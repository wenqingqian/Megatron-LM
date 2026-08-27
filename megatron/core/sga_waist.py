# Copyright (c) 2026, SGA framework team
"""SGA narrow waist (API v1).

The ONLY bridge between Megatron-LM and the SGA plugin framework: a config
exit plus three runtime call points and one static guard, all defined in
AGENTSPACE/notes/sga-waist-api-v1.md. This module contains no scheme code
and no scheme names; it lazily imports the external ``sga`` package (the
SGA repository, expected on PYTHONPATH) and delegates everything to a
single engine object built from the opaque ``--sga-config`` JSON.

Contract highlights (full normative text lives in the API note):
  - Without ``--sga-config`` every entry point is a no-op and the ``sga``
    package is never imported (stock runs must not require it).
  - With ``--sga-config`` set but the ``sga`` package unimportable, the
    failure surfaces at engine-build time (first attach), never at the
    first optimizer step.
  - ``grad_sync`` does not take an iteration: the engine tracks iteration
    internally and advances it only on successful optimizer steps.
"""

import argparse
import json

SGA_API_VERSION = 1

_engine = None
_engine_failed = False


def add_sga_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Config-extension exit: one opaque JSON argument (schema-stable as plugins grow)."""
    group = parser.add_argument_group('SGA')
    group.add_argument(
        '--sga-config', type=str, default='', dest='sga_config',
        help='Canonical (sorted-key) JSON string describing the SGA four-field '
             'plugin configuration (score/selection/error_feedback/backend). '
             'Empty (default) disables SGA entirely.',
    )
    group.add_argument(
        '--sga-allow-config-override', action='store_true',
        dest='sga_allow_config_override',
        help='Allow resuming from a checkpoint whose saved sga_config differs '
             'from the current one (refused by default).',
    )
    return parser


def _config_from_args():
    from megatron.training import get_args

    raw = getattr(get_args(), 'sga_config', '')
    if not raw:
        return None
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f'--sga-config is not valid JSON: {e}') from e
    if not isinstance(config, dict):
        raise ValueError('--sga-config must decode to a JSON object')
    return config


def _get_engine():
    """Lazy singleton; None when SGA is disabled. Never retries a failed build."""
    global _engine, _engine_failed
    if _engine is not None or _engine_failed:
        return _engine
    config = _config_from_args()
    if config is None:
        return None
    try:
        from sga.runtime import build_engine
    except ImportError as e:
        _engine_failed = True
        raise ImportError(
            '--sga-config is set but the sga package cannot be imported '
            f'({e}); is the SGA repository on PYTHONPATH (source env.sh)?'
        ) from e
    _engine = build_engine(config, api_version=SGA_API_VERSION)
    return _engine


def active() -> bool:
    """Static guard for the dense zero_grad_buffer path (never mutable cross-call state)."""
    return _get_engine() is not None


def attach(model_chunk) -> None:
    """Call point 1: bucket walking / BucketState creation. Invoked at DDP.__init__ tail."""
    engine = _get_engine()
    if engine is None:
        return
    engine.attach(model_chunk)


def grad_sync(model_chunk, force_all_reduce: bool) -> bool:
    """Call point 2: selective gradient synchronization.

    Returns True when the engine handled every bucket group of this chunk
    (sparse collectives, or delegation to the dense path for full-sync
    steps); False when the caller should run the stock dense sync.
    """
    engine = _engine
    if engine is None:
        return False
    return engine.grad_sync(model_chunk, force_all_reduce)


def optimizer_step(optimizer, model_chunks, iteration: int):
    """Call point 3: engine-owned optimizer step.

    ``iteration`` is the fork's 0-based train_step iteration; the engine's
    cycle accounting is 1-based (converted here).

    Returns None when inactive (caller runs the stock step), else the
    (update_successful, grad_norm, num_zeros_in_grad) tuple with stock
    semantics. On success the engine also performs the post-update
    selective buffer reset that replaces next iteration's dense
    zero_grad_buffer (see the static guard at the train_step site).
    """
    engine = _engine
    if engine is None:
        return None
    return engine.optimizer_step(optimizer, model_chunks, iteration + 1)
