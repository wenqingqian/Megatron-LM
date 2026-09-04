# Copyright (c) 2026, SGA framework team
"""SGA narrow waist (API v2).

The ONLY bridge between Megatron-LM and the SGA plugin framework: an
arguments exit plus three runtime call points and one static guard. This
module contains no scheme code and no scheme names; it lazily imports
the external ``sga`` package (the SGA repository, expected on PYTHONPATH)
and delegates everything to a single engine object.

Config transport (v2): FLAGS ARE THE TRANSPORT. The training entry
registers the flat ``--sga-*`` vocabulary on megatron's argparse
(``sga/config/arguments.py`` in the SGA repo, via the entry's
extra_args_provider); after parsing, the configuration simply LIVES on
the args namespace as ``sga_*`` attributes. This waist collects those
attributes — the ``sga_`` prefix and its own override switch are all it
knows; the vocabulary itself stays in the SGA repo — and hands the flat
dict to ``build_engine``. There is no JSON string and no re-serialization
(anything checkpointable rides inside the checkpointed args namespace,
which is exactly what the v1 JSON bought).

Contract highlights:
  - Without ``sga_enabled`` true (stock runs carry no SGA attributes at
    all) every entry point is a no-op and the ``sga`` package is never
    imported (stock runs must not require it).
  - With SGA active but the ``sga`` package unimportable, the failure
    surfaces at engine-build time (first attach), never at the first
    optimizer step.
  - ``grad_sync`` does not take an iteration: the engine tracks iteration
    internally and advances it only on successful optimizer steps.
"""

import argparse

SGA_API_VERSION = 2

_engine = None
_engine_failed = False


def add_sga_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments exit: the resume-override switch only.

    The ``--sga-*`` config vocabulary is registered by the SGA training
    entry's own extra_args_provider (the SGA repo owns its declarations);
    this fork-side exit exists so a resume may be forced past the config
    guard without any SGA import.
    """
    group = parser.add_argument_group('SGA')
    group.add_argument(
        '--sga-allow-config-override', action='store_true',
        dest='sga_allow_config_override',
        help='Allow resuming from a checkpoint whose saved sga '
             'configuration differs from the current one (refused by '
             'default).',
    )
    return parser


def _config_from_args():
    from megatron.training import get_args

    args = get_args()
    if getattr(args, 'sga_enabled', False) is not True:
        return None
    # Collect every sga_* attribute (argparse defaults included — the SGA
    # package diffs them against its declared defaults). The non-config
    # sga_ keys are this waist's own override switch and the retired v1
    # JSON-transport key (still present on old checkpoints); the
    # fork-registered-key set stays pinned by the SGA-side invariant tests.
    skip = ('sga_allow_config_override', 'sga_config')
    return {key: value for key, value in vars(args).items()
            if key.startswith('sga_') and key not in skip}


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
            'SGA is enabled (sga_enabled) but the sga package cannot be '
            f'imported ({e}); is the SGA repository on PYTHONPATH '
            '(source env.sh)?'
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
