"""
sim.scenarios - Deterministic scenario generation for the stopping benchmark.

sim.scenarios.strata      Stratum type contracts (StratumType, ScenarioInstance)
                          and the fixed per-stratum episode counts.
sim.scenarios.generator   Cryptographically-salted, reproducible generation of
                          ScenarioInstance episodes per stratum, with a
                          physically-verified recoverability guarantee (via
                          sim.plant.analytic_plant, read-only) for all six
                          nominal strata.
"""

from .generator import (
    DEFAULT_MASTER_SEED,
    derive_seed,
    generate_episode,
    generate_stratum,
    generate_stratum_sample,
    iter_all,
    iter_stratum,
    manifest_digest,
)
from .strata import (
    ALL_STRATA,
    EPISODES_PER_NOMINAL_STRATUM,
    EPISODES_PER_SHIFT_STRATUM,
    NOMINAL_STRATA,
    SHIFT_STRATA,
    TOTAL_EPISODES,
    TOTAL_NOMINAL_EPISODES,
    TOTAL_SHIFT_EPISODES,
    ScenarioInstance,
    StratumType,
    episodes_for,
)

__all__ = [
    "StratumType",
    "ScenarioInstance",
    "NOMINAL_STRATA",
    "SHIFT_STRATA",
    "ALL_STRATA",
    "EPISODES_PER_NOMINAL_STRATUM",
    "EPISODES_PER_SHIFT_STRATUM",
    "TOTAL_NOMINAL_EPISODES",
    "TOTAL_SHIFT_EPISODES",
    "TOTAL_EPISODES",
    "episodes_for",
    "DEFAULT_MASTER_SEED",
    "derive_seed",
    "generate_episode",
    "generate_stratum",
    "generate_stratum_sample",
    "iter_all",
    "iter_stratum",
    "manifest_digest",
]
