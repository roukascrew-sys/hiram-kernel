"""
Scenario generation and strata definitions for Milestone M0 custody.
"""

from .strata import (
    StratumType,
    ScenarioInstance,
    derive_domain_seed,
)
from .generator import (
    DEFAULT_MASTER_SEED,
    NOMINAL_EPISODES_PER_STRATUM,
    SHIFT_EPISODES_PER_STRATUM,
    derive_seed,
    generate_instance,
    generate_stratum_sample,
    manifest_digest,
)

# Backwards compatibility alias
generate_episode = generate_instance

__all__ = [
    "StratumType",
    "ScenarioInstance",
    "derive_domain_seed",
    "derive_seed",
    "generate_instance",
    "generate_episode",
    "generate_stratum_sample",
    "manifest_digest",
    "DEFAULT_MASTER_SEED",
    "NOMINAL_EPISODES_PER_STRATUM",
    "SHIFT_EPISODES_PER_STRATUM",
]