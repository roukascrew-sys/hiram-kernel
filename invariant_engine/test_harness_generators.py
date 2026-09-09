"""Unit tests for :mod:`harness_generators`.

Run with any pytest on Python 3.12::

    python -m pytest test_harness_generators.py -q

The graph helpers are tested first and separately. They are the oracles the
generator tests lean on -- an unverified oracle proves nothing about the thing
it is checking -- so they get their own assertions against hand-built graphs
whose answers are known by inspection rather than by running the code.

Where a test asserts an empirical property (cycle merging rates, fan-out
exhaustion), the property is one the module documents, and the test is what
keeps the documentation honest.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pytest

from harness_generators import (
    AERO_SOURCE_NOTE,
    count_paths,
    discretize_state,
    find_cycles_in_rules,
    forward_chain,
    generate_aero_telemetry_fixture,
    generate_diamond_graph,
    generate_horn_rules,
    generate_random_dag,
    is_acyclic,
    topological_order,
)

SEEDS = list(range(12))


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def edges_of(adjacency: Mapping[str, Sequence[str]]) -> List[Tuple[str, str]]:
    """Flatten an adjacency mapping into (parent, child) pairs."""
    return [(p, c) for p, children in adjacency.items() for c in children]


def indegrees(adjacency: Mapping[str, Sequence[str]]) -> Dict[str, int]:
    """In-degree per node, counting every node in the mapping."""
    degree = {node: 0 for node in adjacency}
    for _, child in edges_of(adjacency):
        degree[child] += 1
    return degree


def var_index(name: str) -> int:
    """``"x12" -> 12``. Both generators name variables ``<letter><index>``."""
    return int(name[1:])


class RecordingThresholds(dict):
    """A ``dict`` that remembers which keys were read via ``__getitem__``.

    Used to prove the aero fixture's queries actually exercise every threshold
    it ships, rather than merely declaring them.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read: set = set()

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return super().__getitem__(key)


# --------------------------------------------------------------------------- #
# Graph helpers (the oracles)
# --------------------------------------------------------------------------- #


class TestGraphHelpers:
    def test_is_acyclic_accepts_a_dag(self) -> None:
        assert is_acyclic({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []})

    def test_is_acyclic_rejects_a_cycle(self) -> None:
        assert not is_acyclic({"a": ["b"], "b": ["c"], "c": ["a"]})

    def test_is_acyclic_rejects_a_self_loop(self) -> None:
        assert not is_acyclic({"a": ["a"]})

    def test_is_acyclic_handles_nodes_only_seen_as_children(self) -> None:
        # "b" is never a key. A sloppy implementation misses it entirely.
        assert is_acyclic({"a": ["b"]})

    def test_topological_order_places_parents_first(self) -> None:
        graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
        order = topological_order(graph)
        position = {node: i for i, node in enumerate(order)}
        assert len(order) == 4
        for parent, child in edges_of(graph):
            assert position[parent] < position[child]

    def test_topological_order_raises_on_a_cycle(self) -> None:
        with pytest.raises(ValueError, match="cycle"):
            topological_order({"a": ["b"], "b": ["a"]})

    def test_count_paths_on_a_hand_counted_diamond(self) -> None:
        assert count_paths({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}, "a", "d") == 2

    def test_count_paths_source_equals_sink(self) -> None:
        assert count_paths({"a": ["b"], "b": []}, "a", "a") == 1

    def test_count_paths_when_sink_is_unreachable(self) -> None:
        assert count_paths({"a": [], "b": []}, "a", "b") == 0

    def test_count_paths_raises_on_a_cycle(self) -> None:
        with pytest.raises(ValueError, match="cycle"):
            count_paths({"a": ["b"], "b": ["a"]}, "a", "b")

    def test_find_cycles_detects_a_self_loop(self) -> None:
        rules = [{"antecedents": ["a"], "consequent": "a"}]
        assert find_cycles_in_rules(rules) == [["a"]]

    def test_find_cycles_detects_a_two_cycle(self) -> None:
        rules = [
            {"antecedents": ["a"], "consequent": "b"},
            {"antecedents": ["b"], "consequent": "a"},
        ]
        assert find_cycles_in_rules(rules) == [["a", "b"]]

    def test_find_cycles_returns_empty_for_an_acyclic_rule_set(self) -> None:
        rules = [
            {"antecedents": ["a"], "consequent": "b"},
            {"antecedents": ["b"], "consequent": "c"},
        ]
        assert find_cycles_in_rules(rules) == []

    def test_forward_chain_reaches_the_least_fixpoint(self) -> None:
        rules = [
            {"antecedents": ["a"], "consequent": "b"},
            {"antecedents": ["b", "c"], "consequent": "d"},
        ]
        assert forward_chain(["a"], rules) == {"a", "b"}
        assert forward_chain(["a", "c"], rules) == {"a", "b", "c", "d"}

    def test_forward_chain_terminates_on_a_cyclic_rule_set(self) -> None:
        # Monotone inference over definite clauses cannot diverge. If this test
        # ever hangs rather than fails, that premise has been broken.
        rules = [
            {"antecedents": ["a"], "consequent": "b"},
            {"antecedents": ["b"], "consequent": "a"},
        ]
        assert forward_chain(["a"], rules) == {"a", "b"}


# --------------------------------------------------------------------------- #
# 1. generate_random_dag
# --------------------------------------------------------------------------- #


class TestGenerateRandomDag:
    def test_returns_adjacency_and_priors(self) -> None:
        adjacency, priors = generate_random_dag(10, 3, 2, seed=0)
        assert isinstance(adjacency, dict)
        assert isinstance(priors, dict)
        assert all(isinstance(v, list) for v in adjacency.values())
        assert all(isinstance(p, float) for p in priors.values())

    def test_every_variable_is_a_key_in_both_returns(self) -> None:
        adjacency, priors = generate_random_dag(15, 4, 3, seed=1)
        expected = {f"v{i}" for i in range(15)}
        assert set(adjacency) == expected
        assert set(priors) == expected

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize(
        ("num_vars", "depth", "max_fanout"),
        [(1, 1, 1), (5, 5, 1), (12, 4, 3), (30, 6, 4), (50, 3, 10), (40, 40, 2)],
    )
    def test_is_always_acyclic(
        self, num_vars: int, depth: int, max_fanout: int, seed: int
    ) -> None:
        adjacency, _ = generate_random_dag(num_vars, depth, max_fanout, seed=seed)
        assert is_acyclic(adjacency)

    @pytest.mark.parametrize("max_fanout", [1, 2, 3, 7])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_max_fanout_is_never_exceeded(self, max_fanout: int, seed: int) -> None:
        adjacency, _ = generate_random_dag(24, 5, max_fanout, seed=seed)
        assert max(len(children) for children in adjacency.values()) <= max_fanout

    @pytest.mark.parametrize("max_indegree", [1, 2, 5])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_max_indegree_is_never_exceeded(self, max_indegree: int, seed: int) -> None:
        adjacency, _ = generate_random_dag(
            24, 5, 6, seed=seed, max_indegree=max_indegree
        )
        assert max(indegrees(adjacency).values()) <= max_indegree

    @pytest.mark.parametrize("seed", SEEDS)
    def test_priors_are_valid_probabilities(self, seed: int) -> None:
        _, priors = generate_random_dag(20, 5, 3, seed=seed)
        assert all(0.0 < p < 1.0 for p in priors.values())

    @pytest.mark.parametrize("seed", SEEDS)
    def test_root_and_leak_priors_use_their_documented_ranges(self, seed: int) -> None:
        # Classification is by actual in-degree, not by layer: fan-out
        # exhaustion can leave a later-layer node parentless, and such a node is
        # genuinely a root and must get a root marginal, not a leak term.
        adjacency, priors = generate_random_dag(30, 5, 3, seed=seed)
        degree = indegrees(adjacency)
        for node, prior in priors.items():
            if degree[node] == 0:
                assert 0.05 <= prior <= 0.95, f"{node} is a root, prior={prior}"
            else:
                assert 0.01 <= prior <= 0.15, f"{node} has parents, prior={prior}"

    def test_same_seed_reproduces_the_same_graph(self) -> None:
        assert generate_random_dag(20, 4, 3, seed=99) == generate_random_dag(
            20, 4, 3, seed=99
        )

    def test_different_seeds_produce_different_graphs(self) -> None:
        results = {
            str(generate_random_dag(20, 4, 3, seed=s)) for s in range(8)
        }
        assert len(results) > 1

    def test_unseeded_calls_vary(self) -> None:
        results = {str(generate_random_dag(20, 4, 3)) for _ in range(8)}
        assert len(results) > 1

    def test_single_variable_graph_has_no_edges(self) -> None:
        adjacency, priors = generate_random_dag(1, 1, 1, seed=0)
        assert adjacency == {"v0": []}
        assert 0.05 <= priors["v0"] <= 0.95  # a lone node is a root

    def test_depth_one_produces_only_roots(self) -> None:
        adjacency, priors = generate_random_dag(8, 1, 3, seed=0)
        assert edges_of(adjacency) == []
        assert all(0.05 <= p <= 0.95 for p in priors.values())

    @pytest.mark.parametrize("seed", SEEDS)
    def test_fanout_exhaustion_leaves_extra_roots_rather_than_breaching_the_cap(
        self, seed: int
    ) -> None:
        # Documented behaviour: with 2 layers and max_fanout=1, the first layer
        # cannot parent every node in the second, so the surplus become roots.
        # The cap must hold and the graph must stay a valid DAG regardless.
        adjacency, priors = generate_random_dag(12, 2, 1, seed=seed)
        assert is_acyclic(adjacency)
        assert max(len(c) for c in adjacency.values()) <= 1
        assert len(adjacency) == 12
        assert set(priors) == set(adjacency)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"num_vars": 0, "depth": 1, "max_fanout": 1}, "num_vars"),
            ({"num_vars": -3, "depth": 1, "max_fanout": 1}, "num_vars"),
            ({"num_vars": 5, "depth": 0, "max_fanout": 1}, "depth"),
            ({"num_vars": 5, "depth": 9, "max_fanout": 1}, "cannot exceed"),
            ({"num_vars": 5, "depth": 2, "max_fanout": 0}, "max_fanout"),
        ],
    )
    def test_rejects_inconsistent_arguments(
        self, kwargs: Dict[str, int], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            generate_random_dag(**kwargs)

    def test_rejects_bad_max_indegree(self) -> None:
        with pytest.raises(ValueError, match="max_indegree"):
            generate_random_dag(5, 2, 2, max_indegree=0)


# --------------------------------------------------------------------------- #
# 2. generate_diamond_graph
# --------------------------------------------------------------------------- #


class TestGenerateDiamondGraph:
    @pytest.mark.parametrize("n", [4, 5, 6, 7, 10, 13, 22, 31, 40])
    def test_returns_exactly_n_named_nodes(self, n: int) -> None:
        chain = generate_diamond_graph(n)
        assert set(chain) == {f"v{i}" for i in range(n)}

    @pytest.mark.parametrize("n", [4, 5, 6, 7, 10, 13, 22, 31, 40])
    def test_chain_is_acyclic(self, n: int) -> None:
        assert is_acyclic(generate_diamond_graph(n))

    @pytest.mark.parametrize("n", [4, 5, 6, 7, 10, 13, 22, 31, 40, 61])
    def test_chain_path_count_is_two_to_the_diamond_count(self, n: int) -> None:
        # The whole point of the fixture: path count exponential in node count.
        k = (n - 1) // 3
        assert count_paths(generate_diamond_graph(n), "v0", f"v{n - 1}") == 2**k

    def test_path_count_grows_exponentially_not_linearly(self) -> None:
        counts = [count_paths(generate_diamond_graph(n), "v0", f"v{n-1}")
                  for n in (4, 13, 22, 31)]
        assert counts == [2, 16, 128, 1024]

    def test_large_chain_is_counted_exactly_without_enumeration(self) -> None:
        # 151 nodes -> 2**50 paths. This must return, and return an exact int.
        assert count_paths(generate_diamond_graph(151), "v0", "v150") == 2**50

    @pytest.mark.parametrize("n", [4, 7, 13, 31])
    def test_source_has_no_parents_and_sink_no_children(self, n: int) -> None:
        chain = generate_diamond_graph(n)
        assert indegrees(chain)["v0"] == 0
        assert chain[f"v{n - 1}"] == []

    @pytest.mark.parametrize("n", [4, 5, 6, 7, 13, 22])
    def test_every_edge_runs_forward_in_the_node_numbering(self, n: int) -> None:
        for parent, child in edges_of(generate_diamond_graph(n)):
            assert var_index(parent) < var_index(child)

    def test_single_diamond_has_the_textbook_shape(self) -> None:
        assert generate_diamond_graph(4) == {
            "v0": ["v1", "v2"],
            "v1": ["v3"],
            "v2": ["v3"],
            "v3": [],
        }

    @pytest.mark.parametrize(("n", "tail"), [(5, ["v4"]), (6, ["v4", "v5"])])
    def test_leftover_nodes_become_a_linear_tail(self, n: int, tail: List[str]) -> None:
        chain = generate_diamond_graph(n)
        assert count_paths(chain, "v0", f"v{n - 1}") == 2  # tail adds no branching
        assert set(chain) >= set(tail)
        assert chain["v3"] == [tail[0]]  # the join feeds the tail

    @pytest.mark.parametrize("n", [3, 5, 13, 40])
    def test_wide_mode_path_count_is_linear(self, n: int) -> None:
        wide = generate_diamond_graph(n, mode="wide")
        assert count_paths(wide, "v0", f"v{n - 1}") == n - 2
        assert is_acyclic(wide)
        assert set(wide) == {f"v{i}" for i in range(n)}

    def test_wide_mode_shape(self) -> None:
        assert generate_diamond_graph(4, mode="wide") == {
            "v0": ["v1", "v2"],
            "v1": ["v3"],
            "v2": ["v3"],
            "v3": [],
        }

    def test_wide_and_chain_diverge_once_there_is_room_to(self) -> None:
        # n=4 is the one size where both modes are the same single diamond.
        assert generate_diamond_graph(4) == generate_diamond_graph(4, mode="wide")
        assert generate_diamond_graph(7) != generate_diamond_graph(7, mode="wide")

    def test_is_deterministic(self) -> None:
        assert generate_diamond_graph(22) == generate_diamond_graph(22)

    @pytest.mark.parametrize("n", [0, 1, 2, 3, -5])
    def test_chain_rejects_n_below_one_diamond(self, n: int) -> None:
        with pytest.raises(ValueError, match="n >= 4"):
            generate_diamond_graph(n)

    @pytest.mark.parametrize("n", [0, 1, 2, -5])
    def test_wide_rejects_n_below_three(self, n: int) -> None:
        with pytest.raises(ValueError, match="n >= 3"):
            generate_diamond_graph(n, mode="wide")

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="chain.*wide"):
            generate_diamond_graph(10, mode="lattice")


# --------------------------------------------------------------------------- #
# 3. generate_horn_rules
# --------------------------------------------------------------------------- #


class TestGenerateHornRules:
    @pytest.mark.parametrize("num_rules", [0, 1, 5, 40, 120])
    def test_returns_exactly_num_rules_clauses(self, num_rules: int) -> None:
        assert len(generate_horn_rules(num_rules, 10, seed=0)) == num_rules

    def test_clause_shape(self) -> None:
        for rule in generate_horn_rules(30, 10, 0.2, seed=0):
            assert set(rule) == {"antecedents", "consequent", "circular"}
            assert isinstance(rule["antecedents"], list)
            assert all(isinstance(a, str) for a in rule["antecedents"])
            assert isinstance(rule["consequent"], str)
            assert isinstance(rule["circular"], bool)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_all_variables_come_from_the_declared_pool(self, seed: int) -> None:
        pool = {f"x{i}" for i in range(9)}
        for rule in generate_horn_rules(40, 9, 0.25, seed=seed):
            assert rule["consequent"] in pool
            assert set(rule["antecedents"]) <= pool

    @pytest.mark.parametrize(
        ("num_rules", "fraction"), [(40, 0.1), (40, 0.25), (40, 0.5), (33, 0.3), (7, 1.0)]
    )
    def test_injected_circular_count_matches_the_requested_fraction(
        self, num_rules: int, fraction: float
    ) -> None:
        rules = generate_horn_rules(num_rules, 12, fraction, seed=5)
        assert sum(1 for r in rules if r["circular"]) == round(num_rules * fraction)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_zero_fraction_yields_a_genuinely_acyclic_rule_set(self, seed: int) -> None:
        rules = generate_horn_rules(60, 12, 0.0, seed=seed)
        assert all(not r["circular"] for r in rules)
        assert find_cycles_in_rules(rules) == []

    @pytest.mark.parametrize("seed", SEEDS)
    def test_full_fraction_marks_every_clause_circular(self, seed: int) -> None:
        rules = generate_horn_rules(20, 12, 1.0, seed=seed)
        assert all(r["circular"] for r in rules)
        assert find_cycles_in_rules(rules) != []

    @pytest.mark.parametrize("seed", SEEDS)
    def test_injected_cycles_are_actually_detectable(self, seed: int) -> None:
        rules = generate_horn_rules(40, 12, 0.25, seed=seed)
        assert find_cycles_in_rules(rules) != []

    def test_a_budget_of_one_degenerates_to_a_self_loop(self) -> None:
        # round(4 * 0.25) == 1, and a single clause can only close a cycle on
        # itself.
        circular = [r for r in generate_horn_rules(4, 6, 0.25, seed=1) if r["circular"]]
        assert len(circular) == 1
        assert circular[0]["antecedents"] == [circular[0]["consequent"]]

    @pytest.mark.parametrize("seed", SEEDS)
    def test_non_circular_clauses_always_point_forward(self, seed: int) -> None:
        # The acyclicity guarantee for the non-injected half rests entirely on
        # this ordering.
        for rule in generate_horn_rules(60, 12, 0.25, seed=seed):
            if rule["circular"]:
                continue
            for antecedent in rule["antecedents"]:
                assert var_index(antecedent) < var_index(rule["consequent"])

    @pytest.mark.parametrize("seed", SEEDS)
    def test_non_circular_clauses_are_deduplicated(self, seed: int) -> None:
        # Regression: an earlier revision returned ~5 literal duplicates per
        # 40-rule draw, so a caller asking for 40 rules got ~35 distinct ones.
        rules = generate_horn_rules(40, 12, 0.25, seed=seed)
        keys = [
            (tuple(r["antecedents"]), r["consequent"]) for r in rules if not r["circular"]
        ]
        assert len(keys) == len(set(keys))

    def test_exact_count_wins_when_the_clause_space_is_too_small(self) -> None:
        # 2 variables admit very few distinct clauses; the contract is that the
        # list is still exactly num_rules long, duplicates and all.
        rules = generate_horn_rules(50, 2, 0.0, seed=0)
        assert len(rules) == 50

    @pytest.mark.parametrize("max_antecedents", [1, 2, 5])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_max_antecedents_is_respected(
        self, max_antecedents: int, seed: int
    ) -> None:
        rules = generate_horn_rules(
            40, 12, 0.0, seed=seed, max_antecedents=max_antecedents
        )
        assert max(len(r["antecedents"]) for r in rules) <= max_antecedents

    @pytest.mark.parametrize(
        ("fact_fraction", "expected_facts"), [(0.0, 0), (1.0, 30)]
    )
    def test_fact_fraction_extremes(
        self, fact_fraction: float, expected_facts: int
    ) -> None:
        rules = generate_horn_rules(30, 8, 0.0, seed=3, fact_fraction=fact_fraction)
        assert sum(1 for r in rules if not r["antecedents"]) == expected_facts

    def test_same_seed_reproduces_the_same_rule_set(self) -> None:
        assert generate_horn_rules(40, 12, 0.25, seed=7) == generate_horn_rules(
            40, 12, 0.25, seed=7
        )

    def test_different_seeds_produce_different_rule_sets(self) -> None:
        assert len({str(generate_horn_rules(40, 12, 0.25, seed=s)) for s in range(8)}) > 1

    def test_cycles_are_not_clustered_at_the_head_of_the_list(self) -> None:
        # The list is shuffled; a consumer that only reads a prefix must not get
        # a systematically different rule population.
        positions = [
            i
            for s in range(30)
            for i, r in enumerate(generate_horn_rules(40, 12, 0.25, seed=s))
            if r["circular"]
        ]
        assert max(positions) > 30  # injected cycles reach the tail of the list

    @pytest.mark.parametrize("seed", range(50))
    def test_injected_cycles_can_merge_into_larger_sccs(self, seed: int) -> None:
        # Documented caveat: circular_fraction is an INJECTION rate, not a
        # promise about cycle count or size. Injected back-edges combine with
        # the forward edges of the acyclic clauses to close larger cycles. This
        # test pins that documented behaviour so nobody "fixes" the docs.
        cycles = find_cycles_in_rules(generate_horn_rules(40, 12, 0.25, seed=seed))
        assert any(len(c) > 4 for c in cycles)  # 4 is the largest injected cycle

    @pytest.mark.parametrize("seed", SEEDS)
    def test_forward_chaining_terminates_on_a_cyclic_generated_set(
        self, seed: int
    ) -> None:
        rules = generate_horn_rules(40, 12, 0.5, seed=seed)
        facts = [r["consequent"] for r in rules if not r["antecedents"]]
        derived = forward_chain(facts, rules)
        assert derived <= {f"x{i}" for i in range(12)}

    @pytest.mark.parametrize(
        ("args", "kwargs", "match"),
        [
            ((-1, 10), {}, "num_rules"),
            ((10, 1), {}, "num_vars"),
            ((10, 0), {}, "num_vars"),
            ((10, 10, 1.5), {}, "circular_fraction"),
            ((10, 10, -0.1), {}, "circular_fraction"),
            ((10, 10), {"max_antecedents": 0}, "max_antecedents"),
            ((10, 10), {"fact_fraction": 2.0}, "fact_fraction"),
        ],
    )
    def test_rejects_out_of_range_arguments(
        self, args: tuple, kwargs: Dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            generate_horn_rules(*args, **kwargs)


# --------------------------------------------------------------------------- #
# 4. generate_aero_telemetry_fixture
# --------------------------------------------------------------------------- #

FIXTURE = generate_aero_telemetry_fixture()
QUERY_IDS = [q["name"] for q in FIXTURE["queries"]]


class TestAeroFixtureStructure:
    def test_top_level_blocks(self) -> None:
        assert set(FIXTURE) == {"meta", "thresholds", "premises", "rules", "queries"}

    def test_each_call_returns_a_fresh_object(self) -> None:
        # Fixtures that share mutable state let one test silently poison the
        # next. Mutate hard, then confirm a new call is untouched.
        first = generate_aero_telemetry_fixture()
        first["thresholds"]["VMO"]["value"] = 1.0
        first["rules"].clear()
        first["queries"].clear()
        second = generate_aero_telemetry_fixture()
        assert second["thresholds"]["VMO"]["value"] == 340.0
        assert second["rules"]
        assert second["queries"]

    def test_declares_the_five_requested_premise_channels(self) -> None:
        channels = set(FIXTURE["meta"]["telemetry_channels"])
        assert {
            "airspeed_kias",
            "altitude_ft",
            "aoa_deg",
            "flap_setting_deg",
            "engines",
        } <= channels

    def test_nominal_premises_match_their_own_discretization(self) -> None:
        premises = FIXTURE["premises"]
        assert premises["atoms"] == discretize_state(
            premises["state"], FIXTURE["thresholds"]
        )

    def test_nominal_cruise_premises_are_a_safe_state(self) -> None:
        conclusions = forward_chain(FIXTURE["premises"]["atoms"], FIXTURE["rules"])
        assert "envelope_exceedance" not in conclusions


class TestAeroProvenance:
    """CLAUDE.md rule: unverified numbers must be flagged as such, in the data."""

    def test_every_threshold_is_flagged_unverified(self) -> None:
        assert all(t["verified"] is False for t in FIXTURE["thresholds"].values())
        assert FIXTURE["meta"]["all_thresholds_verified"] is False

    def test_every_threshold_carries_unit_limit_state_and_basis(self) -> None:
        for name, t in FIXTURE["thresholds"].items():
            assert set(t) == {"value", "unit", "limit_state", "basis", "verified"}
            assert isinstance(t["value"], float), name
            assert t["unit"], name
            assert t["limit_state"], name
            assert len(t["basis"]) > 20, f"{name} basis is too thin to be provenance"

    def test_every_threshold_basis_admits_the_value_is_synthetic(self) -> None:
        for name, t in FIXTURE["thresholds"].items():
            assert "SYNTHETIC" in t["basis"], f"{name} does not flag its value"

    def test_source_note_is_explicit_about_provenance(self) -> None:
        assert "SYNTHETIC" in AERO_SOURCE_NOTE
        assert "UNVERIFIED" in AERO_SOURCE_NOTE
        assert FIXTURE["meta"]["source_note"] == AERO_SOURCE_NOTE

    def test_stall_warning_margin_documents_the_cas_versus_aoa_caveat(self) -> None:
        # 14 CFR 25.207 states the certified margin in CAS, not AoA. Converting
        # needs a lift curve this fixture does not have; the record must say so
        # rather than imply regulatory backing.
        basis = FIXTURE["thresholds"]["ALPHA_WARNING_MARGIN"]["basis"]
        assert "CAS" in basis
        assert "NOT the regulatory margin" in basis

    def test_every_rule_names_a_limit_state_and_a_basis(self) -> None:
        for rule in FIXTURE["rules"]:
            assert rule["limit_state"], rule
            assert rule["basis"], rule

    def test_rule_limit_states_are_declared_by_some_threshold(self) -> None:
        declared = {t["limit_state"] for t in FIXTURE["thresholds"].values()}
        for rule in FIXTURE["rules"]:
            assert rule["limit_state"] in declared, rule["limit_state"]


class TestAeroRules:
    def test_rule_set_is_acyclic(self) -> None:
        # The aero fixture is the known-good control for the cycle detector.
        assert find_cycles_in_rules(FIXTURE["rules"]) == []

    def test_rules_share_the_generate_horn_rules_shape(self) -> None:
        for rule in FIXTURE["rules"]:
            assert isinstance(rule["antecedents"], list)
            assert rule["antecedents"]  # no bare facts in the rule block
            assert isinstance(rule["consequent"], str)

    def test_every_rule_antecedent_is_producible(self) -> None:
        # A rule whose body can never be satisfied is dead weight that silently
        # inflates the apparent coverage of the fixture.
        producible = {
            atom
            for q in FIXTURE["queries"]
            for atom in discretize_state(q["state"], FIXTURE["thresholds"])
        }
        producible |= {r["consequent"] for r in FIXTURE["rules"]}
        for rule in FIXTURE["rules"]:
            assert set(rule["antecedents"]) <= producible, rule

    def test_every_rule_fires_in_at_least_one_query(self) -> None:
        fired = set()
        for q in FIXTURE["queries"]:
            derived = forward_chain(
                discretize_state(q["state"], FIXTURE["thresholds"]), FIXTURE["rules"]
            )
            for i, rule in enumerate(FIXTURE["rules"]):
                if all(a in derived for a in rule["antecedents"]):
                    fired.add(i)
        never = [FIXTURE["rules"][i] for i in range(len(FIXTURE["rules"])) if i not in fired]
        assert not never, f"rules never exercised by any query: {never}"

    def test_every_threshold_is_read_by_at_least_one_query(self) -> None:
        recording = RecordingThresholds(FIXTURE["thresholds"])
        for q in FIXTURE["queries"]:
            discretize_state(q["state"], recording)
        unread = set(FIXTURE["thresholds"]) - recording.read
        assert not unread, f"thresholds never exercised by any query: {sorted(unread)}"

    def test_stall_warning_does_not_imply_an_exceedance(self) -> None:
        # An annunciation is not a violation. Conflating them is the classic
        # false-positive in envelope monitors.
        derived = forward_chain(["aoa_above_stall_warning_aoa"], FIXTURE["rules"])
        assert "stall_warning_active" in derived
        assert "envelope_exceedance" not in derived

    @pytest.mark.parametrize(
        "limit_atom",
        [
            "flap_structural_overload",
            "airframe_overspeed",
            "aerodynamic_stall",
            "engine_out_ceiling_exceedance",
            "altitude_envelope_exceedance",
        ],
    )
    def test_every_named_limit_state_rolls_up_to_the_violation_atom(
        self, limit_atom: str
    ) -> None:
        assert "envelope_exceedance" in forward_chain([limit_atom], FIXTURE["rules"])


class TestAeroQueries:
    def test_covers_both_safe_and_violating_states(self) -> None:
        verdicts = {q["expected_safe"] for q in FIXTURE["queries"]}
        assert verdicts == {True, False}

    def test_query_records_are_well_formed(self) -> None:
        for q in FIXTURE["queries"]:
            assert set(q) == {
                "name",
                "description",
                "state",
                "expected_conclusions",
                "expected_safe",
            }
            assert isinstance(q["expected_safe"], bool)
            assert q["description"]

    def test_query_names_are_unique(self) -> None:
        assert len(QUERY_IDS) == len(set(QUERY_IDS))

    @pytest.mark.parametrize("query", FIXTURE["queries"], ids=QUERY_IDS)
    def test_safety_verdict_matches_ground_truth(self, query: Dict[str, Any]) -> None:
        atoms = discretize_state(query["state"], FIXTURE["thresholds"])
        derived = forward_chain(atoms, FIXTURE["rules"])
        assert ("envelope_exceedance" not in derived) is query["expected_safe"]

    @pytest.mark.parametrize("query", FIXTURE["queries"], ids=QUERY_IDS)
    def test_expected_conclusions_are_derived(self, query: Dict[str, Any]) -> None:
        atoms = discretize_state(query["state"], FIXTURE["thresholds"])
        derived = forward_chain(atoms, FIXTURE["rules"])
        assert set(query["expected_conclusions"]) <= derived

    @pytest.mark.parametrize("query", FIXTURE["queries"], ids=QUERY_IDS)
    def test_query_atoms_stay_within_the_declared_vocabulary(
        self, query: Dict[str, Any]
    ) -> None:
        vocabulary = set(FIXTURE["meta"]["atom_vocabulary"])
        for atom in discretize_state(query["state"], FIXTURE["thresholds"]):
            if atom.startswith("flaps_detent_"):
                assert "flaps_detent_<d>" in vocabulary
                continue
            assert atom in vocabulary, atom


class TestDiscretizeState:
    """Boundary behaviour. These are the tests that catch off-by-one bugs."""

    @staticmethod
    def state(**overrides: Any) -> Dict[str, Any]:
        base = {
            "airspeed_kias": 250.0,
            "mach": 0.60,
            "altitude_ft": 20000.0,
            "aoa_deg": 3.0,
            "flap_setting_deg": 0,
            "engines": ["operating", "operating"],
        }
        base.update(overrides)
        return base

    def atoms(self, **overrides: Any) -> List[str]:
        return discretize_state(self.state(**overrides), FIXTURE["thresholds"])

    def test_returns_a_sorted_list(self) -> None:
        atoms = self.atoms(flap_setting_deg=15, airspeed_kias=190.0)
        assert atoms == sorted(atoms)

    @pytest.mark.parametrize(
        ("airspeed", "expected"), [(204.9, False), (205.0, False), (205.1, True)]
    )
    def test_vfe_comparison_is_strict(self, airspeed: float, expected: bool) -> None:
        # A placard limit is a MAXIMUM PERMISSIBLE value: exactly on it is legal.
        atoms = self.atoms(flap_setting_deg=15, airspeed_kias=airspeed)
        assert ("airspeed_above_vfe" in atoms) is expected

    @pytest.mark.parametrize(
        ("airspeed", "expected"), [(339.9, False), (340.0, False), (340.1, True)]
    )
    def test_vmo_comparison_is_strict(self, airspeed: float, expected: bool) -> None:
        assert ("airspeed_above_vmo" in self.atoms(airspeed_kias=airspeed)) is expected

    @pytest.mark.parametrize(
        ("mach", "expected"), [(0.819, False), (0.82, False), (0.821, True)]
    )
    def test_mmo_comparison_is_strict(self, mach: float, expected: bool) -> None:
        assert ("mach_above_mmo" in self.atoms(mach=mach)) is expected

    @pytest.mark.parametrize(
        ("altitude", "expected"), [(40999.0, False), (41000.0, False), (41001.0, True)]
    )
    def test_max_altitude_comparison_is_strict(
        self, altitude: float, expected: bool
    ) -> None:
        atoms = self.atoms(altitude_ft=altitude)
        assert ("altitude_above_max_operating" in atoms) is expected

    @pytest.mark.parametrize(
        ("aoa", "expected"), [(15.4, False), (15.5, False), (15.6, True)]
    )
    def test_alpha_crit_comparison_is_strict(self, aoa: float, expected: bool) -> None:
        assert ("aoa_above_alpha_crit" in self.atoms(aoa_deg=aoa)) is expected

    def test_alpha_crit_depends_on_flap_configuration(self) -> None:
        # 12.5 deg is well inside the clean limit and past the flaps-down one.
        # A single fixed alpha_crit misses this entirely.
        assert "aoa_above_alpha_crit" not in self.atoms(aoa_deg=12.5)
        assert "aoa_above_alpha_crit" in self.atoms(
            aoa_deg=12.5, flap_setting_deg=30, airspeed_kias=160.0
        )

    def test_stall_warning_precedes_the_stall(self) -> None:
        warning_only = self.atoms(aoa_deg=14.0)
        assert "aoa_above_stall_warning_aoa" in warning_only
        assert "aoa_above_alpha_crit" not in warning_only

    def test_flaps_up_and_extended_are_mutually_exclusive(self) -> None:
        assert "flaps_up" in self.atoms(flap_setting_deg=0)
        assert "flaps_extended" not in self.atoms(flap_setting_deg=0)
        extended = self.atoms(flap_setting_deg=5, airspeed_kias=200.0)
        assert "flaps_extended" in extended and "flaps_up" not in extended
        assert "flaps_detent_5" in extended

    def test_engine_status_atoms_are_mutually_exclusive(self) -> None:
        both = self.atoms()
        assert "all_engines_operating" in both and "engine_out" not in both
        one_out = self.atoms(engines=["operating", "inoperative"])
        assert "engine_out" in one_out and "all_engines_operating" not in one_out

    @pytest.mark.parametrize(
        ("altitude", "expected"), [(24999.0, False), (25000.0, False), (25001.0, True)]
    )
    def test_single_engine_ceiling_only_applies_with_an_engine_out(
        self, altitude: float, expected: bool
    ) -> None:
        engines_out = self.atoms(
            altitude_ft=altitude, engines=["operating", "inoperative"]
        )
        assert ("altitude_above_single_engine_ceiling" in engines_out) is expected
        # ... and never when both engines are running.
        assert "altitude_above_single_engine_ceiling" not in self.atoms(
            altitude_ft=altitude
        )

    def test_unknown_flap_detent_raises_rather_than_assuming_no_limit(self) -> None:
        # Silently treating an undefined detent as unlimited is the dangerous
        # failure mode, so it must raise.
        with pytest.raises(KeyError, match="VFE_FLAP_20"):
            discretize_state(
                self.state(flap_setting_deg=20), FIXTURE["thresholds"]
            )

    @pytest.mark.parametrize(
        "channel",
        ["airspeed_kias", "mach", "altitude_ft", "aoa_deg", "flap_setting_deg", "engines"],
    )
    def test_missing_telemetry_channel_raises(self, channel: str) -> None:
        state = self.state()
        del state[channel]
        with pytest.raises(KeyError, match=channel):
            discretize_state(state, FIXTURE["thresholds"])
