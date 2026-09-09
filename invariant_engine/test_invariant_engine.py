"""Unit tests for :mod:`invariant_engine`.

Run with any pytest on Python 3.12::

    python -m pytest test_invariant_engine.py -q

This is a trusted computing base, so the suite is organised around what the
kernel promises rather than around its functions:

* untrusted data cannot enter by construction (:class:`TestTrustBoundary`);
* every admission gate rejects rather than raises (:class:`TestAdmitRules`,
  :class:`TestAdmitPriors`);
* a defective trace is never accepted, and the reason given is the right one
  (:class:`TestVerifyTrace`);
* **no proposal can change a verdict** -- the soundness property, tested against
  every defect class the harness can produce (:class:`TestCheckGoalSoundness`);
* SAFE is unreachable without an explicit closed-world assumption
  (:class:`TestEnvelopeMonitor`).

Exact inference is checked against brute-force enumeration of the joint
distribution rather than against itself, so :func:`marginal` is graded by an
oracle that shares none of its machinery.
"""

from __future__ import annotations

import itertools
import math
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pytest

from harness_generators import (
    generate_aero_telemetry_fixture,
    generate_candidate_traces,
    generate_diamond_graph,
    generate_horn_rules,
    generate_random_dag,
    inject_rule_hallucinations,
)
from invariant_engine import (
    EliminationStats,
    Factor,
    Finding,
    GoalReport,
    InvariantReport,
    NoisyOr,
    Rejection,
    RuleBase,
    Status,
    TraceStep,
    Verdict,
    admit_priors,
    admit_rules,
    check_aero_fixture,
    check_envelope,
    check_goal,
    derive,
    marginal,
    parse_trace,
    verify_trace,
)

SEEDS = list(range(8))

# A small, hand-checkable rule base. derive("goal") should emit six steps:
# p, q (premises), r (rule 0), s (rule 1), t (premise), goal (rule 2).
CLAUSES: List[Dict[str, Any]] = [
    {"antecedents": ["p", "q"], "consequent": "r"},
    {"antecedents": ["r"], "consequent": "s"},
    {"antecedents": ["s", "t"], "consequent": "goal"},
]
FACTS = ["p", "q", "t"]


def admitted(clauses: Sequence[Mapping[str, Any]] = CLAUSES, **kwargs: Any) -> RuleBase:
    """Admit ``clauses`` and assert nothing was rejected."""
    rulebase, findings = admit_rules(clauses, **kwargs)
    assert not findings, findings
    return rulebase


def reasons(findings: Iterable[Finding]) -> set:
    return {f.reason for f in findings}


# --------------------------------------------------------------------------- #
# The trust boundary
# --------------------------------------------------------------------------- #


class TestTrustBoundary:
    def test_rulebase_cannot_be_constructed_directly(self) -> None:
        with pytest.raises(TypeError, match="admit_rules"):
            RuleBase((), frozenset())

    def test_rulebase_cannot_be_forged_with_a_guessed_token(self) -> None:
        for guess in (None, object(), "admit", 0, True):
            with pytest.raises(TypeError, match="admit_rules"):
                RuleBase((), frozenset(), guess)

    def test_admit_rules_is_the_only_door_in(self) -> None:
        rulebase = admitted()
        assert isinstance(rulebase, RuleBase)
        assert len(rulebase) == 3

    def test_rulebase_is_immutable(self) -> None:
        rulebase = admitted()
        with pytest.raises(Exception):
            rulebase.clauses = ()  # type: ignore[misc]

    def test_accessors_agree_with_the_admitted_clauses(self) -> None:
        rulebase = admitted()
        assert rulebase.antecedents(0) == ("p", "q")
        assert rulebase.consequent(0) == "r"
        assert rulebase.vocabulary == frozenset({"p", "q", "r", "s", "t", "goal"})

    def test_kernel_does_not_import_the_harness_at_module_level(self) -> None:
        # The verifier must not depend on the fixtures it verifies. Measured in
        # a clean subprocess: an in-process sys.modules check would only tell us
        # what this test runner happens to have loaded.
        code = "import sys, invariant_engine; print('harness_generators' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"

    def test_kernel_imports_no_third_party_package(self) -> None:
        code = (
            "import sys, invariant_engine;"
            "bad=[m for m in ('numpy','scipy','networkx','torch','pandas')"
            " if m in sys.modules];"
            "print(bad)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]"


# --------------------------------------------------------------------------- #
# admit_rules
# --------------------------------------------------------------------------- #


class TestAdmitRules:
    def test_clean_acyclic_proposal_is_admitted_in_full(self) -> None:
        rulebase, findings = admit_rules(CLAUSES)
        assert len(rulebase) == 3
        assert findings == ()

    def test_empty_proposal_yields_an_empty_base(self) -> None:
        rulebase, findings = admit_rules([])
        assert len(rulebase) == 0
        assert findings == ()

    @pytest.mark.parametrize(
        "junk",
        [
            None,
            42,
            "not a clause",
            {},
            {"antecedents": ["a"]},
            {"consequent": "b"},
            {"antecedents": ["a"], "consequent": ""},
            {"antecedents": ["a"], "consequent": 7},
            {"antecedents": "a", "consequent": "b"},
            {"antecedents": [7], "consequent": "b"},
            {"antecedents": [""], "consequent": "b"},
            {"antecedents": [None], "consequent": "b"},
        ],
    )
    def test_malformed_clauses_are_rejected_not_raised(self, junk: Any) -> None:
        # A kernel that crashes on bad input is a denial-of-service surface.
        rulebase, findings = admit_rules([junk])
        assert len(rulebase) == 0
        assert reasons(findings) == {Rejection.MALFORMED_CLAUSE}

    def test_malformed_clauses_do_not_take_good_ones_with_them(self) -> None:
        rulebase, findings = admit_rules([CLAUSES[0], None, CLAUSES[1]])
        assert len(rulebase) == 2
        assert [f.location for f in findings] == [1]

    def test_finding_locations_are_indices_into_the_original_proposal(self) -> None:
        rulebase, findings = admit_rules([None, CLAUSES[0], None, CLAUSES[1]])
        assert sorted(f.location for f in findings) == [0, 2]
        assert len(rulebase) == 2

    def test_extra_keys_are_ignored_not_trusted(self) -> None:
        clause = dict(CLAUSES[0], circular=True, limit_state="whatever", hallucinated=True)
        rulebase, findings = admit_rules([clause])
        assert len(rulebase) == 1
        assert findings == ()

    def test_hallucinated_atoms_are_rejected_against_a_vocabulary(self) -> None:
        vocabulary = {"p", "q", "r", "s", "t", "goal"}
        poisoned = dict(CLAUSES[0], consequent="phantom_predicate")
        rulebase, findings = admit_rules([*CLAUSES, poisoned], vocabulary=vocabulary)
        assert len(rulebase) == 3
        assert reasons(findings) == {Rejection.HALLUCINATED_ATOM}
        assert findings[0].location == 3
        assert "phantom_predicate" in findings[0].detail

    def test_hallucinated_antecedents_are_caught_too(self) -> None:
        vocabulary = {"p", "q", "r"}
        clause = {"antecedents": ["p", "invented"], "consequent": "r"}
        rulebase, findings = admit_rules([clause], vocabulary=vocabulary)
        assert len(rulebase) == 0
        assert reasons(findings) == {Rejection.HALLUCINATED_ATOM}

    def test_vocabulary_none_disables_the_check(self) -> None:
        clause = {"antecedents": ["anything"], "consequent": "at_all"}
        rulebase, findings = admit_rules([clause], vocabulary=None)
        assert len(rulebase) == 1
        assert findings == ()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_every_planted_hallucination_is_caught_and_no_clean_rule_is(
        self, seed: int
    ) -> None:
        vocabulary = [f"x{i}" for i in range(12)]
        clean = generate_horn_rules(40, 12, 0.0, seed=seed)
        poisoned = inject_rule_hallucinations(clean, vocabulary, 0.25, seed=seed)
        _, findings = admit_rules(poisoned, vocabulary=vocabulary)
        caught = {f.location for f in findings if f.reason is Rejection.HALLUCINATED_ATOM}
        planted = {i for i, r in enumerate(poisoned) if r["hallucinated"]}
        assert caught == planted

    def test_self_loops_are_rejected(self) -> None:
        rulebase, findings = admit_rules([{"antecedents": ["a"], "consequent": "a"}])
        assert len(rulebase) == 0
        assert reasons(findings) == {Rejection.CIRCULAR_RULE}

    def test_two_cycles_are_rejected(self) -> None:
        cycle = [
            {"antecedents": ["a"], "consequent": "b"},
            {"antecedents": ["b"], "consequent": "a"},
        ]
        rulebase, findings = admit_rules(cycle)
        assert len(rulebase) == 0
        assert reasons(findings) == {Rejection.CIRCULAR_RULE}

    def test_acyclic_clauses_survive_alongside_a_cycle(self) -> None:
        mixed = [
            {"antecedents": ["p", "q"], "consequent": "r"},  # keep
            {"antecedents": ["a"], "consequent": "b"},  # cycle
            {"antecedents": ["b"], "consequent": "a"},  # cycle
        ]
        rulebase, findings = admit_rules(mixed)
        assert len(rulebase) == 1
        assert rulebase.consequent(0) == "r"
        assert len(findings) == 2

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_admitted_base_is_always_acyclic(self, seed: int) -> None:
        # The property that matters: whatever the proposer sends, what comes out
        # of admission has no cycle in it.
        rulebase, _ = admit_rules(generate_horn_rules(40, 12, 0.5, seed=seed))
        graph: Dict[str, List[str]] = {}
        for rule_id in range(len(rulebase)):
            graph.setdefault(rulebase.consequent(rule_id), [])
            for antecedent in rulebase.antecedents(rule_id):
                graph.setdefault(antecedent, []).append(rulebase.consequent(rule_id))
        # Kahn: a full topological order exists iff the graph is acyclic.
        indegree = {n: 0 for n in graph}
        for children in graph.values():
            for child in children:
                indegree[child] += 1
        queue = [n for n, d in indegree.items() if d == 0]
        seen = 0
        while queue:
            node = queue.pop()
            seen += 1
            for child in graph.get(node, ()):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        assert seen == len(indegree)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_admission_never_invents_a_clause(self, seed: int) -> None:
        proposal = generate_horn_rules(40, 12, 0.25, seed=seed)
        rulebase, _ = admit_rules(proposal)
        offered = {(tuple(r["antecedents"]), r["consequent"]) for r in proposal}
        for rule_id in range(len(rulebase)):
            assert (rulebase.antecedents(rule_id), rulebase.consequent(rule_id)) in offered

    def test_reject_cycles_false_admits_a_cycle(self) -> None:
        cycle = [
            {"antecedents": ["a"], "consequent": "b"},
            {"antecedents": ["b"], "consequent": "a"},
        ]
        rulebase, findings = admit_rules(cycle, reject_cycles=False)
        assert len(rulebase) == 2
        assert findings == ()


# --------------------------------------------------------------------------- #
# admit_priors
# --------------------------------------------------------------------------- #


ADJACENCY = {"a": ["b"], "b": []}


class TestAdmitPriors:
    def test_clean_priors_are_admitted(self) -> None:
        clean, findings = admit_priors({"a": 0.3, "b": 0.1}, ADJACENCY)
        assert clean == {"a": 0.3, "b": 0.1}
        assert findings == ()

    @pytest.mark.parametrize("value", [1.0001, -0.0001, 2.0, -1.0])
    def test_out_of_range_priors_are_rejected(self, value: float) -> None:
        clean, findings = admit_priors({"a": value, "b": 0.1}, ADJACENCY)
        assert "a" not in clean
        assert Rejection.PRIOR_OUT_OF_RANGE in reasons(findings)

    @pytest.mark.parametrize("value", [0.0, 1.0, 0, 1])
    def test_the_closed_unit_interval_is_admissible(self, value: Any) -> None:
        # Bernoulli priors of exactly 0 or 1 are degenerate but well-formed.
        clean, _ = admit_priors({"a": value, "b": 0.1}, ADJACENCY)
        assert clean["a"] == float(value)

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf"), "0.5", None, [0.5], True, False]
    )
    def test_malformed_priors_are_rejected(self, value: Any) -> None:
        clean, findings = admit_priors({"a": value, "b": 0.1}, ADJACENCY)
        assert "a" not in clean
        assert Rejection.MALFORMED_PRIOR in reasons(findings)

    def test_booleans_are_not_probabilities(self) -> None:
        # bool is a subclass of int; accepting True as 1.0 would silently turn a
        # type error into a degenerate certainty.
        _, findings = admit_priors({"a": True, "b": 0.1}, ADJACENCY)
        assert Rejection.MALFORMED_PRIOR in reasons(findings)

    def test_priors_for_phantom_variables_are_rejected(self) -> None:
        clean, findings = admit_priors({"a": 0.3, "b": 0.1, "ghost": 0.5}, ADJACENCY)
        assert "ghost" not in clean
        assert Rejection.UNKNOWN_VARIABLE in reasons(findings)

    def test_missing_priors_are_reported_for_every_node(self) -> None:
        clean, findings = admit_priors({"a": 0.3}, ADJACENCY)
        missing = [f for f in findings if f.reason is Rejection.MISSING_PRIOR]
        assert [f.location for f in missing] == ["b"]
        assert clean == {"a": 0.3}

    def test_a_rejected_prior_also_reports_as_missing(self) -> None:
        # Rejecting a value leaves a hole, and the hole is what stops inference.
        _, findings = admit_priors({"a": 5.0, "b": 0.1}, ADJACENCY)
        assert {Rejection.PRIOR_OUT_OF_RANGE, Rejection.MISSING_PRIOR} <= reasons(findings)

    def test_nodes_seen_only_as_children_still_need_priors(self) -> None:
        _, findings = admit_priors({"a": 0.3}, {"a": ["child_only"]})
        missing = {f.location for f in findings if f.reason is Rejection.MISSING_PRIOR}
        assert missing == {"child_only"}

    @pytest.mark.parametrize("seed", SEEDS)
    def test_generated_priors_are_admitted_cleanly(self, seed: int) -> None:
        adjacency, priors = generate_random_dag(20, 4, 3, seed=seed)
        clean, findings = admit_priors(priors, adjacency)
        assert findings == ()
        assert clean == priors


# --------------------------------------------------------------------------- #
# parse_trace
# --------------------------------------------------------------------------- #


class TestParseTrace:
    def test_valid_steps_parse(self) -> None:
        steps, findings = parse_trace(
            [
                {"conclusion": "p", "rule_id": None, "support": []},
                {"conclusion": "r", "rule_id": 0, "support": ["p", "q"]},
            ]
        )
        assert findings == ()
        assert steps == (TraceStep("p", None, ()), TraceStep("r", 0, ("p", "q")))

    def test_support_defaults_to_empty(self) -> None:
        steps, findings = parse_trace([{"conclusion": "p", "rule_id": None}])
        assert findings == ()
        assert steps[0].support == ()

    @pytest.mark.parametrize(
        "step",
        [
            {},
            {"conclusion": "p"},
            {"rule_id": None},
            {"conclusion": "", "rule_id": None},
            {"conclusion": 7, "rule_id": None},
            {"conclusion": "p", "rule_id": "0"},
            {"conclusion": "p", "rule_id": 1.5},
            {"conclusion": "p", "rule_id": True},
            {"conclusion": "p", "rule_id": 0, "support": "pq"},
            {"conclusion": "p", "rule_id": 0, "support": [1, 2]},
        ],
    )
    def test_malformed_steps_are_reported_not_raised(self, step: Any) -> None:
        steps, findings = parse_trace([step])
        assert reasons(findings) == {Rejection.MALFORMED_STEP}
        assert steps == ()

    def test_parsing_aborts_at_the_first_malformed_step(self) -> None:
        steps, findings = parse_trace(
            [{"conclusion": "p", "rule_id": None}, {}, {"conclusion": "q", "rule_id": None}]
        )
        assert len(steps) == 1
        assert findings[0].location == 1


# --------------------------------------------------------------------------- #
# verify_trace -- the kernel's core
# --------------------------------------------------------------------------- #


class TestVerifyTrace:
    @staticmethod
    def sound() -> List[TraceStep]:
        return [
            TraceStep("p", None, ()),
            TraceStep("q", None, ()),
            TraceStep("r", 0, ("p", "q")),
            TraceStep("s", 1, ("r",)),
            TraceStep("t", None, ()),
            TraceStep("goal", 2, ("s", "t")),
        ]

    def test_a_sound_trace_is_accepted(self) -> None:
        result = verify_trace(self.sound(), admitted(), FACTS)
        assert result.accepted
        assert "goal" in result.established
        assert result.rejection is None

    def test_an_empty_trace_is_vacuously_accepted_and_proves_nothing(self) -> None:
        result = verify_trace([], admitted(), FACTS)
        assert result.accepted
        assert result.established == frozenset()

    def test_support_order_does_not_matter(self) -> None:
        steps = self.sound()
        steps[2] = TraceStep("r", 0, ("q", "p"))
        assert verify_trace(steps, admitted(), FACTS).accepted

    def test_duplicated_support_is_harmless(self) -> None:
        steps = self.sound()
        steps[2] = TraceStep("r", 0, ("p", "q", "p"))
        assert verify_trace(steps, admitted(), FACTS).accepted

    def test_premise_step_citing_a_non_fact_is_ungrounded(self) -> None:
        steps = self.sound()
        steps[0] = TraceStep("smuggled", None, ())
        result = verify_trace(steps, admitted(), FACTS)
        assert not result.accepted
        assert result.rejection.reason is Rejection.UNGROUNDED_PREMISE
        assert result.rejection.location == 0

    @pytest.mark.parametrize("rule_id", [3, 99, -1, -5])
    def test_citing_a_nonexistent_rule_is_a_hallucination(self, rule_id: int) -> None:
        steps = self.sound()
        steps[2] = TraceStep("r", rule_id, ("p", "q"))
        result = verify_trace(steps, admitted(), FACTS)
        assert not result.accepted
        assert result.rejection.reason is Rejection.HALLUCINATED_RULE_ID
        assert result.rejection.location == 2

    def test_conclusion_that_is_not_the_rules_consequent_is_rejected(self) -> None:
        steps = self.sound()
        steps[2] = TraceStep("s", 0, ("p", "q"))
        result = verify_trace(steps, admitted(), FACTS)
        assert not result.accepted
        assert result.rejection.reason is Rejection.CONCLUSION_MISMATCH

    @pytest.mark.parametrize(
        "support", [("p",), ("p", "q", "t"), ("t",), (), ("p", "wrong")]
    )
    def test_support_must_match_the_rules_antecedents_exactly(
        self, support: Tuple[str, ...]
    ) -> None:
        steps = self.sound()
        steps[2] = TraceStep("r", 0, support)
        result = verify_trace(steps, admitted(), FACTS)
        assert not result.accepted
        assert result.rejection.reason is Rejection.ANTECEDENT_MISMATCH

    def test_support_established_only_by_a_later_step_is_unfounded(self) -> None:
        # This is the clause that makes circular support unprovable.
        steps = self.sound()
        rotated = [steps[-1]] + steps[:-1]
        result = verify_trace(rotated, admitted(), FACTS)
        assert not result.accepted
        assert result.rejection.reason is Rejection.UNFOUNDED_SUPPORT
        assert result.rejection.location == 0

    def test_a_fact_still_needs_an_explicit_premise_step(self) -> None:
        # A trace must stand on its own rather than leaning on ambient context,
        # so citing a trusted fact without introducing it is still unfounded.
        steps = [TraceStep("r", 0, ("p", "q"))]
        result = verify_trace(steps, admitted(), FACTS)
        assert not result.accepted
        assert result.rejection.reason is Rejection.UNFOUNDED_SUPPORT
        assert "p" in result.rejection.detail

    def test_checking_stops_at_the_first_bad_step(self) -> None:
        steps = self.sound()
        steps[0] = TraceStep("smuggled", None, ())  # defect at 0
        steps[3] = TraceStep("s", 99, ("r",))  # defect at 3
        result = verify_trace(steps, admitted(), FACTS)
        assert result.rejection.location == 0
        assert result.rejection.reason is Rejection.UNGROUNDED_PREMISE

    def test_established_holds_what_was_proved_before_the_failure(self) -> None:
        steps = self.sound()
        steps[3] = TraceStep("s", 99, ("r",))
        result = verify_trace(steps, admitted(), FACTS)
        assert not result.accepted
        assert result.established == frozenset({"p", "q", "r"})

    def test_a_rule_with_no_antecedents_is_an_axiom_step(self) -> None:
        rulebase = admitted([{"antecedents": [], "consequent": "axiom"}])
        result = verify_trace([TraceStep("axiom", 0, ())], rulebase, [])
        assert result.accepted
        assert result.established == frozenset({"axiom"})


class TestVerifyTraceAgainstGeneratedDefects:
    """Every defect class the harness can produce must be rejected, and no
    sound trace may ever be."""

    @staticmethod
    def setup_case(seed: int, defect_rate: float = 0.6, n: int = 200):
        clauses = generate_horn_rules(40, 12, 0.0, seed=seed)
        rulebase, findings = admit_rules(clauses)
        # Rule ids are positional. With an acyclic proposal nothing is dropped,
        # so trace ids line up with the admitted base -- which is exactly why
        # traces must be generated against the base the kernel will use.
        assert not findings and len(rulebase) == len(clauses)
        facts = sorted({r["consequent"] for r in clauses if not r["antecedents"]})
        traces = generate_candidate_traces(
            clauses, facts, n, defect_rate=defect_rate, seed=seed
        )
        return rulebase, facts, traces

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_sound_trace_is_ever_rejected(self, seed: int) -> None:
        rulebase, facts, traces = self.setup_case(seed)
        sound = [t for t in traces if t["defect"] is None]
        assert sound, "fixture produced no sound traces to check"
        for trace in sound:
            steps, findings = parse_trace(trace["steps"])
            assert findings == ()
            result = verify_trace(steps, rulebase, facts)
            assert result.accepted, (trace["goal"], result.rejection)
            assert trace["goal"] in result.established

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_defective_trace_is_ever_accepted(self, seed: int) -> None:
        rulebase, facts, traces = self.setup_case(seed)
        defective = [t for t in traces if t["defect"] is not None]
        assert defective, "fixture produced no defective traces to check"
        for trace in defective:
            steps, findings = parse_trace(trace["steps"])
            if findings:
                continue  # rejected at the parse gate, which also counts
            result = verify_trace(steps, rulebase, facts)
            assert not (
                result.accepted and trace["goal"] in result.established
            ), f"{trace['defect']} trace was accepted: {trace}"

    def test_every_defect_class_is_actually_exercised(self) -> None:
        # Guards against the suite passing because the fixture stopped producing
        # defects at all.
        _, _, traces = self.setup_case(0, defect_rate=1.0, n=400)
        produced = {t["defect"] for t in traces if t["defect"]}
        assert len(produced) == 5, produced

    @pytest.mark.parametrize(
        ("defect", "expected"),
        [
            ("hallucinated_rule", Rejection.HALLUCINATED_RULE_ID),
            ("ungrounded_premise", Rejection.UNGROUNDED_PREMISE),
            ("circular_support", Rejection.UNFOUNDED_SUPPORT),
        ],
    )
    def test_the_rejection_reason_identifies_the_actual_defect(
        self, defect: str, expected: Rejection
    ) -> None:
        # Rejecting for the wrong reason is a debugging trap: it sends the
        # operator after the wrong failure.
        rulebase, facts, traces = self.setup_case(1, defect_rate=1.0, n=400)
        matching = [t for t in traces if t["defect"] == defect]
        assert matching, f"fixture produced no {defect} traces"
        for trace in matching:
            steps, _ = parse_trace(trace["steps"])
            result = verify_trace(steps, rulebase, facts)
            assert not result.accepted
            assert result.rejection.reason is expected, (defect, result.rejection)


# --------------------------------------------------------------------------- #
# derive
# --------------------------------------------------------------------------- #


class TestDerive:
    def test_derives_a_chain_and_emits_it_in_dependency_order(self) -> None:
        proof = derive("goal", admitted(), FACTS)
        assert proof is not None
        assert proof[-1].conclusion == "goal"
        assert len(proof) == 6

    def test_a_base_fact_derives_as_a_single_premise_step(self) -> None:
        proof = derive("p", admitted(), FACTS)
        assert proof == (TraceStep("p", None, ()),)

    def test_returns_none_for_an_underivable_goal(self) -> None:
        assert derive("goal", admitted(), ["p"]) is None
        assert derive("nonexistent", admitted(), FACTS) is None

    def test_none_means_not_established_not_false(self) -> None:
        # Adding the missing premise establishes it, so the earlier None was
        # ignorance, not refutation.
        assert derive("goal", admitted(), ["p", "q"]) is None
        assert derive("goal", admitted(), FACTS) is not None

    def test_a_shared_subterm_is_emitted_once(self) -> None:
        clauses = [
            {"antecedents": ["a"], "consequent": "b"},
            {"antecedents": ["a"], "consequent": "c"},
            {"antecedents": ["b", "c"], "consequent": "d"},
        ]
        proof = derive("d", admitted(clauses), ["a"])
        conclusions = [s.conclusion for s in proof]
        assert conclusions.count("a") == 1
        assert len(conclusions) == len(set(conclusions))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_every_kernel_proof_verifies(self, seed: int) -> None:
        # The kernel grades itself here, and must not be able to mint a proof it
        # would reject from anyone else.
        clauses = generate_horn_rules(40, 12, 0.0, seed=seed)
        rulebase, _ = admit_rules(clauses)
        facts = sorted({r["consequent"] for r in clauses if not r["antecedents"]})
        for atom in sorted(rulebase.vocabulary):
            proof = derive(atom, rulebase, facts)
            if proof is None:
                continue
            result = verify_trace(proof, rulebase, facts)
            assert result.accepted, (atom, result.rejection)
            assert atom in result.established

    def test_a_cyclic_base_admitted_deliberately_cannot_prove_its_cycle(self) -> None:
        # With reject_cycles=False the fixpoint contains a and b, but neither has
        # a well-founded derivation, so derive() must return None rather than
        # dressing the fixpoint up as a proof.
        cycle = [
            {"antecedents": ["a"], "consequent": "b"},
            {"antecedents": ["b"], "consequent": "a"},
        ]
        rulebase, _ = admit_rules(cycle, reject_cycles=False)
        assert derive("a", rulebase, []) is None
        assert derive("b", rulebase, []) is None


# --------------------------------------------------------------------------- #
# check_goal -- the soundness property
# --------------------------------------------------------------------------- #


class TestCheckGoal:
    def test_verified_without_a_proposal(self) -> None:
        report = check_goal("goal", admitted(), FACTS)
        assert report.status is Status.VERIFIED
        assert report.proposal_accepted is None
        assert report.proof_source == "kernel"

    def test_unknown_without_a_proposal(self) -> None:
        report = check_goal("goal", admitted(), ["p"])
        assert report.status is Status.UNKNOWN
        assert report.proof == ()
        assert report.proof_source == "none"

    def test_a_sound_proposal_is_used_as_the_proof(self) -> None:
        proposal = TestVerifyTrace.sound()
        report = check_goal("goal", admitted(), FACTS, proposal=proposal)
        assert report.status is Status.VERIFIED
        assert report.proposal_accepted is True
        assert report.proof_source == "proposal"
        assert report.proof == tuple(proposal)

    def test_a_defective_proposal_falls_back_to_the_kernel(self) -> None:
        broken = TestVerifyTrace.sound()
        broken[2] = TraceStep("r", 99, ("p", "q"))
        report = check_goal("goal", admitted(), FACTS, proposal=broken)
        assert report.status is Status.VERIFIED
        assert report.proposal_accepted is False
        assert report.proof_source == "kernel"
        assert Rejection.HALLUCINATED_RULE_ID in reasons(report.findings)

    def test_a_sound_proposal_for_the_wrong_goal_is_not_accepted(self) -> None:
        proposal = [TraceStep("p", None, ())]  # sound, but proves only "p"
        report = check_goal("goal", admitted(), FACTS, proposal=proposal)
        assert report.proposal_accepted is False
        assert Rejection.CONCLUSION_MISMATCH in reasons(report.findings)
        assert report.status is Status.VERIFIED  # the kernel derives it anyway
        assert report.proof_source == "kernel"

    def test_a_defective_proposal_for_an_underivable_goal_stays_unknown(self) -> None:
        broken = [TraceStep("goal", 99, ())]
        report = check_goal("goal", admitted(), ["p"], proposal=broken)
        assert report.status is Status.UNKNOWN
        assert report.proposal_accepted is False


class TestCheckGoalSoundness:
    """The property the whole design exists to provide."""

    @staticmethod
    def setup_case(seed: int):
        return TestVerifyTraceAgainstGeneratedDefects.setup_case(seed)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_proposal_can_never_change_a_verdict(self, seed: int) -> None:
        # The crispest statement of the threat model: whatever the untrusted
        # proposer sends -- sound, hallucinated, circular, ungrounded -- the
        # verdict is identical to the one reached with no proposal at all. The
        # proposal can only save work or be discarded.
        rulebase, facts, traces = self.setup_case(seed)
        for trace in traces:
            steps, findings = parse_trace(trace["steps"])
            if findings:
                continue
            goal = trace["goal"]
            without = check_goal(goal, rulebase, facts)
            with_proposal = check_goal(goal, rulebase, facts, proposal=steps)
            assert with_proposal.status is without.status, trace["defect"]

    @pytest.mark.parametrize("seed", SEEDS)
    def test_every_verified_verdict_carries_a_proof_that_verifies(
        self, seed: int
    ) -> None:
        rulebase, facts, traces = self.setup_case(seed)
        for trace in traces:
            steps, findings = parse_trace(trace["steps"])
            if findings:
                continue
            report = check_goal(trace["goal"], rulebase, facts, proposal=steps)
            if report.status is not Status.VERIFIED:
                continue
            result = verify_trace(report.proof, rulebase, facts)
            assert result.accepted
            assert report.goal in result.established

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_defective_proposal_is_never_the_accepted_proof(self, seed: int) -> None:
        rulebase, facts, traces = self.setup_case(seed)
        checked = 0
        for trace in traces:
            if trace["defect"] is None:
                continue
            steps, findings = parse_trace(trace["steps"])
            if findings:
                continue
            report = check_goal(trace["goal"], rulebase, facts, proposal=steps)
            assert report.proposal_accepted is False
            assert report.proof_source != "proposal"
            checked += 1
        assert checked, "no defective traces were exercised"


# --------------------------------------------------------------------------- #
# Probabilistic layer
# --------------------------------------------------------------------------- #


class TestNoisyOr:
    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
    def test_rejects_an_out_of_range_activation(self, bad: float) -> None:
        with pytest.raises(ValueError, match="edge_activation"):
            NoisyOr(bad)

    def test_no_true_parent_leaves_the_leak(self) -> None:
        assert NoisyOr(0.8).conditional(0.05, 0) == pytest.approx(0.05)

    def test_more_true_parents_never_lowers_the_probability(self) -> None:
        policy = NoisyOr(0.6)
        values = [policy.conditional(0.05, k) for k in range(5)]
        assert values == sorted(values)

    def test_zero_activation_makes_parents_irrelevant(self) -> None:
        policy = NoisyOr(0.0)
        assert policy.conditional(0.3, 4) == pytest.approx(0.3)

    def test_full_activation_makes_one_true_parent_sufficient(self) -> None:
        assert NoisyOr(1.0).conditional(0.05, 1) == pytest.approx(1.0)

    def test_a_certain_leak_stays_certain(self) -> None:
        assert NoisyOr(0.5).conditional(1.0, 3) == pytest.approx(1.0)


class TestFactor:
    def test_rejects_a_table_of_the_wrong_size(self) -> None:
        with pytest.raises(ValueError, match="table size"):
            Factor(("a", "b"), (0.5, 0.5))

    def test_multiply_over_the_same_variable(self) -> None:
        product = Factor(("a",), (0.3, 0.7)).multiply(Factor(("a",), (0.5, 0.5)))
        assert product.variables == ("a",)
        assert product.table == pytest.approx((0.15, 0.35))

    def test_multiply_unions_the_variables(self) -> None:
        product = Factor(("a",), (0.4, 0.6)).multiply(Factor(("b",), (0.1, 0.9)))
        assert set(product.variables) == {"a", "b"}
        assert sum(product.table) == pytest.approx(1.0)

    def test_sum_out_reduces_width_and_totals_correctly(self) -> None:
        reduced = Factor(("a", "b"), (0.1, 0.2, 0.3, 0.4)).sum_out("a")
        assert reduced.variables == ("b",)
        assert reduced.table == pytest.approx((0.4, 0.6))

    def test_sum_out_of_an_absent_variable_is_the_identity(self) -> None:
        factor = Factor(("a",), (0.3, 0.7))
        assert factor.sum_out("z") is factor

    def test_restrict_selects_one_slice(self) -> None:
        restricted = Factor(("a", "b"), (0.1, 0.2, 0.3, 0.4)).restrict("a", True)
        assert restricted.variables == ("b",)
        assert restricted.table == pytest.approx((0.3, 0.4))


def brute_force_marginal(
    adjacency: Mapping[str, Sequence[str]],
    priors: Mapping[str, float],
    target: str,
    policy: NoisyOr,
    evidence: Optional[Mapping[str, bool]] = None,
) -> float:
    """Enumerate the full joint distribution. The oracle for :func:`marginal`.

    Deliberately shares no machinery with the engine: no factors, no elimination
    order, just 2**n assignments and the CPD definition. Exponential and
    therefore only usable on small graphs, which is the point -- it is the thing
    variable elimination has to agree with.
    """
    nodes = set(adjacency)
    for children in adjacency.values():
        nodes |= set(children)
    order = sorted(nodes)
    parents: Dict[str, List[str]] = {n: [] for n in order}
    for node, children in adjacency.items():
        for child in children:
            parents[child].append(node)

    total = 0.0
    hit = 0.0
    for pattern in itertools.product([False, True], repeat=len(order)):
        assignment = dict(zip(order, pattern))
        if evidence and any(assignment[k] != v for k, v in evidence.items()):
            continue
        probability = 1.0
        for node in order:
            node_parents = parents[node]
            if node_parents:
                p_true = policy.conditional(
                    priors[node], sum(assignment[p] for p in node_parents)
                )
            else:
                p_true = priors[node]
            probability *= p_true if assignment[node] else 1.0 - p_true
        total += probability
        if assignment[target]:
            hit += probability
    return hit / total


class TestMarginal:
    def test_a_lone_root_returns_its_prior(self) -> None:
        p, stats = marginal({"a": []}, {"a": 0.37}, "a")
        assert p == pytest.approx(0.37)
        assert stats.max_factor_width <= 1

    def test_a_two_node_chain_matches_the_hand_computation(self) -> None:
        adjacency = {"a": ["b"], "b": []}
        priors = {"a": 0.4, "b": 0.05}
        policy = NoisyOr(0.8)
        expected = 0.4 * policy.conditional(0.05, 1) + 0.6 * 0.05
        p, _ = marginal(adjacency, priors, "b", policy=policy)
        assert p == pytest.approx(expected)

    @pytest.mark.parametrize("n", [4, 5, 6, 7, 10, 13])
    def test_diamond_marginals_match_brute_force_enumeration(self, n: int) -> None:
        adjacency = generate_diamond_graph(n)
        priors = {node: 0.2 if node == "v0" else 0.1 for node in adjacency}
        policy = NoisyOr(0.7)
        for target in (f"v{n - 1}", "v0", "v1"):
            exact, _ = marginal(adjacency, priors, target, policy=policy)
            oracle = brute_force_marginal(adjacency, priors, target, policy)
            assert exact == pytest.approx(oracle, abs=1e-12), target

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_dag_marginals_match_brute_force_enumeration(self, seed: int) -> None:
        adjacency, priors = generate_random_dag(12, 4, 3, seed=seed)
        policy = NoisyOr(0.65)
        for target in sorted(adjacency)[:4]:
            exact, _ = marginal(adjacency, priors, target, policy=policy)
            oracle = brute_force_marginal(adjacency, priors, target, policy)
            assert exact == pytest.approx(oracle, abs=1e-12), target

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_conditional_marginals_match_brute_force_enumeration(self, seed: int) -> None:
        adjacency, priors = generate_random_dag(10, 3, 3, seed=seed)
        policy = NoisyOr(0.55)
        evidence = {"v0": True}
        for target in sorted(adjacency)[-3:]:
            exact, _ = marginal(
                adjacency, priors, target, evidence=evidence, policy=policy
            )
            oracle = brute_force_marginal(adjacency, priors, target, policy, evidence)
            assert exact == pytest.approx(oracle, abs=1e-12), target

    def test_observing_a_variable_makes_it_certain(self) -> None:
        adjacency = generate_diamond_graph(7)
        priors = {node: 0.3 for node in adjacency}
        p_true, _ = marginal(adjacency, priors, "v3", evidence={"v3": True})
        p_false, _ = marginal(adjacency, priors, "v3", evidence={"v3": False})
        assert p_true == pytest.approx(1.0)
        assert p_false == pytest.approx(0.0)

    def test_evidence_at_the_source_raises_the_sink(self) -> None:
        adjacency = generate_diamond_graph(13)
        priors = {node: 0.2 if node == "v0" else 0.1 for node in adjacency}
        free, _ = marginal(adjacency, priors, "v12")
        given, _ = marginal(adjacency, priors, "v12", evidence={"v0": True})
        assert given > free

    @pytest.mark.parametrize("n", [13, 31, 61, 151])
    def test_induced_width_stays_bounded_as_paths_explode(self, n: int) -> None:
        # 2**((n-1)//3) source-to-sink paths; cost must track width, not paths.
        adjacency = generate_diamond_graph(n)
        priors = {node: 0.2 if node == "v0" else 0.1 for node in adjacency}
        p, stats = marginal(adjacency, priors, f"v{n - 1}")
        assert 0.0 <= p <= 1.0
        assert stats.max_factor_width <= 4
        assert stats.max_table_entries <= 16

    def test_a_150_node_diamond_is_answered_at_all(self) -> None:
        # 2**50 paths. If the implementation ever regresses to enumeration this
        # test does not fail, it hangs -- which is its own clear signal.
        adjacency = generate_diamond_graph(151)
        priors = {node: 0.1 for node in adjacency}
        p, stats = marginal(adjacency, priors, "v150")
        assert 0.0 <= p <= 1.0
        assert stats.eliminated >= 100

    def test_rejects_an_unknown_target(self) -> None:
        with pytest.raises(ValueError, match="not a node"):
            marginal({"a": []}, {"a": 0.5}, "ghost")

    def test_refuses_to_invent_a_missing_prior(self) -> None:
        # Defaulting here would put a fabricated number into every downstream
        # marginal, which is precisely what admit_priors exists to prevent.
        with pytest.raises(ValueError, match="MISSING_PRIOR"):
            marginal({"a": ["b"], "b": []}, {"a": 0.5}, "b")

    def test_rejects_evidence_for_an_unknown_variable(self) -> None:
        with pytest.raises(ValueError, match="not in the graph"):
            marginal({"a": []}, {"a": 0.5}, "a", evidence={"ghost": True})

    def test_the_policy_is_an_engine_choice_that_changes_the_answer(self) -> None:
        # Documented caveat: edge_activation is an assumption, not fixture data.
        adjacency = {"a": ["b"], "b": []}
        priors = {"a": 0.5, "b": 0.05}
        weak, _ = marginal(adjacency, priors, "b", policy=NoisyOr(0.1))
        strong, _ = marginal(adjacency, priors, "b", policy=NoisyOr(0.9))
        assert strong > weak

    def test_is_deterministic(self) -> None:
        adjacency, priors = generate_random_dag(12, 4, 3, seed=0)
        first, _ = marginal(adjacency, priors, "v11")
        second, _ = marginal(adjacency, priors, "v11")
        assert first == second


# --------------------------------------------------------------------------- #
# Envelope monitor
# --------------------------------------------------------------------------- #


ENVELOPE = [
    {"antecedents": ["too_fast"], "consequent": "overspeed"},
    {"antecedents": ["overspeed"], "consequent": "envelope_exceedance"},
]


class TestEnvelopeMonitor:
    def test_a_violation_is_reported_with_a_proof_and_a_limit_state(self) -> None:
        report = check_envelope(
            "q", ["too_fast"], admitted(ENVELOPE), limit_state_atoms=["overspeed"]
        )
        assert report.verdict is Verdict.VIOLATED
        assert report.violated_limit_states == ("overspeed",)
        assert report.proof

    def test_no_derivable_violation_is_unknown_by_default(self) -> None:
        # Absence of proof is not proof of absence.
        report = check_envelope("q", ["slow"], admitted(ENVELOPE))
        assert report.verdict is Verdict.UNKNOWN
        assert report.proof == ()

    def test_safe_requires_an_explicit_closed_world_assumption(self) -> None:
        report = check_envelope("q", ["slow"], admitted(ENVELOPE), closed_world=True)
        assert report.verdict is Verdict.SAFE

    def test_closed_world_cannot_turn_a_violation_into_safety(self) -> None:
        report = check_envelope("q", ["too_fast"], admitted(ENVELOPE), closed_world=True)
        assert report.verdict is Verdict.VIOLATED

    @pytest.mark.parametrize("closed_world", [True, False])
    def test_every_verdict_records_its_assumptions(self, closed_world: bool) -> None:
        for atoms in (["too_fast"], ["slow"]):
            report = check_envelope(
                "q", atoms, admitted(ENVELOPE), closed_world=closed_world
            )
            assert report.assumptions

    def test_the_closed_world_assumption_is_stated_not_implied(self) -> None:
        report = check_envelope("q", ["slow"], admitted(ENVELOPE), closed_world=True)
        assert any("closed-world" in a for a in report.assumptions)

    def test_atoms_are_deduplicated_and_sorted(self) -> None:
        report = check_envelope("q", ["b", "a", "b"], admitted(ENVELOPE))
        assert report.atoms == ("a", "b")

    def test_a_defective_proposal_cannot_manufacture_a_violation(self) -> None:
        # The safety-critical direction of the soundness property.
        forged = [TraceStep("envelope_exceedance", 99, ())]
        report = check_envelope(
            "q", ["slow"], admitted(ENVELOPE), closed_world=True, proposal=forged
        )
        assert report.verdict is Verdict.SAFE
        assert Rejection.HALLUCINATED_RULE_ID in reasons(report.findings)

    def test_a_sound_proposal_reaches_the_same_violation(self) -> None:
        proposal = [
            TraceStep("too_fast", None, ()),
            TraceStep("overspeed", 0, ("too_fast",)),
            TraceStep("envelope_exceedance", 1, ("overspeed",)),
        ]
        report = check_envelope(
            "q", ["too_fast"], admitted(ENVELOPE), proposal=proposal
        )
        assert report.verdict is Verdict.VIOLATED


# --------------------------------------------------------------------------- #
# Aero fixture adapter, end to end
# --------------------------------------------------------------------------- #

FIXTURE = generate_aero_telemetry_fixture()
QUERY_IDS = [q["name"] for q in FIXTURE["queries"]]


class TestAeroAdapter:
    def test_every_query_verdict_matches_the_fixture_ground_truth(self) -> None:
        expected = {q["name"]: q["expected_safe"] for q in FIXTURE["queries"]}
        for report in check_aero_fixture(FIXTURE):
            want = Verdict.SAFE if expected[report.name] else Verdict.VIOLATED
            assert report.verdict is want, report.name

    def test_every_violation_carries_a_proof_that_verifies(self) -> None:
        rulebase, findings = admit_rules(FIXTURE["rules"])
        assert not findings
        for report in check_aero_fixture(FIXTURE):
            if report.verdict is not Verdict.VIOLATED:
                continue
            result = verify_trace(report.proof, rulebase, report.atoms)
            assert result.accepted, (report.name, result.rejection)
            assert "envelope_exceedance" in result.established

    def test_every_violation_names_at_least_one_limit_state(self) -> None:
        # A violation that cannot say which limit it breached is not actionable.
        for report in check_aero_fixture(FIXTURE):
            if report.verdict is Verdict.VIOLATED:
                assert report.violated_limit_states, report.name

    def test_open_world_downgrades_safe_to_unknown_and_leaves_violations_alone(
        self,
    ) -> None:
        closed = {r.name: r.verdict for r in check_aero_fixture(FIXTURE)}
        open_world = {
            r.name: r.verdict for r in check_aero_fixture(FIXTURE, closed_world=False)
        }
        for name, verdict in closed.items():
            if verdict is Verdict.SAFE:
                assert open_world[name] is Verdict.UNKNOWN
            else:
                assert open_world[name] is verdict

    def test_the_fixture_rule_set_is_admitted_without_a_single_rejection(self) -> None:
        _, findings = admit_rules(FIXTURE["rules"])
        assert findings == ()

    def test_a_cyclic_aero_rule_set_is_refused_by_the_adapter(self) -> None:
        poisoned = dict(FIXTURE)
        poisoned["rules"] = list(FIXTURE["rules"]) + [
            {
                "antecedents": ["envelope_exceedance"],
                "consequent": "aerodynamic_stall",
                "limit_state": "aerodynamic_stall",
                "basis": "injected cycle",
            }
        ]
        with pytest.raises(ValueError, match="admit cleanly"):
            check_aero_fixture(poisoned)
