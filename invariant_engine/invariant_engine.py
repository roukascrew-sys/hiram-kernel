"""HIRAM v4.0 -- trusted verification kernel for neuro-symbolic invariants.

Standard library only. No solver bindings, no tensor library, no network calls.

Threat model
------------
Everything the proposal engine produces is UNTRUSTED. That includes candidate
deduction traces, root priors, and autoformalised rule dictionaries -- exactly
the three outputs :mod:`harness_generators` mocks. The proposer may be sound,
confused, or adversarial, and the kernel cannot tell which by inspection. So it
does not try: it re-checks everything it is handed against its own rules.

Two consequences run through the whole module:

1. **A proposal can never raise a verdict to VERIFIED.** Only a kernel-checked
   proof does that. A trace is a *hint* about where to look; if it is sound the
   kernel confirms it, and if it is not the kernel derives the goal itself or
   returns UNKNOWN. :func:`check_goal` asserts this invariant on every call.
2. **Untrusted data cannot enter the kernel by construction.** :class:`RuleBase`
   refuses to be built directly; the only way in is :func:`admit_rules`, which
   validates shape, vocabulary, and acyclicity first. This is a code-level gate,
   not a convention someone has to remember.

What counts as the trusted computing base
-----------------------------------------
The TCB is the section marked ``TRUSTED KERNEL`` below: :func:`admit_rules`,
:func:`admit_priors`, :func:`verify_trace`, :func:`derive`, and
:func:`check_goal`. Everything after it -- probabilistic inference, the envelope
monitor, the fixture adapters -- is built on those and adds no new trust. Audit
the kernel and you have audited what can produce a VERIFIED.

Absence of proof is not proof of absence
----------------------------------------
:func:`check_envelope` will not report SAFE merely because it failed to derive a
violation. Under the open-world default that is UNKNOWN. SAFE requires the
caller to assert ``closed_world=True``, and the assumption is then recorded in
the report rather than left implicit. A monitor that says SAFE by default is
worse than no monitor.

Path explosion
--------------
Probabilistic queries use exact variable elimination, never path enumeration.
The chained-diamond fixtures have 2**k source-to-sink paths; the kernel answers
them in time linear in node count with factors bounded by the graph's induced
width. :func:`marginal` reports that width so the claim is checkable rather than
asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

__all__ = [
    "Rejection",
    "Finding",
    "RuleBase",
    "TraceStep",
    "TraceResult",
    "Status",
    "GoalReport",
    "admit_rules",
    "admit_priors",
    "parse_trace",
    "verify_trace",
    "derive",
    "check_goal",
    "NoisyOr",
    "Factor",
    "EliminationStats",
    "marginal",
    "Verdict",
    "InvariantReport",
    "check_envelope",
    "check_aero_fixture",
]


# =========================================================================== #
#                              TRUSTED KERNEL                                 #
# =========================================================================== #


class Rejection(Enum):
    """Why the kernel refused something. Each value is a distinct failure mode,
    so a caller can tell a hallucinated rule from a circular one."""

    MALFORMED_CLAUSE = "malformed clause"
    HALLUCINATED_ATOM = "atom outside the declared vocabulary"
    CIRCULAR_RULE = "clause participates in a dependency cycle"
    MALFORMED_STEP = "malformed trace step"
    HALLUCINATED_RULE_ID = "trace cites a rule that does not exist"
    CONCLUSION_MISMATCH = "step conclusion is not the cited rule's consequent"
    ANTECEDENT_MISMATCH = "cited support is not the cited rule's antecedents"
    UNFOUNDED_SUPPORT = "support not established by a strictly earlier step"
    UNGROUNDED_PREMISE = "premise step cites a non-fact"
    MALFORMED_PRIOR = "prior is not a finite real number"
    PRIOR_OUT_OF_RANGE = "prior outside [0, 1]"
    UNKNOWN_VARIABLE = "prior given for a variable not in the graph"
    MISSING_PRIOR = "no prior supplied for a variable in the graph"


@dataclass(frozen=True)
class Finding:
    """One rejection, with enough detail to act on without re-running."""

    reason: Rejection
    detail: str
    location: Any = None

    def __str__(self) -> str:
        where = "" if self.location is None else f" at {self.location!r}"
        return f"{self.reason.value}{where}: {self.detail}"


_ADMISSION = object()
"""Private token. Possessing it is what distinguishes admitted clauses from
untrusted ones, and only :func:`admit_rules` has it."""


@dataclass(frozen=True)
class RuleBase:
    """An immutable, validated set of definite clauses.

    Cannot be constructed directly -- :func:`admit_rules` is the only door in.
    That is deliberate: a `RuleBase` in hand is a proof that shape, vocabulary,
    and acyclicity were all checked, and no amount of downstream carelessness
    can forge one.
    """

    clauses: Tuple[Tuple[Tuple[str, ...], str], ...]
    vocabulary: FrozenSet[str]
    _token: Any = None

    def __post_init__(self) -> None:
        if self._token is not _ADMISSION:
            raise TypeError(
                "RuleBase must be built by admit_rules(); untrusted clauses "
                "cannot enter the kernel by direct construction"
            )

    def __len__(self) -> int:
        return len(self.clauses)

    def antecedents(self, rule_id: int) -> Tuple[str, ...]:
        return self.clauses[rule_id][0]

    def consequent(self, rule_id: int) -> str:
        return self.clauses[rule_id][1]


def _scc(graph: Mapping[str, Sequence[str]]) -> List[List[str]]:
    """Tarjan's algorithm, iterative. Kept local so the kernel depends on
    nothing outside the standard library."""
    nodes: List[str] = list(graph)
    seen: Set[str] = set(nodes)
    for children in graph.values():
        for child in children:
            if child not in seen:
                seen.add(child)
                nodes.append(child)

    index_of: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    out: List[List[str]] = []
    counter = 0

    for root in nodes:
        if root in index_of:
            continue
        work: List[Tuple[str, int]] = [(root, 0)]
        while work:
            node, child_index = work.pop()
            if child_index == 0:
                index_of[node] = lowlink[node] = counter
                counter += 1
                stack.append(node)
                on_stack[node] = True
            recursed = False
            children = graph.get(node, ())
            while child_index < len(children):
                child = children[child_index]
                child_index += 1
                if child not in index_of:
                    work.append((node, child_index))
                    work.append((child, 0))
                    recursed = True
                    break
                if on_stack.get(child, False):
                    lowlink[node] = min(lowlink[node], index_of[child])
            if recursed:
                continue
            if lowlink[node] == index_of[node]:
                component: List[str] = []
                while True:
                    popped = stack.pop()
                    on_stack[popped] = False
                    component.append(popped)
                    if popped == node:
                        break
                out.append(component)
            if work:
                lowlink[work[-1][0]] = min(lowlink[work[-1][0]], lowlink[node])
    return out


def admit_rules(
    candidates: Sequence[Mapping[str, Any]],
    *,
    vocabulary: Optional[Iterable[str]] = None,
    reject_cycles: bool = True,
) -> Tuple[RuleBase, Tuple[Finding, ...]]:
    """Validate untrusted clause dictionaries and admit the survivors.

    Never raises on bad input. Rejecting a proposal is normal operation, not an
    error: the proposer is expected to be wrong sometimes, and a kernel that
    crashes on malformed input is a denial-of-service surface.

    Three gates, in order:

    1. **Shape.** ``antecedents`` must be a list of strings, ``consequent`` a
       non-empty string. Extra keys (``circular``, ``limit_state``, ...) are
       ignored, not trusted.
    2. **Vocabulary.** If ``vocabulary`` is given, every atom must be in it.
       This is what catches an autoformaliser inventing predicates: a
       hallucinated atom can never be established and never refuted, so it
       silently poisons every rule it touches.
    3. **Acyclicity.** Clauses closing a dependency cycle are dropped, since
       cyclic support is unfounded support. Dropping is iterative -- removing
       one clause can dissolve other cycles -- and terminates because each pass
       removes at least one clause.

    Args:
        candidates: Untrusted clause dictionaries, e.g. from
            ``generate_horn_rules`` or ``inject_rule_hallucinations``.
        vocabulary: Declared legitimate atoms. ``None`` disables the check,
            which is the right choice only when the domain is genuinely open.
        reject_cycles: Leave True unless you are deliberately studying cyclic
            rule sets; a `RuleBase` admitted with False can produce fixpoints
            that are not proofs.

    Returns:
        ``(rulebase, findings)``. ``findings`` is empty iff every candidate was
        admitted, and each finding's ``location`` is the candidate's index.
    """
    declared = None if vocabulary is None else set(vocabulary)
    findings: List[Finding] = []
    survivors: List[Tuple[int, Tuple[str, ...], str]] = []

    for index, candidate in enumerate(candidates):
        try:
            antecedents = candidate["antecedents"]
            consequent = candidate["consequent"]
        except (KeyError, TypeError):
            findings.append(
                Finding(Rejection.MALFORMED_CLAUSE, "missing antecedents/consequent", index)
            )
            continue
        if not isinstance(consequent, str) or not consequent:
            findings.append(
                Finding(Rejection.MALFORMED_CLAUSE, "consequent is not a non-empty str", index)
            )
            continue
        if not isinstance(antecedents, (list, tuple)) or not all(
            isinstance(a, str) and a for a in antecedents
        ):
            findings.append(
                Finding(Rejection.MALFORMED_CLAUSE, "antecedents is not a list of atoms", index)
            )
            continue

        atoms = set(antecedents) | {consequent}
        if declared is not None and not atoms <= declared:
            invented = sorted(atoms - declared)
            findings.append(
                Finding(Rejection.HALLUCINATED_ATOM, f"undeclared atoms {invented}", index)
            )
            continue

        survivors.append((index, tuple(antecedents), consequent))

    if reject_cycles:
        while True:
            graph: Dict[str, List[str]] = {}
            for _, antecedents, consequent in survivors:
                graph.setdefault(consequent, [])
                for antecedent in antecedents:
                    graph.setdefault(antecedent, []).append(consequent)
            cyclic: Set[str] = set()
            for component in _scc(graph):
                if len(component) > 1:
                    cyclic |= set(component)
            offenders = [
                i
                for i, (_, antecedents, consequent) in enumerate(survivors)
                if consequent in antecedents
                or (consequent in cyclic and cyclic & set(antecedents))
            ]
            if not offenders:
                break
            for i in reversed(offenders):
                index, antecedents, consequent = survivors.pop(i)
                findings.append(
                    Finding(
                        Rejection.CIRCULAR_RULE,
                        f"{list(antecedents)} -> {consequent} closes a cycle",
                        index,
                    )
                )

    clauses = tuple((antecedents, consequent) for _, antecedents, consequent in survivors)
    admitted_atoms = frozenset(
        atom for antecedents, consequent in clauses for atom in (*antecedents, consequent)
    )
    return RuleBase(clauses, admitted_atoms, _ADMISSION), tuple(findings)


def admit_priors(
    priors: Mapping[str, Any], adjacency: Mapping[str, Sequence[str]]
) -> Tuple[Dict[str, float], Tuple[Finding, ...]]:
    """Validate untrusted Bernoulli priors against a graph.

    The proposal engine supplies these, so they may be out of range, non-finite,
    for variables that do not exist, or simply absent. Each case is a distinct
    finding; the returned dict contains only what survived.

    Note that a missing prior is reported but cannot be defaulted: inventing a
    number here would put a fabricated value into every downstream marginal.
    :func:`marginal` refuses to run on an incomplete prior set for that reason.

    Args:
        priors: Untrusted variable -> probability mapping.
        adjacency: The graph the priors are meant to describe.

    Returns:
        ``(clean_priors, findings)``.
    """
    nodes: Set[str] = set(adjacency)
    for children in adjacency.values():
        nodes |= set(children)

    findings: List[Finding] = []
    clean: Dict[str, float] = {}

    for name, value in priors.items():
        if name not in nodes:
            findings.append(
                Finding(Rejection.UNKNOWN_VARIABLE, "not a node in the graph", name)
            )
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            findings.append(Finding(Rejection.MALFORMED_PRIOR, f"got {value!r}", name))
            continue
        value = float(value)
        if not math.isfinite(value):
            findings.append(Finding(Rejection.MALFORMED_PRIOR, f"got {value!r}", name))
            continue
        if not 0.0 <= value <= 1.0:
            findings.append(Finding(Rejection.PRIOR_OUT_OF_RANGE, f"got {value}", name))
            continue
        clean[name] = value

    for missing in sorted(nodes - set(clean)):
        findings.append(
            Finding(Rejection.MISSING_PRIOR, "no admissible prior supplied", missing)
        )

    return clean, tuple(findings)


@dataclass(frozen=True)
class TraceStep:
    """One line of a proposed derivation.

    ``rule_id`` of ``None`` marks a base-premise citation. Every other step must
    name a rule in the trusted base and list exactly that rule's antecedents as
    its support.
    """

    conclusion: str
    rule_id: Optional[int]
    support: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceResult:
    """Outcome of checking a proposed trace."""

    accepted: bool
    established: FrozenSet[str]
    findings: Tuple[Finding, ...] = ()

    @property
    def rejection(self) -> Optional[Finding]:
        """The first thing that went wrong, or None if the trace was accepted."""
        return self.findings[0] if self.findings else None


def parse_trace(raw: Sequence[Mapping[str, Any]]) -> Tuple[Tuple[TraceStep, ...], Tuple[Finding, ...]]:
    """Convert untrusted step dictionaries into typed steps.

    Shape validation only -- soundness is :func:`verify_trace`'s job. Returns
    the steps parsed so far alongside any findings; a malformed step aborts the
    parse, because the steps after it cannot be interpreted with confidence.
    """
    steps: List[TraceStep] = []
    for index, item in enumerate(raw):
        try:
            conclusion = item["conclusion"]
            rule_id = item["rule_id"]
            support = item.get("support", ())
        except (KeyError, TypeError):
            return tuple(steps), (
                Finding(Rejection.MALFORMED_STEP, "missing conclusion/rule_id", index),
            )
        if not isinstance(conclusion, str) or not conclusion:
            return tuple(steps), (
                Finding(Rejection.MALFORMED_STEP, "conclusion is not an atom", index),
            )
        if rule_id is not None and (isinstance(rule_id, bool) or not isinstance(rule_id, int)):
            return tuple(steps), (
                Finding(Rejection.MALFORMED_STEP, f"rule_id {rule_id!r} is not an int", index),
            )
        if not isinstance(support, (list, tuple)) or not all(
            isinstance(a, str) for a in support
        ):
            return tuple(steps), (
                Finding(Rejection.MALFORMED_STEP, "support is not a list of atoms", index),
            )
        steps.append(TraceStep(conclusion, rule_id, tuple(support)))
    return tuple(steps), ()


def verify_trace(
    steps: Sequence[TraceStep], rulebase: RuleBase, facts: Iterable[str]
) -> TraceResult:
    """Check a proposed derivation line by line. This is the kernel's core.

    A step is accepted only if all of the following hold:

    * A premise step (``rule_id is None``) concludes something in ``facts``.
      Otherwise the proposer has smuggled in an assumption -- an ungrounded
      premise, which is how a plausible-looking chain reaches a false goal.
    * A rule step cites a ``rule_id`` that exists in the trusted base. A cited
      rule that is not there is a hallucination, whatever else is true of it.
    * The step's conclusion is exactly that rule's consequent, and its support
      is exactly that rule's antecedents as a set.
    * **Every cited atom was concluded by a strictly earlier step.** This one
      clause is what makes circular support unprovable. Forward chaining over
      definite clauses reaches a fixpoint on a cyclic rule set and will happily
      report the atom as "derived"; a fixpoint is not a proof, and the ordering
      requirement is what separates the two. Base facts are no exception -- they
      must appear as explicit premise steps, so a trace stands on its own rather
      than leaning on ambient context.

    Checking stops at the first bad step: a proof is invalid where it first goes
    wrong, and the steps after it are conditioned on a claim that never held.

    Args:
        steps: The proposed derivation, in order.
        rulebase: The trusted rule base. Steps cite it by index.
        facts: Trusted base premises.

    Returns:
        A :class:`TraceResult`. ``established`` holds what the trace legitimately
        proved -- everything up to the failure point, if there was one.
    """
    fact_set = set(facts)
    established: Set[str] = set()

    for index, step in enumerate(steps):
        if step.rule_id is None:
            if step.conclusion not in fact_set:
                return TraceResult(
                    False,
                    frozenset(established),
                    (
                        Finding(
                            Rejection.UNGROUNDED_PREMISE,
                            f"{step.conclusion!r} is not a trusted premise",
                            index,
                        ),
                    ),
                )
            established.add(step.conclusion)
            continue

        if not 0 <= step.rule_id < len(rulebase):
            return TraceResult(
                False,
                frozenset(established),
                (
                    Finding(
                        Rejection.HALLUCINATED_RULE_ID,
                        f"rule {step.rule_id} not in a base of {len(rulebase)}",
                        index,
                    ),
                ),
            )

        expected_consequent = rulebase.consequent(step.rule_id)
        if step.conclusion != expected_consequent:
            return TraceResult(
                False,
                frozenset(established),
                (
                    Finding(
                        Rejection.CONCLUSION_MISMATCH,
                        f"rule {step.rule_id} concludes {expected_consequent!r}, "
                        f"step claims {step.conclusion!r}",
                        index,
                    ),
                ),
            )

        expected_support = set(rulebase.antecedents(step.rule_id))
        if set(step.support) != expected_support:
            return TraceResult(
                False,
                frozenset(established),
                (
                    Finding(
                        Rejection.ANTECEDENT_MISMATCH,
                        f"rule {step.rule_id} needs {sorted(expected_support)}, "
                        f"step cites {sorted(set(step.support))}",
                        index,
                    ),
                ),
            )

        unfounded = sorted(set(step.support) - established)
        if unfounded:
            return TraceResult(
                False,
                frozenset(established),
                (
                    Finding(
                        Rejection.UNFOUNDED_SUPPORT,
                        f"{unfounded} not established by an earlier step",
                        index,
                    ),
                ),
            )

        established.add(step.conclusion)

    return TraceResult(True, frozenset(established))


def derive(
    goal: str, rulebase: RuleBase, facts: Iterable[str]
) -> Optional[Tuple[TraceStep, ...]]:
    """Independently derive ``goal``, returning a trace the kernel built itself.

    Provenance is recorded only when every antecedent is already established, so
    the resulting trace is acyclic by construction and always passes
    :func:`verify_trace` -- which this function asserts before returning, so a
    bug here fails loudly rather than minting an unchecked proof.

    Returns:
        A minimal-by-first-derivation trace, or ``None`` if ``goal`` is not
        entailed. ``None`` means "not established", never "false".
    """
    fact_set = set(facts)
    provenance: Dict[str, Optional[Tuple[int, Tuple[str, ...]]]] = {
        fact: None for fact in fact_set
    }
    changed = True
    while changed:
        changed = False
        for rule_id in range(len(rulebase)):
            antecedents = rulebase.antecedents(rule_id)
            consequent = rulebase.consequent(rule_id)
            if consequent in provenance:
                continue
            if all(a in provenance for a in antecedents):
                provenance[consequent] = (rule_id, antecedents)
                changed = True

    if goal not in provenance:
        return None

    steps: List[TraceStep] = []
    emitted: Set[str] = set()

    def visit(atom: str) -> None:
        if atom in emitted:
            return
        record = provenance[atom]
        if record is None:
            steps.append(TraceStep(atom, None, ()))
        else:
            rule_id, support = record
            for antecedent in support:
                visit(antecedent)
            steps.append(TraceStep(atom, rule_id, support))
        emitted.add(atom)

    visit(goal)
    proof = tuple(steps)

    result = verify_trace(proof, rulebase, fact_set)
    if not result.accepted:  # pragma: no cover -- a kernel bug, not an input bug
        raise RuntimeError(f"kernel produced an invalid proof: {result.rejection}")
    return proof


class Status(Enum):
    """Verdict on a goal. There is no FALSE: definite clauses cannot express
    negation, so the honest opposite of VERIFIED is UNKNOWN."""

    VERIFIED = "VERIFIED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class GoalReport:
    """What the kernel concluded, and on what evidence."""

    goal: str
    status: Status
    proof: Tuple[TraceStep, ...] = ()
    proposal_accepted: Optional[bool] = None
    findings: Tuple[Finding, ...] = ()

    @property
    def proof_source(self) -> str:
        """Who produced the accepted proof: the proposer, the kernel, or nobody."""
        if self.status is not Status.VERIFIED:
            return "none"
        return "proposal" if self.proposal_accepted else "kernel"


def check_goal(
    goal: str,
    rulebase: RuleBase,
    facts: Iterable[str],
    *,
    proposal: Optional[Sequence[TraceStep]] = None,
) -> GoalReport:
    """Decide a goal, optionally using an untrusted proposed trace as a hint.

    The order matters and is the whole design:

    1. If a proposal is supplied, check it. If it is sound *and* actually
       establishes ``goal``, that checked trace is the proof.
    2. Otherwise derive independently. A rejected proposal says nothing about
       whether the goal holds -- a bad argument for a true claim is still a bad
       argument, and the claim is still true. Both the rejection and the
       kernel's own proof are reported.
    3. Failing that, UNKNOWN.

    The proposal can only ever save work or be discarded. There is no path
    through this function where untrusted input produces VERIFIED without a
    kernel-checked trace, and the assertion before the return enforces it.

    Args:
        goal: Atom to establish.
        rulebase: Trusted rule base.
        facts: Trusted base premises.
        proposal: An untrusted candidate trace, if one is available.

    Returns:
        A :class:`GoalReport`. ``proposal_accepted`` is ``None`` when no
        proposal was offered.
    """
    findings: List[Finding] = []
    proposal_accepted: Optional[bool] = None

    if proposal is not None:
        result = verify_trace(proposal, rulebase, facts)
        proposal_accepted = result.accepted and goal in result.established
        if result.accepted and goal not in result.established:
            findings.append(
                Finding(
                    Rejection.CONCLUSION_MISMATCH,
                    f"trace is sound but never establishes the goal {goal!r}",
                    None,
                )
            )
        findings.extend(result.findings)
        if proposal_accepted:
            report = GoalReport(
                goal, Status.VERIFIED, tuple(proposal), True, tuple(findings)
            )
            _assert_sound(report, rulebase, facts)
            return report

    proof = derive(goal, rulebase, facts)
    if proof is None:
        return GoalReport(goal, Status.UNKNOWN, (), proposal_accepted, tuple(findings))

    report = GoalReport(goal, Status.VERIFIED, proof, proposal_accepted, tuple(findings))
    _assert_sound(report, rulebase, facts)
    return report


def _assert_sound(report: GoalReport, rulebase: RuleBase, facts: Iterable[str]) -> None:
    """Re-check any VERIFIED verdict before it leaves the kernel.

    Cheap, and it converts a whole class of future refactoring mistakes from
    silent unsoundness into a loud crash.
    """
    if report.status is not Status.VERIFIED:
        return
    result = verify_trace(report.proof, rulebase, facts)
    if not result.accepted or report.goal not in result.established:
        raise RuntimeError(  # pragma: no cover -- kernel bug
            f"VERIFIED verdict for {report.goal!r} is not backed by a valid proof"
        )


# =========================================================================== #
#                     PROBABILISTIC LAYER (built on the kernel)               #
# =========================================================================== #


@dataclass(frozen=True)
class NoisyOr:
    """CPD policy turning the fixture's leak terms into full conditionals.

    **This is an engine modelling choice, not fixture data.** The harness
    supplies one leak probability per non-root node and nothing about edge
    strengths, so something has to fill that gap, and it should be visible
    rather than buried. Under this policy::

        P(child = True | parents) = 1 - (1 - leak) * prod_{p true} (1 - w)

    with a single shared activation ``w`` for every edge. A real model would
    learn per-edge weights; substituting a different policy is a matter of
    passing a different ``NoisyOr``. Any marginal computed here inherits the
    arbitrariness of ``edge_activation`` and should be read as a structural
    result, not a calibrated probability.
    """

    edge_activation: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.edge_activation <= 1.0:
            raise ValueError(
                f"edge_activation must be in [0, 1], got {self.edge_activation}"
            )

    def conditional(self, leak: float, parents_true: int) -> float:
        """P(node = True | exactly ``parents_true`` of its parents are True)."""
        return 1.0 - (1.0 - leak) * (1.0 - self.edge_activation) ** parents_true


@dataclass(frozen=True)
class Factor:
    """A discrete factor over binary variables.

    ``table`` is indexed by the bit pattern of the assignment, with
    ``variables[0]`` the most significant bit.
    """

    variables: Tuple[str, ...]
    table: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.table) != 1 << len(self.variables):
            raise ValueError("factor table size does not match its variable count")

    @property
    def width(self) -> int:
        return len(self.variables)

    def _index(self, assignment: Mapping[str, bool]) -> int:
        index = 0
        for variable in self.variables:
            index = (index << 1) | int(assignment[variable])
        return index

    def multiply(self, other: "Factor") -> "Factor":
        variables = self.variables + tuple(
            v for v in other.variables if v not in self.variables
        )
        table: List[float] = []
        for pattern in range(1 << len(variables)):
            assignment = {
                variable: bool((pattern >> (len(variables) - 1 - i)) & 1)
                for i, variable in enumerate(variables)
            }
            table.append(self.table[self._index(assignment)] * other.table[other._index(assignment)])
        return Factor(variables, tuple(table))

    def sum_out(self, variable: str) -> "Factor":
        if variable not in self.variables:
            return self
        remaining = tuple(v for v in self.variables if v != variable)
        table = [0.0] * (1 << len(remaining))
        for pattern in range(1 << len(self.variables)):
            assignment = {
                v: bool((pattern >> (len(self.variables) - 1 - i)) & 1)
                for i, v in enumerate(self.variables)
            }
            index = 0
            for v in remaining:
                index = (index << 1) | int(assignment[v])
            table[index] += self.table[pattern]
        return Factor(remaining, tuple(table))

    def restrict(self, variable: str, value: bool) -> "Factor":
        if variable not in self.variables:
            return self
        remaining = tuple(v for v in self.variables if v != variable)
        table = [0.0] * (1 << len(remaining))
        for pattern in range(1 << len(self.variables)):
            assignment = {
                v: bool((pattern >> (len(self.variables) - 1 - i)) & 1)
                for i, v in enumerate(self.variables)
            }
            if assignment[variable] != value:
                continue
            index = 0
            for v in remaining:
                index = (index << 1) | int(assignment[v])
            table[index] = self.table[pattern]
        return Factor(remaining, tuple(table))


@dataclass(frozen=True)
class EliminationStats:
    """Evidence that the query did not enumerate paths.

    ``max_factor_width`` is the induced width the elimination order actually
    hit. It bounds the work: the query costs O(nodes * 2**max_factor_width),
    independent of how many source-to-sink paths the graph has.
    """

    eliminated: int
    max_factor_width: int
    max_table_entries: int


def _parents_of(adjacency: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
    parents: Dict[str, List[str]] = {node: [] for node in adjacency}
    for node, children in adjacency.items():
        for child in children:
            parents.setdefault(child, []).append(node)
    for node in list(parents):
        parents.setdefault(node, [])
    return parents


def _min_degree_order(
    nodes: Iterable[str], parents: Mapping[str, Sequence[str]], keep: Set[str]
) -> List[str]:
    """Greedy min-degree elimination order over the moral graph.

    Not optimal -- finding the optimal order is NP-hard -- but it is what keeps
    the chained-diamond fixtures at width 3 instead of degenerating into
    something exponential, and :class:`EliminationStats` reports what it
    actually achieved rather than what it hoped for.
    """
    neighbours: Dict[str, Set[str]] = {node: set() for node in nodes}
    for child, ps in parents.items():
        if child not in neighbours:
            continue
        for parent in ps:
            if parent not in neighbours:
                continue
            neighbours[child].add(parent)
            neighbours[parent].add(child)
        for i, a in enumerate(ps):  # moralise: marry the parents
            for b in ps[i + 1 :]:
                if a in neighbours and b in neighbours:
                    neighbours[a].add(b)
                    neighbours[b].add(a)

    order: List[str] = []
    remaining = {n for n in neighbours if n not in keep}
    while remaining:
        node = min(remaining, key=lambda n: (len(neighbours[n] & remaining), n))
        order.append(node)
        clique = neighbours[node] & remaining - {node}
        for a in clique:
            for b in clique:
                if a != b:
                    neighbours[a].add(b)
        remaining.discard(node)
    return order


def marginal(
    adjacency: Mapping[str, Sequence[str]],
    priors: Mapping[str, float],
    target: str,
    *,
    evidence: Optional[Mapping[str, bool]] = None,
    policy: NoisyOr = NoisyOr(),
) -> Tuple[float, EliminationStats]:
    """Exact P(target = True | evidence) by variable elimination.

    Exact, not sampled, and it never enumerates paths -- which is the point of
    the diamond fixtures. A chained diamond of 151 nodes has 2**50 source-to-sink
    paths and is answered here in milliseconds, because elimination cost depends
    on induced width, not path count.

    Args:
        adjacency: Parent -> children. Must be acyclic.
        priors: Complete, already-admitted priors. Root nodes use theirs as the
            marginal; non-roots use theirs as the noisy-OR leak.
        target: Variable to query.
        evidence: Observed variable -> value.
        policy: CPD policy. See :class:`NoisyOr` -- its ``edge_activation`` is
            an engine assumption, not data.

    Returns:
        ``(probability, stats)``.

    Raises:
        ValueError: if ``target`` is not in the graph, if any node lacks a prior
            (the kernel will not invent one), or if evidence names an unknown
            variable.
    """
    evidence = dict(evidence or {})
    parents = _parents_of(adjacency)
    nodes = set(parents)

    if target not in nodes:
        raise ValueError(f"target {target!r} is not a node in the graph")
    missing = sorted(nodes - set(priors))
    if missing:
        raise ValueError(
            f"no prior for {missing}; admit_priors() reports these as "
            "MISSING_PRIOR and the kernel will not fabricate a value"
        )
    unknown = sorted(set(evidence) - nodes)
    if unknown:
        raise ValueError(f"evidence names variables not in the graph: {unknown}")

    factors: List[Factor] = []
    for node in sorted(nodes):
        node_parents = tuple(sorted(parents[node]))
        variables = (node,) + node_parents
        table: List[float] = []
        for pattern in range(1 << len(variables)):
            bits = [(pattern >> (len(variables) - 1 - i)) & 1 for i in range(len(variables))]
            node_true = bool(bits[0])
            parents_true = sum(bits[1:])
            if node_parents:
                p_true = policy.conditional(priors[node], parents_true)
            else:
                p_true = priors[node]
            table.append(p_true if node_true else 1.0 - p_true)
        factors.append(Factor(variables, tuple(table)))

    # Restrict on every observed variable EXCEPT the target. Restricting the
    # target would eliminate the very variable being queried, leaving nothing to
    # read the answer off; its own observation is applied as an indicator at the
    # end instead, which also keeps zero-probability evidence detectable.
    for variable, value in evidence.items():
        if variable == target:
            continue
        factors = [f.restrict(variable, value) for f in factors]

    keep = {target} | set(evidence)
    order = _min_degree_order(nodes, parents, keep)

    max_width = max((f.width for f in factors), default=0)
    max_entries = max((len(f.table) for f in factors), default=0)
    eliminated = 0

    for variable in order:
        involved = [f for f in factors if variable in f.variables]
        if not involved:
            continue
        factors = [f for f in factors if variable not in f.variables]
        product = involved[0]
        for other in involved[1:]:
            product = product.multiply(other)
        max_width = max(max_width, product.width)
        max_entries = max(max_entries, len(product.table))
        factors.append(product.sum_out(variable))
        eliminated += 1

    result = Factor((), (1.0,))
    for factor in factors:
        result = result.multiply(factor)
    max_width = max(max_width, result.width)

    if result.variables != (target,):
        result_target = result
        for variable in result.variables:
            if variable != target:
                result_target = result_target.sum_out(variable)
        result = result_target

    if target in evidence:  # apply the target's own observation as an indicator
        observed = bool(evidence[target])
        result = Factor(
            result.variables,
            tuple(
                value if bool(index) == observed else 0.0
                for index, value in enumerate(result.table)
            ),
        )

    total = result.table[0] + result.table[1]
    if total <= 0.0:
        raise ValueError("evidence has probability zero under this model")
    probability = result.table[1] / total
    return probability, EliminationStats(eliminated, max_width, max_entries)


# =========================================================================== #
#                    ENVELOPE MONITOR (built on the kernel)                   #
# =========================================================================== #


class Verdict(Enum):
    """Invariant verdict.

    SAFE is only reachable under an explicit closed-world assumption. Without
    one, failing to derive a violation is UNKNOWN -- absence of proof is not
    proof of absence, and for a safety monitor that distinction is the entire
    job.
    """

    SAFE = "SAFE"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class InvariantReport:
    """Verdict for one state, with its proof and its assumptions."""

    name: str
    verdict: Verdict
    atoms: Tuple[str, ...]
    violated_limit_states: Tuple[str, ...] = ()
    proof: Tuple[TraceStep, ...] = ()
    assumptions: Tuple[str, ...] = ()
    findings: Tuple[Finding, ...] = ()


def check_envelope(
    name: str,
    atoms: Iterable[str],
    rulebase: RuleBase,
    *,
    violation_atom: str = "envelope_exceedance",
    limit_state_atoms: Iterable[str] = (),
    closed_world: bool = False,
    proposal: Optional[Sequence[TraceStep]] = None,
) -> InvariantReport:
    """Check one discretised state against an admitted envelope rule base.

    Args:
        name: Label for the state, carried into the report.
        atoms: Ground atoms describing the state. These are the trusted
            premises; producing them from raw telemetry is the adapter's job,
            deliberately outside the kernel.
        rulebase: Admitted envelope rules.
        violation_atom: The atom whose derivation means the invariant failed.
        limit_state_atoms: Intermediate limit states to report individually, so
            a violation says *which* limit was breached rather than just that
            one was.
        closed_world: Assert that ``rulebase`` is complete for this vocabulary.
            Only then can "no violation derivable" become SAFE. The assumption
            is recorded in the report either way.
        proposal: An untrusted candidate trace for the violation, if available.

    Returns:
        An :class:`InvariantReport`.
    """
    atoms = tuple(sorted(set(atoms)))
    report = check_goal(violation_atom, rulebase, atoms, proposal=proposal)

    if report.status is Status.VERIFIED:
        breached = tuple(
            sorted(
                atom
                for atom in limit_state_atoms
                if check_goal(atom, rulebase, atoms).status is Status.VERIFIED
            )
        )
        return InvariantReport(
            name,
            Verdict.VIOLATED,
            atoms,
            breached,
            report.proof,
            ("violation established by kernel-checked derivation",),
            report.findings,
        )

    if closed_world:
        return InvariantReport(
            name,
            Verdict.SAFE,
            atoms,
            (),
            (),
            (
                f"closed-world: the rule base is complete for {violation_atom!r} "
                "over the declared vocabulary",
                "the discretisation is faithful to the underlying telemetry",
            ),
            report.findings,
        )

    return InvariantReport(
        name,
        Verdict.UNKNOWN,
        atoms,
        (),
        (),
        ("open-world: no violation was derivable, which is not a safety claim",),
        report.findings,
    )


# =========================================================================== #
#            FIXTURE ADAPTERS (outside the TCB, add no new trust)             #
# =========================================================================== #


def check_aero_fixture(
    fixture: Mapping[str, Any], *, closed_world: bool = True
) -> List[InvariantReport]:
    """Run the aero telemetry fixture end to end.

    Imports :mod:`harness_generators` lazily and only here. The kernel must not
    depend on the fixtures it verifies -- a verifier that needs its test data to
    import is not a verifier -- so the coupling is confined to this adapter.

    ``closed_world`` defaults to True for this fixture specifically, because its
    discretisation is total over the declared vocabulary and its rule set is
    complete for ``envelope_exceedance``. That is a property of this fixture,
    not a safe default for telemetry in general.
    """
    from harness_generators import discretize_state  # local: adapter-only coupling

    rulebase, findings = admit_rules(
        fixture["rules"], vocabulary=None, reject_cycles=True
    )
    if findings:
        raise ValueError(f"the aero fixture should admit cleanly, got {findings}")

    limit_states = tuple(
        sorted({rule["limit_state"] for rule in fixture["rules"]} & rulebase.vocabulary)
    )

    reports: List[InvariantReport] = []
    for query in fixture["queries"]:
        atoms = discretize_state(query["state"], fixture["thresholds"])
        reports.append(
            check_envelope(
                query["name"],
                atoms,
                rulebase,
                violation_atom=fixture["meta"]["violation_atom"],
                limit_state_atoms=limit_states,
                closed_world=closed_world,
            )
        )
    return reports


# --------------------------------------------------------------------------- #
# Demonstration / smoke test
# --------------------------------------------------------------------------- #


def _banner(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def _main() -> int:  # pragma: no cover -- exercised by running the module
    import sys

    from harness_generators import (
        generate_aero_telemetry_fixture,
        generate_candidate_traces,
        generate_diamond_graph,
        generate_horn_rules,
        generate_random_dag,
        inject_rule_hallucinations,
    )

    failures: List[str] = []

    def check(condition: bool, label: str) -> None:
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
        if not condition:
            failures.append(label)

    # ------------------------------------------------------------------ 1 -- #
    _banner("1. Admission control: untrusted rules cannot enter the kernel")
    try:
        RuleBase((), frozenset())
        forged = True
    except TypeError:
        forged = False
    check(not forged, "RuleBase cannot be forged by direct construction")

    cyclic = generate_horn_rules(40, 12, 0.25, seed=42)
    rulebase, findings = admit_rules(cyclic)
    circular = [f for f in findings if f.reason is Rejection.CIRCULAR_RULE]
    print(f"  proposed={len(cyclic)}  admitted={len(rulebase)}  rejected={len(findings)}")
    print(f"  first rejection: {circular[0] if circular else '-'}")
    check(len(circular) > 0, "circular clauses are rejected, not silently admitted")
    check(len(rulebase) < len(cyclic), "the admitted base is strictly smaller")

    vocabulary = sorted({f"x{i}" for i in range(12)})
    poisoned = inject_rule_hallucinations(
        generate_horn_rules(40, 12, 0.0, seed=1), vocabulary, 0.25, seed=1
    )
    _, poison_findings = admit_rules(poisoned, vocabulary=vocabulary)
    caught = {f.location for f in poison_findings if f.reason is Rejection.HALLUCINATED_ATOM}
    planted = {i for i, r in enumerate(poisoned) if r["hallucinated"]}
    print(f"  hallucinated atoms planted={len(planted)}  caught={len(caught)}")
    check(caught == planted, "every hallucinated atom is caught, and no clean rule is")

    # ------------------------------------------------------------------ 2 -- #
    _banner("2. Trace verification: sound accepted, defective rejected, by reason")
    clean_rules = generate_horn_rules(40, 12, 0.0, seed=3)
    clean_base, clean_findings = admit_rules(clean_rules)
    check(not clean_findings, "an acyclic proposal is admitted in full")
    facts = sorted({r["consequent"] for r in clean_rules if not r["antecedents"]})

    traces = generate_candidate_traces(clean_rules, facts, 400, defect_rate=0.6, seed=5)
    tally: Dict[str, Dict[str, int]] = {}
    for raw in traces:
        steps, parse_findings = parse_trace(raw["steps"])
        if parse_findings:
            outcome = "REJECTED"
        else:
            outcome = "ACCEPTED" if verify_trace(steps, clean_base, facts).accepted else "REJECTED"
        bucket = tally.setdefault(raw["defect"] or "(sound)", {"ACCEPTED": 0, "REJECTED": 0})
        bucket[outcome] += 1

    print(f"  {'proposed trace':<24}{'accepted':>10}{'rejected':>10}")
    print(f"  {'-' * 44}")
    for label in sorted(tally):
        print(f"  {label:<24}{tally[label]['ACCEPTED']:>10}{tally[label]['REJECTED']:>10}")

    check(tally["(sound)"]["REJECTED"] == 0, "no sound trace is ever rejected")
    for defect in sorted(k for k in tally if k != "(sound)"):
        check(tally[defect]["ACCEPTED"] == 0, f"no {defect} trace is ever accepted")

    # ------------------------------------------------------------------ 3 -- #
    _banner("3. A rejected proposal does not make the goal false")
    goal = next(
        r["goal"] for r in traces if r["defect"] is not None
    )
    bad = next(r for r in traces if r["defect"] is not None and r["goal"] == goal)
    steps, _ = parse_trace(bad["steps"])
    report = check_goal(goal, clean_base, facts, proposal=steps)
    print(f"  goal={goal!r}  proposal defect={bad['defect']}")
    print(f"  proposal_accepted={report.proposal_accepted}  status={report.status.value}")
    print(f"  proof_source={report.proof_source}  proof_steps={len(report.proof)}")
    check(report.proposal_accepted is False, "the defective proposal is rejected")
    check(report.status is Status.VERIFIED, "the kernel still derives the goal itself")
    check(report.proof_source == "kernel", "the accepted proof came from the kernel")

    unreachable = check_goal("x0", clean_base, [], proposal=None)
    check(unreachable.status is Status.UNKNOWN, "an underivable goal is UNKNOWN, not false")

    # ------------------------------------------------------------------ 4 -- #
    _banner("4. Probabilistic layer: exact marginals without path enumeration")
    adjacency, raw_priors = generate_random_dag(24, 5, 3, seed=7)
    priors, prior_findings = admit_priors(raw_priors, adjacency)
    check(not prior_findings, "well-formed priors are admitted cleanly")

    corrupt = dict(raw_priors)
    corrupt["v0"] = 1.4
    corrupt["v1"] = float("nan")
    corrupt["ghost"] = 0.5
    _, bad_findings = admit_priors(corrupt, adjacency)
    reasons = {f.reason for f in bad_findings}
    print(f"  corrupt-prior rejections: {sorted(r.name for r in reasons)}")
    check(Rejection.PRIOR_OUT_OF_RANGE in reasons, "out-of-range prior rejected")
    check(Rejection.MALFORMED_PRIOR in reasons, "NaN prior rejected")
    check(Rejection.UNKNOWN_VARIABLE in reasons, "prior for a phantom variable rejected")

    print()
    print(f"  {'diamond':<10}{'paths':>18}{'P(sink)':>12}{'width':>8}{'table':>9}")
    print(f"  {'-' * 57}")
    for n in (13, 31, 61, 151):
        chain = generate_diamond_graph(n)
        chain_priors = {node: 0.2 if node == "v0" else 0.1 for node in chain}
        p, stats = marginal(chain, chain_priors, f"v{n - 1}")
        print(
            f"  n={n:<8}{2 ** ((n - 1) // 3):>18,}{p:>12.6f}"
            f"{stats.max_factor_width:>8}{stats.max_table_entries:>9}"
        )
        check(stats.max_factor_width <= 4, f"n={n}: induced width stayed bounded")
        check(0.0 <= p <= 1.0, f"n={n}: marginal is a probability")

    chain = generate_diamond_graph(13)
    chain_priors = {node: 0.2 if node == "v0" else 0.1 for node in chain}
    p_free, _ = marginal(chain, chain_priors, "v12")
    p_given, _ = marginal(chain, chain_priors, "v12", evidence={"v0": True})
    print(f"  conditioning: P(v12)={p_free:.6f}  P(v12 | v0=True)={p_given:.6f}")
    check(p_given > p_free, "evidence at the source raises the sink's marginal")

    # ------------------------------------------------------------------ 5 -- #
    _banner("5. Envelope monitor on the aero fixture")
    fixture = generate_aero_telemetry_fixture()
    reports = check_aero_fixture(fixture)
    expected = {q["name"]: q["expected_safe"] for q in fixture["queries"]}

    print(f"  {'query':<32}{'verdict':<11}{'proof':>6}  limit states")
    print(f"  {'-' * 72}")
    for report in reports:
        print(
            f"  {report.name:<32}{report.verdict.value:<11}{len(report.proof):>6}  "
            f"{', '.join(report.violated_limit_states) or '-'}"
        )
        want = Verdict.SAFE if expected[report.name] else Verdict.VIOLATED
        check(report.verdict is want, f"{report.name}: verdict matches ground truth")
        if report.verdict is Verdict.VIOLATED:
            check(bool(report.proof), f"{report.name}: violation carries a proof")
            check(
                bool(report.violated_limit_states),
                f"{report.name}: violation names its limit state",
            )

    open_world = check_aero_fixture(fixture, closed_world=False)
    safe_names = {r.name for r in reports if r.verdict is Verdict.SAFE}
    unknown_names = {r.name for r in open_world if r.verdict is Verdict.UNKNOWN}
    print()
    print(f"  open-world: {len(safe_names)} SAFE verdicts become UNKNOWN")
    check(safe_names == unknown_names, "SAFE requires an explicit closed-world assumption")
    check(
        all(r.assumptions for r in reports),
        "every verdict records the assumptions it rests on",
    )

    # --------------------------------------------------------------------- #
    _banner("SUMMARY")
    if failures:
        print(f"  {len(failures)} check(s) FAILED:")
        for label in failures:
            print(f"    - {label}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_main())
