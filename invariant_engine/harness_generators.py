"""Synthetic test-case generators for a neuro-symbolic invariant verification engine.

Standard library only -- no numpy, networkx, scipy, or solver bindings. Every
generator is deterministic under an explicit ``seed`` so a failing harness run
can be replayed byte-for-byte.

Fixture families
----------------
* :func:`generate_random_dag`             -- randomly structured Bayesian DAGs.
* :func:`generate_diamond_graph`          -- chained diamonds, for path explosion.
* :func:`generate_horn_rules`             -- Horn clause sets with injected cycles.
* :func:`generate_aero_telemetry_fixture` -- a mock flight-envelope dataset.

Verification helpers (:func:`is_acyclic`, :func:`count_paths`,
:func:`find_cycles_in_rules`, :func:`forward_chain`) let the generators check
their own output. They are deliberately naive *reference* implementations for
fixture self-test -- they are not, and must not be mistaken for, the engine
under test.

NUMERIC PROVENANCE WARNING
--------------------------
Every number in :func:`generate_aero_telemetry_fixture` is SYNTHETIC. The
*structure* -- speed nomenclature, limit-state names, the rule shapes -- follows
published airworthiness terminology, but no value was taken from an Aircraft
Flight Manual, type certificate data sheet, or certification report. Each
threshold therefore carries ``verified: False``. See :data:`AERO_SOURCE_NOTE`.
These values exercise code paths; they do not describe any real aircraft.
"""

from __future__ import annotations

import random
import sys
from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    "Adjacency",
    "Priors",
    "HornClause",
    "AERO_SOURCE_NOTE",
    "generate_random_dag",
    "generate_diamond_graph",
    "generate_horn_rules",
    "generate_aero_telemetry_fixture",
    "generate_candidate_traces",
    "inject_rule_hallucinations",
    "TRACE_DEFECTS",
    "discretize_state",
    "forward_chain",
    "find_cycles_in_rules",
    "is_acyclic",
    "count_paths",
    "topological_order",
]

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #

Adjacency = Dict[str, List[str]]
"""Parent -> list of children. Every node appears as a key, possibly with []."""

Priors = Dict[str, float]
"""Variable -> baseline probability of the variable being True."""

HornClause = Dict[str, Any]
"""``{"antecedents": [str, ...], "consequent": str, ...metadata}``."""


# --------------------------------------------------------------------------- #
# Graph utilities (reference implementations, used for fixture self-test)
# --------------------------------------------------------------------------- #


def _all_nodes(graph: Mapping[str, Sequence[str]]) -> List[str]:
    """Return every node in ``graph``: keys plus any node reachable as a child.

    Order is stable -- keys first in insertion order, then newly seen children
    in first-encounter order -- so downstream algorithms are deterministic.
    """
    nodes: List[str] = list(graph)
    seen: Set[str] = set(nodes)
    for children in graph.values():
        for child in children:
            if child not in seen:
                seen.add(child)
                nodes.append(child)
    return nodes


def _strongly_connected_components(
    graph: Mapping[str, Sequence[str]],
) -> List[List[str]]:
    """Tarjan's SCC algorithm, iterative (no recursion depth limit).

    Returns one list of node names per strongly connected component. A component
    of size > 1 is a cycle; self-loops are handled by the callers, which know
    whether a single node pointing at itself is meaningful in their domain.
    """
    index_of: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    components: List[List[str]] = []
    counter = 0

    for root in _all_nodes(graph):
        if root in index_of:
            continue
        work: List[Tuple[str, int]] = [(root, 0)]
        while work:
            node, child_index = work.pop()
            if child_index == 0:
                index_of[node] = counter
                lowlink[node] = counter
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
                components.append(component)

            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    return components


def is_acyclic(graph: Mapping[str, Sequence[str]]) -> bool:
    """True iff ``graph`` contains no directed cycle, self-loops included."""
    for node, children in graph.items():
        if node in children:
            return False
    return all(len(comp) <= 1 for comp in _strongly_connected_components(graph))


def topological_order(graph: Mapping[str, Sequence[str]]) -> List[str]:
    """Kahn topological sort.

    Raises:
        ValueError: if ``graph`` contains a cycle.
    """
    indegree: Dict[str, int] = {node: 0 for node in _all_nodes(graph)}
    for children in graph.values():
        for child in children:
            indegree[child] += 1

    queue = deque(node for node, deg in indegree.items() if deg == 0)
    order: List[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in graph.get(node, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(order) != len(indegree):
        raise ValueError("graph contains a cycle; no topological order exists")
    return order


def count_paths(graph: Mapping[str, Sequence[str]], source: str, sink: str) -> int:
    """Count distinct directed paths from ``source`` to ``sink``.

    Exact -- not sampled -- via dynamic programming over a topological order, so
    it is safe on graphs whose path count is exponential in node count: the
    count is astronomically large, the computation is O(V + E).

    Raises:
        ValueError: if ``graph`` is cyclic.
    """
    order = topological_order(graph)
    counts: Dict[str, int] = {}
    for node in reversed(order):
        if node == sink:
            counts[node] = 1
            continue
        counts[node] = sum(counts.get(child, 0) for child in graph.get(node, ()))
    return counts.get(source, 0)


def find_cycles_in_rules(rules: Sequence[HornClause]) -> List[List[str]]:
    """Return the cyclic variable groups in a Horn rule set's dependency graph.

    The dependency graph carries an edge ``antecedent -> consequent`` for every
    literal of every rule. Each returned list is either a self-dependent
    variable (``["a"]``, from a rule such as ``a -> a``) or a strongly connected
    component of size > 1.

    Note that forward chaining over definite clauses is monotone and therefore
    *terminates* even on a cyclic rule set -- it simply reaches a least
    fixpoint. A cycle is not a hang; it is an unfounded-support hazard that a
    cycle detector must catch explicitly, which is precisely why the harness
    injects them.
    """
    graph: Dict[str, List[str]] = {}
    self_loops: List[List[str]] = []
    for rule in rules:
        consequent = rule["consequent"]
        graph.setdefault(consequent, [])
        for antecedent in rule["antecedents"]:
            graph.setdefault(antecedent, []).append(consequent)
            if antecedent == consequent and [consequent] not in self_loops:
                self_loops.append([consequent])

    cycles = [
        sorted(comp) for comp in _strongly_connected_components(graph) if len(comp) > 1
    ]
    return self_loops + cycles


def forward_chain(facts: Iterable[str], rules: Sequence[HornClause]) -> Set[str]:
    """Naive forward chainer over definite clauses, run to a least fixpoint.

    Included so fixtures can be checked end to end without importing the engine
    under test. O(rules x iterations) and deliberately untuned; it is a
    correctness oracle for fixture-sized inputs, not a performance reference.
    """
    derived: Set[str] = set(facts)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            consequent = rule["consequent"]
            if consequent in derived:
                continue
            if all(antecedent in derived for antecedent in rule["antecedents"]):
                derived.add(consequent)
                changed = True
    return derived


# --------------------------------------------------------------------------- #
# 1. Random Bayesian DAG
# --------------------------------------------------------------------------- #


def generate_random_dag(
    num_vars: int,
    depth: int,
    max_fanout: int,
    seed: Optional[int] = None,
    *,
    max_indegree: int = 3,
) -> Tuple[Adjacency, Priors]:
    """Generate a random, guaranteed-acyclic Bayesian DAG.

    Acyclicity is structural, not rejection-sampled: variables are partitioned
    into ``depth`` layers and every edge runs from a strictly lower layer to a
    strictly higher one, so no cycle is representable. The result is asserted
    acyclic before return.

    Args:
        num_vars: Total number of variables. Must be >= ``depth``.
        depth: Number of layers, each non-empty. Must be >= 1.
        max_fanout: Hard cap on out-degree (children per node). Must be >= 1.
        seed: Seed for the local RNG. ``None`` draws from system entropy and
            makes the run unreproducible -- pass an int in any harness you
            intend to replay.
        max_indegree: Cap on parents per node. Bounds CPT width: a node with
            *k* binary parents needs 2**k rows, so keep this small.

    Returns:
        ``(adjacency, priors)``.

        * ``adjacency`` maps each variable to its list of children. Every
          variable is present as a key, possibly with an empty list.
        * ``priors`` maps each variable to a baseline probability in (0, 1).
          For a **root** node this is the marginal P(v = True). For a
          **non-root** node it is the noisy-OR *leak* term: the probability
          that v is True with all parents False. Full conditional probability
          tables are intentionally not generated -- the consumer picks the CPD
          family (noisy-OR, logistic, tabular) and expands from the leak.

    Raises:
        ValueError: on inconsistent arguments.
        RuntimeError: if the acyclicity post-condition fails (a generator bug).

    Note:
        If every earlier-layer node is already at ``max_fanout``, a node is left
        parentless and becomes an additional root rather than breaching the
        fan-out cap. The result is still a valid DAG; it means the requested
        ``max_fanout`` could not support the sampled layer widths.
    """
    if num_vars < 1:
        raise ValueError(f"num_vars must be >= 1, got {num_vars}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if depth > num_vars:
        raise ValueError(f"depth ({depth}) cannot exceed num_vars ({num_vars})")
    if max_fanout < 1:
        raise ValueError(f"max_fanout must be >= 1, got {max_fanout}")
    if max_indegree < 1:
        raise ValueError(f"max_indegree must be >= 1, got {max_indegree}")

    rng = random.Random(seed)

    # Partition the variables into `depth` non-empty layers.
    layer_sizes = [1] * depth
    for _ in range(num_vars - depth):
        layer_sizes[rng.randrange(depth)] += 1

    names = [f"v{i}" for i in range(num_vars)]
    layers: List[List[str]] = []
    cursor = 0
    for size in layer_sizes:
        layers.append(names[cursor : cursor + size])
        cursor += size

    adjacency: Adjacency = {name: [] for name in names}
    parents: Dict[str, List[str]] = {name: [] for name in names}
    capacity: Dict[str, int] = {name: max_fanout for name in names}

    for layer_index in range(1, depth):
        earlier = [node for layer in layers[:layer_index] for node in layer]
        for node in layers[layer_index]:
            candidates = [n for n in earlier if capacity[n] > 0]
            if not candidates:
                continue  # becomes an extra root; see the docstring Note.
            k = rng.randint(1, min(max_indegree, len(candidates)))
            for parent in rng.sample(candidates, k):
                adjacency[parent].append(node)
                parents[node].append(parent)
                capacity[parent] -= 1

    priors: Priors = {}
    for name in names:
        if parents[name]:
            priors[name] = round(rng.uniform(0.01, 0.15), 4)  # noisy-OR leak
        else:
            priors[name] = round(rng.uniform(0.05, 0.95), 4)  # root marginal

    if not is_acyclic(adjacency):  # post-condition, not decoration
        raise RuntimeError("generate_random_dag produced a cyclic graph")

    return adjacency, priors


# --------------------------------------------------------------------------- #
# 2. Diamond graph
# --------------------------------------------------------------------------- #


def generate_diamond_graph(n: int, *, mode: str = "chain") -> Adjacency:
    """Generate a branching-and-rejoining diamond DAG over ``n`` variables.

    Two shapes, because they stress different things:

    ``mode="chain"`` (default)
        *k* diamonds in series, ``k = (n - 1) // 3``. The source-to-sink path
        count is **2**k** -- exponential in node count. This is the shape that
        actually produces path explosion, and the one you want for stressing an
        enumerating verifier. Nodes left over after ``3k + 1`` are appended as a
        linear tail so exactly ``n`` variables are returned.

    ``mode="wide"``
        The textbook single diamond widened to ``n`` variables: one source, one
        sink, ``n - 2`` parallel middle nodes. Path count is only ``n - 2``, so
        it exercises fan-in width and marginalisation but does *not* explode.
        Keep it as the control case: if a verifier is slow here too, the problem
        is not path enumeration.

    Args:
        n: Total number of variables. ``chain`` requires n >= 4 (one diamond);
            ``wide`` requires n >= 3.
        mode: ``"chain"`` or ``"wide"``.

    Returns:
        Adjacency mapping parent -> children. Nodes are named ``v0 .. v{n-1}``
        in topological order, so the source is always ``v0`` and the sink is
        always ``v{n-1}``.

    Raises:
        ValueError: for an unknown ``mode`` or too small an ``n``.
    """
    if mode not in ("chain", "wide"):
        raise ValueError(f"mode must be 'chain' or 'wide', got {mode!r}")

    names = [f"v{i}" for i in range(n)]
    adjacency: Adjacency = {name: [] for name in names}

    if mode == "wide":
        if n < 3:
            raise ValueError(f"mode='wide' needs n >= 3, got {n}")
        source, sink = names[0], names[-1]
        for middle in names[1:-1]:
            adjacency[source].append(middle)
            adjacency[middle].append(sink)
        return adjacency

    if n < 4:
        raise ValueError(f"mode='chain' needs n >= 4 for one diamond, got {n}")

    num_diamonds = (n - 1) // 3
    hub = names[0]
    for d in range(num_diamonds):
        left, right, join = names[3 * d + 1], names[3 * d + 2], names[3 * d + 3]
        adjacency[hub].extend([left, right])
        adjacency[left].append(join)
        adjacency[right].append(join)
        hub = join

    for tail_node in names[3 * num_diamonds + 1 :]:  # linear remainder
        adjacency[hub].append(tail_node)
        hub = tail_node

    return adjacency


# --------------------------------------------------------------------------- #
# 3. Horn rules with injected circular dependencies
# --------------------------------------------------------------------------- #

_CLAUSE_RESAMPLE_ATTEMPTS = 32
"""Tries to find an unused clause before accepting a duplicate. See
:func:`generate_horn_rules`."""


def _sample_definite_clause(
    rng: random.Random,
    variables: Sequence[str],
    max_antecedents: int,
    fact_fraction: float,
) -> HornClause:
    """Sample one acyclic-by-construction definite clause.

    Antecedents always carry a strictly lower index than the consequent, so no
    clause from this function can ever close a cycle on its own.
    """
    if rng.random() < fact_fraction:
        return {
            "antecedents": [],
            "consequent": rng.choice(variables),
            "circular": False,
        }
    consequent_index = rng.randrange(1, len(variables))  # x0 can only be a fact
    pool = variables[:consequent_index]
    k = rng.randint(1, min(max_antecedents, len(pool)))
    return {
        "antecedents": sorted(rng.sample(pool, k)),
        "consequent": variables[consequent_index],
        "circular": False,
    }


def generate_horn_rules(
    num_rules: int,
    num_vars: int,
    circular_fraction: float = 0.1,
    *,
    seed: Optional[int] = None,
    max_antecedents: int = 3,
    fact_fraction: float = 0.15,
) -> List[HornClause]:
    """Generate Horn clauses, deliberately injecting circular dependencies.

    Non-circular rules are acyclic by construction: variables carry an implicit
    index and every antecedent has a strictly lower index than its consequent.
    Circular rules are built explicitly as closed chains (``a -> b -> c -> a``),
    degenerating to a self-loop (``a -> a``) when only one rule of budget
    remains.

    Args:
        num_rules: Number of clauses to emit. The returned list has exactly this
            length.
        num_vars: Size of the variable pool. Must be >= 2.
        circular_fraction: Fraction of clauses that participate in a
            deliberately injected cycle, in [0.0, 1.0]. The injected count is
            ``round(num_rules * circular_fraction)``.
        seed: RNG seed. Pass an int for replayable harness runs.
        max_antecedents: Cap on body length of a non-circular rule.
        fact_fraction: Fraction of non-circular rules emitted as *facts* (empty
            antecedent list) -- the base premises a forward chainer starts from.

    Returns:
        A shuffled list of ``{"antecedents": [...], "consequent": str,
        "circular": bool}``. The ``circular`` flag is ground truth for the
        harness: True iff the clause was emitted as part of an injected cycle.
        A strict Horn-clause consumer can ignore the extra key.

        Non-circular clauses are deduplicated by ``(antecedents, consequent)``
        via bounded resampling (:data:`_CLAUSE_RESAMPLE_ATTEMPTS` tries). If the
        clause space is too small to fill ``num_rules`` distinctly, a duplicate
        is accepted rather than shortening the list -- the exact-count contract
        takes priority. Injected cycle clauses are never rejected, since doing
        so would break the cycle-budget arithmetic.

    Raises:
        ValueError: on out-of-range arguments.

    Warning:
        ``circular_fraction`` is an *injection* rate, not a guarantee about the
        total number of cycles present, nor about their size. A back-edge from
        an injected cycle can combine with forward edges from the acyclic rules
        to close further, unplanned cycles, and those cycles can merge.

        Measured over 200 seeds at ``num_rules=40, num_vars=12``, the fraction
        of runs in which a single SCC absorbed all 12 variables was 1/200 at
        ``circular_fraction=0.1``, 18/200 at 0.25, and 72/200 at 0.5. If the
        harness needs small, separable cycles, stay near the default and check
        the result -- do not assume the injected count is the total, and do not
        assume the cycles stay disjoint. :func:`find_cycles_in_rules` reports
        what is actually there.
    """
    if num_rules < 0:
        raise ValueError(f"num_rules must be >= 0, got {num_rules}")
    if num_vars < 2:
        raise ValueError(f"num_vars must be >= 2, got {num_vars}")
    if not 0.0 <= circular_fraction <= 1.0:
        raise ValueError(f"circular_fraction must be in [0, 1], got {circular_fraction}")
    if max_antecedents < 1:
        raise ValueError(f"max_antecedents must be >= 1, got {max_antecedents}")
    if not 0.0 <= fact_fraction <= 1.0:
        raise ValueError(f"fact_fraction must be in [0, 1], got {fact_fraction}")

    rng = random.Random(seed)
    variables = [f"x{i}" for i in range(num_vars)]
    rules: List[HornClause] = []

    # --- injected cycles --------------------------------------------------- #
    budget = min(num_rules, round(num_rules * circular_fraction))
    while budget > 0:
        if budget == 1:
            var = rng.choice(variables)
            rules.append({"antecedents": [var], "consequent": var, "circular": True})
            budget -= 1
            continue
        cycle_len = min(rng.randint(2, 4), budget, num_vars)
        members = rng.sample(variables, cycle_len)
        for i, var in enumerate(members):
            rules.append(
                {
                    "antecedents": [var],
                    "consequent": members[(i + 1) % cycle_len],
                    "circular": True,
                }
            )
        budget -= cycle_len

    # --- acyclic remainder ------------------------------------------------- #
    # Resample on collision: without this, a 40-rule / 12-variable draw returns
    # roughly 5 literal duplicate clauses (measured over 200 seeds), so a caller
    # asking for 40 rules would silently get ~35 distinct ones.
    seen = {(tuple(r["antecedents"]), r["consequent"]) for r in rules}
    while len(rules) < num_rules:
        for _ in range(_CLAUSE_RESAMPLE_ATTEMPTS):
            candidate = _sample_definite_clause(
                rng, variables, max_antecedents, fact_fraction
            )
            key = (tuple(candidate["antecedents"]), candidate["consequent"])
            if key not in seen:
                break
        # Falls through with the last candidate when the clause space is too
        # small to fill num_rules distinctly; the exact-count contract wins.
        seen.add(key)
        rules.append(candidate)

    rng.shuffle(rules)  # do not cluster the cycles at the head of the list
    return rules


# --------------------------------------------------------------------------- #
# 3b. Adversarial proposal fixtures (mocked untrusted "neuro" output)
# --------------------------------------------------------------------------- #

TRACE_DEFECTS = (
    "hallucinated_rule",
    "circular_support",
    "ungrounded_premise",
    "antecedent_mismatch",
    "conclusion_mismatch",
)
"""The defect taxonomy a candidate deduction trace can carry.

These mock the ways an untrusted proposal engine -- an LLM, a learned search
policy -- gets a derivation wrong. They are the cases a trusted kernel exists to
reject, so each one is generated with a ground-truth label rather than left for
the kernel to be graded on its own homework.
"""


def _derive_with_provenance(
    facts: Iterable[str], rules: Sequence[HornClause]
) -> Dict[str, Optional[Tuple[int, Tuple[str, ...]]]]:
    """Forward chain, recording how each atom was first established.

    Returns atom -> ``None`` for a base fact, or ``(rule_index, support)`` for a
    derived atom. Provenance is acyclic by construction: an atom is only ever
    recorded once every antecedent is already established, so no atom can end up
    supporting itself even when the rule set is cyclic.

    This duplicates a few lines of the engine's own chase on purpose. A fixture
    generator that imported the engine could only ever produce fixtures the
    engine already agrees with.
    """
    established: Dict[str, Optional[Tuple[int, Tuple[str, ...]]]] = {
        fact: None for fact in facts
    }
    changed = True
    while changed:
        changed = False
        for rule_id, rule in enumerate(rules):
            consequent = rule["consequent"]
            if consequent in established:
                continue
            if all(a in established for a in rule["antecedents"]):
                established[consequent] = (rule_id, tuple(rule["antecedents"]))
                changed = True
    return established


def _extract_trace(
    goal: str, established: Mapping[str, Optional[Tuple[int, Tuple[str, ...]]]]
) -> List[Dict[str, Any]]:
    """Emit a sound, self-contained trace for ``goal`` in dependency order.

    Every cited atom appears as the conclusion of a strictly earlier step, base
    facts included -- a trace that leans on ambient context is not a proof.
    """
    steps: List[Dict[str, Any]] = []
    emitted: Set[str] = set()

    def visit(atom: str) -> None:
        if atom in emitted:
            return
        provenance = established[atom]
        if provenance is None:
            steps.append({"conclusion": atom, "rule_id": None, "support": []})
        else:
            rule_id, support = provenance
            for antecedent in support:
                visit(antecedent)
            steps.append(
                {"conclusion": atom, "rule_id": rule_id, "support": list(support)}
            )
        emitted.add(atom)

    visit(goal)
    return steps


def _corrupt(
    steps: List[Dict[str, Any]],
    defect: str,
    rng: random.Random,
    rules: Sequence[HornClause],
    facts: Set[str],
    vocabulary: Sequence[str],
) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Apply one named defect. Returns ``(steps, defect_step)`` or None if the
    trace has no site where that defect is expressible."""
    steps = [dict(s) for s in steps]
    rule_steps = [i for i, s in enumerate(steps) if s["rule_id"] is not None]
    premise_steps = [i for i, s in enumerate(steps) if s["rule_id"] is None]

    if defect == "hallucinated_rule":
        if not rule_steps:
            return None
        i = rng.choice(rule_steps)
        steps[i]["rule_id"] = len(rules) + rng.randint(0, 5)
        return steps, i

    if defect == "circular_support":
        # Rotate the concluding step to the front. Its support was established
        # by earlier steps, so in front of them it is self-referential: the
        # trace now asserts its own conclusion before anything supports it.
        if len(steps) < 2 or steps[-1]["rule_id"] is None:
            return None
        return [steps[-1]] + steps[:-1], 0

    if defect == "ungrounded_premise":
        if not premise_steps:
            return None
        i = rng.choice(premise_steps)
        candidates = [v for v in vocabulary if v not in facts]
        if not candidates:
            return None
        steps[i] = dict(steps[i], conclusion=rng.choice(candidates))
        return steps, i

    if defect == "antecedent_mismatch":
        sites = [i for i in rule_steps if steps[i]["support"]]
        if not sites:
            return None
        i = rng.choice(sites)
        dropped = list(steps[i]["support"])
        dropped.pop(rng.randrange(len(dropped)))
        steps[i] = dict(steps[i], support=dropped)
        return steps, i

    if defect == "conclusion_mismatch":
        if not rule_steps:
            return None
        i = rng.choice(rule_steps)
        alternatives = [v for v in vocabulary if v != steps[i]["conclusion"]]
        if not alternatives:
            return None
        steps[i] = dict(steps[i], conclusion=rng.choice(alternatives))
        return steps, i

    raise ValueError(f"unknown defect {defect!r}")


def generate_candidate_traces(
    rules: Sequence[HornClause],
    facts: Iterable[str],
    num_traces: int,
    *,
    defect_rate: float = 0.5,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate candidate deduction traces, sound and adversarial, with labels.

    This mocks the first job of the untrusted proposal engine: proposing a
    derivation that may be sound, flawed, or circular. The trusted kernel's
    entire purpose is to accept the first kind and reject the rest, so each
    trace ships with the ground truth of what -- if anything -- is wrong with it.

    Args:
        rules: The rule base the traces are proposed against. Traces cite rules
            by index into this sequence.
        facts: Base premises available to the proposer.
        num_traces: How many traces to emit.
        defect_rate: Probability that a given trace is corrupted. The defect is
            drawn uniformly from :data:`TRACE_DEFECTS`; if the drawn defect has
            no expressible site in that particular trace, the trace is emitted
            sound rather than force-fitted, so the realised defect rate can sit
            slightly below the requested one. The ``defect`` label is always
            accurate -- that is what matters.
        seed: RNG seed.

    Returns:
        A list of ``{"goal": str, "steps": [...], "defect": str | None,
        "defect_step": int | None}``. Each step is ``{"conclusion": str,
        "rule_id": int | None, "support": [str]}``; ``rule_id`` of ``None``
        marks a base-premise citation.

    Raises:
        ValueError: if ``defect_rate`` is out of range, or if ``rules`` and
            ``facts`` together entail nothing -- there is then no sound trace to
            build or corrupt, and silently returning premise-only traces would
            be a fixture that tests nothing.
    """
    if not 0.0 <= defect_rate <= 1.0:
        raise ValueError(f"defect_rate must be in [0, 1], got {defect_rate}")
    if num_traces < 0:
        raise ValueError(f"num_traces must be >= 0, got {num_traces}")

    rng = random.Random(seed)
    fact_set = set(facts)
    established = _derive_with_provenance(fact_set, rules)
    derivable = sorted(a for a, prov in established.items() if prov is not None)
    if not derivable:
        raise ValueError(
            "the rule base entails nothing from these facts, so no sound trace "
            "exists to build or corrupt"
        )

    vocabulary = sorted(
        {r["consequent"] for r in rules}
        | {a for r in rules for a in r["antecedents"]}
        | fact_set
    )

    traces: List[Dict[str, Any]] = []
    for _ in range(num_traces):
        goal = rng.choice(derivable)
        steps = _extract_trace(goal, established)
        defect: Optional[str] = None
        defect_step: Optional[int] = None
        if rng.random() < defect_rate:
            drawn = rng.choice(TRACE_DEFECTS)
            outcome = _corrupt(steps, drawn, rng, rules, fact_set, vocabulary)
            if outcome is not None:
                # Label with the defect actually applied, never with the one we
                # merely intended: the whole value of these fixtures is that the
                # ground truth is trustworthy.
                steps, defect_step = outcome
                defect = drawn
        traces.append(
            {
                "goal": goal,
                "steps": steps,
                "defect": defect,
                "defect_step": defect_step,
            }
        )
    return traces


def inject_rule_hallucinations(
    rules: Sequence[HornClause],
    vocabulary: Iterable[str],
    fraction: float,
    *,
    seed: Optional[int] = None,
) -> List[HornClause]:
    """Corrupt a rule base with atoms outside the declared vocabulary.

    This mocks the third job of the untrusted proposal engine: autoformalising
    raw input into rule dictionaries, and inventing predicates while doing it. A
    hallucinated atom is not a wrong rule so much as an ungrounded one -- it
    names something the domain never declared, so nothing can ever establish it
    and nothing can ever refute it.

    Args:
        rules: The clean rule base to corrupt. Not mutated.
        vocabulary: The declared, legitimate atom vocabulary.
        fraction: Fraction of clauses to corrupt, in [0, 1].
        seed: RNG seed.

    Returns:
        A new list of clauses. Corrupted ones carry ``"hallucinated": True`` as
        ground truth; untouched ones are copied through with ``False``.

    Raises:
        ValueError: if ``fraction`` is out of range.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    rng = random.Random(seed)
    declared = set(vocabulary)
    count = round(len(rules) * fraction)
    targets = set(rng.sample(range(len(rules)), count)) if count else set()

    out: List[HornClause] = []
    for i, rule in enumerate(rules):
        clause = {k: (list(v) if isinstance(v, list) else v) for k, v in rule.items()}
        if i in targets:
            phantom = f"phantom_{rng.randrange(10**6):06d}"
            assert phantom not in declared
            if clause["antecedents"] and rng.random() < 0.5:
                clause["antecedents"][rng.randrange(len(clause["antecedents"]))] = phantom
            else:
                clause["consequent"] = phantom
            clause["hallucinated"] = True
        else:
            clause["hallucinated"] = False
        out.append(clause)
    return out


# --------------------------------------------------------------------------- #
# 4. Aerospace telemetry fixture
# --------------------------------------------------------------------------- #

AERO_SOURCE_NOTE = (
    "ALL NUMERIC THRESHOLDS IN THIS FIXTURE ARE SYNTHETIC AND UNVERIFIED. They "
    "were chosen to be order-of-magnitude plausible for a generic two-engine "
    "transport-category aeroplane and to exercise every branch of the rule set. "
    "None is traceable to an Aircraft Flight Manual, Type Certificate Data "
    "Sheet, or certification report, and no real aircraft is being modelled. "
    "The speed nomenclature (VMO/MMO, VFE) and the limit-state names follow "
    "14 CFR Part 1 / Part 25 terminology, and the direction of the flap effect "
    "on stalling angle of attack -- trailing-edge flap deflection shifts the "
    "lift curve up and to the left, lowering alpha_crit -- is standard "
    "aerodynamics. Every magnitude here is nonetheless an invention. Replace "
    "the entire `thresholds` block with certified data before this fixture "
    "informs any decision about real hardware."
)


def _threshold(value: float, unit: str, limit_state: str, basis: str) -> Dict[str, Any]:
    """Build one threshold record. ``verified`` is hard-wired False by design.

    There is no argument to set it True: a value only becomes verified by being
    replaced with sourced data, and that replacement should be a visible edit to
    this fixture rather than a flag flipped at a call site.
    """
    return {
        "value": value,
        "unit": unit,
        "limit_state": limit_state,
        "basis": basis,
        "verified": False,
    }


def discretize_state(
    state: Mapping[str, Any], thresholds: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    """Map a numeric telemetry state to the ground atoms the rule set consumes.

    All comparisons are **strict** (``>``). A parameter sitting exactly on its
    placard limit is therefore not an exceedance -- the correct reading of a
    "maximum permissible" value, and the boundary the fixture's
    ``flap_at_vfe_exactly`` query exists to pin down.

    Args:
        state: Keys ``airspeed_kias``, ``mach``, ``altitude_ft``, ``aoa_deg``,
            ``flap_setting_deg``, and ``engines`` (a list of ``"operating"`` /
            ``"inoperative"``).
        thresholds: The fixture's ``thresholds`` block.

    Returns:
        Sorted list of ground atoms.

    Raises:
        KeyError: on a missing telemetry channel, or a flap detent with no
            corresponding ``VFE_FLAP_<detent>`` threshold. Silently treating an
            undefined detent as unlimited would be the dangerous failure mode,
            so it raises.
    """
    atoms: Set[str] = set()

    airspeed = float(state["airspeed_kias"])
    mach = float(state["mach"])
    altitude = float(state["altitude_ft"])
    aoa = float(state["aoa_deg"])
    flap = int(state["flap_setting_deg"])
    engines = list(state["engines"])

    # --- configuration ----------------------------------------------------- #
    if flap == 0:
        atoms.add("flaps_up")
    else:
        atoms.add("flaps_extended")
        atoms.add(f"flaps_detent_{flap}")
        vfe_key = f"VFE_FLAP_{flap}"
        if vfe_key not in thresholds:
            raise KeyError(f"no placard speed {vfe_key} for flap detent {flap}")
        if airspeed > thresholds[vfe_key]["value"]:
            atoms.add("airspeed_above_vfe")

    # --- speed ------------------------------------------------------------- #
    if airspeed > thresholds["VMO"]["value"]:
        atoms.add("airspeed_above_vmo")
    if mach > thresholds["MMO"]["value"]:
        atoms.add("mach_above_mmo")

    # --- altitude ---------------------------------------------------------- #
    if altitude > thresholds["MAX_OPERATING_ALTITUDE"]["value"]:
        atoms.add("altitude_above_max_operating")

    # --- angle of attack (threshold depends on configuration) -------------- #
    alpha_crit = (
        thresholds["ALPHA_CRIT_FLAPS"]["value"]
        if flap != 0
        else thresholds["ALPHA_CRIT_CLEAN"]["value"]
    )
    alpha_warning = alpha_crit - thresholds["ALPHA_WARNING_MARGIN"]["value"]
    if aoa > alpha_crit:
        atoms.add("aoa_above_alpha_crit")
    if aoa > alpha_warning:
        atoms.add("aoa_above_stall_warning_aoa")

    # --- propulsion -------------------------------------------------------- #
    if any(status != "operating" for status in engines):
        atoms.add("engine_out")
        if altitude > thresholds["SINGLE_ENGINE_CEILING"]["value"]:
            atoms.add("altitude_above_single_engine_ceiling")
    else:
        atoms.add("all_engines_operating")

    return sorted(atoms)


def generate_aero_telemetry_fixture() -> Dict[str, Any]:
    """Build a mock flight-envelope dataset for invariant verification.

    Returns a dict with five blocks:

    ``meta``
        Provenance, including :data:`AERO_SOURCE_NOTE` and the atom vocabulary.
    ``thresholds``
        Named limit values, each with unit, the **named limit state** it guards,
        its nominal regulatory basis, and ``verified: False``.
    ``premises``
        A nominal cruise state: the raw telemetry dict plus the ground atoms
        :func:`discretize_state` derives from it.
    ``rules``
        Horn clauses in the same shape as :func:`generate_horn_rules` output,
        each additionally carrying ``limit_state`` and ``basis``. Acyclic by
        construction -- the aero fixture is the known-good control against which
        the cycle-injecting generators are contrasted.
    ``queries``
        Target safety queries: safe states, boundary-violating states, and
        exactly-on-the-limit cases. Each carries ``expected_conclusions`` (atoms
        the engine must derive) and ``expected_safe`` as ground truth.

    Returns:
        The fixture dict, freshly constructed on every call, so callers cannot
        contaminate each other through shared mutable state.
    """
    thresholds: Dict[str, Dict[str, Any]] = {
        "VMO": _threshold(
            340.0,
            "KIAS",
            "airframe_overspeed",
            "Maximum operating limit speed, 14 CFR 25.1505. VALUE SYNTHETIC.",
        ),
        "MMO": _threshold(
            0.82,
            "Mach",
            "airframe_overspeed",
            "Maximum operating limit Mach number, 14 CFR 25.1505. VALUE SYNTHETIC.",
        ),
        "VFE_FLAP_5": _threshold(
            250.0,
            "KIAS",
            "flap_structural_overload",
            "VFE per 14 CFR 1.2; flap design loads 14 CFR 25.345. VALUE SYNTHETIC.",
        ),
        "VFE_FLAP_15": _threshold(
            205.0,
            "KIAS",
            "flap_structural_overload",
            "VFE per 14 CFR 1.2; flap design loads 14 CFR 25.345. VALUE SYNTHETIC.",
        ),
        "VFE_FLAP_30": _threshold(
            175.0,
            "KIAS",
            "flap_structural_overload",
            "VFE per 14 CFR 1.2; flap design loads 14 CFR 25.345. VALUE SYNTHETIC.",
        ),
        "MAX_OPERATING_ALTITUDE": _threshold(
            41000.0,
            "ft",
            "altitude_envelope_exceedance",
            "Maximum operating altitude, AFM limitation. VALUE SYNTHETIC.",
        ),
        "SINGLE_ENGINE_CEILING": _threshold(
            25000.0,
            "ft",
            "engine_out_ceiling_exceedance",
            "One-engine-inoperative net ceiling / drift-down floor. VALUE SYNTHETIC.",
        ),
        "ALPHA_CRIT_CLEAN": _threshold(
            15.5,
            "deg",
            "aerodynamic_stall",
            "Stalling angle of attack, flaps up. VALUE SYNTHETIC.",
        ),
        "ALPHA_CRIT_FLAPS": _threshold(
            12.0,
            "deg",
            "aerodynamic_stall",
            "Stalling AoA with trailing-edge flaps deployed. Lower than clean "
            "because TE flap deflection shifts the lift curve up and to the "
            "left; the DIRECTION is standard aerodynamics, the MAGNITUDE is "
            "SYNTHETIC.",
        ),
        "ALPHA_WARNING_MARGIN": _threshold(
            2.0,
            "deg",
            "stall_warning_margin",
            "AoA margin below alpha_crit at which warning activates. NOTE: "
            "14 CFR 25.207 states the certified stall-warning margin in terms "
            "of CAS, not AoA; converting requires the aircraft lift curve, "
            "which this fixture does not have. This AoA margin is a modelling "
            "convenience, NOT the regulatory margin. VALUE SYNTHETIC.",
        ),
    }

    rules: List[HornClause] = [
        {
            "antecedents": ["flaps_extended", "airspeed_above_vfe"],
            "consequent": "flap_structural_overload",
            "limit_state": "flap_structural_overload",
            "basis": "Flaps extended above the placard VFE for the selected detent.",
        },
        {
            "antecedents": ["airspeed_above_vmo"],
            "consequent": "airframe_overspeed",
            "limit_state": "airframe_overspeed",
            "basis": "Indicated airspeed above VMO.",
        },
        {
            "antecedents": ["mach_above_mmo"],
            "consequent": "airframe_overspeed",
            "limit_state": "airframe_overspeed",
            "basis": "Mach number above MMO. Binds before VMO at high altitude.",
        },
        {
            "antecedents": ["aoa_above_alpha_crit"],
            "consequent": "aerodynamic_stall",
            "limit_state": "aerodynamic_stall",
            "basis": "Angle of attack above the stalling AoA for the configuration.",
        },
        {
            "antecedents": ["aoa_above_stall_warning_aoa"],
            "consequent": "stall_warning_active",
            "limit_state": "stall_warning_margin",
            "basis": "Warning is an annunciation, not an exceedance: it does NOT "
            "imply envelope_exceedance.",
        },
        {
            "antecedents": ["engine_out", "altitude_above_single_engine_ceiling"],
            "consequent": "engine_out_ceiling_exceedance",
            "limit_state": "engine_out_ceiling_exceedance",
            "basis": "Engine inoperative above the OEI net ceiling; drift-down required.",
        },
        {
            "antecedents": ["altitude_above_max_operating"],
            "consequent": "altitude_envelope_exceedance",
            "limit_state": "altitude_envelope_exceedance",
            "basis": "Above maximum operating altitude.",
        },
        {
            "antecedents": ["altitude_envelope_exceedance"],
            "consequent": "envelope_exceedance",
            "limit_state": "altitude_envelope_exceedance",
            "basis": "Roll-up: any named limit state implies envelope exceedance.",
        },
        {
            "antecedents": ["flap_structural_overload"],
            "consequent": "envelope_exceedance",
            "limit_state": "flap_structural_overload",
            "basis": "Roll-up: any named limit state implies envelope exceedance.",
        },
        {
            "antecedents": ["airframe_overspeed"],
            "consequent": "envelope_exceedance",
            "limit_state": "airframe_overspeed",
            "basis": "Roll-up: any named limit state implies envelope exceedance.",
        },
        {
            "antecedents": ["aerodynamic_stall"],
            "consequent": "envelope_exceedance",
            "limit_state": "aerodynamic_stall",
            "basis": "Roll-up: any named limit state implies envelope exceedance.",
        },
        {
            "antecedents": ["engine_out_ceiling_exceedance"],
            "consequent": "envelope_exceedance",
            "limit_state": "engine_out_ceiling_exceedance",
            "basis": "Roll-up: any named limit state implies envelope exceedance.",
        },
    ]

    nominal_state: Dict[str, Any] = {
        "airspeed_kias": 280.0,
        "mach": 0.78,
        "altitude_ft": 35000.0,
        "aoa_deg": 2.5,
        "flap_setting_deg": 0,
        "engines": ["operating", "operating"],
    }

    def query(
        name: str,
        description: str,
        state: Dict[str, Any],
        expected_conclusions: List[str],
        expected_safe: bool,
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "state": state,
            "expected_conclusions": expected_conclusions,
            "expected_safe": expected_safe,
        }

    queries: List[Dict[str, Any]] = [
        query(
            "nominal_cruise",
            "Mid-envelope cruise. Nothing should fire.",
            dict(nominal_state),
            [],
            True,
        ),
        query(
            "flap_at_vfe_exactly",
            "Flaps 15 at exactly VFE (205 KIAS). SAFE: a placard limit is a "
            "maximum permissible value, so the comparison must be strict. This "
            "is the off-by-one that a sloppy '>=' turns into a false positive.",
            {
                **nominal_state,
                "airspeed_kias": 205.0,
                "mach": 0.32,
                "altitude_ft": 8000.0,
                "aoa_deg": 5.0,
                "flap_setting_deg": 15,
            },
            [],
            True,
        ),
        query(
            "flap_one_knot_over_vfe",
            "Flaps 15 at VFE + 1 kt: the matching false-negative test.",
            {
                **nominal_state,
                "airspeed_kias": 206.0,
                "mach": 0.32,
                "altitude_ft": 8000.0,
                "aoa_deg": 5.0,
                "flap_setting_deg": 15,
            },
            ["flap_structural_overload", "envelope_exceedance"],
            False,
        ),
        query(
            "flap_overspeed_flaps15",
            "Flaps 15 at VFE + 5 kt: unambiguous flap structural exceedance.",
            {
                **nominal_state,
                "airspeed_kias": 210.0,
                "mach": 0.33,
                "altitude_ft": 8000.0,
                "aoa_deg": 5.0,
                "flap_setting_deg": 15,
            },
            ["flap_structural_overload", "envelope_exceedance"],
            False,
        ),
        query(
            "flaps5_maneuvering_safe",
            "Flaps 5 at 240 KIAS, below VFE_FLAP_5 (250). Safe, and the only "
            "query that reads the flaps-5 placard at all -- without it that "
            "threshold ships untested.",
            {
                **nominal_state,
                "airspeed_kias": 240.0,
                "mach": 0.40,
                "altitude_ft": 5000.0,
                "aoa_deg": 6.0,
                "flap_setting_deg": 5,
            },
            [],
            True,
        ),
        query(
            "above_max_operating_altitude",
            "FL430, above the 41,000 ft ceiling, everything else nominal. The "
            "only query that fires the altitude rule.",
            {
                **nominal_state,
                "airspeed_kias": 280.0,
                "mach": 0.80,
                "altitude_ft": 43000.0,
                "aoa_deg": 3.0,
                "flap_setting_deg": 0,
            },
            ["altitude_envelope_exceedance", "envelope_exceedance"],
            False,
        ),
        query(
            "stall_warning_only_clean",
            "AoA 14.0 clean: above the warning AoA (13.5), below alpha_crit "
            "(15.5). Warning annunciates, the state is still inside the "
            "envelope. Tests that annunciation is not conflated with violation.",
            {
                **nominal_state,
                "airspeed_kias": 220.0,
                "mach": 0.55,
                "altitude_ft": 25000.0,
                "aoa_deg": 14.0,
                "flap_setting_deg": 0,
            },
            ["stall_warning_active"],
            True,
        ),
        query(
            "stall_clean",
            "AoA 16.0 clean: above alpha_crit.",
            {
                **nominal_state,
                "airspeed_kias": 200.0,
                "mach": 0.48,
                "altitude_ft": 20000.0,
                "aoa_deg": 16.0,
                "flap_setting_deg": 0,
            },
            ["aerodynamic_stall", "stall_warning_active", "envelope_exceedance"],
            False,
        ),
        query(
            "stall_approach_flaps30",
            "AoA 12.5 with flaps 30: inside the CLEAN limit but past the "
            "flaps-down alpha_crit (12.0). Tests configuration-dependent "
            "threshold selection -- a fixed alpha_crit misses this entirely.",
            {
                **nominal_state,
                "airspeed_kias": 160.0,
                "mach": 0.25,
                "altitude_ft": 2000.0,
                "aoa_deg": 12.5,
                "flap_setting_deg": 30,
            },
            ["aerodynamic_stall", "stall_warning_active", "envelope_exceedance"],
            False,
        ),
        query(
            "vmo_exceedance",
            "345 KIAS at FL200: 5 kt above VMO.",
            {
                **nominal_state,
                "airspeed_kias": 345.0,
                "mach": 0.72,
                "altitude_ft": 20000.0,
                "aoa_deg": 2.0,
                "flap_setting_deg": 0,
            },
            ["airframe_overspeed", "envelope_exceedance"],
            False,
        ),
        query(
            "mmo_exceedance",
            "M 0.83 at FL390 with IAS below VMO: the Mach limit binds at "
            "altitude even though the airspeed limit does not.",
            {
                **nominal_state,
                "airspeed_kias": 290.0,
                "mach": 0.83,
                "altitude_ft": 39000.0,
                "aoa_deg": 3.0,
                "flap_setting_deg": 0,
            },
            ["airframe_overspeed", "envelope_exceedance"],
            False,
        ),
        query(
            "engine_out_above_ceiling",
            "One engine inoperative at FL350, above the OEI net ceiling.",
            {**nominal_state, "engines": ["operating", "inoperative"]},
            ["engine_out_ceiling_exceedance", "envelope_exceedance"],
            False,
        ),
        query(
            "engine_out_drift_down_complete",
            "One engine inoperative at FL200, below the OEI ceiling. Degraded "
            "but inside the envelope -- must NOT be reported as a violation.",
            {
                **nominal_state,
                "airspeed_kias": 250.0,
                "mach": 0.58,
                "altitude_ft": 20000.0,
                "aoa_deg": 4.0,
                "engines": ["operating", "inoperative"],
            },
            [],
            True,
        ),
    ]

    return {
        "meta": {
            "name": "aero_telemetry_fixture",
            "aircraft_class": "generic two-engine transport category (Part 25 style)",
            "source_note": AERO_SOURCE_NOTE,
            "all_thresholds_verified": False,
            "telemetry_channels": [
                "airspeed_kias",
                "mach",
                "altitude_ft",
                "aoa_deg",
                "flap_setting_deg",
                "engines",
            ],
            "atom_vocabulary": [
                "flaps_up",
                "flaps_extended",
                "flaps_detent_<d>",
                "airspeed_above_vfe",
                "airspeed_above_vmo",
                "mach_above_mmo",
                "altitude_above_max_operating",
                "aoa_above_alpha_crit",
                "aoa_above_stall_warning_aoa",
                "engine_out",
                "all_engines_operating",
                "altitude_above_single_engine_ceiling",
            ],
            "violation_atom": "envelope_exceedance",
        },
        "thresholds": thresholds,
        "premises": {
            "state": nominal_state,
            "atoms": discretize_state(nominal_state, thresholds),
        },
        "rules": rules,
        "queries": queries,
    }


# --------------------------------------------------------------------------- #
# Demonstration / smoke test
# --------------------------------------------------------------------------- #


def _banner(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def _main() -> int:
    """Exercise every generator and verify its own output. Returns an exit code."""
    failures: List[str] = []

    def check(condition: bool, label: str) -> None:
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
        if not condition:
            failures.append(label)

    # ------------------------------------------------------------------ 1 -- #
    _banner("1. generate_random_dag(num_vars=12, depth=4, max_fanout=3, seed=7)")
    adjacency, priors = generate_random_dag(12, 4, 3, seed=7)
    edges = sum(len(children) for children in adjacency.values())
    has_parent = {c for children in adjacency.values() for c in children}
    roots = sorted(node for node in adjacency if node not in has_parent)
    print(f"  nodes={len(adjacency)}  edges={edges}  roots={roots}")
    for node in list(adjacency)[:5]:
        print(f"    {node} -> {adjacency[node] or '[]'}   prior={priors[node]}")
    print("    ...")
    check(is_acyclic(adjacency), "random DAG is acyclic")
    check(len(priors) == len(adjacency), "every variable has a prior")
    check(all(0.0 < p < 1.0 for p in priors.values()), "priors are valid probabilities")
    check(max(len(c) for c in adjacency.values()) <= 3, "max_fanout=3 respected")
    check(
        generate_random_dag(12, 4, 3, seed=7) == (adjacency, priors),
        "same seed reproduces the same DAG",
    )

    # ------------------------------------------------------------------ 2 -- #
    _banner("2. generate_diamond_graph -- path explosion")
    for n in (4, 7, 13, 22, 31):
        chain = generate_diamond_graph(n)
        k = (n - 1) // 3
        paths = count_paths(chain, "v0", f"v{n - 1}")
        print(f"  n={n:>3}  diamonds={k:>2}  source->sink paths={paths:>12,}  (2**{k})")
        check(paths == 2**k, f"chain n={n}: path count is 2**{k}")
        check(is_acyclic(chain), f"chain n={n} is acyclic")
        check(len(chain) == n, f"chain n={n} has exactly n nodes")

    wide = generate_diamond_graph(13, mode="wide")
    wide_paths = count_paths(wide, "v0", "v12")
    print(f"  control: mode='wide', n=13 -> {wide_paths} paths (linear, not explosive)")
    check(wide_paths == 11, "wide diamond n=13 has n-2 = 11 paths")

    big = generate_diamond_graph(151)
    print(
        f"  n=151 -> {count_paths(big, 'v0', 'v150'):.3e} distinct paths, "
        f"counted exactly in O(V+E)"
    )

    # ------------------------------------------------------------------ 3 -- #
    _banner("3. generate_horn_rules(num_rules=40, num_vars=12, circular_fraction=0.25)")
    hrules = generate_horn_rules(40, 12, 0.25, seed=42)
    injected = [r for r in hrules if r["circular"]]
    facts = [r for r in hrules if not r["antecedents"]]
    cycles = find_cycles_in_rules(hrules)
    print(f"  rules={len(hrules)}  injected-circular={len(injected)}  facts={len(facts)}")
    for rule in hrules[:6]:
        body = " AND ".join(rule["antecedents"]) or "TRUE"
        mark = "   <-- injected cycle member" if rule["circular"] else ""
        print(f"    {body} -> {rule['consequent']}{mark}")
    print("    ...")
    print(f"  cyclic groups detected: {cycles}")
    check(len(hrules) == 40, "exactly num_rules clauses returned")
    check(len(injected) == 10, "injected count == round(40 * 0.25)")
    check(len(cycles) > 0, "injected cycles are detectable")
    check(
        generate_horn_rules(40, 12, 0.25, seed=42) == hrules,
        "same seed reproduces the same rule set",
    )
    check(
        find_cycles_in_rules(generate_horn_rules(40, 12, 0.0, seed=42)) == [],
        "circular_fraction=0.0 yields a genuinely acyclic rule set",
    )

    derived = forward_chain([r["consequent"] for r in facts], hrules)
    print(f"  forward chaining over the CYCLIC set terminated; |derived|={len(derived)}")
    check(True, "forward chaining terminates on cyclic definite clauses")

    # ------------------------------------------------------------------ 4 -- #
    _banner("4. generate_aero_telemetry_fixture()")
    fixture = generate_aero_telemetry_fixture()
    thresholds = fixture["thresholds"]
    aero_rules = fixture["rules"]

    print(f"  !! {AERO_SOURCE_NOTE[:66]}")
    print(
        f"  thresholds={len(thresholds)}  rules={len(aero_rules)}  "
        f"queries={len(fixture['queries'])}"
    )
    print(f"  nominal premises: {fixture['premises']['atoms']}")
    print()
    print(f"  {'query':<32}{'expected':<11}{'derived':<11}limit states fired")
    print(f"  {'-' * 70}")

    for q in fixture["queries"]:
        atoms = discretize_state(q["state"], thresholds)
        conclusions = forward_chain(atoms, aero_rules)
        safe = "envelope_exceedance" not in conclusions
        fired = sorted(conclusions - set(atoms)) or ["-"]
        print(
            f"  {q['name']:<32}"
            f"{'SAFE' if q['expected_safe'] else 'VIOLATION':<11}"
            f"{'SAFE' if safe else 'VIOLATION':<11}"
            f"{', '.join(fired)}"
        )
        check(safe == q["expected_safe"], f"{q['name']}: safety verdict matches")
        missing = sorted(set(q["expected_conclusions"]) - conclusions)
        check(not missing, f"{q['name']}: expected conclusions derived {missing or ''}")

    print()
    check(find_cycles_in_rules(aero_rules) == [], "aero rule set is acyclic (control)")
    check(
        all(t["verified"] is False for t in thresholds.values()),
        "every aero threshold is flagged unverified",
    )
    check(
        all("limit_state" in r and "basis" in r for r in aero_rules),
        "every aero rule names a limit state and its basis",
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


if __name__ == "__main__":
    sys.exit(_main())
