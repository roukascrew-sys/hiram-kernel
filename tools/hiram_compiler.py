from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

from hiram_ac import ACNode, ArithmeticCircuit, NodeType
from hiram_checker import Rule


@dataclass(frozen=True)
class Literal:
    """Represents a propositional variable or its negation."""

    var_id: int
    is_negated: bool


@dataclass
class Clause:
    """A disjunction of literals (L1 v L2 v ... v Lk)."""

    literals: List[Literal]

    def __hash__(self) -> int:
        # Canonical order-independent, deduplicated hash of literals
        return hash(
            frozenset((lit.var_id, lit.is_negated) for lit in self.literals)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Clause):
            return False
        return frozenset(
            (lit.var_id, lit.is_negated) for lit in self.literals
        ) == frozenset((lit.var_id, lit.is_negated) for lit in other.literals)

    @classmethod
    def from_horn(cls, premises: List[int], conclusion: int) -> "Clause":
        """Translates a Horn rule (P1 ^ P2 ^ ... -> C) into a CNF clause:

        (~P1 v ~P2 v ... v C).
        """
        lits = [Literal(var_id=p, is_negated=True) for p in premises]
        lits.append(Literal(var_id=conclusion, is_negated=False))
        return cls(literals=lits)

    @classmethod
    def from_or(cls, lits: List[Tuple[int, bool]]) -> "Clause":
        """Constructs a clause from a list of (var_id, is_negated) tuples."""
        return cls(
            literals=[Literal(var_id=v, is_negated=neg) for v, neg in lits]
        )

    @classmethod
    def from_prohibition(cls, vars: List[int]) -> "Clause":
        """Constructs a mutual exclusion / prohibition clause ~(v1 ^ v2 ^ ...) <=> (~v1 v ~v2 v ...)."""
        return cls(
            literals=[Literal(var_id=v, is_negated=True) for v in vars]
        )


def condition_clauses(
    clauses: List[Clause], var_id: int, assignment: bool
) -> Optional[List[Clause]]:
    """Conditions a set of CNF clauses on a truth assignment (var_id = assignment).

    - If any literal in a clause is satisfied: the clause is subsumed (dropped).
    - If a literal in a clause is falsified: that literal is eliminated from the clause.
    - If a clause becomes empty: a contradiction occurred; returns None.
    """
    simplified: List[Clause] = []

    for clause in clauses:
        clause_satisfied = False
        remaining_lits: List[Literal] = []

        for lit in clause.literals:
            if lit.var_id == var_id:
                is_lit_true = (assignment and not lit.is_negated) or (
                    not assignment and lit.is_negated
                )
                if is_lit_true:
                    clause_satisfied = True
                    break
            else:
                remaining_lits.append(lit)

        if clause_satisfied:
            continue

        if not remaining_lits:
            # All literals in this clause were falsified -> unsatisfiable branch
            return None

        simplified.append(Clause(literals=remaining_lits))

    return simplified


class CompilerContext:
    """Manages node ID allocation, literal node caching, and subproblem memoization."""

    def __init__(self, circuit: ArithmeticCircuit):
        self.circuit = circuit
        self.next_id = 1
        self.lit_cache: Dict[Tuple[int, bool], int] = {}
        # Caches (frozenset of clauses, tuple of remaining variables) -> node_id
        self.subproblem_cache: Dict[
            Tuple[frozenset[Clause], Tuple[int, ...]], Optional[int]
        ] = {}

    def alloc_id(self) -> int:
        nid = self.next_id
        self.next_id += 1
        return nid

    def get_or_create_literal(self, var_id: int, is_negated: bool) -> int:
        key = (var_id, is_negated)
        if key not in self.lit_cache:
            nid = self.alloc_id()
            self.circuit.add_node(
                ACNode(
                    node_id=nid,
                    node_type=NodeType.LITERAL,
                    var_id=var_id,
                    is_negated=is_negated,
                )
            )
            self.lit_cache[key] = nid
        return self.lit_cache[key]


def _compile_recursive(
    ctx: CompilerContext,
    clauses: List[Clause],
    remaining_vars: List[int],
    priors: Dict[int, Fraction],
) -> Optional[int]:
    """Recursively compiles propositional clauses into a minimal Arithmetic Circuit DAG

    using Shannon decomposition and subproblem sharing (OBDD/d-DNNF reduction).
    """
    # Base Case: All variables have been assigned
    if not remaining_vars:
        one_node = ctx.alloc_id()
        ctx.circuit.add_node(ACNode(one_node, NodeType.PROD, children=[]))
        return one_node

    frozen_clauses = frozenset(clauses)
    cache_key = (frozen_clauses, tuple(remaining_vars))

    if cache_key in ctx.subproblem_cache:
        return ctx.subproblem_cache[cache_key]

    var_id = remaining_vars[0]
    tail_vars = remaining_vars[1:]
    prior_true = priors[var_id]
    prior_false = Fraction(1, 1) - prior_true

    # 1. Branch True (var_id = True)
    clauses_true = condition_clauses(clauses, var_id, True)
    if clauses_true is None:
        child_true: Optional[int] = None
    else:
        sub_true = _compile_recursive(ctx, clauses_true, tail_vars, priors)
        if sub_true is None:
            child_true = None
        else:
            lit_true = ctx.get_or_create_literal(var_id, is_negated=False)
            child_true = ctx.alloc_id()
            ctx.circuit.add_node(
                ACNode(child_true, NodeType.PROD, children=[lit_true, sub_true])
            )

    # 2. Branch False (var_id = False)
    clauses_false = condition_clauses(clauses, var_id, False)
    if clauses_false is None:
        child_false: Optional[int] = None
    else:
        sub_false = _compile_recursive(ctx, clauses_false, tail_vars, priors)
        if sub_false is None:
            child_false = None
        else:
            lit_false = ctx.get_or_create_literal(var_id, is_negated=True)
            child_false = ctx.alloc_id()
            ctx.circuit.add_node(
                ACNode(
                    child_false, NodeType.PROD, children=[lit_false, sub_false]
                )
            )

    # 3. Synthesize the SUM Node
    result_node: Optional[int] = None

    if child_true is None and child_false is None:
        result_node = None
    elif child_true is not None and child_false is None:
        sum_id = ctx.alloc_id()
        ctx.circuit.add_node(
            ACNode(
                sum_id,
                NodeType.SUM,
                children=[child_true],
                weights=[prior_true],
            )
        )
        result_node = sum_id
    elif child_true is None and child_false is not None:
        sum_id = ctx.alloc_id()
        ctx.circuit.add_node(
            ACNode(
                sum_id,
                NodeType.SUM,
                children=[child_false],
                weights=[prior_false],
            )
        )
        result_node = sum_id
    else:
        sum_id = ctx.alloc_id()
        ctx.circuit.add_node(
            ACNode(
                sum_id,
                NodeType.SUM,
                children=[child_true, child_false],
                weights=[prior_true, prior_false],
            )
        )
        result_node = sum_id

    ctx.subproblem_cache[cache_key] = result_node
    return result_node


def compile_circuit(
    variables: List[int],
    priors: Dict[int, Fraction],
    clauses: List[Clause],
) -> Tuple[ArithmeticCircuit, Optional[int]]:
    """Compiles propositional CNF clauses over variables into an Arithmetic Circuit."""
    circuit = ArithmeticCircuit()
    ctx = CompilerContext(circuit)
    root_id = _compile_recursive(ctx, clauses, variables, priors)
    return circuit, root_id


@dataclass
class CompiledKnowledgeBase:
    """The immutable runtime representation of a compiled avionics knowledge base."""

    circuit: ArithmeticCircuit
    root_id: int
    var_to_id: Dict[str, int]
    id_to_var: Dict[int, str]


def compile_from_rules(
    rules: List[Rule],
    priors: Dict[str, Fraction],
) -> CompiledKnowledgeBase:
    """Compiles symbolic Horn rules and string priors into an executable Arithmetic Circuit DAG."""
    var_names: List[str] = list(priors.keys())

    for r in rules:
        for p in r.premises:
            if p not in priors:
                raise ValueError(
                    f"Missing prior probability for rule premise: '{p}'"
                )
        if r.conclusion not in priors:
            raise ValueError(
                f"Missing prior probability for rule conclusion: '{r.conclusion}'"
            )

    var_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(var_names)}
    id_to_var: Dict[int, str] = {idx: name for idx, name in enumerate(var_names)}

    int_priors: Dict[int, Fraction] = {
        var_to_id[name]: p for name, p in priors.items()
    }

    clauses: List[Clause] = []
    for r in rules:
        premise_ids = [var_to_id[p] for p in r.premises]
        conclusion_id = var_to_id[r.conclusion]
        clauses.append(Clause.from_horn(premise_ids, conclusion_id))

    variables = list(range(len(var_names)))
    circuit, root_id = compile_circuit(variables, int_priors, clauses)

    if root_id is None:
        raise ValueError(
            "Knowledge base rules form an unsatisfiable contradiction (0 possible worlds)."
        )

    return CompiledKnowledgeBase(
        circuit=circuit,
        root_id=root_id,
        var_to_id=var_to_id,
        id_to_var=id_to_var,
    )