from fractions import Fraction
from hiram_ac import Circuit, NODE_LITERAL, NODE_PROD, NODE_SUM

class Clause:
    def __init__(self, literals=None):
        if literals is None:
            self.literals = ()
        else:
            self.literals = tuple(literals)

    @classmethod
    def from_or(cls, literals):
        return cls(literals)

    def condition(self, var_id, val):
        new_lits = []
        for v, is_neg in self.literals:
            if v == var_id:
                lit_is_true = (not is_neg) if val else is_neg
                if lit_is_true:
                    return None
            else:
                new_lits.append((v, is_neg))
        return Clause(new_lits)

def compile_circuit(variables, priors, clauses):
    # 1. Validation: variables list cannot be empty
    if not variables:
        raise ValueError("Variables list cannot be empty")

    var_set = set(variables)
    if len(var_set) != len(variables):
        raise ValueError("Duplicate variable ID in variables list")

    # 2. Validation: priors must exist for all variables and lie in [0, 1]
    for v in variables:
        if v not in priors:
            raise KeyError(f"Variable {v} missing prior probability")

    for v, p in priors.items():
        if v not in var_set:
            raise ValueError(f"Prior declared for undeclared variable {v}")
        if p < 0 or p > 1:
            raise ValueError(f"Prior probability for variable {v} must be in [0, 1], got {p}")

    # 3. Validation: clauses must only reference declared variables
    for c in clauses:
        for v, _ in c.literals:
            if v not in var_set:
                raise ValueError(f"Clause references undeclared variable {v}")

    # 4. If any clause is empty, the CNF formula is unsatisfiable
    if any(len(c.literals) == 0 for c in clauses):
        return Circuit(), None

    circuit = Circuit()
    memo = {}

    lit_nodes = {}
    for v in variables:
        lit_nodes[(v, False)] = circuit.add_node(NODE_LITERAL, var_id=v, is_negated=0)
        lit_nodes[(v, True)] = circuit.add_node(NODE_LITERAL, var_id=v, is_negated=1)

    true_node = circuit.add_node(NODE_PROD, children=[])

    def build(var_idx, current_clauses):
        if any(len(c.literals) == 0 for c in current_clauses):
            return None

        if var_idx == len(variables):
            return true_node

        v = variables[var_idx]
        p = priors[v]

        clause_key = tuple(sorted(tuple(sorted(c.literals)) for c in current_clauses))
        key = (var_idx, clause_key)
        if key in memo:
            return memo[key]

        # Condition on v = True
        c_true = []
        true_dead = False
        for c in current_clauses:
            cond = c.condition(v, True)
            if cond is not None:
                if len(cond.literals) == 0:
                    true_dead = True
                    break
                c_true.append(cond)
        true_child = None if true_dead else build(var_idx + 1, c_true)

        # Condition on v = False
        c_false = []
        false_dead = False
        for c in current_clauses:
            cond = c.condition(v, False)
            if cond is not None:
                if len(cond.literals) == 0:
                    false_dead = True
                    break
                c_false.append(cond)
        false_child = None if false_dead else build(var_idx + 1, c_false)

        if true_child is None and false_child is None:
            memo[key] = None
            return None

        branches = []
        weights = []

        if true_child is not None:
            prod_true = circuit.add_node(NODE_PROD, children=[lit_nodes[(v, False)], true_child])
            branches.append(prod_true)
            weights.append(p)

        if false_child is not None:
            prod_false = circuit.add_node(NODE_PROD, children=[lit_nodes[(v, True)], false_child])
            branches.append(prod_false)
            weights.append(1 - p)

        sum_node = circuit.add_node(NODE_SUM, children=branches, weights=weights)
        memo[key] = sum_node
        return sum_node

    root = build(0, list(clauses))
    return circuit, root