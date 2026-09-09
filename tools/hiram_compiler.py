from fractions import Fraction
from hiram_ac import Circuit, eval_node, eval_evidence, NodeType, NODE_LITERAL, NODE_PROD, NODE_SUM

FLIGHT_VAR_NAMES = [
    "terrain_clear",          # 0
    "altitude_stable",        # 1
    "descent_permitted",      # 2
    "low_airspeed",           # 3
    "high_angle_of_attack",   # 4
    "stall_hazard",           # 5
    "freezing_temp",          # 6
    "high_humidity",          # 7
    "icing_hazard",           # 8
    "high_g_load",            # 9
    "airframe_stress",        # 10
    "structural_hazard"       # 11
]

class Clause:
    def __init__(self, literals=None):
        self.literals = tuple(literals) if literals is not None else ()

    @classmethod
    def from_or(cls, literals):
        return cls(literals)

    def condition(self, var_id, val):
        new_lits = []
        for v, is_neg in self.literals:
            if v == var_id:
                lit_is_true = (not is_neg) if val else is_neg
                if lit_is_true:
                    return None  # Clause satisfied
            else:
                new_lits.append((v, is_neg))
        return Clause(new_lits)

class Rule:
    """Strict Implication Rule: Antecedents => Consequent."""
    def __init__(self, antecedents, consequent):
        self.antecedents = tuple(antecedents)
        self.consequent = consequent

    def to_clause(self, var_to_id):
        # A1 and A2 => C  <=>  ~A1 or ~A2 or C
        lits = []
        for ant in self.antecedents:
            if isinstance(ant, tuple) and len(ant) == 2:
                var, is_neg = ant
                vid = var_to_id[var] if isinstance(var, str) else var
                lits.append((vid, not bool(is_neg)))
            else:
                vid = var_to_id[ant] if isinstance(ant, str) else ant
                lits.append((vid, True))  # Antecedent positive -> negated in clause

        if isinstance(self.consequent, tuple) and len(self.consequent) == 2:
            var, is_neg = self.consequent
            vid = var_to_id[var] if isinstance(var, str) else var
            lits.append((vid, bool(is_neg)))
        else:
            vid = var_to_id[self.consequent] if isinstance(self.consequent, str) else self.consequent
            lits.append((vid, False))  # Consequent positive -> positive in clause

        return Clause.from_or(lits)

CANONICAL_PRIORS = {
    "terrain_clear": Fraction(99, 100),
    "altitude_stable": Fraction(95, 100),
    "descent_permitted": Fraction(1, 2),
    "low_airspeed": Fraction(1, 20),
    "high_angle_of_attack": Fraction(1, 10),
    "stall_hazard": Fraction(1, 100),
    "freezing_temp": Fraction(1, 5),
    "high_humidity": Fraction(1, 3),
    "icing_hazard": Fraction(1, 50),
    "high_g_load": Fraction(1, 50),
    "airframe_stress": Fraction(1, 4),
    "structural_hazard": Fraction(1, 500)
}

CANONICAL_RULES = [
    Rule(["terrain_clear", "altitude_stable"], "descent_permitted"),
    Rule(["low_airspeed", "high_angle_of_attack"], "stall_hazard"),
    Rule(["freezing_temp", "high_humidity"], "icing_hazard"),
    Rule(["high_g_load", "airframe_stress"], "structural_hazard")
]

class VarMap(dict):
    """Dictionary mapping variable names to IDs with integer fallback."""
    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        if isinstance(key, int):
            return key
        raise KeyError(key)

    def get(self, key, default=None):
        if key in self:
            return super().__getitem__(key)
        if isinstance(key, int):
            return key
        return default

class CompiledKB(tuple):
    def __new__(cls, circuit, root, variables, priors, clauses, var_to_id, id_to_var, rules):
        return super().__new__(cls, (circuit, root))

    def __init__(self, circuit, root, variables, priors, clauses, var_to_id, id_to_var, rules):
        self.circuit = circuit
        self.root = root
        self.root_id = root
        self.variables = variables
        self.priors = priors
        self.clauses = clauses
        self.var_to_id = var_to_id
        self.id_to_var = id_to_var
        self.rules = rules

    def _normalize_evidence(self, ev):
        norm = {}
        if hasattr(ev, 'observed_mask') and hasattr(ev, 'values_mask'):
            for v in range(32):
                if ev.observed_mask & (1 << v):
                    norm[v] = bool(ev.values_mask & (1 << v))
            return norm

        items = ev.items() if isinstance(ev, dict) else ev
        for k, val in items:
            if isinstance(val, str):
                raise TypeError(f"Evidence value for '{k}' must be boolean, got string '{val}'")
            if not isinstance(val, (bool, int)):
                raise TypeError(f"Evidence value for '{k}' must be boolean, got {type(val)}")

            if isinstance(k, str):
                if k not in self.var_to_id:
                    raise KeyError(f"Unknown flight variable name: '{k}'")
                vid = self.var_to_id[k]
            else:
                vid = int(k)
                if vid not in self.id_to_var:
                    raise KeyError(f"Unknown flight variable ID: {vid}")
            norm[vid] = bool(val)
        return norm

    def eval_evidence(self, ev):
        return eval_node(self.circuit, self.root, self._normalize_evidence(ev))

    def eval_node(self, ev):
        return self.eval_evidence(ev)

    def eval(self, ev):
        return self.eval_evidence(ev)

def compile_circuit(variables, priors, clauses):
    if not variables:
        raise ValueError("Variables list cannot be empty")
    var_set = set(variables)
    if len(var_set) != len(variables):
        raise ValueError("Duplicate variable ID detected")

    for v in variables:
        if v not in priors:
            raise KeyError(f"Variable {v} missing prior probability")
    for v, p in priors.items():
        if v not in var_set:
            raise ValueError(f"Prior declared for undeclared variable {v}")
        if p < 0 or p > 1:
            raise ValueError(f"Prior for variable {v} must be in [0, 1], got {p}")

    for c in clauses:
        for v, _ in c.literals:
            if v not in var_set:
                raise ValueError(f"Clause references undeclared variable {v}")

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
            branches.append(circuit.add_node(NODE_PROD, children=[lit_nodes[(v, False)], true_child]))
            weights.append(p)
        if false_child is not None:
            branches.append(circuit.add_node(NODE_PROD, children=[lit_nodes[(v, True)], false_child]))
            weights.append(1 - p)

        sum_node = circuit.add_node(NODE_SUM, children=branches, weights=weights)
        memo[key] = sum_node
        return sum_node

    root = build(0, list(clauses))
    return circuit, root

def compile_from_rules(*args, **kwargs):
    rules = args[0] if len(args) >= 1 else kwargs.get('rules', CANONICAL_RULES)
    priors = args[1] if len(args) >= 2 else kwargs.get('priors', CANONICAL_PRIORS)
    variables = args[2] if len(args) >= 3 else kwargs.get('variables', None)

    if priors is None:
        raise ValueError("Priors must be provided")

    if variables is None:
        raw_vars = list(priors.keys())
    else:
        raw_vars = list(variables)

    var_to_id = VarMap()
    id_to_var = {}
    for i, v in enumerate(raw_vars):
        if isinstance(v, str):
            var_to_id[v] = i
            id_to_var[i] = v
        else:
            name = FLIGHT_VAR_NAMES[v] if v < len(FLIGHT_VAR_NAMES) else f"var_{v}"
            var_to_id[name] = v
            id_to_var[v] = name

    int_variables = [var_to_id[v] if isinstance(v, str) else v for v in raw_vars]
    int_priors = {var_to_id[k] if isinstance(k, str) else k: p for k, p in priors.items()}

    clauses = []
    parsed_rules = []
    for r in (rules or []):
        if isinstance(r, Rule):
            rule_obj = r
        elif isinstance(r, (list, tuple)) and len(r) == 2:
            rule_obj = Rule(r[0], r[1])
        elif isinstance(r, (list, tuple)) and len(r) == 3 and all(isinstance(x, (str, int)) for x in r):
            rule_obj = Rule(r[:2], r[2])
        elif isinstance(r, Clause):
            clauses.append(r)
            parsed_rules.append(r)
            continue
        else:
            raise TypeError(f"Cannot parse rule specification: {r}")

        clauses.append(rule_obj.to_clause(var_to_id))
        parsed_rules.append(rule_obj)

    circuit, root = compile_circuit(int_variables, int_priors, clauses)
    return CompiledKB(circuit, root, raw_vars, priors, clauses, var_to_id, id_to_var, parsed_rules)

def build_reference_kb():
    return compile_from_rules(CANONICAL_RULES, CANONICAL_PRIORS)