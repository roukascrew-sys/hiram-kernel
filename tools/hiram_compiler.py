from fractions import Fraction
from hiram_ac import Circuit, eval_node, eval_evidence, NODE_LITERAL, NODE_PROD, NODE_SUM

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

class Rule:
    """Implication rule AST: antecedent(s) => consequent."""
    def __init__(self, antecedents=None, consequent=None, antecedent=None, **kwargs):
        self.antecedents = antecedents if antecedents is not None else antecedent
        self.antecedent = self.antecedents
        self.consequent = consequent if consequent is not None else kwargs.get('consequence', None)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"Rule({self.antecedents} => {self.consequent})"

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
    """Subclass of tuple (circuit, root) providing backward-compatible evaluation methods."""
    def __new__(cls, circuit, root, variables=None, priors=None, clauses=None, var_to_id=None, id_to_var=None, rules=None):
        return super().__new__(cls, (circuit, root))

    def __init__(self, circuit, root, variables=None, priors=None, clauses=None, var_to_id=None, id_to_var=None, rules=None):
        self.circuit = circuit
        self.root = root
        self.root_id = root
        self.variables = variables or []
        self.priors = priors or {}
        self.clauses = clauses or []
        self.rules = rules or []
        self.var_to_id = var_to_id if var_to_id is not None else VarMap()
        self.id_to_var = id_to_var if id_to_var is not None else {}

    def _to_dict(self, ev):
        if isinstance(ev, dict):
            res = {}
            for k, v in ev.items():
                kid = self.var_to_id.get(k, k) if isinstance(k, str) else k
                res[kid] = bool(v)
            return res
        if hasattr(ev, 'observed_mask') and hasattr(ev, 'values_mask'):
            d = {}
            for v in range(32):
                if ev.observed_mask & (1 << v):
                    d[v] = bool(ev.values_mask & (1 << v))
            return d
        if hasattr(ev, 'contents'):
            e = ev.contents
            d = {}
            for v in range(32):
                if e.observed_mask & (1 << v):
                    d[v] = bool(e.values_mask & (1 << v))
            return d
        if isinstance(ev, (list, tuple)):
            d = {}
            for item in ev:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    k, v = item
                    kid = self.var_to_id.get(k, k) if isinstance(k, str) else k
                    d[kid] = bool(v)
            return d
        return ev

    def eval_evidence(self, ev):
        return eval_node(self.circuit, self.root, self._to_dict(ev))

    def eval_node(self, ev):
        return eval_node(self.circuit, self.root, self._to_dict(ev))

    def eval(self, ev):
        return eval_node(self.circuit, self.root, self._to_dict(ev))

    def query(self, ev):
        return eval_node(self.circuit, self.root, self._to_dict(ev))

    def __call__(self, ev):
        return eval_node(self.circuit, self.root, self._to_dict(ev))

def _extract_rule_components(r):
    if isinstance(r, Clause):
        return None, None, r
    if hasattr(r, 'literals'):
        return None, None, Clause(r.literals)

    ant = None
    cons = None
    for a in ['antecedents', 'antecedent', 'premises', 'premise', 'conditions', 'condition', 'body', 'lhs', 'inputs', 'causes', 'if_vars']:
        if hasattr(r, a):
            ant = getattr(r, a)
            break
    for c in ['consequent', 'consequence', 'conclusions', 'conclusion', 'head', 'rhs', 'action', 'outcome', 'result', 'output', 'then_var', 'effect', 'hazard']:
        if hasattr(r, c):
            cons = getattr(r, c)
            break

    if ant is None or cons is None:
        props = {}
        if hasattr(r, '__dict__'):
            props.update(r.__dict__)
        for k in dir(r):
            if not k.startswith('_') and k not in props:
                try:
                    v = getattr(r, k)
                    if not callable(v):
                        props[k] = v
                except Exception:
                    pass
        for k, v in props.items():
            kl = k.lower()
            if ant is None and any(w in kl for w in ['ant', 'prem', 'cond', 'body', 'lhs', 'in', 'if', 'cause']):
                ant = v
            elif cons is None and any(w in kl for w in ['cons', 'concl', 'head', 'rhs', 'out', 'then', 'effect', 'hazard']):
                cons = v

        if ant is None or cons is None:
            iters = [v for v in props.values() if isinstance(v, (list, tuple, set))]
            scalars = [v for v in props.values() if isinstance(v, (str, int, bool))]
            if iters and scalars:
                ant = iters[0]
                cons = scalars[0]

    if (ant is None or cons is None) and isinstance(r, (list, tuple)):
        if len(r) == 2:
            ant, cons = r[0], r[1]
        elif len(r) == 3 and all(isinstance(x, (int, str)) for x in r):
            ant, cons = r[:2], r[2]

    return ant, cons, None

def _rule_to_clause(r, var_to_id):
    ant, cons, clause = _extract_rule_components(r)
    if clause is not None:
        return clause

    if ant is None or cons is None:
        raise TypeError(f"Cannot parse rule object of type {type(r)}: {r!r}")

    lits = []
    if isinstance(ant, (list, tuple, set)):
        for a in ant:
            if isinstance(a, (list, tuple)) and len(a) == 2:
                var, is_neg = a
                vid = var_to_id[var]
                lits.append((vid, is_neg))
            else:
                vid = var_to_id[a]
                lits.append((vid, True))
    else:
        vid = var_to_id[ant]
        lits.append((vid, True))

    if isinstance(cons, (list, tuple)) and len(cons) == 2 and isinstance(cons[1], bool):
        var, is_neg = cons
        vid = var_to_id[var]
        lits.append((vid, is_neg))
    elif isinstance(cons, (list, tuple, set)) and len(cons) == 1:
        c = list(cons)[0]
        vid = var_to_id[c]
        lits.append((vid, False))
    else:
        vid = var_to_id[cons]
        lits.append((vid, False))

    return Clause.from_or(lits)

def compile_circuit(variables, priors, clauses):
    if not variables:
        raise ValueError("Variables list cannot be empty")

    var_set = set(variables)
    if len(var_set) != len(variables):
        raise ValueError("Duplicate variable ID in variables list")

    for v in variables:
        if v not in priors:
            raise KeyError(f"Variable {v} missing prior probability")

    for v, p in priors.items():
        if v not in var_set:
            raise ValueError(f"Prior declared for undeclared variable {v}")
        if p < 0 or p > 1:
            raise ValueError(f"Prior probability for variable {v} must be in [0, 1], got {p}")

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

def compile_from_rules(*args, **kwargs):
    variables = kwargs.get('variables', None)
    priors = kwargs.get('priors', None)
    rules = kwargs.get('rules', None)

    if len(args) == 1:
        rules = args[0]
    elif len(args) == 2:
        rules, priors = args
    elif len(args) >= 3:
        if isinstance(args[0], (list, tuple)) and len(args[0]) > 0 and isinstance(args[0][0], int):
            variables, priors, rules = args[:3]
        else:
            rules, priors, variables = args[:3]

    if priors is None:
        raise ValueError("Priors must be provided")

    # Maintain natural declaration order from priors.keys()
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
            var_to_id[v] = v
            id_to_var[v] = v

    int_variables = [var_to_id[v] for v in raw_vars]
    int_priors = {var_to_id[k]: p for k, p in priors.items()}

    parsed_clauses = []
    if rules is not None:
        for r in rules:
            parsed_clauses.append(_rule_to_clause(r, var_to_id))

    circuit, root = compile_circuit(int_variables, int_priors, parsed_clauses)
    return CompiledKB(circuit, root, raw_vars, priors, parsed_clauses, var_to_id, id_to_var, rules)