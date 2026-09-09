from fractions import Fraction
from enum import IntEnum

class NodeType(IntEnum):
    NODE_LITERAL = 0
    NODE_PROD = 1
    NODE_SUM = 2

NODE_LITERAL = NodeType.NODE_LITERAL
NODE_PROD = NodeType.NODE_PROD
NODE_SUM = NodeType.NODE_SUM

class Circuit:
    def __init__(self):
        self.nodes = []

    def add_node(self, node_type, **kwargs):
        nid = len(self.nodes)
        node = {'id': nid, 'type': int(node_type), **kwargs}
        self.nodes.append(node)
        return nid

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __len__(self):
        return len(self.nodes)

def eval_node(circuit, root_id, evidence):
    if root_id is None:
        return Fraction(0)
    memo = {}

    def _eval(nid):
        if nid in memo:
            return memo[nid]
        node = circuit[nid]
        ntype = node['type']

        if ntype == NODE_LITERAL:
            vid = node['var_id']
            if vid not in evidence:
                res = Fraction(1)
            else:
                val = evidence[vid]
                neg = node['is_negated']
                sat = (val != bool(neg))
                res = Fraction(1) if sat else Fraction(0)
        elif ntype == NODE_PROD:
            res = Fraction(1)
            for ch in node['children']:
                res *= _eval(ch)
                if res == 0:
                    break
        elif ntype == NODE_SUM:
            res = Fraction(0)
            for ch, w in zip(node['children'], node['weights']):
                res += w * _eval(ch)
        else:
            raise ValueError(f"Unknown node type: {ntype}")

        memo[nid] = res
        return res

    return _eval(root_id)

eval_evidence = eval_node