from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import product
from typing import Dict, List, Optional

import math


class NodeType(Enum):
    SUM = "+"
    PROD = "*"
    LITERAL = "LIT"


@dataclass
class ACNode:
    node_id: int
    node_type: NodeType
    # Used if node_type == NodeType.LITERAL
    var_id: Optional[int] = None
    is_negated: bool = False
    # Used if node_type in (NodeType.SUM, NodeType.PROD)
    children: List[int] = field(default_factory=list)
    weights: List[Fraction] = field(default_factory=list)  # 1:1 mapping with children for SUM


class ArithmeticCircuit:
    def __init__(self):
        self.nodes: Dict[int, ACNode] = {}
        self.root_id: Optional[int] = None

    def add_node(self, node: ACNode) -> int:
        self.nodes[node.node_id] = node
        return node.node_id

    def set_root(self, node_id: int):
        self.root_id = node_id

    def get(self, node_id: int) -> ACNode:
        return self.nodes[node_id]

import math

def eval_node(
    circuit: ArithmeticCircuit,
    node_id: int,
    evidence: Dict[int, bool],
    memo: Optional[Dict[int, Fraction]] = None,
) -> Fraction:
    """Evaluates the probability mass of the circuit under partial evidence.

    Guarantees strict O(|Circuit|) execution time via post-order memoization.
    """
    if memo is None:
        memo = {}

    if node_id in memo:
        return memo[node_id]

    node = circuit.nodes[node_id]

    if node.node_type == NodeType.LITERAL:
        # Literal indicator logic:
        # If unobserved, indicator evaluates to 1
        if node.var_id not in evidence:
            val = Fraction(1, 1)
        else:
            observed_val = evidence[node.var_id]
            is_satisfied = (observed_val and not node.is_negated) or (
                not observed_val and node.is_negated
            )
            val = Fraction(1, 1) if is_satisfied else Fraction(0, 1)

    elif node.node_type == NodeType.PROD:
        val = Fraction(1, 1)
        for child_id in node.children:
            child_val = eval_node(circuit, child_id, evidence, memo)
            val *= child_val
            if val == Fraction(0, 1):
                break  # Short-circuit product if zero

    elif node.node_type == NodeType.SUM:
        val = Fraction(0, 1)
        for child_id, weight in zip(node.children, node.weights):
            child_val = eval_node(circuit, child_id, evidence, memo)
            val += weight * child_val

    else:
        raise ValueError(f"Unknown node type: {node.node_type}")

    memo[node_id] = val
    return val
    