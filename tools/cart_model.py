"""
tools.cart_model - Compiler-driven causal DAG construction for the HIRAM
safety kernel's cart model (Task 2.1).

Builds a Bayesian-network-shaped intermediate representation (IR) for the
1D cart stopping problem's causal structure -- physical latent state
(TrackCondition, TrueObstacle, DecelCapability) and the noisy sensor
observations of it (LidarObs, TofObs, WheelSlipObs) -- plus a decision-loss
matrix over actions and a subset of that causal state, and compiles both to
a flat, row-major, statically-indexable IR suitable for a Cortex-M7 static
memory buffer: every CPT and the loss matrix map to one contiguous array
with precomputed parent/dependency strides, so a target evaluator can index
into them with pure integer arithmetic and no dynamic allocation.

Zero external dependencies: standard library only (json, math, hashlib,
dataclasses, typing). No numpy/scipy/pgmpy.

Invariant preservation:
  * Every CPT and loss-matrix float is round-tripped through float.hex()
    (IEEE 754 bit-exact hex), not just str()/repr(), so cpt_hex/matrix_hex
    and the compiled model's digest are lossless -- float.fromhex() on any
    entry reconstructs the exact original float, with no decimal rounding
    ambiguity.
  * Sensor-observation nodes carry a literal "DROPOUT" state (state index 2
    for every sensor node in this model) representing a missing/unobserved
    reading -- the tri-state analogue of the None-valued dropout semantics
    used elsewhere in this benchmark (sim.sensors.pipeline.SensorObservation
    sets d_critical/d_marginal/sensor_disagree to None during a dropout
    window). compile() records each node's dropout_state_index (None if the
    node has no such state) so a downstream evaluator can treat that index
    as "no evidence" consistently, without needing to special-case a
    string comparison against "DROPOUT" itself.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "1.0.0"
DROPOUT_STATE_NAME = "DROPOUT"


def _row_major_strides(dims: Sequence[int]) -> List[int]:
    """
    C-order (row-major) strides for a tensor of shape `dims`: the last axis
    is fastest-varying (stride 1), each preceding axis's stride is the
    product of every dimension after it. flat_index = sum(idx[i] * strides[i]).
    """
    strides = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        strides[i] = strides[i + 1] * dims[i + 1]
    return strides


def _product(dims: Sequence[int]) -> int:
    total = 1
    for d in dims:
        total *= d
    return total


@dataclass(frozen=True)
class _NodeSpec:
    name: str
    states: Tuple[str, ...]
    parents: Tuple[str, ...]
    cpt: Tuple[float, ...]


@dataclass(frozen=True)
class _LossSpec:
    actions: Tuple[str, ...]
    state_dependencies: Tuple[str, ...]
    matrix: Tuple[float, ...]


class CausalDAGBuilder:
    """
    Incrementally builds a causal DAG (nodes + CPTs) and an associated
    decision-loss matrix, then compiles both to a flat IR.

    Node registration is deliberately permissive about ordering: add_node()
    does not require a node's parents to already be registered (nor does it
    validate cpt length/row sums against them), so that a genuinely cyclic
    graph *can* be constructed and then rejected by topological_sort() /
    compile() -- if parents had to pre-exist, no cycle could ever be built
    in the first place, and the cycle-detection path would be untestable.
    All structural validation (unknown parents, cycles, cpt shape, row
    sums, loss-matrix shape) happens in topological_sort() / compile().
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, _NodeSpec] = {}
        self._loss: Optional[_LossSpec] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_node(
        self,
        name: str,
        states: Sequence[str],
        parents: Sequence[str],
        cpt: Sequence[float],
    ) -> None:
        if not name:
            raise ValueError("node name must be a non-empty string")
        if name in self._nodes:
            raise ValueError(f"node '{name}' already added")
        if not states:
            raise ValueError(f"node '{name}': states must be non-empty")
        if len(set(states)) != len(states):
            raise ValueError(f"node '{name}': states must be unique, got {states!r}")
        if name in parents:
            raise ValueError(f"node '{name}': cannot list itself as its own parent")
        if not cpt:
            raise ValueError(f"node '{name}': cpt must be non-empty")

        self._nodes[name] = _NodeSpec(
            name=name,
            states=tuple(states),
            parents=tuple(parents),
            cpt=tuple(float(v) for v in cpt),
        )

    def set_loss_matrix(
        self,
        actions: Sequence[str],
        state_dependencies: Sequence[str],
        matrix: Sequence[float],
    ) -> None:
        if not actions:
            raise ValueError("loss matrix: actions must be non-empty")
        if len(set(actions)) != len(actions):
            raise ValueError(f"loss matrix: actions must be unique, got {actions!r}")
        if not matrix:
            raise ValueError("loss matrix: matrix must be non-empty")

        self._loss = _LossSpec(
            actions=tuple(actions),
            state_dependencies=tuple(state_dependencies),
            matrix=tuple(float(v) for v in matrix),
        )

    # ------------------------------------------------------------------
    # Graph algorithms
    # ------------------------------------------------------------------
    def topological_sort(self) -> List[str]:
        """
        A deterministic valid topological order of the registered nodes
        (parents strictly before children). Ties are broken by add_node()
        insertion order -- among all currently-ready nodes, the one added
        earliest is always emitted first -- so the result does not depend
        on dict/set iteration order and is stable across runs.

        Raises ValueError if any node references a parent that was never
        registered, or if the graph contains a cycle.
        """
        insertion_order = list(self._nodes.keys())

        for name, spec in self._nodes.items():
            for parent in spec.parents:
                if parent not in self._nodes:
                    raise ValueError(f"node '{name}' references unknown parent '{parent}'")

        children: Dict[str, List[str]] = {name: [] for name in self._nodes}
        in_degree: Dict[str, int] = {name: 0 for name in self._nodes}
        for name, spec in self._nodes.items():
            for parent in spec.parents:
                children[parent].append(name)
                in_degree[name] += 1

        ready = [n for n in insertion_order if in_degree[n] == 0]
        order: List[str] = []
        while ready:
            ready.sort(key=insertion_order.index)
            n = ready.pop(0)
            order.append(n)
            for child in children[n]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)

        if len(order) != len(self._nodes):
            unresolved = [n for n in insertion_order if n not in order]
            raise ValueError(f"cycle detected: could not order node(s) {unresolved}")

        return order

    def compute_strides(self, node_name: str) -> Tuple[Dict[str, int], int]:
        """
        Row-major strides for `node_name`'s CPT, treating its flat array as
        a tensor of shape (parent_1_states, ..., parent_k_states,
        own_states) -- parents in the order given to add_node(), own state
        fastest-varying (implicit stride 1, not included in the returned
        dict, which is keyed by parent name).

        Returns (parent_strides, total_parameter_count).
        """
        spec = self._require_node(node_name)
        parent_sizes = [len(self._require_node(p).states) for p in spec.parents]
        dims = parent_sizes + [len(spec.states)]
        strides = _row_major_strides(dims)
        return dict(zip(spec.parents, strides[:-1])), _product(dims)

    def compute_loss_strides(self) -> Tuple[int, Dict[str, int], int]:
        """
        Row-major strides for the loss matrix, treating its flat array as a
        tensor of shape (n_actions, dep_1_states, ..., dep_m_states) --
        actions outermost/slowest, dependencies in the given order after
        that (first dependency slower than later ones).

        Returns (action_stride, dependency_strides, total_parameter_count).
        """
        if self._loss is None:
            raise ValueError("loss matrix has not been set")
        dep_sizes = [len(self._require_node(d).states) for d in self._loss.state_dependencies]
        dims = [len(self._loss.actions)] + dep_sizes
        strides = _row_major_strides(dims)
        return strides[0], dict(zip(self._loss.state_dependencies, strides[1:])), _product(dims)

    def flat_cpt_index(self, node_name: str, parent_state_indices: Sequence[int], state_index: int) -> int:
        """Flat index of one (parent-config, own-state) cell in node_name's CPT."""
        spec = self._require_node(node_name)
        if len(parent_state_indices) != len(spec.parents):
            raise ValueError(
                f"node '{node_name}': expected {len(spec.parents)} parent indices, got {len(parent_state_indices)}"
            )
        strides, total = self.compute_strides(node_name)

        # Validate every axis individually, not just the resulting flat
        # index against the total -- an invalid state_index can otherwise
        # still land within [0, total) by coincidence and silently alias a
        # neighboring cell instead of being caught.
        for parent, i in zip(spec.parents, parent_state_indices):
            parent_size = len(self._nodes[parent].states)
            if not 0 <= i < parent_size:
                raise IndexError(
                    f"node '{node_name}': parent '{parent}' state index {i} out of range [0, {parent_size})"
                )
        n_states = len(spec.states)
        if not 0 <= state_index < n_states:
            raise IndexError(f"node '{node_name}': state index {state_index} out of range [0, {n_states})")

        idx = sum(i * strides[p] for p, i in zip(spec.parents, parent_state_indices))
        idx += state_index
        assert 0 <= idx < total  # guaranteed by the per-axis checks above
        return idx

    def flat_loss_index(self, action_index: int, dependency_state_indices: Sequence[int]) -> int:
        """Flat index of one (action, dependency-state-config) cell in the loss matrix."""
        if self._loss is None:
            raise ValueError("loss matrix has not been set")
        if len(dependency_state_indices) != len(self._loss.state_dependencies):
            raise ValueError(
                f"loss matrix: expected {len(self._loss.state_dependencies)} dependency indices, "
                f"got {len(dependency_state_indices)}"
            )
        action_stride, dep_strides, total = self.compute_loss_strides()

        n_actions = len(self._loss.actions)
        if not 0 <= action_index < n_actions:
            raise IndexError(f"loss matrix: action index {action_index} out of range [0, {n_actions})")
        for dep, i in zip(self._loss.state_dependencies, dependency_state_indices):
            dep_size = len(self._nodes[dep].states)
            if not 0 <= i < dep_size:
                raise IndexError(f"loss matrix: dependency '{dep}' state index {i} out of range [0, {dep_size})")

        idx = action_index * action_stride
        idx += sum(i * dep_strides[d] for d, i in zip(self._loss.state_dependencies, dependency_state_indices))
        assert 0 <= idx < total  # guaranteed by the per-axis checks above
        return idx

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------
    def compile(self) -> dict:
        """
        Validate the full DAG (acyclic, every CPT row sums to 1.0, the loss
        matrix's shape matches its declared dependencies) and export a
        structured, JSON-serializable IR: flat row-major CPTs and loss
        matrix, precomputed strides, lossless float.hex() encodings of
        every parameter, and a SHA-256 digest sealing the whole IR.
        """
        order = self.topological_sort()

        nodes_ir: Dict[str, dict] = {}
        for name in order:
            spec = self._nodes[name]
            parent_strides, total = self.compute_strides(name)

            if len(spec.cpt) != total:
                raise ValueError(
                    f"node '{name}': cpt has {len(spec.cpt)} entries, expected {total} "
                    f"({' x '.join(str(len(self._nodes[p].states)) for p in spec.parents)}"
                    f"{' x ' if spec.parents else ''}{len(spec.states)})"
                )

            n_states = len(spec.states)
            n_rows = total // n_states
            for row in range(n_rows):
                row_vals = spec.cpt[row * n_states : (row + 1) * n_states]
                row_sum = math.fsum(row_vals)
                if not math.isclose(row_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        f"node '{name}' row {row} (values {row_vals}) sums to {row_sum!r}, expected 1.0"
                    )

            dropout_index = spec.states.index(DROPOUT_STATE_NAME) if DROPOUT_STATE_NAME in spec.states else None

            nodes_ir[name] = {
                "name": name,
                "states": list(spec.states),
                "parents": list(spec.parents),
                "cpt": list(spec.cpt),
                "cpt_hex": [v.hex() for v in spec.cpt],
                "parent_strides": parent_strides,
                "own_state_stride": 1,
                "total_parameters": total,
                "dropout_state_index": dropout_index,
                "is_sensor_variable": dropout_index is not None,
            }

        loss_ir: Optional[dict] = None
        if self._loss is not None:
            for dep in self._loss.state_dependencies:
                if dep not in self._nodes:
                    raise ValueError(f"loss matrix depends on unknown node '{dep}'")

            action_stride, dep_strides, total = self.compute_loss_strides()
            if len(self._loss.matrix) != total:
                raise ValueError(
                    f"loss matrix has {len(self._loss.matrix)} entries, expected {total} "
                    f"({len(self._loss.actions)} actions x "
                    f"{' x '.join(str(len(self._nodes[d].states)) for d in self._loss.state_dependencies)} states)"
                )

            loss_ir = {
                "actions": list(self._loss.actions),
                "state_dependencies": list(self._loss.state_dependencies),
                "matrix": list(self._loss.matrix),
                "matrix_hex": [v.hex() for v in self._loss.matrix],
                "action_stride": action_stride,
                "dependency_strides": dep_strides,
                "total_parameters": total,
            }

        ir = {
            "schema_version": SCHEMA_VERSION,
            "topological_order": order,
            "nodes": nodes_ir,
            "loss_matrix": loss_ir,
        }
        ir["digest"] = self._compute_digest(ir)
        return ir

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_node(self, name: str) -> _NodeSpec:
        if name not in self._nodes:
            raise ValueError(f"unknown node '{name}'")
        return self._nodes[name]

    @staticmethod
    def _compute_digest(ir: dict) -> str:
        """
        SHA-256 over the IR's structure and every parameter's lossless
        float.hex() encoding -- never over decimal repr()/str(), which can
        alias distinct bit patterns to the same rounded decimal string.
        """
        hasher = hashlib.sha256()
        hasher.update(f"schema={ir['schema_version']}\n".encode("utf-8"))
        hasher.update(f"order={','.join(ir['topological_order'])}\n".encode("utf-8"))
        for name in ir["topological_order"]:
            node = ir["nodes"][name]
            hasher.update(
                (
                    f"node={name}|states={','.join(node['states'])}|"
                    f"parents={','.join(node['parents'])}|dropout={node['dropout_state_index']}\n"
                ).encode("utf-8")
            )
            hasher.update((",".join(node["cpt_hex"]) + "\n").encode("utf-8"))
        lm = ir["loss_matrix"]
        if lm is not None:
            hasher.update(
                (
                    f"loss_actions={','.join(lm['actions'])}|"
                    f"loss_deps={','.join(lm['state_dependencies'])}\n"
                ).encode("utf-8")
            )
            hasher.update((",".join(lm["matrix_hex"]) + "\n").encode("utf-8"))
        return hasher.hexdigest()


def build_default_cart_model() -> CausalDAGBuilder:
    """The cart model's causal topology: physical latent state, its noisy
    sensor observations, and the action/state-dependent decision-loss table."""
    builder = CausalDAGBuilder()

    builder.add_node(
        name="TrackCondition",
        states=["DRY", "WET", "ICY"],
        parents=[],
        cpt=[0.70, 0.20, 0.10],
    )
    builder.add_node(
        name="TrueObstacle",
        states=["CLEAR", "BLOCKED"],
        parents=["TrackCondition"],
        cpt=[0.95, 0.05, 0.90, 0.10, 0.85, 0.15],
    )
    builder.add_node(
        name="DecelCapability",
        states=["NOMINAL", "DEGRADED", "CRITICAL"],
        parents=["TrackCondition"],
        cpt=[0.92, 0.07, 0.01, 0.20, 0.70, 0.10, 0.05, 0.35, 0.60],
    )
    builder.add_node(
        name="LidarObs",
        states=["CLEAR", "DETECTED", "DROPOUT"],
        parents=["TrueObstacle"],
        cpt=[0.98, 0.01, 0.01, 0.02, 0.95, 0.03],
    )
    builder.add_node(
        name="TofObs",
        states=["CLEAR", "DETECTED", "DROPOUT"],
        parents=["TrueObstacle"],
        cpt=[0.96, 0.02, 0.02, 0.04, 0.92, 0.04],
    )
    builder.add_node(
        name="WheelSlipObs",
        states=["LOW", "HIGH", "DROPOUT"],
        parents=["DecelCapability"],
        cpt=[0.95, 0.04, 0.01, 0.30, 0.68, 0.02, 0.05, 0.90, 0.05],
    )

    builder.set_loss_matrix(
        actions=["ACCEL", "COAST", "EMERGENCY_BRAKE"],
        state_dependencies=["TrueObstacle", "DecelCapability"],
        matrix=[
            0.0, 0.0, 0.0, 500.0, 1000.0, 2000.0,
            2.0, 2.0, 2.0, 100.0, 300.0, 800.0,
            25.0, 25.0, 25.0, 5.0, 20.0, 50.0,
        ],
    )

    return builder


def main() -> None:
    builder = build_default_cart_model()
    ir = builder.compile()

    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "data" / "models" / "causal_dag.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ir, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"Compiled causal DAG: {len(ir['nodes'])} nodes, topological order: {ir['topological_order']}")
    print(f"Loss matrix: {len(ir['loss_matrix']['matrix'])} entries over "
          f"{ir['loss_matrix']['actions']} x {ir['loss_matrix']['state_dependencies']}")
    print(f"Digest: {ir['digest']}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
