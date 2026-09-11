"""
tools.inference_engine - Factor reduction & expected-loss inference engine
for the HIRAM safety kernel's cart model (Task 2.2).

Loads the sealed causal DAG IR compiled by tools.cart_model (Task 2.1) and
answers two questions given tri-state sensor evidence:

  1. What is the posterior P(TrueObstacle, DecelCapability | evidence)?
     TrackCondition is the single latent common cause of both TrueObstacle
     and DecelCapability, so it is analytically marginalized (summed) out;
     it is never part of the returned posterior. Each sensor's CPT depends
     only on its own parent (LidarObs/TofObs on TrueObstacle, WheelSlipObs
     on DecelCapability), so its likelihood factors independently and an
     unobserved sensor (evidence value None) contributes a factor of
     exactly 1.0 -- marginalizing it out analytically rather than summing
     over its states, since a proper CPT's rows already sum to 1.0.

  2. Given that posterior, which action (ACCEL / COAST / EMERGENCY_BRAKE)
     minimizes expected loss under the model's decision-loss matrix?

Zero external dependencies: standard library only (json, math, hashlib,
pathlib, typing). No numpy/scipy/pgmpy.

Sealed custody: __init__ verifies the loaded model's "ir_digest" two ways --
(a) internal self-consistency: recomputing the canonical digest from the
loaded content (the same algorithm tools.cart_model.CausalDAGBuilder.compile
uses) must match the file's own stored "ir_digest" field, catching a
tamperer who edits a CPT/loss value but leaves the stored digest string
untouched; (b) baseline match: the (verified-consistent) digest must equal
this specific cart model's sealed value, catching a tamperer who edits
content *and* correctly recomputes+rewrites ir_digest, but is not looking at
the certified model. Either mismatch raises ValueError.

Determinism & bounds: every CPT/loss lookup is a plain list index computed
from the model's own precomputed strides (never re-derived from scratch),
so inference is a fixed, small number of arithmetic operations over
statically-shaped row-major arrays -- no dynamic allocation, no recursion,
no data-dependent control flow beyond which evidence values were supplied.

Tri-state evidence: each sensor's evidence value is an int in {0, 1, 2}
(matching that sensor's 3 declared states, index 2 always being "DROPOUT"
per Task 2.1's tri-state invariant) or None for "not observed this step".

Dropout ("2") vs unobserved ("None") contract: by design, a reported
DROPOUT reading carries no information about the sensor's parent -- it
means the telemetry link produced nothing usable, not "the parent is in
some particular state." So DROPOUT and "not observed" are semantically the
same event, and BayesianInferenceEngine treats them identically by default
(marginalize_dropout_state=True): both marginalize the sensor out
analytically (likelihood factor 1.0 for every parent state), not just an
unobserved key. This is exact, not an approximation, because a CPT's row
already sums to 1.0: summing a sensor's likelihood over all 3 of its states
(including DROPOUT) is mathematically identical to never having asked the
CPT at all. Pass marginalize_dropout_state=False to instead treat an
observed DROPOUT value as ordinary evidence (i.e. condition on "the sensor
reported dropout", using its DROPOUT-row CPT entries, which is a different
and non-trivial computation from marginalizing it out).

Python Reference Engine: this module is the algorithmic reference
implementation of the inference/decision logic, prioritizing clarity and
direct correspondence to the mathematical definitions above. Its "zero
dynamic memory allocation" and "static row-major array" properties describe
the *shape* of the computation (fixed-size lookups and loops driven by the
model's precomputed strides, no data-dependent recursion or growth) -- they
are not a literal guarantee about the CPython interpreter, which allocates
objects for lists/tuples/floats as a matter of course. The actual
zero-allocation guarantee is a target for the downstream C implementation
(Phase 3), which this module's arithmetic is designed to port to directly.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EvidenceValue = Optional[int]

# The three observation nodes this engine conditions on, and the two latent
# decision-relevant nodes (TrueObstacle, DecelCapability) whose joint
# posterior it returns, with TrackCondition marginalized out as the common
# cause. This structural role assignment is fixed to *this* sealed model;
# it is safe to hardcode because a digest mismatch (any topology change)
# is rejected before any of it is relied upon.
_SENSOR_NODES = ("LidarObs", "TofObs", "WheelSlipObs")
_TRACK_NODE = "TrackCondition"
_OBSTACLE_NODE = "TrueObstacle"
_DECEL_NODE = "DecelCapability"

# Lower rank = preferred when multiple actions tie on expected loss.
_SAFETY_TIE_BREAK_PRIORITY = {"EMERGENCY_BRAKE": 0, "COAST": 1, "ACCEL": 2}

_ZERO_LIKELIHOOD_GUARD = 1e-15


def _canonical_digest(model: dict) -> str:
    """
    The exact canonical-digest algorithm tools.cart_model.CausalDAGBuilder
    .compile() uses for "ir_digest": SHA-256 over the model dict (minus any
    digest fields) via canonical JSON (sort_keys + fixed separators).
    Reimplemented here (rather than imported) so this module has no
    compile-time dependency on tools.cart_model -- it only ever reads
    already-compiled model files, never builds them.
    """
    without_digests = {k: v for k, v in model.items() if k not in ("digest", "ir_digest")}
    return hashlib.sha256(
        json.dumps(without_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class BayesianInferenceEngine:
    """
    Loads a sealed causal-DAG IR and performs exact inference (via direct
    marginalization -- the model is small and fixed-shape, so a general
    variable-elimination engine would be unneeded machinery) over
    TrueObstacle/DecelCapability given tri-state sensor evidence, followed
    by expected-loss minimization over the model's action set.
    """

    #: This cart model's sealed baseline digest (Task 2.1, hardened per the
    #: Astra audit). A freshly-compiled model that doesn't match this exact
    #: value is not the certified model this engine was built against.
    EXPECTED_IR_DIGEST = "b39845824f903e9230c75110ffd822ccea36cd89a5ebb4eae1235ff4860b57e7"

    def __init__(
        self,
        model_path: str = "data/models/causal_dag.json",
        marginalize_dropout_state: bool = True,
    ) -> None:
        """
        marginalize_dropout_state: when True (the default -- see the module
        docstring's "Dropout vs unobserved" section), an evidence value
        equal to a sensor's DROPOUT state index is treated identically to
        that sensor being unobserved (None): marginalized out analytically.
        When False, an observed DROPOUT value is instead used as ordinary
        evidence (conditioning on "the sensor reported dropout").
        """
        self.marginalize_dropout_state = marginalize_dropout_state

        path = Path(model_path)
        try:
            model = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load causal DAG model from '{model_path}': {exc}") from exc

        if not isinstance(model, dict) or "ir_digest" not in model:
            raise ValueError(f"model at '{model_path}' is missing the 'ir_digest' custody seal")

        stored_digest = model["ir_digest"]
        recomputed_digest = _canonical_digest(model)
        if recomputed_digest != stored_digest:
            raise ValueError(
                f"ir_digest custody violation loading '{model_path}': stored digest "
                f"{stored_digest!r} does not match the digest recomputed from the file's own "
                f"content ({recomputed_digest!r}) -- the model has been tampered with"
            )
        if stored_digest != self.EXPECTED_IR_DIGEST:
            raise ValueError(
                f"ir_digest custody violation loading '{model_path}': digest {stored_digest!r} "
                f"(internally self-consistent) does not match the sealed baseline "
                f"{self.EXPECTED_IR_DIGEST!r} -- this is not the certified cart model"
            )

        # Past this point, `model`'s content is byte-identical to the sealed
        # baseline, so its structure (node names, state counts, parent
        # roles) can be trusted without further defensive parsing.
        self._model = model
        self._nodes: Dict[str, dict] = model["nodes"]
        self._loss: dict = model["loss_matrix"]

        self._n_obstacle = len(self._nodes[_OBSTACLE_NODE]["states"])
        self._n_decel = len(self._nodes[_DECEL_NODE]["states"])
        self._n_track = len(self._nodes[_TRACK_NODE]["states"])
        self._n_posterior = self._n_obstacle * self._n_decel

        self._track_cpt: List[float] = self._nodes[_TRACK_NODE]["cpt"]
        self._obstacle_cpt: List[float] = self._nodes[_OBSTACLE_NODE]["cpt"]
        self._decel_cpt: List[float] = self._nodes[_DECEL_NODE]["cpt"]
        self._sensor_cpt: Dict[str, List[float]] = {s: self._nodes[s]["cpt"] for s in _SENSOR_NODES}
        self._dropout_state_index: Dict[str, Optional[int]] = {
            s: self._nodes[s].get("dropout_state_index") for s in _SENSOR_NODES
        }

        self._track_to_obstacle_stride = self._nodes[_OBSTACLE_NODE]["parent_strides"][_TRACK_NODE]
        self._track_to_decel_stride = self._nodes[_DECEL_NODE]["parent_strides"][_TRACK_NODE]
        self._obstacle_to_lidar_stride = self._nodes["LidarObs"]["parent_strides"][_OBSTACLE_NODE]
        self._obstacle_to_tof_stride = self._nodes["TofObs"]["parent_strides"][_OBSTACLE_NODE]
        self._decel_to_wheel_stride = self._nodes["WheelSlipObs"]["parent_strides"][_DECEL_NODE]

        self._actions: List[str] = self._loss["actions"]
        self._loss_values: List[float] = self._loss["matrix"]
        self._loss_action_stride: int = self._loss["action_stride"]

        if "EMERGENCY_BRAKE" not in self._actions:
            raise ValueError(
                f"model's loss matrix has no 'EMERGENCY_BRAKE' action ({self._actions!r}); "
                "this engine's safety fallbacks require it to exist"
            )
        self._emergency_brake_index = self._actions.index("EMERGENCY_BRAKE")

        # The true physical prior P(TrueObstacle, DecelCapability), computed
        # once with no evidence at all (TrackCondition marginalized, no
        # sensor factors applied). Used as the conservative fallback
        # distribution when evidence is contradictory/degenerate (see
        # _compute_posterior_with_diagnostics) instead of an arbitrary
        # uniform guess. Safe to compute via the normal inference path here:
        # with empty evidence the likelihood is always comfortably nonzero
        # for a well-formed model (every CPT row sums to 1.0), so this call
        # can never itself hit the fallback branch that would reference
        # self._prior_marginal before it exists.
        self._prior_marginal, _ = self._compute_posterior_with_diagnostics({})

    # ------------------------------------------------------------------
    # Evidence validation
    # ------------------------------------------------------------------
    def _extract_evidence_value(self, evidence: Dict[str, EvidenceValue], sensor_name: str) -> EvidenceValue:
        value = evidence.get(sensor_name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"evidence for '{sensor_name}' must be an int or None, got {value!r}")
        n_states = len(self._nodes[sensor_name]["states"])
        if not 0 <= value < n_states:
            raise ValueError(f"evidence for '{sensor_name}' = {value} out of range [0, {n_states})")

        if self.marginalize_dropout_state:
            dropout_idx = self._dropout_state_index.get(sensor_name)
            if dropout_idx is not None and value == dropout_idx:
                # DROPOUT === unobserved under this contract (see module
                # docstring): return None so this reuses the exact same
                # "marginalize out" code path as a genuinely missing key,
                # rather than a separately-implemented equivalent branch.
                return None

        return value

    def _validate_evidence_keys(self, evidence: Dict[str, EvidenceValue]) -> None:
        unknown = set(evidence) - set(_SENSOR_NODES)
        if unknown:
            raise ValueError(f"unknown evidence key(s) {sorted(unknown)}; expected a subset of {_SENSOR_NODES}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def compute_posterior(self, evidence: Dict[str, EvidenceValue]) -> List[float]:
        """
        P(TrueObstacle=o, DecelCapability=d | evidence), as a flat 6-element
        row-major list (index = o * n_decel + d = o * 3 + d).

        evidence may supply any subset of {"LidarObs", "TofObs",
        "WheelSlipObs"}; a missing key, an explicit value of None, or (when
        marginalize_dropout_state, the default, is set) that sensor's
        DROPOUT value all mean "not observed" and marginalize that sensor
        out analytically (likelihood factor 1.0), rather than summing over
        its 3 states -- a CPT row already sums to 1.0, so the two are
        mathematically equivalent, but the former is O(1) per sensor
        instead of O(states).

        If the supplied evidence is contradictory/degenerate under this
        model (joint likelihood below the zero-likelihood guard), the
        model's true physical prior P(TrueObstacle, DecelCapability) is
        returned instead of dividing by (near) zero -- see
        _compute_posterior_with_diagnostics for the flag that reports when
        this happened; step() also surfaces it as "fallback_active".
        """
        posterior, _fallback_active = self._compute_posterior_with_diagnostics(evidence)
        return posterior

    def _compute_posterior_with_diagnostics(self, evidence: Dict[str, EvidenceValue]) -> Tuple[List[float], bool]:
        """compute_posterior()'s actual implementation, additionally
        reporting whether the zero-likelihood fallback was used -- kept
        separate so step() can force a safe action when it was, without
        compute_posterior()'s public return type needing to change."""
        self._validate_evidence_keys(evidence)
        lidar_val = self._extract_evidence_value(evidence, "LidarObs")
        tof_val = self._extract_evidence_value(evidence, "TofObs")
        wheel_val = self._extract_evidence_value(evidence, "WheelSlipObs")

        unnormalized = [0.0] * self._n_posterior

        for t in range(self._n_track):
            p_t = self._track_cpt[t]

            for o in range(self._n_obstacle):
                p_o_given_t = self._obstacle_cpt[t * self._track_to_obstacle_stride + o]

                if lidar_val is None:
                    f_lidar = 1.0
                else:
                    f_lidar = self._sensor_cpt["LidarObs"][o * self._obstacle_to_lidar_stride + lidar_val]
                if tof_val is None:
                    f_tof = 1.0
                else:
                    f_tof = self._sensor_cpt["TofObs"][o * self._obstacle_to_tof_stride + tof_val]
                obstacle_evidence_factor = f_lidar * f_tof

                for d in range(self._n_decel):
                    p_d_given_t = self._decel_cpt[t * self._track_to_decel_stride + d]

                    if wheel_val is None:
                        f_wheel = 1.0
                    else:
                        f_wheel = self._sensor_cpt["WheelSlipObs"][d * self._decel_to_wheel_stride + wheel_val]

                    unnormalized[o * self._n_decel + d] += (
                        p_t * p_o_given_t * p_d_given_t * obstacle_evidence_factor * f_wheel
                    )

        likelihood = math.fsum(unnormalized)
        if likelihood < _ZERO_LIKELIHOOD_GUARD:
            # Contradictory/degenerate evidence under this model: rather
            # than divide by (near) zero -- or return an arbitrary uniform
            # guess -- fall back to the model's true physical prior
            # P(TrueObstacle, DecelCapability), and flag that this
            # happened so callers (step()) can force a safe decision.
            return list(self._prior_marginal), True

        return [v / likelihood for v in unnormalized], False

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def evaluate_expected_loss(self, posterior: List[float]) -> Tuple[List[float], str, int]:
        """
        Expected loss of each action under `posterior`:
        E[loss | action] = sum_{o,d} posterior[o,d] * loss[action, o, d].

        Returns (expected_losses, best_action_name, best_action_index).
        Ties are broken deterministically toward safety: EMERGENCY_BRAKE is
        preferred over COAST, which is preferred over ACCEL, regardless of
        the actions list's declared order.

        `posterior` must be an exactly-normalized probability distribution:
        the right length, every entry finite and in [0.0, 1.0], summing to
        1.0 within 1e-12. Any violation raises ValueError -- this method
        does not attempt to silently renormalize or clip a malformed input,
        since that could mask an upstream bug.

        If the loss matrix itself somehow yields a non-finite expected loss
        for some action (never possible with the sealed model, whose loss
        values are validated finite at compile time -- this is defense in
        depth against a corrupted matrix at runtime), the safety tie-break
        alone is not trustworthy: comparisons against NaN are neither true
        nor false in a consistent way, so ordinary min()-based selection
        could silently pick an unsafe action. In that case EMERGENCY_BRAKE
        is selected outright, never ACCEL.
        """
        self._validate_posterior(posterior)

        n_actions = len(self._actions)
        stride = self._loss_action_stride
        expected_losses = []
        for a in range(n_actions):
            block = self._loss_values[a * stride : a * stride + stride]
            expected_losses.append(math.fsum(p * loss for p, loss in zip(posterior, block)))

        if not all(math.isfinite(v) for v in expected_losses):
            best_idx = self._emergency_brake_index
        else:
            best_idx = self._select_best_action_index(expected_losses)
        return expected_losses, self._actions[best_idx], best_idx

    def _validate_posterior(self, posterior: List[float]) -> None:
        if len(posterior) != self._n_posterior:
            raise ValueError(f"posterior must have {self._n_posterior} entries, got {len(posterior)}")
        for p in posterior:
            if not math.isfinite(p):
                raise ValueError(f"posterior entry {p!r} is not finite (NaN/Inf are not valid probabilities)")
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"posterior entry {p!r} is out of the valid probability range [0.0, 1.0]")
        total = math.fsum(posterior)
        if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"posterior does not sum to 1.0 (sum={total!r}); expected a normalized distribution")

    def _select_best_action_index(self, expected_losses: List[float]) -> int:
        return min(
            range(len(expected_losses)),
            key=lambda i: (
                expected_losses[i],
                _SAFETY_TIE_BREAK_PRIORITY.get(self._actions[i], len(self._actions)),
            ),
        )

    # ------------------------------------------------------------------
    # High-level pipeline
    # ------------------------------------------------------------------
    def step(self, evidence: Dict[str, EvidenceValue]) -> dict:
        """
        Run inference then decision selection in one call.

        If the supplied evidence was contradictory/degenerate (see
        compute_posterior), the returned posterior is the model's true
        prior rather than something actually informed by this evidence --
        trusting expected-loss minimization over it would be misplaced
        confidence, so the action is forced to EMERGENCY_BRAKE outright in
        that case, and "fallback_active" is set to True so callers can tell
        the difference from an ordinary, evidence-driven decision.
        """
        posterior, fallback_active = self._compute_posterior_with_diagnostics(evidence)

        if fallback_active:
            expected_losses, _ignored_action, _ignored_idx = self.evaluate_expected_loss(posterior)
            best_idx = self._emergency_brake_index
            best_action_name = self._actions[best_idx]
        else:
            expected_losses, best_action_name, best_idx = self.evaluate_expected_loss(posterior)

        return {
            "posterior": posterior,
            "posterior_hex": [v.hex() for v in posterior],
            "expected_losses": expected_losses,
            "expected_losses_hex": [v.hex() for v in expected_losses],
            "selected_action": best_action_name,
            "selected_action_index": best_idx,
            "fallback_active": fallback_active,
        }


def main() -> None:
    engine = BayesianInferenceEngine()

    scenarios = {
        "no evidence (prior)": {},
        "clear on both range sensors, low slip": {"LidarObs": 0, "TofObs": 0, "WheelSlipObs": 0},
        "obstacle detected on both range sensors": {"LidarObs": 1, "TofObs": 1, "WheelSlipObs": 0},
        "high wheel slip only": {"WheelSlipObs": 1},
        "partial dropout (TofObs unobserved)": {"LidarObs": 1, "TofObs": None, "WheelSlipObs": 0},
    }

    for label, evidence in scenarios.items():
        result = engine.step(evidence)
        print(f"{label}: evidence={evidence}")
        print(f"  posterior (o*3+d)   = {[round(p, 4) for p in result['posterior']]}")
        print(f"  expected losses     = {[round(v, 4) for v in result['expected_losses']]}")
        print(f"  selected action     = {result['selected_action']}")
        print()


if __name__ == "__main__":
    main()
