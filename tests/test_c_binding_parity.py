"""
Task 3.2 differential parity test: the compiled C kernel
(tools.hiram_c_binding, src/hiram_kernel.c) vs. the Python reference engine
(tools.inference_engine.BayesianInferenceEngine, Task 2.2) must agree
exactly on the selected action, and match to within 1e-12 on the posterior,
across every tri-state sensor evidence combination.

Frozen acceptance criteria (.clauderules.txt rule 1): this file is not to be
loosened to make an implementation pass -- a failure here means the C
kernel's arithmetic or decision logic has drifted from the sealed Python
reference, not that the test's tolerance is wrong.
"""

import itertools
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.inference_engine import BayesianInferenceEngine
from tools.hiram_c_binding import HIRAM_UNOBSERVED, HiramCKernel, step_c_kernel

# Every combination of {unobserved, state 0, state 1, DROPOUT} for each of
# the three tri-state sensors: 4^3 = 64 total evidence vectors.
_SENSOR_VALUES = (HIRAM_UNOBSERVED, 0, 1, 2)
_ALL_64_COMBINATIONS = list(itertools.product(_SENSOR_VALUES, repeat=3))

_POSTERIOR_TOLERANCE = 1e-12


def _c_value_to_python_evidence(value: int):
    """C's HIRAM_UNOBSERVED (-1) <-> Python's missing-key/None; every other
    value (0, 1, 2) is passed straight through, matching hiram_evidence_t's
    fields directly."""
    return None if value == HIRAM_UNOBSERVED else value


class TestCBindingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.py_engine = BayesianInferenceEngine()
        cls.c_kernel = HiramCKernel()

    def test_all_64_tri_state_combinations_agree(self) -> None:
        self.assertEqual(len(_ALL_64_COMBINATIONS), 64)

        mismatched_actions = []
        max_posterior_diff = 0.0

        for lidar, tof, slip in _ALL_64_COMBINATIONS:
            py_evidence = {
                "LidarObs": _c_value_to_python_evidence(lidar),
                "TofObs": _c_value_to_python_evidence(tof),
                "WheelSlipObs": _c_value_to_python_evidence(slip),
            }
            py_result = self.py_engine.step(py_evidence)
            c_result = self.c_kernel.step(lidar, tof, slip)

            case_label = f"(lidar={lidar}, tof={tof}, slip={slip})"

            if py_result["selected_action"] != c_result["selected_action"]:
                mismatched_actions.append(
                    f"{case_label}: py={py_result['selected_action']} c={c_result['selected_action']}"
                )

            self.assertEqual(
                py_result["fallback_active"],
                c_result["fallback_active"],
                f"fallback_active mismatch at {case_label}",
            )

            self.assertEqual(
                len(py_result["posterior"]),
                len(c_result["posterior"]),
                f"posterior length mismatch at {case_label}",
            )
            for py_p, c_p in zip(py_result["posterior"], c_result["posterior"]):
                diff = math.fabs(py_p - c_p)
                max_posterior_diff = max(max_posterior_diff, diff)

        self.assertEqual(
            mismatched_actions,
            [],
            f"{len(mismatched_actions)}/64 action disagreements:\n" + "\n".join(mismatched_actions),
        )
        self.assertLessEqual(
            max_posterior_diff,
            _POSTERIOR_TOLERANCE,
            f"max posterior abs difference {max_posterior_diff!r} exceeds tolerance {_POSTERIOR_TOLERANCE!r}",
        )

    def test_integer_overflow_underflow_rejection(self) -> None:
        """Astra audit finding (post-1891234): ctypes.c_int32 silently
        *wraps* an out-of-i32-range Python int at struct-field assignment
        time -- 2**32 -> 0, -(2**32) -> 0 -- so step_c_kernel(2**32, 0, 0)
        used to produce status=0 (HIRAM_OK) and ACCEL, exactly as if
        lidar=0 had been passed. tools.hiram_c_binding now validates every
        sensor argument in full Python-int precision *before* any ctypes
        narrowing (see _validate_sensor_value), so every case below must
        either raise ValueError outright, or -- if the wrapper's contract
        ever changes to a non-raising one -- come back as the same safe
        EMERGENCY_BRAKE-over-the-prior fallback the C kernel itself uses
        for HIRAM_ERR_INVALID_EVIDENCE (fallback_active=True, status=2).
        Never status=0, never ACCEL, for any of these.
        """
        bad_values = [
            2**32,  # 4294967296 -- wraps to 0 under ctypes.c_int32
            -(2**32),  # -4294967296 -- also wraps to 0
            3,  # in i32 range, but outside hiram_evidence_t's {-1,0,1,2} domain
            -2,  # ditto
        ]

        for bad_value in bad_values:
            for position in ("lidar", "tof", "slip"):
                kwargs = {"lidar": 0, "tof": 0, "slip": 0}
                kwargs[position] = bad_value
                case_label = f"step_c_kernel({kwargs['lidar']}, {kwargs['tof']}, {kwargs['slip']})"

                with self.subTest(bad_value=bad_value, position=position):
                    try:
                        result = step_c_kernel(**kwargs)
                    except ValueError:
                        continue  # rejecting the malformed call outright satisfies the contract

                    self.assertNotEqual(
                        result.get("status"), 0, f"{case_label} must not report HIRAM_OK (status=0)"
                    )
                    self.assertNotEqual(
                        result.get("selected_action"), "ACCEL", f"{case_label} must never select ACCEL"
                    )
                    self.assertTrue(
                        result.get("fallback_active"), f"{case_label} must set fallback_active=True"
                    )
                    self.assertEqual(
                        result.get("selected_action"),
                        "EMERGENCY_BRAKE",
                        f"{case_label} must fall back to EMERGENCY_BRAKE",
                    )

    def test_non_integer_and_none_rejected(self) -> None:
        """Non-int evidence (float, str, bool, None) must be rejected the
        same way as an out-of-domain int -- never silently truncated/coerced
        by ctypes into some in-range value."""
        for bad_value in (2.0, "1", None, True, False):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(ValueError):
                    step_c_kernel(bad_value, 0, 0)


if __name__ == "__main__":
    unittest.main()
