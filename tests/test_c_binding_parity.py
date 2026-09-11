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
from tools.hiram_c_binding import HIRAM_UNOBSERVED, HiramCKernel

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


if __name__ == "__main__":
    unittest.main()
