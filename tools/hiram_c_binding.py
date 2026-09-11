"""
tools.hiram_c_binding - Python ctypes bridge to the compiled C kernel
(Task 3.2), src/hiram_kernel.c / include/hiram_kernel.h.

Loads build/libhiram_kernel.{dll,so} (built by the root Makefile's
$(KERNEL_SHARED) target, `-fPIC -shared` off the same source and flags as
the static library and C test binary -- see Makefile), maintains a
persistent hiram_workspace_t, and exposes step_c_kernel() as a thin,
allocation-free-on-the-C-side wrapper returning a dict shaped exactly like
tools.inference_engine.BayesianInferenceEngine.step()'s return value, so the
two can be compared field-for-field (see tests/test_c_binding_parity.py).

The three ctypes.Structure classes below are laid out field-for-field, in
declaration order, to match the C99 structs in include/hiram_kernel.h
exactly -- do not reorder or add fields without updating both sides.
"""

import ctypes
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"

_SHARED_LIB_NAME = "libhiram_kernel.dll" if sys.platform == "win32" else "libhiram_kernel.so"
SHARED_LIB_PATH = BUILD_DIR / _SHARED_LIB_NAME

# Mirrors include/hiram_tables.h's HIRAM_ACTION_NAMES / HIRAM_ACTION_* --
# both sealed to this cart model's loss_matrix.actions order (verified
# against data/models/causal_dag.json: ["ACCEL", "COAST", "EMERGENCY_BRAKE"]).
HIRAM_ACTION_NAMES = ("ACCEL", "COAST", "EMERGENCY_BRAKE")

HIRAM_UNOBSERVED = -1

HIRAM_STATUS_NAMES = {
    0: "HIRAM_OK",
    1: "HIRAM_ERR_CUSTODY_MISMATCH",
    2: "HIRAM_ERR_INVALID_EVIDENCE",
    3: "HIRAM_ERR_NULL_POINTER",
}

HIRAM_ERR_INVALID_EVIDENCE = 2

# The only values hiram_evidence_t's int32 fields are ever valid for --
# mirrors hiram_evidence_value_in_range() in src/hiram_kernel.c exactly.
# ctypes.c_int32 silently *wraps* an out-of-range Python int at struct
# assignment time (e.g. 2**32 -> 0, -(2**32) -> 0, 2**32 + 1 -> 1) rather
# than raising, so that narrowing must never happen before this set has
# been checked in full Python-int precision -- see
# HiramCKernel._validate_sensor_value / .step().
_VALID_SENSOR_VALUES = frozenset({HIRAM_UNOBSERVED, 0, 1, 2})


def _validate_sensor_value(value: object, field_name: str) -> int:
    """Validates one hiram_evidence_t field value in full Python-int
    precision, before it is ever narrowed into a ctypes.c_int32. Raises
    ValueError for anything outside {-1, 0, 1, 2} -- including values a
    ctypes.c_int32 assignment would silently wrap into range (e.g.
    2**32 == 0, -(2**32) == 0), non-integers (float, str, None), and bool
    (which is technically an int subclass in Python but not a meaningful
    sensor reading).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int in {{-1, 0, 1, 2}} (-1 = HIRAM_UNOBSERVED), "
            f"got {value!r} ({type(value).__name__})"
        )
    if value not in _VALID_SENSOR_VALUES:
        raise ValueError(
            f"{field_name}={value!r} is out of range; must be one of {{-1, 0, 1, 2}} "
            "(-1 = HIRAM_UNOBSERVED, 2 = DROPOUT) -- refusing to narrow this into a "
            "ctypes.c_int32, which would silently wrap it (e.g. 2**32 -> 0) instead "
            "of the C kernel's own HIRAM_ERR_INVALID_EVIDENCE rejection"
        )
    return value


class HiramEvidence(ctypes.Structure):
    """Mirrors hiram_evidence_t (include/hiram_kernel.h)."""

    _fields_ = [
        ("lidar_obs", ctypes.c_int32),
        ("tof_obs", ctypes.c_int32),
        ("wheel_slip_obs", ctypes.c_int32),
    ]


class HiramDecision(ctypes.Structure):
    """Mirrors hiram_decision_t (include/hiram_kernel.h)."""

    _fields_ = [
        ("posterior", ctypes.c_double * 6),
        ("expected_losses", ctypes.c_double * 3),
        ("selected_action", ctypes.c_int32),
        ("selected_action_index", ctypes.c_uint32),
        ("fallback_active", ctypes.c_bool),
        ("status", ctypes.c_int32),
    ]


class HiramWorkspace(ctypes.Structure):
    """Mirrors hiram_workspace_t (include/hiram_kernel.h)."""

    _fields_ = [
        ("posterior_scratch", ctypes.c_double * 6),
    ]


def _build_shared_lib() -> None:
    """Builds build/libhiram_kernel.{dll,so} via the root Makefile's
    dedicated target, rather than re-deriving gcc flags here -- so this
    binding can never drift out of step with the flags (-std=c99 -Wall
    -Wextra -Werror -Wvla -pedantic -O2) the rest of the kernel is built
    and audited under."""
    target = str(SHARED_LIB_PATH.relative_to(REPO_ROOT)).replace(os.sep, "/")
    subprocess.run(["make", target], cwd=str(REPO_ROOT), check=True)


def _load_library() -> ctypes.CDLL:
    if not SHARED_LIB_PATH.exists():
        _build_shared_lib()

    if sys.platform == "win32":
        import shutil

        gcc_bin = shutil.which("gcc")
        if gcc_bin:
            try:
                os.add_dll_directory(os.path.dirname(os.path.abspath(gcc_bin)))
            except (OSError, AttributeError):
                pass
        try:
            os.add_dll_directory(str(SHARED_LIB_PATH.parent))
        except (OSError, AttributeError):
            pass
        lib = ctypes.CDLL(str(SHARED_LIB_PATH), winmode=0)
    else:
        lib = ctypes.CDLL(str(SHARED_LIB_PATH))

    lib.hiram_kernel_verify_custody.argtypes = []
    lib.hiram_kernel_verify_custody.restype = ctypes.c_int32

    lib.hiram_kernel_step.argtypes = [
        ctypes.POINTER(HiramEvidence),
        ctypes.POINTER(HiramWorkspace),
        ctypes.POINTER(HiramDecision),
    ]
    lib.hiram_kernel_step.restype = ctypes.c_int32
    return lib


class HiramCKernel:
    """Owns one loaded copy of the shared library plus the persistent,
    caller-owned scratch workspace hiram_kernel_step() writes through --
    mirrors the embedded contract (include/hiram_kernel.h: "caller-owned
    scratch memory ... so a caller can place it in a specific memory
    region") of allocating the workspace once and reusing it across calls,
    rather than a fresh ctypes struct per step."""

    def __init__(self) -> None:
        self._lib = _load_library()
        self._workspace = HiramWorkspace()
        custody = self._lib.hiram_kernel_verify_custody()
        if custody != 0:
            raise ValueError(
                f"hiram_kernel_verify_custody() failed with status {custody} "
                f"({HIRAM_STATUS_NAMES.get(custody, 'UNKNOWN')}) -- the compiled "
                "kernel's HIRAM_FROZEN_IR_DIGEST does not match the sealed baseline"
            )

    def step(self, lidar: int, tof: int, slip: int) -> dict:
        """Runs one hiram_kernel_step() cycle and returns a dict shaped
        exactly like BayesianInferenceEngine.step()'s return value:
        posterior / posterior_hex / expected_losses / expected_losses_hex /
        selected_action / selected_action_index / fallback_active -- plus a
        "status" field (the raw hiram_status_t) that the Python reference
        has no equivalent for.

        Raises ValueError if `lidar`/`tof`/`slip` is not an int in
        {-1, 0, 1, 2} -- checked in full Python-int precision *before* any
        ctypes.c_int32 field assignment, so an out-of-i32-range or
        out-of-domain value is rejected outright rather than silently
        wrapping (see _validate_sensor_value)."""
        lidar = _validate_sensor_value(lidar, "lidar")
        tof = _validate_sensor_value(tof, "tof")
        slip = _validate_sensor_value(slip, "slip")

        evidence = HiramEvidence(lidar_obs=lidar, tof_obs=tof, wheel_slip_obs=slip)
        decision = HiramDecision()

        status = self._lib.hiram_kernel_step(
            ctypes.byref(evidence), ctypes.byref(self._workspace), ctypes.byref(decision)
        )

        posterior = list(decision.posterior)
        expected_losses = list(decision.expected_losses)

        return {
            "posterior": posterior,
            "posterior_hex": [v.hex() for v in posterior],
            "expected_losses": expected_losses,
            "expected_losses_hex": [v.hex() for v in expected_losses],
            "selected_action": HIRAM_ACTION_NAMES[decision.selected_action],
            "selected_action_index": decision.selected_action_index,
            "fallback_active": bool(decision.fallback_active),
            "status": status,
        }


_default_kernel: "HiramCKernel | None" = None


def _get_default_kernel() -> "HiramCKernel":
    global _default_kernel
    if _default_kernel is None:
        _default_kernel = HiramCKernel()
    return _default_kernel


def step_c_kernel(lidar: int, tof: int, slip: int) -> dict:
    """Module-level convenience wrapper around a lazily-created, persistent
    HiramCKernel (and therefore a persistent hiram_workspace_t) -- see
    HiramCKernel.step() for the return contract. `lidar`/`tof`/`slip` are
    each in {HIRAM_UNOBSERVED (-1), 0, 1, 2} (2 == DROPOUT), matching
    hiram_evidence_t's fields directly. Raises ValueError for anything
    outside that domain (including a value only invalid because a
    ctypes.c_int32 would silently wrap it, e.g. 2**32) -- see
    HiramCKernel.step() / _validate_sensor_value.
    """
    return _get_default_kernel().step(lidar, tof, slip)


def main() -> None:
    kernel = HiramCKernel()
    scenarios = {
        "no evidence (prior)": (HIRAM_UNOBSERVED, HIRAM_UNOBSERVED, HIRAM_UNOBSERVED),
        "clear on both range sensors, low slip": (0, 0, 0),
        "obstacle detected on both range sensors": (1, 1, 0),
        "high wheel slip only": (HIRAM_UNOBSERVED, HIRAM_UNOBSERVED, 1),
        "partial dropout (TofObs unobserved)": (1, 2, 0),
    }
    for label, (lidar, tof, slip) in scenarios.items():
        result = kernel.step(lidar, tof, slip)
        print(f"{label}: evidence=(lidar={lidar}, tof={tof}, slip={slip})")
        print(f"  posterior (o*3+d)   = {[round(p, 4) for p in result['posterior']]}")
        print(f"  expected losses     = {[round(v, 4) for v in result['expected_losses']]}")
        print(f"  selected action     = {result['selected_action']}")
        print()


if __name__ == "__main__":
    main()
