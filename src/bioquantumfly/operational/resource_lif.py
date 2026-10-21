"""Isolated compiled synthetic LIF model; not the earlier Brian2 protocol."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np

_LIB = None


def kernel():
    global _LIB
    if _LIB is None:
        src = Path(__file__).with_suffix(".cpp")
        fingerprint = hashlib.sha256(src.read_bytes()).hexdigest()[:20]
        directory = Path(tempfile.gettempdir()) / "bioquantumfly-resource-lif" / fingerprint
        directory.mkdir(parents=True, exist_ok=True)
        library = directory / "lif.so"
        if not library.exists():
            subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                            str(src), "-o", str(library)], check=True, capture_output=True)
        _LIB = ctypes.CDLL(str(library))
        ptr = np.ctypeslib.ndpointer
        _LIB.resource_lif.argtypes = [
            ctypes.c_int, ctypes.c_int, ptr(np.float64, flags="C_CONTIGUOUS"),
            ptr(np.float64, flags="C_CONTIGUOUS"), ctypes.c_double,
            ptr(np.int32, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_uint64, ctypes.c_double, ctypes.c_double,
            ptr(np.uint8, flags="C_CONTIGUOUS"), ptr(np.uint32, flags="C_CONTIGUOUS"),
            ptr(np.uint32, flags="C_CONTIGUOUS"), ptr(np.float64, flags="C_CONTIGUOUS")]
        _LIB.resource_lif.restype = None
    return _LIB.resource_lif


def sample(neurons, preparation, gates, gain, word, query, shots, *, seed,
           dt_ms=.1, jitter=.15, background_hz=20.):
    """Independent trials, one specified monitor, no state overwrite at gates."""
    prep = np.ascontiguousarray(preparation, dtype=np.float64)
    gates = np.ascontiguousarray(gates, dtype=np.float64)
    if (neurons not in (2, 8, 16) or type(shots) is not int or not 1 <= shots <= 200000
            or prep.shape != (2,) or gates.shape != (2, 2, 2)
            or not np.all(np.isfinite(prep)) or not np.all(np.isfinite(gates))
            or np.any(prep < 0) or np.any(prep > 3) or np.any(np.abs(gates) > 3)
            or not 0 <= gain <= 8 or query not in (0, 1, 2)
            or not isinstance(word, str) or len(word) > 4 or set(word)-set("FWI")
            or dt_ms not in (.1, .05, .025)
            or not 0 <= jitter <= .5 or not 0 <= background_hz <= 100
            or type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError("outside registered synthetic LIF limits")
    codes = np.ascontiguousarray([{"F": 0, "W": 1, "I": -1}[g] for g in word], dtype=np.int32)
    out = np.empty(shots, dtype=np.uint8)
    activity = np.empty(shots, dtype=np.uint32)
    noise = np.empty(shots, dtype=np.uint32)
    cost = np.empty(shots)
    run = kernel()
    start = time.perf_counter()
    run(neurons, shots, prep, gates, gain, codes, len(word), query, dt_ms,
        seed, jitter, background_hz, out, activity, noise, cost)
    return out, {
        "trials": shots, "neurons": neurons, "duration_ms": 19+10*len(word),
        "spikes": int(activity.sum()), "noise_events": int(noise.sum()),
        "absolute_drive_neuron_ms": float(cost.sum()),
        "readout_bits_per_trial": 2 if query == 2 else 1,
        "simulator_seconds": time.perf_counter()-start,
        "activity": activity, "noise": noise, "drive": cost}
