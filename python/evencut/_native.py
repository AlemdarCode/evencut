"""ctypes binding to the native shared library (evencut.dll / libevencut.so / libevencut.dylib)."""

from __future__ import annotations

import array
import ctypes
import os
from pathlib import Path
from typing import Optional, Sequence

from ._types import EvencutError

_SIZE_T_CODE = "Q" if ctypes.sizeof(ctypes.c_size_t) == 8 else "I"
NO_PARENT = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1

_OK, _INVALID, _BUFFER, _NOMEM, _INTERNAL = 0, 1, 2, 3, 4
_MIN_MAX, _SUM_SQUARES = 0, 1


class _Options(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("objective", ctypes.c_int32),
        ("forbidden_cuts", ctypes.POINTER(ctypes.c_size_t)),
        ("forbidden_count", ctypes.c_size_t),
        ("secondary", ctypes.POINTER(ctypes.c_uint64)),
        ("secondary_limit", ctypes.c_uint64),
        ("polish", ctypes.c_int32),
    ]


class _Summary(ctypes.Structure):
    _fields_ = [
        ("parts", ctypes.c_size_t),
        ("max_load", ctypes.c_uint64),
        ("total", ctypes.c_uint64),
        ("imbalance", ctypes.c_double),
        ("sum_of_squares", ctypes.c_char * 48),
    ]


_LIBRARY_NAMES = ("evencut.dll", "libevencut.dll", "libevencut.so", "libevencut.dylib")


def _candidates() -> list[Path]:
    override = os.environ.get("EVENCUT_LIBRARY")
    if override:
        return [Path(override)]
    here = Path(__file__).resolve().parent
    return [here / name for name in _LIBRARY_NAMES]


def load() -> Optional[ctypes.CDLL]:
    """Loads the shared library next to this package (or from $EVENCUT_LIBRARY)."""
    for path in _candidates():
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            _declare(lib)
            if lib.evencut_abi_version() != 1:
                raise ImportError(f"{path}: unsupported evencut ABI version {lib.evencut_abi_version()}")
            return lib
    return None


def _declare(lib: ctypes.CDLL) -> None:
    sz = ctypes.c_size_t
    u64p = ctypes.POINTER(ctypes.c_uint64)
    szp = ctypes.POINTER(ctypes.c_size_t)
    lib.evencut_options_init.argtypes = [ctypes.POINTER(_Options)]
    lib.evencut_options_init.restype = None
    lib.evencut_partition.argtypes = [u64p, sz, sz, ctypes.POINTER(_Options), szp, sz, ctypes.POINTER(_Summary)]
    lib.evencut_partition.restype = ctypes.c_int
    lib.evencut_evaluate.argtypes = [u64p, sz, szp, sz, ctypes.POINTER(_Summary)]
    lib.evencut_evaluate.restype = ctypes.c_int
    lib.evencut_partition_tree.argtypes = [szp, u64p, sz, sz, szp, sz, ctypes.POINTER(_Summary)]
    lib.evencut_partition_tree.restype = ctypes.c_int
    lib.evencut_last_error.argtypes = []
    lib.evencut_last_error.restype = ctypes.c_char_p
    lib.evencut_version.argtypes = []
    lib.evencut_version.restype = ctypes.c_char_p
    lib.evencut_abi_version.argtypes = []
    lib.evencut_abi_version.restype = ctypes.c_int


def _u64_buffer(values: Sequence[int]):
    arr = array.array("Q", values)
    return arr, (ctypes.c_uint64 * len(arr)).from_buffer(arr) if len(arr) else None


def _size_buffer(values: Sequence[int]):
    arr = array.array(_SIZE_T_CODE, values)
    return arr, (ctypes.c_size_t * len(arr)).from_buffer(arr) if len(arr) else None


def _check(lib: ctypes.CDLL, status: int) -> None:
    if status == _OK:
        return
    message = (lib.evencut_last_error() or b"").decode("utf-8", "replace")
    if status == _INVALID:
        raise EvencutError(message)
    if status == _NOMEM:
        raise MemoryError(message)
    raise RuntimeError(message or f"evencut: native call failed with status {status}")


def partition(lib, weights: list[int], k: int, objective: str, forbidden: list[int],
              secondary: Optional[list[int]], secondary_limit: int, polish: bool) -> list[int]:
    w_arr, w_ptr = _u64_buffer(weights)
    opt = _Options()
    lib.evencut_options_init(ctypes.byref(opt))
    opt.objective = _SUM_SQUARES if objective == "sum_squares" else _MIN_MAX
    opt.polish = 1 if polish else 0
    f_arr, f_ptr = _size_buffer(forbidden)
    if forbidden:
        opt.forbidden_cuts = f_ptr
        opt.forbidden_count = len(forbidden)
    s_arr = None
    if secondary is not None:
        s_arr, s_ptr = _u64_buffer(secondary)
        opt.secondary = s_ptr
        opt.secondary_limit = secondary_limit
    capacity = max(1, min(k, len(weights)))
    ends = (ctypes.c_size_t * capacity)()
    summary = _Summary()
    _check(lib, lib.evencut_partition(w_ptr, len(weights), k, ctypes.byref(opt), ends, capacity, ctypes.byref(summary)))
    del w_arr, f_arr, s_arr  # buffers had to outlive the call
    return list(ends[: summary.parts])


def partition_tree(lib, parents: list[int], weights: list[int], k: int) -> list[int]:
    p_arr, p_ptr = _size_buffer(parents)
    w_arr, w_ptr = _u64_buffer(weights)
    out = (ctypes.c_size_t * max(1, len(weights)))()
    summary = _Summary()
    _check(lib, lib.evencut_partition_tree(p_ptr, w_ptr, len(weights), k, out, len(out), ctypes.byref(summary)))
    del p_arr, w_arr
    return list(out[: len(weights)])

