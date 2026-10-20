# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
"""Low-level Metal access for the sonicray compositor: builds the native library on first use, then wraps its C ABI with ctypes.

    g = GPU(W, H)
    h = g.mmap("layer.f32")           # zero-copy GPU view of a layer file
    a = g.alloc3()                     # W*H*3 float32 working image (numpy can read/write it through g.view(a, 3))
    g.begin(); g.axpy(a, [(h, 0, (1, 1, 1))], init=True); g.end()

Nothing here knows about films; see compositor.py for the frame pipeline.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.join(HERE, "native")
_LIB = None


class SonicrayError(RuntimeError):
    """Raised with a message that says what to do next (these are read by humans *and* by AI assistants)."""


def cache_dir():
    return os.environ.get("SONICRAY_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "sonicray"))


def build_native(force=False):
    """Compile native/sonicray_comp.mm into a dylib under ~/.cache/sonicray (needs only the Xcode Command Line Tools)."""
    src = os.path.join(NATIVE, "sonicray_comp.mm")
    digest = hashlib.sha1(open(src, "rb").read()).hexdigest()[:12]
    out_dir = os.path.join(cache_dir(), "lib-" + digest)
    out = os.path.join(out_dir, "libsonicray_comp.dylib")
    if os.path.exists(out) and not force:
        return out
    if sys.platform != "darwin":
        raise SonicrayError("sonicray's GPU compositor needs macOS on Apple Silicon (Metal).")
    cxx = shutil.which("clang++")
    if not cxx:
        raise SonicrayError("clang++ not found. Install the Xcode Command Line Tools:  xcode-select --install")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [cxx, "-O2", "-std=c++17", "-fobjc-arc", "-ObjC++", "-dynamiclib", "-fPIC", "-framework", "Metal", "-framework", "Foundation",
           "-o", out, src]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SonicrayError("building the Metal bridge failed:\n" + r.stderr[-2000:])
    return out


def _lib():
    global _LIB
    if _LIB is None:
        L = ctypes.CDLL(build_native())
        L.sc_last_error.restype = ctypes.c_char_p
        L.sc_device_name.restype = ctypes.c_char_p
        L.sc_ptr.restype = ctypes.c_void_p
        L.sc_alloc.argtypes = [ctypes.c_size_t]
        L.sc_mmap.argtypes = [ctypes.c_char_p]
        L.sc_op.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _LIB = L
    return _LIB


def gauss_taps(sigma):
    """scipy.ndimage.gaussian_filter1d weights (truncate = 4)."""
    r = int(4.0 * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * x * x / (sigma * sigma))
    return r, (k / k.sum()).astype(np.float32)


class GPU:
    def __init__(self, W, H):
        self.L = _lib()
        self.W, self.H = int(W), int(H)
        if self.L.sc_init(os.path.join(NATIVE, "sonicray_comp.metal").encode()) != 0:
            raise SonicrayError(self.err())
        self.name = self.L.sc_device_name().decode()
        self._ptrs = {}
        self.profile = None                                    # set to a dict to collect seconds per kernel (each op then syncs)
        self.S1, self.S2 = self.alloc3(), self.alloc3()          # blur scratch (full size, 3 channels)

    def err(self):
        return self.L.sc_last_error().decode()

    # ------------------------------------------------------------------ buffers
    def mmap(self, path):
        h = self.L.sc_mmap(path.encode())
        if h < 0:
            raise SonicrayError(self.err())
        return h

    def alloc_bytes(self, n):
        h = self.L.sc_alloc(int(n))
        if h < 0:
            raise SonicrayError(self.err())
        return h

    def alloc1(self):
        return self.alloc_bytes(self.W * self.H * 4)

    def alloc3(self):
        return self.alloc_bytes(self.W * self.H * 12)

    def ptr(self, h):
        if h not in self._ptrs:
            self._ptrs[h] = self.L.sc_ptr(h)
        return self._ptrs[h]

    def view(self, h, ch, dtype=np.float32):
        """numpy view onto a shared buffer (zero copy)."""
        n = self.W * self.H * ch
        buf = (ctypes.c_char * (n * np.dtype(dtype).itemsize)).from_address(self.ptr(h))
        a = np.frombuffer(buf, dtype=dtype, count=n)
        return a.reshape(self.H, self.W, ch) if ch > 1 else a.reshape(self.H, self.W)

    def write(self, h, arr):
        arr = np.ascontiguousarray(arr)
        ctypes.memmove(self.ptr(h), arr.ctypes.data, arr.nbytes)

    def zero(self, h, nbytes):
        ctypes.memset(self.ptr(h), 0, nbytes)

    def read_bytes(self, h, nbytes):
        return ctypes.string_at(self.ptr(h), nbytes)

    # ------------------------------------------------------------------ command stream
    def begin(self):
        if self.L.sc_begin() != 0:
            raise SonicrayError(self.err())

    def end(self):
        if self.L.sc_end() != 0:
            raise SonicrayError(self.err())

    def op(self, kernel, bufs, params=b"", gw=None, gh=None):
        arr = (ctypes.c_int * len(bufs))(*bufs)
        if self.L.sc_op(kernel.encode(), arr, len(bufs), params, len(params), gw or self.W, gh or self.H) != 0:
            raise SonicrayError(self.err())
        if self.profile is not None:                           # profiling: run this op alone and time it
            t0 = time.time()
            self.end(); self.begin()
            self.profile[kernel] = self.profile.get(kernel, 0.0) + time.time() - t0

    # ------------------------------------------------------------------ kernels
    def axpy(self, out, terms, init):
        """out (+)= sum w * layer.  terms: [(handle, mode, (wr, wg, wb)), ...]; mode 0 = rgb, 1 = red broadcast, 2 = green broadcast."""
        terms = list(terms)
        for s in range(0, max(len(terms), 1), 8):
            chunk = terms[s:s + 8]
            hs = [t[0] for t in chunk] + [out] * (8 - len(chunk))
            modes = [t[1] for t in chunk] + [-1] * (8 - len(chunk))
            w = []
            for t in chunk:
                w += [float(x) for x in t[2]]
            w += [0.0] * (24 - len(w))
            p = struct.pack("3i8i24f", self.W, self.H, 1 if (init and s == 0) else 0, *modes, *w)
            self.op("axpy8", [out] + hs, p)

    def blur(self, src, dst, sigma, ch):
        """Gaussian blur matching the reference compositor: direct for sigma <= 6, else box-down / blur / bilinear-up."""
        W, H = self.W, self.H
        if sigma <= 6:
            r, k = gauss_taps(sigma)
            for d, (a, b) in enumerate(((src, self.S1), (self.S1, dst))):
                self.op("blur1d", [a, b], struct.pack("5i130f", W, H, ch, r, d, *(list(k) + [0.0] * (130 - len(k)))))
            return
        k = int(sigma / 3)
        w2, h2 = max(W // k, 2), max(H // k, 2)
        self.op("box_down", [src, self.S1], struct.pack("5i", W, H, w2, h2, ch), w2, h2)
        r, kk = gauss_taps(sigma / k)
        for d, (a, b) in enumerate(((self.S1, self.S2), (self.S2, self.S1))):
            self.op("blur1d", [a, b], struct.pack("5i130f", w2, h2, ch, r, d, *(list(kk) + [0.0] * (130 - len(kk)))), w2, h2)
        self.op("bilinear_up", [self.S1, dst], struct.pack("5i", w2, h2, W, H, ch))
