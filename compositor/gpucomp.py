"""ctypes wrapper for the sonicray GPU compositor (libsonicray_comp.dylib + sonicray_comp.metal).

    g = GPU(W, H)
    h = g.mmap("layer.f32")          # zero-copy GPU view of a layer file
    a = g.alloc3()                    # W*H*3 float32 working image (also readable/writable from numpy via g.view(a, 3))
    g.begin(); g.axpy(a, [(h, 0, (1, 1, 1))]); ...; g.end()
"""
import ctypes, math, os, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        path = os.path.join(HERE, "libsonicray_comp.dylib")
        if not os.path.exists(path):
            raise RuntimeError("build the library first:  compositor/build.sh")
        L = ctypes.CDLL(path)
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
        self.W, self.H = W, H
        if self.L.sc_init(os.path.join(HERE, "sonicray_comp.metal").encode()) != 0:
            raise RuntimeError(self.err())
        self.name = self.L.sc_device_name().decode()
        self._ptrs = {}
        # scratch for blurs (full-size, 3 channels) and a zero-weight dummy for unused kernel slots
        self.S1, self.S2 = self.alloc3(), self.alloc3()

    def err(self):
        return self.L.sc_last_error().decode()

    # ------------------------------------------------------------------ buffers
    def mmap(self, path):
        h = self.L.sc_mmap(path.encode())
        if h < 0:
            raise RuntimeError(self.err())
        return h

    def alloc_bytes(self, n):
        h = self.L.sc_alloc(int(n))
        if h < 0:
            raise RuntimeError(self.err())
        return h

    def alloc1(self):
        return self.alloc_bytes(self.W * self.H * 4)

    def alloc3(self):
        return self.alloc_bytes(self.W * self.H * 12)

    def ptr(self, h):
        if h not in self._ptrs:
            self._ptrs[h] = self.L.sc_ptr(h)
        return self._ptrs[h]

    def view(self, h, ch, dtype=np.float32, shape=None):
        """numpy view onto a shared buffer (zero copy)."""
        n = (self.W * self.H * ch) if shape is None else int(np.prod(shape))
        buf = (ctypes.c_char * (n * np.dtype(dtype).itemsize)).from_address(self.ptr(h))
        a = np.frombuffer(buf, dtype=dtype, count=n)
        return a.reshape(shape) if shape is not None else a.reshape(self.H, self.W, ch) if ch > 1 else a.reshape(self.H, self.W)

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
            raise RuntimeError(self.err())

    def end(self):
        if self.L.sc_end() != 0:
            raise RuntimeError(self.err())

    def op(self, kernel, bufs, params=b"", gw=None, gh=None):
        arr = (ctypes.c_int * len(bufs))(*bufs)
        r = self.L.sc_op(kernel.encode(), arr, len(bufs), params, len(params), gw or self.W, gh or self.H)
        if r != 0:
            raise RuntimeError(self.err())

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
        """gaussian blur matching Scene.gblur(): direct for sigma <= 6, else box-down / blur / bilinear-up."""
        W, H = self.W, self.H
        if sigma <= 6:
            r, k = gauss_taps(sigma)
            for d, (a, b) in enumerate(((src, self.S1), (self.S1, dst))):
                p = struct.pack("5i130f", W, H, ch, r, d, *(list(k) + [0.0] * (130 - len(k))))
                self.op("blur1d", [a, b], p)
            return
        k = int(sigma / 3)
        w2, h2 = max(W // k, 2), max(H // k, 2)
        self.op("box_down", [src, self.S1], struct.pack("5i", W, H, w2, h2, ch), w2, h2)
        r, kk = gauss_taps(sigma / k)
        for d, (a, b) in enumerate(((self.S1, self.S2), (self.S2, self.S1))):
            p = struct.pack("5i130f", w2, h2, ch, r, d, *(list(kk) + [0.0] * (130 - len(kk))))
            self.op("blur1d", [a, b], p, w2, h2)
        self.op("bilinear_up", [self.S1, dst], struct.pack("5i", w2, h2, W, H, ch))
