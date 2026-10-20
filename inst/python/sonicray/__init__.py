"""sonicray: GPU (Metal) compositing for light-layer films on Apple Silicon.  See AGENTS.md in the repository root first."""
import os as _os

# numpy's BLAS spins up one thread per core even for 3x3 products; with several worker processes that melts the machine
# (measured: 40x more CPU burn, half the speed).  Must be set before numpy is first imported.
for _k in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

__version__ = "0.2.0"

from .framespec import (Beacon, Burst, FrameSpec, Orb, SatScan, SkyDay, Term,  # noqa: F401
                        PLANE_ALK, PLANE_CHLA, PLANE_SHIMMER, PLANE_TOC, PLANE_WHITE)
