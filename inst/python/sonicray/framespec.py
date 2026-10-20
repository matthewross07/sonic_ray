"""FrameSpec: a plain-data description of ONE frame. A *story* (e.g. films/poudre.py) turns a time t into a FrameSpec; the GPU
compositor (compositor.py) turns a FrameSpec into pixels. Keeping the two apart is deliberate: all creative decisions (what lights up
when, colours, seasons, sun path) live in the story; all pixel work lives on the GPU. To make a new film, write a new story -- do not
edit the compositor.

Layer names refer to files `<layers_dir>/<name>.f32` (float32 RGB, row-major, W*H*3).  Conventional names the compositor knows:
    amb        ambient (sky-fill) light on the terrain           required
    mask       1 where terrain is, 0 where sky is (1 channel)    required
    riv        river mask: R = highlighted river, G = all water  required (may be all zeros)
    stars      starfield (optional)       uv_u, uv_v  per-pixel map coordinates for satellite scans (optional)
Everything else (sun_0.., sky_0.., fwk_*, bea_*, ...) is referenced by name from the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

RGB = Tuple[float, float, float]

# Plane / colour order used by spark primitives.
PLANE_TOC, PLANE_CHLA, PLANE_ALK, PLANE_SHIMMER, PLANE_WHITE = 0, 1, 2, 3, 4
PLANE_OF = {"toc": PLANE_TOC, "chla": PLANE_CHLA, "alk": PLANE_ALK}


@dataclass
class Term:
    """out += layer * weight.  mode 0 = use the layer's RGB, 1 = broadcast its red channel, 2 = broadcast its green channel."""
    layer: str
    weight: RGB
    mode: int = 0


@dataclass
class SatScan:
    """Landsat-style scan lines sweeping across the scene using the uv_u / uv_v layers."""
    t: float
    events: List[Tuple[float, float]]            # (start time, u-centre of the swath)
    second_layer: Optional[str] = None           # layer (e.g. 'sun_1') mixed into the scan lighting; amb is always used
    half_width: float = 0.18
    duration: float = 4.0
    tau: float = 1.76
    color: RGB = (0.84, 0.93, 1.00)


@dataclass
class Burst:
    """One firework: a sphere of curved ballistic sparks, or (dt < 0) the rising rocket before it."""
    uid: int                                     # seeds the random sparks -> the same uid always looks the same
    plane: int                                   # PLANE_TOC / PLANE_CHLA / PLANE_ALK
    t0: float                                    # burst time (s)
    origin: Tuple[float, float, float]           # world x, y, z of the burst centre
    launch_y: float                              # world y the rocket rises from
    n_sparks: int
    gain: float                                  # G   (brightness)
    v_max: float                                 # outward speed
    rise: float                                  # rocket rise duration (s)
    life: Tuple[float, float]                    # spark lifetime range (before the SLOW factor)
    trail_dt: float                              # tail length in seconds
    cell: bool = False                           # True for the small WQP-cell blips (fewer, simpler)


@dataclass
class SkyDay:
    """Daytime sky: two path-traced sky layers blended by f, normalised, tinted horizon->zenith and mapped."""
    layer_a: str
    layer_b: str
    f: float
    sky_exp: float
    zenith: RGB
    horizon_tint: RGB
    dawn_band: float = 0.0
    set_band: float = 0.0
    venus: RGB = (0.92, 0.46, 0.30)
    sunset: RGB = (1.00, 0.42, 0.16)


@dataclass
class Orb:
    """Analytic sun: mode 1 = overhead noon corona, mode 2 = descending disc with Mie aureole."""
    mode: int
    x: float
    y: float
    radius: float = 0.0
    visibility: float = 0.0
    dusk: float = 0.0
    amount: float = 0.0
    color: RGB = (1.0, 0.94, 0.80)


@dataclass
class Beacon:
    """A lit sensor: glowing core on the surface plus an upward plume (screen-space)."""
    x: float
    y: float
    beam: float
    pulse: float


@dataclass
class FrameSpec:
    t: float = 0.0
    terms: List[Term] = field(default_factory=list)              # summed into the terrain light
    beacon_terms: List[Term] = field(default_factory=list)       # 'lighthouse' layers: tinted blue, also drive the river halo
    sat: Optional[SatScan] = None
    bursts: Sequence[Burst] = ()
    sky: Optional[SkyDay] = None                                 # None -> night (stars only)
    star_amount: float = 0.0                                     # 0 disables stars
    orb: Optional[Orb] = None
    aerial: float = 0.0                                          # atmospheric-perspective strength (needs sky for the horizon colour)
    beacons: List[Beacon] = field(default_factory=list)
    fade: float = 1.0
    seed: int = 0                                                # dither seed
