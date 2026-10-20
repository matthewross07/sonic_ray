// sonicray GPU compositor kernels (Metal Shading Language).
//
// Everything a light-layer film compositor does per pixel, on the GPU:
//   weighted layer sums -> satellite scan -> halo -> tone map -> sharpen -> bloom -> sky/terrain blend
//   -> firework overlay -> beacons -> dither -> uint8 RGB.
// All images are row-major interleaved float32 (1 or 3 channels) in shared (unified) memory, the same layout as the
// *.f32 light-layer files, so a layer file can be mmap'ed straight into a buffer with no copy.
//
// Parameter structs are plain sequences of 4-byte ints/floats (no float3 members) so the host can pack them with
// struct.pack; the field order here is the contract with compositor/gpucomp.py.
#include <metal_stdlib>
using namespace metal;

typedef packed_float3 pf3;

inline int reflect_idx(int i, int n) {            // scipy.ndimage 'reflect' (d c b a | a b c d)
    if (i < 0) i = -i - 1;
    if (i >= n) i = 2 * n - i - 1;
    return clamp(i, 0, n - 1);
}
inline uint pcg(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
inline float rnd01(uint v) { return float(pcg(v) >> 8) * (1.0f / 16777216.0f); }

// ----------------------------------------------------------------------------------------------- layer accumulation
// out (+)= sum_k layer_k * w_k  (per-channel weights; mode 0 = rgb, 1 = broadcast red, 2 = broadcast green, -1 = unused)
struct AxpyP { int W; int H; int init; int mode[8]; float w[24]; };
kernel void axpy8(device pf3* out [[buffer(0)]],
                  device const pf3* l0 [[buffer(1)]], device const pf3* l1 [[buffer(2)]],
                  device const pf3* l2 [[buffer(3)]], device const pf3* l3 [[buffer(4)]],
                  device const pf3* l4 [[buffer(5)]], device const pf3* l5 [[buffer(6)]],
                  device const pf3* l6 [[buffer(7)]], device const pf3* l7 [[buffer(8)]],
                  constant AxpyP& p [[buffer(9)]], uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= p.W || (int)g.y >= p.H) return;
    uint i = g.y * (uint)p.W + g.x;
    float3 acc = p.init != 0 ? float3(0.0f) : float3(out[i]);
    device const pf3* L[8] = {l0, l1, l2, l3, l4, l5, l6, l7};
    for (int k = 0; k < 8; ++k) {
        int m = p.mode[k];
        if (m < 0) continue;
        float3 v = float3(L[k][i]);
        if (m == 1) v = float3(v.x); else if (m == 2) v = float3(v.y);
        acc += v * float3(p.w[3 * k], p.w[3 * k + 1], p.w[3 * k + 2]);
    }
    out[i] = pf3(acc);
}

// ----------------------------------------------------------------------------------------------- satellite scan
struct SatP { int W; int H; int nev; int has_sun1; float t; float half_w; float dur; float tau; float ts[4]; float uc[4]; float col[3]; };
kernel void sat_add(device pf3* lin [[buffer(0)]], device const pf3* amb [[buffer(1)]], device const pf3* sun1 [[buffer(2)]],
                    device const float* uu [[buffer(3)]], device const float* vv [[buffer(4)]], device const pf3* riv [[buffer(5)]],
                    constant SatP& p [[buffer(6)]], uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= p.W || (int)g.y >= p.H) return;
    uint i = g.y * (uint)p.W + g.x;
    float u = uu[i], v = clamp(vv[i], 0.0f, 1.0f), lvl = 0.0f;
    for (int e = 0; e < p.nev; ++e) {
        float sw = clamp((p.half_w - fabs(u - p.uc[e])) / 0.014f, 0.0f, 1.0f);
        float dtp = p.t - (p.ts[e] + p.dur * v);
        float ramp = clamp(dtp / 0.22f, 0.0f, 1.0f); ramp = ramp * ramp * (3.0f - 2.0f * ramp);
        float glow = exp(-max(dtp, 0.0f) / p.tau) * ramp * 0.68f;
        float line = 2.15f * exp(-pow((dtp - 0.06f) / 0.11f, 2.0f));
        lvl = max(lvl, (glow + line) * sw);
    }
    float3 a = float3(amb[i]);
    float3 base = a * 0.42f + (p.has_sun1 != 0 ? float3(sun1[i]) * 0.62f : a * 0.42f);
    float3 c = float3(p.col[0], p.col[1], p.col[2]);
    float3 o = float3(lin[i]) + base * (lvl * c * 1.35f)
             + float(riv[i].x) * float3(0.04f, 0.28f, 0.62f) * clamp(lvl * 0.28f, 0.0f, 0.65f);
    lin[i] = pf3(o);
}

// ----------------------------------------------------------------------------------------------- halo around lit rivers
kernel void lum_mean(device const pf3* bl [[buffer(0)]], device float* lum [[buffer(1)]],
                     constant int2& d [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= d.x || (int)g.y >= d.y) return;
    uint i = g.y * (uint)d.x + g.x;
    float3 v = float3(bl[i]);
    lum[i] = (v.x + v.y + v.z) / 3.0f;
}
struct HaloP { int W; int H; float c1; float c2; float a; float b; float col[3]; };
kernel void halo_add(device pf3* lin [[buffer(0)]], device const float* g1 [[buffer(1)]], device const float* g2 [[buffer(2)]],
                     device const pf3* riv [[buffer(3)]], constant HaloP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= p.W || (int)g.y >= p.H) return;
    uint i = g.y * (uint)p.W + g.x;
    float halo = clamp(g1[i] * p.c1 + g2[i] * p.c2, 0.0f, 1.0f);
    float3 r = float3(riv[i]);
    lin[i] = pf3(float3(lin[i]) + (r.x * p.a + r.y * p.b) * halo * float3(p.col[0], p.col[1], p.col[2]));
}

// ----------------------------------------------------------------------------------------------- tone map (+bloom source)
kernel void tonemap(device const pf3* lin [[buffer(0)]], device const float* m [[buffer(1)]],
                    device pf3* terrain [[buffer(2)]], device pf3* bright [[buffer(3)]], device atomic_uint* bmax [[buffer(4)]],
                    constant int2& d [[buffer(5)]], uint2 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float local = 0.0f;
    if ((int)g.x < d.x && (int)g.y < d.y) {
        uint i = g.y * (uint)d.x + g.x;
        float3 l = max(float3(lin[i]), 0.0f);
        float mx = max(l.x, max(l.y, l.z));
        float mc = mx > 0.55f ? 0.55f + (mx - 0.55f) / (1.0f + 1.6f * (mx - 0.55f)) : mx;
        l = l * (mc / max(mx, 1e-6f));
        float3 mapped = clamp((l * (2.51f * l + 0.03f)) / (l * (2.43f * l + 0.59f) + 0.14f), 0.0f, 1.0f);
        terrain[i] = pf3(pow(mapped, float3(1.0f / 2.2f)));
        float3 b = max(mapped - 0.5f, 0.0f) * m[i];
        bright[i] = pf3(b);
        local = max(b.x, max(b.y, b.z));
    }
    float sm = simd_max(local);
    if (lane == 0 && sm > 0.0f) atomic_fetch_max_explicit(bmax, as_type<uint>(sm), memory_order_relaxed);
}
struct SharpP { int W; int H; float amount; };
kernel void sharpen_comb(device pf3* terrain [[buffer(0)]], device const pf3* blurred [[buffer(1)]], constant SharpP& p [[buffer(2)]],
                         uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= p.W || (int)g.y >= p.H) return;
    uint i = g.y * (uint)p.W + g.x;
    float3 t = float3(terrain[i]);
    terrain[i] = pf3(clamp(t + p.amount * (t - float3(blurred[i])), 0.0f, 1.0f));
}

// ----------------------------------------------------------------------------------------------- gaussian blur machinery
// Separable gaussian with scipy 'reflect' borders; large sigmas go through box-down / blur / bilinear-up like the CPU gblur().
struct BlurP { int w; int h; int ch; int r; int dir; float k[130]; };
kernel void blur1d(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant BlurP& p [[buffer(2)]],
                   uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= p.w || (int)g.y >= p.h) return;
    for (int c = 0; c < p.ch; ++c) {
        float acc = 0.0f;
        for (int j = -p.r; j <= p.r; ++j) {
            int xx = p.dir == 0 ? reflect_idx((int)g.x + j, p.w) : (int)g.x;
            int yy = p.dir == 0 ? (int)g.y : reflect_idx((int)g.y + j, p.h);
            acc += p.k[j + p.r] * src[((uint)yy * (uint)p.w + (uint)xx) * (uint)p.ch + (uint)c];
        }
        dst[(g.y * (uint)p.w + g.x) * (uint)p.ch + (uint)c] = acc;
    }
}
struct ResizeP { int sw; int sh; int dw; int dh; int ch; };
kernel void box_down(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant ResizeP& p [[buffer(2)]],
                     uint2 g [[thread_position_in_grid]]) {     // PIL Image.BOX: equal weights over the pixels whose centres fall in the window
    if ((int)g.x >= p.dw || (int)g.y >= p.dh) return;
    float sx = (float)p.sw / (float)p.dw, sy = (float)p.sh / (float)p.dh;
    float cx = ((float)g.x + 0.5f) * sx, cy = ((float)g.y + 0.5f) * sy;
    int x0 = max((int)(cx - 0.5f * sx + 0.5f), 0), x1 = min((int)(cx + 0.5f * sx + 0.5f), p.sw);
    int y0 = max((int)(cy - 0.5f * sy + 0.5f), 0), y1 = min((int)(cy + 0.5f * sy + 0.5f), p.sh);
    if (x1 <= x0) x1 = min(x0 + 1, p.sw);
    if (y1 <= y0) y1 = min(y0 + 1, p.sh);
    float inv = 1.0f / (float)((x1 - x0) * (y1 - y0));
    for (int c = 0; c < p.ch; ++c) {
        float acc = 0.0f;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) acc += src[((uint)y * (uint)p.sw + (uint)x) * (uint)p.ch + (uint)c];
        dst[(g.y * (uint)p.dw + g.x) * (uint)p.ch + (uint)c] = acc * inv;
    }
}
kernel void bilinear_up(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant ResizeP& p [[buffer(2)]],
                        uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= p.dw || (int)g.y >= p.dh) return;
    float fx = ((float)g.x + 0.5f) * (float)p.sw / (float)p.dw - 0.5f;
    float fy = ((float)g.y + 0.5f) * (float)p.sh / (float)p.dh - 0.5f;
    int x0 = (int)floor(fx), y0 = (int)floor(fy);
    float tx = fx - (float)x0, ty = fy - (float)y0;
    int xa = clamp(x0, 0, p.sw - 1), xb = clamp(x0 + 1, 0, p.sw - 1), ya = clamp(y0, 0, p.sh - 1), yb = clamp(y0 + 1, 0, p.sh - 1);
    for (int c = 0; c < p.ch; ++c) {
        float a = src[((uint)ya * (uint)p.sw + (uint)xa) * (uint)p.ch + (uint)c], b = src[((uint)ya * (uint)p.sw + (uint)xb) * (uint)p.ch + (uint)c];
        float d = src[((uint)yb * (uint)p.sw + (uint)xa) * (uint)p.ch + (uint)c], e = src[((uint)yb * (uint)p.sw + (uint)xb) * (uint)p.ch + (uint)c];
        dst[(g.y * (uint)p.dw + g.x) * (uint)p.ch + (uint)c] = mix(mix(a, b, tx), mix(d, e, tx), ty);
    }
}

// ----------------------------------------------------------------------------------------------- sky (starfield only; sun-era sky comes from the host)
struct SkyP { int W; int H; float t; float amt; };
kernel void sky_stars(device const pf3* stars [[buffer(0)]], device pf3* sky [[buffer(1)]], constant SkyP& p [[buffer(2)]],
                      uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= p.W || (int)g.y >= p.H) return;
    uint i = g.y * (uint)p.W + g.x;
    float tw = 0.85f + 0.15f * sin(p.t * 1.6f + (float)g.x * 0.17f + (float)g.y * 0.11f);
    float3 st = pow(clamp(float3(stars[i]) * 22.0f, 0.0f, 1.0f), float3(0.75f));
    sky[i] = pf3(clamp(st * tw * p.amt, 0.0f, 1.0f));
}

// ----------------------------------------------------------------------------------------------- firework overlay
// Five 8-bit planes drawn by the host at 2x resolution (toc, chla, alk, shimmer, white) -> 2x2 box average -> coloured, summed.
struct OvP { int W; int H; int active[5]; float col[15]; };
kernel void ov_from_planes(device const uchar* p0 [[buffer(0)]], device const uchar* p1 [[buffer(1)]], device const uchar* p2 [[buffer(2)]],
                           device const uchar* p3 [[buffer(3)]], device const uchar* p4 [[buffer(4)]],
                           device pf3* ov [[buffer(5)]], device atomic_uint* omax [[buffer(6)]], constant OvP& p [[buffer(7)]],
                           uint2 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float local = 0.0f;
    if ((int)g.x < p.W && (int)g.y < p.H) {
        device const uchar* P[5] = {p0, p1, p2, p3, p4};
        uint W2 = (uint)p.W * 2u;
        uint a = (g.y * 2u) * W2 + g.x * 2u, b = a + 1u, c = a + W2, d = c + 1u;
        float3 o = float3(0.0f);
        for (int k = 0; k < 5; ++k) {
            if (!p.active[k]) continue;
            float m = floor(((float)P[k][a] + (float)P[k][b] + (float)P[k][c] + (float)P[k][d]) * 0.25f + 0.5f) / 255.0f;
            o += m * float3(p.col[3 * k], p.col[3 * k + 1], p.col[3 * k + 2]);
        }
        o *= 0.92f;
        ov[g.y * (uint)p.W + g.x] = pf3(o);
        local = max(o.x, max(o.y, o.z));
    }
    float sm = simd_max(local);
    if (lane == 0 && sm > 0.0f) atomic_fetch_max_explicit(omax, as_type<uint>(sm), memory_order_relaxed);
}

// ----------------------------------------------------------------------------------------------- final frame
struct FinalP { int W; int H; int has_orb; int use_atm; int use_ov; int dither; uint seed; int nb;
                float atm; float hor[3]; float fade; float scale; float bcn[96]; };   // bcn: nb x (px, py, beam_h, pulse)
kernel void final_frame(device const pf3* terrain [[buffer(0)]], device const float* m [[buffer(1)]], device const pf3* sky [[buffer(2)]],
                        device const pf3* orb_core [[buffer(3)]], device const pf3* orb_glow [[buffer(4)]], device const float* aerial [[buffer(5)]],
                        device const pf3* b1 [[buffer(6)]], device const pf3* b2 [[buffer(7)]], device const pf3* b3 [[buffer(8)]],
                        device const pf3* ov [[buffer(9)]], device const pf3* ob1 [[buffer(10)]], device const pf3* ob2 [[buffer(11)]],
                        device const uint* bmax [[buffer(12)]], device const uint* omax [[buffer(13)]], device uchar* out [[buffer(14)]],
                        constant FinalP& p [[buffer(15)]], uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= p.W || (int)g.y >= p.H) return;
    uint i = g.y * (uint)p.W + g.x;
    float m3 = m[i];
    float3 t = float3(terrain[i]);
    if (p.use_atm != 0) {
        float aw = aerial[i] * p.atm;
        t = t * (1.0f - aw) + float3(p.hor[0], p.hor[1], p.hor[2]) * aw;
    }
    float3 img = float3(sky[i]) * (1.0f - m3) + t * m3;
    if (p.has_orb != 0) img += float3(orb_core[i]) * (1.0f - m3) + float3(orb_glow[i]) * (1.0f - 0.72f * m3);
    if (as_type<float>(bmax[0]) > 0.01f) img += (float3(b1[i]) * 0.56f + float3(b2[i]) * 0.72f + float3(b3[i]) * 0.54f) * 0.52f;
    if (p.use_ov != 0 && as_type<float>(omax[0]) > 0.0f) {
        float3 ot = float3(ov[i]) * 1.12f + float3(ob1[i]) * 0.26f + float3(ob2[i]) * 0.20f;
        float mo = max(ot.x, max(ot.y, ot.z));
        float3 os = ot * ((1.0f - exp(-1.12f * mo)) / max(mo, 1e-6f));
        img = 1.0f - (1.0f - clamp(img, 0.0f, 1.0f)) * (1.0f - clamp(os, 0.0f, 0.96f));
    }
    // sonde beacons: luminous core on the stream surface + soft upward plume
    float x = (float)g.x, y = (float)g.y, sc = p.scale;
    for (int k = 0; k < p.nb; ++k) {
        float px = p.bcn[4 * k], py = p.bcn[4 * k + 1], bh = p.bcn[4 * k + 2], pulse = p.bcn[4 * k + 3];
        float dx = x - px, dy = py - y;
        if (dy >= -2.5f * sc && dy <= bh * 1.65f && fabs(dx) <= 16.0f * sc) {
            float by = clamp(dy / bh, 0.0f, 1.65f);
            float sigma_x = (1.15f + 1.55f * by) * sc;
            float taper_y = clamp(1.0f - pow(by / 1.65f, 1.8f), 0.0f, 1.0f); taper_y *= taper_y;
            float taper_b = clamp((dy + 2.5f * sc) / (2.5f * sc), 0.0f, 1.0f);
            float wb = exp(-0.5f * (dx / sigma_x) * (dx / sigma_x)) * exp(-1.35f * by) * taper_y * taper_b * pulse;
            img += wb * float3(0.25f, 0.68f, 1.05f);
        }
        float rmax = 8.0f * sc;
        if (fabs(dx) <= rmax && fabs(dy) <= rmax) {
            float r2 = dx * dx + dy * dy;
            float taper_r = clamp(1.0f - r2 / (rmax * rmax), 0.0f, 1.0f); taper_r *= taper_r;
            float wd = exp(-0.5f * r2 / ((1.6f * sc) * (1.6f * sc))) * taper_r * pulse;
            img += wd * float3(0.65f, 0.90f, 1.15f);
        }
    }
    img = clamp(img, 0.0f, 1.0f) * p.fade;
    if (p.dither != 0) {
        for (int c = 0; c < 3; ++c) {
            uint k = (i * 3u + (uint)c) * 2u;
            img[c] += (rnd01(k ^ p.seed) - rnd01((k + 1u) ^ (p.seed * 2654435761u))) / 255.0f;
        }
    }
    img = clamp(img, 0.0f, 1.0f);
    out[i * 3u + 0u] = (uchar)(img.x * 255.0f + 0.5f);
    out[i * 3u + 1u] = (uchar)(img.y * 255.0f + 0.5f);
    out[i * 3u + 2u] = (uchar)(img.z * 255.0f + 0.5f);
}
