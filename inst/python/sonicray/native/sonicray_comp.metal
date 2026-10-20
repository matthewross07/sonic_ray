// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
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

// ----------------------------------------------------------------------------------------------- sky
// Day sky = blend of two path-traced sky layers, normalised by their 92nd percentile (GPU log-histogram, no CPU readback),
// tinted zenith->horizon, ACES-mapped, plus dawn/sunset bands and a twinkling starfield.  Night (p <= 0) is stars only.
struct SkyP { int W; int H; int day; int use_stars; float f; float t; float star_amt; float y_edge; float ox1; float dawn_band; float set_band;
              float sky_exp; float zen[3]; float hort[3]; float venus[3]; float sunc[3]; float lo; float step; int nb; };
inline float3 sky_base(uint i, int x, int y, device const pf3* sa, device const pf3* sb, constant SkyP& p, float p90) {
    float3 raw = float3(sa[i]) * (1.0f - p.f) + float3(sb[i]) * p.f;
    float yn = (float)y / (float)(p.H - 1);
    float3 zt = (1.0f - yn) * float3(p.zen[0], p.zen[1], p.zen[2]) + yn * float3(p.hort[0], p.hort[1], p.hort[2]);
    float3 xx = max(raw / p90 * zt * p.sky_exp, 0.0f);
    float3 mapped = clamp((xx * (2.51f * xx + 0.03f)) / (xx * (2.43f * xx + 0.59f) + 0.14f), 0.0f, 1.0f);
    float3 s = pow(mapped, float3(1.0f / 2.2f));
    float yb = exp(-pow(((float)y - p.y_edge) / (0.09f * (float)p.H), 2.0f));
    float xs = exp(-pow(((float)x - p.ox1) / (0.32f * (float)p.W), 2.0f));
    s = clamp(s + yb * (float3(p.venus[0], p.venus[1], p.venus[2]) * (0.26f * p.dawn_band) +
                        float3(p.sunc[0], p.sunc[1], p.sunc[2]) * (0.34f * p.set_band * (0.45f + 0.55f * xs))), 0.0f, 1.0f);
    return s;
}
kernel void sky_hist(device const pf3* sa [[buffer(0)]], device const pf3* sb [[buffer(1)]], device atomic_uint* hist [[buffer(2)]],
                     constant SkyP& p [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    int x = (int)g.x * 4, y = (int)g.y * 4;                       // every 4th pixel in each direction is plenty for a percentile
    if (x >= p.W || y >= p.H) return;
    uint i = (uint)y * (uint)p.W + (uint)x;
    float3 raw = float3(sa[i]) * (1.0f - p.f) + float3(sb[i]) * p.f;
    for (int c = 0; c < 3; ++c) {
        float v = max(raw[c], 1e-9f);
        int b = clamp((int)((log2(v) - p.lo) / p.step), 0, p.nb - 1);
        atomic_fetch_add_explicit(&hist[b], 1u, memory_order_relaxed);
    }
}
kernel void sky_pctl(device const uint* hist [[buffer(0)]], device float* out [[buffer(1)]], constant SkyP& p [[buffer(2)]],
                     uint2 g [[thread_position_in_grid]]) {
    if (g.x != 0 || g.y != 0) return;
    uint n = 0;
    for (int b = 0; b < p.nb; ++b) n += hist[b];
    float target = 0.92f * (float)(n > 0 ? n - 1 : 0);
    uint c = 0; int hit = p.nb - 1;
    for (int b = 0; b < p.nb; ++b) { c += hist[b]; if ((float)c > target) { hit = b; break; } }
    out[0] = max(exp2(p.lo + ((float)hit + 0.5f) * p.step), 1e-3f);
}
kernel void sky_hor(device const pf3* sa [[buffer(0)]], device const pf3* sb [[buffer(1)]], device const float* p90 [[buffer(2)]],
                    device float* hor [[buffer(3)]], constant SkyP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x != 0 || g.y != 0) return;
    if (p.day == 0) { hor[0] = 0.45f; hor[1] = 0.55f; hor[2] = 0.72f; return; }
    int y = min((int)p.y_edge, p.H - 1), x = p.W / 2;
    float3 s = sky_base((uint)y * (uint)p.W + (uint)x, x, y, sa, sb, p, p90[0]);
    hor[0] = s.x; hor[1] = s.y; hor[2] = s.z;
}
kernel void sky_render(device const pf3* sa [[buffer(0)]], device const pf3* sb [[buffer(1)]], device const float* p90 [[buffer(2)]],
                       device const pf3* stars [[buffer(3)]], device pf3* sky [[buffer(4)]], constant SkyP& p [[buffer(5)]],
                       uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= p.W || (int)g.y >= p.H) return;
    uint i = g.y * (uint)p.W + g.x;
    float3 s = p.day != 0 ? sky_base(i, (int)g.x, (int)g.y, sa, sb, p, p90[0]) : float3(0.0f);
    if (p.use_stars != 0) {
        float tw = 0.85f + 0.15f * sin(p.t * 1.6f + (float)g.x * 0.17f + (float)g.y * 0.11f);
        float3 st = pow(clamp(float3(stars[i]) * 22.0f, 0.0f, 1.0f), float3(0.75f));
        s = clamp(s + st * tw * p.star_amt, 0.0f, 1.0f);
    }
    sky[i] = pf3(s);
}

// ----------------------------------------------------------------------------------------------- firework sparks
// Host-generated primitives (filaments = thick lines, star heads = ellipses), painted per plane in draw order, binned into 32x32
// tiles.  Each output pixel takes 2x2 sub-samples (the CPU path drew at 2x resolution and box-filtered), five planes
// (toc, chla, alk, shimmer, white) -> coloured, summed, scaled by 0.92.
struct Prim { float a; float b; float c; float d; float w; float fill; int plane; int kind; };
struct SparkP { int W; int H; int ntx; float col[15]; };
kernel void spark_raster(device const Prim* prims [[buffer(0)]], device const uint* tile_off [[buffer(1)]], device const uint* tile_idx [[buffer(2)]],
                         device pf3* ov [[buffer(3)]], device atomic_uint* omax [[buffer(4)]], constant SparkP& p [[buffer(5)]],
                         uint2 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float local = 0.0f;
    if ((int)g.x < p.W && (int)g.y < p.H) {
        uint tile = (g.y / 32u) * (uint)p.ntx + (g.x / 32u);
        float4 v[5] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        float2 q[4];
        for (int s = 0; s < 4; ++s) q[s] = float2((float)(g.x * 2u + (uint)(s & 1)), (float)(g.y * 2u + (uint)(s >> 1)));
        for (uint k = tile_off[tile]; k < tile_off[tile + 1]; ++k) {
            Prim pr = prims[tile_idx[k]];
            for (int s = 0; s < 4; ++s) {
                bool hit;
                if (pr.kind == 0) {
                    float2 a = float2(pr.a, pr.b), d = float2(pr.c - pr.a, pr.d - pr.b), r = q[s] - a;
                    float L2 = dot(d, d);
                    if (L2 < 1e-6f) hit = length(r) <= pr.w * 0.5f - 0.10f;
                    else {
                        float L = sqrt(L2), t = dot(r, d) / L, perp = fabs(r.x * d.y - r.y * d.x) / L;
                        hit = perp <= pr.w * 0.5f - 0.10f && t >= -0.5f && t <= L + 0.5f;
                    }
                } else {
                    float2 c = float2((pr.a + pr.c) * 0.5f, (pr.b + pr.d) * 0.5f), rad = float2((pr.c - pr.a) * 0.5f + 0.4f, (pr.d - pr.b) * 0.5f + 0.4f);
                    float2 e = (q[s] - c) / rad;
                    hit = dot(e, e) <= 1.0f;
                }
                if (hit) v[pr.plane][s] = pr.fill;
            }
        }
        float3 o = float3(0.0f);
        for (int k = 0; k < 5; ++k) {
            float m = floor((v[k].x + v[k].y + v[k].z + v[k].w) * 0.25f + 0.5f) / 255.0f;
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
// orb_mode: 0 none, 1 noon corona (glow only), 2 descending solar disk + Mie aureole.  orb[]: ox, oy, r, vis, dusk, amp, sun colour rgb.
struct FinalP { int W; int H; int orb_mode; int use_atm; int use_ov; int dither; uint seed; int nb;
                float atm; float fade; float scale; float orb[9]; float bcn[96]; };   // bcn: nb x (px, py, beam_h, pulse)
kernel void final_frame(device const pf3* terrain [[buffer(0)]], device const float* m [[buffer(1)]], device const pf3* sky [[buffer(2)]],
                        device const float* hor [[buffer(3)]], device const float* aerial [[buffer(4)]],
                        device const pf3* b1 [[buffer(5)]], device const pf3* b2 [[buffer(6)]], device const pf3* b3 [[buffer(7)]],
                        device const pf3* ov [[buffer(8)]], device const pf3* ob1 [[buffer(9)]], device const pf3* ob2 [[buffer(10)]],
                        device const uint* bmax [[buffer(11)]], device const uint* omax [[buffer(12)]], device uchar* out [[buffer(13)]],
                        constant FinalP& p [[buffer(14)]], uint2 g [[thread_position_in_grid]]) {
    if ((int)g.x >= p.W || (int)g.y >= p.H) return;
    uint i = g.y * (uint)p.W + g.x;
    float m3 = m[i];
    float3 t = float3(terrain[i]);
    if (p.use_atm != 0) {
        float aw = aerial[i] * p.atm;
        t = t * (1.0f - aw) + float3(hor[0], hor[1], hor[2]) * aw;
    }
    float3 img = float3(sky[i]) * (1.0f - m3) + t * m3;
    float x = (float)g.x, y = (float)g.y, sc = p.scale;
    if (p.orb_mode != 0) {
        float H = (float)p.H;
        float ox = p.orb[0], oy = p.orb[1];
        float d2 = (x - ox) * (x - ox) + (y - oy) * (y - oy);
        float3 sun_col = float3(p.orb[6], p.orb[7], p.orb[8]);
        float3 core = float3(0.0f), glow = float3(0.0f);
        if (p.orb_mode == 1) {
            glow = exp(-d2 / (2.0f * (0.14f * H) * (0.14f * H))) * p.orb[5] * sun_col;
        } else {
            float r = p.orb[2], vis = p.orb[3], dusk = p.orb[4], d = sqrt(d2);
            float u_limb = clamp(1.0f - d2 / max(r * r, 1e-4f), 0.0f, 1.0f), mu = sqrt(u_limb);
            float disc = (u_limb > 0.0f ? 1.0f : 0.0f) * (0.35f + 0.65f * mu) * clamp((r - d) / (0.22f * r) + 0.5f, 0.0f, 1.0f);
            float th = d / (0.12f * H);
            float mie_in = 1.0f / pow(1.0f + pow(th / 0.14f, 1.95f), 1.45f);
            float mie_out = exp(-d2 / (2.0f * (0.09f * H) * (0.09f * H))) * 0.16f;
            core = disc * (sun_col * 0.35f + 0.65f) * (1.15f * vis * (1.0f - dusk));
            glow = (mie_in * 0.48f + mie_out) * vis * sun_col;
        }
        img += core * (1.0f - m3) + glow * (1.0f - 0.72f * m3);
    }
    if (as_type<float>(bmax[0]) > 0.01f) img += (float3(b1[i]) * 0.56f + float3(b2[i]) * 0.72f + float3(b3[i]) * 0.54f) * 0.52f;
    if (p.use_ov != 0 && as_type<float>(omax[0]) > 0.0f) {
        float3 ot = float3(ov[i]) * 1.12f + float3(ob1[i]) * 0.26f + float3(ob2[i]) * 0.20f;
        float mo = max(ot.x, max(ot.y, ot.z));
        float3 os = ot * ((1.0f - exp(-1.12f * mo)) / max(mo, 1e-6f));
        img = 1.0f - (1.0f - clamp(img, 0.0f, 1.0f)) * (1.0f - clamp(os, 0.0f, 0.96f));
    }
    // sonde beacons: luminous core on the stream surface + soft upward plume
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
