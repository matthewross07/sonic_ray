// =============================================================================
// inst/metal/sonicray_compositor.metal
// GPU Compute Pipeline for Linear Light-Layer Accumulation, Satellite Swaths,
// River Glint, Soft-Knee Highlight Roll-Off, and ACES Filmic Tone Mapping.
// =============================================================================
#include <metal_stdlib>
using namespace metal;

struct FrameUniforms {
    uint          width;
    uint          height;
    uint          num_active_layers;
    float         amb_weight;
    packed_float3 sun_tint;
    float         sun_weight_a;
    float         sun_weight_b;
    float         poudre_riv_gain;
    float         sat_active;
    float         sat_uc;
    float         sat_half;
    float         sat_dt_offset;
    float         sat_dur;
    float         sat_tau;
    float         fade;
};

struct ActiveLayerWeight {
    uint          layer_index;
    packed_float3 tint_weight; // w_i(t) * col_i
};

inline float smooth01(float x) {
    float c = clamp(x, 0.0f, 1.0f);
    return c * c * (3.0f - 2.0f * c);
}

inline float3 aces_tonemap(float3 x) {
    return clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
}

kernel void compose_linear_layers_kernel(
    uint2                         gid           [[thread_position_in_grid]],
    constant FrameUniforms&       u             [[buffer(0)]],
    device const packed_float3*   amb_layer     [[buffer(1)]],
    device const packed_float3*   sun_layer_a   [[buffer(2)]],
    device const packed_float3*   sun_layer_b   [[buffer(3)]],
    device const packed_float2*   uv_layer      [[buffer(4)]],
    device const packed_float2*   riv_layer     [[buffer(5)]],
    device const packed_float3*   light_layers  [[buffer(6)]],
    constant ActiveLayerWeight*   active_list   [[buffer(7)]],
    device packed_float3*         out_tonemap   [[buffer(8)]]
) {
    if (gid.x >= u.width || gid.y >= u.height) return;
    uint pix = gid.y * u.width + gid.x;
    uint num_pixels = u.width * u.height;

    float3 amb = float3(amb_layer[pix]);
    float3 lin = amb * u.amb_weight;

    // 1. Interpolated Sun Layer + Poudre Cyan-Azure River Highlight
    float3 sun_mix = float3(sun_layer_a[pix]) * u.sun_weight_a + float3(sun_layer_b[pix]) * u.sun_weight_b;
    lin += sun_mix * float3(u.sun_tint);

    float2 riv = float2(riv_layer[pix]);
    lin += riv.x * float3(0.04f, 0.30f, 0.65f) * u.poudre_riv_gain;

    // 2. Satellite WRS-2 Orbital Scanline Swath (evaluated in world UV space)
    if (u.sat_active > 0.0f) {
        float2 uv = float2(uv_layer[pix]);
        float sw = clamp((u.sat_half - fabs(uv.x - u.sat_uc)) / 0.014f, 0.0f, 1.0f);
        float dtp = u.sat_dt_offset - u.sat_dur * clamp(uv.y, 0.0f, 1.0f);
        float ramp = smooth01(dtp / 0.22f);
        float glow = exp(-max(dtp, 0.0f) / u.sat_tau) * ramp * 0.68f;
        float line = 2.15f * exp(-pow((dtp - 0.06f) / 0.11f, 2.0f));
        float sat_lvl = (glow + line) * sw * u.sat_active;
        float3 sat_base = amb * 0.42f + float3(sun_layer_a[pix]) * 0.62f;
        lin += sat_base * (sat_lvl * float3(0.84f, 0.93f, 1.00f) * 1.35f);
    }

    // 3. Linear Superposition of Active Firework & Beacon Light Layers
    for (uint i = 0; i < u.num_active_layers; ++i) {
        uint idx = active_list[i].layer_index;
        float3 w_col = float3(active_list[i].tint_weight);
        lin += float3(light_layers[idx * num_pixels + pix]) * w_col;
    }

    // 4. Chromaticity-Preserving Soft-Knee Highlight Roll-Off + ACES Tone-Mapping
    lin = max(lin, float3(0.0f));
    float mx = max(lin.r, max(lin.g, lin.b));
    float mc = (mx > 0.55f) ? (0.55f + (mx - 0.55f) / (1.0f + 1.6f * (mx - 0.55f))) : mx;
    lin *= (mc / max(mx, 1e-6f));

    float3 mapped = pow(aces_tonemap(lin), float3(1.0f / 2.2f)) * u.fade;
    out_tonemap[pix] = packed_float3(mapped);
}
