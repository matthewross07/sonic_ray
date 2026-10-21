// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
// =============================================================================
// inst/metal/sonicray_pathtracer.metal
// Apple Silicon Metal 3 Hardware Ray-Tracing + Multi-Light NEE + AOV Kernel
// Compiled at runtime via [MTLDevice newLibraryWithSource:...] (no Xcode.app required).
// =============================================================================
#include <metal_stdlib>
#include <metal_raytracing>

using namespace metal;
using namespace metal::raytracing;

// -----------------------------------------------------------------------------
// Shared CPU/GPU Structures (must match src/metal_context.h)
// -----------------------------------------------------------------------------
struct CameraUniforms {
     packed_float3 lookfrom;
     float         fov_rad;
     packed_float3 cam_forward;
     float         aspect;
     packed_float3 cam_right;
     uint          width;
     packed_float3 cam_up;
     uint          height;
     uint          samples;
     uint          max_bounces;
     uint          seed_offset;
     float         clamp_val;
     packed_float3 sky_high;
     float         ambient_intensity;
     packed_float3 sky_low;
     uint          num_lights;
};

// Light types: 0 = invisible sphere light (firework / beacon), 1 = directional/distant sun disk
struct LightSpec {
    packed_float3 position;   // World position (or direction * sun_dist for sun)
    float         radius;     // Sphere radius (km)
    packed_float3 emission;   // Linear RGB radiant intensity
    uint          light_type; // 0 = sphere point/area, 1 = distant solar disk
};

// -----------------------------------------------------------------------------
// Fast PCG32 + Owen-Scrambled Hash RNG for Monte Carlo Path Tracing
// -----------------------------------------------------------------------------
struct RNG {
    uint state;

    inline uint next_uint() {
        uint old = state;
        state = old * 747796405u + 2891336453u;
        uint word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
        return (word >> 22u) ^ word;
    }

    inline float next_float() {
        return float(next_uint() >> 8) * (1.0f / 16777216.0f);
    }

    inline float2 next_float2() {
        return float2(next_float(), next_float());
    }
};

inline RNG make_rng(uint2 gid, uint sample_idx, uint seed_offset) {
    RNG rng;
    uint h = (gid.y * 4099u + gid.x) * 1664525u + sample_idx * 1013904223u + seed_offset;
    rng.state = h ^ 0x9E3779B9u;
    rng.next_uint();
    return rng;
}

// -----------------------------------------------------------------------------
// Orthonormal Basis & Cosine-Weighted Hemisphere Sampling
// -----------------------------------------------------------------------------
inline void build_onb(float3 n, thread float3 &t, thread float3 &b) {
    float s = (n.z >= 0.0f) ? 1.0f : -1.0f;
    float a = -1.0f / (s + n.z);
    float d = n.x * n.y * a;
    t = float3(1.0f + s * n.x * n.x * a, s * d, -s * n.x);
    b = float3(d, s + n.y * n.y * a, -n.y);
}

inline float3 sample_cosine_hemisphere(float2 u, float3 n) {
    float phi = 2.0f * M_PI_F * u.x;
    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(max(0.0f, 1.0f - u.y));
    float3 t, b;
    build_onb(n, t, b);
    return normalize(t * (cos_theta * cos(phi)) + b * (cos_theta * sin(phi)) + n * sin_theta);
}

// -----------------------------------------------------------------------------
// Kernel 1: Primary Ambient + First-Hit Auxiliary AOV Pass (amb, mask, albedo, normal, uv, riv)
// Renders the ambient sky-fill layer AND the first-hit buffers (mask, albedo, normal, uv, river mask)
// and world-space UV position maps for satellite scanlines in a SINGLE pass.
// -----------------------------------------------------------------------------
kernel void render_ambient_and_aovs_kernel(
    uint2                                   gid          [[thread_position_in_grid]],
    constant CameraUniforms&                cam          [[buffer(0)]],
    primitive_acceleration_structure        accel_struct [[buffer(1)]],
    device const packed_float3*             vertices     [[buffer(2)]],
    device const packed_float3*             normals      [[buffer(3)]],
    device const packed_float2*             uvs          [[buffer(4)]],
    device const packed_uint3*              indices      [[buffer(5)]],
    texture2d<float, access::sample>        albedo_tex   [[texture(0)]],
    texture2d<float, access::sample>        river_tex    [[texture(1)]],
    device packed_float3*                   out_amb      [[buffer(6)]],
    device float*                           out_mask     [[buffer(7)]],
    device packed_float3*                   out_albedo   [[buffer(8)]],
    device packed_float3*                   out_normal   [[buffer(9)]],
    device packed_float2*                   out_uv       [[buffer(10)]],
    device packed_float2*                   out_riv      [[buffer(11)]]
) {
    if (gid.x >= cam.width || gid.y >= cam.height) return;
    uint pix_idx = gid.y * cam.width + gid.x;

    constexpr sampler lin_sampler(coord::normalized, address::clamp_to_edge, filter::linear);
    intersector<triangle_data> inter;
    inter.assume_geometry_type(geometry_type::triangle);
    inter.force_opacity(forced_opacity::opaque);

    float3 accum_amb    = float3(0.0f);
    float3 accum_albedo = float3(0.0f);
    float3 accum_normal = float3(0.0f);
    float2 accum_uv     = float2(0.0f);
    float2 accum_riv    = float2(0.0f);
    float  accum_hit    = 0.0f;

    float tan_half_fov = tan(0.5f * cam.fov_rad);
    float3 lookfrom    = float3(cam.lookfrom);
    float3 cam_f       = float3(cam.cam_forward);
    float3 cam_r       = float3(cam.cam_right);
    float3 cam_u       = float3(cam.cam_up);

    for (uint s = 0; s < cam.samples; ++s) {
        RNG rng = make_rng(gid, s, cam.seed_offset);
        float2 jitter = rng.next_float2();
        float ndc_x = ((float(gid.x) + jitter.x) / float(cam.width)) * 2.0f - 1.0f;
        float ndc_y = 1.0f - ((float(gid.y) + jitter.y) / float(cam.height)) * 2.0f;

        ray primary_ray;
        primary_ray.origin = lookfrom;
        primary_ray.direction = normalize(
            cam_f + cam_r * (ndc_x * tan_half_fov * cam.aspect) + cam_u * (ndc_y * tan_half_fov)
        );
        primary_ray.min_distance = 1e-3f;
        primary_ray.max_distance = INFINITY;

        auto hit = inter.intersect(primary_ray, accel_struct);
        if (hit.type == intersection_type::none) {
            continue;
        }

        accum_hit += 1.0f;
        uint prim_id = hit.primitive_id;
        uint3 tri = uint3(indices[prim_id]);
        float2 bary = hit.triangle_barycentric_coord;
        float w0 = 1.0f - bary.x - bary.y;
        float w1 = bary.x;
        float w2 = bary.y;

        float3 n0 = normalize(
            w0 * float3(normals[tri.x]) +
            w1 * float3(normals[tri.y]) +
            w2 * float3(normals[tri.z])
        );
        if (dot(n0, primary_ray.direction) > 0.0f) n0 = -n0;

        float2 uv0 = w0 * float2(uvs[tri.x]) + w1 * float2(uvs[tri.y]) + w2 * float2(uvs[tri.z]);
        // Sample linear albedo & linear river mask (no sRGB gamma corruption on data channels!)
        float3 rho0 = albedo_tex.sample(lin_sampler, float2(uv0.x, 1.0f - uv0.y)).rgb;
        float2 riv0 = river_tex.sample(lin_sampler, float2(uv0.x, 1.0f - uv0.y)).rg;

        accum_albedo += rho0;
        accum_normal += n0;
        accum_uv     += uv0;
        accum_riv    += riv0;

        // Ambient hemisphere bounce
        float3 p0 = primary_ray.origin + primary_ray.direction * hit.distance;
        float3 bounce_dir = sample_cosine_hemisphere(rng.next_float2(), n0);

        ray amb_ray;
        amb_ray.origin = p0 + n0 * 1.5e-3f;
        amb_ray.direction = bounce_dir;
        amb_ray.min_distance = 1e-3f;
        amb_ray.max_distance = INFINITY;

        inter.accept_any_intersection(true);
        auto occ = inter.intersect(amb_ray, accel_struct);
        inter.accept_any_intersection(false);

        if (occ.type == intersection_type::none) {
            float sky_t = clamp(0.5f * (bounce_dir.y + 1.0f), 0.0f, 1.0f);
            float3 sky_col = mix(float3(cam.sky_low), float3(cam.sky_high), sky_t) * cam.ambient_intensity;
            accum_amb += min(rho0 * sky_col, float3(cam.clamp_val));
        }
    }

    float inv_s = 1.0f / float(max(cam.samples, 1u));
    float inv_h = (accum_hit > 0.0f) ? (1.0f / accum_hit) : 0.0f;

    out_amb[pix_idx]    = packed_float3(accum_amb * inv_s);
    out_mask[pix_idx]   = accum_hit * inv_s;
    out_albedo[pix_idx] = packed_float3(accum_albedo * inv_h);
    out_normal[pix_idx] = packed_float3((accum_hit > 0.0f) ? normalize(accum_normal * inv_h) : float3(0.0f));
    out_uv[pix_idx]     = packed_float2(accum_uv * inv_h);
    out_riv[pix_idx]    = packed_float2(accum_riv * inv_h);
}

// -----------------------------------------------------------------------------
// Kernel 2: Batched Multi-Light Path-Tracing Kernel (Up to K = 32 Lights per Pass)
// Evaluates Next-Event Estimation (NEE) across K lights simultaneously for both
// direct lighting (bounce 0) and 1-bounce diffuse interreflection (bounce 1),
// writing exact, zero-crosstalk RGB layers to `out_layers[k * num_pixels + pix_idx]`.
// -----------------------------------------------------------------------------
constant uint MAX_BATCH_LIGHTS = 32;

// Exact illuminance (per unit radiance) of a uniform spherical emitter seen under half-angle sigma, at a surface whose normal makes
// angle theta with the direction to the sphere centre (Lagarde & de Rousiers, "Moving Frostbite to PBR").  Handles the horizon
// partially cutting the sphere, and is exact for any distance (the plain "solid angle x cos" shortcut overestimates up to 2x
// when a surface is close to the light).
inline float illuminance_sphere(float cosTheta, float sinSigmaSqr) {
    float sinTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f));
    float ill = 0.0f;
    if (cosTheta * cosTheta > sinSigmaSqr) {
        ill = M_PI_F * sinSigmaSqr * clamp(cosTheta, 0.0f, 1.0f);
    } else {
        float x = sqrt(max(1.0f / max(sinSigmaSqr, 1e-8f) - 1.0f, 0.0f));
        float y = clamp(-x * (cosTheta / max(sinTheta, 1e-6f)), -1.0f, 1.0f);
        float sinThetaSqrtY = sinTheta * sqrt(max(1.0f - y * y, 0.0f));
        ill = (cosTheta * acos(y) - x * sinThetaSqrtY) * sinSigmaSqr + atan(sinThetaSqrtY / max(x, 1e-6f));
    }
    return max(ill, 0.0f);
}

// Direct light from sphere `light` at (p, n): analytic unshadowed illuminance x one stochastic shadow ray for visibility.
// (Exact when fully lit or fully shadowed; a slightly biased but very low-noise estimate inside penumbrae.)
inline float3 eval_light_nee(
    float3                           p,
    float3                           n,
    float3                           rho,
    constant LightSpec&              light,
    float2                           u_rand,
    primitive_acceleration_structure accel_struct
) {
    float3 to_center = float3(light.position) - p;
    float dist_sq = dot(to_center, to_center);
    float dist = sqrt(dist_sq);
    if (dist <= light.radius) return float3(0.0f);                  // surface point inside the emitter

    float3 dir_c = to_center / dist;
    float sin_sig_sq = min((light.radius * light.radius) / dist_sq, 0.999f);
    float E = illuminance_sphere(dot(n, dir_c), sin_sig_sq);
    if (E <= 0.0f) return float3(0.0f);

    // Visibility: a point on the disk through the sphere centre, perpendicular to the view direction
    float3 t, b;
    build_onb(dir_c, t, b);
    float r_disk = light.radius * sqrt(u_rand.x);
    float phi = 2.0f * M_PI_F * u_rand.y;
    float3 sample_pt = float3(light.position) + r_disk * (cos(phi) * t + sin(phi) * b);
    float3 L_vec = sample_pt - p;
    float L_dist = length(L_vec);
    float3 wi = L_vec / max(L_dist, 1e-4f);
    if (dot(n, wi) <= 0.0f) wi = (dot(n, dir_c) > 0.0f) ? dir_c : normalize(dir_c - n * dot(n, dir_c) + 1e-3f * n);

    intersector<triangle_data> shadow_inter;
    shadow_inter.assume_geometry_type(geometry_type::triangle);
    shadow_inter.force_opacity(forced_opacity::opaque);
    shadow_inter.accept_any_intersection(true);
    ray shadow_ray;
    shadow_ray.origin = p + n * 1.5e-3f;
    shadow_ray.direction = wi;
    shadow_ray.min_distance = 1e-3f;
    shadow_ray.max_distance = max(L_dist - light.radius, 2e-3f);     // stop at (or before) the emitter's surface
    auto shadow_hit = shadow_inter.intersect(shadow_ray, accel_struct);
    if (shadow_hit.type != intersection_type::none) return float3(0.0f);

    return (rho * M_1_PI_F) * float3(light.emission) * E;            // Lo = rho/pi * L * E
}

kernel void render_multilight_batch_kernel(
    uint2                                   gid          [[thread_position_in_grid]],
    constant CameraUniforms&                cam          [[buffer(0)]],
    primitive_acceleration_structure        accel_struct [[buffer(1)]],
    device const packed_float3*             vertices     [[buffer(2)]],
    device const packed_float3*             normals      [[buffer(3)]],
    device const packed_float2*             uvs          [[buffer(4)]],
    device const packed_uint3*              indices      [[buffer(5)]],
    constant LightSpec*                     lights       [[buffer(6)]],
    texture2d<float, access::sample>        albedo_tex   [[texture(0)]],
    device packed_float3*                   out_layers   [[buffer(7)]]
) {
    if (gid.x >= cam.width || gid.y >= cam.height) return;
    uint pix_idx = gid.y * cam.width + gid.x;
    uint num_pixels = cam.width * cam.height;
    uint K = min(cam.num_lights, MAX_BATCH_LIGHTS);

    float3 accum[MAX_BATCH_LIGHTS];
    for (uint k = 0; k < K; ++k) {
        accum[k] = float3(0.0f);
    }

    constexpr sampler lin_sampler(coord::normalized, address::clamp_to_edge, filter::linear);
    intersector<triangle_data> inter;
    inter.assume_geometry_type(geometry_type::triangle);
    inter.force_opacity(forced_opacity::opaque);

    float tan_half_fov = tan(0.5f * cam.fov_rad);
    float3 lookfrom    = float3(cam.lookfrom);
    float3 cam_f       = float3(cam.cam_forward);
    float3 cam_r       = float3(cam.cam_right);
    float3 cam_u       = float3(cam.cam_up);

    const uint max_vertices = max(cam.max_bounces, 1u);             // surface vertices per path (1 = direct light only)
    for (uint s = 0; s < cam.samples; ++s) {
        RNG rng = make_rng(gid, s, cam.seed_offset);
        float2 jitter = rng.next_float2();
        float ndc_x = ((float(gid.x) + jitter.x) / float(cam.width)) * 2.0f - 1.0f;
        float ndc_y = 1.0f - ((float(gid.y) + jitter.y) / float(cam.height)) * 2.0f;

        ray cur;
        cur.origin = lookfrom;
        cur.direction = normalize(cam_f + cam_r * (ndc_x * tan_half_fov * cam.aspect) + cam_u * (ndc_y * tan_half_fov));
        cur.min_distance = 1e-3f;
        cur.max_distance = INFINITY;
        float3 throughput = float3(1.0f);

        for (uint depth = 0; depth < max_vertices; ++depth) {
            auto hit = inter.intersect(cur, accel_struct);
            if (hit.type == intersection_type::none) break;

            uint3 tri = uint3(indices[hit.primitive_id]);
            float2 bc = hit.triangle_barycentric_coord;
            float w0 = 1.0f - bc.x - bc.y;
            float3 n = normalize(w0 * float3(normals[tri.x]) + bc.x * float3(normals[tri.y]) + bc.y * float3(normals[tri.z]));
            if (dot(n, cur.direction) > 0.0f) n = -n;
            float2 uv = w0 * float2(uvs[tri.x]) + bc.x * float2(uvs[tri.y]) + bc.y * float2(uvs[tri.z]);
            float3 rho = albedo_tex.sample(lin_sampler, float2(uv.x, 1.0f - uv.y)).rgb;
            float3 p = cur.origin + cur.direction * hit.distance;

            // Next-event estimation for every light in the batch at this vertex (shared path => exact RGB, no cross-talk)
            float2 u_nee = rng.next_float2();
            for (uint k2 = 0; k2 < K; ++k2) {
                float3 Ld = throughput * eval_light_nee(p, n, rho, lights[k2], u_nee, accel_struct);
                accum[k2] += min(Ld, float3(cam.clamp_val));
            }
            if (depth + 1 >= max_vertices) break;

            // Continue the path: cosine-weighted bounce, throughput *= albedo; Russian roulette after 2 bounces
            float3 thr_next = throughput * rho;
            if (depth >= 2) {
                float q = clamp(max(thr_next.x, max(thr_next.y, thr_next.z)), 0.05f, 0.95f);
                if (rng.next_float() > q) break;
                thr_next /= q;
            }
            throughput = thr_next;
            cur.origin = p + n * 1.5e-3f;
            cur.direction = sample_cosine_hemisphere(rng.next_float2(), n);
            cur.min_distance = 1e-3f;
            cur.max_distance = INFINITY;
        }
    }

    float inv_s = 1.0f / float(max(cam.samples, 1u));
    for (uint k3 = 0; k3 < K; ++k3) {
        out_layers[k3 * num_pixels + pix_idx] = packed_float3(accum[k3] * inv_s);
    }
}
