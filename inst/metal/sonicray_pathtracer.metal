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

// Compile-time switch: scenes without water get kernels with ALL water code removed (zero cost). Set by the host when the scene is uploaded.
constant bool HAS_WATER [[function_constant(0)]];

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

// Water: a homogeneous dielectric volume {y < level} over the terrain's x/z footprint (water never extends beyond the terrain); with `cutaway` the
// footprint box also has vertical glass walls (a diorama).  Coefficients are per scene unit.  Must match HostWaterUniforms in src/metal_context.h.
struct WaterUniforms {
    float level;          // world y of the water surface
    float ior;            // index of refraction (1.333)
    float roughness;      // capillary-wave tilt of the surface normal (0 = mirror flat)
    uint  cutaway;        // 1: volume is the x/z box of the terrain with side walls
    float sigma_t[3];     // extinction = absorption + scattering   (Beer-Lambert)
    float sigma_s[3];     // scattering (single-scatter in-scatter)
    float bmin[3];        // terrain bounds (x, y, z)
    float bmax[3];
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
// Water: exact dielectric Fresnel, the interface of the water volume, Beer-Lambert transmittance
// -----------------------------------------------------------------------------
inline float fresnel_dielectric(float cos_i, float n_i, float n_t) {      // unpolarised, exact; 1 on total internal reflection
    cos_i = clamp(cos_i, 0.0f, 1.0f);
    float r = n_i / n_t;
    float sin_t2 = r * r * (1.0f - cos_i * cos_i);
    if (sin_t2 >= 1.0f) return 1.0f;
    float cos_t = sqrt(1.0f - sin_t2);
    float rs = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t);
    float rp = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t);
    return 0.5f * (rs * rs + rp * rp);
}
inline float3 w_sigma_t(constant WaterUniforms& w) { return float3(w.sigma_t[0], w.sigma_t[1], w.sigma_t[2]); }
inline float3 w_sigma_s(constant WaterUniforms& w) { return float3(w.sigma_s[0], w.sigma_s[1], w.sigma_s[2]); }
inline bool   w_any_scatter(constant WaterUniforms& w) { return (w.sigma_s[0] + w.sigma_s[1] + w.sigma_s[2]) > 0.0f; }

inline bool water_in_footprint(float3 p, constant WaterUniforms& w) {
    return p.x > w.bmin[0] && p.x < w.bmax[0] && p.z > w.bmin[2] && p.z < w.bmax[2];
}
inline bool water_contains(float3 p, constant WaterUniforms& w) {
    if (p.y >= w.level) return false;
    if (w.cutaway == 0u) return water_in_footprint(p, w);
    return water_in_footprint(p, w) && p.y > w.bmin[1];
}

// Where does the ray meet the water volume's boundary?  Outside: the entry (the top face; with cutaway also the side walls).
// Inside: the exit (the top face; with cutaway also the walls).  n_out is the volume's outward normal at that point.
inline bool water_interface(float3 o, float3 d, bool inside, constant WaterUniforms& w, thread float& t_hit, thread float3& n_out) {
    if (w.cutaway == 0u) {                                          // top surface only, within the terrain footprint
        if (inside ? (d.y <= 1e-6f) : (d.y >= -1e-6f)) return false;
        t_hit = (w.level - o.y) / d.y; n_out = float3(0, 1, 0);
        return t_hit > 1e-4f && water_in_footprint(o + d * t_hit, w);
    }
    float3 bmin = float3(w.bmin[0], w.bmin[1], w.bmin[2]), bmax = float3(w.bmax[0], w.level, w.bmax[2]);
    float3 inv = 1.0f / select(d, float3(1e-9f), fabs(d) < 1e-9f);
    float3 t0 = (bmin - o) * inv, t1 = (bmax - o) * inv;
    float3 tlo = min(t0, t1), thi = max(t0, t1);
    float tn = max(tlo.x, max(tlo.y, tlo.z)), tf = min(thi.x, min(thi.y, thi.z));
    if (inside) {
        if (tf <= 1e-4f) return false;
        t_hit = tf;
        n_out = (tf == thi.x) ? float3(d.x > 0 ? 1.0f : -1.0f, 0, 0) : ((tf == thi.y) ? float3(0, d.y > 0 ? 1.0f : -1.0f, 0) : float3(0, 0, d.z > 0 ? 1.0f : -1.0f));
        return true;
    }
    if (tf <= tn || tn <= 1e-4f) return false;
    t_hit = tn;
    n_out = (tn == tlo.x) ? float3(d.x > 0 ? -1.0f : 1.0f, 0, 0) : ((tn == tlo.y) ? float3(0, d.y > 0 ? -1.0f : 1.0f, 0) : float3(0, 0, d.z > 0 ? -1.0f : 1.0f));
    return true;
}

// Transmittance of the STRAIGHT segment a->b (used by light rays: caustics/refraction of light rays are not modelled): Beer-Lambert over the part
// below the surface, times the Fresnel transmission of the surface crossing.
inline float3 water_segment_transmittance(float3 a, float3 b, constant WaterUniforms& w) {
    float3 d = b - a;
    float len = length(d);
    if (len < 1e-6f) return float3(1.0f);
    float ya = a.y - w.level, yb = b.y - w.level;
    float f0 = 0.0f, f1 = 1.0f;
    bool crosses = (ya > 0.0f) != (yb > 0.0f);
    if (ya > 0.0f && yb > 0.0f) return float3(1.0f);
    if (crosses) { float tc = ya / (ya - yb); if (ya > 0.0f) f0 = tc; else f1 = tc; }
    {                                                              // clip to the terrain footprint
        for (int ax = 0; ax < 3; ax += 2) {
            float lo = w.bmin[ax], hi = w.bmax[ax], oa = ax == 0 ? a.x : a.z, da = ax == 0 ? d.x : d.z;
            if (fabs(da) < 1e-9f) { if (oa < lo || oa > hi) return float3(1.0f); continue; }
            float u0 = (lo - oa) / da, u1 = (hi - oa) / da;
            f0 = max(f0, min(u0, u1)); f1 = min(f1, max(u0, u1));
        }
        if (f1 <= f0) return float3(1.0f);
    }
    float3 T = exp(-w_sigma_t(w) * ((f1 - f0) * len));
    if (crosses) T *= (1.0f - fresnel_dielectric(fabs(d.y) / len, 1.0f, w.ior));
    return T;
}

// Transmittance of an infinite ray from p along dir to the sky (for ambient light reaching an underwater point).
inline float3 water_sky_transmittance(float3 p, float3 dir, constant WaterUniforms& w) {
    if (p.y >= w.level || dir.y <= 1e-4f) return float3(1.0f);
    float l = (w.level - p.y) / dir.y;
    return exp(-w_sigma_t(w) * l) * (1.0f - fresnel_dielectric(dir.y, 1.0f, w.ior));
}

inline bool ray_sphere(float3 o, float3 d, float3 c, float r, thread float& t) {
    float3 oc = o - c;
    float b = dot(oc, d), cc = dot(oc, oc) - r * r, disc = b * b - cc;
    if (disc < 0.0f) return false;
    float sq = sqrt(disc), t0 = -b - sq;
    if (t0 > 1e-4f) { t = t0; return true; }
    float t1 = -b + sq;
    if (t1 > 1e-4f) { t = t1; return true; }
    return false;
}

// Slightly tilt the surface normal (capillary waves)
inline float3 perturb_normal(float3 n, float roughness, float2 u) {
    if (roughness <= 0.0f) return n;
    float r = sqrt(-2.0f * log(max(u.x, 1e-7f))) * roughness, ph = 2.0f * M_PI_F * u.y;
    float3 t, b; build_onb(n, t, b);
    return normalize(n + r * (cos(ph) * t + sin(ph) * b));
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
    device packed_float2*                   out_riv      [[buffer(11)]],
    constant WaterUniforms&                 water        [[buffer(12)]],
    device float*                           out_wdepth   [[buffer(13)]]
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
    float  accum_hit    = 0.0f;          // terrain hits (normalises the first-hit buffers)
    float  accum_cov    = 0.0f;          // pixel coverage: terrain OR the water surface
    float  accum_wdepth = 0.0f;          // path length inside the water to the terrain hit

    float tan_half_fov = tan(0.5f * cam.fov_rad);
    float3 lookfrom    = float3(cam.lookfrom);
    float3 cam_f       = float3(cam.cam_forward);
    float3 cam_r       = float3(cam.cam_right);
    float3 cam_u       = float3(cam.cam_up);
    float3 sky_mean    = 0.5f * (float3(cam.sky_low) + float3(cam.sky_high)) * cam.ambient_intensity;

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

        float3 thr = float3(1.0f);
        float  wdepth = 0.0f;
        bool   in_water = false, via_water = false, covered = false;
        uint   events = 0;
        if (HAS_WATER) in_water = water_contains(lookfrom, water);

        intersection_result<triangle_data> hit;
        bool got = false;
        while (true) {
            hit = inter.intersect(primary_ray, accel_struct);
            got = hit.type != intersection_type::none;
            float t_s = got ? hit.distance : INFINITY;
            if (HAS_WATER) {
                float t_w = 0.0f; float3 n_w = float3(0.0f);
                bool wev = water_interface(primary_ray.origin, primary_ray.direction, in_water, water, t_w, n_w) && t_w < t_s;
                if (wev && !in_water && water.cutaway != 0u && fabs(n_w.y) < 0.5f) {      // side wall: only where the point is above the terrain
                    ray up; up.origin = primary_ray.origin + primary_ray.direction * t_w; up.direction = float3(0, 1, 0); up.min_distance = 1e-3f; up.max_distance = INFINITY;
                    inter.accept_any_intersection(true);
                    if (inter.intersect(up, accel_struct).type != intersection_type::none) wev = false;
                    inter.accept_any_intersection(false);
                }
                float seg = wev ? t_w : t_s;
                if (in_water) {
                    if (!isfinite(seg)) break;
                    float3 st = w_sigma_t(water), Tseg = exp(-st * seg);
                    wdepth += seg;
                    if (w_any_scatter(water)) accum_amb += min(thr * (1.0f - Tseg) * w_sigma_s(water) / max(st, float3(1e-6f)) * sky_mean, float3(cam.clamp_val));
                    thr *= Tseg;
                    if (max(thr.x, max(thr.y, thr.z)) < 1e-5f) break;
                }
                if (wev) {
                    covered = true;
                    if (++events > 8u) break;
                    float3 d = primary_ray.direction;
                    float3 N = n_w; if (dot(N, d) > 0.0f) N = -N;
                    N = perturb_normal(N, water.roughness, rng.next_float2());
                    if (dot(N, d) > 0.0f) N = n_w * (dot(n_w, d) > 0.0f ? -1.0f : 1.0f);
                    float cosi = clamp(-dot(N, d), 0.0f, 1.0f);
                    float n_i = in_water ? water.ior : 1.0f, n_t = in_water ? 1.0f : water.ior;
                    float3 dn;
                    if (rng.next_float() < fresnel_dielectric(cosi, n_i, n_t)) dn = reflect(d, N);
                    else { dn = refract(d, N, n_i / n_t); in_water = !in_water; }
                    primary_ray.origin = primary_ray.origin + d * t_w + dn * 2e-3f;
                    primary_ray.direction = dn;
                    primary_ray.min_distance = 1e-3f;
                    primary_ray.max_distance = INFINITY;
                    via_water = true;
                    continue;
                }
            }
            break;
        }
        if (!got) {
            if (via_water) {                                          // a reflection of the sky off the water
                float sky_t = clamp(0.5f * (primary_ray.direction.y + 1.0f), 0.0f, 1.0f);
                accum_amb += min(thr * mix(float3(cam.sky_low), float3(cam.sky_high), sky_t) * cam.ambient_intensity, float3(cam.clamp_val));
            }
            if (covered) accum_cov += 1.0f;
            continue;
        }

        accum_hit += 1.0f;
        accum_cov += 1.0f;
        accum_wdepth += wdepth;
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
            float3 Lsky = rho0 * sky_col;
            if (HAS_WATER) Lsky *= thr * water_sky_transmittance(p0, bounce_dir, water);     // view path and sky path through the water
            accum_amb += min(Lsky, float3(cam.clamp_val));
        }
    }

    float inv_s = 1.0f / float(max(cam.samples, 1u));
    float inv_h = (accum_hit > 0.0f) ? (1.0f / accum_hit) : 0.0f;

    out_amb[pix_idx]    = packed_float3(accum_amb * inv_s);
    out_mask[pix_idx]   = (HAS_WATER ? accum_cov : accum_hit) * inv_s;
    out_albedo[pix_idx] = packed_float3(accum_albedo * inv_h);
    out_normal[pix_idx] = packed_float3((accum_hit > 0.0f) ? normalize(accum_normal * inv_h) : float3(0.0f));
    out_uv[pix_idx]     = packed_float2(accum_uv * inv_h);
    out_riv[pix_idx]    = packed_float2(accum_riv * inv_h);
    out_wdepth[pix_idx] = accum_wdepth * inv_h;
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
    primitive_acceleration_structure accel_struct,
    constant WaterUniforms&          water
) {
    float3 to_center = float3(light.position) - p;
    float dist_sq = dot(to_center, to_center);
    float dist = sqrt(dist_sq);
    if (dist <= light.radius) return float3(0.0f);                  // surface point inside the emitter

    float3 dir_c = to_center / dist;
    float sin_sig_sq = min((light.radius * light.radius) / dist_sq, 0.999f);
    float E = illuminance_sphere(dot(n, dir_c), sin_sig_sq);
    if (E <= 0.0f) return float3(0.0f);
    float3 w_atten = float3(1.0f);
    if (HAS_WATER) {                                   // light reaching an underwater (or crossing) point is attenuated; skip the shadow ray when it is all gone
        w_atten = water_segment_transmittance(p, float3(light.position), water);
        if (max(w_atten.x, max(w_atten.y, w_atten.z)) < 1e-5f) return float3(0.0f);
    }

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

    return (rho * M_1_PI_F) * float3(light.emission) * E * w_atten;  // Lo = rho/pi * L * E (* water transmittance)
}

// Light scattered toward the path by the water at point q: fluence from sphere light `light` (solid angle x radiance), shadowed and attenuated.
inline float3 eval_light_scatter(float3 q, constant LightSpec& light, float2 u_rand, primitive_acceleration_structure accel_struct, constant WaterUniforms& water) {
    float3 to_c = float3(light.position) - q;
    float dist_sq = dot(to_c, to_c), dist = sqrt(dist_sq);
    if (dist <= light.radius) return float3(0.0f);
    float3 dir_c = to_c / dist;
    float sin2 = min((light.radius * light.radius) / dist_sq, 0.999f);
    float omega = 2.0f * M_PI_F * (1.0f - sqrt(1.0f - sin2));
    float3 t, b; build_onb(dir_c, t, b);
    float r_disk = light.radius * sqrt(u_rand.x), phi = 2.0f * M_PI_F * u_rand.y;
    float3 sample_pt = float3(light.position) + r_disk * (cos(phi) * t + sin(phi) * b);
    float3 atten = water_segment_transmittance(q, sample_pt, water);
    if (max(atten.x, max(atten.y, atten.z)) < 1e-5f) return float3(0.0f);
    float3 L_vec = sample_pt - q;
    float L_dist = length(L_vec);
    intersector<triangle_data> sh;
    sh.assume_geometry_type(geometry_type::triangle);
    sh.force_opacity(forced_opacity::opaque);
    sh.accept_any_intersection(true);
    ray r; r.origin = q; r.direction = L_vec / max(L_dist, 1e-4f); r.min_distance = 1e-3f; r.max_distance = max(L_dist - light.radius, 2e-3f);
    if (sh.intersect(r, accel_struct).type != intersection_type::none) return float3(0.0f);
    return float3(light.emission) * omega * atten;
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
    device packed_float3*                   out_layers   [[buffer(7)]],
    constant WaterUniforms&                 water        [[buffer(8)]]
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
        bool in_water = false, spec_chain = false;                   // spec_chain: only water interfaces since the camera (lets lights be seen in reflections/refractions)
        uint events = 0;
        if (HAS_WATER) in_water = water_contains(lookfrom, water);

        uint depth = 0;
        while (depth < max_vertices) {
            auto hit = inter.intersect(cur, accel_struct);
            bool got = hit.type != intersection_type::none;
            float t_s = got ? hit.distance : INFINITY;
            if (HAS_WATER) {
                float t_w = 0.0f; float3 n_w = float3(0.0f);
                bool wev = water_interface(cur.origin, cur.direction, in_water, water, t_w, n_w) && t_w < t_s;
                if (wev && !in_water && water.cutaway != 0u && fabs(n_w.y) < 0.5f) {      // side wall: only where the point is above the terrain
                    ray up; up.origin = cur.origin + cur.direction * t_w; up.direction = float3(0, 1, 0); up.min_distance = 1e-3f; up.max_distance = INFINITY;
                    inter.accept_any_intersection(true);
                    if (inter.intersect(up, accel_struct).type != intersection_type::none) wev = false;
                    inter.accept_any_intersection(false);
                }
                float seg = wev ? t_w : t_s;
                if (spec_chain) {                                                       // sun glints on the water; the sun seen from below
                    for (uint k4 = 0; k4 < K; ++k4) {
                        float te;
                        if (ray_sphere(cur.origin, cur.direction, float3(lights[k4].position), lights[k4].radius, te) && te < seg) {
                            float3 Tte = in_water ? exp(-w_sigma_t(water) * te) : float3(1.0f);
                            accum[k4] += min(throughput * Tte * float3(lights[k4].emission), float3(cam.clamp_val));
                        }
                    }
                }
                if (in_water) {
                    if (!isfinite(seg)) break;
                    float3 st = w_sigma_t(water);
                    if (w_any_scatter(water)) {                                         // single scattering: one shared point along the segment for all lights
                        float s_len = rng.next_float() * seg;
                        float3 q = cur.origin + cur.direction * s_len, Ts = exp(-st * s_len), ss = w_sigma_s(water);
                        float2 u_sc = rng.next_float2();
                        for (uint k5 = 0; k5 < K; ++k5) {
                            float3 Li = eval_light_scatter(q, lights[k5], u_sc, accel_struct, water);
                            accum[k5] += min(throughput * Ts * ss * (0.25f * M_1_PI_F) * Li * seg, float3(cam.clamp_val));
                        }
                    }
                    throughput *= exp(-st * seg);
                    if (max(throughput.x, max(throughput.y, throughput.z)) < 1e-5f) break;
                }
                if (wev) {                                                              // dielectric interface: reflect with probability F, else refract
                    if (++events > 8u) break;
                    float3 d = cur.direction;
                    float3 N = n_w; if (dot(N, d) > 0.0f) N = -N;
                    N = perturb_normal(N, water.roughness, rng.next_float2());
                    if (dot(N, d) > 0.0f) N = n_w * (dot(n_w, d) > 0.0f ? -1.0f : 1.0f);
                    float cosi = clamp(-dot(N, d), 0.0f, 1.0f);
                    float n_i = in_water ? water.ior : 1.0f, n_t = in_water ? 1.0f : water.ior;
                    float3 dn;
                    if (rng.next_float() < fresnel_dielectric(cosi, n_i, n_t)) dn = reflect(d, N);
                    else { dn = refract(d, N, n_i / n_t); in_water = !in_water; }
                    cur.origin = cur.origin + d * t_w + dn * 2e-3f;
                    cur.direction = dn;
                    cur.min_distance = 1e-3f;
                    cur.max_distance = INFINITY;
                    spec_chain = true;
                    continue;
                }
            }
            if (!got) break;

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
                float3 Ld = throughput * eval_light_nee(p, n, rho, lights[k2], u_nee, accel_struct, water);
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
            spec_chain = false;
            ++depth;
        }
    }

    float inv_s = 1.0f / float(max(cam.samples, 1u));
    for (uint k3 = 0; k3 < K; ++k3) {
        out_layers[k3 * num_pixels + pix_idx] = packed_float3(accum[k3] * inv_s);
    }
}
