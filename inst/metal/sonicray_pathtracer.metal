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
// Renders the ambient sky-fill layer AND outputs noise-free guide buffers for Intel OIDN
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
    if (dist <= 1e-4f) return float3(0.0f);

    // Sample point on sphere light
    float3 t, b;
    float3 dir_c = to_center / dist;
    build_onb(dir_c, t, b);
    float r_disk = light.radius * sqrt(u_rand.x);
    float phi = 2.0f * M_PI_F * u_rand.y;
    float3 sample_pt = float3(light.position) + r_disk * (cos(phi) * t + sin(phi) * b);

    float3 L_vec = sample_pt - p;
    float L_dist = length(L_vec);
    float3 wi = L_vec / max(L_dist, 1e-4f);
    float cos_theta = dot(n, wi);
    if (cos_theta <= 0.0f) return float3(0.0f);

    // Hardware shadow ray (early-exit on any hit)
    intersector<triangle_data> shadow_inter;
    shadow_inter.assume_geometry_type(geometry_type::triangle);
    shadow_inter.force_opacity(forced_opacity::opaque);
    shadow_inter.accept_any_intersection(true);

    ray shadow_ray;
    shadow_ray.origin = p + n * 1.5e-3f;
    shadow_ray.direction = wi;
    shadow_ray.min_distance = 1e-3f;
    shadow_ray.max_distance = max(L_dist - light.radius * 0.25f, 2e-3f);

    auto shadow_hit = shadow_inter.intersect(shadow_ray, accel_struct);
    if (shadow_hit.type != intersection_type::none) {
        return float3(0.0f);
    }

    // Solid angle & Lambertian BRDF (rho / pi)
    float sin_alpha_max_sq = clamp((light.radius * light.radius) / max(dist_sq, 1e-4f), 0.0f, 0.999f);
    float omega = 2.0f * M_PI_F * (1.0f - sqrt(1.0f - sin_alpha_max_sq));
    return (rho * M_1_PI_F) * float3(light.emission) * (cos_theta * omega);
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

    for (uint s = 0; s < cam.samples; ++s) {
        RNG rng = make_rng(gid, s, cam.seed_offset);
        float2 jitter = rng.next_float2();
        float ndc_x = ((float(gid.x) + jitter.x) / float(cam.width)) * 2.0f - 1.0f;
        float ndc_y = 1.0f - ((float(gid.y) + jitter.y) / float(cam.height)) * 2.0f;

        ray r0;
        r0.origin = lookfrom;
        r0.direction = normalize(
            cam_f + cam_r * (ndc_x * tan_half_fov * cam.aspect) + cam_u * (ndc_y * tan_half_fov)
        );
        r0.min_distance = 1e-3f;
        r0.max_distance = INFINITY;

        auto hit0 = inter.intersect(r0, accel_struct);
        if (hit0.type == intersection_type::none) continue;

        uint3 tri0 = uint3(indices[hit0.primitive_id]);
        float2 b0 = hit0.triangle_barycentric_coord;
        float w0 = 1.0f - b0.x - b0.y;
        float3 n0 = normalize(w0 * float3(normals[tri0.x]) + b0.x * float3(normals[tri0.y]) + b0.y * float3(normals[tri0.z]));
        if (dot(n0, r0.direction) > 0.0f) n0 = -n0;
        float2 uv0 = w0 * float2(uvs[tri0.x]) + b0.x * float2(uvs[tri0.y]) + b0.y * float2(uvs[tri0.z]);
        float3 rho0 = albedo_tex.sample(lin_sampler, float2(uv0.x, 1.0f - uv0.y)).rgb;
        float3 p0 = r0.origin + r0.direction * hit0.distance;

        // 1. Direct NEE across all K lights in the batch (sharing primary ray hit!)
        float2 u_nee0 = rng.next_float2();
        for (uint k = 0; k < K; ++k) {
            float3 L_dir = eval_light_nee(p0, n0, rho0, lights[k], u_nee0, accel_struct);
            accum[k] += min(L_dir, float3(cam.clamp_val));
        }

        // 2. Shared 1-bounce diffuse interreflection across all K lights
        if (cam.max_bounces >= 2) {
            float3 dir1 = sample_cosine_hemisphere(rng.next_float2(), n0);
            ray r1;
            r1.origin = p0 + n0 * 1.5e-3f;
            r1.direction = dir1;
            r1.min_distance = 1e-3f;
            r1.max_distance = 28.0f; // Local canyon/valley bounce radius (km)

            auto hit1 = inter.intersect(r1, accel_struct);
            if (hit1.type != intersection_type::none) {
                uint3 tri1 = uint3(indices[hit1.primitive_id]);
                float2 b1 = hit1.triangle_barycentric_coord;
                float w1_0 = 1.0f - b1.x - b1.y;
                float3 n1 = normalize(w1_0 * float3(normals[tri1.x]) + b1.x * float3(normals[tri1.y]) + b1.y * float3(normals[tri1.z]));
                if (dot(n1, r1.direction) > 0.0f) n1 = -n1;
                float2 uv1 = w1_0 * float2(uvs[tri1.x]) + b1.x * float2(uvs[tri1.y]) + b1.y * float2(uvs[tri1.z]);
                float3 rho1 = albedo_tex.sample(lin_sampler, float2(uv1.x, 1.0f - uv1.y)).rgb;
                float3 p1 = r1.origin + r1.direction * hit1.distance;

                float2 u_nee1 = rng.next_float2();
                for (uint k = 0; k < K; ++k) {
                    float3 L_indir = rho0 * eval_light_nee(p1, n1, rho1, lights[k], u_nee1, accel_struct);
                    accum[k] += min(L_indir, float3(cam.clamp_val * 0.5f));
                }
            }
        }
    }

    float inv_s = 1.0f / float(max(cam.samples, 1u));
    for (uint k = 0; k < K; ++k) {
        out_layers[k * num_pixels + pix_idx] = packed_float3(accum[k] * inv_s);
    }
}
