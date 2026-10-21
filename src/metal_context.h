// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
// =============================================================================
// sonicray/src/metal_context.h
// Shared C++ / Objective-C++ host structures matching sonicray_pathtracer.metal
// =============================================================================
#pragma once
#include <cstdint>
#include <vector>
#include <string>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

struct PackedFloat2 {
    float x, y;
};

struct PackedFloat3 {
    float x, y, z;
};

struct PackedUint3 {
    uint32_t x, y, z;
};

// Matches MSL CameraUniforms (16-byte aligned blocks where required)
struct HostCameraUniforms {
    PackedFloat3 lookfrom;
    float        fov_rad;
    PackedFloat3 cam_forward;
    float        aspect;
    PackedFloat3 cam_right;
    uint32_t     width;
    PackedFloat3 cam_up;
    uint32_t     height;
    uint32_t     samples;
    uint32_t     max_bounces;
    uint32_t     seed_offset;
    float        clamp_val;
    PackedFloat3 sky_high;
    float        ambient_intensity;
    PackedFloat3 sky_low;
    uint32_t     num_lights;
};

struct HostLightSpec {
    PackedFloat3 position;
    float        radius;
    PackedFloat3 emission;
    uint32_t     light_type;
};

// Matches MSL WaterUniforms (16 floats). Coefficients are per scene unit.
struct HostWaterUniforms {
    float    level;
    float    ior;
    float    roughness;
    uint32_t cutaway;
    float    sigma_t[3];
    float    sigma_s[3];
    float    bmin[3];
    float    bmax[3];
};
static_assert(sizeof(HostWaterUniforms) == 64, "HostWaterUniforms must match the Metal struct");

#ifdef __OBJC__
// Persistent GPU Scene held in an Rcpp::XPtr<SonicrayScene> across all layer renders
struct SonicrayScene {
    id<MTLDevice>                 device;
    id<MTLCommandQueue>           queue;
    id<MTLLibrary>                library;
    id<MTLComputePipelineState>   amb_pipeline;
    id<MTLComputePipelineState>   multilight_pipeline;
    id<MTLAccelerationStructure>  accel_struct;
    id<MTLBuffer>                 vertex_buf;
    id<MTLBuffer>                 normal_buf;
    id<MTLBuffer>                 uv_buf;
    id<MTLBuffer>                 index_buf;
    id<MTLTexture>                albedo_tex;
    id<MTLTexture>                river_tex;
    uint32_t                      num_vertices;
    uint32_t                      num_triangles;
    bool                          has_water = false;
    HostWaterUniforms             water{};
};
#endif
