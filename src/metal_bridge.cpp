// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
// =============================================================================
// sonicray/src/metal_bridge.mm
// Runtime Metal 3 Shader Compiler, Hardware BVH Acceleration Structure Builder,
// and Batched Multi-Light Path-Tracing Dispatcher for R.
// =============================================================================
#include <Rcpp.h>
#include "metal_context.h"

using namespace Rcpp;

// Helper to build a hardware MTLAccelerationStructure once on the GPU
static id<MTLAccelerationStructure> build_triangle_bvh(
    id<MTLDevice>       device,
    id<MTLCommandQueue> queue,
    id<MTLBuffer>       vertex_buf,
    uint32_t            num_vertices,
    id<MTLBuffer>       index_buf,
    uint32_t            num_triangles
) {
    MTLAccelerationStructureTriangleGeometryDescriptor* geom =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geom.vertexBuffer = vertex_buf;
    geom.vertexStride = sizeof(PackedFloat3);
    // (MTLAccelerationStructureTriangleGeometryDescriptor has no vertexCount: the vertex count is implied by the index buffer)
    (void)num_vertices;
    geom.indexBuffer  = index_buf;
    geom.indexType    = MTLIndexTypeUInt32;
    geom.triangleCount = num_triangles;
    geom.opaque       = YES;

    MTLPrimitiveAccelerationStructureDescriptor* desc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    desc.geometryDescriptors = @[geom];

    MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:desc];
    id<MTLAccelerationStructure> accel = [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch = [device newBufferWithLength:sizes.buildScratchBufferSize
                                                options:MTLResourceStorageModePrivate];

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> enc = [cmd accelerationStructureCommandEncoder];
    [enc buildAccelerationStructure:accel descriptor:desc scratchBuffer:scratch scratchBufferOffset:0];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return accel;
}

// [[Rcpp::export]]
List cpp_metal_device_info() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        return List::create(
            _["available"] = false,
            _["name"] = "None"
        );
    }
    return List::create(
        _["available"] = true,
        _["name"] = std::string([[dev name] UTF8String]),
        _["supports_raytracing"] = (bool)[dev supportsRaytracing],
        _["unified_memory"] = (bool)[dev hasUnifiedMemory],
        _["recommended_max_working_set_gb"] = (double)[dev recommendedMaxWorkingSetSize] / (1024.0 * 1024.0 * 1024.0)
    );
}

// [[Rcpp::export]]
SEXP cpp_upload_scene(
    NumericMatrix vertices,    // N_v x 3
    NumericMatrix normals,     // N_v x 3
    NumericMatrix uvs,         // N_v x 2
    IntegerMatrix indices,     // N_t x 3 (0-indexed)
    NumericVector albedo_rgba, // W_tex * H_tex * 4 (linear float [0,1])
    int           tex_w,
    int           tex_h,
    NumericVector river_rg,    // W_tex * H_tex * 2 (linear float [0,1])
    std::string   metal_shader_src,
    NumericVector water = NumericVector::create()   // empty = no water; else (level, ior, roughness, cutaway, sigma_t[3], sigma_s[3]) per scene unit
) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev || ![dev supportsRaytracing]) {
        stop("sonicray requires an Apple Silicon GPU with Metal Ray Tracing support.");
    }

    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"      // fastMathEnabled is only used on macOS < 15
    if (@available(macOS 15.0, *)) { opts.mathMode = MTLMathModeFast; } else { opts.fastMathEnabled = YES; }
#pragma clang diagnostic pop
    NSString* src_ns = [NSString stringWithUTF8String:metal_shader_src.c_str()];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src_ns options:opts error:&err];
    if (!lib) {
        std::string msg = err ? [[err localizedDescription] UTF8String] : "Unknown Metal compile error";
        stop("Failed to compile sonicray Metal shader at runtime:\n" + msg);
    }

    // Water code is compiled in only when the scene has water (a Metal function constant), so dry scenes pay nothing for it.
    const bool has_water = water.size() >= 10;
    MTLFunctionConstantValues* fc = [[MTLFunctionConstantValues alloc] init];
    [fc setConstantValue:&has_water type:MTLDataTypeBool atIndex:0];
    id<MTLFunction> fn_amb = [lib newFunctionWithName:@"render_ambient_and_aovs_kernel" constantValues:fc error:&err];
    id<MTLFunction> fn_ml  = [lib newFunctionWithName:@"render_multilight_batch_kernel" constantValues:fc error:&err];
    if (!fn_amb || !fn_ml) stop(std::string("Failed to specialise the Metal kernels: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
    id<MTLComputePipelineState> p_amb = [dev newComputePipelineStateWithFunction:fn_amb error:&err];
    id<MTLComputePipelineState> p_ml  = [dev newComputePipelineStateWithFunction:fn_ml error:&err];
    if (!p_amb || !p_ml) {
        std::string msg = err ? [[err localizedDescription] UTF8String] : "Pipeline creation error";
        stop("Failed to create Metal compute pipeline:\n" + msg);
    }

    uint32_t nv = vertices.nrow();
    uint32_t nt = indices.nrow();
    std::vector<PackedFloat3> v_vec(nv), n_vec(nv);
    std::vector<PackedFloat2> uv_vec(nv);
    for (uint32_t i = 0; i < nv; ++i) {
        v_vec[i]  = {(float)vertices(i, 0), (float)vertices(i, 1), (float)vertices(i, 2)};
        n_vec[i]  = {(float)normals(i, 0),  (float)normals(i, 1),  (float)normals(i, 2)};
        uv_vec[i] = {(float)uvs(i, 0),      (float)uvs(i, 1)};
    }
    std::vector<PackedUint3> idx_vec(nt);
    for (uint32_t i = 0; i < nt; ++i) {
        idx_vec[i] = {(uint32_t)indices(i, 0), (uint32_t)indices(i, 1), (uint32_t)indices(i, 2)};
    }

    id<MTLBuffer> v_buf   = [dev newBufferWithBytes:v_vec.data()   length:nv * sizeof(PackedFloat3) options:MTLResourceStorageModeShared];
    id<MTLBuffer> n_buf   = [dev newBufferWithBytes:n_vec.data()   length:nv * sizeof(PackedFloat3) options:MTLResourceStorageModeShared];
    id<MTLBuffer> uv_buf  = [dev newBufferWithBytes:uv_vec.data()  length:nv * sizeof(PackedFloat2) options:MTLResourceStorageModeShared];
    id<MTLBuffer> idx_buf = [dev newBufferWithBytes:idx_vec.data() length:nt * sizeof(PackedUint3)  options:MTLResourceStorageModeShared];

    // Linear RGBA16Float / RG16Float textures (avoids 8-bit sRGB gamma distortion completely)
    MTLTextureDescriptor* alb_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                        width:tex_w
                                                                                       height:tex_h
                                                                                    mipmapped:NO];
    id<MTLTexture> alb_tex = [dev newTextureWithDescriptor:alb_desc];
    std::vector<float> alb_f32(albedo_rgba.begin(), albedo_rgba.end());
    [alb_tex replaceRegion:MTLRegionMake2D(0, 0, tex_w, tex_h)
               mipmapLevel:0
                 withBytes:alb_f32.data()
               bytesPerRow:tex_w * 4 * sizeof(float)];

    MTLTextureDescriptor* riv_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float
                                                                                        width:tex_w
                                                                                       height:tex_h
                                                                                    mipmapped:NO];
    id<MTLTexture> riv_tex = [dev newTextureWithDescriptor:riv_desc];
    std::vector<float> riv_f32(river_rg.begin(), river_rg.end());
    [riv_tex replaceRegion:MTLRegionMake2D(0, 0, tex_w, tex_h)
               mipmapLevel:0
                 withBytes:riv_f32.data()
               bytesPerRow:tex_w * 2 * sizeof(float)];

    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLAccelerationStructure> bvh = build_triangle_bvh(dev, queue, v_buf, nv, idx_buf, nt);

    SonicrayScene* sc = new SonicrayScene{
        dev, queue, lib, p_amb, p_ml, bvh,
        v_buf, n_buf, uv_buf, idx_buf, alb_tex, riv_tex, nv, nt
    };
    sc->has_water = has_water;
    if (has_water) {
        HostWaterUniforms& w = sc->water;
        w.level = (float)water[0]; w.ior = (float)water[1]; w.roughness = (float)water[2]; w.cutaway = water[3] != 0.0 ? 1u : 0u;
        for (int i = 0; i < 3; ++i) { w.sigma_t[i] = (float)water[4 + i]; w.sigma_s[i] = (float)water[7 + i]; }
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};       // terrain bounds (used by the cutaway box)
        for (uint32_t i = 0; i < nv; ++i) {
            float c[3] = {v_vec[i].x, v_vec[i].y, v_vec[i].z};
            for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], c[k]); hi[k] = std::max(hi[k], c[k]); }
        }
        for (int k = 0; k < 3; ++k) { w.bmin[k] = lo[k] - (k == 1 ? 1e-3f : 0.0f); w.bmax[k] = hi[k]; }
    }
    return XPtr<SonicrayScene>(sc, true);
}

// Helper to compute camera basis vectors matching rayrender / sonicray convention
static HostCameraUniforms make_host_camera(
    NumericVector lookfrom,
    NumericVector lookat,
    double        fov_deg,
    int           width,
    int           height,
    int           samples,
    int           max_bounces,
    int           seed_offset,
    double        clamp_val,
    NumericVector sky_high,
    NumericVector sky_low,
    double        ambient_intensity,
    int           num_lights
) {
    double fx = lookat[0] - lookfrom[0];
    double fy = lookat[1] - lookfrom[1];
    double fz = lookat[2] - lookfrom[2];
    double flen = std::sqrt(fx * fx + fy * fy + fz * fz);
    fx /= flen; fy /= flen; fz /= flen;

    // Cross(f, (0, 1, 0)) -> right = (-fz, 0, fx)
    double rx = -fz, ry = 0.0, rz = fx;
    double rlen = std::sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rlen; ry /= rlen; rz /= rlen;

    // Cross(r, f) -> up
    double ux = ry * fz - rz * fy;
    double uy = rz * fx - rx * fz;
    double uz = rx * fy - ry * fx;

    HostCameraUniforms cam{};
    cam.lookfrom          = {(float)lookfrom[0], (float)lookfrom[1], (float)lookfrom[2]};
    cam.fov_rad           = (float)(fov_deg * M_PI / 180.0);
    cam.cam_forward       = {(float)fx, (float)fy, (float)fz};
    cam.aspect            = (float)width / (float)height;
    cam.cam_right         = {(float)rx, (float)ry, (float)rz};
    cam.width             = (uint32_t)width;
    cam.cam_up            = {(float)ux, (float)uy, (float)uz};
    cam.height            = (uint32_t)height;
    cam.samples           = (uint32_t)samples;
    cam.max_bounces       = (uint32_t)max_bounces;
    cam.seed_offset       = (uint32_t)seed_offset;
    cam.clamp_val         = (float)clamp_val;
    cam.sky_high          = {(float)sky_high[0], (float)sky_high[1], (float)sky_high[2]};
    cam.ambient_intensity = (float)ambient_intensity;
    cam.sky_low           = {(float)sky_low[0],  (float)sky_low[1],  (float)sky_low[2]};
    cam.num_lights        = (uint32_t)num_lights;
    return cam;
}

// [[Rcpp::export]]
List cpp_render_ambient_and_aovs(
    SEXP          scene_xptr,
    NumericVector lookfrom,
    NumericVector lookat,
    double        fov_deg,
    int           width,
    int           height,
    int           samples,
    double        clamp_val
) {
    XPtr<SonicrayScene> sc(scene_xptr);
    size_t npix = (size_t)width * (size_t)height;

    HostCameraUniforms cam = make_host_camera(
        lookfrom, lookat, fov_deg, width, height, samples, 1, 1337, clamp_val,
        NumericVector::create(0.043, 0.094, 0.251),
        NumericVector::create(0.043, 0.094, 0.251),
        1.0, 0
    );

    id<MTLBuffer> b_amb  = [sc->device newBufferWithLength:npix * sizeof(PackedFloat3) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_mask = [sc->device newBufferWithLength:npix * sizeof(float)        options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_alb  = [sc->device newBufferWithLength:npix * sizeof(PackedFloat3) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_nrm  = [sc->device newBufferWithLength:npix * sizeof(PackedFloat3) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_uv   = [sc->device newBufferWithLength:npix * sizeof(PackedFloat2) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_riv  = [sc->device newBufferWithLength:npix * sizeof(PackedFloat2) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_wdep = [sc->device newBufferWithLength:npix * sizeof(float)        options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [sc->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:sc->amb_pipeline];
    [enc setBytes:&cam length:sizeof(HostCameraUniforms) atIndex:0];
    [enc setAccelerationStructure:sc->accel_struct atBufferIndex:1];
    [enc setBuffer:sc->vertex_buf offset:0 atIndex:2];
    [enc setBuffer:sc->normal_buf offset:0 atIndex:3];
    [enc setBuffer:sc->uv_buf     offset:0 atIndex:4];
    [enc setBuffer:sc->index_buf  offset:0 atIndex:5];
    [enc setBuffer:b_amb          offset:0 atIndex:6];
    [enc setBuffer:b_mask         offset:0 atIndex:7];
    [enc setBuffer:b_alb          offset:0 atIndex:8];
    [enc setBuffer:b_nrm          offset:0 atIndex:9];
    [enc setBuffer:b_uv           offset:0 atIndex:10];
    [enc setBuffer:b_riv          offset:0 atIndex:11];
    [enc setBytes:&sc->water length:sizeof(HostWaterUniforms) atIndex:12];
    [enc setBuffer:b_wdep         offset:0 atIndex:13];
    [enc setTexture:sc->albedo_tex atIndex:0];
    [enc setTexture:sc->river_tex  atIndex:1];

    MTLSize grid = MTLSizeMake(width, height, 1);
    MTLSize tg   = MTLSizeMake(8, 8, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    const float* p_amb  = (const float*)[b_amb contents];
    const float* p_mask = (const float*)[b_mask contents];
    const float* p_alb  = (const float*)[b_alb contents];
    const float* p_nrm  = (const float*)[b_nrm contents];
    const float* p_uv   = (const float*)[b_uv contents];
    const float* p_riv  = (const float*)[b_riv contents];
    const float* p_wdep = (const float*)[b_wdep contents];

    return List::create(
        _["amb"]    = NumericVector(p_amb,  p_amb  + npix * 3),
        _["mask"]   = NumericVector(p_mask, p_mask + npix),
        _["albedo"] = NumericVector(p_alb,  p_alb  + npix * 3),
        _["normal"] = NumericVector(p_nrm,  p_nrm  + npix * 3),
        _["uv"]     = NumericVector(p_uv,   p_uv   + npix * 2),
        _["riv"]    = NumericVector(p_riv,  p_riv  + npix * 2),
        _["water_depth"] = NumericVector(p_wdep, p_wdep + npix)
    );
}

// [[Rcpp::export]]
NumericVector cpp_render_multilight_batch(
    SEXP          scene_xptr,
    NumericVector lookfrom,
    NumericVector lookat,
    double        fov_deg,
    int           width,
    int           height,
    int           samples,
    int           max_bounces,
    double        clamp_val,
    NumericMatrix light_pos_r_intensity // K x 5: (x, y, z, radius, intensity)
) {
    XPtr<SonicrayScene> sc(scene_xptr);
    int K = light_pos_r_intensity.nrow();
    if (K <= 0 || K > 32) stop("Batch size K must be in 1..32.");
    size_t npix = (size_t)width * (size_t)height;

    HostCameraUniforms cam = make_host_camera(
        lookfrom, lookat, fov_deg, width, height, samples, max_bounces, 4242, clamp_val,
        NumericVector::create(0, 0, 0), NumericVector::create(0, 0, 0), 0.0, K
    );

    std::vector<HostLightSpec> lspecs(K);
    for (int k = 0; k < K; ++k) {
        float I = (float)light_pos_r_intensity(k, 4);
        lspecs[k].position   = {(float)light_pos_r_intensity(k, 0), (float)light_pos_r_intensity(k, 1), (float)light_pos_r_intensity(k, 2)};
        lspecs[k].radius     = (float)light_pos_r_intensity(k, 3);
        lspecs[k].emission   = {I, I, I};
        lspecs[k].light_type = 0;
    }

    id<MTLBuffer> b_lights = [sc->device newBufferWithBytes:lspecs.data()
                                                     length:K * sizeof(HostLightSpec)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_out    = [sc->device newBufferWithLength:(size_t)K * npix * sizeof(PackedFloat3)
                                                     options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [sc->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:sc->multilight_pipeline];
    [enc setBytes:&cam length:sizeof(HostCameraUniforms) atIndex:0];
    [enc setAccelerationStructure:sc->accel_struct atBufferIndex:1];
    [enc setBuffer:sc->vertex_buf offset:0 atIndex:2];
    [enc setBuffer:sc->normal_buf offset:0 atIndex:3];
    [enc setBuffer:sc->uv_buf     offset:0 atIndex:4];
    [enc setBuffer:sc->index_buf  offset:0 atIndex:5];
    [enc setBuffer:b_lights       offset:0 atIndex:6];
    [enc setBuffer:b_out          offset:0 atIndex:7];
    [enc setBytes:&sc->water length:sizeof(HostWaterUniforms) atIndex:8];
    [enc setTexture:sc->albedo_tex atIndex:0];

    MTLSize grid = MTLSizeMake(width, height, 1);
    MTLSize tg   = MTLSizeMake(8, 8, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    const float* p_out = (const float*)[b_out contents];
    return NumericVector(p_out, p_out + (size_t)K * npix * 3);
}
