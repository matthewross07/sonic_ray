// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
// sonicray GPU compositor host library: a tiny C ABI over Metal compute, driven from Python (ctypes) or anything else.
//
//   sc_init(metal_source_path)            compile the kernels at runtime (works with Command Line Tools only)
//   sc_mmap(path)                         map a read-only layer file straight into a GPU buffer (no copy, shared page cache)
//   sc_alloc(bytes) / sc_ptr(handle)      shared-memory working buffers the CPU can also read and write
//   sc_begin / sc_op(...) / sc_end        encode a frame's worth of kernels into one command buffer, then commit + wait
//
// Build: compositor/build.sh   (clang++ -ObjC++, -framework Metal -framework Foundation)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_lib = nil;
static NSMutableArray<id<MTLBuffer>>* g_bufs = nil;
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* g_pso = nil;
static id<MTLCommandBuffer> g_cmd = nil;
static std::string g_err;

static int fail(const std::string& m) { g_err = m; return -1; }

extern "C" {

const char* sc_last_error(void) { return g_err.c_str(); }

int sc_init(const char* metal_path) {
  @autoreleasepool {
    if (g_dev) return 0;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) return fail("no Metal device");
    g_queue = [g_dev newCommandQueue];
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:metal_path] encoding:NSUTF8StringEncoding error:&err];
    if (!src) return fail(std::string("cannot read ") + metal_path);
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.languageVersion = MTLLanguageVersion3_0;
    g_lib = [g_dev newLibraryWithSource:src options:opt error:&err];
    if (!g_lib) return fail(std::string("Metal compile failed: ") + [[err localizedDescription] UTF8String]);
    g_bufs = [NSMutableArray new];
    g_pso = [NSMutableDictionary new];
    return 0;
  }
}

const char* sc_device_name(void) { return g_dev ? [[g_dev name] UTF8String] : ""; }

// read-only layer file -> GPU buffer without copying (private COW mapping; never written)
int sc_mmap(const char* path) {
  @autoreleasepool {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(std::string("open failed: ") + path);
    struct stat st; fstat(fd, &st);
    size_t pg = (size_t)getpagesize(), len = ((size_t)st.st_size + pg - 1) / pg * pg;
    const char* mode = getenv("SONICRAY_MMAP");            // experiments: "copy" | "shared" | (default) private
    id<MTLBuffer> b = nil;
    if (mode && std::string(mode) == "copy") {
      void* q = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      if (q == MAP_FAILED) return fail(std::string("mmap failed: ") + path);
      b = [g_dev newBufferWithLength:len options:MTLResourceStorageModeShared];
      if (!b) return fail(std::string("buffer allocation failed: ") + path);
      memcpy([b contents], q, (size_t)st.st_size);
      munmap(q, len);
    } else {
      bool sh = mode && std::string(mode) == "shared";
      void* p = mmap(nullptr, len, sh ? PROT_READ : (PROT_READ | PROT_WRITE), sh ? MAP_SHARED : MAP_PRIVATE, fd, 0);
      close(fd);
      if (p == MAP_FAILED) return fail(std::string("mmap failed: ") + path);
      b = [g_dev newBufferWithBytesNoCopy:p length:len options:MTLResourceStorageModeShared deallocator:nil];
      if (!b) return fail(std::string("newBufferWithBytesNoCopy failed: ") + path);
    }
    [g_bufs addObject:b];
    return (int)[g_bufs count] - 1;
  }
}

int sc_alloc(size_t bytes) {
  @autoreleasepool {
    id<MTLBuffer> b = [g_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) return fail("buffer allocation failed");
    memset([b contents], 0, bytes);
    [g_bufs addObject:b];
    return (int)[g_bufs count] - 1;
  }
}

void* sc_ptr(int h) { return [g_bufs[h] contents]; }

int sc_begin(void) {
  g_cmd = [g_queue commandBuffer];
  return g_cmd ? 0 : fail("commandBuffer failed");
}

// Dispatch kernel `name` over a gw x gh grid with `nb` buffers bound at 0..nb-1 and `params` bound at index nb.
int sc_op(const char* name, const int* bufs, int nb, const void* params, int plen, int gw, int gh) {
  @autoreleasepool {
    NSString* key = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> pso = g_pso[key];
    if (!pso) {
      id<MTLFunction> fn = [g_lib newFunctionWithName:key];
      if (!fn) return fail(std::string("no kernel ") + name);
      NSError* err = nil;
      pso = [g_dev newComputePipelineStateWithFunction:fn error:&err];
      if (!pso) return fail(std::string("pipeline failed for ") + name + ": " + [[err localizedDescription] UTF8String]);
      g_pso[key] = pso;
    }
    id<MTLComputeCommandEncoder> enc = [g_cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (int i = 0; i < nb; ++i) [enc setBuffer:g_bufs[bufs[i]] offset:0 atIndex:i];
    if (plen > 0) [enc setBytes:params length:plen atIndex:nb];
    [enc dispatchThreads:MTLSizeMake(gw, gh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    return 0;
  }
}

int sc_end(void) {
  @autoreleasepool {
    [g_cmd commit];
    [g_cmd waitUntilCompleted];
    if ([g_cmd status] != MTLCommandBufferStatusCompleted) {
      std::string m = "command buffer failed";
      if ([g_cmd error]) m += std::string(": ") + [[[g_cmd error] localizedDescription] UTF8String];
      g_cmd = nil;
      return fail(m);
    }
    g_cmd = nil;
    return 0;
  }
}

}  // extern "C"
