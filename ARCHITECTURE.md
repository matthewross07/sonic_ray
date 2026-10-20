# `sonicray` Architecture & Engineering Blueprint

This document synthesizes the original `raymetal` spike plan (`visuals/ideas/metal_gpu_rayrender_plan.md`) with the production lessons learned building the Denver Water (`denver_3d_v2`) and Cache la Poudre (`poudre_3d_v2`) 3D network films.

---

## 1. Why a Flattened Metal 3 Engine Instead of Porting `rayrender` C++

`rayrender` is an expressive CPU path tracer built around object-oriented C++ (`hitable` base classes, virtual `hit()` and `scattering_pdf()` methods, heap-allocated `std::shared_ptr<material>`, and recursive ray bounces).

Apple's Metal Shading Language (MSL) executes on massively parallel SIMD-groups (32 threads/wavefront) where:
1. **Virtual functions and heap pointers do not exist** (or tank occupancy via function tables).
2. **BVH construction and ray-triangle traversal are built directly into silicon** on Apple M3/M4/M5 GPUs (`MTLPrimitiveAccelerationStructureDescriptor` + `metal::raytracing::intersector<triangle_data>`), and run via fast compute-shader BVH hardware on M1/M2.
3. **Unified Memory (`MTLStorageModeShared`)** allows R's C++ layer to populate contiguous `float3` / `uint3` buffers once and hand them to the GPU with **zero PCIe copy overhead**.

### Scene Flattening Contract
Instead of an R `tibble` of thousands of `rayrender` objects, `sonicray` flattens a scene into five GPU buffers held inside an `Rcpp::XPtr<SonicrayScene>`:
- `vertices` (`float3[N_v]`): world-space $(x, y, z)$ coordinates in km.
- `normals` (`float3[N_v]`): smooth vertex normals (computed once on upload—avoiding `rayrender`'s `recalculate_normals = TRUE` segfault when paired with image textures).
- `uvs` (`float2[N_v]`): normalized $(u, v) \in [0, 1]^2$ texture coordinates.
- `indices` (`uint3[N_t]`): triangle vertex index triplets.
- `textures`:
  - `albedo_tex` (`MTLPixelFormatRGBA8Unorm_sRGB` or `RGBA16Float`): hardware-linearized terrain color.
  - `river_tex` (`MTLPixelFormatRGBA8Unorm`): linear-space river/reservoir mask (Channel 0 = highlighted mainstem, Channel 1 = all NHD rivers/lakes, avoiding `rayrender`'s sRGB $x^{2.2}$ gamma distortion on data masks).

---

## 2. Native Multi-Light Layer Path Tracing (Wavefront / Array NEE)

### The Linearity Principle
Because light transport is linear, a 3,900-frame film with $M \approx 150$ dynamic lights (fireworks, sonde beacons, WQP cells, sun elevations) can be factored as:
$$I(x, y, t) = w_{\text{amb}}(t)\,L_{\text{amb}}(x, y) + \sum_{m=1}^{M} w_m(t)\,\mathbf{c}_m \odot L_m(x, y)$$
where each layer $L_m(x, y)$ is the path-traced illumination of **unit-white light $m$ alone** on the terrain.

### Why CPU RGB Multiplexing Was Only a Half-Step
On CPU (`rayrender`), rendering $M = 150$ lights individually required 150 full renders. We cut that by $3\times$ by packing 3 lights per render as pure Red `#ff0000`, Green `#00ff00`, and Blue `#0000ff` emitters and reconstructing the missing channels from an emissive albedo pass (`I[, , k] * (A[, , c] + eps) / (A[, , k] + eps)`).
- **Limitation 1**: Still required $\lceil 150 / 3 \rceil = 50$ separate CPU path-tracing passes (each rebuilding the BVH).
- **Limitation 2**: Introduced a 5–10% color bias whenever 2nd-bounce indirect light hit adjacent canyon walls of a different color.

### How `sonicray` Solves It on GPU (`batch_size = 16` or `32`)
In `inst/metal/sonicray_pathtracer.metal`, the kernel `pathtrace_multilight_kernel` accepts a batch of $K \le 32$ lights (`constant LightSpec* lights`, `uint num_lights`):
1. **Primary Ray + First-Hit Shading (Shared Across All $K$ Lights)**:
   - Each pixel traces its camera ray against the hardware `primitive_acceleration_structure` **once per sample**.
   - Interpolates smooth normal $\mathbf{n}$, UV $(u, v)$, and samples linear `albedo` $\boldsymbol{\rho}_0$.
2. **Direct Lighting Next-Event Estimation (NEE) Loop Over $K$ Lights**:
   - For each light $k \in \{0, \dots, K-1\}$ whose bounding sphere is within cutoff range of hit point $\mathbf{p}$:
     - Samples a point on sphere light $k$ using solid-angle importance sampling.
     - Traces a fast `accept_any_intersection(true)` shadow ray (`intersector.intersect(shadow_ray, accel_struct)`).
     - If unoccluded, accumulates $\boldsymbol{\rho}_0 \cdot \frac{\cos\theta_i}{\pi} \cdot L_{\text{emit}, k} \cdot \Omega_k$ directly into `layer_accum[k]`.
3. **Shared Cosine-Weighted Indirect Bounce**:
   - Generates **one** cosine-weighted hemisphere bounce ray $\omega_1$ from $\mathbf{p}$, intersects the terrain once at $\mathbf{p}_2$ (albedo $\boldsymbol{\rho}_1$), and evaluates NEE shadow rays from $\mathbf{p}_2$ toward the active lights $k$.
   - Result: **1 primary ray + 1 bounce ray serves all $K = 16$ lights**, and every light layer $L_k$ gets exact 3-channel RGB interreflection with **zero RGB cross-talk**!

---

## 3. Auxiliary AOVs + Intel Open Image Denoise (OIDN Metal Backend)

Path-tracing variance scales as $\mathcal{O}(1 / \sqrt{N_{\text{spp}}})$. Going from 64 spp to 512 spp costs $8\times$ more compute.

Intel Open Image Denoise (`OIDN >= 2.2`, which includes a native **Apple Silicon Metal / BNNS backend**) removes Monte Carlo noise in **~18 ms per 1080p layer** while preserving 100% of the fine 100m DEM ridge detail and river channels—**if and only if** it is given noise-free auxiliary feature buffers (AOVs):
- `aov_albedo` (`float3[H, W]`): exact surface albedo $\boldsymbol{\rho}_0$ at the first camera hit (supersampled across pixel sub-samples).
- `aov_normal` (`float3[H, W]`): camera-space or world-space smooth shading normal $\mathbf{n}_0$ at the first hit.

Because all $M$ light layers share the exact same camera viewpoint and static terrain mesh, `sonicray` renders `aov_albedo` and `aov_normal` **once** during the initial `amb` pass and reuses them as the guide buffers to denoise every sun layer and firework/beacon layer (`src/oidn_denoiser.cpp`).

### Important Rule: Pre-Divide or Use `hdr = true` + `cleanAux = true`
When denoising an individual dark light layer $L_k$ with `OIDN`:
- Set `oidnSetFilterBool(filter, "hdr", true)` and `oidnSetFilterBool(filter, "cleanAux", true)`.
- Alternatively, denoise the **irradiance** $E_k = L_k / (\boldsymbol{\rho}_0 + \epsilon)$ and multiply back by $\boldsymbol{\rho}_0$ after denoising (`albedo demodulation`), which guarantees that high-frequency texture detail (rivers, reservoirs, rock strata) can never be blurred by the denoiser! `sonicray` supports demodulated denoising out of the box (`demodulate_albedo = TRUE`).

---

## 4. Runtime Metal Compilation (Working Without Full `Xcode.app`)

A major practical gotcha on macOS is that users with only **Command Line Tools** (`/Library/Developer/CommandLineTools`) have `clang++` and `Metal.framework`, but **do not** have `xcrun metal` or `xcrun metallib`.

`sonicray` avoids `xcrun metal` entirely:
- `src/Makevars` compiles `src/metal_bridge.cpp` with `clang++ -ObjC++ -framework Metal -framework Foundation -framework CoreGraphics`.
- At runtime, `metal_bridge.cpp` loads `inst/metal/sonicray_pathtracer.metal` as an `NSString*` and calls `[device newLibraryWithSource:source options:compileOptions error:&err]`, which invokes Apple's in-process Metal driver compiler (`MTLCompilerService`)—available on **every macOS installation** with zero Xcode requirement!

---

## 5. Bridging Back to `rayrender` and `rayshader`

Once `sonicray` is validated as a standalone package, upstream integration requires only a ~30-line hook in `rayrender::render_scene()`:
```r
if (identical(backend, "metal")) {
  if (!requireNamespace("sonicray", quietly = TRUE)) {
    stop("Install the 'sonicray' package for Metal GPU rendering.")
  }
  return(sonicray::sr_render_rayrender_scene(scene, width = width, height = height, samples = samples, ...))
}
```
And `rayshader::render_highquality()` automatically inherits GPU acceleration via its `...` pass-through to `rayrender::render_scene()`.
