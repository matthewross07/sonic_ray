// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "comp_engine.h"
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "comp_sparks.h"

namespace sr {
namespace {

// ---- parameter blocks: field order is the contract with inst/metal/sonicray_comp.metal ----
struct AxpyP { int W, H, init, y0; int mode[8]; float w[24]; };
struct SatP { int W, H, nev, has_sun1, y0; float t, half_w, dur, tau; float ts[4], uc[4]; float col[3]; };
struct Dims { int W, H, y0; };
struct HaloP { int W, H, y0; float c1, c2, a, b; float col[3]; };
struct SharpP { int W, H, y0; float amount; };
struct BlurP { int w, h, ch, r, dir; float k[130]; };
struct ResizeP { int sw, sh, dw, dh, ch; };
struct SkyP { int W, H, day, use_stars; float f, t, star_amt, y_edge, ox1, dawn_band, set_band, sky_exp; float zen[3], hort[3], venus[3], sunc[3]; float lo, step; int nb; };
struct SparkP { int W, H, ntx, tile; float col[15]; };
struct FinalP { int W, H, orb_mode, use_atm, use_ov, dither; uint32_t seed; int nb; float atm, fade, scale; float orb[9]; float bcn[96]; };

constexpr size_t MAX_PRIMS = 600000, MAX_TILE_ENTRIES = 3000000;
constexpr int HIST_NB = 8192;

std::vector<float> gauss_taps(double sigma, int& r) {          // scipy.ndimage.gaussian_filter1d weights (truncate = 4)
  r = (int)(4.0 * sigma + 0.5);
  std::vector<float> k(2 * r + 1);
  double sum = 0; std::vector<double> d(2 * r + 1);
  for (int i = -r; i <= r; ++i) { d[i + r] = std::exp(-0.5 * i * i / (sigma * sigma)); sum += d[i + r]; }
  for (size_t i = 0; i < d.size(); ++i) k[i] = (float)(d[i] / sum);
  return k;
}

}  // namespace

struct Compositor::Impl {
  int W = 0, H = 0, y0 = 0;
  Look look;
  SceneInfo scene;
  Geometry geo;
  std::string dir;
  std::set<std::string> names;
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> lib = nil;
  std::map<std::string, id<MTLComputePipelineState>> pso;
  std::mutex layer_mu;
  std::map<std::string, id<MTLBuffer>> layers;
  id<MTLBuffer> b_meff = nil, b_aerial = nil, b_uu = nil, b_vv = nil;
  std::unique_ptr<SparkSystem> sparks;

  struct Ctx {
    id<MTLBuffer> lin, bl, terrain, tblur, bright, b1, b2, b3, ov, ob1, ob2, sky, S1, S2;
    id<MTLBuffer> lum, g1, g2, out, bmax, omax, p90, hor, hist, prims, toff, tidx;
    std::vector<Prim> prim_vec; std::vector<uint32_t> off_vec, idx_vec;
  };
  std::vector<Ctx> ctxs;

  id<MTLBuffer> alloc(size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("GPU buffer allocation failed");
    memset([b contents], 0, bytes);
    return b;
  }
  id<MTLBuffer> map_file(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st; fstat(fd, &st);
    size_t pg = (size_t)getpagesize(), len = ((size_t)st.st_size + pg - 1) / pg * pg;
    void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);   // private copy-on-write mapping that we never write
    close(fd);
    if (p == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
    id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:p length:len options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) throw std::runtime_error("cannot map " + path + " into a GPU buffer");
    return b;
  }
  id<MTLBuffer> layer(const std::string& name) {
    std::lock_guard<std::mutex> lk(layer_mu);
    auto it = layers.find(name);
    if (it != layers.end()) return it->second;
    if (!names.count(name)) throw std::runtime_error("layer '" + name + "' not found in " + dir + " (expected " + name + ".f32)");
    id<MTLBuffer> b = map_file(dir + "/" + name + ".f32");
    layers[name] = b;
    return b;
  }
  id<MTLComputePipelineState> pipeline(const std::string& name) {
    auto it = pso.find(name);
    if (it == pso.end()) throw std::runtime_error("missing kernel " + name);
    return it->second;
  }

  // ---- command encoding helpers (one encoder per kernel; Metal tracks the buffer hazards)
  struct Enc {
    Impl* im; id<MTLCommandBuffer> cmd;
    void op(const char* kernel, std::initializer_list<id<MTLBuffer>> bufs, const void* params, size_t plen, int gw, int gh) {
      id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
      [e setComputePipelineState:im->pipeline(kernel)];
      int i = 0;
      for (id<MTLBuffer> b : bufs) [e setBuffer:b offset:0 atIndex:i++];
      if (plen) [e setBytes:params length:plen atIndex:i];
      [e dispatchThreads:MTLSizeMake(gw, gh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [e endEncoding];
    }
    void axpy(id<MTLBuffer> out, const std::vector<std::pair<id<MTLBuffer>, const Term*>>& terms, bool init) {
      size_t n = std::max<size_t>(terms.size(), 1);
      for (size_t s = 0; s < n; s += 8) {
        AxpyP p{}; p.W = im->W; p.H = im->H; p.init = (init && s == 0) ? 1 : 0; p.y0 = im->y0;
        id<MTLBuffer> l[8];
        for (int k = 0; k < 8; ++k) {
          if (s + k < terms.size()) { l[k] = terms[s + k].first; const Term* t = terms[s + k].second; p.mode[k] = t->mode; for (int c = 0; c < 3; ++c) p.w[3 * k + c] = t->w[c]; }
          else { l[k] = out; p.mode[k] = -1; }
        }
        op("axpy8", {out, l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]}, &p, sizeof(p), im->W, im->H - im->y0);
      }
    }
    // Gaussian blur matching the reference compositor: direct for sigma <= 6, else box-down / blur / bilinear-up
    void blur(id<MTLBuffer> src, id<MTLBuffer> dst, id<MTLBuffer> S1, id<MTLBuffer> S2, double sigma, int ch) {
      int W = im->W, H = im->H;
      if (sigma <= 6) {
        int r; auto k = gauss_taps(sigma, r);
        for (int d = 0; d < 2; ++d) {
          BlurP p{}; p.w = W; p.h = H; p.ch = ch; p.r = r; p.dir = d; std::copy(k.begin(), k.end(), p.k);
          op("blur1d", {d == 0 ? src : S1, d == 0 ? S1 : dst}, &p, sizeof(p), W, H);
        }
        return;
      }
      int k = (int)(sigma / 3), w2 = std::max(W / k, 2), h2 = std::max(H / k, 2);
      ResizeP dn{W, H, w2, h2, ch};
      op("box_down", {src, S1}, &dn, sizeof(dn), w2, h2);
      int r; auto kk = gauss_taps(sigma / k, r);
      for (int d = 0; d < 2; ++d) {
        BlurP p{}; p.w = w2; p.h = h2; p.ch = ch; p.r = r; p.dir = d; std::copy(kk.begin(), kk.end(), p.k);
        op("blur1d", {d == 0 ? S1 : S2, d == 0 ? S2 : S1}, &p, sizeof(p), w2, h2);
      }
      ResizeP up{w2, h2, W, H, ch};
      op("bilinear_up", {S1, dst}, &up, sizeof(up), W, H);
    }
  };
};

Compositor::Compositor(const std::string& layers_dir, const std::string& metal_source, const SceneInfo& scene, const Look& look, int n_contexts)
    : impl_(new Impl) {
  Impl& m = *impl_;
  m.W = scene.W; m.H = scene.H; m.scene = scene; m.look = look; m.dir = layers_dir;
  const int W = m.W, H = m.H;
  DIR* d = opendir(layers_dir.c_str());
  if (!d) throw std::runtime_error("cannot open layers directory " + layers_dir);
  while (dirent* e = readdir(d)) { std::string f = e->d_name; if (f.size() > 4 && f.substr(f.size() - 4) == ".f32") m.names.insert(f.substr(0, f.size() - 4)); }
  closedir(d);
  for (const char* need : {"amb", "mask", "riv"})
    if (!m.names.count(need)) throw std::runtime_error(layers_dir + " is missing required layer " + need + ".f32 (see the layer contract in docs/BRING_YOUR_OWN_DATA.md)");

  m.dev = MTLCreateSystemDefaultDevice();
  if (!m.dev) throw std::runtime_error("no Metal device (needs an Apple Silicon Mac)");
  m.queue = [m.dev newCommandQueue];
  NSError* err = nil;
  MTLCompileOptions* opt = [MTLCompileOptions new];
  opt.languageVersion = MTLLanguageVersion3_0;
  m.lib = [m.dev newLibraryWithSource:[NSString stringWithUTF8String:metal_source.c_str()] options:opt error:&err];
  if (!m.lib) throw std::runtime_error(std::string("Metal compile failed: ") + [[err localizedDescription] UTF8String]);
  for (const char* k : {"axpy8", "sat_add", "lum_mean", "halo_add", "tonemap", "sharpen_comb", "blur1d", "box_down", "bilinear_up", "sky_hist", "sky_pctl",
                        "sky_hor", "sky_render", "spark_raster", "final_frame"}) {
    id<MTLFunction> fn = [m.lib newFunctionWithName:[NSString stringWithUTF8String:k]];
    if (!fn) throw std::runtime_error(std::string("missing Metal kernel ") + k);
    id<MTLComputePipelineState> ps = [m.dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) throw std::runtime_error(std::string("pipeline failed for ") + k);
    m.pso[k] = ps;
  }

  // ---- static buffers derived from the terrain mask
  const float* mask = (const float*)[m.layer("mask") contents];
  m.b_meff = m.alloc((size_t)W * H * 4);
  float* meff = (float*)[m.b_meff contents];
  for (size_t i = 0; i < (size_t)W * H; ++i) meff[i] = std::min(std::max(mask[i], 0.0f), 1.0f);
  int top = -1, bot = -1;
  for (int y = 0; y < H && top < 0; ++y) { for (int x = 0; x < W; ++x) if (mask[(size_t)y * W + x] != 0.0f) { top = y; break; } }
  for (int y = H - 1; y >= 0 && bot < 0; --y) { for (int x = 0; x < W; ++x) if (mask[(size_t)y * W + x] != 0.0f) { bot = y; break; } }
  Geometry& g = m.geo;
  g.y_top = top >= 0 ? top : 0.31 * H; g.y_bot = bot >= 0 ? bot : H - 1;
  g.ox0 = 0.36 * W; g.ox1 = 0.54 * W;
  int ycol = -1; for (int y = 0; y < H; ++y) if (mask[(size_t)y * W + (int)g.ox1] > 0.5f) { ycol = y; break; }
  g.y_edge = ycol >= 0 ? ycol : 0.32 * H;
  g.orb_r = 0.015 * H; g.orb_y_top = -0.03 * H; g.orb_y_set = g.y_edge + 0.018 * H;
  m.y0 = std::min(std::max((int)g.y_top, 0), H - 1);                  // terrain passes skip the sky rows above the terrain's bounding box
  m.b_aerial = m.alloc((size_t)W * H * 4);
  float* aer = (float*)[m.b_aerial contents];
  for (int y = 0; y < H; ++y) {
    double far = std::pow(std::min(std::max(1.0 - (y - g.y_top) / std::max(g.y_bot - g.y_top, 1.0), 0.0), 1.0), 2.1);
    for (int x = 0; x < W; ++x) aer[(size_t)y * W + x] = (float)(far * meff[(size_t)y * W + x]);
  }
  if (m.names.count("uv_u") && m.names.count("uv_v")) {
    const float* U = (const float*)[m.layer("uv_u") contents]; const float* V = (const float*)[m.layer("uv_v") contents];
    m.b_uu = m.alloc((size_t)W * H * 4); m.b_vv = m.alloc((size_t)W * H * 4);
    float* uu = (float*)[m.b_uu contents]; float* vv = (float*)[m.b_vv contents];
    auto gam = [](float c) { return std::pow(std::min(std::max(c, 0.0f), 1.0f), 1.0f / 2.2f); };   // 8-bit textures come back as value^2.2
    for (size_t i = 0; i < (size_t)W * H; ++i) { uu[i] = gam(U[i * 3]) + gam(U[i * 3 + 1]) / 255.0f; vv[i] = gam(V[i * 3]) + gam(V[i * 3 + 1]) / 255.0f; }
  }
  m.sparks.reset(new SparkSystem(scene, look.spark_slow));
  const size_t n3 = (size_t)W * H * 12, n1 = (size_t)W * H * 4;
  const int ntx = (W + TILE - 1) / TILE, nty = (H + TILE - 1) / TILE;
  m.ctxs.resize(std::max(n_contexts, 1));
  for (auto& c : m.ctxs) {
    c.lin = m.alloc(n3); c.bl = m.alloc(n3); c.terrain = m.alloc(n3); c.tblur = m.alloc(n3); c.bright = m.alloc(n3);
    c.b1 = m.alloc(n3); c.b2 = m.alloc(n3); c.b3 = m.alloc(n3); c.ov = m.alloc(n3); c.ob1 = m.alloc(n3); c.ob2 = m.alloc(n3); c.sky = m.alloc(n3);
    c.S1 = m.alloc(n3); c.S2 = m.alloc(n3);
    c.lum = m.alloc(n1); c.g1 = m.alloc(n1); c.g2 = m.alloc(n1);
    c.out = m.alloc((size_t)W * H * 3);
    c.bmax = m.alloc(16); c.omax = m.alloc(16); c.p90 = m.alloc(16); c.hor = m.alloc(16);
    c.hist = m.alloc(HIST_NB * 4);
    c.prims = m.alloc(MAX_PRIMS * sizeof(Prim)); c.toff = m.alloc((size_t)(ntx * nty + 1) * 4); c.tidx = m.alloc(MAX_TILE_ENTRIES * 4);
  }
}
Compositor::~Compositor() = default;
int Compositor::width() const { return impl_->W; }
int Compositor::height() const { return impl_->H; }
const Geometry& Compositor::geometry() const { return impl_->geo; }
Look& Compositor::look() { return impl_->look; }
bool Compositor::has_layer(const std::string& n) const { return !n.empty() && impl_->names.count(n) > 0; }
const std::set<std::string>& Compositor::layer_names() const { return impl_->names; }

void Compositor::render(int ci, const FrameSpec& spec, uint8_t* out) {
  Impl& m = *impl_;
  Impl::Ctx& c = m.ctxs.at(ci);
  const int W = m.W, H = m.H;
  const double s = W / 480.0;
  const Look& look = m.look;
  @autoreleasepool {
    std::vector<std::pair<id<MTLBuffer>, const Term*>> terms, bea_terms;
    for (const Term& t : spec.terms) terms.push_back({m.layer(t.layer), &t});
    for (const Term& t : spec.beacon_terms) bea_terms.push_back({m.layer(t.layer), &t});

    // ---- CPU: spark primitives (geometry) -> shared GPU memory
    int use_ov = 0;
    const int ntx = (W + TILE - 1) / TILE;
    if (!spec.bursts.empty()) {
      m.sparks->primitives(spec.t, spec.bursts, c.prim_vec);
      if (!c.prim_vec.empty()) {
        if (c.prim_vec.size() > MAX_PRIMS) throw std::runtime_error("too many spark primitives in one frame");
        m.sparks->bin_tiles(c.prim_vec, c.off_vec, c.idx_vec);
        if (c.idx_vec.size() > MAX_TILE_ENTRIES) throw std::runtime_error("too many spark/tile entries in one frame; reduce n_sparks");
        memcpy([c.prims contents], c.prim_vec.data(), c.prim_vec.size() * sizeof(Prim));
        memcpy([c.toff contents], c.off_vec.data(), c.off_vec.size() * 4);
        memcpy([c.tidx contents], c.idx_vec.data(), c.idx_vec.size() * 4);
        use_ov = 1;
      }
    }
    const SkyDay& sky = spec.sky;
    const int day = sky.active ? 1 : 0;
    const int use_stars = (spec.star_amount > 0 && m.names.count("stars")) ? 1 : 0;
    id<MTLBuffer> dummy = m.layer("amb");
    id<MTLBuffer> sa = day ? m.layer(sky.layer_a) : dummy, sb = day ? m.layer(sky.layer_b) : dummy, stars = use_stars ? m.layer("stars") : dummy;
    SkyP skyp{}; skyp.W = W; skyp.H = H; skyp.day = day; skyp.use_stars = use_stars; skyp.f = (float)sky.f; skyp.t = (float)spec.t;
    skyp.star_amt = (float)spec.star_amount; skyp.y_edge = (float)m.geo.y_edge; skyp.ox1 = (float)m.geo.ox1;
    if (day) {
      skyp.dawn_band = (float)sky.dawn_band; skyp.set_band = (float)sky.set_band; skyp.sky_exp = (float)sky.sky_exp;
      for (int i = 0; i < 3; ++i) { skyp.zen[i] = (float)sky.zenith[i]; skyp.hort[i] = (float)sky.horizon_tint[i]; skyp.venus[i] = (float)sky.venus[i]; skyp.sunc[i] = (float)sky.sunset[i]; }
    }
    skyp.lo = -24.0f; skyp.step = 40.0f / HIST_NB; skyp.nb = HIST_NB;

    memset([c.bmax contents], 0, 16); memset([c.omax contents], 0, 16);
    if (day) memset([c.hist contents], 0, HIST_NB * 4);
    const int cropH = H - m.y0;
    Dims d2{W, H, m.y0};
    id<MTLCommandBuffer> cmd = [m.queue commandBuffer];
    Impl::Enc enc{&m, cmd};
    enc.axpy(c.lin, terms, true);
    const SatScan& sat = spec.sat;
    if (sat.active && !sat.events.empty() && m.b_uu != nil) {
      SatP p{}; p.W = W; p.H = H; p.nev = (int)std::min<size_t>(sat.events.size(), 4); p.y0 = m.y0;
      bool second = m.names.count(sat.second_layer) > 0; p.has_sun1 = second;
      p.t = (float)sat.t; p.half_w = (float)sat.half_width; p.dur = (float)sat.duration; p.tau = (float)sat.tau;
      for (int i = 0; i < p.nev; ++i) { p.ts[i] = (float)sat.events[i].first; p.uc[i] = (float)sat.events[i].second; }
      for (int i = 0; i < 3; ++i) p.col[i] = sat.color[i];
      enc.op("sat_add", {c.lin, dummy, second ? m.layer(sat.second_layer) : dummy, m.b_uu, m.b_vv, m.layer("riv")}, &p, sizeof(p), W, cropH);
    }
    if (!bea_terms.empty()) {
      enc.axpy(c.bl, bea_terms, true);
      Term tint; tint.layer = ""; tint.mode = 0; for (int i = 0; i < 3; ++i) tint.w[i] = look.beacon_tint[i];
      enc.axpy(c.lin, {{c.bl, &tint}}, false);
      enc.op("lum_mean", {c.bl, c.lum}, &d2, sizeof(d2), W, cropH);
      enc.blur(c.lum, c.g1, c.S1, c.S2, 12 * s, 1);
      enc.blur(c.lum, c.g2, c.S1, c.S2, 30 * s, 1);
      HaloP hp{}; hp.W = W; hp.H = H; hp.y0 = m.y0; hp.c1 = 7.5f; hp.c2 = 14.0f; hp.a = 1.15f; hp.b = 0.45f; hp.col[0] = 0.12f * 0.476f; hp.col[1] = 0.44f * 0.476f; hp.col[2] = 0.92f * 0.476f;
      enc.op("halo_add", {c.lin, c.g1, c.g2, m.layer("riv")}, &hp, sizeof(hp), W, cropH);
    }
    enc.op("tonemap", {c.lin, m.b_meff, c.terrain, c.bright, c.bmax}, &d2, sizeof(d2), W, cropH);
    if (look.sharpen > 0) {
      enc.blur(c.terrain, c.tblur, c.S1, c.S2, std::max(1.2 * W / 1920.0, 0.5), 3);
      SharpP sp{W, H, m.y0, (float)look.sharpen};
      enc.op("sharpen_comb", {c.terrain, c.tblur}, &sp, sizeof(sp), W, cropH);
    }
    enc.blur(c.bright, c.b1, c.S1, c.S2, 2.5 * s, 3);
    enc.blur(c.bright, c.b2, c.S1, c.S2, 8.5 * s, 3);
    enc.blur(c.bright, c.b3, c.S1, c.S2, 22 * s, 3);
    if (day) {
      enc.op("sky_hist", {sa, sb, c.hist}, &skyp, sizeof(skyp), (W + 3) / 4, (H + 3) / 4);
      enc.op("sky_pctl", {c.hist, c.p90}, &skyp, sizeof(skyp), 1, 1);
    }
    enc.op("sky_hor", {sa, sb, c.p90, c.hor}, &skyp, sizeof(skyp), 1, 1);
    enc.op("sky_render", {sa, sb, c.p90, stars, c.sky}, &skyp, sizeof(skyp), W, H);
    if (use_ov) {
      SparkP sp{}; sp.W = W; sp.H = H; sp.ntx = ntx; sp.tile = TILE;
      const float* cols[5] = {look.spark_colors[0], look.spark_colors[1], look.spark_colors[2], nullptr, nullptr};
      static const float shimmer[3] = {0.56f, 0.46f, 1.00f}, white[3] = {1.0f, 1.0f, 1.0f};
      cols[3] = shimmer; cols[4] = white;
      for (int k = 0; k < 5; ++k) for (int i = 0; i < 3; ++i) sp.col[3 * k + i] = cols[k][i];
      enc.op("spark_raster", {c.prims, c.toff, c.tidx, c.ov, c.omax}, &sp, sizeof(sp), W, H);
      enc.blur(c.ov, c.ob1, c.S1, c.S2, 0.85 * s, 3);
      enc.blur(c.ov, c.ob2, c.S1, c.S2, 2.6 * s, 3);
    }
    FinalP fp{}; fp.W = W; fp.H = H; fp.orb_mode = spec.orb.mode; fp.use_atm = spec.aerial > 0 ? 1 : 0; fp.use_ov = use_ov; fp.dither = look.dither ? 1 : 0;
    fp.seed = spec.seed & 0x7FFFFFFFu; fp.atm = (float)spec.aerial; fp.fade = (float)spec.fade; fp.scale = (float)(W / 640.0);
    if (spec.orb.mode) {
      const Orb& o = spec.orb; double v[9] = {o.x, o.y, o.radius, o.visibility, o.dusk, o.amount, o.color[0], o.color[1], o.color[2]};
      for (int i = 0; i < 9; ++i) fp.orb[i] = (float)v[i];
    }
    fp.nb = (int)std::min<size_t>(spec.beacons.size(), 24);
    for (int i = 0; i < fp.nb; ++i) { fp.bcn[4 * i] = (float)spec.beacons[i].x; fp.bcn[4 * i + 1] = (float)spec.beacons[i].y; fp.bcn[4 * i + 2] = (float)spec.beacons[i].beam; fp.bcn[4 * i + 3] = (float)spec.beacons[i].pulse; }
    enc.op("final_frame", {c.terrain, m.b_meff, c.sky, c.hor, m.b_aerial, c.b1, c.b2, c.b3, c.ov, c.ob1, c.ob2, c.bmax, c.omax, c.out}, &fp, sizeof(fp), W, H);
    [cmd commit];
    [cmd waitUntilCompleted];
    if ([cmd status] != MTLCommandBufferStatusCompleted) throw std::runtime_error("GPU command buffer failed");
    memcpy(out, [c.out contents], (size_t)W * H * 3);
  }
}

}  // namespace sr
