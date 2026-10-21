// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
#include "comp_sparks.h"
#include <algorithm>
#include <cmath>

namespace sr {

// ------------------------------------------------------------------------------------------------ numpy-compatible RNG
NumpyRng::NumpyRng(uint32_t seed) {
  uint32_t s = seed;
  for (int i = 0; i < 624; ++i) {                  // numpy mt19937_seed (= init_genrand)
    key_[i] = s;
    s = 1812433253u * (s ^ (s >> 30)) + (uint32_t)i + 1u;
  }
  pos_ = 624;
}
uint32_t NumpyRng::next32() {
  constexpr int N = 624, M = 397;
  constexpr uint32_t UPPER = 0x80000000u, LOWER = 0x7fffffffu, MATRIX_A = 0x9908b0dfu;
  if (pos_ == N) {
    int kk;
    for (kk = 0; kk < N - M; ++kk) { uint32_t y = (key_[kk] & UPPER) | (key_[kk + 1] & LOWER); key_[kk] = key_[kk + M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u); }
    for (; kk < N - 1; ++kk) { uint32_t y = (key_[kk] & UPPER) | (key_[kk + 1] & LOWER); key_[kk] = key_[kk + (M - N)] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u); }
    uint32_t y = (key_[N - 1] & UPPER) | (key_[0] & LOWER);
    key_[N - 1] = key_[M - 1] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
    pos_ = 0;
  }
  uint32_t y = key_[pos_++];
  y ^= (y >> 11); y ^= (y << 7) & 0x9d2c5680u; y ^= (y << 15) & 0xefc60000u; y ^= (y >> 18);
  return y;
}
double NumpyRng::uniform01() {
  uint32_t a = next32() >> 5, b = next32() >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}
double NumpyRng::normal() {
  if (has_gauss_) { has_gauss_ = false; return gauss_; }
  double x1, x2, r2;
  do { x1 = 2.0 * uniform01() - 1.0; x2 = 2.0 * uniform01() - 1.0; r2 = x1 * x1 + x2 * x2; } while (r2 >= 1.0 || r2 == 0.0);
  double f = std::sqrt(-2.0 * std::log(r2) / r2);
  gauss_ = f * x1; has_gauss_ = true;
  return f * x2;
}
int NumpyRng::choice(const std::vector<double>& p) {
  std::vector<double> cdf(p.size());
  double s = 0; for (size_t i = 0; i < p.size(); ++i) { s += p[i]; cdf[i] = s; }
  for (auto& c : cdf) c /= s;
  double u = uniform01();
  return (int)(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin());     // searchsorted(side = 'right')
}

// ------------------------------------------------------------------------------------------------ geometry
namespace {
constexpr int SS = 2, M_PTS = 9;

// numpy.linalg.qr (LAPACK Householder, 3x3): returns Q with LAPACK's sign convention
void qr3(const double A_in[9], double Q[9]) {
  double A[9]; std::copy(A_in, A_in + 9, A);
  for (int i = 0; i < 9; ++i) Q[i] = (i % 4 == 0) ? 1.0 : 0.0;
  for (int j = 0; j < 2; ++j) {
    double alpha = A[j * 3 + j], xn2 = 0;
    for (int i = j + 1; i < 3; ++i) xn2 += A[i * 3 + j] * A[i * 3 + j];
    if (xn2 == 0.0) continue;
    double beta = -std::copysign(std::sqrt(alpha * alpha + xn2), alpha), tau = (beta - alpha) / beta;
    double v[3] = {0, 0, 0}; v[j] = 1.0;
    for (int i = j + 1; i < 3; ++i) v[i] = A[i * 3 + j] / (alpha - beta);
    for (int c = j; c < 3; ++c) {                      // A = H A
      double dot = 0; for (int i = j; i < 3; ++i) dot += v[i] * A[i * 3 + c];
      for (int i = j; i < 3; ++i) A[i * 3 + c] -= tau * v[i] * dot;
    }
    for (int r = 0; r < 3; ++r) {                      // Q = Q H
      double dot = 0; for (int i = j; i < 3; ++i) dot += Q[r * 3 + i] * v[i];
      for (int i = j; i < 3; ++i) Q[r * 3 + i] -= tau * dot * v[i];
    }
  }
}
inline double itrunc(double x) { return std::trunc(x); }
}  // namespace

SparkSystem::SparkSystem(const SceneInfo& s, double slow) : W_(s.W), H_(s.H), slow_(slow) {
  px_ = W_ / 480.0;
  for (int i = 0; i < 3; ++i) { lf_[i] = s.lookfrom[i]; f_[i] = s.lookat[i] - s.lookfrom[i]; }
  double fn = std::sqrt(f_[0] * f_[0] + f_[1] * f_[1] + f_[2] * f_[2]); for (auto& x : f_) x /= fn;
  r_[0] = f_[1] * 0 - f_[2] * 1; r_[1] = f_[2] * 0 - f_[0] * 0; r_[2] = f_[0] * 1 - f_[1] * 0;     // cross(f, up = (0,1,0))
  double rn = std::sqrt(r_[0] * r_[0] + r_[1] * r_[1] + r_[2] * r_[2]); for (auto& x : r_) x /= rn;
  u_[0] = r_[1] * f_[2] - r_[2] * f_[1]; u_[1] = r_[2] * f_[0] - r_[0] * f_[2]; u_[2] = r_[0] * f_[1] - r_[1] * f_[0];
  tf_ = std::tan(s.fov * M_PI / 360.0);
  w_tail_ = std::max(1, (int)std::lround(0.28 * SS * px_));
  w_mid_  = std::max(1, (int)std::lround(0.42 * SS * px_));
  w_head_ = std::max(2, (int)std::lround(0.58 * SS * px_));
}

void SparkSystem::project(const double* P, int n, double* sx, double* sy) const {
  for (int i = 0; i < n; ++i) {
    double x = P[i * 3] - lf_[0], y = P[i * 3 + 1] - lf_[1], z = P[i * 3 + 2] - lf_[2];
    double xc = x * r_[0] + y * r_[1] + z * r_[2], yc = x * u_[0] + y * u_[1] + z * u_[2], zc = std::max(x * f_[0] + y * f_[1] + z * f_[2], 1e-3);
    sx[i] = (xc / (zc * tf_ * W_ / H_) * 0.5 + 0.5) * (W_ * SS);
    sy[i] = (0.5 - yc / (zc * tf_) * 0.5) * (H_ * SS);
  }
}

std::shared_ptr<const SparkSystem::Static> SparkSystem::spark_static(const Burst& b) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cache_.find(b.uid);
    if (it != cache_.end()) return it->second;
  }
  NumpyRng rng((uint32_t)(b.uid * 13 + 7));
  const int N = b.n_sparks;
  double qr[9]; for (int i = 0; i < 9; ++i) qr[i] = (double)(float)rng.normal();
  double Q[9]; qr3(qr, Q);
  auto s = std::make_shared<Static>();
  s->v.resize(N * 3); s->sp.resize(N); s->life.resize(N); s->phase.resize(N);
  std::vector<double> v0(N * 3);
  for (int i = 0; i < N; ++i) {
    double idx = i + 0.5, phi = std::acos(1.0 - 2.0 * idx / N), theta = M_PI * (1.0 + std::sqrt(5.0)) * idx;
    v0[i * 3] = std::sin(phi) * std::cos(theta); v0[i * 3 + 1] = std::cos(phi); v0[i * 3 + 2] = std::sin(phi) * std::sin(theta);
  }
  std::vector<double> noise(N * 3); for (auto& x : noise) x = (double)(float)(0.0 + 0.11 * rng.normal());
  for (int i = 0; i < N; ++i) {
    double w[3];
    for (int j = 0; j < 3; ++j) w[j] = v0[i * 3] * Q[0 * 3 + j] + v0[i * 3 + 1] * Q[1 * 3 + j] + v0[i * 3 + 2] * Q[2 * 3 + j] + noise[i * 3 + j];
    double n = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    for (int j = 0; j < 3; ++j) s->v[i * 3 + j] = w[j] / n;
  }
  for (int i = 0; i < N; ++i) { double u = (double)(float)rng.uniform(0.0, 1.0); s->sp[i] = b.v_max * (0.12 + 0.90 * std::pow(u, 0.52)) * (0.94 + 0.04 * (b.uid % 4)); }
  for (int i = 0; i < N; ++i) s->life[i] = (double)(float)rng.uniform(b.life0, b.life1) * slow_;
  for (int i = 0; i < N; ++i) s->phase[i] = (double)(float)rng.uniform(0.0, 2.0 * M_PI);
  std::lock_guard<std::mutex> lk(mu_);
  cache_[b.uid] = s;
  return s;
}

void SparkSystem::primitives(double t, const std::vector<Burst>& bursts, std::vector<Prim>& out) {
  struct Rec { int64_t ord; Prim p; };
  std::vector<Rec> recs; recs.reserve(1 << 16);
  auto add = [&](double a, double b, double c, double d, double w, double fill, int plane, int kind, int64_t ord) {
    recs.push_back({ord, Prim{(float)a, (float)b, (float)c, (float)d, (float)w, (float)fill, plane, kind}});
  };
  auto line = [&](double x0, double y0, double x1, double y1, double w, double fill, int plane, int64_t ord) {
    double off = (std::fmod(w, 2.0) == 0.0) ? 0.5 : 0.0;
    add(itrunc(x0) + off, itrunc(y0) + off, itrunc(x1) + off, itrunc(y1) + off, w, fill, plane, 0, ord);
  };
  for (size_t rank = 0; rank < bursts.size(); ++rank) {
    const Burst& b = bursts[rank];
    const int64_t base = (int64_t)rank * 10000000;
    const double dt = t - b.t0;
    if (dt < 0) {                                                        // rising rocket
      double f = 1.0 + dt / b.rise;
      if (f <= 0) continue;
      const int n_rseg = 5;
      double P[6 * 3], sx[6], sy[6];
      for (int k = 0; k <= n_rseg; ++k) {
        double fs = std::max(f - 0.28 * ((double)k / n_rseg), 0.0);
        P[k * 3] = b.origin[0]; P[k * 3 + 1] = b.launch_y + std::pow(fs, 0.82) * (b.origin[1] - b.launch_y); P[k * 3 + 2] = b.origin[2];
      }
      project(P, 6, sx, sy);
      for (int k = 0; k < n_rseg; ++k) {
        double seg_a = std::min(255.0, itrunc(220.0 * f * b.gain * std::pow(1.0 - (double)k / n_rseg, 1.4)));
        double seg_w = k == 0 ? w_head_ : (k < 3 ? w_mid_ : w_tail_);
        line(sx[k + 1], sy[k + 1], sx[k], sy[k], seg_w, seg_a, b.plane, base + 2 * k);
        line(sx[k + 1], sy[k + 1], sx[k], sy[k], w_tail_, itrunc(seg_a * 0.42), PLANE_SHIMMER, base + 2 * k + 1);
      }
      double rh = 1.15 * SS * px_;
      add(sx[0] - rh, sy[0] - rh, sx[0] + rh, sy[0] + rh, 0, std::min(255.0, itrunc(245.0 * f * b.gain)), PLANE_WHITE, 1, base + 20);
      continue;
    }
    auto st = spark_static(b);
    const bool early = dt < 0.055 * slow_;
    const double tr = std::min(b.trail_dt, dt * 0.90);
    for (int i = 0; i < b.n_sparks; ++i) {
      const double life = st->life[i];
      if (dt > life) continue;
      const double age = dt / life;
      double base_a = std::pow(std::max(0.0, 1.0 - age), 1.35);
      double glitter = age > 0.55 ? 0.72 + 0.28 * std::sin(dt * 36.0 + st->phase[i]) : 1.0;
      double ak = std::min(std::min(std::max(base_a * glitter * b.gain, 0.0), 1.15), 1.0);
      if (!(ak > 0.02)) continue;
      double P[M_PTS * 3], sx[M_PTS], sy[M_PTS];
      for (int k = 0; k < M_PTS; ++k) {
        double frac = (double)k / (M_PTS - 1);
        double tt = std::max(dt - frac * tr, 0.0), tp = tt / slow_, drag = (1.0 - std::exp(-2.30 * tp)) / 2.30, m = st->sp[i] * drag;
        P[k * 3] = b.origin[0] + st->v[i * 3] * m;
        P[k * 3 + 1] = b.origin[1] + st->v[i * 3 + 1] * m - 0.5 * 1.38 * tp * tp;
        P[k * 3 + 2] = b.origin[2] + st->v[i * 3 + 2] * m;
      }
      project(P, M_PTS, sx, sy);
      const int64_t order0 = base + (int64_t)i * 100;
      for (int seg = M_PTS - 1; seg >= 1; --seg) {
        double head_w = std::pow(1.0 - (seg - 0.5) / (M_PTS - 1), 1.65);
        double fill = itrunc(255.0 * ak * head_w);
        if (!(fill > 3)) continue;
        double w_seg = seg == 1 ? w_head_ : (seg <= 3 ? w_mid_ : w_tail_);
        int64_t o = order0 + (int64_t)(M_PTS - 1 - seg) * 3;
        if (early && seg <= 2) line(sx[seg], sy[seg], sx[seg - 1], sy[seg - 1], w_seg, itrunc(fill * 0.75), PLANE_WHITE, o);
        line(sx[seg], sy[seg], sx[seg - 1], sy[seg - 1], w_seg, fill, b.plane, o + 1);
        if (seg >= 4 && !early) line(sx[seg], sy[seg], sx[seg - 1], sy[seg - 1], w_tail_, itrunc(fill * 0.28), PLANE_SHIMMER, o + 2);
      }
      double rs = 0.58 * SS * px_ * (0.75 + 0.30 * ak), rc = 0.28 * SS * px_ * (0.75 + 0.30 * ak);
      add(sx[0] - rs, sy[0] - rs, sx[0] + rs, sy[0] + rs, 0, itrunc(255.0 * ak), b.plane, 1, order0 + 30);
      add(sx[0] - rc, sy[0] - rc, sx[0] + rc, sy[0] + rc, 0, itrunc(225.0 * std::pow(ak, 0.9)), PLANE_WHITE, 1, order0 + 31);
    }
    double a_fl = std::exp(-dt / 0.075);                                 // compact central flash
    if (a_fl > 0.03) {
      double sx, sy; project(b.origin, 1, &sx, &sy);
      double r = (3.2 + 6.5 * (1.0 - a_fl)) * SS * px_;
      for (int j = 0; j < 5; ++j) {
        double rj = r * (1.0 - 0.18 * j) * 0.5;
        add(sx - rj, sy - rj, sx + rj, sy + rj, 0, std::min(255.0, itrunc(210.0 * a_fl * b.gain * (0.10 + 0.15 * j))), j >= 3 ? PLANE_WHITE : b.plane, 1, base + 9000000 + j);
      }
    }
  }
  std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b) { return a.ord < b.ord; });
  out.resize(recs.size());
  for (size_t i = 0; i < recs.size(); ++i) out[i] = recs[i].p;
}

int SparkSystem::bin_tiles(const std::vector<Prim>& prims, std::vector<uint32_t>& off, std::vector<uint32_t>& idx) const {
  const int ntx = (W_ + TILE - 1) / TILE, nty = (H_ + TILE - 1) / TILE;
  off.assign((size_t)ntx * nty + 1, 0);
  struct Box { int x0, x1, y0, y1; };
  std::vector<Box> boxes(prims.size());
  for (size_t i = 0; i < prims.size(); ++i) {
    const Prim& p = prims[i];
    double pad = p.kind == 0 ? p.w * 0.5 + 1.0 : 1.5;
    double x0 = (std::min(p.a, p.c) - pad) / SS, x1 = (std::max(p.a, p.c) + pad) / SS, y0 = (std::min(p.b, p.d) - pad) / SS, y1 = (std::max(p.b, p.d) + pad) / SS;
    int tx0 = (int)std::floor(x0 / TILE), tx1 = (int)std::floor(x1 / TILE), ty0 = (int)std::floor(y0 / TILE), ty1 = (int)std::floor(y1 / TILE);
    if (tx1 < 0 || tx0 >= ntx || ty1 < 0 || ty0 >= nty) { boxes[i] = {1, 0, 1, 0}; continue; }
    boxes[i] = {std::max(tx0, 0), std::min(tx1, ntx - 1), std::max(ty0, 0), std::min(ty1, nty - 1)};
  }
  for (const Box& b : boxes) for (int ty = b.y0; ty <= b.y1; ++ty) for (int tx = b.x0; tx <= b.x1; ++tx) off[(size_t)ty * ntx + tx + 1]++;
  for (size_t i = 1; i < off.size(); ++i) off[i] += off[i - 1];
  idx.assign(std::max<uint32_t>(off.back(), 1), 0);
  std::vector<uint32_t> fillpos(off.begin(), off.end() - 1);
  for (size_t i = 0; i < boxes.size(); ++i) {                            // prims in draw order -> stable inside every tile
    const Box& b = boxes[i];
    for (int ty = b.y0; ty <= b.y1; ++ty) for (int tx = b.x0; tx <= b.x1; ++tx) idx[fillpos[(size_t)ty * ntx + tx]++] = (uint32_t)i;
  }
  return ntx;
}

}  // namespace sr
