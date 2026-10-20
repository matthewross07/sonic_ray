// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
// =============================================================================
// sonicray/src/mesh_utils.cpp
// Plain C++ mesh helpers (no Metal): smooth vertex normals for large triangle meshes.
// =============================================================================
#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

// Area-weighted smooth vertex normals (the cross product length is proportional to triangle area).
// [[Rcpp::export]]
NumericMatrix cpp_vertex_normals(NumericMatrix vertices, IntegerMatrix indices) {
    const int nv = vertices.nrow(), nt = indices.nrow();
    NumericMatrix n(nv, 3);
    for (int t = 0; t < nt; ++t) {
        const int a = indices(t, 0), b = indices(t, 1), c = indices(t, 2);
        const double ux = vertices(b, 0) - vertices(a, 0), uy = vertices(b, 1) - vertices(a, 1), uz = vertices(b, 2) - vertices(a, 2);
        const double vx = vertices(c, 0) - vertices(a, 0), vy = vertices(c, 1) - vertices(a, 1), vz = vertices(c, 2) - vertices(a, 2);
        const double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const int vs[3] = {a, b, c};
        for (int k = 0; k < 3; ++k) { n(vs[k], 0) += nx; n(vs[k], 1) += ny; n(vs[k], 2) += nz; }
    }
    for (int i = 0; i < nv; ++i) {
        double len = std::sqrt(n(i, 0) * n(i, 0) + n(i, 1) * n(i, 1) + n(i, 2) * n(i, 2));
        if (len < 1e-20) { n(i, 0) = 0; n(i, 1) = 1; n(i, 2) = 0; } else { n(i, 0) /= len; n(i, 1) /= len; n(i, 2) /= len; }
    }
    return n;
}
