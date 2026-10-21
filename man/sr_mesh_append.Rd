\name{sr_mesh_append}
\alias{sr_mesh_append}
\title{Append a mesh (for example a submerged object) to another mesh}
\description{Append a mesh (for example a submerged object) to another mesh}
\usage{sr_mesh_append(a, b)}
\arguments{
\item{a,b}{meshes with `vertices`, `normals`, `uvs`, `indices` (0-based triangles), as returned by [sr_mesh_from_heightfield()] and [sr_sphere_mesh()].}
}
\value{the combined mesh. Objects take their colour from the albedo texture at their uv; see [sr_paint_albedo()].}
