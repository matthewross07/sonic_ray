\name{sr_render_light_layers}
\alias{sr_render_light_layers}
\title{Render all light layers (Ambient + AOVs + Sun Arc + Sonde/WQP Multi-Light Batches) on Metal GPU}
\description{Render all light layers (Ambient + AOVs + Sun Arc + Sonde/WQP Multi-Light Batches) on Metal GPU}
\usage{sr_render_light_layers( gpu_scene, light_table, lookfrom   = c(-0.5, 15.8, 70.0), lookat     = c(-0.5,  1.5, -2.0), fov        = 22.8, width      = 1920L, height     = 1080L, amb_spp    = 128L, light_spp  = 48L, max_bounces = 4L, clamp      = 50, batch_size = 16L, out_dir    = NULL )}
\arguments{
\item{gpu_scene}{A `sonicray_gpu_scene` returned by [sr_upload_scene()].}
\item{light_table}{Data frame of lights with columns `name`, `x`, `y`, `z`, `r`, `intensity`.}
\item{lookfrom,lookat}{Camera 3-vectors in scene coordinates (the defaults frame the Cache la Poudre scene; set your own).}
\item{fov}{Vertical field-of-view in degrees (default 22.8).}
\item{width,height}{Output layer resolution (default 1920 x 1080).}
\item{amb_spp,light_spp}{Samples per pixel for ambient and point/sphere light batches.}
\item{max_bounces}{Surface vertices per path with next-event estimation (1 = direct light only, 4 = three indirect bounces).}
\item{clamp}{Per-sample brightness clamp (fireflies); the CPU film pipeline used 12, which also clips genuine near-light highlights.}
\item{batch_size}{Number of lights evaluated simultaneously per BVH traversal (1..32, default 16).}
\item{out_dir}{Optional directory to write `.f32` layer files compatible with `02_compose_video.py`.}
}
