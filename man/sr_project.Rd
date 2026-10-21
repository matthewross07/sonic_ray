\name{sr_project}
\alias{sr_project}
\title{Project world points to pixel coordinates (same convention as the compositor)}
\description{Project world points to pixel coordinates (same convention as the compositor)}
\usage{sr_project(xyz, lookfrom, lookat, fov, width, height)}
\arguments{
\item{xyz}{n x 3 matrix of scene coordinates (km).}
\item{lookfrom,lookat}{camera.}
\item{fov}{vertical field of view (degrees).}
\item{width,height}{image size. @return n x 2 matrix (px, py), y down.}
}
