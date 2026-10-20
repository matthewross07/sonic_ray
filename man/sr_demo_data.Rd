\name{sr_demo_data}
\alias{sr_demo_data}
\title{Synthetic demo data (invented terrain, sensors and samples)}
\description{Generates a fractal river-valley terrain, a river and lakes, \\code\{n_sensors\} invented sensors along the river, and a table of invented "samples" (three made-up parameters, \\code\{param_a/b/c\}) with made-up seasonal values. **Everything is synthetic**; it is only an example of the *shape* of data the package expects. See \\code\{docs/BRING_YOUR_OWN_DATA.md\} to replace it with real sensors, real coordinates and a real digital elevation model.}
\usage{sr_demo_data(seed = 7L, n_sensors = 12L, duration_s = 18, n = 360L, tex = 2048L)}
\arguments{
\item{seed}{random seed (the demo is fully reproducible).}
\item{n_sensors}{number of invented sensors.}
\item{duration_s}{length of the film in seconds (sets the event times).}
\item{n}{terrain grid size (even number).}
\item{tex}{albedo texture size in pixels. (km), \\code\{y0\}, \\code\{vexag\}, \\code\{albedo\} (h x w x 3, linear), \\code\{river\} and \\code\{water\} (masks, \\code\{n x n\}), \\code\{sensors\} (data frame), \\code\{events\} (data frame), \\code\{cam\}, \\code\{synthetic = TRUE\}.}
}
\value{a list of class \\code\{sonicray_demo_data\} with elements \\code\{H\} (elevation matrix, metres), \\code\{xr\}, \\code\{zr\}}
