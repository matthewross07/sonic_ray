\name{sr_intake}
\alias{sr_intake}
\title{Print the questions an assistant should ask the user before rendering (colours, quality, speed, ...)}
\description{Print the questions an assistant should ask the user before rendering (colours, quality, speed, ...)}
\usage{sr_intake(json = FALSE) invisible(system2(sr_cli(), c("intake", if (json) "--json")))}
\arguments{
\item{json}{machine-readable output.}
}
