\name{sr_compose}
\alias{sr_compose}
\title{Compose a film from light layers on the GPU}
\description{One process: worker threads render frames on the GPU while the main thread streams them to ffmpeg.}
\usage{sr_compose(layers, out, story = "network", quality = c("draft", "review", "final"), encoder = NULL, threads = 3L, t0 = 0, t1 = -1, audio = NULL, audio_offset = 0, ...)}
\arguments{
\item{layers}{directory with `scene_meta.json` and `*.f32` layers.}
\item{out}{output .mp4 (use a NEW name).}
\item{story}{"network" (data-driven) or "poudre".}
\item{quality}{"draft", "review" or "final" (see [sr_presets()]).}
\item{encoder}{"x264" (small files, best for final) or "videotoolbox" (Apple hardware encoder: fast, frees the CPU, bigger files). Default from `quality`.}
\item{threads}{frames rendered at once (default 3).}
\item{t0,t1}{render only this time range in seconds.}
\item{audio}{optional .wav/.m4a to mux in.}
\item{audio_offset}{seconds to trim from the start of the audio.}
\item{...}{overrides: `fps`, `crf`, `preset`, `sharpen`, `dither`, `codec`, `vt_quality`.}
}
\value{invisibly, a list with `seconds`, `frames` and `out`.}
