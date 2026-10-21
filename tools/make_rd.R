# Development tool: generate man/*.Rd from the roxygen-style comments in R/*.R (roxygen2 itself is not required).
esc <- function(s) gsub("([%{}\\\\])", "\\\\\\1", s)
exports <- regmatches(readLines("NAMESPACE"), gregexpr("sr_[A-Za-z_]+", readLines("NAMESPACE"))); exports <- unique(unlist(exports))
unlink(list.files("man", full.names = TRUE)); dir.create("man", showWarnings = FALSE)
for (f in list.files("R", pattern = "\\.R$", full.names = TRUE)) {
  L <- readLines(f); i <- 1
  while (i <= length(L)) {
    if (!startsWith(L[i], "#'")) { i <- i + 1; next }
    j <- i; while (j <= length(L) && startsWith(L[j], "#'")) j <- j + 1
    blk <- trimws(sub("^#'", "", L[i:(j - 1)])); hdr <- if (j <= length(L)) L[j] else ""
    m <- regmatches(hdr, regexec("^([A-Za-z_.0-9]+) <- function\\(", hdr))[[1]]
    i <- j
    if (length(m) < 2 || !(m[2] %in% exports) || any(grepl("@keywords internal", blk))) next
    name <- m[2]; sig <- hdr; k <- j
    while (lengths(regmatches(sig, gregexpr("\\(", sig))) > lengths(regmatches(sig, gregexpr("\\)", sig)))) { k <- k + 1; sig <- paste(sig, trimws(L[k])) }
    args <- sub("\\)\\s*\\{?\\s*$", "", sub("^[A-Za-z_.0-9]+ <- function\\(", "", sig))
    title <- blk[1]; desc <- character(); params <- list(); value <- ""
    for (b in blk[-1]) {
      if (startsWith(b, "@param")) { for (pp in strsplit(b, "\\s@param\\s")[[1]]) { pp <- sub("^@param\\s+", "", pp); nm <- sub("\\s.*$", "", pp); params[[length(params) + 1]] <- c(nm, trimws(sub("^\\S+\\s*", "", pp))) } }
      else if (startsWith(b, "@return")) value <- trimws(sub("^@return", "", b))
      else if (startsWith(b, "@") || !nzchar(b)) next
      else if (length(params)) params[[length(params)]][2] <- paste(params[[length(params)]][2], b)
      else desc <- c(desc, b)
    }
    out <- c(sprintf("\\name{%s}", name), sprintf("\\alias{%s}", name), sprintf("\\title{%s}", esc(title)), sprintf("\\description{%s}", esc(paste(c(desc, if (!length(desc)) title), collapse = " "))),
             sprintf("\\usage{%s(%s)}", name, esc(args)))
    if (length(params)) out <- c(out, "\\arguments{", vapply(params, function(p) sprintf("\\item{%s}{%s}", esc(p[1]), esc(p[2])), ""), "}")
    if (nzchar(value)) out <- c(out, sprintf("\\value{%s}", esc(value)))
    writeLines(out, file.path("man", paste0(name, ".Rd")))
  }
}
cat("wrote", length(list.files("man")), "Rd files\n")
