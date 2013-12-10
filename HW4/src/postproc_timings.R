#!/usr/bin/env Rscript
# postproc_timings.R
# Author: Nick Ulle
# Description:
#   Processes the CPU/GPU timings into nice plots.

library(lattice)

processSampling <- function() {
    data <- read.csv('../out/timings.csv', header = FALSE)

    n <- nrow(data)

    data <- data.frame(Time = c(data[[1]], data[[2]]), 
                       Method = rep(c('GPU', 'CPU'), each = n),
                       k = seq_len(n))

    out <- xyplot(Time ~ k, data, groups = Method, type = 'l',
                  scales = list(y = list(log = TRUE)),
                  auto.key = list(lines = TRUE, points = FALSE),
                  main = 'Performance Timings (sampling)',
                  ylab = 'Time (log-scale seconds)')

    png('../tex/res/timings.png', width = 8, height = 8, units = 'in',
        res = 300)
    print(out)
    dev.off()
}

processMCMC <- function() {
    files <- paste0('../out/time_0', 1:5, '.csv')
    data <- lapply(files, read.csv, header = FALSE)
    data <- do.call(rbind, data)

    write.table(format(data, digits = 3, nsmall = 2),
                '../tex/res/timings_mcmc.tex', quote = FALSE,
                sep = ' & ', eol = ' \\\\\n', col.names = FALSE)

    n <- nrow(data)

    data <- data.frame(Time = c(data[, 1], data[, 2]), 
                       Method = rep(c('GPU', 'CPU'), each = n),
                       Dataset = seq_len(n))

    out <- xyplot(Time ~ Dataset, data, groups = Method, type = 'l',
                  scales = list(y = list(log = TRUE)),
                  auto.key = list(lines = TRUE, points = FALSE),
                  main = 'Performance Timings (probit MCMC)',
                  ylab = 'Time (log-scale seconds)')

    png('../tex/res/timings_mcmc.png', width = 8, height = 8, units = 'in',
        res = 300)
    print(out)
    dev.off()
}

processSampling()
processMCMC()

