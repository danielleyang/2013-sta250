#!/usr/bin/env Rscript
# postproc.R
# Author: Nick Ulle
# Description:
#   This script performs post-processing on the output from mcmc_demos.py,
#   such as creating plots and calculating diagonstics. R and the coda
#   package were used here because of the scant documentation included with
#   Pymc.

library(lattice)
library(coda)
library(optparse)

density_overlay <- function(x, f, curve_args = list(lty = 'dotted'), pch = 20,
                            cex = 0.01, xlab = 'Value',  ...) {
    densityplot(x, pch = pch, cex = cex, xlab = xlab, ...,
                prepanel = function(x, ...) {
                    lims = prepanel.default.densityplot(x, ...)
                    lims$ylim = range(f(x), lims$ylim)
                    lims
                },
                panel = function(x, ...) {
                    panel.densityplot(x, ...)
                    do.call(panel.curve, c(bquote(f), curve_args))
                })
}

dlaplace <- function(x, mean = 0, rate = 1) {
    0.5 * rate * exp(-abs(x - mean) * rate)
}

normal_mixture_gentle <- function(x) {
    0.2 * dnorm(x, -6, 0.5) + 0.8 * dnorm(x, 8, 1)
}

normal_mixture <- function(x) {
    0.25 * dnorm(x, -20, 0.25) +
    0.25 * dnorm(x, -10, 0.25) + 
    0.25 * dnorm(x, 0, 0.25) + 
    0.25 * dnorm(x, 10, 0.25)
}

mcmc_diagnostics <- function(x) {
    x <- mcmc(x)
    structure(c(acf(x, plot = FALSE)$acf[[2]], effectiveSize(x), 
                pnorm(geweke.diag(x)$z, lower.tail = FALSE)),
              names = c('Autocorrelation', 'Effective Size', 
                        'Geweke $p$-value'))
             
}

write_tex <- function(x, path, row.names = FALSE, col.names = FALSE,
                      digits = 1, nsmall = 2, scientific = FALSE, ...) {
    x <- format(x, digits = digits, nsmall = nsmall, scientific = FALSE, ...)
    write.table(x, path, sep = ' & ', eol = ' \\\\\n', quote = FALSE, 
                row.names = row.names, col.names = col.names)
}

process <- function(filenames, in_dir, out_dir, d, xlim) {
    dir.create(out_dir, showWarnings = FALSE)

    # Load the data.
    files <- paste0(in_dir, filenames, '.csv')
    data <- lapply(files, 
                   function(file) {
                       read.csv(file, header = FALSE)[[1]]
                   })

    # Make a TeX table of diagnostic values.
    diagnostics <- sapply(data, mcmc_diagnostics)
    acceptance <- read.csv(paste0(in_dir, 'acceptance.csv'), header = FALSE)
    diagnostics <- rbind(t(acceptance), diagnostics)
    rownames(diagnostics)[[1]] <- 'Acceptance'

    out_file <- paste0(out_dir, 'diagnostics.tex')
    write_tex(diagnostics, out_file, row.names = TRUE)
    cat('Wrote', out_file, '\n')

    # Make density plots.
    if (missing(xlim)) {
        xlim <- range(data)
    }
    lapply(seq_along(data),
           function(i) {
               out_file <- paste0(out_dir, 'density_0', i - 1, '.png')
               png(out_file, width = 8, height = 8, units = 'in', res = 300)
               print(density_overlay(data[[i]], d, main = '', xlim = xlim))
               dev.off()
               cat('Wrote', out_file, '\n')
           })

    invisible()
}

demo_laplace <- function() {
    filenames <- c('metropolis_01', paste0('parallel_0', 1:4))
    in_dir <- 'out/demo_laplace/'
    out_dir <- 'tex/res/demo_laplace/'

    process(filenames, in_dir, out_dir, dlaplace)
}

demo_01 <- function() {
    filenames <- c('metropolis_01', paste0('parallel_0', 1:4))
    in_dir <- 'out/demo_01/'
    out_dir <- 'tex/res/demo_01/'

    process(filenames, in_dir, out_dir, normal_mixture_gentle)
}

demo_02 <- function() {
    filenames <- c('metropolis_01', paste0('parallel_0', 1:4))
    in_dir <- 'out/demo_02/'
    out_dir <- 'tex/res/demo_02/'

    process(filenames, in_dir, out_dir, normal_mixture, c(-25, 15))
}

demo_03 <- function() {
    filenames <- paste0('parallel_0', 1:5)
    in_dir <- 'out/demo_03/'
    out_dir <- 'tex/res/demo_03/'

    process(filenames, in_dir, out_dir, normal_mixture, c(-25, 15))
}

main <- function() {
    parser <- OptionParser(description = 'Post-process the output from 
                           mcmc_demos.py')
    parser <- add_option(parser, c('-d', '--demo'), type = 'integer',
                         help = 'Post-process demonstration D.', default = 0) 
    args <- parse_args(parser)

    demos <- c(demo_laplace, demo_01, demo_02, demo_03)
    demos[[args$demo + 1]]()
}

main()

