#!/usr/bin/env Rscript
# BLB_index_plot.R
# Author: Nick Ulle

library(lattice)

data <- read.table('final/blb_lin_reg_data_s5_r50_SE.txt', header = TRUE)[[1]]

png('../tex/res/01_index_plot.png', 2400, 2400)
xyplot(data ~ seq_along(data), main = 'Index Plot', xlab = 'Index',
       ylab = 'Estimated Standard Error',
       panel = function(x, y, ...) {
           panel.xyplot(x, y, ...)
           panel.abline(h = 0.01)
       })
dev.off()

cat('Plot generated successfully!\n')

