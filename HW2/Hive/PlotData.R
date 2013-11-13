#!/usr/bin/env Rscript
# PlotData.R
# Author: Nick Ulle
# Description:
#   This script produces the mean-variance plot for HW2, part 3, provided
#   the data has been preloaded to the output/ subdirectory.

library(lattice)

means <- read.csv('output/means.csv', header = FALSE)
vars <- read.csv('output/variances.csv', header = FALSE)

data <- merge(means, vars, by = 'V1')
names(data) <- c('Groups', 'Mean', 'Variance')

png('../tex/res/mean_var.png', 2400, 2400)
xyplot(Variance ~ Mean, data, main = 'Mean-Variance Plot')
dev.off()

cat('Plot generated successfully!\n')

