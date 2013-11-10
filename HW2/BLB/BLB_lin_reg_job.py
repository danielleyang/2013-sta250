#!/bin/env python2
# BLB_lin_reg_job.py
# Author: Nick Ulle

import argparse as ap
import csv

# External modules
import numpy as np
from wls import *

# Global variables
# Number of little bootstraps.
S = 5
# Samples per little bootstrap.
R = 50
# Exponent determining number of unique elements per little bootstrap.
GAMMA = 0.7
# The mini data file.
MINI_FILE = 'data/blb_lin_reg_mini.txt'
MINI_NCOL = 41
# The full data file.
FILE = ''
NCOL = 101

def main():
    # Set up command line arguments.
    parser = ap.ArgumentParser(
        description = 'This program takes a little bootstrap sample from the \
        specified CSV data set and fits a linear regression. The last column \
        of the data set is assumed to be the response. When no arguments are \
        specified, the mini data set is used, provided it is in the data \
        sub-directory.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', metavar = 'JOB',
                       help = 'job number (using the full data set)')
    group.add_argument('-f', nargs = 2, metavar = ('FILE', 'NCOL'),
                       help = 'use NCOL columns from FILE as the data set')
    args = parser.parse_args()

    if args.i:
        file = FILE
        ncol = NCOL
        print 'This is job {0}; using full data set...'.format(args.i)
    elif args.f:
        file = args.f[0]
        ncol = int(args.f[1])
        print 'Using {0} columns from file:'.format(ncol)
        print 2 * ' ' + file
    else:
        file = MINI_FILE
        ncol = MINI_NCOL
        print 'No job specified; using mini data set...'

    n = countLines(file)

    # Draw a little bootstrap seed of size b = n^GAMMA.
    b = int(np.ceil(n ** GAMMA))
    print 'n = {0};  b = {1}'.format(n, b)

    rows = np.random.choice(n, b)

    # Get the seed rows.
    data = readRows(file, rows, ncol)
    y = data[:, -1]
    X = data[:, :-1]
    
    print 'y shape is {0};  X shape is {1}'.format(y.shape, X.shape)

    # Get the weights for each seed row.
    weights = np.random.multinomial(n, np.ones(b) / b)

    # Perform the linear regression.
    # wls(y, X, w, verbose=False):
    beta = wls(y, X, weights, False)

    s_index = 5; r_index = 1
    outfile = 'output/coef_{0:02d}_{1:02d}.txt'.format(s_index, r_index)

def readRows(file, rows, ncol):
    # Read the specified rows from a file.
    max_row = np.max(rows)
    data = np.zeros((len(rows), ncol))
    i = 0
    with open(file, 'r') as lines:
        lines = csv.reader(lines)
        for (row, line) in enumerate(lines):
            if row in rows:
                # Store rows in a matrix.
                data[i, :] = np.float64(line)
                i += 1
            elif row > max_row:
                # Stop reading through the file once we've got all rows.
                break
    return data

def countLines(file):
    # Count number of lines n in the data file.
    # Faster to use `wc -l` for this.
    n = 0
    with open(file, 'r') as lines:
        for line in lines:
            n += 1
    return n

# Allow this script to be loaded as a module.
if __name__ == '__main__':
    main()

