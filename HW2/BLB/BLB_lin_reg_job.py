#!/bin/env python2
# BLB_lin_reg_job.py
# Author: Nick Ulle

import argparse as ap
import csv

# ----- External modules
import numpy as np
from wls import wls

# ----- Global constants
# Number of little bootstraps.
S = 5
# Samples per little bootstrap.
R = 50
# Exponent determining number of unique elements per little bootstrap.
GAMMA = 0.7
# The mini data file.
MINI_N = 10 ** 4
MINI_P = 40
MINI_IN = 'data/blb_lin_reg_mini.txt'
MINI_OUT = 'output/coef_mini.txt'
# The full data file.
FULL_N = 10 ** 6
FULL_P = 1000
FULL_IN = '/home/pdbaines/data/blb_lin_reg_data.txt'

# ----- Functions
def main():
    # Process command line arguments.
    (n, p, in_file, out_file, seeds) = parseArgs()

    # Draw a little bootstrap subset of size b = n^GAMMA.
    b = int(np.ceil(n ** GAMMA))
    print 'n = {0}, p = {1}, b = {2}'.format(n, p, b)

    if seeds:
        # Ensure every job with the same s_index draws the same subset.
        np.random.seed(seeds[0])
    rows = np.random.choice(n, b, replace = False)

    # Get the subset rows.
    data = readRows(in_file, rows, p + 1)
    y = data[:, -1]
    X = data[:, :-1]
    
    print 'y shape is {0};  X shape is {1}\n'.format(y.shape, X.shape)

    # Draw weights for each subset row.
    if seeds:
        # Ensure every job draws different weights.
        np.random.seed(seeds[1])
    weights = np.random.multinomial(n, np.ones(b) / b)

    # Perform the linear regression.
    # wls(y, X, w, verbose=False):
    beta = wls(y, X, weights, False)

    # Store the result.
    with open(out_file, 'w') as writer:
        writer.write('beta\n')
        for elt in beta:
            writer.write(str(elt) + '\n')

    print 'Completed successfully.'

def readRows(file, rows, ncol):
    # Read the specified rows from a file.
    rows = np.sort(rows)
    data = np.zeros((len(rows), ncol))
    i = 0
    with open(file, 'r') as reader:
        lines = csv.reader(reader)
        for (row, line) in enumerate(lines):
            if row == rows[i]:
                # Store rows in a matrix.
                data[i, :] = np.float64(line[0:ncol])
                i += 1
                if i == len(rows):
                    assert row == rows[-1]
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

def parseArgs():
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
    group.add_argument('-f', nargs = 4, metavar = ('N', 'P', 'IN', 'OUT'),
                       help = 'use data set IN with N observations and P \
                       covariates; results are written to OUT')
    args = parser.parse_args()

    if args.i:
        # Use the full data set.
        index = int(args.i)
        s_index = ((index - 1) // R) + 1
        r_index = index - (s_index - 1) * R
        # The little bootstrap subset must be the same for all jobs with the
        # same s_index, while the little bootstrap sample created from the
        # subset must be different for all jobs.
        # So the first seed depends on s_index, the second depends on index.
        seeds = (4141 * s_index, 1157 * index)
        out_file = 'output/coef_{0:02d}_{1:02d}.txt'.format(s_index, r_index)

        print 'Using full data set. This is job {0} (s = {1}, r = {2}).' \
            .format(args.i, s_index, r_index)
        print 'Random seeds are {0} and {1}.\n'.format(seeds[0], seeds[1])
        return (FULL_N, FULL_P, FULL_IN, out_file, seeds)
    elif args.f:
        # Use the user-specified data set.
        (n, p, in_file, out_file) = args.f
        n = int(n)
        p = int(p)

        print 'Using input file:'
        print 2 * ' ' + in_file
        print 'Using output file:'
        print 2 * ' ' + out_file + '\n'
        return (n, p, in_file, out_file, None)
    else:
        # Use the mini data set.
        print 'Using mini data set (no arguments specified).\n'
        return (MINI_N, MINI_P, MINI_IN, MINI_OUT, None)

# Allow this script to be loaded as a module.
if __name__ == '__main__':
    main()

