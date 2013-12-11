#!/usr/bin/env python2
# mcmc_demos.py
# Author: Nick Ulle
'''Demonstrations of MCMC methods.

This module has demonstrations of the Metropolis and parallel tempering
algorithms, as discussed in the accompanying paper.
'''

import argparse as ap
import os

import numpy as np

from mcmc_algorithms import *

# ----- Demo Functions
def demo_laplace(seed):
    dir_path = 'out/demo_laplace/'
    make_dir(dir_path)

    u = lambda t: log_inf(dlaplace(t))
    propose = lambda t: np.random.normal(t, 3)
    num_iter = 10**5
    burnin = int(0.05 * num_iter)

    a = np.empty(5)

    print '\nSampling Laplace(0, 1).'
    print 'Using {} iterations (after {} burn-in).'.format(num_iter, burnin)

    print '\nRunning Metropolis algorithm...',
    np.random.seed(seed)
    (y, a[0]) = metropolis(u, propose, num_iter, burnin)

    np.savetxt(dir_path + 'metropolis_01.csv', y, delimiter = ',')
    print 'complete (acceptance {:.2%}).'.format(a[0])

    temperatures_list = [np.arange(6, 1, -1),
                         2 * np.arange(5, 0, -1),
                         np.repeat(1.5, 5) ** np.arange(5, 0, -1),
                         np.repeat(2, 5) ** np.arange(5, 0, -1)]

    for (i, temperatures) in enumerate(temperatures_list):
        print '\nRunning parallel tempering {} with temperature ladder:' \
            .format(i + 1)
        print '  {}'.format(temperatures)

        np.random.seed(seed)
        (x, a[i + 1]) = \
            parallel_tempering(u, temperatures, propose, num_iter, burnin)

        np.savetxt(dir_path + 'parallel_0{}.csv'.format(i + 1), x,
                   delimiter = ',')
        print 'Complete (acceptance {:.2%}).'.format(a[i + 1])

    np.savetxt(dir_path + 'acceptance.csv', a)

    print '\nOutput directory: {}\n'.format(dir_path)

def demo_01(seed):
    dir_path = 'out/demo_01/'
    make_dir(dir_path)

    u = normal_mixture_gentle
    propose = lambda t: np.random.normal(t, 3)
    num_iter = 10**5
    burnin = int(0.05 * num_iter)

    a = np.empty(5)

    print '\nSampling 20% N(-6, 0.25), 80% N(8, 1) mixture.'
    print 'Using {} iterations (after {} burn-in).'.format(num_iter, burnin)

    print '\nRunning Metropolis algorithm...',
    np.random.seed(seed)
    (y, a[0]) = metropolis(u, propose, num_iter, burnin)

    np.savetxt(dir_path + 'metropolis_01.csv', y, delimiter = ',')
    print 'complete (acceptance {:.2%}).'.format(a[0])

    temperatures_list = [np.arange(6, 1, -1),
                         np.arange(11, 1, -2),
                         np.repeat(1.5, 5) ** np.arange(5, 0, -1),
                         np.repeat(2, 5) ** np.arange(5, 0, -1)]

    for (i, temperatures) in enumerate(temperatures_list):
        print '\nRunning parallel tempering {} with temperature ladder:' \
            .format(i + 1)
        print '  {}'.format(temperatures)

        np.random.seed(seed)
        (x, a[i + 1]) = \
            parallel_tempering(u, temperatures, propose, num_iter, burnin)

        np.savetxt(dir_path + 'parallel_0{}.csv'.format(i + 1), x,
                   delimiter = ',')
        print 'Complete (acceptance {:.2%}).'.format(a[i + 1])

    np.savetxt(dir_path + 'acceptance.csv', a)

    print '\nOutput directory: {}\n'.format(dir_path)

def demo_02(seed):
    dir_path = 'out/demo_02/'
    make_dir(dir_path)

    u = normal_mixture
    propose = lambda t: np.random.normal(t, 1)
    num_iter = 10**5
    burnin = int(0.05 * num_iter)

    a = np.empty(5)

    print '\nSampling 25% N(j, 0.25) mixture (j = -20, -10, 0, 10).'
    print 'Using {} iterations (after {} burn-in).'.format(num_iter, burnin)

    print '\nRunning Metropolis algorithm...',
    np.random.seed(seed)
    (y, a[0]) = metropolis(u, propose, num_iter, burnin, -10)

    np.savetxt(dir_path + 'metropolis_01.csv', y, delimiter = ',')
    print 'complete (acceptance {:.2%}).'.format(a[0])

    temperatures_list = [np.arange(6, 1, -1),
                         np.arange(11, 1, -1),
                         np.repeat(1.5, 5) ** np.arange(5, 0, -1),
                         np.repeat(1.5, 10) ** np.arange(10, 0, -1)]

    for (i, temperatures) in enumerate(temperatures_list):
        print '\nRunning parallel tempering {} with temperature ladder:' \
            .format(i + 1)
        print '  {}'.format(temperatures)

        np.random.seed(seed)
        (x, a[i + 1]) = \
            parallel_tempering(u, temperatures, propose, num_iter, burnin, -10)

        np.savetxt(dir_path + 'parallel_0{}.csv'.format(i + 1), x,
                   delimiter = ',')
        print 'Complete (acceptance {:.2%}).'.format(a[i + 1])

    np.savetxt(dir_path + 'acceptance.csv', a)

    print '\nOutput directory: {}\n'.format(dir_path)

def demo_03(seed):
    dir_path = 'out/demo_03/'
    make_dir(dir_path)

    u = normal_mixture
    propose = lambda t: np.random.normal(t, 1)
    num_iter = int(2.5 * 10**5)
    burnin = int(0.05 * num_iter)

    a = np.empty(5)

    print '\nSampling 25% N(j, 0.25) mixture (j = -20, -10, 0, 10).'
    print 'Using {} iterations (after {} burn-in).'.format(num_iter, burnin)

    temperatures_list = [np.arange(26, 1, -5),
                         np.repeat(1.1, 10) ** np.arange(10, 0, -1),
                         np.repeat(1.5, 10) ** np.arange(10, 0, -1),
                         np.repeat(2, 5) ** np.arange(5, 0, -1),
                         np.repeat(3, 5) ** np.arange(5, 0, -1)]

    for (i, temperatures) in enumerate(temperatures_list):
        print '\nRunning parallel tempering {} with temperature ladder:' \
            .format(i + 1)
        print '  {}'.format(temperatures)

        np.random.seed(seed)
        (x, a[i]) = \
            parallel_tempering(u, temperatures, propose, num_iter, burnin, -10)

        np.savetxt(dir_path + 'parallel_0{}.csv'.format(i + 1), x,
                   delimiter = ',')
        print 'Complete (acceptance {:.2%}).'.format(a[i])

    np.savetxt(dir_path + 'acceptance.csv', a)

    print '\nOutput directory: {}\n'.format(dir_path)

# ----- Demo Helper Functions
def make_dir(path):
    '''Makes a directory, without raising an error if it already exists.

    Args:
        path    -- the path to the directory.
    '''
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def log_inf(x):
    if x == 0:
        return -float('inf')
    else:
        return log(x)

def normal_mixture_gentle(x):
    return log_inf(0.2 * dnorm(x, -6, 0.5) + \
                   0.8 * dnorm(x, 8, 1) \
                   )

def normal_mixture(x):
    return log_inf(0.25 * dnorm(x, -20, 0.25) + \
                   0.25 * dnorm(x, -10, 0.25) + \
                   0.25 * dnorm(x, 0, 0.25) + \
                   0.25 * dnorm(x, 10, 0.25) \
                   )

def wild_mixture(x):
    likelihood = 0.75 * dunif(x, -10, -8) + 0.25 * dexp(x, 1, 2)
    log_inf(likelihood)

# ----- Script
def main():
    args = parse_args()

    switch = {0: demo_laplace, 
              1: demo_01,
              2: demo_02,
              3: demo_03}
    switch[args.d](args.seed)

def parse_args():
    parser = ap.ArgumentParser(description = 'Demonstrate the parallel \
                               tempering algorithm.')
    parser.add_argument('-d', help = 'Run demonstration D.', type = int,
                        default = 0, choices = [1, 2, 3])
    parser.add_argument('seed', metavar = 'SEED',
                        help = 'Use SEED as the random seed.',
                        type = int, nargs = '?', default = 250)
    return parser.parse_args()

if __name__ == '__main__':
    main()

