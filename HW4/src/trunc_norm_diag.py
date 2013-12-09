#!/usr/bin/env python2.7
# trunc_norm_diag.py
# Author: Nick Ulle

from __future__ import division
from math import pi, exp, sqrt, erfc
import argparse as ap

import numpy as np

import pycuda.driver as cuda

from cpu_trunc_norm import *
from gpu_trunc_norm import *

def run_timings(k, mean = 0, sd = 1, a = -2, b = 2, seed = 101):
    n = 10 ** k
    mean = np.ones(n) * mean
    sd = np.ones(n) * sd
    a = np.ones(n) * a
    b = np.ones(n) * b

    start = cuda.Event()
    end = cuda.Event()

    gpu_curand_init(seed)
    start.record()
    gpu_trunc_norm(n, mean, sd, a, b)
    end.record()
    end.synchronize()
    gpu_time = start.time_till(end) * 1e-3
    gpu_curand_deinit()

    set_seed(seed)
    start.record()
    cpu_trunc_norm(n, mean, sd, a, b)
    end.record()
    end.synchronize()
    cpu_time = start.time_till(end) * 1e-3

    return (gpu_time, cpu_time)

def run_diagnostics(n, mean = 0, sd = 1, a = -2, b = 2, seed = 101):
    true_mean = trunc_norm_mean(mean, sd, a, b)

    mean = np.ones(n) * mean
    sd = np.ones(n) * sd
    a = np.ones(n) * a
    b = np.ones(n) * b

    gpu_curand_init(seed)
    gpu_mean = np.mean(gpu_trunc_norm(n, mean, sd, a, b))
    gpu_curand_deinit()

    set_seed(seed)
    cpu_mean = np.mean(cpu_trunc_norm(n, mean, sd, a, b))

    return (gpu_mean, cpu_mean, true_mean)

def trunc_norm_mean(mean, sd, a, b):
    alpha = (a - mean) / sd
    beta = (b - mean) / sd
    Z = norm_cdf(beta) - norm_cdf(alpha)
    return mean + sd * (norm_pdf(alpha) - norm_pdf(beta)) / Z

def norm_cdf(x):
    return 0.5 * erfc(-x / sqrt(2))

def norm_pdf(x):
    return exp(-0.5 * x**2) / sqrt(2 * pi)

def main():
    args = parse_args()

    MAX_K = 8

    if args.t:
        # Run CPU and GPU code timings.
        print 'Printing GPU and CPU timings for 10^k.'
        timings = np.empty((MAX_K, 2))
        for k in range(1, MAX_K + 1):
            timings[k - 1, :] =  run_timings(k, 2, 1, 0, 1.5)
            print '{}: {}'.format(k, timings[k - 1, :])
        if args.o is not None:
            np.savetxt(args.o, timings, delimiter = ',')
    else:
        # Run CPU and GPU code diagnostics.
        means = np.empty((3, 3))
        means[0, :] = run_diagnostics(10**5, mean = 1, sd = 2)
        means[1, :] = run_diagnostics(10**5, a = 5, b = np.float32('inf'))
        means[2, :] = run_diagnostics(10**5, a = -np.float32('inf'), b = -10)
        print 'Printing GPU, CPU, and theoretical means.'
        print 'N(1, 2) on (-2, 2): {}'.format(means[0, :])
        print 'N(0, 1) on (5, Inf): {}'.format(means[1, :])
        print 'N(0, 1) on (-Inf, -10): {}'.format(means[2, :])
        if args.o is not None:
            np.savetxt(args.o, means, delimiter = ',')

def parse_args():
    parser = ap.ArgumentParser(description = 'Perform diagnostics on CPU and \
                               GPU truncated normal code.')
    parser.add_argument('-t', action = 'store_true', help = 'run timings')
    parser.add_argument('-o', metavar = 'DEST', help = 'output results to DEST')
    return parser.parse_args()

if __name__ == '__main__':
    main()

