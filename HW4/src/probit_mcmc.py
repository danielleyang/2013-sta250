#!/usr/bin/env python2.7
# probit_mcmc.py
# Author: Nick Ulle

import argparse as ap
import numpy as np

import pycuda.driver as cuda

from cpu_trunc_norm import *
from gpu_trunc_norm import *

def probit_mcmc(y, X, beta_0, sigma_0_inv, niter = 2000, burnin = 500, 
                seed = 2121, gpu = True, verbose = False):
    # Select the appropriate truncated normal sampling function.
    if gpu:
        gpu_curand_init(seed)
        trunc_norm = gpu_trunc_norm
    else:
        set_seed(seed)
        trunc_norm = cpu_trunc_norm

    (n, p) = X.shape
    if verbose:
        print 'GPU {0}, n {1}, p {2}'.format(gpu, n, p)

    # When y == 0, use left tail. When y == 1, use right tail.
    a = np.zeros_like(y)
    b = np.zeros_like(y)
    a[y == 0] = -np.float32('inf')
    b[y == 1] = np.float32('inf')

    # Run the Gibbs sampler. The steps are:
    #   1. Sample from Z | beta, Y
    #   2. Sample from beta | Z, Y
    # This produces samples of from Z, beta | Y.
    beta = np.zeros((burnin + niter + 1, p))
    posterior_cov = np.linalg.inv(X.transpose().dot(X) + sigma_0_inv)

    for i in range(1, burnin + niter + 1):
        z = trunc_norm(n, X.dot(beta[i - 1]), np.ones(n), a, b)
        
        posterior_mean = posterior_cov \
            .dot(X.transpose().dot(z) + sigma_0_inv.dot(beta_0))
        beta[i, :] = \
            np.random.multivariate_normal(posterior_mean, posterior_cov)

    # Clean up the GPU memory if used.
    if gpu:
        gpu_curand_deinit()

    # Discard starting value + burnin.
    return beta[(burnin + 1):, :]

def main():
    args = parse_args()

    if args.n is not None:
        run_dataset(int(args.n))
    else:
        run_dataset(0)

def parse_args():
    parser = ap.ArgumentParser(description = 'Run probit MCMC on the GPU and \
                               CPU.')
    parser.add_argument('-n', help = 'run probit MCMC on specified dataset',
                        choices = ['1', '2', '3', '4', '5'])
    return parser.parse_args()

def run_dataset(k):
    # Load the dataset.
    if k == 0:
        data_path = '../mini_data.txt'
        beta_path = '../out/mini_beta.csv'
        time_path = '../out/mini_time.csv'
        print 'Running mini dataset.'
    else:
        data_path = '../data_0{}.txt'.format(k)
        beta_path = '../out/beta_0{}.csv'.format(k)
        time_path = '../out/time_0{}.csv'.format(k)
        print 'Running dataset {}.'.format(k)

    data = np.loadtxt(data_path, skiprows = 1)
    y = data[:, 0]
    X = data[:, 1:]
    (n, p) = X.shape

    beta_0 = np.zeros(p)
    sigma_0_inv = np.zeros((p, p))

    beta = np.empty((p, 2))

    # Set up for timing.
    time = np.empty((1, 2))
    start = cuda.Event()
    end = cuda.Event()

    # Run the MCMC using the GPU.
    start.record()
    gpu_beta = probit_mcmc(y, X, beta_0, sigma_0_inv)
    end.record()
    end.synchronize()
    time[0, 0] = start.time_till(end) * 1e-3
    beta[:, 0] = gpu_beta.mean(axis = 0)
    print 'GPU method took {} seconds.'.format(time[0, 0])

    start.record()
    # Run the MCMC using the CPU
    cpu_beta = probit_mcmc(y, X, beta_0, sigma_0_inv, gpu = False)
    end.record()
    end.synchronize()
    time[0, 1] = start.time_till(end) * 1e-3
    beta[:, 1] = cpu_beta.mean(axis = 0)
    print 'CPU method took {} seconds.'.format(time[0, 1])

    # Save the results.
    np.savetxt(beta_path, beta, delimiter = ',')
    np.savetxt(time_path, time, delimiter = ',')


if __name__ == '__main__':
    main()

