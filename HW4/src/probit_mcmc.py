#!/usr/bin/env python2.7
# probit_mcmc.py
# Author: Nick Ulle

import numpy as np

from gpu_trunc_norm import *

def probit_mcmc(y, X, beta_0, sigma_0_inv, niter = 2000, burnin = 500, 
                use_gpu = True, verbose = False):
    # Select the appropriate truncated normal sampling function.
    tr_norm = trunc_norm
    if use_gpu:
        gpu_random_activate(2121)
        tr_norm = gpu_trunc_norm

    (n, p) = X.shape
    if verbose:
        print 'GPU {0}, n {1}, p {2}'.format(use_gpu, n, p)

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
        z = tr_norm(n, X.dot(beta[i - 1]), np.ones(n), a, b)
        
        posterior_mean = posterior_cov \
            .dot(X.transpose().dot(z) + sigma_0_inv.dot(beta_0))
        beta[i, :] = \
            np.random.multivariate_normal(posterior_mean, posterior_cov)

    # Clean up the GPU memory if used.
    if use_gpu:
        gpu_random_deactivate()

    # Discard starting value + burnin.
    return beta[(burnin + 1):, :]

def main():
    data = np.loadtxt('mini_data.txt', skiprows = 1)
    y = data[:, 0]
    X = data[:, 1:]
    (n, p) = X.shape

    beta = probit_mcmc(y, X, np.zeros(p), np.zeros((p, p)))

if __name__ == '__main__':
    main()

