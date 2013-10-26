#!/bin/python2
# BLR_fit.py
# Author: Nick Ulle
# Description: Python code for HW 1, problem 2.

from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt

def bayes_logreg(m, y, X, beta_0, Sigma_0_inv, niter = 10000, burnin = 1000,
                 print_every = 1000, retune = 100, verbose = True):
    # Use Metropolis-Hastings to create a sample of size niter from the
    # posterior distribution.
    p = beta_0.size
    scale = 0.005
    cov = scale * np.eye(p)
    accept_count = 0
    sample = np.zeros((burnin + niter) * p).reshape((burnin + niter, p))

    if verbose:
        print 'p is {0}.'.format(p)

    for i in range(1, burnin + niter):
        # Draw from proposal distribution.
        draw = np.random.multivariate_normal(sample[i - 1, :], cov)

        # Accept with calcuated probability.
        U = np.random.uniform()
        A = log_posterior(draw, m, y, X, beta_0, Sigma_0_inv) \
            - log_posterior(sample[i - 1, :], m, y, X, beta_0, Sigma_0_inv)
        A = np.minimum(1, A)

        if np.log(U) < A:
            sample[i, :] = draw
            accept_count += 1
        else:
            sample[i, :] = sample[i - 1, :]

        # Print a status update periodically.
        if (i % print_every) == 0:
            print 'Iteration {0}. Acceptance {1:.2%}.' \
                .format(i, accept_count / i)

        # Retune periodically during the burn-in period.
        if i < burnin and (i % retune) == 0:
            mult = np.exp(1.5 * (accept_count / i - 0.35))
            cov *= mult
            scale *= mult
            if verbose:
                print 'Iteration {0}. Acceptance {1:04.2%}.' \
                    .format(i, accept_count / i)
                print '    Retuned by {0:04.3f}. Scale is {1:04.3f}.' \
                    .format(mult, scale)

    print 'Done!'
    # Remove burn-in draws from the sample.
    sample = sample[burnin:]

    # Compute percentiles iteratively because Numpy doesn't support vectorized
    # percentiles yet.
    pct = np.zeros(99 * p).reshape(99, p)

    for j in range(0, p):
        for i in range(0, 99):
            pct[i, j] = np.percentile(sample[:, j], i + 1)

    if verbose:
        print '    25th percentile: {0}'.format(pct[24, :])
        print '    50th percentile: {0}'.format(pct[49, :])
        print '    75th percentile: {0}'.format(pct[74, :])

    return pct

def log_posterior(beta, m, y, X, beta_0, Sigma_0_inv):
    likelihood = -0.5 * (beta - beta_0).dot(Sigma_0_inv).dot(beta - beta_0)
    likelihood += y.dot(X).dot(beta)
    factor = np.log( 1 + np.exp(X.dot(beta)) )
    likelihood -= m.dot(factor)
    return likelihood

# ----- Main Script -----
beta_0 = np.zeros(2)
Sigma_0_inv = np.eye(2)

if len(sys.argv) >= 3 and sys.argv[1] == '-i':
    job = '1{0:03}'.format(int(sys.argv[2]))
    print 'Running job {0}.'.format(job)
else:
    print 'Invalid syntax. Use BLR_fit.py -i <job>'
    sys.exit()

# Read values from the data file.
data_file = 'data/blr_data_{0}.csv'.format(job)
data = np.loadtxt(data_file, delimiter = ',', skiprows = 1)
y = data[:, 0].astype('i')
m = data[:, 1].astype('i')
X = data[:, 2:4]

# Run the logistic regression.
pct = bayes_logreg(m, y, X, beta_0, Sigma_0_inv)

# Save the results.
result_file = 'results/blr_res_{0}.csv'.format(job)
np.savetxt(result_file, pct, delimiter = ',', fmt = '%.8f')

