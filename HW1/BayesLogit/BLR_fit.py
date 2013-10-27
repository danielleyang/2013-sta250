#!/bin/python2
# BLR_fit.py
# Author: Nick Ulle
# Description: Python code for HW 1, problem 2.

from __future__ import division
import sys
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt

def bayes_logreg(m, y, X, beta_0, Sigma_0_inv, niter = 10000, burnin = 1000,
                 print_every = 1000, retune = 100, verbose = True):
    # Set up proposal covariance matrix.
    p = beta_0.size
    scale = 0.005
    cov = scale * np.eye(p)
    # Track number of accepted draws.
    accept_count = 0
    # Pre-allocate memory.
    sample = np.zeros((burnin + niter) * p).reshape((burnin + niter, p))

    if verbose:
        print 'Parameter dimension p is {0}.'.format(p)

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
                print 4 * ' ' + 'Retuned by {0:04.3f}. Scale is {1:04.3f}.' \
                    .format(mult, scale)

    print 'Sampling completed.'
    # Remove burn-in draws from the sample.
    sample = sample[burnin:]

    # Compute percentiles iteratively because Numpy doesn't support vectorized
    # percentiles yet.
    pct = np.zeros(99 * p).reshape(99, p)

    for j in range(0, p):
        for i in range(0, 99):
            pct[i, j] = np.percentile(sample[:, j], i + 1)

    if verbose:
        print 4 * ' ', '25th percentile: {0}'.format(pct[24, :])
        print 4 * ' ', '50th percentile: {0}'.format(pct[49, :])
        print 4 * ' ', '75th percentile: {0}'.format(pct[74, :])

    return (sample, pct)

def log_posterior(beta, m, y, X, beta_0, Sigma_0_inv):
    likelihood = -0.5 * (beta - beta_0).dot(Sigma_0_inv).dot(beta - beta_0)
    likelihood += y.dot(X).dot(beta)
    factor = np.log( 1 + np.exp(X.dot(beta)) )
    likelihood -= m.dot(factor)
    return likelihood

def fit_sim(job):
    # Add the extra '1' in front of the job number.
    job = '1{0:03}'.format(job)
    print 'Running job {0}.'.format(job)

    # Set up prior parameters.
    beta_0 = np.zeros(2)
    Sigma_0_inv = np.eye(2)
    
    # Read values from the data file.
    data_file = 'data/blr_data_{0}.csv'.format(job)
    data = np.loadtxt(data_file, delimiter = ',', skiprows = 1,
                      dtype = 'i4, i4, 2f8')
    y = data['f0']
    m = data['f1']
    X = data['f2']
    
    # Run the logistic regression.
    (samp, pct) = bayes_logreg(m, y, X, beta_0, Sigma_0_inv)
    
    # Save the results.
    result_file = 'results/blr_res_{0}.csv'.format(job)
    np.savetxt(result_file, pct, delimiter = ',', fmt = '%.8f')

def fit_breast_cancer():
    # Set up prior parameters.
    beta_0 = np.zeros(10)
    Sigma_0_inv = np.eye(10)

    # Read values from the data file.
    data = np.loadtxt('breast_cancer.txt', skiprows = 1, dtype = '10f8, i2', 
                      converters = {10: lambda s: 1 if s == '"M"' else 0})
    X = data['f0']
    y = data['f1']
    m = np.ones(y.size)

    # Run the logistic regression.
    (samp, pct) = bayes_logreg(m, y, X, beta_0, Sigma_0_inv)

    # Save the results.

# ----- Main Script -----
# Set up command line options.
parser = ap.ArgumentParser(description = 'Perform Bayesian logistic \
                           regression.')
group = parser.add_mutually_exclusive_group(required = True)
group.add_argument('-i', help = 'fit a simulated data set', metavar = 'JOB',
                   type = int)
group.add_argument('-b', help = 'fit the breast cancer data set',
                   action = 'store_true')
args = parser.parse_args()

# Run the program.
if args.b:
    fit_breast_cancer()
else:
    fit_sim(args.i)

