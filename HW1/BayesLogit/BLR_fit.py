#!/bin/python2
# BLR_fit.py
# Author: Nick Ulle
# Description: Python code for HW 1.

# Built-in modules:
from __future__ import division
import sys
import argparse as ap
import functools as ft

# External modules:
import numpy as np
import matplotlib.pyplot as plt

def gibbs_sampler(sample, p, scale, accept_count, log_likelihood):
    for j in range(0, p):
        # Use Metropolis-Hastings to sample from marginal posterior.
        # Draw from proposal distribution.
        marginal_draw = np.random.normal(sample[0, j], scale[j])

        # Acceptance probability is based on samples from the previous iteration
        # and this iteration.
        prev_draw = np.zeros(p)
        prev_draw[1:j] = sample[1, 1:j]
        prev_draw[j:] = sample[0, j:]
        draw = np.copy(prev_draw)
        draw[j] = marginal_draw

        A = log_likelihood(draw) - log_likelihood(prev_draw)

        U = np.random.uniform()
        if np.log(U) < A:
            sample[1, j] = marginal_draw
            accept_count[j] += 1
        else:
            sample[1, j] = sample[0, j]

    return sample[1, :], accept_count

def metropolis(sample, p, scale, accept_count, log_likelihood):
    # Draw from proposal distribution.
    draw = np.random.multivariate_normal(sample[0, :], np.diag(scale))

    # Accept with calcuated probability.
    A = log_likelihood(draw) - log_likelihood(sample[0, :])

    U = np.random.uniform()
    if np.log(U) < A:
        sample[1, :] = draw
        accept_count += 1
    else:
        sample[1, :] = sample[0, :]

    return sample[1, :], accept_count

def bayes_logreg(m, y, X, beta_0, Sigma_0_inv, scale, niter = 10000, 
                 burnin = 1000, print_every = 1000, retune = 100, 
                 verbose = True, gibbs = False):
    p = beta_0.size
    if verbose: 
        print 'Parameter dimension p is {0}.'.format(p), '\n'

    # Pre-allocate memory.
    sample = np.zeros((burnin + niter) * p).reshape((burnin + niter, p))
    # Partially apply the log posterior function.
    log_likelihood = ft.partial(log_posterior, m = m, y = y, X = X, 
                                beta_0 = beta_0, Sigma_0_inv = Sigma_0_inv)

    # Determine which sampling algorithm to use.
    if gibbs:
        sample_func = gibbs_sampler
        accept_count = np.zeros(p)
        if verbose: 
            print 'Running Metropolis-Hastings within a Gibbs sampler:'
    else:
        sample_func = metropolis
        accept_count = np.array([0])
        if verbose: 
            print 'Running Metropolis-Hastings algorithm:'

    # Gather the sample.
    for i in range(1, burnin + niter):
        (sample[i, :], accept_count) = \
            sample_func(sample[i - 1:i + 1, :], p, scale, accept_count, 
                        log_likelihood)

        # Retune periodically during the burn-in period.
        if i < burnin and (i % retune) == 0:
            mult = np.exp(1.25 * (accept_count / i - 0.40))
            scale *= mult
            if verbose:
                print 'Iteration {0}:'.format(i)
                print '  {0: <12}'.format('Acceptance'), 
                print ' '.join('{0: >6.2%}'.format(s) for s in accept_count / i)
                print '  {0: <12}'.format('Retune'), 
                print ' '.join('{0: >6.3f}'.format(s) for s in mult)
                print '  {0: <12}'.format('Scale'), 
                print ' '.join('{0: >6.3f}'.format(s) for s in scale)
        # Print a status update periodically.
        elif (i % print_every) == 0:
            print 'Iteration {0}:'.format(i)
            print '  {0: <12}'.format('Acceptance'), 
            print ' '.join('{0: >6.2%}'.format(s) for s in accept_count / i)

    print 'Sampling completed!'
    # Remove burn-in draws from the sample.
    sample = sample[burnin:]

    return sample

def log_posterior(beta, m, y, X, beta_0, Sigma_0_inv):
    likelihood = -0.5 * (beta - beta_0).dot(Sigma_0_inv).dot(beta - beta_0)
    likelihood += y.dot(X).dot(beta)
    factor = np.log( 1 + np.exp(X.dot(beta)) )
    likelihood -= m.dot(factor)
    return likelihood

def fit_sim(job):
    # Add the extra '1' in front of the job number.
    job = '1{0:03}'.format(job)
    print 'This is job {0}.'.format(job)

    p = 2
    # Set up prior parameters.
    beta_0 = np.zeros(p)
    Sigma_0_inv = np.eye(p)
    
    # Read values from the data file.
    data_file = 'data/blr_data_{0}.csv'.format(job)
    data = np.genfromtxt(data_file, delimiter = ',', skip_header = 1, 
                         dtype = 'i4,i4,2f8')
    y = data['f0']
    m = data['f1']
    X = data['f2']
    
    # Run the logistic regression.
    scale = np.repeat(0.005, p)
    samp = bayes_logreg(m, y, X, beta_0, Sigma_0_inv, scale)
    
    # Compute percentiles iteratively because Numpy doesn't support vectorized
    # percentiles yet.
    pct = np.zeros(99 * p).reshape(99, p)

    for j in range(0, p):
        for i in range(0, 99):
            pct[i, j] = np.percentile(samp[:, j], i + 1)

    print
    print '25th percentile: {0}'.format(pct[24, :])
    print '50th percentile: {0}'.format(pct[49, :])
    print '75th percentile: {0}'.format(pct[74, :])

    # Save the results.
    result_file = 'results/blr_res_{0}.csv'.format(job)
    np.savetxt(result_file, pct, delimiter = ',', fmt = '%.8f')

# This function samples from the posterior distribution of the breast cancer
# data set model.
def fit_breast_cancer():
    # Read values from the data file.
    data = np.loadtxt('breast_cancer.txt', skiprows = 1, dtype = '10f8, i2', 
                      converters = {10: lambda s: 1 if s == '"M"' else 0})
    X = data['f0']
    y = data['f1']
    n = y.size
    m = np.ones(n)
    # Add an intercept term:
    intercept = np.ones(n).reshape((n, 1))
    X = np.append(X, intercept, 1)
    (n, p) = X.shape

    # Set up prior parameters.
    beta_0 = np.zeros(p)
    Sigma_0_inv = np.eye(p)

    scale = np.array([0.001, 0.8, 0.7, 0.8, 0.8, 0.002, 0.02, 0.8, 0.7, 0.02,
                      0.5])
    # Run the logistic regression.
    samp = bayes_logreg(m, y, X, beta_0, Sigma_0_inv, scale, 
                        niter = 20000, gibbs = True)

    # Save the results.
    result_file = 'results/breast_cancer_res.txt'
    np.savetxt(result_file, samp, fmt = '%.8f')

# This function calculates autocorrelations, parameter estimates, and so on for
# the breast cancer data set, after being fit. Post-processing is in a separate
# function because it takes a long time to fit the data set.
def post_process_breast_cancer():
    data = np.loadtxt('breast_cancer.txt', skiprows = 1, dtype = '10f8, i2', 
                      converters = {10: lambda s: 1 if s == '"M"' else 0})
    X = data['f0']
    y = data['f1']
    n = y.size

    intercept = np.ones(n).reshape((n, 1))
    X = np.append(X, intercept, 1)
    (n, p) = X.shape

    sample_file = 'results/breast_cancer_res.txt'
    samp = np.loadtxt(sample_file)

    # Calculate lag-1 autocorrelations.
    corr = np.zeros(p)
    for j in range(0, p):
        corr[j] = autocorr(samp[:, j])

    print 'Lag-1 autocorrelations:'
    print '  ' + ' '.join('{0: >6.4f}'.format(s) for s in corr)

    ac_file = 'results/breast_cancer_ac.txt'
    np.savetxt(result_file, samp, fmt = '%.8f')

    # Calculate postierior mean.
    est = np.mean(samp, axis = 0)
    print '\nPosterior means:'
    print '  ' + ' '.join('{0: >6.4f}'.format(s) for s in est)

    mean_file = 'results/breast_cancer_mean.txt'
    np.savetxt(result_file, est, fmt = '%.8f')

    # Do a posterior predictive check of the mean.
    sim = predictive_sim(est, X, y)

    y_mean = np.mean(y)
    sim_means = np.mean(sim, axis = 1)
    plt.hist(sim_means, 20)
    plt.vlines(y_mean, 0, 1000, color = 'r', linewidth = 2.0)
    plt.ylim(ymax = n)
    plt.savefig('ppc_hist.png', dpi = 300)

def autocorr(x, lags = 1, normalize = True):
    ac = np.correlate(x, x, mode = 'full')[len(x) - 1:len(x) + lags]
    if normalize:
        ac = ac / ac[0]
    return ac[1:]

def predictive_sim(estimate, X, y, size = 1000):
    prob = np.exp(X.dot(estimate)) / (1 + np.exp(X.dot(estimate)))
    n = y.size

    simulated = np.zeros(size * n).reshape((size, n))
    for i in range(0, n):
        simulated[i, :] = np.random.binomial(1, prob)

    return simulated

# ----- Main Script -----
# Set up command line options.
parser = ap.ArgumentParser(description = 'Perform Bayesian logistic \
                           regression.')
group = parser.add_mutually_exclusive_group(required = True)
group.add_argument('-i', help = 'fit a simulated data set', metavar = 'JOB',
                   type = int)
group.add_argument('-b', help = 'fit the breast cancer data set',
                   action = 'store_true')
group.add_argument('-p', help = 'post-process the breast cancer model fit',
                   action = 'store_true')
args = parser.parse_args()

# Run the program.
if args.b:
    fit_breast_cancer()
elif args.p:
    post_process_breast_cancer()
else:
    fit_sim(args.i)

