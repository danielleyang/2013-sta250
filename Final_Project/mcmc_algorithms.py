# mcmc_algorithms.py
# Author: Nick Ulle
'''Markov chain Monte Carlo algorithms.

This module includes an implementation of the Metropolis algorithm and the
parallel tempering algorithm, as well as several useful probability densities.
'''

from __future__ import division
from math import log, exp, sqrt, pi

import numpy as np

# ----- MCMC Functions
def metropolis(
    u,
    propose,
    num_iter = 10**4,
    burnin = 500,
    start_val = 0):
    '''Sample from a distribution using the Metropolis algorithm.

    Args:
        u               -- a univariate function; the potential function of the
                           target distribution. That is, the log-likelihood.
        propose         -- a univariate function; the proposal function, which
                           proposes a new state given the current state.
        num_iter        -- an integer; the number of sample points to generate.
        burnin          -- an integer; the number of initial sample points to
                           discard.
        start_val       -- the starting value for the chain.

    Returns:
        A tuple consisting of an ndarray of sample points, and the acceptance
        rate.
    '''
    x = np.empty(1 + burnin + num_iter)
    x[0] = start_val

    acceptance = 0
    for iter in range(1, 1 + burnin + num_iter):
        # Update the target chain.
        (x[iter], accepted) = _metropolis_update(x[iter - 1], propose, u)
        acceptance += accepted

    acceptance = acceptance / (burnin + num_iter)
    return x[(1 + burnin):], acceptance

def parallel_tempering(
    u, 
    temperatures, 
    propose, 
    num_iter = 10**4, 
    burnin = 500,
    start_val = 0):
    '''Sample from a distribution using parallel tempering.

    Args:
        u               -- a univariate function; the potential function of the
                           target distribution. That is, the log-likelihood.
        temperatures    -- an ndarray; the temperature ladder, in descending
                           order. This should not include the temperature of
                           the target distribution (i.e., 1).
        propose         -- a univariate function; the proposal function, which
                           proposes a new state given the current state.
        num_iter        -- an integer; the number of sample points to generate.
        burnin          -- an integer; the number of initial sample points to
                           discard.
        start_val       -- the starting value for the chain.

    Returns:
        An ndarray of sample points.
    '''

    m = temperatures.size

    # Only store the current state of the non-target chains.
    chains = np.zeros(m) + start_val

    x = np.empty(1 + burnin + num_iter)
    x[0] = start_val

    acceptance = 0
    iter = 1
    while iter < 1 + burnin + num_iter:
        # Update the m non-target chains.
        for j in range(0, m):
            # Update chain j.
            (chains[j], _) = \
                _metropolis_update(chains[j], propose, 
                                   lambda t: u(t) / temperatures[j])

        # Update the target chain.
        (x[iter], accepted) = _metropolis_update(x[iter - 1], propose, u)
        acceptance += accepted

        iter +=1
        if iter >= 1 + burnin + num_iter:
            break

        # Perform a swap between two chains.
        k = np.random.randint(0, m)
        if k == m - 1:
            # Swap with the target chain.
            (chains[k], x[iter], accepted) = \
                _swap_update(chains[k], x[iter - 1], temperatures[k], 1, u)
            acceptance += accepted

            # Count the update to the target chain as an iteration.
            iter += 1
        else:
            # Swap non-target chains.
            (chains[k], chains[k + 1], _) = \
                _swap_update(chains[k], chains[k + 1], 
                             temperatures[k], temperatures[k + 1], u)

    acceptance = acceptance / (burnin + num_iter)
    return x[(1 + burnin):], acceptance

# ----- MCMC Helper Functions
def _metropolis_update(current, propose, log_likelihood):
    '''Perform a Metropolis update.

    Args:
        current         -- the current state.
        propose         -- a univariate function; the proposal function, which
                           proposes a new state given the current state.
        log_likelihood  -- the log-likelihood of the target distribution.

    Returns:
        The new state of the chain.
    '''
    proposal = propose(current)
    log_odds = log_likelihood(proposal) - log_likelihood(current)

    if log(np.random.uniform()) < log_odds:
        return (proposal, 1)
    else:
        return (current, 0)

def _swap_update(high, low, high_temp, low_temp, u):
    '''Attempt a swap update between high and low temperature states.

    Args:
        high        -- the current high temperature state.
        low         -- the current low temperature state.
        high_temp   -- the high temperature.
        low_temp    -- the low temperature.
        u           -- the potential function.

    Returns:
        A tuple containing the new high and low temperature states,
        respectively.
    '''
    log_odds = (u(low) - u(high)) / high_temp + (u(high) - u(low)) / low_temp 

    if log(np.random.uniform()) < log_odds:
        return (low, high, 1)
    else:
        return (high, low, 0)

# ----- Densities
def dexp(x, rate = 1, a = 0):
    '''Evaluates a (shifted) exponential density.

    Args:
        x       -- the evaluation point.
        rate    -- the rate parameter of the distribution.
        a       -- the left (or right) endpoint of the distribution.

    Returns:
        The value of the specified exponential density at the evaluation point.
    '''
    if a <= x:
        return rate * exp(-rate * (x - a))
    else:
        return 0

def dunif(x, a = 0, b = 1):
    '''Evaluates a uniform density.

    Args:
        x   -- the evaluation point.
        a   -- the left endpoint of the distribution.
        b   -- the right endpoint of the distribution.

    Returns:
        The value of the specified uniform density at the evaluation point.
    '''
    if a <= x and x <= b:
        return x / (a - b)
    else:
        return 0

def dnorm(x, mean = 0, sd = 1):
    '''Evaluates a normal density.

    Args:
        x       -- the evaluation point.
        mean    -- the mean of the distribution.
        sd      -- the standard deviation of the distribution.

    Returns:
        The value of the specified normal density at the evaluation point.
    '''
    return exp(-((x - mean) / sd)**2 / 2) / (sd * sqrt(2 * pi))

def dlaplace(x, mean = 0, rate = 1):
    '''Evaluates a Laplace density.

    Args:
        x       -- the evaluation point.
        mean    -- the mean of the distribution.
        rate    -- the rate of the distribution.

    Returns:
        The value of the specified Laplace density at the evaluation point.
    '''
    return 0.5 * rate * exp(-abs(x - mean) * rate)


