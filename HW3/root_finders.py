#!/bin/usr/env python
# root_finders.py
# Author: Nick Ulle
# Description:
#   This module includes implementations of the bisection algorithm and the
#   Newton-Raphson algorithm for root-finding.

from __future__ import division

def root_bisection(f, interval, tol = 10 ** -6, max_iter = 10 ** 6, 
                   verbose = False):
    '''Find a root of a function using the bisection algorithm.

    Args:
        f           -- target function.
        interval    -- a tuple with lower and upper endpoints for the search.
        tol         -- tolerance for stopping.
        max_iter    -- maximum number of iterations.
        verbose     -- whether to print debug messages. If a positive integer n
                       is supplied, debug messages will be printed every n
                       iterations.

    Returns:
        A tuple containing the root, the value of the function at the root, and
        the number of iterations run.

    Raises:
        Exception: Invalid interval, signs at endpoints are the same.
    '''
    (l, u) = interval
    if f(l) * f(u) > 0:
        # Throw some kind of error.
        raise Exception('Invalid interval, signs at endpoints are the same.')

    iter = 0
    while True:
        iter += 1
        mid = (l + u) / 2
        value = f(mid)

        if verbose and iter % verbose == 0:
            print 'Iteration {0}:'.format(iter)
            print '  Lower:  f({0:.4g}) = {1:.4g}'.format(l, f(l))
            print '  Middle: f({0:.4g}) = {1:.4g}'.format(mid, f(mid))
            print '  Upper:  f({0:.4g}) = {1:.4g}'.format(u, f(u))

        if abs(value) < tol or iter >= max_iter:
            # Nothing left to do.
            break
        elif f(l) * value < 0:
            # The new interval is (l, mid).
            u = mid
        else:
            # The new interval is (mid, u).
            l = mid

    return (mid, value, iter)


def root_newton(f, df, x0, tol = 10 ** -6, max_iter = 10 ** 6, 
                verbose = False):
    '''Find a root of a function using the Newton-Raphson algorithm.

    Args:
        f           -- target function.
        df          -- derivative of target function.
        x0          -- initial guess.
        tol         -- tolerance for stopping.
        max_iter    -- maximum number of iterations.
        verbose     -- whether to print debug messages. If a positive integer n
                       is supplied, debug messages will be printed every n
                       iterations.

    Returns:
        A tuple containing the root, the value of the function at the root, and
        the number of iterations run.
    '''
    pass

