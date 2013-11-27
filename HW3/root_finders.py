#!/usr/bin/env python2
# root_finders.py
# Author: Nick Ulle
# Description:
'''Root-finding methods.

This module includes implementations of the bisection algorithm and the
Newton-Raphson algorithm for root-finding.
'''

from __future__ import division
import math

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

    for iter in range(1, max_iter + 1):
        mid = (l + u) / 2
        value = f(mid)

        if verbose and iter % verbose == 0:
            print 'Iteration {0}:'.format(iter)
            print '  Lower:  f({0:.4g}) = {1:.4g}'.format(l, f(l))
            print '  Middle: f({0:.4g}) = {1:.4g}'.format(mid, f(mid))
            print '  Upper:  f({0:.4g}) = {1:.4g}'.format(u, f(u))

        if abs(value) < tol:
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
    x = x0
    for iter in range(1, max_iter + 1):
        # This loop could be made marginally faster by reducing the number of
        # calls to f, but at the cost of clarity.
        if verbose and iter % verbose == 0:
            print 'Iteration {0}:'.format(iter)
            print '  f({0:.4g}) = {1:.4g}'.format(x, f(x))
            print '  df({0:.4g}) = {1:.4g}'.format(x, df(x))

        if abs(f(x)) < tol:
            break

        x = x - (f(x) / df(x))

    return (x, f(x), iter)

def main():
    '''Demo the algorithms on Rao\'s linkage problem.'''
    print '\nRunning as a script.'
    print 'MLEs for Rao\'s linkage problem will be computed.'
    print 'Use `import root_finders` to use the root-finding functions' + \
        ' directly.\n'

    def log_likelihood(x):
        return 125*math.log(2 + x) + 38*math.log(1 - x) + 34*math.log(x)

    def score(x):
        return 125 / (2 + x) - 38 / (1 - x) + 34 / x

    def d_score(x):
        return -125 / (2 + x)**2 - 38 / (1 - x)**2 - 34 / x**2

    TOL = 10**-10

    print 'On (-2, 0):'
    (root, _, iter) = root_bisection(score, (-1.9999, -0.0001), tol = TOL)
    print '  Bisection found {0:.4} in {1} iterations' \
        .format(root, iter)
    (root, _, iter) = root_newton(score, d_score, -1, tol = TOL)
    print '  Newton-Raphson found {0:.4} in {1} iterations' \
        .format(root, iter)

    print 'On (0, 1):'
    (root, _, iter) = root_bisection(score, (0.0001, 0.9999), tol = TOL)
    print '  Bisection found {0:.4} in {1} iterations, value is {2:.4}' \
        .format(root, iter, log_likelihood(root))
    (root, _, iter) = root_newton(score, d_score, 0.5, tol = TOL)
    print '  Newton-Raphson found {0:.4} in {1} iterations, value is {2:.4}' \
        .format(root, iter, log_likelihood(root))

if __name__ == '__main__':
    main()

