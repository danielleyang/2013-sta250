#!/usr/bin/env python2.7
# gpu_trunc_norm.py
# Author: Nick Ulle

from __future__ import division
from math import ceil, sqrt

import pycuda.autoinit
import pycuda.driver as cuda

import numpy as np

# ----- PTX Setup
_gpu_module = cuda.module_from_file('gpu_trunc_norm.ptx')

_gpu_random_activate = _gpu_module.get_function('gpu_random_activate')
_gpu_random_deactivate = _gpu_module.get_function('gpu_random_deactivate')
_gpu_trunc_norm = _gpu_module.get_function('gpu_trunc_norm')

# ----- Globals & Constants
_rng_state = None

# This should match NUM_RNG in gpu_trunc_norm.cu
_NUM_THREADS = 128

_MAX_ITER = 100

# ----- Main Script

def gpu_random_activate(seed):
    global _rng_state, _NUM_THREADS

    if _rng_state:
        print "RNG already initialized! Use gpu_random_free() to free."
        return

    seed = np.int32(seed)
    _rng_state = np.zeros(_NUM_THREADS, np.intp)

    _gpu_random_activate(seed, cuda.Out(_rng_state), 
                         block = (_NUM_THREADS, 1, 1), grid = (1, 1))

def gpu_random_deactivate():
    global _rng_state, _NUM_THREADS

    _gpu_random_deactivate(cuda.In(_rng_state), 
                           block = (_NUM_THREADS, 1, 1), grid = (1, 1))
    _rng_state = None

def gpu_trunc_norm(n, mean, sd, a, b):
    blocks = int(ceil(sqrt(n / _NUM_THREADS)))
    #if blocks > 512:
    #    raise Exception('Maximum sample size in a single call is 33,554,432.')

    n = np.int32(n)
    mean = mean.astype(np.float32)
    sd = sd.astype(np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    result = np.zeros(n, np.float32)

    _gpu_trunc_norm(n, cuda.In(mean), cuda.In(sd), cuda.In(a), cuda.In(b),
                    cuda.Out(result), cuda.In(_rng_state),
                    block = (_NUM_THREADS, 1, 1), grid = (blocks, blocks))
    return result

def tail_norm(mean, sd, a):
    a = (a - mean) / sd
    tail = -1 if a < 0 else 1
    a *= tail
    for i in range(0, _MAX_ITER):
        # Generate z ~ EXP(alpha) + a.
        alpha = (a + np.sqrt(a**2 + 4)) / 2
        u = np.random.uniform()
        z = -np.log(1 - u) / alpha + a

        # Compute g(z).
        gz = np.exp(-(z - alpha)**2 / 2)

        u = np.random.uniform()
        if u <= gz:
            break

    return sd * tail * z + mean

def trunc_norm(n, mean, sd, a, b):
    result = np.zeros(n)
    for idx in range(0, n):
        if not np.isfinite(a[idx]) and  b[idx] <= mean[idx]:
            draw = tail_norm(mean[idx], sd[idx], b[idx])
        elif not np.isfinite(b[idx]) and a[idx] >= mean[idx]:
            draw = tail_norm(mean[idx], sd[idx], a[idx])
        else:
            for i in range(0, _MAX_ITER):
                draw = np.random.normal(mean[idx], sd[idx], 1)
                if a[idx] <= draw and draw <= b[idx]:
                    break
        result[idx] = draw
    return result

def time_sample(k, mean = 2, sd = 1, a = 0, b = 1.5):
    n = 10 ** k
    mean = np.ones(n) * mean
    sd = np.ones(n) * sd
    a = np.ones(n) * a
    b = np.ones(n) * b

    start = cuda.Event()
    end = cuda.Event()

    start.record()
    gpu_trunc_norm(n, mean, sd, a, b)
    end.record()
    end.synchronize()
    gpu_time = start.time_till(end) * 1e-3

    start.record()
    trunc_norm(n, mean, sd, a, b)
    end.record()
    end.synchronize()
    cpu_time = start.time_till(end) * 1e-3

    return gpu_time, cpu_time

def check_sample(n = 10 ** 5, mean = 2, sd = 1, a = 0, b = 1.5):
    n = 10 ** 5
    mean = np.ones(n) * mean
    sd = np.ones(n) * sd
    a = np.ones(n) * a
    b = np.ones(n) * b

    gpu_samp = gpu_trunc_norm(n, mean, sd, a, b)
    cpu_samp = trunc_norm(n, mean, sd, a, b)

    return np.mean(gpu_samp), np.mean(cpu_samp)

def main():
    gpu_random_activate(140)
    np.random.seed(150)

    # Print sample means as diagnostics. These should agree with each other and
    # with the theoretical values.
    print 'N(2, 1) on (0, 1.5) sample means: ', check_sample()
    print 'N(0, 2) on (4, Inf) sample means: ', \
        check_sample(mean = 0, sd = 2, a = 4, b = np.float32('inf'))
    print 'N(-1, 1) on (-Inf, -4) sample means: ', \
        check_sample(mean = -1, sd = 1, a = -np.float32('inf'), b = -4)

    # Run timings.
    time = np.zeros((2, 8))
    for k in range(1, 9):
        time[:, k - 1] = time_sample(k)
        print 'Finished k = {0}.'.format(k)

    np.savetxt('output/timings.csv', time, delimiter = ',')

    gpu_random_deactivate()

if __name__ == '__main__':
    main()

