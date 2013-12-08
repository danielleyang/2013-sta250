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
    if blocks > 512:
        raise Exception('Maximum sample size in a single call is 33,554,432.')

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

def main():
    gpu_random_activate(4140)

    # Try generating from a truncated normal. For vectors larger than about
    # 10^6, numpy.mean becomes numerically unstable.
    n = np.int32(10 ** 5)
    mean = np.zeros(n, np.float32)
    sd = np.ones(n, np.float32)
    a = np.ones(n, np.float32) * -np.float32('inf')
    b = np.ones(n, np.float32) * -8

    res = gpu_trunc_norm(n, mean, sd, a, b)

    print np.mean(res)

    gpu_random_deactivate()

if __name__ == '__main__':
    main()

