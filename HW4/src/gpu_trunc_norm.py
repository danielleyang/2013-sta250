#!/usr/bin/env python2.7
# gpu_trunc_norm.py
# Author: Nick Ulle

from __future__ import division
from math import ceil, sqrt

import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda

# ----- PTX Module Setup
# Import PTX module and setup its functions.
_gpu_module = cuda.module_from_file('bin/gpu_trunc_norm.ptx')

_gpu_trunc_norm = _gpu_module.get_function('gpu_trunc_norm')
_gpu_curand_init = _gpu_module.get_function('gpu_curand_init')
_gpu_curand_deinit = _gpu_module.get_function('gpu_curand_deinit')

# ----- Globals
# A pointer to the curand RNG states.
_rng_state = None

# The total number of threads per block.
# This should match NUM_RNG in gpu_trunc_norm.cu
_NUM_THREADS = 128

# ----- Functions
def gpu_trunc_norm(n, mean, sd, a, b):
    blocks = int(ceil(sqrt(n / _NUM_THREADS)))

    n = np.int32(n)
    mean = mean.astype(np.float32)
    sd = sd.astype(np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    result = np.empty(n, np.float32)

    _gpu_trunc_norm(n, cuda.In(mean), cuda.In(sd), cuda.In(a), cuda.In(b),
                    cuda.Out(result), cuda.In(_rng_state),
                    block = (_NUM_THREADS, 1, 1), grid = (blocks, blocks))

    return result

def gpu_curand_init(seed):
    global _rng_state
    global _NUM_THREADS

    if _rng_state is not None:
        gpu_curand_deinit()

    seed = np.int32(seed)
    _rng_state = np.zeros(_NUM_THREADS, np.intp)

    _gpu_curand_init(seed, cuda.Out(_rng_state), 
                     block = (_NUM_THREADS, 1, 1), grid = (1, 1))

def gpu_curand_deinit():
    global _rng_state
    global _NUM_THREADS

    _gpu_curand_deinit(cuda.In(_rng_state), 
                       block = (_NUM_THREADS, 1, 1), grid = (1, 1))
    _rng_state = None

# ----- Script
def main():
    gpu_curand_init(15)

    n = 10
    mean = np.zeros(n)
    sd = np.ones(n)
    a = np.ones(n) * 4
    b = np.ones(n) * np.float32('inf')

    res = gpu_trunc_norm(n, mean, sd, a, b)
    
    print res

    gpu_curand_deinit()

if __name__ == '__main__':
    main()

