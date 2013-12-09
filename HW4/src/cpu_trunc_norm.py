#!/usr/bin/env python2.7
# cpu_trunc_norm.py
# Author: Nick Ulle

import ctypes as ct

import numpy as np

# ----- C Library Setup
# Define a C float pointer type.
_c_float_p = np.ctypeslib.ndpointer(dtype = np.float32)

# Import C library and setup its functions.
_cpu_lib = np.ctypeslib.load_library('cpu_trunc_norm.so', './bin')

_set_seed = _cpu_lib.set_seed
_set_seed.argtypes = [ct.c_int]
_set_seed.restype = None

_cpu_trunc_norm = _cpu_lib.cpu_trunc_norm
_cpu_trunc_norm.argtypes = [ct.c_int] + [_c_float_p] * 5
_cpu_trunc_norm.restype = None

# ----- Functions
def cpu_trunc_norm(n, mean, sd, a, b):
    mean = mean.astype(np.float32)
    sd = sd.astype(np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    result = np.empty(n, np.float32)

    _cpu_trunc_norm(n, mean, sd, a, b, result)

    return result

def set_seed(seed):
    _set_seed(seed)

# ----- Script
def main():
    set_seed(15)

    n = 10
    mean = np.zeros(n)
    sd = np.ones(n)
    a = np.ones(n) * 4
    b = np.ones(n) * np.float32('inf')

    res = cpu_trunc_norm(n, mean, sd, a, b)
    
    print res

if __name__ == '__main__':
    main()

