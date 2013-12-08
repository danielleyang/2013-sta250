#!/usr/bin/env python2.7
# gpu_trunc_norm.py
# Author: Nick Ulle

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np

def main():
    print "Hello!"
    mod = cuda.module_from_file('gpu_trunc_norm.ptx')
    gpu_trunc_normal = mod.get_function('gpu_trunc_normal')

    # Try getting generating some values.
    n = np.int32(100)
    mean = np.zeros(n, np.float32)
    sd = np.ones(n, np.float32)
    a = np.ones(n, np.float32) * 10
    b = np.ones(n, np.float32) / 0
    res = np.zeros(n, np.float32)
    print b

    gpu_trunc_normal(n, cuda.In(mean), cuda.In(sd), cuda.In(a),
                     cuda.In(b), cuda.Out(res),
                     block = (7,1,1), grid = (16,1))

    print res

if __name__ == '__main__':
    main()

