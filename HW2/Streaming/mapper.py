#!/usr/bin/env python
# mapper.py
# Author: Nick Ulle
# Description:
#   This is the mapper script for HW2, part 2. It takes tab-delimited data from
#   stdin and writes tab-delimited (key, value) pairs to stdout.

import sys

# ----- External Modules
import numpy as np

# ----- Main Script
for line in sys.stdin:
    words = line.strip().split('\t', 1)

    # Try to convert to floats.
    try:
        values = np.float64(words) * 10
    except ValueError:
        continue

    # Floor and ceiling to one decimal place. 
    # This gives (x_lo, y_lo) and (x_hi, y_hi).
    (x_lo, y_lo) = np.floor(values) / 10
    (x_hi, y_hi) = np.ceil(values) / 10

    # Write the (key, value) pair to stdout.
    print '{0:.1f},{1:.1f},{2:.1f},{3:.1f}\t1'.format(x_lo, x_hi, y_lo, y_hi)

