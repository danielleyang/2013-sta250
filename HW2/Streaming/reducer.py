#!/usr/bin/env python2
# reducer.py
# Author: Nick Ulle

import sys

# ----- Main Script
bin = None
current_bin = None
current_count = 0

for line in sys.stdin:
    (bin, count) = line.strip().split('\t', 1)

    # Try to convert count to an integer.
    try:
        count = int(count)
    except ValueError:
        continue

    # Check whether this element is in the current bin.
    if bin == current_bin:
        current_count += count
    else:
        # This element is in a new bin, so output results for the current bin.
        if current_bin:
            print '{0},{1}'.format(current_bin, current_count)

        current_bin = bin
        current_count = 1

# Make sure to output the results for the final bin.
if bin == current_bin:
    print '{0},{1}'.format(current_bin, current_count)

