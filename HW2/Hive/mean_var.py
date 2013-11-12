#!/usr/bin/env python2
# mean_var.py
# Author: Nick Ulle

import numpy as np

# ----- Main Script
i = 0
groups = np.zeros(1000, np.int)
values = np.zeros(1000, np.float)
with open('mini_groups.txt', 'r') as lines:
    for line in lines:
        # Do resizing at beginning of loop.
        if i == groups.size:
            groups.resize((groups.size + 1000, ))
            values.resize((values.size + 1000, ))
            print 'Resize needed at i = {0}'.format(i)

        (group, value) = line.split('\t', 1)
        groups[i] = np.int(group)
        values[i] = np.float(value)

        i += 1

groups = groups[:i]
values = values[:i]

# Sort the groups.
ordering = np.argsort(groups)
groups = groups[ordering]
values = values[ordering]

means = np.zeros(groups[-1])
vars = np.zeros(groups[-1])
for group in range(groups[0], groups[-1] + 1):
    means[group - 1] = np.mean(values[groups == group])
    vars[group - 1] = np.var(values[groups == group])

print '\nMeans are:'
print means
print '\nVariances are:'
print vars

