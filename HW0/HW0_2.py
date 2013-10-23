# !/bin/python2
# HW0_2.py
# Author: Nick Ulle
# Description:

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
#import numpy.random

#numpy.random.uniform(size = 10)
x = scipy.stats.uniform.rvs(0, 2 * np.pi, 10000)
y = scipy.stats.uniform.rvs(0, 1, 10000)

u = y * np.cos(x)
v = y * np.sin(x)

plt.plot(u, v, ',')
plt.show()

r = np.sqrt(u**2 + v**2)

plt.hist(r, 50, facecolor = 'g', normed = 1, alpha = 0.75)
plt.grid(True)
plt.show()

