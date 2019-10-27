gapfit
======
Gapfit is a module created to fit lines to the exoplanet radius gap and provide boostrapped errors. However, it could theoretically be used to fit a line to a gap in any dataset.

Gapfit cannot find gaps on its own, it can only find the parameters for a line that best fits that gap. It requires human guidance to get it started with a good guess at the line of the gap. It works by minimizing the density of the data points along the gap. Their density is estimated using Kernel Density Estimation to ensure smooth changes in density as different fit lines are attempted.


# Quick Start
The below is straight from the gapfit.test() function :)

```
import numpy as np
from matplotlib import pyplot as plt
import gapfit

# make some fake normally distributed data
np.random.seed(42)
n = 600
x, y = np.random.randn(2, n)

# now define the parameters of a line
#  y = (x0 - x)*m + y0
# to describe a fake gap
x0 = 0
m = 1
y0 = -0.5

# and clear out points within the vicinity fo that gap
ygap = gap_line(x, x0, y0, m)
dy = y - ygap
remove = np.abs(dy) < 0.3
x, y = [a[~remove] for a in (x, y)]

# now bootstrap fits to the gappy data
y0_guess = -0.6
m_guess = 0.9
sig = 0.15
y0_rng = 0.5
boots = bootstrap_fit_gap(x, y, x0, y0_guess, m_guess, sig, y0_rng,
                          nboots=300)

# print result
triplets = uncertainty_from_boots(boots)
y0trip, mtrip = triplets
print("Input gap line parameters:")
print("  y0 = {}".format(y0))
print("  m = {}".format(m))
print("Retrieved gap line parameters:")
print("  y0 = {:.2f} +{:.2f}/-{:.2f}".format(*y0trip))
print("  m = {:.2f} +{:.2f}/-{:.2f}".format(*mtrip))
# Input gap line parameters:
#  y0 = -0.5
#  m = 1
# Retrieved gap line parameters:
#  y0 = -0.55 +0.02/-0.02
#  m = 1.01 +0.02/-0.03

# now plot up the results
fig, ax = plt.subplots(1,1)
ax.plot(x, y, 'k.')
xlim = np.array(ax.get_xlim())
ygap = gap_line(xlim, x0, y0, m)
ax.plot(xlim, ygap, color='C0')

# with an error region
boots = np.asarray(boots)
y0boot, mboot = boots.T
xgrid = np.linspace(*xlim, num=300)
ygap_grid = gap_line(xgrid[None,:], x0, y0boot[:,None], mboot[:,None])
ygap_lo, ygap_hi = np.percentile(ygap_grid, (16, 84), axis=0)
ax.fill_between(xgrid, ygap_lo, ygap_hi, color='C0', alpha=0.3)
