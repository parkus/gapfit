from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from matplotlib import pyplot as plt
import inspect
import sys


# save myself time and effort by standardizing the descriptions of variables
# in the function docstrings
__std_lbls = {
    "x": "x-values of data",
    "y": "y-values of data",
    "x0" : """reference x value in the gap line equation
           (x - x0)*m + y0""",
    "y0" : """y value of the gap line at x0 in 
           (x - x0)*m + y0""",
    "m" : """slope of the gap in the equation 
          (x - x0)*m + y0
          i.e., dy/dx""",
    "sig" : """kernel width for use in KDE
            If rescale is False, then this is in units of y. A value of 0.15 
            usually yields good results.
            If rescale is True, then this is in units of the sample standard
            deviation of y.
            E.g., if the sample standard deviation of the y data after 
            aligning the data to the gap is 2., sig is 0.1, and rescale is  
            True then the kernel width for the KDE will be 0.2.""",
    "rescale" : """True or False
                Whether to rescale the data by their sample standard 
                deviation after aligning to the gap, before computing the 
                KDE at the line center.""",
    "y0_guess" : "Guess at y0 of gap line to start search.",
    "m_guess" : "Guess at slope of gap line to start search.",
    "y0_rng" : """The maximum by which the gap lines are allowed to deviate  \
               from y0 when searching for the best fit.""",
    "min_kws" : """Dictioary of keywords to be passed to scipy.optimize.minimize 
                when 
                numerically searching for a fit. Can be used, e.g., 
                to change the minimization method used."""}


def gap_line(x, x0, y0, m):
    """
    Return the y-value of the gap at the provided x values given.

    Simply computes the function for a line
    y = (x - x0) * m + y0
    """
    return (x - x0) * m + y0



def kde(xx, x, xsig):
    """
    Compute the density of points at xx using Kernel Density Estimation (KDE).

    Parameters
    ----------
    xx : points at which to evaluate kde
    x : {x}
    xsig : kernel width for kde in units of x

    Returns
    -------
    KDE estimate of point density at xx.
    """

    # groom input
    if np.isscalar(xsig):
        xsig = xsig*np.ones_like(x)
    if np.isscalar(xx):
        xx = np.array([xx])

    # compute values of Gaussian centered on the data x at every user point xx
    # and sum them up
    normfacs = 1./(xsig*np.sqrt(2*np.pi))/len(x)
    expfacs = 1./2/xsig**2
    exponents = -(x[:,None] - xx[None,:]) ** 2 * expfacs[:,None]
    terms = normfacs[:,None] * np.exp(exponents)
    return np.sum(terms, 0)
kde.__doc__.format(**__std_lbls)


def density_in_gap(x, y, x0, y0, m, sig, rescale=True):
    """
    Compute the depth of the gap after "aligning" the data to the provided 
    parameters for a line.
    
    "Aligning" is loosely used here -- the data logRp
    
    Parameters
    ----------
    x : {x}
    y : {y}
    x0 : {x0}
    y0 : {y0}
    m : {m}
    sig : {sig}
    rescale : {rescale}

    Returns
    -------
    The average point density (per y unit) along the gap, estimated using KDE.

    """

    # subtract the gap y values from the data y values
    Y = y - gap_line(x, x0, y0, m)

    # normalize the standard deviation of the aligned data
    if rescale:
        Y = Y/np.std(Y)

    # compute the denisyt of points at the gap (at 0 in the aligned Y
    # coordinates)
    return kde(0, Y, sig)
density_in_gap.__doc__.format(**__std_lbls)


def bootstrap_fit_gap(x, y, x0, y0_guess, m_guess, sig, y0_rng=0.2,
                      nboots=1000, min_kws=None, bad_fits="50% tolerance"):
    """

    Parameters
    ----------
    x : {x}
    y : {y}
    x0 : {x0}
    y0_guess : {y0_guess}
    m_guess : {m_guess}
    sig : {sig}
    y0_rng : {y0_rng}
    nboots : Number of times to bootstrap the data and fit a gap.
    min_kws : {min_kws}
    bad_fits : "XX% tolerance"|"raise"|"ignore"
        How to handle fits that do not succeed. Default is "50% tolerance"
            "XX% tolerance" : If XX% of the fits do not succeed, raise an error.
            "raise" : Raise an error immediately if any fit does not succeed.
            "ignore" : Ignore any fits that don't succeed and just return
                       those that do.

    Returns
    -------
    boots : list
        Bootstrapped fit values as a list of fit pairs,
        e.g., [(y00, m0), (y01, m1), ..., (y0n, mn)] where n=nboots]

    """
    npts = len(x)
    iall = np.arange(npts) # indices of all points
    boots = []
    print("Bootstrapping...")
    for _ in tqdm(range(nboots)):
        # detrmine which points will be bootstrapped
        ichoice = np.random.choice(iall, npts, replace=True)
        xx, yy = x[ichoice], y[ichoice]
        try:
            # now try to fit those points
            fit = fit_gap(xx, yy, x0, y0_guess, m_guess, sig, y0_rng,
                          min_kws=min_kws)
            boots.append(fit)

        # if it doesn't work and the user wants them to work, raise the error
        except AssertionError:
            if bad_fits == "raise":
                raise

    # if the user is okay with some fraction  of fits not working, check that
    # that fraction was not exceed
    if "tolerance" in bad_fits:
        tol = float(bad_fits[:2])/100
        if len(boots)/nboots < tol:
            raise ValueError("More than {}% of fits caused errors."
                             "".format(tol*100))

    return boots
bootstrap_fit_gap.__doc__.format(**__std_lbls)


def fit_gap(x, y, x0, y0_guess, m_guess, sig, y0_rng, min_kws=None):
    """

    Parameters
    ----------
    x : {x}
    y : {y}
    x0 : {x0}
    y0_guess : {y0_guess}
    m_guess : {m_guess}
    sig : {sig}
    y0_rng : {y0_rng}
    min_kws : {min_kws}

    Returns
    -------
    y0, m : Parameters of the best fit line
    """

    if min_kws is None:
        min_kws = {}
    # for some reason most minimizing methods don't converge, but Nelder-Mead
    # does
    min_kws.setdefault("method", 'Nelder-Mead')

    # create cost function that is simply the density of points along  the
    # gap line
    def cost_function(vec):
        logr0, m = vec

        # if outside of the allowable range for y0, return infinity
        if np.abs(logr0 - y0_guess) > y0_rng/2.:
            return np.inf

        return density_in_gap(x, y, x0, logr0, m, sig, rescale=True)

    # now minimize the cost_function
    result = minimize(cost_function, (y0_guess, m_guess), **min_kws)

    assert result.success # make sure the fit succeeded

    # return  the best fit y0, m
    return result.x
fit_gap.__doc__.format(**__std_lbls)


def uncertainty_from_boots(boots):
    """
    Convenience function to return the median with 1-sigma error bars (or,
    technically 16-84th percentile range) from bootstrapped fits.

    Parameters
    ----------
    boots : output from bootstrap_fit_gap

    Returns
    -------
    triplets : array
        triplets[0] is the median value, positive uncertainty, and negative
          uncertainty of the y0 of the bootstrapped fit values
        triplets[1] is the same for m
    """
    fit = np.median(boots, axis=0)
    boots = np.vstack(boots)
    intervals = np.percentile(boots, [16, 84], axis=0)
    err1 = intervals[1] - fit
    err2 = fit - intervals[0]
    return np.vstack((fit, err1, err2)).T


def test():
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
    print("")
    print("")
    print("Input gap line parameters:")
    print("  y0 = {}".format(y0))
    print("  m = {}".format(m))
    print("Retrieved gap line parameters:")
    print("  y0 = {:.2f} +{:.2f}/-{:.2f}".format(*y0trip))
    print("  m = {:.2f} +{:.2f}/-{:.2f}".format(*mtrip))

    # now plot up the results
    fig, ax = plt.subplots(1,1)
    ax.plot(x, y, 'k.')
    xlim = np.array(ax.get_xlim())
    ygap = gap_line(xlim, x0, y0, m)
    ax.plot(xlim, ygap, color='C0')

    # ugh, I hate making error regions
    boots = np.asarray(boots)
    y0boot, mboot = boots.T
    xgrid = np.linspace(*xlim, num=300)
    ygap_grid = gap_line(xgrid[None,:], x0, y0boot[:,None], mboot[:,None])
    ygap_lo, ygap_hi = np.percentile(ygap_grid, (16, 84), axis=0)
    ax.fill_between(xgrid, ygap_lo, ygap_hi, color='C0', alpha=0.3)