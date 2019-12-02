#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Dirk Eilander (contact: dirk.eilander@vu.nl)
# Created: Nov 2nd 2018
# 

import xarray as xr
import numpy as np
from scipy.stats import norm

__all__ = ['confidence_interval']

def confidence_interval(da, statfunction=np.nanmean, alpha=0.05, 
    n_samples=10000, method='bca', dim='time'):
    """Returns the bootstrap confidence interval for <statfunction> on 
    one dimension of the xarray DataArray.
    
    Parameters
    ----------
    da: xarray DataArray
        Input data array
    statfunction: statistical function (default np.nanmean)
        This function should accept samples of data from ``data``. 
        It is applied to these samples individually. 
    alpha: float (default=0.05)
        The percentiles to use for the confidence interval. If this
        is a float the returned values are (alpha/2, 1-alpha/2) 
        percentile confidence intervals. If it is an iterable, 
        alpha is assumed to be an iterable of each desired percentile.
    n_samples: int (default=10000)
        The number of bootstrap samples to use 
    method: option of 'bca' and 'simple'
        choice of method to calculate the confidence interval.
        simple -> bootstrap interval 
        bca -> bias-corrected and accelerated bootstrap interval
    dim : str, optional
        name of the core dimension (the default is 'time')
    
    Returns
    -------
    xarray DataArray
        The core dimension is reduced to the upper and lower confidence 
        boundary (size=2) and returned at the last dimension.
    """
    # confidence interval parameters
    ci_kwargs = dict(
        statfunction=statfunction, 
        alpha=alpha, 
        n_samples=n_samples, 
        method=method
        )
    # apply ufunc parameters
    kwargs = dict(               
        input_core_dims=[[dim]], 
        output_core_dims=[['ci_bounds']],
        dask='parallelized', 
        output_dtypes=[float],       
        # on output, <dim> is reduced to length 2 with upper and lower ci bounds
        output_sizes={'ci_bounds': 2},
        # vectorize to apply over 1 dim
        vectorize=True
    )
    # apply ci_nd over dim
    ci = xr.apply_ufunc(_ci, da, kwargs=ci_kwargs, **kwargs)
    ci['ci_bounds'] = xr.Variable('ci_bounds', np.array([alpha/2, 1-alpha/2]))
    return ci

def _ci(data, statfunction=np.nanmean, alpha=0.05, n_samples=10000, method='bca'):
    """Returns the bootstrap confidence interval for <statfunction> on data on 1D arrays. 

    Parameters
    ----------
    data: array_like
        Input data
    statfunction: statistical function (default np.nanmean)
        This function should accept samples of data from ``data``. 
        It is applied to these samples individually. 
    alpha: float (default=0.05)
        The percentiles to use for the confidence interval. If this
        is a float the returned values are (alpha/2, 1-alpha/2) 
        percentile confidence intervals. If it is an iterable, 
        alpha is assumed to be an iterable of each desired percentile.
    n_samples: int (default=10000)
        The number of bootstrap samples to use 
    method: option of 'bca' and 'simple'
        choice of method to calculate the confidence interval.
        simple -> bootstrap interval 
        bca -> bias-corrected and accelerated bootstrap interval

    Returns
    -------
    numpy array with shape (2,)
        upper and lower confidence boundary 
    """
    assert data.ndim==1, "only tested for 1D arrays"
    n = data.shape[0]
    alphas = np.array([alpha/2, 1-alpha/2])
    stat = statfunction(data[np.random.randint(n, size=(n, n_samples))], axis=0)
    stat.sort()

    if method == 'simple':
        avals = alphas

    elif method == 'bca':
        ostat = statfunction(data, axis=0)
        # The bias correction value.
        z0 = norm.ppf(( 1.0*np.sum(stat < ostat, axis=0)) / n_samples )
        # Statistics of the jackknife distribution
        jstat = statfunction(data[_jackknife_sample(np.arange(n, dtype=np.int16))], axis=0)
        jmean = jstat.mean(axis=0)
        # Acceleration value
        av = np.sum((jmean - jstat)**3, axis=0) / (6.0 * np.sum((jmean - jstat)**2, axis=0)**1.5)
        zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)
        # Bias corrected alphas
        avals = norm.cdf(z0 + zs/(1-av*zs))

    #get confidence interval based on sorted bootstrapped statistic
    nvals = np.round((n_samples-1)*avals)
    nvals = np.nan_to_num(nvals).astype('int')

    return stat[nvals]

def _jackknife_sample(data):
    """jackknife sample from data"""
    n = data.shape[0]
    jsample = np.empty([n, n-1], dtype=data.dtype)
    for i in range(n):
        jsample[i] = np.delete(data, i)
    return jsample