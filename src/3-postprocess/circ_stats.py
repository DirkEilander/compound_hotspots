#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Dirk Eilander (contact: dirk.eilander@vu.nl) and Anais Couasnon (contact anais.couasnon@vu.nl)
# Created: Nov 2nd 2018
#
# Xarray wrapper around astropy.stats.circstats functions 

# TODO: find a way to implement weights, both if weights == None, type(weights) == np.ndarray or type(weights) == xr.DataArray 

import xarray as xr
import numpy as np

__all__ = ['circ_mean', 'circ_var', 'circ_corr', 'rayleightest', 'angle_diff']

# circular stats
def circ_mean(circ_data, dim='time'):
    """Returns the mean of circular data [radian].
    
    Parameters
    ----------
    circ_data : xarray DataArray
        circular data [radian]
    dim : str, optional
        name of the core dimension (the default is 'time')
    
    Returns
    -------
    xarray DataArray
        circular mean
    """
    # wrap numpy function
    theta = xr.apply_ufunc(_circmean, circ_data, #kwargs={'weights':weights}, 
        input_core_dims=[[dim]], dask='parallelized', output_dtypes=[float])
    theta.name='theta'
    theta.attrs.update(unit='radian', description='circular mean')
    return theta

def circ_var(circ_data, dim='time'):
    """Returns the variance of circular data [radian].
    
    Parameters
    ----------
    circ_data : xarray DataArray
        circular data [radian]
    dim : str, optional
        name of the core dimension (the default is 'time')
    
    Returns
    -------
    xarray DataArray
        circular variance
    """
    circvar = xr.apply_ufunc(_circvar, circ_data, #kwargs={'weights':weights},
        input_core_dims=[[dim]], dask='parallelized', output_dtypes=[float])
    circvar.name='circ_var'
    circvar.attrs.update(unit='radian', description='circular variance')
    return circvar

def circ_corr(alpha, beta, dim='time'):
    """Returns the circular correlation coefficient between two arrays of
     circular data. [radian]. 
    
    Parameters
    ----------
    alpha : xarray DataArray
        circular data [radian]
    beta : xarray DataArray
        circular data [radian]
    dim : str, optional
        name of the core dimension (the default is 'time')
    
    Returns
    -------
    xarray DataArray
        circular correlation coefficient
    """
    # wrap numpy function
    rho = xr.apply_ufunc(_circcorrcoef, alpha, beta, 
        # kwargs={'weights_alpha':weights_alpha, 'weights_beta':weights_beta}, 
        input_core_dims=[[dim], [dim]], dask='parallelized', output_dtypes=[float])
    rho.name = 'circ_corrcoef'
    rho.attrs.update(unit='-', description='circular correlation coefficient')    
    return rho

def rayleightest(circ_data, dim='time'):
    """Returns the p-value for the Rayleigh test of uniformity

    This test is  used to identify a non-uniform distribution, i.e. it is
    designed for detecting an unimodal deviation from uniformity. More
    precisely, it assumes the following hypotheses:
    - H0 (null hypothesis): The population is distributed uniformly around the
    circle.
    - H1 (alternative hypothesis): The population is not distributed uniformly
    around the circle.

    Parameters
    ----------
    circ_data : xarray DataArray
        circular data [radian]
    weights : xarray DataArray, optional
        weights of the circular data (the default is None)
    dim : str, optional
        name of the core dimension (the default is 'time')
    
    Returns
    -------
    xarray DataArray
        p-value
    """
    p_value = xr.apply_ufunc(_rayleightest, circ_data, #kwargs={'weights':weights},
        input_core_dims=[[dim]], dask='parallelized', output_dtypes=[float])
    p_value.name = 'rayleigh_p'
    p_value.attrs.update(unit='', description='p-value for rayleigh test of uniformity')
    return p_value

def angle_diff(rad1, rad2):
    """Returns the smallest angle between two circular angles [rad]

    Parameters
    ----------
    rad1 : xarray DataArray
        circular data [radian]
    rad2 : xarray DataArray
        circular data [radian]
    
    Returns
    -------
    xarray DataArray
        p-value
    """
    diff = xr.apply_ufunc(_angle_diff, rad1, rad2,
        dask='parallelized', output_dtypes=[float])
    diff.name = 'angle_diff'
    diff.attrs.update(unit='', description='smalles angle difference')
    return diff

# utils
def _angle_diff(rad1, rad2):
    """Returns the differences between two angles [radian]"""
    msg = "circular doy should be in [-pi, pi] range"
    assert (np.abs(rad1) <= np.pi).all() and (np.abs(rad2) <= np.pi).all(), msg
    # input circdata in range [-pi, pi]
    diff = rad2 - rad1
    abs_diff = np.abs(diff)
    # extract the smallest angle between two angles range [-pi, pi]
    diff = np.where(abs_diff>=np.pi, 2*np.pi-abs_diff, diff)
    return diff

# numpy functions from https://github.com/astropy/astropy/blob/v3.0.x/astropy/stats/circstats.py
# Copyright (c) 2011-2017, Astropy Developers
# copied to avoid astropy dependecy
# edits 
# -use nansum by default instead of sum
# -default axis is set to -1
# -added axis and newaxis where necessary to deal with ndarrays

def _components(data, p=1, phi=0.0, weights=None, axis=-1):
    """ Generalized rectangular components."""
    if weights is None:
        weights = np.ones((1,))
    try:
        weights = np.broadcast_to(weights, data.shape)
    except ValueError:
        raise ValueError('Weights and data have inconsistent shape.')

    # nansum instead of sum
    C = np.nansum(weights * np.cos(p * (data - phi)), axis)/np.nansum(weights, axis)
    S = np.nansum(weights * np.sin(p * (data - phi)), axis)/np.nansum(weights, axis)

    return C, S


def _angle(data, p=1, phi=0.0, weights=None, axis=-1):
    """ Generalized sample mean angle."""
    C, S = _components(data, p, phi, weights, axis)

    # theta will be an angle in the interval [-np.pi, np.pi)
    theta = np.arctan2(S, C)

    return theta


def _length(data, p=1, phi=0.0, weights=None, axis=-1):
    """ Generalized sample length."""
    C, S = _components(data, p, phi, weights, axis)
    return np.hypot(S, C)


def _circmean(data, weights=None, axis=-1):
    """ Circular mean."""
    return _angle(data, 1, 0.0, weights, axis)


def _circvar(data, weights=None, axis=-1):
    """ Circular variance."""
    return 1.0 - _length(data, 1, 0.0, weights, axis)


def _circcorrcoef(alpha, beta, weights_alpha=None, weights_beta=None, axis=-1):
    """ Circular correlation coefficient.
    edited to deal with dimensions"""
    if(np.size(alpha, axis) != np.size(beta, axis)):
        raise ValueError("alpha and beta must be arrays of the same size")

    mu_a = _circmean(alpha, weights_alpha, axis)
    mu_b = _circmean(beta, weights_beta, axis)

    # added newaxis to deal with multi dimensions
    sin_a = np.sin(alpha - mu_a[..., None]) 
    sin_b = np.sin(beta - mu_b[..., None])

    # changed sum into nansum and added axis to deal with dimensions
    rho = np.nansum(sin_a*sin_b, axis=axis)/np.sqrt(np.nansum(sin_a**2, axis=axis)*np.nansum(sin_b**2, axis=axis))

    return rho


def _rayleightest(data, weights=None, axis=-1):
    """Rayleigh test of uniformity."""
    n = np.sum(np.isfinite(data), axis=axis) # changed in to count of finite values 
    Rbar = _length(data, 1, 0.0, weights, axis)
    z = n*Rbar*Rbar

    # see original astropy script for references
    # adapted to to work for ndim array
    tmp = np.where(
        n < 50,
        1. + (2.*z - z**2)/(4.*n) - (24.*z - 132.*z**2 + 76.*z**3 - 9.*z**4)/(288. * n**2),
        1.
    )

    p_value = np.exp(-z)*tmp
    return p_value