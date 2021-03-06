#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Dirk Eilander (contact: dirk.eilander@vu.nl)
# Created: Nov 2nd 2018

import xarray as xr
import numpy as np
import scipy

__all__ = ['get_peaks', 'peaks_over_threshold', 'annual_max']

def get_peaks(ts, min_dist=1, dim='time', chunks={}):
    """Returns a DataArray with peak values, all other values are set to NaN.
    
    Peaks are defined as a high data point surrounded by lower data points. 
    If multiple high data points are surrounded by lower data points (a flat peak) 
    only the first high data point is selected. The minimum distance between peaks 
    can be set using the min_dist argument.

    Parameters
    ----------
    ts : xarray DataArray
        time series
    min_dist : int, optional
        minimum distance between peaks [ts step] (the default is 1)
    dim : str, optional
        name of time dimension in ts (the default is 'time')
    
    Returns
    -------
    xarray DataArray
        timeseries of peaks
    """
    # get sign of trend
    trend = xr.ufuncs.sign(ts.diff(dim=dim))
    # set flats to negative trend to deal with flat peaks
    trend = xr.where(trend==0, -1, trend)
    # peaks where previous and next points are lower
    peaks = ts.where(trend.diff(dim=dim, label='lower')==-2).chunk(chunks)
    if min_dist > 1:
        # import pdb; pdb.set_trace()
        # max_in_wdw = peaks.rolling(center=True, min_periods=1, **{dim: min_dist*2}).max()
        # see git issue https://github.com/pydata/xarray/issues/3165
        max_in_wdw = peaks.rolling(center=True, min_periods=1, **{dim: min_dist*2}).construct('window').max('window')
        peaks = peaks.where(max_in_wdw==peaks)
    return peaks

def peaks_over_threshold(ts, threshold, min_dist=1, dim='time', chunks={}):
    """Returns a DataArray with Peaks Over Threshold (POT), all other values are 
    set to NaN.
    
    Peaks are defined as a high data point surrounded by lower data points. 
    If multiple high data points are surrounded by lower data points (a flat peak) 
    only the first high data point is selected. The minimum distance between peaks 
    can be set using the min_dist argument.
 
    Parameters
    ----------
    ts : xarray DataArray
        time series
    threshold : float
        threshold value
    min_dist : int, optional
        minimum distance between peaks [ts step] (the default is 0)
    dim : str, optional
        name of time dimension in ts (the default is 'time')
    
    Returns
    -------
    xarray DataArray
        timeseries of peaks over threshold
    """
    peaks = get_peaks(ts, dim=dim, min_dist=min_dist, chunks=chunks)
    # peaks over threshold (POT)
    peaks = peaks.where(peaks > threshold)
    return peaks

def annual_max(da, min_dist=1, dim='time', reduce=False):
    """Returns a DataArray with Annual Maxima (AM) peaks

    Peaks are defined as a high data point surrounded by lower data points. 
    If multiple high data points are surrounded by lower data points (a flat peak) 
    only the first high data point is selected. The minimum distance between peaks 
    can be set using the min_dist argument.

    Parameters
    ----------
    ts : xarray DataArray
        time series
    threshold : float
        threshold value
    min_dist : int, optional
        minimum distance between peaks [ts step] (the default is 0)
    dim : str, optional
        name of time dimension in ts (the default is 'time')
    reduce : bool, optional
        if True, reduce the AM series to a year timestep; if False, keep full timeseries 
        with all values but the AM set to NaN (the default is False)

    Returns
    -------
    xarray DataArray
        timeseries of annual maxima peaks
    """
    peaks = get_peaks(da, min_dist=min_dist, dim=dim)
    grp = '{}.year'.format(dim)
    peaks_grp = peaks.groupby(grp)
    if reduce == False:
        peaks_am = peaks.where(peaks_grp.max(dim=dim) == peaks_grp)
    else:
        peaks_am = peaks_grp.max(dim=dim)
    return peaks_am

def nanpercentile(da, q, dim='time', interpolation='linear'):
    """Returns the qth percentile of the data along the specified core dimension,
    while ignoring nan values.
    
    Parameters
    ----------
    da: xarray DataArray
        Input data array
    q : float in range of [0,100] (or sequence of floats)
        Percentile to compute, which must be between 0 and 100 inclusive.
    dim : str, optional
        name of the core dimension (the default is 'time')
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired quantile lies between two data points
    
    Returns
    -------
    percentile : xarray DataArray
        The core dimension is reduce to quantiles and returned at the last dimension.
    """
    def _nanpercentile(*args, **kwargs):
        """nanpercentile, but with q moved to the last axis"""
        return np.moveaxis(np.nanpercentile(*args, **kwargs), 0, -1)
    # nanpercentile parameters
    q = np.atleast_1d(q)
    q_kwargs = dict(q=q, axis=-1, interpolation=interpolation)
    # apply_ufunc parameters
    kwargs = dict(               
        input_core_dims=[[dim]], 
        output_core_dims=[['percentile']],
        dask='parallelized', 
        output_dtypes=[float],    
        output_sizes={'percentile': q.size} # on output, <dim> is reduced to length q.size 
    )
    if 'percentile' in da.coords:
        da = da.drop('percentile')
    percentile = xr.apply_ufunc(_nanpercentile, da.chunk({dim: -1}), kwargs=q_kwargs, **kwargs)
    percentile['percentile'] = xr.Variable('percentile', q)
    return percentile.squeeze() # if q.size=1 remove dim


