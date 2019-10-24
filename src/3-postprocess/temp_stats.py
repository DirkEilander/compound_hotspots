#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Dirk Eilander (contact: dirk.eilander@vu.nl) and Anais Couasnon (contact anais.couasnon@vu.nl)
# Created: Nov 2nd 2018

import xarray as xr
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
import calendar

__all__ = [
    'time_lag_crosscorr', 'doy_to_circ', 'circ_to_doy',
    'flood_day', 'corr_flood_day', 'mean_flood_day', 'mean_flood_day_diff', 'mean_flood_day_stats'
    ]
    
from circ_stats import (circ_mean, circ_corr, circ_var, rayleightest, angle_diff)
from skill_stats import pearson_correlation 
from peaks import get_peaks

# crosscorr 
def time_lag_crosscorr(sim, obs, quantile=None, lags=np.arange(-10,11,1), 
                        t_unit='days', dim='time'):
    """Returns the time lag between two time series based on a lag time 
    with the maximum pearson correlation.
    
    Parameters
    ----------
    sim : xarray DataArray
        simulations time series
    obs : xarray DataArray
        observations time series
    quantile : numpy ndarray, optional
        quantile based threshold (the default is None, which does not use any threshold)
    lags : numpy ndarray, optional
        range of considered lag times (the default is np.arange(-10,11,1))
    t_unit : str, optional
        time unit used to parse lags to timedelta format (the default is 'days')
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')
    
    Returns
    -------
    xarray DataSet
        lag time and associated correlation coefficient
    """

    if quantile:
        obs.load()        
        obs = obs.where(obs>=obs.quantile(quantile, dim=dim))
    # loop through time lags and calculate cross correlation
    r = []
    lags = np.asarray(lags)
    time_org = sim[dim].to_index()
    for dt in lags:
        time_new = time_org + timedelta(**{t_unit: float(dt)})
        ts = slice(max(time_org.min(), time_new.min()), min(time_org.max(), time_new.max()))
        sim[dim] = time_new
        r.append(pearson_correlation(sim.sel(**{dim:ts}), obs.sel(**{dim:ts})))
    sim[dim] = time_org # reset time
    pearsonr = xr.concat(r, dim='dt')
    pearsonr['dt'] = xr.Variable('dt', lags)
    # get maximum cross corr
    pearsonr_max = pearsonr.max(dim='dt')
    pearsonr_max.name = 'lag_rho'
    pearsonr_max.attrs.update(description='maximum pearson coefficient for given time lag')
    # get lag time of maximum cross corr
    # NOTE that we assume a evenly spaced lag times
    lag = xr.where(
        xr.ufuncs.isfinite(pearsonr).sum(dim='dt')==lags.size,
        pearsonr.argmax(dim='dt', skipna=False), 
        np.nan)*np.diff(lags)[0] + lags.min()
    lag.name = 'lag'
    lag.attrs.update(description='time lag with maximum pearson coefficient', unit=t_unit)
    # merge max cross corr and lag tiem
    return xr.merge([lag, pearsonr_max])

# day of the year stats
def corr_flood_day(sim_doy, obs_doy, dim='year'):
    """Returns the circular correlation coefficient of the flood days,
    i.e. the day of the year (DOY) of annual maxima (AM) in two time series.  
    
    Parameters
    ----------
    sim_doy : xarray DataArray
        annual simulation time series of flood day (day of the year of annual maximum)
    obs_doy : xarray DataArray
        annual observations time series  of flood day (day of the year of annual maximum)
    dim : str, optional
        name of time dimension in sim and obs (the default is 'year')
    
    Returns
    -------
    xarray DataArray
        circular correlation coefficient [-]
    """
    sim_doy_circ = doy_to_circ(sim_doy, dim=dim) # angle [rad] representation of doy 
    obs_doy_circ = doy_to_circ(obs_doy, dim=dim)
    corrcoef = circ_corr(sim_doy_circ, obs_doy_circ, dim=dim)
    corrcoef.name = 'doy_corr'
    corrcoef.attrs.update(
        unit='-', 
        description='circular correlation coeficient of flood days (i.e.annual maximum doy)'
    )
    return corrcoef

def mean_flood_day_diff(sim_doy, obs_doy, dim='year'):
    """Returns the difference in mean flood day, i.e. the difference in mean DOY of 
    annual maxima between two time series.  
    
    Parameters
    ----------
    sim_doy : xarray DataArray
        annual simulation time series of flood day (day of the year of annual maximum)
    obs : xarray DataArray
        annual observations time series  of flood day (day of the year of annual maximum)
    dim : str, optional
        name of time dimension in sim and obs (the default is 'year')
    
    Returns
    -------
    tuple of xarray DataArray
        difference in mean DOY of annual max [days]
    """
    # mean annual maximum doy
    sim_doy_circ = doy_to_circ(sim_doy, dim=dim) # [-pi,pi] 
    sim_theta = circ_mean(sim_doy_circ, dim=dim) 
    obs_doy_obs = doy_to_circ(obs_doy, dim=dim) # [-pi,pi] 
    obs_theta = circ_mean(obs_doy_obs, dim=dim) 
    # angle between mean max doy. positive means simulation after observation
    theta_diff = angle_diff(obs_theta, sim_theta) # [-pi, pi]
    diff_days = theta_diff*365/(2*np.pi) # abs diff anlge expressed in days
    diff_days.name = 'doy_diff'
    diff_days.attrs.update(unit='day', description='lag in mean annual maximum day of the year')
    return diff_days

def mean_flood_day(ts_doy, dim='year'):
    """Returns the mean flood day, i.e. the day of the year (DOY) of annual maxima (AM) in ts.
    
    Parameters
    ----------
    ts_doy : xarray DataArray
        annual time series of flood day (day of the year of annual maximum)
    dim : str, optional
        name of time dimension in ts (the default is 'year')
    
    Returns
    -------
    xarray DataArray
        mean annual flood day [julian calander day], 
    """
    circ_data = doy_to_circ(ts_doy, dim=dim) # [-pi,pi] 
    theta = circ_mean(circ_data, dim=dim)  # [-pi,pi] 
    mean_flood_day = circ_to_doy(theta, dim=None) # [1,365]
    mean_flood_day.name = 'doy_mean'
    mean_flood_day.attrs.update( 
        description='mean annual maximum day of the year', 
        unit='julian calendar day')
    return mean_flood_day

def mean_flood_day_stats(ts, min_dist=1, dim='time'):
    """Returns the mean flood day, i.e. the day of the year (DOY) of annual maxima (AM) in ts.
    
    Parameters
    ----------
    ts : xarray DataArray
        time series
    dim : str, optional
        name of time dimension in ts (the default is 'time')
    
    Returns
    -------
    xarray DataSet
        mean annual flood day [julian calander day], 
        variance of annual flood day [-], 
        rayleigh test p-value [-]
    """
    # get annual max doys [rad]
    am = flood_day(ts, min_dist=min_dist, dim=dim)
    # get mean flood day
    am_doy_mean = mean_flood_day(am['doy'], dim='year')
    # variance = 1 - geralized rectangular length
    circ_data = doy_to_circ(am['doy'], dim='year') # [-pi,pi] 
    var_flood_day = circ_var(circ_data, dim='year')  
    var_flood_day.name = 'doy_var'
    var_flood_day.attrs.update( description='variance of annual maximum day of the year', unit='-')
    # uniformity test
    p_value = rayleightest(circ_data, dim='year') # test non-uniformity of circ_data
    p_value.name = 'doy_uniform_p'
    return xr.merge([am, am_doy_mean, var_flood_day, p_value])

def flood_day(ts, min_dist=1, dim='time'):
    """Returns a yearly time series with the flood day and magnitude.
    i.e. the day of the year (DOY) of the annual maxima (AM) in ts.
    
    Parameters
    ----------
    ts : xarray DataArray
        time series with daily time step
    dim : str, optional
        name of time dimension in ts (the default is 'time')
    
    Returns
    -------
    xarray DataSet
        AM 
        DOY of annual maxima per year [julian calander day]
        
    """
    # make sure first (last) year start (end) at start (end) of the year and daily time step
    # t = ts[dim].to_index()
    # tstart =  t[0] if (t[0].month == 1) and (t[0].day == 1) else date(t[0].year+1, 1, 1)
    # tend =  t[-1] if (t[-1].month == 12) and (t[-1].day == 31) else date(t[-1].year-1, 12, 31)
    # ts = ts.sel(**{dim: pd.date_range(tstart, tend, )}, method='nearest')
    grp = '{}.year'.format(dim)
    # get peaks only with optional min dist argument
    peaks = get_peaks(ts, min_dist=min_dist, dim=dim)
    # get annual maxima
    am = peaks.groupby(grp).max(dim=dim).chunk({'year': -1})
    am.name = 'annual_max'
    am.attrs.update(description='annual maximum', unit=ts.attrs.get('unit', 'unknown'))
    # now argmax for doy annual maxima
    # NOTE: argmax on dask chuncks returns zeros with xr version 0.10.8 and dask version 0.18.2
    # doy = ts.groupby(grp).reduce(np.argmax, dim=dim)
    # fillna to avoid "all-NaNs error"
    doy = peaks.fillna(-np.inf).groupby(grp).argmax(dim=dim)
    # deal with year all-NaNs years and start counting from 1 (jan 1st) instead of python zero indexing
    doy = xr.where(xr.ufuncs.isfinite(am), doy+1, np.nan).chunk({'year': -1})
    doy.name = 'doy'
    doy.attrs.update(description='annual maximum day of the year', unit='julian calendar day')
    return xr.merge([am, doy])


def doy_to_circ(doy, dim='year'):
    """Translate the day of the year (DOY) to circular angle [radian] with range [-pi, pi]"""
    # idx = np.isfinite(doy)
    # assert (doy[idx]>=1).all() and (doy[idx]<=366).all(), "doy should be in [1, 366] range"
    ndays = _get_ndays_year(doy, dim=dim)
    circ_doy = doy*2*np.pi / ndays # range [0, 2pi]
    circ_doy = xr.where(circ_doy>np.pi, -2*np.pi+circ_doy, circ_doy) # range [-pi, pi]
    return circ_doy

def circ_to_doy(circ_doy, dim='year'):
    """Translate a circular angle [radian] with range [-pi, pi] to day of the year (DOY) with range [0, 366]"""
    # idx = np.isfinite(circ_doy)
    # assert (np.abs(circ_doy[idx]) <= np.pi).all(), "circular doy should be in [-pi, pi] range"
    ndays = _get_ndays_year(circ_doy, dim=dim)
    circ_doy = xr.where(circ_doy<0, circ_doy+2*np.pi, circ_doy) #Convert to interval [0, 2pi]
    doy = circ_doy * ndays / (2*np.pi) #[0, ndays-1]
    doy = xr.where(doy<1, ndays, doy) # zeros should be ndays [1, ndays]
    return doy

# helper DOY functions
def _doy_to_month(doy, dim='year'):
    """translate day of the year to month of the year taking into account leap years"""
    def _moy(doy, dim='year'):
        year = doy[dim].values
        f = lambda x: (datetime(year, 1, 1) + timedelta(x - 1)).month
        return np.vectorize(f)(doy)
    # Month Of the Year
    moy = doy.groupby(dim).apply(_moy, dim=dim)
    return moy

def _get_ndays_year(doy, dim='year'):
    """get number of days in a year"""
    if dim:
        ndays = np.array([365+calendar.isleap(yr) for yr in doy[dim].values])
        ndays = xr.DataArray(ndays, dims=[dim], coords=[doy[dim].values])
    else:
        ndays = xr.apply_ufunc(lambda x: np.maximum(365, x), doy, 
            dask='parallelized', output_dtypes=[float])
    return ndays

