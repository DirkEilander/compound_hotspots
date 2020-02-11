#!/usr/bin/env python
# coding: utf-8

"""validates discharge from CMF with observations from GRDC"""

import xarray as xr
from os.path import join
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import calendar

# local 
import temp_stats as tstats
import skill_stats as sstats


# functions to preprocess the GRDC observation record 
# and make sure the simulation data is identical in terms of missing values and axis

def drop_incomplete_years(ts, threshold_complete, dim='time'):
    """Returns a DataArrays with NaN values in years with insufficient datapoints.
    """
    ndays = lambda year: 365+calendar.isleap(year) # 
    start_year = lambda x: x[dim].to_index().year[0]
    grp = '{}.year'.format(dim)
    drop_incomplete_years = lambda x: x.where((x.count(dim=dim)/ndays(start_year(x)))>=threshold_complete)
    return ts.groupby(grp).apply(drop_incomplete_years)

def preprocess_ts(sim, obs, threshold_complete=0, dim='time'):
    """Returns DataArrays sliced to overlapping time and with NaN 
    values in years with insufficient datapoints or where the observation 
    dataarrays has missing or NaN values. 
    
    Note that the function assumes a complete simulation DataArray, 
    i.e. no missing/NaN values. 
    """
    # drop incomplete years
    if threshold_complete > 0:
        # assume the sim dataarray is always complete
        obs = drop_incomplete_years(obs, threshold_complete, dim=dim)
    # drop values in sim where NaNs in obs
    idx = xr.ufuncs.isfinite(obs) # finite values in observations
    obs = obs.where(idx)
    sim = sim.where(idx)
    # find overlapping time
    latest_start = max(sim.time.to_index()[0], obs.time.to_index()[0])
    earliest_end = min(sim.time.to_index()[-1], obs.time.to_index()[-1])
    sim = sim.sel(time=slice(latest_start, earliest_end)).chunk({dim: -1})
    obs = obs.sel(time=slice(latest_start, earliest_end)).chunk({dim: -1})
    return sim, obs


def run_skill_metrics(sim, obs, dim='time'):
    """run following performance metrics on daily values
    kge
    nse
    log_nse
    rsquared
    time lag
    """
    kge = sstats.kge(sim, obs, dim=dim)
    nse = sstats.nashsutcliffe(sim, obs, dim=dim)
    log_nse = sstats.lognashsutcliffe(sim, obs, dim=dim, epsilon=1e-6)
    rsquared = sstats.rsquared(sim, obs, dim=dim)
    # overall timing
    tl = tstats.time_lag_crosscorr(sim, obs, dim=dim)
    # merge performance metrics
    return xr.merge([kge, nse, log_nse, rsquared, tl])

def run_skill_metrics_annual_max(sim, obs, min_dist=1, dim='time'):
    """get performance metrics for annual maxima values
    doy diff
    doy correlation
    am rank correlation
    am bias
    """
    # get annual max magnitude and timing
    sim_am_stats = tstats.mean_flood_day_stats(sim, min_dist=min_dist, dim='time').chunk({'year':-1})
    obs_am_stats = tstats.mean_flood_day_stats(obs, min_dist=min_dist, dim='time').chunk({'year':-1})
    sim_am, obs_am = sim_am_stats['annual_max'], obs_am_stats['annual_max']
    sim_doy, obs_doy = sim_am_stats['doy'], obs_am_stats['doy']
    # timing
    doy_diff = tstats.mean_flood_day_diff(sim_doy, obs_doy, dim='year')
    doy_corr = tstats.corr_flood_day(sim_doy, obs_doy, dim='year')
    doy_diff.name = 'am_doy_diff'
    doy_corr.name = 'am_doy_corr'
    # magnitude
    am_rank_corr = sstats.spearman_rank_correlation(sim_am, obs_am, dim='year')
    am_rank_corr.name = 'am_rank_corr'
    am_rank_corr.attrs.update(discription = 'spearman rank correlation coefficient for annual maxima')
    # same bias definition as in kge
    am_bias = sim_am.sum(dim='year') / obs_am.sum(dim='year')
    am_bias.name = 'am_bias'
    am_bias.attrs.update(discription = 'bias in annual maxima')
    # merge and keep doy uniformity test p and variance of observed annual max
    obs_am_stats = obs_am_stats.rename({n: f'obs_{n}' for n in obs_am_stats.data_vars.keys()})
    return xr.merge([am_bias, am_rank_corr, doy_diff, doy_corr, obs_am_stats[['doy_uniform_p', 'doy_var']]])

if __name__ == "__main__":
    root = r'/scratch/compound_hotspots'
    ddir = join(root, 'data', '3-model_output')
    grdc_dir = r'/scratch/grdc'
    versions = ['v362'] #, 'v392']
    studies = ['validation'] #, 'sensitivity_hc'] ['compound'] #
    # NOTE: already selected with these values when creating zarr
    min_area = 1e3 #[km2]
    min_yrs = 8 #[%]
    threshold_complete=0.75
    # unfortunatately this file cannot be shared but has to be retreived from GRDC
    fn_grdc = join(grdc_dir, r'grdc_discharge_1980-2014_v20180912.zarr')

    # load metadata and calculate percentage of data available
    # ds_obs = xr.open_zarr(fn_grdc)['discharge'].drop(['lat', 'lon']).persist()
    # df_meta = pd.read_csv(join(r'/home/dirk/datasets/GRDC', '20180912_GRDC_GSIM_NatHuman_metadata.csv'), index_col=0)
    # print(len(df_meta))
    # # calculate how perc data per station
    # ddata = (xr.ufuncs.isfinite(ds_obs).sum(dim='time') / ds_obs.time.size * 100).to_series()
    # df_meta = df_meta.loc[ddata.index, :]
    # df_meta['ddata'] = ddata
    # df_meta.to_csv(fn_grdc.replace('.zarr', '.csv'))

    # load observations from selected stations (df_meta_index)
    obs = xr.open_zarr(fn_grdc)['discharge']
    obs = obs.transpose('grdc_id', 'time').chunk({'time':-1, 'grdc_id':20}).persist()
    obs = drop_incomplete_years(obs, threshold_complete=threshold_complete, dim='time')
    nyrs = (xr.ufuncs.isfinite(obs).groupby('time.year').sum(dim='time')>0).sum('year').load()

    # load meta data
    df_meta = pd.read_csv(fn_grdc.replace('.zarr', '.csv'), index_col=0)
    df_meta['nyrs'] = nyrs.reindex(grdc_id=df_meta.index).values
    # select basedd on minimal up area and minimal no. of years 
    df_meta = df_meta[np.logical_and(df_meta['area']> min_area, df_meta['nyrs'] >= min_yrs)]
    sel = dict(grdc_id=df_meta.index)
    obs = obs.sel(sel)
    len(df_meta)

    # load simulations form CMF
    # use the runoff scenario for validation
    fn_format = join(ddir, r'grdc_{m}_mswep_runoff_v362_1980-2014.nc')
    model = ['anu', 'cnrs', 'ecmwf', 'nerc', 'jrc'] #, 'univu', 'univk']
    t0, t1 = datetime(1980,1,1), datetime(2014,12,31) # disregard last day. has some weird values
    chunks = {'time': -1, 'grdc_id':100}
    # combine cmf outputs
    ds_m = []
    for m in model:
        ds_t = xr.open_dataset(fn_format.format(m=m), chunks=chunks)[['outflw']].sortby('time')
        ds_m.append(ds_t)
    ds_cmf = xr.concat(ds_m, dim='ensemble').transpose('ensemble', 'grdc_id', 'time')
    ds_cmf['ensemble'] = xr.Variable('ensemble', model)
    ds_cmf['time'] = ds_cmf.time.to_index() + timedelta(days=-1)
    ds_cmf = ds_cmf.sel(grdc_id=obs.grdc_id.values).drop(['lat_nc', 'lon_nc'])
    ds_cmf


    ## compute performance statistics
    sim = ds_cmf['outflw']
    # make sure model is string data
    sim['ensemble'] = xr.Variable(data=sim['ensemble'].values.astype('str'), dims=['ensemble']) 
    # select the same 
    sim = sim.sel(**sel).chunk({'time':-1, 'grdc_id':100}).persist()
    # preprocess
    sim, obs = preprocess_ts(sim, obs)
    # daily skill stats
    pm = run_skill_metrics(sim, obs, dim='time').transpose('ensemble', 'grdc_id')
    fn_pm = join(ddir, '../4-postprocessed', r'cmf_v362_e2o_validation_grdc_pm.nc')
    pm.to_netcdf(fn_pm)
    # annual max statistics
    pm_am= run_skill_metrics_annual_max(sim, obs, min_dist=30, dim='time').transpose('ensemble', 'grdc_id')
    fn_pm_am = join(ddir, '../4-postprocessed', r'cmf_v362_e2o_validation_grdc_pm_am.nc')
    pm_am.to_netcdf(fn_pm_am)

