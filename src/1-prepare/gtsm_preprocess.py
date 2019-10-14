
# coding: utf-8

from datetime import datetime, timedelta
import xarray as xr
from os.path import basename, isfile, join, dirname
from dask.diagnostics import ProgressBar
import dask
import dask.threaded
import pandas as pd
import os
import warnings
from multiprocessing import cpu_count, Pool, freeze_support
import numpy as np
warnings.filterwarnings("ignore")

nprocs = 6

cwd = dirname(os.path.realpath(__file__))
stations_fn = join(cwd, 'cmf_gtsm_75km.csv')
ddir = r'/home/glofris1/VIDI/experiments/CaMaFlood_e2o/GTSM'
fn_tide = join(ddir, r'all_fes_data.nc') # all data combined
fn_surge = join(ddir, r'global_model_surgelevel_{}.nc') # data per year


def select(ds, gtsm_idx, time_index, index):
    ds['gtsm_idx'] = xr.Variable(('stations'), ds.stations.values)
    ds = ds.sel(stations=np.unique(gtsm_idx)).sel(time=time_index, method='nearest').load()
    ds['time'] = xr.Variable(('time'), time_index)
    # create new axis to allow for duplite stations
    istat = np.concatenate([np.where(ds.gtsm_idx.values==i)[0] for i in gtsm_idx])
    ds = ds.isel(stations=istat)
    ds['index'] = xr.Variable(('stations'), index)
    return ds

def add_offset(ds, egm_offset, var='tide'):
    ds['egm_offset'] = xr.Variable(('stations'), egm_offset)
    ds[var] = ds[var] + ds['egm_offset']
    return ds

def combine_tide_surge(ds_surge, ds_tide):
    assert np.all(ds_surge.gtsm_idx==ds_tide.gtsm_idx)
    ds_surge['waterlevel'] = xr.Variable(('time', 'stations'), (ds_surge.waterlevel + ds_tide.tide)) # combine surge and tide
    return ds_surge


def select_offset(year):
    # station data
    df = pd.read_csv(stations_fn, index_col=0)
    gtsm_idx = df[['gtsm_idx']].values.astype(int)
    # select resample & write
    fn_wl_out = join(cwd, 'global_model_waterlevel_{:04d}_select.nc'.format(year))
    fn_tide_out = join(cwd, 'all_fes_data_{:04d}_select.nc'.format(year))
    print(fn_wl_out)
    print(fn_tide_out)
    # read data
    chunks={'time':2*24*7*26, 'stations': 1000}
    fn = fn_surge.format(year)
    dss = xr.open_dataset(fn, chunks=chunks)
    global_attrs = {'source_files': ', '.join([basename(fn), basename(fn_tide)])}
    global_attrs.update(dss.attrs)
    dss.attrs = global_attrs
    dst = xr.open_dataset(fn_tide, chunks=chunks)
    global_attrs = {'source_files': basename(fn_tide)}
    global_attrs.update(dst.attrs)
    dst.attrs = global_attrs
    time_index = pd.date_range(start=datetime(year, 1, 1), end=datetime(year+1, 1, 1), freq='30min')
    
    # do the work
    with dask.config.set(get=dask.threaded.get, num_workers=cpu_count()/nprocs*2): #, ProgressBar():
        # select tide & surge # slow
        dss_sel = select(dss, gtsm_idx, time_index, df.index)
        dst_sel = select(dst, gtsm_idx, time_index, df.index)

        # combine tide & surge 
        dswl_sel = combine_tide_surge(dss_sel, dst_sel)
        dswl_sel.to_netcdf(basename(fn_wl_out), encoding={'waterlevel':  {'zlib': True}})
        dst_sel.to_netcdf(basename(fn_tide_out), encoding={'tide':  {'zlib': True}})
        
        # add egm offset
        dst_egm = add_offset(dst_sel, df.loc[dst_sel.stations, 'gtsm_egm_offset'], 'tide')
        dst_egm.to_netcdf(basename(fn_tide_out).replace('.nc','_egm.nc'), encoding={'tide':  {'zlib': True}})
        dswl_egm = add_offset(dswl_sel, df.loc[dswl_sel.stations, 'gtsm_egm_offset'], 'waterlevel')
        dswl_egm.to_netcdf(basename(fn_wl_out).replace('.nc','_egm.nc'), encoding={'waterlevel':  {'zlib': True}})

def surge_seasonal(yr):
    with dask.config.set(num_workers=int(cpu_count()/2), scheduler='threads'):
        print(yr)
        # read selected 30 min data
        fn_out = join(ddir, 'global_model_waterlevel_seasonal_{:04d}_select_egm.nc').format(yr)
        if isfile(fn_out): return
        fn_wl = join(ddir, 'global_model_waterlevel_{:d}_select_egm.nc').format
        fn_tide = join(ddir, 'all_fes_data_{:d}_select_egm.nc').format
        chunks={'stations': 30, 'time':-1}
        # read data
        yr0, yr1 = 1980, 2014
        yrs = np.unique(np.clip(np.array([yr-1, yr, yr+1]), yr0, yr1))
        fns_tide = [fn_tide(y) for y in yrs]
        fns_wl = [fn_wl(y) for y in yrs]
        t0 = max([datetime(yr-1, 11, 15), datetime(yr0, 1, 1)])
        t1 = min([datetime(yr+1, 2, 15), datetime(yr1, 12, 31, 23, 59)])
        ds_tide = xr.open_mfdataset(fns_tide).sortby('time').sel(time=slice(t0, t1)).chunk(chunks)
        ds_wl = xr.open_mfdataset(fns_wl).sortby('time').sel(time=slice(t0, t1)).chunk(chunks)
        da_surge = ds_wl['waterlevel'] - ds_tide['tide']
        # running mean over 90 days
        da_surge_day = da_surge.resample(time='D').mean('time').chunk(chunks)
        da_surge_seas = da_surge_day.rolling(time=90, center=True, min_periods=1).mean()
        tslice = slice(datetime(yr,1,1), datetime(yr,12,31,23,59,59))
        da_surge_seas = da_surge_seas.sel(time=tslice)
        da_surge_seas['dayofyear'] = da_surge_seas.time.dt.dayofyear
        da_surge_seas = da_surge_seas.swap_dims({'time': 'dayofyear'})
        da_tide = ds_tide['tide'].sel(time=tslice)
        # create new waterlevel variable in ds_wl
        da_out = xr.DataArray(
            data = da_tide.groupby('time.dayofyear') + da_surge_seas, 
            coords = [da_tide.time, da_tide.stations], 
            dims = ['time', 'stations'],
            name='waterlevel'
        )
        da_out.attrs.update({
            'source_files': ', '.join(fns_tide + fns_wl),
            'description': '90 days moving average surge'
        })
        # print(da_out)
        # save yearly nc files
        encoding={'waterlevel':  {'zlib': True}}
        da_out.to_netcdf(fn_out, encoding=encoding)


def resample_day(year):
    # read selected 30 min data
    fn_wl = join(ddir, 'global_model_waterlevel_{:04d}_select.nc'.format(year))
    fn_tide = join(ddir, 'all_fes_data_{:04d}_select.nc'.format(year))
    fn_out = join(ddir, 'global_model_waterlevel_{:04d}_select_day.nc'.format(year))
    chunks={'stations': 1000, 'time':-1}
    global_attrs = {
        'source_files': ', '.join([basename(fn_wl), basename(fn_tide)]),
        'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    print(fn_out)
    if os.path.isfile(fn_out): return
    # read data
    ds_tide = xr.open_dataset(basename(fn_tide), chunks=chunks)
    ds_wl = xr.open_dataset(basename(fn_wl), chunks=chunks)
    da_surge = ds_wl['waterlevel'] - ds_tide['tide']
    # combine
    ds_wl_day = ds_wl['waterlevel'].resample(time='D').max('time')
    ds_wl_day.name = 'high_wl'
    ds_wl_day_min = ds_wl['waterlevel'].resample(time='D').min('time')
    ds_wl_day_min.name = 'low_wl'
    ds_tide_day = ds_tide['tide'].resample(time='D').max('time')
    ds_tide_day.name = 'high_tide'
    ds_tide_day_min = ds_tide['tide'].resample(time='D').min('time')
    ds_tide_day_min.name = 'low_tide'
    ds_surge_day = da_surge.resample(time='D').max('time')
    ds_surge_day.name = 'max_surge'
    ds_surge_day_mean = da_surge.resample(time='D').mean('time')
    ds_surge_day_mean.name = 'mean_surge'
    ds_skew_surge_day = ds_wl_day - ds_tide_day
    ds_skew_surge_day.name = 'skew_surge'
    # save combined waterlevel
    ds_out = xr.merge([
        ds_wl_day, ds_wl_day_min, ds_tide_day, ds_tide_day_min, ds_surge_day, ds_surge_day_mean, ds_skew_surge_day, 
        ds_tide[['station_id', 'station_name', 'stations']]]
        ).chunk({'time':-1})
    ds_out = ds_out.set_coords(['station_id', 'station_name'])
    ds_out.attrs.update(global_attrs)
    encoding={v:  {'zlib': True} for v in ds_out.data_vars}
    ds_out.to_netcdf(basename(fn_out), encoding=encoding)
    ds_wl.close()
    ds_tide.close()
    return

def do_years(yrs=range(1980, 2015), num_workers=nprocs, f=select_offset):
    print('using {:d} workers'.format(num_workers))
    if num_workers > 1:
        p = Pool(num_workers)
        p.map(f, yrs)
        p.close()
    else:
        # single process
        for year in yrs:
            f(year)

if __name__ == "__main__":
    freeze_support()
    # # select stations and intant 30 min data + egm_offset
    # do_years(f=select_offset)
    
    # # resample to day
    # do_years(f=resample_day)

    # seasonal bounds
    do_years(f=surge_seasonal, num_workers=4)
