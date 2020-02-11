
# coding: utf-8

from datetime import datetime, timedelta
import xarray as xr
import dask
from os.path import basename, isfile, join, dirname
import pandas as pd
import os
import warnings
from multiprocessing import cpu_count, Pool, freeze_support
import numpy as np
import calendar
warnings.filterwarnings("ignore")

nprocs = 4

cwd = dirname(os.path.realpath(__file__))
stations_fn = join(cwd, 'cmf_gtsm_75km.csv')
ddir = r'/home/glofris1/VIDI/experiments/CaMaFlood_e2o/GTSM'
# ddir = r'/media/data/GTSM'
fn_tide = join(ddir, r'all_fes_data.nc') # all data combined
fn_surge = join(ddir, r'global_model_surgelevel_{}.nc') # data per year
# station data
df = pd.read_csv(stations_fn, index_col=0)
gtsm_idx = df['gtsm_idx'].values.astype(int)
chunks = {'time':2*24*50, 'stations':1000}

def select(ds, gtsm_idx, time_index, index):
    ds['gtsm_idx'] = xr.Variable(('stations'), ds.stations.values)
    ds = ds.sel(stations=gtsm_idx.ravel()).sel(time=time_index, method='nearest') #.load()
    ds['time'] = xr.Variable(('time'), time_index)
    ds['index'] = xr.Variable(('stations'), index)
    return ds

def waterlevel_climatology_forcing(yr):
    fn_out = join(cwd, f"global_model_waterlevel_clim_{yr}_select_egm.nc")
    if isfile(fn_out): return
    fn_tide = join(cwd, f"all_fes_data_{yr}_select_egm.nc")
    fn_surge_clim = join(cwd, 'global_model_surgelevel_select_clim.nc')
    print(fn_out)
    ds_tide = xr.open_dataset(fn_tide, chunks=chunks)
    ds_clim = xr.open_dataset(fn_surge_clim)
    assert np.all(ds_clim['gtsm_idx'].values == ds_tide['gtsm_idx'].values)
    assert np.all(ds_clim['gtsm_idx'].values == gtsm_idx)

    # interpolate monthly values to 30min
    time_index = pd.date_range(start=datetime(yr-1, 12, 1), end=datetime(yr+1, 1, 31), freq='MS') + timedelta(days=14)
    ds_clim_month = xr.DataArray(
        data=dask.array.zeros(shape=(time_index.size, ds_tide.stations.size), dtype=np.float32, chunks=(1000, 14)), 
        coords=[time_index, ds_tide.stations.values], 
        dims=['time', 'stations'], 
        name='surge'
    ).to_dataset()
    ds_clim_month['surge'].data = ds_clim_month['surge'].groupby('time.month') + ds_clim['surge'].astype(np.float32)
    ds_surge_clim = ds_clim_month.interp(time=ds_tide.time, method='linear')
    ds_surge_clim.attrs = {}

    # superimpose surge with tides
    ds_out = xr.merge([
        xr.DataArray(ds_surge_clim['surge'] + ds_tide['tide'], dims=('time', 'stations'), name='waterlevel'),
        ds_tide.drop(('tide', 'station_available', 'station_x_coordinate', 'station_y_coordinate'))
    ]).chunk(chunks)
    encoding = {v: {'zlib': True} for v in ['waterlevel']}
    ds_out.to_netcdf(fn_out, encoding=encoding)
    
def waterlevel_forcing(yr):
    fn_out = join(cwd, f"global_model_waterlevel_{yr}_select_egm.nc")
    if isfile(fn_out): return
    fn_tide = join(cwd, f"all_fes_data_{yr}_select_egm.nc")
    fn_surge = join(cwd, f"global_model_surgelevel_{yr}_select.nc")
    print(fn_out)
    ds_tide = xr.open_dataset(fn_tide, chunks=chunks)
    ds_surge = xr.open_dataset(fn_surge, chunks=chunks)
    # merge with tides
    assert np.all(ds_surge.time==ds_tide.time)
    assert np.all(ds_surge['gtsm_idx'].values == ds_tide['gtsm_idx'].values)
    assert np.all(ds_surge['gtsm_idx'].values == gtsm_idx)
    ds_out = xr.merge([
        xr.DataArray(ds_surge['surge'] + ds_tide['tide'], dims=('time', 'stations'), name='waterlevel'),
        ds_tide.drop(('tide', 'station_available', 'station_x_coordinate', 'station_y_coordinate'))
    ]).chunk(chunks)
    encoding = {v: {'zlib': True} for v in ['waterlevel']}
    ds_out.to_netcdf(fn_out, encoding=encoding) 

def select_tide_year(year):
    # select resample & write
    fn_tide_out = join(cwd, 'all_fes_data_{:04d}_select_egm.nc'.format(year))
    if isfile(fn_tide_out): return
    print(fn_tide_out)
    # read data
    chunks_in = chunks.copy()
    chunks_in.update(time=chunks['time']*10)
    dst = xr.open_dataset(fn_tide, chunks=chunks_in)
    global_attrs = {'source_files': basename(fn_tide)}
    global_attrs.update(dst.attrs)
    dst.attrs = global_attrs
    time_index = pd.date_range(start=datetime(year, 1, 1), end=datetime(year, 12, 31, 23, 59), freq='30min')

    # select tid eand add egm offset
    dst_sel = select(dst, gtsm_idx, time_index, df.index).load().chunk(chunks)
    assert np.all(df.loc[dst_sel.stations, 'gtsm_idx'].values == dst_sel['gtsm_idx'].values)
    dst_sel['egm_offset'] = xr.Variable(('stations'), df.loc[dst_sel.stations, 'gtsm_egm_offset'].values)
    dst_sel['tide'] = dst_sel['tide'] + dst_sel['egm_offset']
    dst_sel.to_netcdf(fn_tide_out, encoding={'tide':  {'zlib': True}})

def select_surge_year(year):
    # select resample & write
    fn_surge_out = join(cwd, 'global_model_surgelevel_{:04d}_select.nc'.format(year))
    if isfile(fn_surge_out): return
    print(fn_surge_out)
    # read data
    chunks_in = chunks.copy()
    chunks_in.update(time=chunks['time']*10)
    fn = fn_surge.format(year)
    dss = xr.open_dataset(fn, chunks=chunks_in)
    global_attrs = {'source_files': ', '.join([basename(fn), basename(fn_tide)])}
    global_attrs.update(dss.attrs)
    dss.attrs = global_attrs
    time_index = pd.date_range(start=datetime(year, 1, 1), end=datetime(year, 12, 31, 23, 59), freq='30min')
    dss_sel = select(dss, gtsm_idx, time_index, df.index).load().chunk(chunks)
    assert np.all(df.loc[dss_sel.stations, 'gtsm_idx'].values == dss_sel['gtsm_idx'].values)
    dss_sel.rename({'waterlevel': 'surge'}).to_netcdf(fn_surge_out, encoding={'surge':  {'zlib': True}})

def surge_climatology():
    dim = 'time'
    encoding = {v: {'zlib': True} for v in ['surge']}
    fns = f"global_model_surgelevel_*_select.nc"
    fn_out = join(cwd, 'global_model_surgelevel_select_clim.nc')
    if isfile(fn_out): return
    ds = xr.open_mfdataset(join(cwd, fns))['surge'].chunk(chunks)
    ds_meta = xr.open_dataset(fn_tide)
    # montly climatology
    grp = '{}.month'.format(dim)
    # ignore period with outliers
    ds = ds.where(ds.time!=slice(datetime(1990,10,1), datetime(1991,3,1)))
    ds_clim = ds.groupby(grp).mean(dim)
    ds_clim.name = 'surge'

    # broadcast to middle of month
    ds_clim = xr.merge([
        ds_clim,  
        ds_meta.drop(('tide', 'station_available', 'station_x_coordinate', 'station_y_coordinate', 'time'))
    ]).chunk({'month':-1, 'stations':1000})
    ds_clim.to_netcdf(fn_out, encoding=encoding)

def resample_day(yr):
    # read selected 30 min data
    fn_tide = join(cwd, f'all_fes_data_{yr}_select_egm.nc')
    fn_wl = join(cwd, f'global_model_waterlevel_{yr}_select_egm.nc')
    fn_wl_clim = join(cwd, f'global_model_waterlevel_clim_{yr}_select_egm.nc')
    fn_surge = join(cwd, f'global_model_surgelevel_{yr}_select.nc')
    
    fn_out = join(cwd, f'gtsm_{yr}_select_day.nc')
    # if isfile(fn_out): return
    global_attrs = {
        'created_by': 'Dirk Eilander (dirk.eilander@vu.nl)',
        'source': "Global Tide and Surge Model (Muis et al., 2016)",
        'source_files': ', '.join([basename(fn_tide), basename(fn_surge), basename(fn_wl), basename(fn_wl_clim)]),
        'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    print(fn_out)
    # read data
    # ds_surge = xr.open_dataset(fn_surge)
    ds_tide = xr.open_dataset(fn_tide)
    da_tide = ds_tide['tide']
    da_wl = xr.open_dataset(fn_wl)['waterlevel']
    da_wl_clim = xr.open_dataset(fn_wl_clim)['waterlevel']
    da_surge = da_wl - da_wl_clim
    da_seas = da_wl_clim - da_tide
    # resample
    da_wl_day = da_wl.resample(time='D')    
    da_tide_day = da_tide.resample(time='D')
    da_seas_day = da_seas.resample(time='D')
    da_surge_day = da_surge.resample(time='D')
    # instant
    time_index = pd.date_range(start=datetime(yr, 1, 1), end=datetime(yr, 12, 31, 0, 0), freq='1D')
    da_tide_inst = da_tide.sel(time=time_index, method='nearest')
    da_seas_inst = da_seas.sel(time=time_index, method='nearest')
    da_surge_inst = da_surge.sel(time=time_index, method='nearest')

    # save combined waterlevel
    dim = ('time', 'stations')
    ds_out = xr.merge([
        #Htide
        xr.DataArray(da_tide_inst, dims=dim, name='Htide'),
        xr.DataArray(da_tide_day.max('time'), dims=dim, name='Htide_day_max'),
        xr.DataArray(da_tide_day.min('time'), dims=dim, name='Htide_day_min'),
        #Hseas
        xr.DataArray(da_seas_inst, dims=dim, name='Hseas'),
        xr.DataArray(da_seas_day.mean('time'), dims=dim, name='Hseas_day_mean'),
        #Hsurge
        xr.DataArray(da_surge_inst, dims=dim, name='Hsurge'),
        xr.DataArray(da_surge_day.max('time'), dims=dim, name='Hsurge_day_max'),
        xr.DataArray(da_surge_day.mean('time'), dims=dim, name='Hsurge_day_mean'),
        #Htot
        xr.DataArray(da_wl_day.max('time'), dims=dim, name='Htot_day_max'),
        xr.DataArray(da_wl_day.mean('time'), dims=dim, name='Htot_day_mean'),
        #meta
        ds_tide[['station_id', 'station_name', 'stations']]
        ]).chunk({'time':-1})
    ds_out['Hskewsurge_day'] = ds_out['Htot_day_max'] - ds_out['Htide_day_max'] 
    ds_out = ds_out.set_coords(['station_id', 'station_name'])
    ds_out.attrs.update(global_attrs)
    encoding={v:  {'zlib': True} for v in ds_out.data_vars}
    ds_out.to_netcdf(fn_out, encoding=encoding)
    return

def do_years(f, yrs=range(1980, 2015), num_workers=nprocs):
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
    # # 1) select stations and instant 30 min data + egm_offset for tid
    # # resuls in surgelevel and tide+egm nc files
    # do_years(f=select_surge_year, num_workers=4)
    # do_years(f=select_tide_year, num_workers=4)
    # # 1b) craete forcing files by combined tide (m+EGM96) + surge
    # do_years(f=waterlevel_forcing)
    
    # # 2) resample surge to monhtly climatology
    # # results in
    # # * monthly climatology (monthly res) 
    # # * interploated to 30min for a leap and 
    # # * idem for noleap year
    # surge_climatology()
    # # 2b) craete forcing files by combined tide (m+EGM96) + surge
    # do_years(f=waterlevel_climatology_forcing)

    # 4) resample to daily values
    do_years(f=resample_day)

