"""get AM riverine water levels events with max H and Q drivers within <wdw>"""

import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gp
from os.path import join, basename
from datetime import date, datetime
import os

# local
from peaks import get_peaks

root = r'/scratch/compound_hotspots'
ddir = join(root, 'data', '4-postprocessed')
wdw=1

# I/O
fn_in = join(ddir, 'rivmth_reanalysis.zarr')
fn_out = join(ddir, f'rivmth_drivers.nc')


scenarios = ['surge']
vars_ = ['h', 'Hskewsurge', 'Q']
rm = {'Hskewsurge_day':'Hskewsurge'}

# read data
ds = xr.open_zarr(fn_in).sel(scen=scenarios).rename(rm)
ds['h'] = ds['WSE']-ds['z0']
ds = ds[vars_].sel(scen='surge').drop('scen')

# window max driver
ds_wdwmax = ds.rolling(time=wdw*2+1, min_periods=1, center=True).construct('window').max('window').astype(np.float32)

# get AM h peaks
peaks  = get_peaks(ds['h'], min_dist=30, dim='time').reset_coords(drop=True).reindex_like(ds).compute()
hpeaks = peaks.where(np.isfinite(peaks), -np.inf)
hpeaks_yr = hpeaks.groupby('time.year')
hpeaks_am = hpeaks_yr == hpeaks_yr.max('time')
hpeaks_doy = hpeaks['time'].dt.dayofyear

# combine h AM peaks, drivers and return periods and keep only an. max.
ds_am_h = xr.merge([
    hpeaks_doy,
    ds_wdwmax,
]).where(hpeaks_am, -np.inf).groupby('time.year').max('time')
ds_am_h = ds_am_h.transpose('year', 'ensemble', 'index').compute()

# write to file
ds_am_h.attrs.update(
    institution = 'Institute for Environmental Studies (IVM) - Vrije Universiteit Amsterdam',
    author = 'Dirk Eilander (dirk.eilander@vu.nl)',
    date_created = str(datetime.now().date()),
    history = f'created using {basename(__file__)}; xarray v{xr.__version__}',
)
ds_am_h.to_netcdf(fn_out)