"""calculate extreme value statistics (GUMBEL) for riverine water levels in different scenarios"""

import xarray as xr
from os.path import join, basename, isfile, isdir
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from lmoments3 import distr
import warnings
warnings.simplefilter("ignore")
from dask.diagnostics import ProgressBar

# load one local library with additional statistics for the xarray datastructure
from peaks import get_peaks
from xlm_fit import xlm_fit, xlm_fit_ci 

# args
min_dist = 14
scenarios = ['surge', 'seas', 'tide']
alpha = 0.1
rps_out = np.array([1.1, 1.5, 2, 5, 10, 20, 30, 50, 100])
chunks = {'ensemble':-1, 'scen':-1, 'time': -1, 'index':100}
#I/O
root = r'/scratch/compound_hotspots'
ddir = join(root, 'data', '4-postprocessed')
# IN
fn_in = join(ddir, 'rivmth_reanalysis.zarr')
fn_out = join(ddir, 'rivmth_ev.nc')

# read
# ----
da_wse = xr.open_zarr(fn_in)['WSE'].sel(scen=scenarios)

# AM peaks & extreme value analysis (fit gumbel)
print(basename(fn_out))
peaks_am  = get_peaks(da_wse, min_dist=min_dist, dim='time').fillna(-np.inf).groupby('time.year').max('time')
peaks_am = peaks_am.rename({'year': 'time'}).chunk(chunks)
ds_rp = xlm_fit(
    peaks_am, 
    fdist=distr.gum, 
    rp=rps_out
    )
ds_out = xr.merge([
    peaks_am.to_dataset().rename({'WSE': 'WSE_am'}),
    ds_rp.rename({'WSE': 'WSE_ev'}),
])
ds_out.chunk(chunks).to_netcdf(fn_out)

# confidence intervals using bootstrap
ds = xr.open_dataset(fn_out, chunks=chunks)
fn_out = fn_out.replace('.nc',f'_ci_p{alpha/2*100:02.0f}.nc')
print(basename(fn_out))
with ProgressBar():
    ds_rp_ci = xlm_fit_ci(
        ds['WSE_am'].chunk(chunks), 
        fdist=distr.gum,
        rp=rps_out, 
        n_samples=1000, 
        alphas=[alpha/2, 1-alpha/2]
        ).load()
ds_out = xr.merge([
    ds,
    ds_rp_ci.to_dataset().rename({'WSE_am': 'WSE_ev_ci'}),
])
ds_out = ds_out.rename({'WSE_am': 'annual_maxima', 'WSE_ev': 'extreme_values', 'WSE_ev_ci': 'extreme_values_ci'})

# to file
ds_out['annual_maxima'].attrs.update(
    description='simulated annual maxima of water surface elevation at the river mouth',
    unit='m+EGM96',
    long_name='water_surface_elevation'
)
ds_out['extreme_values'].attrs.update(
    description='extreme values of water surface elevation at the river mouth based on Gubmel distribion of annual maxima',
    unit='m+EGM96',
    long_name='water_surface_elevation'
)
ds_out['extreme_values_ci'].attrs.update(
    description='confidence intervals around extreme values of water surface elevation',
    unit='m+EGM96',
    long_name='water_surface_elevation'
)
ds_out['params'].attrs.update(
    description='parameters of Gumbel extreme value distibution',
    unit='-',
)
ds_out.attrs.update(
    institution = 'Institute for Environmental Studies (IVM) - Vrije Universiteit Amsterdam',
    author = 'Dirk Eilander (dirk.eilander@vu.nl)',
    date_created = str(datetime.now().date()),
    history = f'created using {basename(__file__)}; xarray v{xr.__version__}',
)

ds_out.to_netcdf(fn_out)