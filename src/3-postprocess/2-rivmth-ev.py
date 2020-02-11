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
scenarios = ['surge', 'seas', 'tide', 'msl']
alpha = 0.05
rps_out = np.array([1.1, 1.5, 2, 5, 10, 20, 30, 50, 100])
chunks = {'ensemble':-1, 'scen':-1, 'time': -1, 'index':100}
#I/O
root = r'/scratch/compound_hotspots'
ddir = join(root, 'data', '4-postprocessed')
# IN
fn_in = join(ddir, 'rivmth_reanalysis.zarr')
fn_out = join(ddir, 'rivmth_swe_am.nc')
# attrs
glob_attrs = dict(
    institution = 'Institute for Environmental Studies (IVM) - Vrije Universiteit Amsterdam',
    author = 'Dirk Eilander (dirk.eilander@vu.nl)',
    date_created = str(datetime.now().date()),
    history = f'created using {basename(__file__)}; xarray v{xr.__version__}',
)

# read
# ----
da_wse = xr.open_zarr(fn_in)['WSE'].sel(scen=scenarios)

# AM peaks & extreme value analysis (fit gumbel)
if not isfile(fn_out):
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
fn_out = fn_out.replace('.nc','_ci.nc')
if not isfile(fn_out):
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
    ds_out.to_netcdf(fn_out)