import xarray as xr
from os.path import join, basename, isfile, isdir
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from lmoments3 import distr

# load one local library with additional statistics for the xarray datastructure
from peaks import peaks_over_threshold, get_peaks
import xstats as xs 
from bootstrap import confidence_interval

# args
min_dist = 14
nyears = 35
window_sizes = [0, 1, 2]
vars_ = ['WSE', 'Htide', 'Hskewsurge', 'Hsurge', 'Htot', 'Q']
scenarios = ['surge', 'seas', 'tide', 'msl']
alpha = 0.05
rps_out = np.array([1.1, 1.5, 2, 5, 10, 20, 30, 50, 100])
chunks = {'ensemble':-1, 'scen':-1, 'time': -1, 'index':100}
force_overwrite = False

#I/O
root = r'/scratch/compound_hotspots'
ddir = join(root, 'data', '4-postprocessed')
# IN
fn_in = join(ddir, 'rivmth_reanalysis.zarr')

# attrs
glob_attrs = dict(
    institution = 'Institute for Environmental Studies (IVM) - Vrije Universiteit Amsterdam',
    author = 'Dirk Eilander (dirk.eilander@vu.nl)',
    date_created = str(datetime.now().date()),
    history = f'created using {basename(__file__)}; xarray v{xr.__version__}',
)

# read
# ----
rm = {'Hskewsurge_day':'Hskewsurge', 'Htot_day_max': 'Htot', 'Htide_day_max': 'Htide', 'Hsurge_day_max': 'Hsurge'}
ds = xr.open_zarr(fn_in).sel(scen=scenarios).drop(['Hsurge', 'Htide']).rename(rm)

# peaks and drivers
# -----------------
fn_peaks = join(ddir, f'rivmth_peaks_d{min_dist}.zarr')
if not isdir(fn_peaks) or force_overwrite:
    print(basename(fn_peaks))
    ds_rivmth = ds[vars_]
    ds_peaks  = get_peaks(ds_rivmth, min_dist=min_dist, dim='time').reset_coords(drop=True).reindex_like(ds_rivmth)
    ds_peaks_rp = xs.xinterp_ev(ds_peaks, ds_rivmth, nyears=nyears) 
    ds_peaks = xr.ufuncs.isfinite(ds_peaks).rename({v:f'{v}_peaks' for v in list(ds_peaks.data_vars.keys())})
    ds_peaks_rp = ds_peaks_rp.rename({v:f'{v}_rp' for v in list(ds_peaks_rp.data_vars.keys())})
    ds_out = xr.merge([ds_rivmth, ds_peaks, ds_peaks_rp]).transpose('scen', 'ensemble', 'time', 'index')
    ds_out.attrs.update(glob_attrs)
    ds_out.chunk(chunks).to_zarr(fn_peaks)
peaks = xr.open_zarr(fn_peaks)

# # select annual max peaks per driver, for compound  scenario
# # ----------------------------------
fn_peaks = join(ddir, f'rivmth_peaks_d{min_dist}.zarr')
peaks = xr.open_zarr(fn_peaks)

drivers = ['WSE', 'Htide', 'Hskewsurge', 'Q']
peaks_ = peaks.sel(scen='surge').drop('scen').fillna(-np.inf)
for wdw in window_sizes[1:]:
    fn_peaks_wdw = join(ddir, f'rivmth_AMpeaks_wdw{wdw}.nc')
    if not isfile(fn_peaks_wdw) or force_overwrite:
        print(basename(fn_peaks_wdw))
        if wdw > 0:
            peaks_wdwmax = peaks_.rolling(time=wdw+1, min_periods=1, center=False).construct('window').max('window').astype(np.float32)
        else:
            peaks_wdwmax = peaks_
        dss = []
        for driver in drivers:
            da_peaks = peaks_[driver].where(peaks_[f'{driver}_peaks'], -np.inf)
            da_group_yr = da_peaks.groupby('time.year')
            da_peaks_am = da_group_yr == da_group_yr.max('time')
            itime = xs.xtopn_idx(da_peaks.where(da_peaks_am, -np.inf), n=nyears).load()
            # itime = da_am_itime.sel(driver=driver)
            ds_topn = peaks_wdwmax.isel(time=itime).transpose('ensemble', 'rank', 'index')
            dss.append(ds_topn)
        ds_out = xr.concat(dss, dim='driver').reset_coords('time')
        ds_out['driver'] = xr.Variable('driver', drivers)
        ds_out['rank'] = xr.Variable('rank', np.arange(nyears).astype(int)+1)
        ds_out.to_netcdf(fn_peaks_wdw)
    
    # stats
    fn_peaks_wdw_stats = join(ddir, f'rivmth_AMpeaks_wdw{wdw}_stats.nc')
    if not isfile(fn_peaks_wdw_stats) or force_overwrite:
        print(basename(fn_peaks_wdw_stats))
        ds = xr.open_dataset(fn_peaks_wdw).sel(driver=drivers)
        ds['h'] = ds['WSE']-ds['z0']
        wl_am_ci = confidence_interval(ds['h'], dim='rank', n_samples=1000, alpha=alpha)
        lowci = wl_am_ci.sel(driver='WSE', ci_bounds=alpha/2).reset_coords(drop=True)
        highci = wl_am_ci.sel(driver=drivers[1:], ci_bounds=1-alpha/2).reset_coords(drop=True)
        wl_diff_sign = lowci > highci
        wl_am = ds['h'].mean('rank').reset_coords(drop=True)
        wl_am_act = wl_am.sel(driver='WSE').drop('driver')
        wl_am_act.name = 'h_actual'
        wl_am_drivers = wl_am.sel(driver=drivers[1:])
        main_driver = wl_am_drivers['driver'].isel(driver=wl_am_drivers.argmax('driver'))
        wl_main_driver = wl_am_drivers.sel(driver=main_driver)
        wl_main_driver.name = 'h_driver'
        wl_diff_sign = wl_diff_sign.sel(driver=main_driver)
        wl_diff_sign.name = 'sign'
        wl_diff_sign.attrs.update(alpha=alpha)
        ds_ratio  = (ds['h'].sel(driver=drivers).mean('rank') / ds['h'].sel(driver='WSE').mean('rank')) * 100
        ds_ratio = xr.merge([
            ds_ratio.sel(driver=d).drop('driver').to_dataset().rename({'h': f'h_ratio_{d}'}) for d in drivers
        ])
        ds_out = xr.merge([
            wl_main_driver,
            wl_am_act,
            wl_diff_sign,
            ds_ratio
        ]).reset_coords()
        ds_out['h_ratio'] = (ds_out['h_driver'] / ds_out['h_actual']) * 100
        ds_out.to_netcdf(fn_peaks_wdw_stats)

# extreme value analysis (fit gumbel)
# -----------------------------------
wl_peaks_am = peaks['WSE'].fillna(-np.inf).resample(time='A').max('time')
fn_peaks_rp = join(ddir, f'rivmth_peaks_gumb.nc')
if not isfile(fn_peaks_rp) or force_overwrite:
    print(basename(fn_peaks_rp))
    ds_rp = xs.xlm_fit(wl_peaks_am, fdist=distr.gum, nyears=None, rp=rps_out)
    ds_out = xr.merge([
        ds_rp.rename({'WSE': 'WSE_ev'}),
        wl_peaks_am.to_dataset().rename({'WSE': 'WSE_am'}),
    ])
    ds_out.to_netcdf(fn_peaks_rp)
fn_peaks_rp_ci = join(ddir, f'rivmth_peaks_gumb_ci_N1e4.nc')
ds_rp = xr.open_dataset(fn_peaks_rp)
if not isfile(fn_peaks_rp_ci) or force_overwrite:
    print(basename(fn_peaks_rp_ci))
    ds_rp_ci = xs.xlm_fit_ci(wl_peaks_am, fdist=distr.gum, nyears=None, rp=rps_out, n_samples=1000, alphas=[alpha/2, 1-alpha/2])
    ds_out = xr.merge([
        ds_rp_ci.to_dataset().rename({'WSE': 'WSE_ev_ci'}),
        ds_rp,
    ])
    ds_out.to_netcdf(fn_peaks_rp_ci)



ds.close()