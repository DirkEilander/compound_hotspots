import xarray as xr
from os.path import join, basename, isfile, isdir
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

# load one local library with additional statistics for the xarray datastructure
from peaks import peaks_over_threshold, get_peaks
import xstats as xs 

# args
q = 95
min_dist = 14
Npeaks = 50
nyears=35
window_sizes = [0, 1, 3] # drivers
alphas = np.array([0.05, 0.1, 0.25, 0.75, 0.9, 0.95])
rps_out = np.array([1,2,5,10,20,30,35])
chunks = {'ensemble':-1, 'scen':-1, 'time': -1, 'index':100}


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
ds = xr.open_zarr(fn_in)

# peaks and drivers
# -----------------
drivers = ['Htot_day_max', 'Hskewsurge_day', 'Htide_day_max', 'Qmsl']
fn_peaks = join(ddir, f'rivmth_peaks_q{q}d{min_dist}_ts.zarr')
if not isdir(fn_peaks):
    print(basename(fn_peaks))
    wl_ts = ds['WSE']
    wl_thresh = xs.xnanpercentile(wl_ts, q, dim='time')
    wl_peaks = peaks_over_threshold(wl_ts.fillna(-np.inf), wl_thresh, min_dist=min_dist, dim='time').reindex_like(wl_ts)
    wl_peaks.name = 'WSE_peaks'
    wl_rp = xs.xinterp_ev(wl_peaks, wl_ts, nyears=nyears)
    wl_rp.name = 'WSE_rp'
    # drivers
    # drivers_ts = ds[drivers]
    # drivers_peaks = get_peaks(drivers_ts.fillna(-np.inf), min_dist=min_dist, dim='time').reset_coords(drop=True).reindex_like(drivers_ts)
    # drivers_rp = xs.xinterp_ev(drivers_peaks, drivers_ts, nyears=nyears)
    # drivers_rp = drivers_rp.rename({n: f'{n}_rp' for n in list(drivers_rp.data_vars.keys())})
    # write to nc file
    ds_out = xr.merge([
        wl_peaks,
        wl_rp,
        # drivers_rp
    ])
    ds_out.attrs.update(glob_attrs)
    ds_out.chunk(chunks).to_zarr(fn_peaks)


# extreme value analysis (emperical)
# ----------------------------------
ds_peaks = xr.open_zarr(fn_peaks)
wl_peaks = ds_peaks['WSE_peaks']
fn_peaks_rp = join(ddir, f'rivmth_peaks_q{q}d{min_dist}_rp.nc')
if not isfile(fn_peaks_rp):
    print(basename(fn_peaks_rp))
    ds_rp = xs.xinterp_rps(wl_peaks, rp=rps_out, nyears=nyears)
    ds_rp.to_netcdf(fn_peaks_rp)
fn_peaks_rp_ci = join(ddir, f'rivmth_peaks_q{q}d{min_dist}_rp_ci_N1e5.nc')
if not isfile(fn_peaks_rp_ci):
    print(basename(fn_peaks_rp_ci))
    ds_rp_ci = xs.xinterp_rps_ci(wl_peaks, rp=rps_out, nyears=nyears, n_samples=10000, alphas=alphas)
    ds_rp_ci.to_netcdf(fn_peaks_rp_ci)
ds_peaks.close()
ds.close()


# def idx_topn(da, n):
#     def _idx_topn(x, n=50):
#         return np.argsort(x)[::-1][:n]
    
#     return xr.apply_ufunc(
#         _idx_topn, 
#         da, 
#         kwargs=dict(n=n),
#         input_core_dims=[['time']], 
#         output_core_dims=[['rank']], 
#         vectorize=True, 
#         dask='allowed', 
#         output_dtypes=[int],
#         output_sizes={'rank':n}
#     )
# # largest n peaks accross scenarios
# # --------------------------------
# # OUT
# fn_peaks_ranked = join(ddir, f'rivmth_peaks_q{q}d{min_dist}_top{Npeaks}_all.nc')
# if not isfile(fn_peaks_ranked):
#     print(basename(fn_peaks_ranked))
#     # IN
#     wl_peaks_ts = xr.open_zarr(fn_peaks)['WSE'].fillna(-np.inf)
#     peaks = xr.ufuncs.isfinite(wl_peaks_ts)
#     scenarios = wl_peaks_ts['scen'].data
#     # find largest peaks within +/- wdw_size days between all scen
#     wl_peaks_wdwmax = wl_peaks_ts.rolling(time=min_dist*2+1, min_periods=1, center=True).construct('window').max('window').astype(np.float32)
#     peaks_max = xr.concat([xr.where(
#             peaks.sel(scen=scen), 
#             wl_peaks_ts.sel(scen=scen) > wl_peaks_wdwmax.sel(scen=[s for s in scenarios if s != scen]).max('scen'), 
#             False
#         ) for scen in scenarios
#     ], dim='scen').chunk({'time':-1})
#     peaks_max.name = 'peaks_max'
#     # select largest Npeaks
#     itime = idx_topn(wl_peaks_ts.where(peaks_max, -np.inf).max('scen'), n=Npeaks).load()
#     wl_peaks_wdwmax_na = wl_peaks_wdwmax.where(xr.ufuncs.isfinite(wl_peaks_wdwmax))
#     wl_peaks_wdwmax_na.name = 'WSE'
#     ds_peaks = xr.merge([wl_peaks_wdwmax_na, peaks_max]).load()
#     # reduce to topn peaks
#     ds_peaks_rank = ds_peaks.isel(time=itime).transpose('scen', 'ensemble', 'rank', 'index')
#     ds_peaks_rank['rank'] = xr.Variable('rank', np.arange(Npeaks).astype(int)+1)
#     # write to nc file
#     ds_peaks_rank.to_netcdf(fn_peaks_ranked)
#     wl_peaks_ts.close()

# # largest n peaks indenpendently per scenario 
# # -------------------------------------------
# fn_peaks_ranked = join(ddir, f'rivmth_peaks_q{q}d{min_dist}_top{Npeaks}_scen.nc')
# if not isfile(fn_peaks_ranked):
#     print(basename(fn_peaks_ranked))
#     wl_peaks_ts = xr.open_zarr(fn_peaks)
#     itime = idx_topn(wl_peaks_ts['WSE'].fillna(-np.inf), n=Npeaks).load()
#     ds_peaks_rank = wl_peaks_ts[['WSE']].isel(time=itime).transpose('scen', 'ensemble', 'rank', 'index')
#     ds_peaks_rank['rank'] = xr.Variable('rank', np.arange(Npeaks).astype(int)+1)
#     ds_peaks_rank['rp'] = xr.Variable('rank', float((Npeaks/(Npeaks / 35.))-1) / ds_peaks_rank['rank'])
#     ds_peaks_rank.to_netcdf(fn_peaks_ranked)
#     wl_peaks_ts.close()

