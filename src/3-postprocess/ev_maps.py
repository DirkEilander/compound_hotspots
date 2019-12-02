from os.path import join, basename
import os 
import xarray as xr
import numpy as np
import pandas as pd
import glob
from dask.diagnostics import ProgressBar
import subprocess
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from peaks import get_peaks
from xlm_fit import xlm_fit 
from lmoments3 import distr

import warnings
warnings.simplefilter("ignore")

rps_out = np.array([1.1, 1.5, 2, 5, 10, 20, 30, 50, 100])
def ev_map(ddir, fn_out, var, mask=None, min_dist=14, rp=rps_out, out_kwargs={}):
    chunks2 = {'latlon':500, 'time':-1}
    fns = glob.glob(join(ddir, f'{var}*.nc'))
    # combine, stack and mask data
    da = xr.open_mfdataset(fns, combine='by_coords')[var]
    if mask is None:
        mask = xr.ufuncs.isfinite(da.isel(time=0)).load()
    da.coords['mask'] = mask
    da_stacked = da.stack(latlon=('lat','lon'))
    da_stacked = da_stacked.where(da_stacked['mask'], drop=True).chunk(chunks2)
    # get AM
    peaks_am_stacked = get_peaks(da_stacked, min_dist=min_dist, dim='time', chunks=chunks2).groupby('time.year').max('time')
    peaks_am_stacked = peaks_am_stacked.rename({'year': 'time'})
    peaks_am_stacked.name = f'{var}_am'
    # fit gumbel
    with ProgressBar():
        ds_rp_stacked = xlm_fit(peaks_am_stacked, fdist=distr.gum, rp=rps_out, nmin=30)
        ds_rp_stacked = ds_rp_stacked.rename({f'{var}_am': f'{var}_ev'})
        # merge and write
        xr.merge([
            peaks_am_stacked, 
            ds_rp_stacked
        ]).unstack().reindex_like(mask).to_netcdf(fn_out, **out_kwargs)

def set_bin_inputs(sdir, fn_swe, rps=np.array([2,10,50])):
    mv=1e+20
    # fn_elevtn = join(sdir, 'map', 'elevtn.tif')
    # elevtn = xr.open_rasterio(fn_elevtn).drop('band').squeeze().rename({'x':'lon', 'y':'lat'})

    # read nc and calc flddph
    swe = xr.open_dataset(fn_swe)['sfcelv_ev']
    # flddph = swe - elevtn 
    # TODO bias correction with flddph for T=1.5 ??
    flddph = swe - swe.sel(T=1.5)
    flddph = xr.where(flddph<0,0,flddph).fillna(mv)

    # write to file per rp
    for T in rps:
        print(f'rp: {T:03.1f}')
        data = flddph.sel(T=T).data.astype('f4')
        fn_out_bin = join(sdir, f'flddph_T{T:03.0f}')
        if os.path.isfile(fn_out_bin): os.unlink(fn_out_bin)
        data.tofile(fn_out_bin)
        # import pdb; pdb.set_trace()


def downscale_flddph(sdir, ddir, rps=np.array([2,10,50])):
    mv=1e+20
    nodata=-9999.
    bbox = -180., -90., 180., 90.
    res = 0.005

    profile = {
        'driver': 'GTiff', 
        'dtype': 'float32', 
        'nodata': nodata, 
        'width': ((bbox[2]-bbox[0])/res), 
        'height': ((bbox[3]-bbox[1])/res), 
        'count': 1, 
        'crs': CRS.from_epsg(4326), 
        'transform': from_origin(bbox[0], bbox[3], res, res),
        'blockxsize': int(2/res), 
        'blockysize': int(2/res), 
        'tiled': True, 
        'compress': 'lzw', 
        'interleave': 'band'
    }

    out_dir = join(ddir, 'flddph')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fn_regions = join(sdir, 'map', 'hires', 'location.txt')
    regions = pd.read_csv(fn_regions, delim_whitespace=True, index_col=0).T \
                .set_index('area').astype(float).to_dict(orient='index')
    # ds_lst = []
    for T in rps:
        fn_out_tif = join(out_dir, f'flddph_T{T:03.0f}.tif')
        fn_flddph_bin = join(sdir, f'flddph_T{T:03.0f}')
        # if os.path.isfile(fn_out_tif): continue
        print(f'rp: {T:03.1f}')        
        with rasterio.open(fn_out_tif, 'w', **profile) as dst:
            pass # write empty file
        # downscale
        for area in regions.keys():
            print(area)
            msg = ['./downscale_flddph', str(area), basename(fn_flddph_bin), '1']
            subprocess.call(msg, cwd=sdir, stderr=subprocess.STDOUT)
            
            fn_fld = join(sdir, '{:s}.flood'.format(area))
            if os.path.isfile(fn_fld):
                # read binary output
                ny, nx = int(regions[area]['ny']), int(regions[area]['nx'])
                with open(fn_fld, 'r') as fid:
                    data1 = np.fromfile(fid, 'f4').reshape(ny, nx)
                    data1 = np.where(data1==mv, nodata, data1)
                # read wdw from tif
                west, north = regions[area]['west'], regions[area]['north']
                wdw_bounds = west, np.round(north-ny*res), np.round(west+nx*res), north #NOTE all bbox are rounded to full degrees
                # append to geotiff
                with rasterio.open(fn_out_tif, 'r+') as dst:
                    if wdw_bounds[2] > 180:
                        nnx = int(np.round((wdw_bounds[2]-180)/res))
                        wdw_bounds2 = -180, np.round(north-ny*res), -180+wdw_bounds[2]-180, north
                        window=rasterio.windows.from_bounds(*wdw_bounds2, transform=dst.transform)
                        data0 = dst.read(1, window=window)
                        data2 = np.where(data1[:,-nnx:] == nodata, data0, data1[:,-nnx:])
                        dst.write(data2, window=window, indexes=1)
                        wdw_bounds = west, np.round(north-ny*res), 180, north
                        data1 = data1[:,:-nnx]
                    window=rasterio.windows.from_bounds(*wdw_bounds, transform=dst.transform)
                    data0 = dst.read(1, window=window)
                    data1 = np.where(data1 == nodata, data0, data1)
                    dst.write(data1, window=window, indexes=1)
                # remove binary output
                os.unlink(fn_fld)
            else:
                print(' '.join(msg))


def flood_impact(sdir, ddir, rps=np.array([2,10,50]), exp_name='worldpop'):
    mv=1e+20
    nodata=-9999.
    bbox = -180., -90., 180., 90.
    res = 0.005

    profile = {
        'driver': 'GTiff', 
        'dtype': 'float32', 
        'nodata': nodata, 
        'width': ((bbox[2]-bbox[0])/res), 
        'height': ((bbox[3]-bbox[1])/res), 
        'count': 1, 
        'crs': CRS.from_epsg(4326), 
        'transform': from_origin(bbox[0], bbox[3], res, res),
        'blockxsize': int(2/res), 
        'blockysize': int(2/res), 
        'tiled': True, 
        'compress': 'lzw', 
        'interleave': 'band'
    }

    out_dir = join(ddir, 'impact')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fn_elevtn = join(sdir, 'map', 'elevtn.tif')
    with rasterio.open(fn_elevtn, 'r') as src:
        mv_mask = src.read(1) == src.nodata
        shape = src.shape
        prof_agg = src.profile

    fn_regions = join(sdir, 'map', 'hires', 'location.txt')
    regions = pd.read_csv(fn_regions, delim_whitespace=True, index_col=0).T \
                .set_index('area').astype(float).to_dict(orient='index')

    for T in rps:
        fn_flddph_bin = join(sdir, f'flddph_T{T:03.0f}')
        fn_out_tif_agg = join(out_dir, f'{exp_name}_agg_T{T:03.0f}.tif')
        fn_out_tif = join(out_dir, f'{exp_name}_T{T:03.0f}.tif')
        # if os.path.isfile(fn_out_tif_agg): continue
        
        # output
        data_agg = np.zeros(shape, dtype=np.float32)
        with rasterio.open(fn_out_tif, 'w', **profile) as dst:
            pass # write empty file
        print(f'rp: {T:03.1f}')
        
        # downscale
        for area in regions.keys():
            print(area)
            msg = ['./flood_impact', str(area), basename(fn_flddph_bin), str(exp_name), '1']
            subprocess.call(msg, cwd=sdir, stderr=subprocess.STDOUT)

            # open binary output and add
            fn_impact_agg = join(sdir, '{:s}.impact_agg'.format(area))
            if os.path.isfile(fn_impact_agg):
                with open(fn_impact_agg, 'r') as fid:
                    data0 = np.fromfile(fid, 'f4').reshape(shape)
                    data0 = np.where(data0==mv, nodata, data0)
                    data_agg += data0
                # remove binary output
                os.unlink(fn_impact_agg)

            fn_impact = join(sdir, '{:s}.impact'.format(area))
            if os.path.isfile(fn_impact):
                # read binary output
                ny, nx = int(regions[area]['ny']), int(regions[area]['nx'])
                with open(fn_impact, 'r') as fid:
                    data1 = np.fromfile(fid, 'f4').reshape(ny, nx)
                    data1 = np.where(data1==mv, nodata, data1)
                # read wdw from tif
                west, north = regions[area]['west'], regions[area]['north']
                wdw_bounds = west, np.round(north-ny*res), np.round(west+nx*res), north #NOTE all bbox are rounded to full degrees
                # append to geotiff
                with rasterio.open(fn_out_tif, 'r+') as dst:
                    if wdw_bounds[2] > 180:
                        nnx = int(np.round((wdw_bounds[2]-180)/res))
                        wdw_bounds2 = -180, np.round(north-ny*res), -180+wdw_bounds[2]-180, north
                        window=rasterio.windows.from_bounds(*wdw_bounds2, transform=dst.transform)
                        data0 = dst.read(1, window=window)
                        data2 = np.where(data1[:,-nnx:] == nodata, data0, data1[:,-nnx:])
                        dst.write(data2, window=window, indexes=1)
                        wdw_bounds = west, np.round(north-ny*res), 180, north
                        data1 = data1[:,:-nnx]
                    window=rasterio.windows.from_bounds(*wdw_bounds, transform=dst.transform)
                    data0 = dst.read(1, window=window)
                    data1 = np.where(data1 == nodata, data0, data1)
                    dst.write(data1, window=window, indexes=1)
                # remove binary output
                os.unlink(fn_impact)
            else:
                print(' '.join(msg))
        
        # write aggregated data to geotiff
        data_agg = np.where(mv_mask, nodata, data_agg)
        with rasterio.open(fn_out_tif_agg, 'w', **prof_agg) as dst:
            dst.write(data_agg, 1)

if __name__ == "__main__":
    #
    root = r'/scratch/compound_hotspots/'
    map_dir = r'/home/dirk/models/cama-flood_bmi_v3.6.2_nc/map'
    ddir = join(root, 'data', '3-model_output')
    sdir = join(map_dir, 'downscale_flddph')
    os.chdir(sdir)
    
    # mask
    map_dir = r'/home/dirk/models/cama-flood_bmi_v3.6.2_nc/map/global_15min'
    rm = {'x':'lon', 'y':'lat'}
    # lecz = xr.open_rasterio(join(map_dir, 'LEZC_10m_basin.tif')).drop('band').squeeze().rename(rm)
    elevtn = xr.open_rasterio(join(sdir, 'map', 'elevtn.tif')).drop('band').squeeze().rename(rm)
    landmask = elevtn != -9999

    for scen in ['cmpnd', 'runoff']:
        ddir0 = join(ddir, f'anu_mswep_{scen}_v362_1980-2014')
        fn_out = join(ddir0, f'ev_map_sfcelv.nc')
        # if not os.path.isfile(fn_out):
        ev_map(
            ddir = ddir0, 
            fn_out = fn_out,
            var = 'sfcelv',
            mask = landmask,
            )
        print('preparing binary input for fortran routines ..')
        fn_swe = join(ddir0, f'ev_map_sfcelv.nc')
        set_bin_inputs(sdir, fn_swe)

        print('downscaling flddph ..')
        downscale_flddph(
            sdir = sdir, 
            ddir = ddir0,
            rps=np.array([2, 10, 50, 100]),
        )

        print('impact assessment ..')
        flood_impact(
            sdir = sdir, 
            ddir = ddir0,
            rps=np.array([2, 10, 50, 100]),
            exp_name= 'worldpop'
        )