#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from os.path import join, dirname, realpath, basename
import rasterio
import click
import glob
import subprocess
import xarray as xr
import pandas as pd
from datetime import timedelta
from rasterio.transform import from_origin

@click.command()
@click.argument('ddir')
@click.argument('area')
@click.argument('time')
@click.option('-p', '--postfix', default='')
def downscale(ddir, area, time, postfix='', dt=-1):
    # parse time
    t = pd.to_datetime(time)
    # read regions info
    sdir = dirname(realpath(__file__))
    fn_regions = join(sdir, 'map', 'hires', 'location.txt')
    click.echo(fn_regions)
    regions = pd.read_csv(fn_regions, delim_whitespace=True, index_col=0).T \
                .set_index('area').astype(float).to_dict(orient='index')

    # read nc
    fn_nc = join(ddir, 'flddph*.nc')
    ds = xr.open_mfdataset(fn_nc, chunks={'time': 10})
    if dt != 0:
        ds['time'] = ds.time.to_index() + timedelta(days=dt)
    data = ds.flddph.sel(time=time).data
    data = np.where(np.isnan(data), 1e+20, data) # mv = 1e20

    # write to bin
    datestr = '{:04d}{:02d}{:02d}'.format(t.year, t.month, t.day)
    fn_out_bin = join(sdir, basename(fn_nc).replace('*.nc', datestr))
    click.echo(fn_out_bin)
    with open(fn_out_bin, 'w') as fid:
        fid.write(data.astype('f4').tobytes())

    # downscale
    click.echo('downscaling...')
    msg = ['./downscale_flddph', str(area), basename(fn_out_bin), '1']
    click.echo(' '.join(msg))
    subprocess.call(msg, cwd=sdir, stderr=subprocess.STDOUT)

    # open binary output
    fn_fld = join(sdir, '{:s}.flood'.format(area))
    ny, nx = int(regions[area]['ny']), int(regions[area]['nx'])
    with open(fn_fld, 'r') as fid:
        data = np.fromfile(fid, 'f4').reshape(ny, nx)

    # write to geotiff
    fn_out_tif = join(ddir, basename(fn_out_bin) + postfix + '.tif')
    click.echo('writing to ' + fn_out_tif)
    west, north, csize = regions[area]['west'], regions[area]['north'], regions[area]['csize']
    transform = from_origin(west, north, csize, csize)
    with rasterio.open(fn_out_tif, 'w', driver='GTiff', height=data.shape[0],
                compress='lzw', width=data.shape[1], count=1, dtype=str(data.dtype),
                crs='+proj=latlong', transform=transform, nodata=-9999) as dst:
        dst.write(data, 1)

    # remove binary output
    os.unlink(fn_out_bin)
    os.unlink(fn_fld)

if __name__ == "__main__":
    downscale()
