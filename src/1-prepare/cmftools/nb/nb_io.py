#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author: Dirk Eilander (contact: dirk.eilancer@vu.nl)
# Created: Nov-2017

from .dd_ops import LDD, NextXY
import rasterio
import numpy as np

def read_dd_rasterio(fn, ddtype='ldd', **ddkwargs):
    with rasterio.open(fn, 'r') as src:
        if ddtype == 'ldd':
            dd_r = LDD(src.read(1), src.transform, nodata=src.nodata, **ddkwargs)
        elif ddtype == 'nextxy':
            dd_r = NextXY(src.read(), src.transform, nodata=src.nodata, **ddkwargs)
    return dd_r 

def read_dd_pcraster(fn, transform, nodata=255):
    import pcraster as pcr
    lddmap = pcr.readmap(str(fn))
    ldd_data = pcr.pcr2numpy(lddmap, nodata)
    ldd = LDD(ldd_data, transform, nodata=nodata)
    return ldd 

def read_dd_cmfbin(fn, transform, height, width, nodata=-9999, **ddkwargs):
    a = np.fromfile(file=str(fn), dtype='int32').reshape((2, int(height), int(width)))
    nextxy = NextXY(a, transform, nodata=nodata, **ddkwargs)
    return nextxy 

def read_raster(fn):
    return None

def write_raster_like(fn_out, fn_like, raster, **kwargs):
    
    # copy input gtiff profile
    with rasterio.open(fn_like, 'r') as src:
        assert src.shape == raster.shape[-2:], "number of rows and cols should match"
        profile = src.profile.copy()
    count = raster.shape[0] if raster.ndim == 3 else 1
    profile.update(count=count, dtype=str(raster.dtype))
    profile.update(**kwargs)
    # write output
    with rasterio.open(fn_out, 'w', **profile) as dst:
        dst.write(raster, 1)