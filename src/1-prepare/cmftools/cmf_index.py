#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Dirk Eilander (contact: dirk.eilancer@vu.nl)
# Created: June 1st

import os
import rasterio
from rasterio.transform import from_origin
import numpy as np

def cmf_index(df, fn_catmxy, fn_lonlat, x_col='lon', y_col='lat'):
    row, col, _ = model_index(df[x_col], df[y_col], fn_catmxy)
    lon_lr, lat_lr, _ = model_xy(row, col, fn_lonlat=fn_lonlat)
    df['cmf_row'], df['cmf_col'] = row, col
    df['cmf_lon'], df['cmf_lat'] = lon_lr, lat_lr 
    return df

def model_index(x, y, fn_catmxy):
    """Get CMF (row, col) indices at low resolution based on xy coordinates.
    The low-res indices are looked up catmxy geotiff file 'reg.catmxy.tif'.

    To convert cama binary maps to geotiff use the 'cama_maps_io.py' script.

    Note that CMF indices smaller than zero should be ignored! This occurs
    for unitcatchments that are not represented in CMF or fall out of
    the CMF domain. for both (row, col) = (-1, -1). CMF is volume conservative,
    so the runoff of ignored unitcatchments is conserved by spreading it
    of other unitcatchments in the same cell.

    Arguments
    ---------
    xy : list of tuples
        list of (x, y) coordinate tuples
    fn_catmxy : str
        filename of CMF catchment index geotiff file

    Returns
    -------
    rc : list of tuples
        list of (row, col) index tuples
    """
    if not os.path.isfile(fn_catmxy):
        raise IOError("{} file not found".format(fn_catmxy))

    with rasterio.open(fn_catmxy, 'r') as ds:
        if ds.count != 2:
            raise ValueError("{} file should have two layers".format(fn_catmxy))
        # sample low res row, col from high res CMF xy grid; using sample as highres may not fit in memory
        c, r = zip(*list(ds.sample(zip(x, y))))
        # go from fortran one-based to python zero-based indices
        r, c = (np.atleast_1d(r)-1).astype(int), (np.atleast_1d(c)-1).astype(int)
        # check valid indices -> -1 & -9999 values should be ignored
        valid = np.logical_and(r>=0, c>=0)
    return r, c, valid

def model_xy(r, c, offset='center', fn_lonlat=None, west=-180, north=90, res=0.25, shape=(np.inf, np.inf)):
    """Get CMF (lon, lat) coordinates at low resolution based on row, col indices.

    To convert cama binary maps to geotiff use the 'cama_maps_io.py' script.

    Arguments
    ---------
    rc : list of tuples
    list of (row, col) index tuples
    offset='center' : str, optional
    Determines if the returned coordinates are for the center of the pixel or for a corner.


    Returns
    -------
    xy : list of tuples
    list of (x, y) coordinate tuples
    """
    # if latlon file is given, read transform and shape from file
    if os.path.isfile(fn_lonlat):
        with rasterio.open(fn_lonlat, 'r') as src:
            transform = src.transform
            nrows, ncols = src.shape
    else: # else, set based on input arguments
        transform = from_origin(west, north, res, res)
        nrows, ncols = shape

    # r, c = zip(*rc)
    r, c = np.atleast_1d(r).astype(int), np.atleast_1d(c).astype(int)

    valid = np.logical_and.reduce((c>=0, r>=0, c<ncols, r<nrows))
    lon, lat = np.zeros_like(c).astype(float), np.zeros_like(r).astype(float)
    lon[valid], lat[valid] = transform * (c[valid], r[valid])
    if offset=='center':
        lon[valid], lat[valid] = lon[valid]+res/2., lat[valid]-res/2.
    return lon, lat, valid

def outlet_xy(r, c, fn_lonlat):
    """Get CMF (lon, lat) coordinates at low resolution based on row, col indices.
    The coordinates are the cell centers if outlets==False (default), 
    or the unit catchment outlets if outlets==True

    To convert cama binary maps to geotiff use the 'cama_maps_io.py' script.

    Arguments
    ---------
    rc : list of tuples
        list of (row, col) index tuples

    Returns
    -------
    xy : list of tuples
        list of (x, y) coordinate tuples
    """
    if not os.path.isfile(fn_lonlat): 
        raise IOError("{} file not found".format(fn_lonlat))

    # r, c = zip(*rc)
    r, c = np.atleast_1d(r).astype(int), np.atleast_1d(c).astype(int)
    # read out outflow lat lon points
    with rasterio.open(fn_lonlat, 'r') as ds:
        if ds.count != 2:
            raise ValueError("{} file should have two layers".format(fn_lonlat))
        nrows, ncols = ds.shape
        valid = np.logical_and.reduce((r>=0, r<nrows, c>=0, c<ncols))
        lat, lon = np.ones(valid.size) * np.nan, np.ones(valid.size) * np.nan   
        lon[valid], lat[valid] = ds.read()[:, r[valid, None], c[valid, None]].squeeze()
    return lon, lat, valid