#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Dirk Eilander (contact: dirk.eilancer@vu.nl)
# Created: May 2018

import os
import pandas as pd
import geopandas as gp 
from  shapely.geometry import Point
import rasterio
import numpy as np
# local libraries
from cmftools.cmf_index import model_xy, outlet_xy, model_index
from nbops import read_dd_raster, catchment_map


def get_outlets(fn_nextxy, fn_lonlat, sample_dict={}, fn_out=None, 
                crs={'init': 'epsg:4326'}, res_str='15min'):
    """
    """
    # get outlets
    for fn in [fn_nextxy, fn_lonlat]:
        if not os.path.isfile(fn): 
            raise IOError("{} file not found".format(fn))
    nextxy = read_dd_raster(fn_nextxy, ddtype='nextxy')
    row, col = nextxy.get_pits()
    lon, lat, _ = outlet_xy(row, col, fn_lonlat)
    lon_lr, lat_lr, _ = model_xy(row, col, fn_lonlat=fn_lonlat)
    meta_dict = {'lon': lon, 'lat': lat, 
                 'lon_{}'.format(res_str): lon_lr, 'lat_{}'.format(res_str): lat_lr,
                 'col_{}'.format(res_str): col, 'row_{}'.format(res_str): row}
    for name in sample_dict:
        meta_dict[name] = sample_map(lon, lat, sample_dict[name])
    # convert outlet points to geopandas
    df = pd.DataFrame(meta_dict)
    if fn_out is not None:
        if fn_out.endswith('.txt') or fn_out.endswith('.csv'):
            df.to_csv(fn_out)
        else:
            pnts = [Point(xy) for xy in zip(lon, lat)]
            gdf = gp.GeoDataFrame(df.drop(['lat', 'lon'], axis=1), crs=crs, geometry=pnts)
            gdf.to_file(fn_out)
    return df

def get_catchments(fn_outlets, fn_nextxy, fn_out=None,
                  index_col=0, x_col='lon', y_col='lat', ddkwargs={}):
    """
    """
    catm = catchment_map(fn_outlets=fn_outlets, fn_dd=fn_nextxy, fn_out=fn_out, 
        index_col=index_col, x_col=x_col, y_col=y_col, 
        ddtype='nextxy', ddkwargs=ddkwargs)
    return catm


def sample_map(x, y, fn_map, fn_catmxy=None, layer=1):
    """
    """
    if not os.path.isfile(fn_map): 
        raise IOError("{} file not found".format(fn_map))
    with rasterio.open(fn_map, 'r') as src:
        if fn_catmxy is None:
            # assume low resolution lat lon coordinates are given
            r, c = src.index(x, y)
            r, c = np.atleast_1d(r).astype(int), np.atleast_1d(c).astype(int)
            nrows, ncols = src.shape
            valid = np.logical_and.reduce((r>=0, r<nrows, c>=0, c<ncols))
        else:
            # convert to low resolution row col using catmxy index
            r, c, valid = model_index(x, y, fn_catmxy)
        # if the fill domain fits into memory that's faster than using the rasterio sample function
        sample = np.ones(r.size, dtype=src.dtypes[layer-1])*np.nan
        sample[valid] = src.read(layer)[r[valid], c[valid]]
    return sample
        

# # convert binary files to geopnadas DataFrame (easy to write to e.g. shapefiles)
# def map2outlets(fn_maps, crs={'init': 'epsg:4326'}, xlim=(-180, 180), ylim=(-90,90)):
#     """derive outlet GeoDataFrames with attribute data from netcdf mapfiles
#     the netcdf file is created using the cama_binmaps2nc.py script"""
#     lats, lons, row, col = slice2latlon(fn_maps, xlim, ylim)
#     gdf_outlets = get_outlets(fn_maps, lats, lons, crs=crs)
#     gdf_outlets['cama_iy'] = row
#     gdf_outlets['cama_ix'] = col
#     return gdf_outlets

# def get_outlets(fn_maps, lats, lons, crs={'init': 'epsg:4326'}, method=None):
#     """derive outlet GeoDataFrames with attribute data from netcdf mapfiles at
#     the netcdf file is created using the cama_binmaps2nc.py script"""
#     with xr.open_dataset(fn_maps) as ds_maps:
#         df = ds_maps.sel_points(dim='index', lat=lats, lon=lons, method=method).to_dataframe()
#         # df = df.drop(['lon', 'lat'], axis=1)

#     # set column types correctly
#     area_var = ['uparea', 'grarea']
#     for v in area_var:
#         if v in df.columns:
#             df[v] = df[v].astype('f8') / 1E6 # km2
#     int_var = ['basin', 'bsncol', 'lsmask', 'upgrid', 'nexty', 'nextx']
#     for v in int_var:
#         if v in df.columns:
#             df[v] = df[v].fillna(-9999)
#             df[v] = df[v].astype('i8')

#     # save outlet point shapefile
#     geometry = xy2point(df, xname='outlon', yname='outlat')
#     gdf_outlet = gp.GeoDataFrame(df, crs=crs, geometry=geometry)
#     return gdf_outlet

# def map2network(fn_maps, crs={'init': 'epsg:4326'}, xlim=(-180, 180), ylim=(-90,90)):
#     """derive outlet and network GeoDataFrames with attribute data from netcdf mapfiles
#     the netcdf file is created using the cama_binmaps2nc.py script"""
#     gfd_outlet = map2outlets(fn_maps, xlim=xlim, ylim=ylim, crs=crs)
#     gdf_network = get_network(fn_maps, gfd_outlet, crs=crs)
#     return gfd_outlet, gdf_network

# def get_network(fn_maps, gfd_outlet, crs={'init': 'epsg:4326'}):
#     """derive network between outlet points based on netcdf mapfiles and
#     GeoDataFrame of outlet points with nextx & netxy attribute data"""
#     # find downstream outlet lat/lon
#     attrs = pd.DataFrame(gfd_outlet).drop(['geometry'], axis=1)
#     idx_next = [i for i,nxty in zip(attrs.index, attrs['nexty'].values)
#                     if (nxty != -9) and np.isfinite(nxty)]
#     attrs = attrs.loc[idx_next, :]
#     del gfd_outlet

#     with xr.open_dataset(fn_maps) as ds_maps:
#         # read unit catchment map data including lat/lon of outlet points
#         row_vec = np.arange(len(ds_maps.lat.values))[::-1] # mirror y/row axis

#         nxt_irows = row_vec[attrs['nexty'].values.astype('int') -1] # convert to python zero indexing
#         nxt_icols = attrs['nextx'].values.astype('int') -1
#         nxt_attrs = ds_maps[['outlon', 'outlat']].isel_points(
#                         dim=attrs.index, lat=nxt_irows, lon=nxt_icols).to_dataframe()
#         nxt_attrs = nxt_attrs.drop(['lon', 'lat'], axis=1)
#         nxt_attrs.rename(columns={'outlat': 'nxtlat', 'outlon': 'nxtlon'}, inplace=True)

#     # concat attrs with next outlet lat/lon info
#     df = pd.concat( [attrs, nxt_attrs], axis=1)

#     # save to geopandas
#     geometry = [LineString(((lon0,lat0),(lon1,lat1))) for lon0, lat0, lon1, lat1
#                     in zip(df.outlon, df.outlat, df.nxtlon, df.nxtlat)]
#     gdf_network = gp.GeoDataFrame(df.drop(['outlon', 'outlat', 'nxtlon', 'nxtlat'], axis=1),
#                                     crs=crs, geometry=geometry)
#     return gdf_network

# def slice2latlon(fn_maps, xlim=(-180, 180), ylim=(-90,90)):
#     """small helper function to get non-missing row/cols from 2D-slice of data"""
#     ds_maps = xr.open_dataset(fn_maps)
#     with xr.open_dataset(fn_maps) as ds_maps:
#         # slice domain from map
#         ds_maps_slice = ds_maps.sel(lon=slice(*xlim), lat=slice(*ylim))
#         row_offset = np.where(ds_maps.lat.values == ds_maps_slice.lat.values[0])[0]
#         col_offset = np.where(ds_maps.lon.values == ds_maps_slice.lon.values[0])[0]
#         lrow, lcol = np.where(np.isfinite(ds_maps_slice.uparea)) # local row/col within domain
#         lats = ds_maps_slice.lat.values[lrow]
#         lons = ds_maps_slice.lon.values[lcol]
#     ds_maps.close()
#     return lats, lons, lrow + row_offset, lcol + col_offset

