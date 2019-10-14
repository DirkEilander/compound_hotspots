#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from os.path import join
import rasterio
import pandas as pd
import rtree
import geopandas as gp
from shapely.geometry import Point
# local
from cmftools.cmf_maps_analyze import get_outlets

def sjoin_nn(points1, points2, max_dist=np.inf, prefix='p2'):
    """function for spatial join of the nearest neighbour from points2 to points1
    :param points1:     geopandas GeoDataFrame with point geometry
    :param points2:     geopandas GeoDataFrame with point geometry
    :param max_dist:    float. maximum distance between two nn points. in km if latlon
    :param prefix:      string used as prefix for metadata from points2
    :param latlon:      if True (calculate distance on unit sphere in lat lon coordinates)"""
    assert points1.crs == points2.crs, 'the point geometries should have the same crs'
    points1 = points1.copy()
    distf = latlon_distance
    x_col, y_col = prefix + '_lon', prefix + '_lat'

    # build spatial rtree index of points2
    tree_idx = rtree.index.Index()
    for i, geom in enumerate(points2.geometry):
        tree_idx.insert(i, geom.coords[:][0])

    # create new columns for output
    dist_col, idx_col = prefix + '_dist', prefix + '_idx'
    result = pd.DataFrame(columns=[x_col, y_col, dist_col, idx_col], index=points1.index)
    # link river outlet and gtsm points
    for idx in points1.index:
        p1 = points1.loc[idx, :]
        # find p2 nearest to p1
        p2_idx = list(tree_idx.nearest(p1.geometry.coords[:][0], 1))[0]
        p2 = points2.iloc[p2_idx, :]
        # calculate distance
        # x1, y1 = (p1.geometry.xy[0][0], p1.geometry.xy[1][0])
        dist = distf(p1.geometry.coords[:][0], p2.geometry.coords[:][0])
        # save link data
        if dist <= max_dist:
            x2, y2 = (p2.geometry.xy[0][0], p2.geometry.xy[1][0])
            result.loc[p1.name, [x_col, y_col, dist_col, idx_col]] = x2, y2, dist, points2.index[p2_idx]
    # remove points without link
    result = result.dropna()

    # merge other attribute data
    cols = [name for name in points2.columns if name != 'geometry']
    new_cols = ['{}_{}'.format(prefix, name) for name in cols]
    for col, new_col in zip(cols, new_cols):
        result[new_col] = points2.loc[result[idx_col], col].values

    return pd.concat([points1, result], axis=1)

# distance 
def latlon_distance(p1, p2, R=6373000):
    """
    the local distance varies with the latitude of the earth. this function
    calculates the local resolution of a grid given its latitude

    :param p1: list of (lon,lat) tuples
    :param p1: list of (lon,lat) tuples
    :param R:   radius earth in local unit (R = 6373 for km)
    :return:    eucledian distance in local unit
    """
    try: # for lists
        lon1, lat1 = zip(*p1)
        lon2, lat2 = zip(*p2)
        lon1, lat1 = np.asarray(lon1), np.asarray(lat1) 
        lon2, lat2 = np.asarray(lon2), np.asarray(lat2)
    except TypeError: # for single points
        lon1, lat1 = p1
        lon2, lat2 = p2
    degrees_to_radians = np.pi/180.0
    # Convert latitude and longitude to spherical coordinates in radians.
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    theta1 = lon1*degrees_to_radians
    theta2 = lon2*degrees_to_radians
    # Compute spherical distance from spherical coordinates.
    cos = (np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) +
        np.cos(phi1)*np.cos(phi2))
    # multiply arc by the radius of the earth in local units to get length.
    d = np.arccos(cos) * R
    return d

def pandas2geopandas(df, x_col='lon', y_col='lat', crs={'init': u'epsg:4326'}):
    geoms = [Point(x, y) for x, y in zip(df[x_col], df[y_col])]
    return gp.GeoDataFrame(df.drop([x_col, y_col], axis=1), geometry=geoms, crs=crs)

# def nn_coupling_rivmth_gtsm(ref_bin_out_fn, gdf_gtsm, fn_nextxy_tif, fn_lonlat_tif, fn_uparea_tif, fn_dist2coast_tif, 
#                             min_uparea=1e9, max_dist=10e3, gtsm_index_col='gtsm_idx'):
#     # find river mouths 
#     sample_dict = {'uparea': fn_uparea_tif, 'dist2coast': fn_dist2coast_tif}
#     rivmth = get_outlets(fn_nextxy_tif, fn_lonlat_tif, sample_dict=sample_dict, res_str='15min')
#     rivmth['dist2coast'] = np.floor(rivmth['dist2coast'].values)
#     # select
#     rivmth[np.logical_and(rivmth['uparea'].values>min_uparea, rivmth['dist2coast'].values<max_dist)]
#     # nn join
#     # TODO find nearest with distance measured over sea
#     gdf_merge = sjoin_nn(pandas2geopandas(rivmth), gdf_gtsm, max_dist=max_dist, prefix='gtsm')
#     # save binary output data
#     ref = np.array([])
#     for idx in gdf_merge.index:
#         cama_iy, cama_ix = gdf_merge.loc[idx, 'row_15min'], gdf_merge.loc[idx, 'col_15min']
#         gtsm_idx = gdf_merge.loc[idx, gtsm_index_col]
#         ref = np.append(ref, cama_iy + 1) # fortran index
#         ref = np.append(ref, cama_ix + 1) # fortran index
#         ref = np.append(ref, gtsm_idx)
#     ref.astype('f').tofile(ref_bin_out_fn)
#     return gdf_merge