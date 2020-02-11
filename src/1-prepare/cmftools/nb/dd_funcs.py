#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author: Dirk Eilander (contact: dirk.eilancer@vu.nl)
# Created: May-2019
import rasterio
import pandas as pd
import os
# local
from .nb_io import read_dd_rasterio, write_raster_like

def catchment_map(fn_outlets, fn_dd, fn_out=None, 
                  index_col=0, x_col='lon', y_col='lat', 
                  ddtype='ldd', ddkwargs={}):
    """get catchments gtif based on outlet x,y points
    
    #TOTO describe input output params
    
    """
    # read x, y of outlets from txt file
    if not os.path.isfile(fn_outlets): 
        raise IOError("{} file not found".format(fn_outlets))
    outlets = pd.read_csv(fn_outlets, index_col=index_col)
    y, x, idx = outlets[y_col].values, outlets[x_col].values, outlets.index
    # read drainage direction file
    dd = read_dd_rasterio(fn_dd, ddtype=ddtype, ddkwargs=ddkwargs)
    # translate x, y to row, col
    rows, cols, valid = dd.index(x, y)
    # get catchments
    catm = dd.get_catchment(rows[valid], cols[valid], idx[valid])
    # copy input gtiff profile
    if fn_out is not None:
        write_raster_like(fn_out, fn_dd, catm)
    return catm