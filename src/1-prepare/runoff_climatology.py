
# coding: utf-8

import xarray as xr
import pandas as pd
from os.path import basename, isfile, join, dirname
import glob
from datetime import datetime, timedelta
import os
import warnings
from shutil import copyfile
warnings.filterwarnings("ignore")


# NOTE: this runs from the data directory
ddir = dirname(os.path.realpath(__file__))
print(ddir)
cwd = os.getcwd()
num_workers=getattr(os.environ, 'OMP_NUM_THREADS', 16)
print('using {:d} workers'.format(num_workers))
encoding={'Runoff':  {'zlib': True}}
# assume the data and script are in the same folder
names = [basename(n).split('_')[1] for n in glob.glob(join(ddir, r'e2o_*_wrr2_glob15_day_Runoff_1980.nc'))]
print(names)
for name in names:
    print(name)
    fn_form = join(ddir, r'e2o_{:s}_wrr2_glob15_day_Runoff_*[0-9].nc'.format(name))
    fns = glob.glob(fn_form)
    # copy data 
    if dirname(fns[0]) != cwd:
        for fn in fns:
            copyfile(fn, basename(fn))
    # read data
    fn_out = '{:s}_climatology.nc'.format('_'.join(fns[0].split('_')[:-1]))
    if isfile(fn_out): continue
    print(fn_out)
    ds = xr.open_mfdataset(fns, chunks={'lat':120, 'lon':120})
    global_attrs = {'source_files': '; '.join(fns)}
    global_attrs.update(ds.attrs)
    # do the work
    with dask.set_options(get=dask.threaded.get, num_workers=num_workers): #, ProgressBar():
        da = ds.Runoff.groupby('time.dayofyear').mean('time').rename({'dayofyear': 'time'})
        da.time.attrs.update({'long_name': 'dayofyear'})
        ds_out = da.to_dataset()
        ds_out.attrs = global_attrs
        ds_out.to_netcdf(basename(fn_out), encoding=encoding)
    ds.close()
    # copy data 
    if dirname(fns[0]) != cwd:
        copyfile(basename(fn_out), fn_out)
        # cleanup
        os.remove(basename(fn_out))
        for fn in fns:
            os.remove(basename(fn))
