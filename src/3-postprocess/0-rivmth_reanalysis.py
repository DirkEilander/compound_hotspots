import xarray as xr
from os.path import join, basename
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import glob

root = r'/scratch/compound_hotspots'
ddir = join(root, 'data')
# IN
fn_csv_coupling = join(root, 'src', '1-prepare', r'cmf_gtsm_75km_update191017.csv')
fn_gtsm = join(root, 'data', '2-preprocessed', f'gtsm_*_select_day.nc')
fn_cmf_format = join(ddir, '3-model_output', r'rivmth_{model}_mswep_{scen}_v362_1980-2014.nc')
#OUT
fn_zarr_out = join(ddir, '4-postprocessed', 'rivmth_reanalysis.zarr')

t0, t1 = datetime(1980,1,2), datetime(2014,12,30)   # NOTE: cama date at 00hrs of 'next day' and missing GTSM values at 31-12-2014
scen_rm = {
    'cmpnd':   'surge', 
    'runoff':  'seas',
    'tide':    'tide',
}
var_rm = {
    'outflw': 'Q',
    'sfcelv': 'WSE'
}

# attrs
glob_attrs = dict(
    institution = 'Institute for Environmental Studies (IVM) - Vrije Universiteit Amsterdam',
    author = 'Dirk Eilander (dirk.eilander@vu.nl)',
    date_created = str(datetime.now().date()),
    history = f'created using {basename(__file__)}; xarray v{xr.__version__}',
)

# read coupling csv
coupling = pd.read_csv(fn_csv_coupling, index_col='index')

# check if all outputs are complete
fns = glob.glob(join(ddir, 'rivmth_*_mswep_*_v362_1980-2014*.nc'))
# check if all complete
for fn in fns:
    if xr.open_dataset(fn).time.to_index()[-1].year != 2015:
        print(fn)
        print(xr.open_dataset(fn).time.data[-1])

# combine cmf outputs
model = ['anu', 'cnrs', 'ecmwf', 'nerc', 'jrc'] #, 'univu', 'univk']
scenarios = ['msl', 'runoff', 'tide', 'cmpnd'] # 'surge',
rm_coords = {'id': 'rivmth_idx'}
chunks = {'time': -1, 'id':100}
ds_m = []
for m in model:
    ds_t = []
    for scen in scenarios:
        ds_t.append(xr.open_dataset(fn_cmf_format.format(model=m, scen=scen), chunks=chunks)[['sfcelv', 'outflw']].sortby('time'))
    ds_t = xr.concat(ds_t, dim='scen')
    ds_t['scen'] = xr.Variable(['scen'], np.asarray(scenarios).astype(str))
    ds_m.append(ds_t)
ds_cmf = xr.concat(ds_m, dim='ensemble').rename(var_rm)
ds_cmf['ensemble'] = xr.Variable(['ensemble'], np.asarray(model).astype(str))
ds_cmf = ds_cmf.rename(**rm_coords).sel(rivmth_idx=coupling.rivmth_idx.values).drop(['lat_nc', 'lon_nc'])
ds_cmf['index'] = xr.DataArray(dims=['rivmth_idx'], data=coupling.index.values, name='index')
ds_cmf = ds_cmf.swap_dims({'rivmth_idx': 'index'})
ds_cmf['rivmth_lat'] = xr.Variable(['index'], coupling['rivmth_lat'].values)
ds_cmf['rivmth_lon'] = xr.Variable(['index'], coupling['rivmth_lon'].values)
ds_cmf = ds_cmf.set_coords(['rivmth_lat', 'rivmth_lon'])
# select and rename scenarios
ds_cmf_sel = xr.merge([
    ds_cmf[['WSE', 'Q']].sel(scen=list(scen_rm.keys())),
    ds_cmf[['Q']].sel(scen='msl').drop('scen').rename({'Q':'Qmsl'}) # take discharge from msl scenario
])
ds_cmf_sel['scen'] = xr.Variable('scen', [scen_rm.get(s,s) for s in ds_cmf_sel['scen'].data])

# combine gtsm outputs
rm = {
    'station_y_coordinate': 'gtsm_lat', 
    'station_x_coordinate': 'gtsm_lon',
    'station_id': 'gtsm_station_id',
    'stations': 'index',
}
drop = ['station_name']
ds_gtsm = xr.open_mfdataset(fn_gtsm).rename(rm).sel(index=coupling.index.values).drop(drop).sortby('time')
assert np.all(ds_gtsm.gtsm_station_id.values.astype('<U20') == coupling.gtsm_station_id.values.astype('<U20'))

# merge
ds_out = xr.merge([ds_cmf_sel, ds_gtsm]).sel(time=slice(t0,t1))
# remove untrusted months -> something funny in gtsm during oct/nov 1990
ds_out = ds_out.where(ds_out.time!=slice(datetime(1990,10,1), datetime(1991,3,1)))

# to zarr
chunks = {'ensemble':-1, 'scen':-1, 'time': -1, 'index':100}
ds_out.attrs.update(glob_attrs)
print(ds_out)
ds_out.transpose('scen', 'ensemble', 'time', 'index').chunk(chunks).to_zarr(fn_zarr_out, mode='w')