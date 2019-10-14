import numpy as np
import geopandas as gp
import pandas as pd
from shapely.geometry import Point, mapping, box
import matplotlib.pyplot as plt
import os
import rasterio
import rasterio.mask
from pyproj import Proj, transform

# raster IO
def read_gtiff_buf(fn, xy, buf, layer=0):
    with rasterio.open(fn, 'r') as src:
        utm_xy, utm_proj = to_utm([xy], Proj(**src.crs))
        utm_mask = Point(utm_xy).buffer(buf)
        mask = gp.GeoDataFrame(geometry=[utm_mask], crs=utm_proj.srs).to_crs(src.crs).geometry[0]
        mask = mask.intersection(box(*src.bounds))
        if mask.is_empty:
            raise ValueError('xy {} outside raster domain'.format(xy))
        mask = mapping(mask)
        data, transform = rasterio.mask.mask(src, [mask], crop=True) 
        if layer is not None:
            data = np.ma.masked_equal(data[layer, :, :], src.nodata)
        else:
            data = np.ma.masked_equal(data, src.nodata)
    return data, transform

# gistools
def haversine_dist(pos1, pos2, r=6378137.0):
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

def calc_haversine_dist(gauge_xy, drains_2d, transform, fill_val=0):
    """calculate distances [m] for True cells in bool array"""
    r, c = np.where(drains_2d)[-2:]
    xy = np.array(list(map(list, zip(*(transform * (c, r))))))
    dist_vec = haversine_dist(np.array(gauge_xy)[None, :], xy) #m
    dist_2d = np.ones(drains_2d.shape, dtype=float)*fill_val
    dist_2d[r, c] = dist_vec
    return np.ma.masked_equal(dist_2d, fill_val) # mask fill_val

def latlon_to_zone_number(latitude, longitude):
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32
    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37
    return int((longitude + 180) / 6) + 1

def to_utm(xy, proj_in):
    xs, ys = zip(*xy) # assume csr epsg:4326
    utm_zone = latlon_to_zone_number(np.mean(ys), np.mean(xs))
    proj_out = Proj(**{'proj': "utm", 'zone': utm_zone, 'datum': 'WGS84'})
    return to_crs(xy, proj_out, proj_in=proj_in), proj_out

def to_crs(xy, proj_out, proj_in):
    return zip(*transform(proj_in, proj_out, *zip(*xy)))

# snapping
def snap_minimum(raster, transform):
    """return row, col and x, y coordinates of smallest value in raster"""
    vmin = raster.min()
    row, col = np.where(raster==vmin)[-2:]
    if row.size > 1:
        row, col = row[:1], col[:1]
    x, y = transform * (col, row)
    return (x[0], y[0]), (int(row[0]), int(col[0])), vmin

def snap_nearest_haversine(gauge_xy, drains_2d, transform):
    """return row, col and x, y coordinates of nearest drain
    The distance is measured using the haversine method.
    """
    # get distance map for drain
    dist_2d = calc_haversine_dist(gauge_xy, drains_2d, transform=transform, fill_val=np.inf)
    # find nearest
    (x, y), (row, col), dist_snap = snap_minimum(dist_2d, transform)
    return (x, y), (row, col), dist_snap, dist_2d

def snap_uparea(gauge_uparea, uparea_2d, drains_2d, transform):
    """Return row, col and x, y coordinates of best fit between the upstream area 
    of drains and the upstream area of a gauge within a given distance.
    The distance is measured using the haversine method.
    Drains are defined based on minimum upstream area threshold 
    """
    upa_reldiff = (uparea_2d - gauge_uparea) / float(gauge_uparea)
    upa_reldiff = np.where(drains_2d, upa_reldiff, np.inf)
    # find closest match
    (x, y), (row, col), _ = snap_minimum(np.abs(upa_reldiff), transform)
    # get distance and uparea of nearest
    upa_snap = uparea_2d[row, col]
    return (x, y), (row, col), upa_snap, upa_reldiff

def snap_combined(gauge_xy, gauge_uparea, uparea_2d, transform, 
                  dist_weight=1., upa_weigth=1., upa_relerr=0.5, max_dist=5e3, 
                  local_upa_perc=99):
    snap_dict = {}
    success = True

    # find drains based on uparea
    uparea_min, uparea_max = (1-upa_relerr)*gauge_uparea, (1+upa_relerr)*gauge_uparea
    local_uparea_min, local_uparea_max = np.percentile(uparea_2d, [local_upa_perc, 100])
    drains_2d = np.logical_and(uparea_2d >= uparea_min, uparea_2d < uparea_max)
    if not np.any(drains_2d): 
        success = False
        drains_2d = np.logical_and(uparea_2d >= local_uparea_min, uparea_2d < local_uparea_max)

    # calculate distance for drains
    dist_2d = calc_haversine_dist(gauge_xy, drains_2d, transform=transform)
    
    # snap to nearest drain
    dist_xy, dist_rc, dist_snap, dist_2d = snap_nearest_haversine(gauge_xy, drains_2d, transform)
    upa_snap = uparea_2d[dist_rc[0], dist_rc[1]]
    dist_relerr_2d = dist_2d / float(max_dist)
    snap_dict['dist'] = dict(lat=dist_xy[1], lon=dist_xy[0], row=dist_rc[0], col=dist_rc[1], uparea=upa_snap, dist=dist_snap)

    # snap to best uparea match
    uparea_xy, upa_rc, upa_snap, upa_reldiff_2d = snap_uparea(gauge_uparea, uparea_2d, drains_2d, transform)        
    dist_snap = dist_2d[upa_rc[0], upa_rc[1]]
    upa_relerr_2d = np.abs(upa_reldiff_2d / float(upa_relerr))
    snap_dict['uparea'] = dict(lat=uparea_xy[1], lon=uparea_xy[0], row=upa_rc[0], col=upa_rc[1], uparea=upa_snap, dist=dist_snap)

    # weighted score
    combi_relerr_2d = (upa_relerr_2d * upa_weigth + dist_relerr_2d * dist_weight) / float(upa_weigth + dist_weight)
    combi_xy, (row, col), _ = snap_minimum(combi_relerr_2d, transform)
    snap_dict['combi'] = dict(lat=combi_xy[1], lon=combi_xy[0], row=row, col=col, uparea=uparea_2d[row, col], dist=dist_2d[row, col], 
                              upa_local_max=local_uparea_max)

    # update dictionaries with rel errors.
    for fname in snap_dict:
        row, col = snap_dict[fname]['row'], snap_dict[fname]['col']
        snap_dict[fname].update(dist_relerr=dist_relerr_2d[row, col],
                                upa_relerr=upa_relerr_2d[row, col],
                                combi_relerr=combi_relerr_2d[row, col])

    return snap_dict, uparea_2d, success

def snap_nearest_drain_ldd():
    # TODO
    return

def snap_gauge(uparea_fn, gauge_xy, gauge_uparea, max_dist=5e3, upa_relerr=0.5, dist_weight=1., upa_weigth=1.,
              local_upa_perc=99):
   
    # read window from data
    uparea_2d, transform = read_gtiff_buf(uparea_fn, gauge_xy, buf=max_dist, layer=0)

    if not np.all(uparea_2d.mask): # no valid data
        snap_dict, uparea_2d, success = snap_combined(gauge_xy, gauge_uparea, uparea_2d, transform, 
            dist_weight=dist_weight, upa_weigth=upa_weigth, upa_relerr=upa_relerr, max_dist=max_dist, 
            local_upa_perc=local_upa_perc)
    else:
        snap_dict, success = {}, 0

    return snap_dict, uparea_2d, transform, success
    
def snap_stations(grdc_meta, uparea_fn, max_dist=5e3, upa_relerr=0.25, dist_weight=1., upa_weigth=1., 
                  fig_dir=None, plot_all=False, uparea_col='AREA', local_upa_perc=99.5):
    import progressbar
    # snap gauge by gauge
    dictionary = {}
    kwargs=dict(max_dist=max_dist, upa_relerr=upa_relerr, dist_weight=dist_weight, upa_weigth=upa_weigth, local_upa_perc=local_upa_perc)
    bar = progressbar.ProgressBar()
    for gauge_id in bar(grdc_meta.index):
        gauge = grdc_meta.loc[gauge_id, :]
        gauge_xy = gauge.geometry.coords[:][0]
        gauge_uparea = gauge[uparea_col]
        try:
            snap_dict, uparea_2d, transform, success = snap_gauge(uparea_fn, gauge_xy, gauge_uparea, **kwargs)
            dictionary[gauge_id] = snap_dict
        except ValueError as e:
            print('skipping gauge {} with error: "{}"'.format(gauge_id, str(e)))
            continue
        
        if (fig_dir is not None) and (plot_all or not success):
            fn_fig = os.path.join(fig_dir, '{}{}.png'.format(str(gauge_id), '_failed' if not success else ''))
            if not os.path.isfile(fn_fig):
                try:
                    fig, _ = plot_snap(dictionary[gauge_id], gauge_id, gauge_xy, gauge_uparea, uparea_2d, transform, **kwargs)
                    plt.savefig(fn_fig, dpi=225, bbox_inches='tight',)
                    plt.close(fig)
                except ValueError as e:
                    print(gauge_id)
                    print(e)

    # reform dict for to convert to multi-index df
    reform = {(outerKey, innerKey): values for outerKey, innerDict in dictionary.items()
              for innerKey, values in innerDict.items()}
    snap_df = pd.DataFrame.from_dict(reform, ).T
    snap_df.index.names = ('id', 'fit')
    # save only combined snap
    snap_df = snap_df.drop(['row', 'col'], axis=1).xs('combi', level='fit')
    return snap_df

def plot_snap(snap_dict, gauge_id, gauge_xy, gauge_uparea, uparea_2d, transform, 
              max_dist, upa_relerr, **kwargs):
    import seaborn as sns
    from gistools import basemap, plot_gtiff
    import matplotlib.colors as mc
    c = sns.color_palette("Set1", n_colors=len(snap_dict.keys())+1, desat=.5)
    cbar_kwargs = dict(width=0.03, rel_height=0.5, label='uparea [km2]',)
    # basemap
    fig, axes = basemap(figsize=(6, 6), gridlines=True, glob=False)
    ax = axes[0]
    plot_gtiff(data=np.where(uparea_2d<=0, np.nan, uparea_2d), transform=transform, cmap='viridis', #vmin=1, vmax=1e5, 
               ax=ax, cbar=True, cbar_kwargs=cbar_kwargs, norm=mc.LogNorm())
    for i, name in enumerate(snap_dict):
        x, y = snap_dict[name]['lon'], snap_dict[name]['lat'], 
        dist, dupa = snap_dict[name]['dist_relerr']*max_dist, snap_dict[name]['upa_relerr']*upa_relerr
        ax.scatter(x, y, edgecolor='white', color=c[i], s=100,
                  label='{:s} \n ($\Delta$A = {:.1f}%; dist = {:.0f} m)'.format(name, dupa*100, dist))
    ax.scatter(*gauge_xy, edgecolor='white', color=c[-1], s=70, marker='^',
              label='original location \n (uparea {:.0f} km2)'.format(gauge_uparea))
    posn = ax.get_position()
    fig.legend(loc='upper left', bbox_to_anchor=(posn.x0+posn.width, posn.height))
    ax.set_title('ID {:s}; \n max_dist={:.0f} m; upa_relerr:{:.1f}%'.format(str(gauge_id), max_dist, upa_relerr*100))
    return fig, ax