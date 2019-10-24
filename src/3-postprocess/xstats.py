import xarray as xr
import numpy as np 
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from lmoments3 import distr


def weibull(peaks, nyears=None):
    peaks = peaks[np.isfinite(peaks)]
    peaks_rank = stats.rankdata(-1 * peaks, method='ordinal')
    if nyears is not None:
        lambd = peaks.size / nyears 
    else:
        lambd = 1.
    rp = float((peaks.size/lambd)+1) / peaks_rank
    return rp, peaks

def _lm_fit(peaks, rp, fdist=distr.gpa, nyears=None):
    peaks = peaks[np.isfinite(peaks)]
    assert peaks.size > 0
    pars = fdist.lmom_fit(peaks)
    pars_out = np.array([v for v in pars.values()])
    rv = fdist(**pars) 
    lambd = 1.
    if nyears is not None:
        lambd = peaks.size / nyears 
    q = 1 - 1. / (rp*lambd)
    return rv.ppf(q), pars_out

def _lm_fit_ci(peaks, rp, fdist=distr.gpa, nyears=None, n_samples=1000, alphas=np.array([0.1, 0.9]) ):
    def bootstrap_indexes(data, n_samples=1000):
        return np.random.randint(data.shape[0], size=(n_samples, data.shape[0]))
    peaks = peaks[np.isfinite(peaks)]
    lambd = 1.
    if nyears is not None:
        lambd = peaks.size / nyears 
    q = 1 - 1. / (rp*lambd)
    bootindexes = bootstrap_indexes(peaks, n_samples=n_samples)
    pars = pd.DataFrame.from_records(np.apply_along_axis(fdist.lmom_fit, arr=peaks[bootindexes], axis=-1))
    par0 =  pd.DataFrame.from_records([fdist.lmom_fit(peaks)]).loc[0]
    bias = par0 - pars.mean()
    pars += bias
    fdist_ppf = lambda pars: fdist(*pars).ppf(q)
    stat = np.apply_along_axis(fdist_ppf, arr=pars.values, axis=-1)
    stat_sorted = np.apply_along_axis(np.sort, arr=stat, axis=0)
    nvals = np.nan_to_num(np.round((n_samples-1)*alphas)).astype('int')
    ci = stat_sorted[nvals, ...]    
    return ci

def _interp_ev(peaks, vals, nyears):
    peaks = peaks[np.isfinite(peaks)]
    peaks.sort()
    rps, peaks = weibull(peaks, nyears=nyears)
    logrps = np.log10(rps)
    kwargs = dict(
        kind='linear', bounds_error=False, assume_sorted=True,
        fill_value=(logrps.min(), logrps.max())
    )
    rp_out = 10**interp1d(peaks, np.log10(rps), **kwargs)(vals)
    return rp_out

def _interp_rps(peaks, rp, nyears):
    peaks = peaks[np.isfinite(peaks)]
    peaks.sort()
    rps_in, peaks = weibull(peaks, nyears=nyears)
    kwargs = dict(
        kind='linear', bounds_error=False, assume_sorted=True,
        fill_value='extrapolate'
    )
    peaks_out = interp1d(np.log10(rps_in), peaks, **kwargs)(np.log10(rp))
    return peaks_out

def _interp_rps_ci(peaks, rp, nyears=None, n_samples=1000, alphas=np.array([0.1, 0.9]) ):
    def bootstrap_indexes(data, n_samples=1000):
        return np.random.randint(data.shape[0], size=(n_samples, data.shape[0]))
    peaks = peaks[np.isfinite(peaks)]
    bootindexes = bootstrap_indexes(peaks, n_samples=n_samples)
    ev = np.apply_along_axis(_interp_rps, arr=peaks[bootindexes], axis=-1, rp=rp, nyears=nyears)
    ev0 =  _interp_rps(peaks, rp=rp, nyears=nyears)
    bias = ev0 - np.nanmean(ev, axis=0)
    ev += bias
    ev_sorted = np.apply_along_axis(np.sort, arr=ev, axis=0)
    nvals = np.nan_to_num(np.round((n_samples-1)*alphas)).astype('int')
    ci = ev_sorted[nvals, ...]   
    return ci

def xlm_fit(da_peaks, fdist=distr.gpa, nyears=None, rp=np.array([2,5,10,25]), dim='time'):
    rp = np.atleast_1d(rp)
    p2r_kwargs = dict(rp=rp, nyears=nyears, fdist=fdist)
    # apply_ufunc parameters
    kwargs = dict(               
        input_core_dims=[[dim]], 
        output_core_dims=[['rp'], ['par']],
        dask='allowed', 
        output_dtypes=[float, float],    
        output_sizes={'rp': len(rp), 'par': 3}, # on output, <dim> is reduced to length q.size 
        vectorize=True
    )
    da_rp, da_par = xr.apply_ufunc(_lm_fit, da_peaks, kwargs=p2r_kwargs, **kwargs)
    da_par.name = 'gev'
    da_out = xr.merge([da_rp, da_par])
    da_out['rp'] = xr.Variable('rp', rp)
    da_out['par'] = xr.Variable('par', ['shape', 'loc', 'scale'])
    return da_out

def xlm_fit_ci(da_peaks, fdist=distr.gpa, nyears=None, rp=np.array([2,5,10,25]), 
               alphas=np.array([0.1, 0.9]), n_samples=10000, dim='time'):
    # confidence interval parameters
    rp = np.atleast_1d(rp)
    alphas = np.atleast_1d(alphas)
    ci_kwargs = dict(
        alphas=alphas, 
        n_samples=n_samples, 
        rp=rp, 
        nyears=nyears,
        fdist=fdist
        )
    # apply ufunc parameters
    kwargs = dict(               
        input_core_dims=[[dim]], 
        output_core_dims=[['alpha', 'rp']],
        dask='allowed', 
        output_dtypes=[float],       
        output_sizes={'alpha': alphas.size, 'rp': rp.size},
        vectorize=True
    )
    # apply ci_nd over dim
    da_out = xr.apply_ufunc(_lm_fit_ci, da_peaks, kwargs=ci_kwargs, **kwargs)
    da_out['alpha'] = xr.Variable('alpha', alphas)
    da_out['rp'] = xr.Variable('rp', rp)

    return da_out.squeeze() 

def xinterp_rps(da_peaks, nyears=None, rp=np.array([2,5,10,25]), dim='time'):
    # confidence interval parameters
    rp = np.atleast_1d(rp)
    ci_kwargs = dict(
        rp=rp, 
        nyears=nyears,
        )
    # apply ufunc parameters
    kwargs = dict(               
        input_core_dims=[[dim]], 
        output_core_dims=[['rp']],
        dask='allowed', 
        output_dtypes=[float],       
        output_sizes={'rp': rp.size},
        vectorize=True
    )
    # apply ci_nd over dim
    da_out = xr.apply_ufunc(_interp_rps, da_peaks, kwargs=ci_kwargs, **kwargs)
    da_out['rp'] = xr.Variable('rp', rp)
    return da_out.squeeze() 

def xinterp_ev(da_peaks, da_vals, nyears=None, dim='time'):
    # confidence interval parameters
    ci_kwargs = dict(
        nyears=nyears,
        )
    # apply ufunc parameters
    kwargs = dict(               
        input_core_dims=[[dim], [dim]], 
        output_core_dims=[[dim]],
        dask='parallelized', 
        output_dtypes=[float],       
        output_sizes={dim: da_peaks[dim].size},
        vectorize=True
    )
    # apply ci_nd over dim
    chunks = {dim: -1}
    da_out = xr.apply_ufunc(_interp_ev, da_peaks.chunk(chunks), da_vals.chunk(chunks), kwargs=ci_kwargs, **kwargs)
    return da_out.squeeze()

def xinterp_rps_ci(da_peaks, nyears=None, rp=np.array([2,5,10,25]), 
               alphas=np.array([0.1, 0.9]), n_samples=10000, dim='time'):
    # confidence interval parameters
    rp = np.atleast_1d(rp)
    alphas = np.atleast_1d(alphas)
    ci_kwargs = dict(
        alphas=alphas, 
        n_samples=n_samples, 
        rp=rp, 
        nyears=nyears,
        )
    # apply ufunc parameters
    kwargs = dict(               
        input_core_dims=[[dim]], 
        output_core_dims=[['alpha', 'rp']],
        dask='allowed', 
        output_dtypes=[float],       
        output_sizes={'alpha': alphas.size, 'rp': rp.size},
        vectorize=True
    )
    # apply ci_nd over dim
    da_out = xr.apply_ufunc(_interp_rps_ci, da_peaks, kwargs=ci_kwargs, **kwargs)
    da_out['alpha'] = xr.Variable('alpha', alphas)
    da_out['rp'] = xr.Variable('rp', rp)
    return da_out.squeeze() 

def xrankdata(da, dim='time', method='average'):
    """Assign ranks to data, dealing with ties appropriately.

    Ranks begin at 1. The method argument controls how ranks are assigned to equal values. 
    See scipy.stats.rankdata for more info on the method.
    
    Parameters
    ----------
    da: xarray DataArray
        Input data array
    dim : str, optional
        name of the core dimension (the default is 'time')
    
    Returns
    -------
    rank : xarray DataArray

    """
    def _rankdata(*args, **kwargs):
        return np.apply_along_axis(stats.rankdata, -1, *args, **kwargs)
    # apply_ufunc parameters
    output_dtype = [int] if method=='ordinal' else [float]
    kwargs = dict(               
        input_core_dims=[[dim]], 
        output_core_dims=[[dim]],
        dask='parallelized', 
        output_dtypes=output_dtype,    
        output_sizes={dim: da[dim].size} 
    )
    rank = xr.apply_ufunc(_rankdata, da, kwargs=dict(method=method), **kwargs)
    return rank

def xnanpercentile(da, q, dim='time', interpolation='linear'):
    """Returns the qth percentile of the data along the specified core dimension,
    while ignoring nan values.
    
    Parameters
    ----------
    da: xarray DataArray
        Input data array
    q : float in range of [0,100] (or sequence of floats)
        Percentile to compute, which must be between 0 and 100 inclusive.
    dim : str, optional
        name of the core dimension (the default is 'time')
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired quantile lies between two data points
    
    Returns
    -------
    percentile : xarray DataArray
        The core dimension is reduce to quantiles and returned at the last dimension.
    """
    def _nanpercentile(*args, **kwargs):
        """nanpercentile, but with q moved to the last axis"""
        return np.moveaxis(np.nanpercentile(*args, **kwargs), 0, -1)
    # nanpercentile parameters
    q = np.atleast_1d(q)
    q_kwargs = dict(q=q, axis=-1, interpolation=interpolation)
    # apply_ufunc parameters
    kwargs = dict(               
        input_core_dims=[[dim]], 
        output_core_dims=[['percentile']],
        dask='parallelized', 
        output_dtypes=[float],    
        output_sizes={'percentile': q.size} # on output, <dim> is reduced to length q.size 
    )
    if 'percentile' in da.coords:
        da = da.drop('percentile')
    percentile = xr.apply_ufunc(_nanpercentile, da.chunk({dim: -1}), kwargs=q_kwargs, **kwargs)
    percentile['percentile'] = xr.Variable('percentile', q)
    return percentile.squeeze() # if q.size=1 remove dim