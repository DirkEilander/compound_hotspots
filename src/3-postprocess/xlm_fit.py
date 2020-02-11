"""NOTE: currenlty only gumbel fits are supported!"""
from lmoments3 import distr
import xarray as xr
import pandas as pd
import numpy as np 
from scipy import stats

def weibull(peaks, nyears=None):
    """weibull plot position"""
    peaks = peaks[np.isfinite(peaks)]
    peaks_rank = stats.rankdata(peaks, 'ordinal')
    P = peaks_rank/(peaks.size+1)
    freq = 1. if nyears is None else peaks.size / nyears
    rp = 1/(1-P) * 1/freq
    return rp

def _lm_fit(peaks, rp, fdist=distr.gum, nmin=3, nyears=None):
    _peaks = peaks[np.isfinite(peaks)]
    success = False
    if _peaks.size >= nmin and _peaks.std() > 0: # min 30 values for robust estimate
        try:
            pars = fdist.lmom_fit(_peaks)
            rv = fdist(**pars) 
            freq = 1. if nyears is None else _peaks.size / nyears
            ev = rv.isf(freq / rp) # 1/(1-P) = freq / rp
            success = True
        except ValueError:
            pass
    if not success:
        _peaks = np.nan if _peaks.size == 0 else np.mean(_peaks)
        ev = np.ones_like(rp)*_peaks
        pars = dict()
    pars_out = np.array([pars.get(k,np.nan) for k in ['shape', 'loc', 'scale']]) 
    return ev, pars_out

def xlm_fit(da_peaks, fdist=distr.gum, nmin=3, nyears=None, rp=np.array([2,5,10,25]), dim='time'):
    rp = np.atleast_1d(rp)
    kwargs = dict(rp=rp, nmin=nmin, nyears=nyears, fdist=fdist)
    # apply_ufunc parameters
    da_rp, da_par = xr.apply_ufunc(
        _lm_fit, 
        da_peaks,         
        input_core_dims=[[dim]], 
        output_core_dims=[['T'], ['par']],
        output_dtypes=[float, float],    
        output_sizes={'T': len(rp), 'par': 3}, # on output, <dim> is reduced to length q.size 
        vectorize=True,
        dask='allowed',
        kwargs=kwargs,
    )
    da_par.name = 'params'
    da_out = xr.merge([da_rp, da_par])
    da_out.coords['T'] = xr.Variable('T', rp)
    da_out.coords['par'] = xr.Variable('par', ['shape', 'loc', 'scale'])
    return da_out

def _lm_fit_ci(peaks, rp, fdist=distr.gum, nyears=None, n_samples=1000, alphas=np.array([0.1, 0.9])):   
    peaks = peaks[np.isfinite(peaks)]
    alphas = np.asarray(alphas)
    if peaks.size > 0 and peaks.std() > 0:
        freq = 1. if nyears is None else peaks.size / nyears
        par0 =  pd.DataFrame.from_records([fdist.lmom_fit(peaks)]).loc[0]
        def _isf(pars):
            return fdist(*pars).isf(freq/rp) # 1(1-P) = freq / rp
        def _bootstrap_indexes(data, n_samples):
            return np.random.randint(data.shape[0], size=(n_samples, data.shape[0]))
        def _lm_fit(peaks):
            if peaks.std() > 0:
                return fdist.lmom_fit(peaks)
            else:
                return par0*np.nan
        # estimate parameters using bootstrap sample
        bootindexes = _bootstrap_indexes(peaks, n_samples=n_samples)
        pars = pd.DataFrame.from_records(np.apply_along_axis(_lm_fit, arr=peaks[bootindexes], axis=-1))
        bias = par0 - pars.mean()
        pars += bias
        # get ci
        stat = np.apply_along_axis(_isf, arr=pars.values, axis=-1)
        stat.sort(axis=0)
        nvals = np.round((n_samples-1)*alphas).astype('int')
        ci = stat[nvals, ...]
    else:
        ci = np.ones((alphas.size, rp.size))*np.nan
    return ci

def xlm_fit_ci(da_peaks, fdist=distr.gum, nyears=None, rp=np.array([2,5,10,25]), 
               alphas=np.array([0.1, 0.9]), n_samples=10000, dim='time'):
    # confidence interval parameters
    kwargs = dict(
        alphas=np.atleast_1d(alphas), 
        n_samples=n_samples, 
        rp=np.atleast_1d(rp), 
        nyears=nyears,
        fdist=fdist
        )
    ufunc = lambda x: np.apply_along_axis(_lm_fit_ci, axis=-1, arr=x, **kwargs)
    da_out = xr.apply_ufunc(
        ufunc,
        da_peaks,
        input_core_dims=[[dim]], 
        output_core_dims=[['alpha', 'T']],
        dask='parallelized', 
        output_dtypes=[float],       
        output_sizes={'alpha': kwargs['alphas'].size, 'T': kwargs['rp'].size},
    )
    da_out.coords['alpha'] = xr.Variable('alpha', alphas)
    da_out.coords['T'] = xr.Variable('T', rp)
    return da_out.squeeze()