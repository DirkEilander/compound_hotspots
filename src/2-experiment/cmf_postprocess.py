
# coding: utf-8

import os
from os.path import join, dirname, realpath, isfile, basename, isdir
import glob
import sys
import warnings
from copy import deepcopy
import click
import itertools
from shutil import copyfile, rmtree
from joblib import Parallel, delayed, cpu_count
import multiprocessing
import json

# local lib
from nc_extract import nc_extract
warnings.filterwarnings("ignore")

# global vars
scriptdir = dirname(realpath(__file__))

## MAIN
@click.command()
@click.argument('settings-json')
@click.argument('datadirs', nargs=-1)
@click.option('-t', '--tmpdir', default=getattr(os.environ, 'TMPDIR', None), help='TMPDIR to do work')
@click.option('-s', '--scriptdir', default=scriptdir, help='directory with scripts and extract files')
@click.option('-n', '--nprocs', default=1, help='Number of parallel runs')
@click.option('-i', '--items', multiple=True, help='Number of parallel runs')
@click.option('--force-overwrite', is_flag=True, help='overwrite output files is exist (default=False)')
def cli(**kwargs):
    main(**kwargs)

def main(settings_json, datadirs, tmpdir=None, scriptdir=scriptdir, nprocs=1, force_overwrite=False, items=[]):
    """
    extract timeseries for regions and points based output files in folder
    """
    # read settings from json file
    if not os.path.isabs(settings_json):
        settings_json= join(scriptdir, settings_json)
    if not isfile(settings_json):
        msg = "setting json file not found: {:s}".format(settings_json)
        raise EnvironmentError(msg)
    else:
        with open(settings_json, 'r') as f:
            settings = json.load(f)

    if items and len(items) > 0:
        settings = {item: settings[item] for item in items}

    runs = []
    num_workers = cpu_count()/nprocs
    for ddir in datadirs:
        if not isdir(ddir): continue
        name = basename(ddir)
        run_tmpdir = join(tmpdir, name) if tmpdir else None
        runs.append(dict(settings=settings, datadir=ddir, 
                         tmpdir=run_tmpdir, scriptdir=scriptdir, 
                         num_workers=num_workers, force_overwrite=force_overwrite))

    if len(runs) == 1 or nprocs == 1:
        print('run {:d} jobs in serial'.format(len(runs)))
        [cmf_postprocess(**kwargs) for kwargs in runs]
    else:
        print('run {:d} jobs in prallel with {} processes'.format(len(runs), nprocs))
        Parallel(n_jobs=nprocs, backend="multiprocessing", batch_size=1)(delayed(cmf_postprocess)(**kwargs) for kwargs in runs)


def cmf_postprocess(settings, datadir, tmpdir=None, scriptdir=scriptdir, num_workers=8, force_overwrite=False):
    name = basename(datadir)
    tmpdir = datadir if tmpdir is None else tmpdir
    if basename(tmpdir) != basename(datadir):
        tmpdir = join(tmpdir, name) # make sure to work in subfolder of tmpdir
    print('tmpdir: {}'.format(tmpdir))
    p = multiprocessing.current_process()
    worker = p.name if p else ''
    try:
        for item in settings.keys():
            kwargs = deepcopy(settings[item])
            # set I/O
            # NOTE: saving output one folder up from input
            tmpout_fn = join(dirname(tmpdir), "{item}_{name}.nc".format(name=name, item=item))
            out_fn = join(dirname(datadir), basename(tmpout_fn))
            kwargs['out_fn'] = tmpout_fn
            if not os.path.isabs(kwargs['extract_fn']):
                kwargs['extract_fn'] = join(scriptdir, kwargs['extract_fn'])
            # skip if already processed
            if not force_overwrite and isfile(out_fn): 
                # print('result file {} already exist. skip.'.format(out_fn))
                print('{}: skip {} - {} ..'.format(worker, name, item))
                continue
            # find input files
            in_fns = []
            for var in kwargs['var_name']:
                in_fns.extend(glob.glob(join(datadir, '{}*.nc'.format(var))))
            if len(in_fns) == 0: continue
            # copy data to scratch
            if tmpdir != datadir:
                if not isdir(tmpdir): os.makedirs(tmpdir)
                in_fns = copy_files(in_fns, tmpdir)
            kwargs['in_fns'] = in_fns
            kwargs['num_workers'] = num_workers
            # postprocessing
            print('{}: processing {} - {}..'.format(worker, name, item))
            nc_extract(**kwargs)
            # copy results back
            if tmpdir != datadir:
                # print('copy output file {} to {}'.format(basename(tmpout_fn), dirname(out_fn)))
                copyfile(tmpout_fn, out_fn)
                os.unlink(tmpout_fn)
    finally:
        if isdir(tmpdir) and (tmpdir != datadir):
            try:
                for fn in glob.glob(join(tmpdir, '*.nc')):
                    os.unlink(fn)
                rmtree(tmpdir)
            except OSError as e:
                print('error while cleaning up tmp dir {}: {}'.format(tmpdir, str(e)))

def copy_files(fn_list, dst_dir):
    fns = []
    for fn in fn_list:
        dst_fn = join(dst_dir, basename(fn))
        fns.append(dst_fn)
        if isfile(dst_fn): 
            continue #NOTE! if filename exists assume they are the same
        copyfile(fn, dst_fn)
    return fns

if __name__ == "__main__":
    cli() 