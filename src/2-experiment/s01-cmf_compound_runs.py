
from pyqsub import qsub 
from os.path import join, dirname, realpath, isdir, basename, isfile
import glob
from itertools import product 
import numpy as np
import calendar
from cmf_run2 import configread, ConfigParser

def check_restart(outdir, year=1980):
    # check status runs from restart folder
    restart_dir = join(outdir, 'restart')
    if isdir(restart_dir):
        rfns = glob.glob(join(restart_dir, 'restart*.nc'))
        if len(rfns) > 0:
            year = max([int(basename(fn).replace('restart', '')[:4]) for fn in rfns])
    return year

if __name__ == '__main__':
    ini_fn = 'cmf_lisa.ini'
    config = configread(ini_fn, cf=ConfigParser(inline_comment_prefixes=('#')))
    outdir = config.get('I/O', 'out_dir')
    # tmpdir = config.get('I/O', 'tmp_dir')
    root = dirname(outdir)    
    jobdir = join(root, 'jobs')
    drm = 'SLURM'
    
    # qsub info
    submit=True
    ijob_start = 34
    njobs_per_node = 2
    parallel = True
    hrsyr = 4  # hrs per model year
    postprocessing_time = 5 # 3hr for postprocessing 
    cwd = dirname(realpath(__file__))
    modules = ['pre2019', 'eb', 'netCDF-Fortran/4.4.4-intel-2016b']
    pbs_args = ['n 16', 'N 1']
    precommand_form = 'source ~/.bashrc \nexport OMP_NUM_THREADS={:d} \nOUTDIR="{}"'

    # runs
    data_sources = ['anu', 'cnrs', 'ecmwf', 'jrc', 'nerc'][:1] #, 'univu', 'univk']
   
    sealvl_bound = ['cli',    'act',    'act',   'tide', 'msl'][1:-1]
    runoff_bound = ['act',    'act',    'cli',   'act',  'act'][1:-1]
    experiment   = ['runoff', 'cmpnd',  'surge', 'tide', 'msl'][1:-1]

    # PDSTMTH
    # variables per run
    spinup, start_year, end_year = 2, 1980, 2014 # make sure this is the same as in the ini file!
    dsbnd_fns = {
        'act': join(root, r'GTSM', r'global_model_waterlevel_{year}_select_egm.nc'),
        'cli': join(root, r'GTSM', r'global_model_waterlevel_clim_{year}_select_egm.nc'),
        'msl': join(root, r'GTSM', r'global_model_waterlevel_{year}_select_egm.nc'),
        'tide': join(root, r'GTSM', r'all_fes_data_{year}_select_egm.nc'),
        }
    runoff_fns = {
        'act': join(root, r'E2O', r'e2o_{data}_wrr2_glob15_day_Runoff_{year}.nc'),
        'cli': join(root, r'E2O', r'e2o_{data}_wrr2_glob15_day_Runoff_{year}clim.nc'),
    }
    dsbnd_args = {  
        'act': ['LBOUNDSL=true', 'DT=1800'],
        'cli': ['LBOUNDSL=true', 'DT=1800'],
        'tide': ['LBOUNDSL=true', 'DT=1800', 'CBNDCDFVAR=tide'],
        'msl': [],
    }


    # main job
    command_temp = 'python cmf_run2.py {ini_fn} {name} -o "$OUTDIR" -t "$TMPDIR" --runoff-fn "{runoff_fn}" --dsbnd-fn "{dsbnd_fn}" {nam_args} --restart --local-io'
    # postprocessing 
    settings_json = join(cwd, 'postprocessing_compound.json')
    postcommand_temp = 'python cmf_postprocess.py  {settings_json} {datadirs} -t "$TMPDIR" -n 1 --force-overwrite'
    
    runs, irun = [], 0
    for nam_args in [['PDSTMTH=12500'], ['PDSTMTH=7500']]:
        postfix = '_' + '_'.join([a[1:].lower().replace('=','') for a in nam_args]) if nam_args else ''
        name_temp = '{data}_mswep_{experiment}_v362_1980-2014' + postfix

        for roff, dsbnd, exp in zip(runoff_bound, sealvl_bound, experiment):
            for data in data_sources:
                name = name_temp.format(data=data, experiment=exp)
                run_outdir = join(outdir, name)
                year0 = check_restart(run_outdir, year=start_year)
                nyears = end_year + 1 - start_year + spinup if year0 == start_year else end_year + 1 - year0 
                if nyears >= 0:
                    runs.append((name, run_outdir, data, roff, dsbnd, exp, nyears, nam_args))
                    print('{:02d}: {:s} - {:d} years'.format(irun+1, name, nyears))
                    irun += 1

    if len(runs) > 0:
        seq = np.arange(irun) #np.argsort(np.array([run[-1] for run in runs]))
        commands, postcommands, times, nyears, ijob = [], [], [], [], ijob_start
        for i in seq:
            name, run_outdir, data, roff, dsbnd, exp, nyr, nam_args = runs[i]
            runoff_fn0 = runoff_fns[roff].format(data=data, year='{year}')
            dsbnd_fn = dsbnd_fns[dsbnd]
            nam_args_str = ' '.join(['--nam {}'.format(arg) for arg in nam_args + dsbnd_args[dsbnd]])
            kwargs = dict(ini_fn=ini_fn, name=name, runoff_fn=runoff_fn0, dsbnd_fn=dsbnd_fn, nam_args=nam_args_str)
            commands.append(kwargs)
            postcommands.append(run_outdir)
            times.append(nyr*hrsyr) 
            nyears.append(nyr)

            if (len(commands)==njobs_per_node) or (i+1 == len(seq)):
                walltime = max(times) if parallel else sum(times)
                commands = [command_temp.format(**kwargs) for nyr, kwargs in zip(nyears, commands) if nyr > 0]
                precommand = precommand_form.format( int(16/len(commands)) if len(commands)>1 else 16, outdir)
                command = ' &\n'.join(commands) if len(commands)>1 else ' \n'.join(commands)
                postcommand = postcommand_temp.format(settings_json=settings_json, datadirs=' '.join(postcommands))
                postcommand = 'wait\n'+postcommand if len(commands)>1 else postcommand
                qsub('cmpnd{:03d}'.format(ijob), command, drm=drm, workdir=cwd, jobdir=jobdir, 
                    lwalltime=min(walltime+postprocessing_time,120), args=pbs_args, modules=modules, 
                    precommand=precommand, postcommand=postcommand,
                    submit=submit, force_overwrite=True, verbose=False)
                commands, postcommands, times, nyears = [], [], [], []
                ijob += 1
