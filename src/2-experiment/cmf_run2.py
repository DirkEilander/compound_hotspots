#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Dirk Eilander (contact: dirk.eilancer@vu.nl)
# Created: June 2017
#
# SVN auto-props
# $ID: $
# $Date: 2018-03-08 12:27:11 +0100 (Thu, 08 Mar 2018) $
# $Author: der240 $
# $Revision: 245 $
# $HeadURL: https://svn.vu.nl/repos/compound_floods/trunk/CaMaFlood/e2o_runs/CaMaFlood_experiment_E2O.py $

import os
from os.path import join, isdir, basename, dirname, realpath, isfile, abspath, relpath, isabs, islink
import shutil
from datetime import datetime
import re, codecs
from configparser import ConfigParser
from collections import OrderedDict
from shutil import copyfile, move, rmtree
from datetime import datetime, timedelta
import click
import glob
import subprocess
import sys
import time
import math
from copy import deepcopy
from threading import Timer
import numpy as np
import multiprocessing

## TEMPLATES
nam_v362 = u"""\
&NRUNVER
IRESTART=2                                  ! 1=> restart;  2=>spinup
CRESTDIR="../restart"                       ! restart directory
CRESTSTO="../restart/restart{year}0101.nc"  ! restart file relative directory
LSTOONLY=.FALSE.                            ! true for restart only from storage
LRESTCDF=.TRUE.                             ! true for netCDF restart file
RESTFREQ=0                                  ! 0: yearly restart file, 1: daily restart file (for monthly restart file use 1. and throw out all but last file)
/
&NSIMTIME
ISYEAR={year}                               ! start year
ISMON=1                                     ! month
ISDAY=1                                     ! day        (assumed at 00UTC)
IEYEAR={next_year}                          ! end year
IEMON=1                                     ! end month
IEDAY=1                                     ! end day    (assumed at 00UTC)
/
&NMAP
LMAPCDF=.FALSE.                             ! true for netCDF map input
CDIMINFO="../map/diminfo_15min_NtoS60.txt"  ! dimention info
CNEXTXY="../map/nextxy.bin"                 ! downstream xy (river network map)
CGRAREA="../map/grarea.bin"                 ! unit-catchment area [m2]
CELEVTN="../map/elevtn.bin"                 ! base elevation      [m]
CNXTDST="../map/nxtdst.bin"                 ! downstream distance [m]
CRIVWTH="../map/rivwth_gwdlr.bin"           ! channel width       [m]
CRIVLEN="../map/rivlen.bin"                 ! channel length      [m]
CRIVHGT="../map/rivhgt.bin"                 ! channel depth       [m]
CFLDHGT="../map/fldhgt.bin"                 ! floodplain elevation profile [m]
CPTHOUT="../map/fldpth.txt"                 ! bifurcation channel list
CRIVCLINC="NONE"                            ! * netCDF river maps
CRIVPARNC="NONE"                            ! * netCDF river width & depth
/
&NINPUT
LINTERP=.TRUE.                              ! true for runoff interpolation using input matrix
LINPCDF=.TRUE.                              ! true for netCDF input
CINPMAT="../map/inpmat-15min_NtoS60.bin"    ! input matrix file name
CRUNOFFDIR=""                               ! runoff input directory
CRUNOFFPRE=""                               ! runoff input prefix
CRUNOFFSUF=""                               ! runoff input suffix
CRUNOFFCDF="../input/runoff{year}.nc"       ! * netCDF input runoff file name
CROFCDFVAR="Runoff"                         ! * netCDF input runoff variable name
SYEARIN={year}                              ! * for netCDF input start date (start of the initial time step)
SMONIN=1
SDAYIN=1
LINTERPCDF=.FALSE.                          ! * true for netCDF input matrix
{ds}
/
&NOUTPUT
LOUTCDF=.TRUE.                              ! true for netCDF output
COUTDIR="./"                              ! output directory ("NONE" for no output)
CRIVOUTDIR="NONE"                              ! river discharge        [m3/s]
CRIVSTODIR="NONE"                           ! river storage          [m3]
CRIVVELDIR="NONE"                           ! river flow velocity    [m/s]
CRIVDPHDIR="./"                              ! river water depth      [m]
CFLDOUTDIR="NONE"                           ! floodplain discharge   [m3/s]
CFLDSTODIR="NONE"                              ! floodplain storage     [m3]
CFLDDPHDIR="./"                           ! floodplain water depth [m]
CFLDFRCDIR="NONE"                              ! flooded area fraction  [m2/m2]
CFLDAREDIR="./"                              ! flooded area           [m2]
CSFCELVDIR="./"                           ! water surface elevation           [m]
COUTFLWDIR="./"                              ! total discharge (rivout+fldout)   [m3/s]
CSTORGEDIR="NONE"                              ! total storage   (rivsto+fldsto)   [m3]
CPTHOUTDIR="NONE"                              ! net bifurcation flow (grid-based) [m3/s]
CPTHFLWDIR="NONE"                              ! bifurcation flow (channel-based)  [m3/s]
COUTINSDIR="NONE"                           ! instantaneous discharge (no river routing, summation of upstream runoff)
LOUTVEC=.FALSE.                             ! for 1-D land-only output (small data size, post processing required)
/
&NCONF                                      ! * NX, NY, NFLP, NXIN, NYIN, INPN, WEST, EAST, NORTH, SOUTH set by diminfo.txt
DT=86400                                    ! time step [sec]
DTIN=86400                                  ! input runoff time step [sec]
DROFUNIT=1.D-3                              ! runoff unit conversion (1.D-3 when input [mm] is converted to [m3/m2]
LADPSTP=.TRUE.                              ! true for adaptive time step
LFLDOUT=.TRUE.                              ! true to activate floodplain discharge
LPTHOUT=.FALSE.                             ! true to activate bifurcation channel flow
LFLD=.TRUE.                                 ! true to activate floodplain inundation
LKINE=.FALSE.                               ! true for kinematic river routing
LMAPEND=.FALSE.                             ! true to convert map data endian
LINPEND=.FALSE.                             ! true to convert input data endian
LLEAPYR=.TRUE.                              ! true for leap year calculatuon, false: always 365days/year
/
&NPARAM
PMANRIV=0.03D0                              ! manning coefficient river
PMANFLD=0.10D0                              ! manning coefficient floodplain
PGRV=9.8D0                                  ! accerelation due to gravity
PDSTMTH=10000.D0                            ! downstream distance at river mouth [m]
PCADP=0.7                                   ! satety coefficient for CFL condition
PMINSLP=1.D-5                               ! * minimum slope (for kinematic wave)
/
"""

nam_v39x = u"""\
&NRUNVER
IRESTART=2                                  ! 1=> restart;  2=>spinup
CRESTDIR="../restart"                       ! restart directory
CRESTSTO="../restart/restart{year}0101.nc"  ! restart file relative directory
LSTOONLY=.FALSE.                            ! true for restart only from storage
LRESTCDF=.TRUE.                             ! true for netCDF restart file
RESTFREQ=0                                  ! 0: yearly restart file, 1: daily restart file (for monthly restart file use 1. and throw out all but last file)
/
&NSIMTIME
ISYEAR={year}                               ! start year
ISMON=1                                     ! month
ISDAY=1                                     ! day        (assumed at 00UTC)
IEYEAR={next_year}                          ! end year
IEMON=1                                     ! end month
IEDAY=1                                     ! end day    (assumed at 00UTC)
/
&NMAP
LMAPCDF=.FALSE.                             ! true for netCDF map input
CDIMINFO="../map/diminfo_15min_NtoS60.txt"  ! dimention info
CNEXTXY="../map/nextxy.bin"                 ! downstream xy (river network map)
CGRAREA="../map/ctmare.bin"                 ! unit-catchment area [m2]
CELEVTN="../map/elevtn.bin"                 ! base elevation      [m]
CNXTDST="../map/nxtdst.bin"                 ! downstream distance [m]
CRIVWTH="../map/rivwth_gwdlr.bin"           ! channel width       [m]
CRIVLEN="../map/rivlen.bin"                 ! channel length      [m]
CRIVHGT="../map/rivhgt.bin"                 ! channel depth       [m]
CRIVMAN=../map/rivman.bin
CFLDHGT="../map/fldhgt.bin"                 ! floodplain elevation profile [m]
CPTHOUT="../map/bifprm.txt"                 ! bifurcation channel list
CRIVCLINC="NONE"                            ! * netCDF river maps
CRIVPARNC="NONE"                            ! * netCDF river width & depth
/
&NINPUT
LINTERP=.TRUE.                              ! true for runoff interpolation using input matrix
LINPCDF=.TRUE.                              ! true for netCDF input
CINPMAT="../map/inpmat-15min_NtoS60.bin"    ! input matrix file name
CRUNOFFDIR=""                               ! runoff input directory
CRUNOFFPRE=""                               ! runoff input prefix
CRUNOFFSUF=""                               ! runoff input suffix
CRUNOFFCDF="../input/runoff{year}.nc"       ! * netCDF input runoff file name
CROFCDFVAR="Runoff"                         ! * netCDF input runoff variable name
SYEARIN={year}                              ! * for netCDF input start date (start of the initial time step)
SMONIN=1
SDAYIN=1
LINTERPCDF=.FALSE.                          ! * true for netCDF input matrix
LMEANSL=.FALSE.                             ! true for mean sea level
CMEANSL=""                                  ! mean sea level
LSEALEV=.FALSE.                             ! true for sea level boundary
LSEALEVCDF=.TRUE.                           ! * true for netCDF sea level boundary
CSEALEVDIR=""                               ! sea level directory
CSEALEVPRE=""                               ! sea level prefix
CSEALEVSUF=""                               ! sea level suffix
CSEALEVCDF="../dsbnd/waterlevel{year}.nc"   ! * netCDF sea level file name
CSLCDFVAR="waterlevel"                      ! * netCDF sea level variable name
DTSL=1800                                   ! sea level time step [sec]
SYEARSL={year}                              ! start year in netCDF sea level
SMONSL=1 			                        ! start month in netCDF sea level
SDAYSL=1    			                    ! start day in netCDF sea level
CSEALEVREF=""                               ! reference table for sea level
/
&NOUTPUT
LOUTCDF=.TRUE.                              ! true for netCDF output
COUTDIR="./"                              ! output directory ("NONE" for no output)
CRIVOUTDIR="NONE"                              ! river discharge        [m3/s]
CRIVSTODIR="NONE"                           ! river storage          [m3]
CRIVVELDIR="NONE"                           ! river flow velocity    [m/s]
CRIVDPHDIR="./"                              ! river water depth      [m]
CFLDOUTDIR="NONE"                           ! floodplain discharge   [m3/s]
CFLDSTODIR="NONE"                              ! floodplain storage     [m3]
CFLDDPHDIR="./"                           ! floodplain water depth [m]
CFLDFRCDIR="NONE"                              ! flooded area fraction  [m2/m2]
CFLDAREDIR="./"                              ! flooded area           [m2]
CSFCELVDIR="./"                           ! water surface elevation           [m]
COUTFLWDIR="./"                              ! total discharge (rivout+fldout)   [m3/s]
CSTORGEDIR="NONE"                              ! total storage   (rivsto+fldsto)   [m3]
CPTHOUTDIR="NONE"                              ! net bifurcation flow (grid-based) [m3/s]
CPTHFLWDIR="NONE"                              ! bifurcation flow (channel-based)  [m3/s]
CMAXDPHDIR="NONE"
COUTINSDIR="NONE"                           ! instantaneous discharge (no river routing, summation of upstream runoff)
LOUTVEC=.FALSE.                             ! for 1-D land-only output (small data size, post processing required)
/
&NCONF                                      ! * NX, NY, NFLP, NXIN, NYIN, INPN, WEST, EAST, NORTH, SOUTH set by diminfo.txt
DT=86400                                    ! time step [sec]
DTIN=86400                                  ! input runoff time step [sec]
DROFUNIT=1.D-3                              ! runoff unit conversion (1.D-3 when input [mm] is converted to [m3/m2]
LADPSTP=.TRUE.                              ! true for adaptive time step
LFLDOUT=.TRUE.                              ! true to activate floodplain discharge
LPTHOUT=.FALSE.                             ! true to activate bifurcation channel flow
LFLD=.TRUE.                                 ! true to activate floodplain inundation
LKINE=.FALSE.                               ! true for kinematic river routing
LMAPEND=.FALSE.                             ! true to convert map data endian
LINPEND=.FALSE.                             ! true to convert input data endian
LLEAPYR=.TRUE.                              ! true for leap year calculatuon, false: always 365days/year
/
&NPARAM
PMANRIV=0.03D0                              ! manning coefficient river
PMANFLD=0.10D0                              ! manning coefficient floodplain
PGRV=9.8D0                                  ! accerelation due to gravity
PDSTMTH=10000.D0                            ! downstream distance at river mouth [m]
PCADP=0.7                                   ! satety coefficient for CFL condition
PMINSLP=1.D-5                               ! * minimum slope (for kinematic wave)
/
"""

ds_v362=u"""\
LMEANSL=.FALSE.                             ! true for mean sea level
CMEANSL=""                                  ! mean sea level
LBOUNDSL=.FALSE.                            ! true for boundary condition for variable sea level
LBOUNDCDF=.TRUE.                            ! true for netCDF sea level boundary
CBOUNDDIR=""                                ! boundary sea level directory
CBOUNDPRE=""                                ! boundary sea level prefix
CBOUNDSUF=""                                ! boundary sea level suffix
CBOUNDCDF=""                                ! * netCDF boundary sea level file name
CBNDCDFVAR="waterlevel"                     ! * netCDF boundary sea level variable name
DTBOUND=1800                                ! * time step for boundary sea level
SYEARBND={year}                             ! * for netCDF input start date (start of the initial time step)
SMONBND=1 			                        ! SHOULD BE MODIFIED TO READ DIRECTLY FROM NETCDF
SDAYBND=1    			                    ! SHOULD BE MODIFIED TO READ DIRECTLY FROM NETCDF
CBOUNDREF=""                                ! refernce table for boundary sea level"""

nam_templates = {'v362': nam_v362.format(ds=ds_v362, year='{year}', next_year='{next_year}'), 
                 'v362org': nam_v362.format(ds='', year='{year}', next_year='{next_year}'), 
                 'v392': nam_v39x,
                 'v393t': nam_v39x
                }

## MAIN
@click.command()
@click.argument('ini')
@click.argument('name') # case name
@click.option('-o', '--outdir', help='output data is saved to outdir/<name>')
@click.option('-t', '--tmpdir', help='work is done in tmpdir/<name>')
@click.option('-d', '--runoff-fn', help='runoff nc file name')
@click.option('-y', '--year', help='run single year from restart file')
@click.option('--dsbnd-fn', help='downstream sea level boundary nc file name')
@click.option('--restart', is_flag=True, help='continue with latest restart file')
@click.option('--local-io', is_flag=True, help='if true copy files else use symbolic links')
@click.option('--timeout', default=0, help='timout in sec')
@click.option('--nam', multiple=True, help='options to be set in nam file; use "option=value" syntax')
def cli(**kwargs):
    """wrapper to run cmf from python
    """
    cmf_runner(**kwargs)

def cmf_runner(ini, name, 
                # these settings overwrite the inifile
                outdir=None, tmpdir=None, year=None,
                runoff_fn=None, dsbnd_fn=None, nam=[],
                restart=False, timeout=None, local_io=False):
    # initiate config
    cmf = cmf_wrapper(name, ini)
    # overwrite settings with args (from cmd)
    if outdir:
        cmf.outdir = outdir
    if tmpdir:
        cmf.tmpdir = tmpdir
    if runoff_fn:
        cmf.runoff_fn = runoff_fn
    if dsbnd_fn:
        cmf.dsbnd_fn = dsbnd_fn
    if year:
        cmf.yrS = int(year)
        cmf.yrE = int(year)+1
        cmf.spinup = 0
    for arg in nam:
        if '=' in arg:
            opt, value = arg.split('=')
            cmf.options[opt] = value
        else:
            raise ValueError('could not parse nam arguments {}. use "option=value" syntax'.format(arg))
    # run
    cmf.initialize(local_io=local_io)
    cmf.run(restart=restart, timeout=timeout)
    

class cmf_wrapper(object):

    def __init__(self, name, ini=None):
        """Run CMF from a temporary directory"""
        self.p = multiprocessing.current_process()
        self.worker = self.p.name if self.p else ''
        self.name = name
        self.timeout = None
        if ini: 
            self.init_config(ini)
            
    def init_config(self, ini):
        # parse config
        config = configread(ini, cf=ConfigParser(inline_comment_prefixes=('#')))
        # -> required sections
        # model
        self.version = config.get('model', 'version', fallback='v362')
        model_dir = config.get('model', 'model_dir')
        prog  = config.get('model', 'prog_fn', fallback='src/MAIN_day')
        map_dir  = config.get('model', 'map_dir', fallback='map/global_15min')
        assert (isabs(prog) and isabs(map_dir)) or isabs(model_dir)
        self.prog = prog if isabs(prog) else join(model_dir, prog)
        self.map_dir = map_dir if isabs(map_dir) else join(model_dir, map_dir)
        # simulation options
        self.spinup = int(config.get('simtime', 'spinup', fallback=1))
        self.yrS = int(config.get('simtime', 'start_yr'))
        self.yrE = int(config.get('simtime', 'end_yr'))
        # -> optional sections
        # I/O (can also be set via cmd)
        self.outdir = config.get('I/O', 'out_dir', fallback=None)
        self.tmpdir = config.get('I/O', 'tmp_dir', fallback=self.outdir)
        self.runoff_fn = config.get('I/O', 'runoff_fn', fallback=None)
        self.dsbnd_fn = config.get('I/O', 'dsbnd_fn', fallback=None)
        # settings
        if config.has_section('settings'):
            nam_temp_fn = config.get('settings', 'nam_temp_fn', fallback=None)
            self.nam_temp_fn = nam_temp_fn if nam_temp_fn is None or isabs(nam_temp_fn) else join(dirname(ini), nam_temp_fn)
            if (self.nam_temp_fn is not None) and (not isfile(self.nam_temp_fn)):
                raise IOError('nam_template file not found {}'.format(self.nam_temp_fn))
            self.options = dict(((opt, config.get('settings', opt)) for opt in config.options('settings') if opt not in ['spinup', 'template']))
        else:
            self.nam_temp_fn = None
            self.options = {}

    def set_timeout(self, timeout):
        self.timeout = None if timeout <= 0 else timeout
        self.end_time = None if timeout <= 0 else time.time() + timeout

    def get_timeout(self):
        self.timeout = self.end_time - time.time() if self.timeout else None
        return self.timeout

    def setup_tmpdir(self):
        # create folder structure in tmpdir
        if not isdir(self.rundir):
            os.makedirs(self.rundir)
        if self.local_io:
            for sdir in ['input', 'restart', 'output', 'dsbnd']:
                if not isdir(join(self.tmpdir, sdir)):
                    os.makedirs(join(self.tmpdir, sdir))
        else:
            if not islink(join(self.tmpdir, 'input')):
                os.symlink(dirname(self.runoff_fn), join(self.tmpdir, 'input'))
            if not islink(join(self.tmpdir, 'output')):
                os.symlink(self.outdir, join(self.tmpdir, 'output'))
            if self.outdir != self.tmpdir and not islink(join(self.tmpdir, 'restart')):
                os.symlink(join(self.outdir, 'restart'), join(self.tmpdir, 'restart'))
            if self.dsbnd and not islink(join(self.tmpdir, 'dsbnd')):
                os.symlink(dirname(self.dsbnd_fn), join(self.tmpdir, 'dsbnd'))

        # map and prog are sym links!
        if not islink(join(self.rundir, 'MAIN_day')):
            print('link prog: {}'.format(self.prog))
            os.symlink(self.prog, join(self.rundir, 'MAIN_day'))
        if not islink(join(self.tmpdir, 'map')):
            print('link map folder: {}'.format(self.map_dir))
            os.symlink(self.map_dir, join(self.tmpdir, 'map'))

    def check_restart(self):
        # check status runs from restart folder
        restart_dir = join(self.outdir, 'restart')
        if isdir(restart_dir):
            rfns = glob.glob(join(restart_dir, 'restart*.nc'))
            if len(rfns) > 0:
                rfns_yr = [int(basename(fn).replace('restart', '')[:4]) for fn in rfns]
                self.yrS = max(rfns_yr)
                self.spinup = 0

    def initialize(self, local_io=False):
        assert self.runoff_fn is not None
        self.local_io = local_io
        if self.yrS > self.yrE:
            print('run {} finished. exit'.format(self.name))
            sys.exit(0)
        # check if downstream boundary
        self.read_nam_template()
        if int(self.version[1:4]) < 390:
            self.dsbnd = 'LBOUNDSL' in self.nam_temp['INPUT'].keys() and self.nam_temp['INPUT']['LBOUNDSL'].lower().strip('.') == 'true'
        else:
            self.dsbnd = 'LSEALEV' in self.nam_temp['INPUT'].keys() and self.nam_temp['INPUT']['LSEALEV'].lower().strip('.') == 'true'
        print('variable downstream sea level boundary: {}'.format(self.dsbnd))
        
        # set file paths
        assert self.tmpdir is not None
        assert self.outdir is not None
        # NOTE: save to subfolder with run_name
        self.tmpdir = join(self.tmpdir, self.name)
        # check tmpdir 
        if len(glob.glob(join(self.tmpdir, '*'))) > 0 or isfile(self.tmpdir):
            raise IOError('temp path not empty {}. delete before running again'.format(self.tmpdir))
        self.outdir = join(self.outdir, self.name)
        self.rundir = join(self.tmpdir, 'run')
        self.nam_fn = join(self.rundir, 'input_flood.nam')
        # create folder structure in out_dir
        print('setting up output dir: {}'.format(self.outdir))
        for sdir_out in ['log', 'restart']:
            if not isdir(join(self.outdir, sdir_out)):
                os.makedirs(join(self.outdir, sdir_out))
        print('setting up temp dir: {:s}'.format(self.tmpdir))        
        self.setup_tmpdir()


    def MAIN_day(self, log_fn=None, **kwargs):
        """run CaMa-Flood MAIN_day executalbe from command line
        write feedback to log file or print (if log_fn is None)
        kill proccess when timeout (seconds) is reached"""
        # from python 3.5 there is a timeout argument in p.wait. this is a workaround for python 2.7
        def kill(p):
            p.kill()
            
        def _run():
            timeout = self.get_timeout()
            cmd = [join(self.rundir, 'MAIN_day')]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=self.rundir, **kwargs)
            t = Timer(timeout, kill, [p])
            t.start()
            for stdout_line in iter(p.stdout.readline, ""):
                yield stdout_line 
            p.stdout.close()
            try:
                return_code = p.wait()
                if (timeout is not None) and (return_code == -9):
                    raise RuntimeError('Timeout reached')
                elif return_code:
                    raise subprocess.CalledProcessError(return_code, ' '.join(cmd))
            finally:
                t.cancel()
        
        if log_fn is not None and int(self.version[1:4]) < 390:
            # in version 362: catch logging via command line
            with open(log_fn, 'w') as f:
                for line in _run():
                    f.write(line)
        else:
            for line in _run():
                pass #print(line)
        return 1

    def run(self, restart=False, timeout=0):
        if restart: self.check_restart()
        # self.timeout = None if timeout <= 0 else timeout
        self.set_timeout(timeout)
        try:
            # spinup years
            for i in range(self.spinup):
                postfix = '_ini{:d}'.format(i)
                self.run_year(self.yrS, restart=i>0, cmf_output=False, postfix=postfix, spinup=True)
            # normal years
            for year in range(self.yrS, self.yrE+1):
                self.run_year(year, restart= restart if year == self.yrS else True)
        finally:
            # cleanup; unlink symlinks; delete other files/folders
            clean_folder2(self.tmpdir)

    def run_year(self, year, restart=False, cmf_output=True, postfix=None, spinup=False):
        # options = deepcopy(self.nam_temp)
        self.setup_tmpdir() # check if all symlinks are still OK. re-link otherwise
        options = {}
        postfix = '_{:04d}'.format(year) if postfix is None else str(postfix)
        log_fn = 'cmf_run2.log'
        
        # copy input
        runoff_src = self.runoff_fn.format(year='{:04d}'.format(year))
        CRUNOFFCDF = join('../input', basename(runoff_src))
        CRESTSTO =  "../restart/restart{year}0101.nc".format(year='{:04d}'.format(year))
        options.update({'CRUNOFFCDF': CRUNOFFCDF, 'CRESTSTO': CRESTSTO})
        if self.dsbnd:
            dsbnd_src = self.dsbnd_fn.format(year='{:04d}'.format(year))
            if int(self.version[1:4]) < 390:
                options.update({'CBOUNDCDF': join('../dsbnd', basename(dsbnd_src))})
            else:
                options.update({'CSEALEVCDF': join('../dsbnd', basename(dsbnd_src))})
        if self.local_io:
            runoff_dst = join(self.tmpdir, 'input', basename(runoff_src))
            copyfile(runoff_src, runoff_dst)
            if restart:
                restart_dst = join(self.tmpdir, 'restart', basename(CRESTSTO))
                if not isfile(restart_dst): # copy from outdir if restart file not in tmpdir
                    restart_src = join(self.outdir, 'restart', basename(CRESTSTO))
                    copyfile(restart_src, restart_dst)
            if self.dsbnd:
                dsbnd_dst = join(self.tmpdir, 'dsbnd', basename(dsbnd_src))
                copyfile(dsbnd_src, dsbnd_dst)

        # set namfile
        options.update({'IRESTART': 1 if restart else 2})
        nam = self._update_nam_temp(options, cmf_output=cmf_output, form_kwargs={'year': year, 'next_year': year+1})
        dict_to_config(nam, self.nam_fn)
        fn_nam_out = basename(self.nam_fn).replace('.nam', '{}.nam'.format(postfix))
        copyfile(self.nam_fn, join(self.outdir, 'log', fn_nam_out))

        ### RUN single year
        exit_code = 0
        try:
            t0, now = time.time(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('{}: run {}; year: {}; start time: {}'.format(self.worker, self.name, year, now))
            log_fn = join(self.rundir, log_fn)
            # kwargs=dict(cwd=self.rundir, timeout=self.timeout)
            # execute([join(self.rundir, 'MAIN_day')], log_fn=log_fn, **kwargs)
            # import pdb; pdb.set_trace()
            exit_code = self.MAIN_day(log_fn=log_fn)
            # with open(join(self.tmpdir, 'restart', 'restart{:04d}0101.nc'.format(year+1)), 'w') as f: f.write('')
            print('{}: run {}; year: {}; elapsed time: {}'.format(self.worker, self.name, year, elapsed_time(time.time()-t0)))
        
        finally:
            # copy log files -> always
            for fn in glob.glob(join(self.rundir, '*.log')):
                fn_log_out = basename(fn).replace('.log', '{}.log'.format(postfix))
                copyfile(fn, join(self.outdir, 'log', fn_log_out))
            for fn in glob.glob(join(self.rundir, '*log*.txt')):
                fn_log_out = basename(fn).replace('.txt', '{}.txt'.format(postfix))
                copyfile(fn, join(self.outdir, 'log', fn_log_out))
            # run finished, copy outputs
            if exit_code == 1 and self.local_io:
                # restart
                restart_src = join(self.tmpdir, 'restart', 'restart{:04d}0101.nc'.format(year+1))
                if not isfile(restart_src):
                    print('restart file {:s} not found'.format(restart_src))
                else:
                    if spinup: # change date of restart file to start of year
                        restart_dst = join(self.tmpdir, 'restart', 'restart{:04d}0101.nc'.format(year))
                        if isfile(restart_dst):
                            os.unlink(restart_dst)
                        copyfile(restart_src, restart_dst)
                        restart_src = restart_dst
                    # copy results to outdir
                    restart_dst = join(self.outdir, 'restart', basename(restart_src))
                    copyfile(restart_src, restart_dst)
                # outmaps
                fns = glob.glob(join(self.tmpdir, 'output', '*{:04d}.nc'.format(year)))
                if len(fns) > 0: 
                    copy_files(fns, self.outdir)

    def read_nam_template(self):
        if self.nam_temp_fn is not None:
            self.nam_temp = config_to_dict(self.nam_temp_fn)
        elif (self.version in nam_templates.keys()):
            self.nam_temp = config_to_dict(nam_templates[self.version])
        else:
            raise ValueError('unknown model version {}'.format(self.version))
        # set some base settings
        options = deepcopy(self.options)
        options.update({'CRESTDIR': '../restart/',
                        'COUTDIR': '../output/',
                        'CRUNOFFDIR': '../input/',
                        'CBOUNDDIR': '../dsbnd/'})
        self.nam_temp = self._update_nam_temp(options)

    def _update_nam_temp(self, options, cmf_output=True, form_kwargs={}):
        # Read in the file
        nam = deepcopy(self.nam_temp)
        # Replace the target string
        for header in nam.keys():
            for key in nam[header].keys():
                if key in options.keys():
                    nam[header][key] = options[key]

                if (not cmf_output) and (header == 'OUTPUT') and key.endswith('DIR'):
                    # set outputdir to NONE for spinup (cmf_output == False)
                    nam[header][key] = 'NONE'
                else:
                    try:
                        nam[header][key] = nam[header][key].format(**form_kwargs)
                    except:
                        pass # when format key not given
        return nam

## NAM parser
class NamConfigParser(ConfigParser):
    def __init__(self, **kwargs):
        defaults = dict(comment_prefixes=('!', '/'),
                        inline_comment_prefixes=('!'),
                        delimiters=('='))
        defaults.update(**kwargs)
        super(NamConfigParser, self).__init__(**defaults)
        self.SECTCRE = re.compile(r"&N(?P<header>[^]]+)")

    def write(self, fp, space_around_delimiters=False):
        """Write an .ini-format representation of the configuration state.
        If `space_around_delimiters' is True (the default), delimiters
        between keys and values are surrounded by spaces.
        """
        super(NamConfigParser, self).write(fp, space_around_delimiters=space_around_delimiters)

    def _write_section(self, fp, section_name, section_items, delimiter):
        """Write a single section to the specified `fp'."""
        fp.write(u"&N{}\n".format(section_name))
        for key, value in section_items:
            if value.lower().strip('.') in ['true', 'false']:
                value = '.TRUE.' if value.lower().strip('.')=='true' else '.FALSE.'
            else:
                try:
                    float(value.replace('D', 'e'))
                except:
                    if not value.startswith('"'):
                        value = '"{}'.format(value)
                    if not value.endswith('"'):
                        value = '{}"'.format(value)
            value = self._interpolation.before_write(self, section_name, key, value)
            if value is not None or not self._allow_no_value:
                value = delimiter + str(value).replace('\n', '\n\t')
            else:
                value = ""
            fp.write("{}{}\n".format(key.upper(), value))
        fp.write("/\n")

## UTILS


def elapsed_time(s):
    hours, s = divmod(s, 3600) #if s > 3600 else 0, s
    minutes, seconds = divmod(s, 60) #if s > 60 else 0, s
    return '{:02d}:{:02d}:{:.1f}'.format(int(hours), int(minutes), seconds)

def copy_files(fn_list, dst_dir):
    for fn in fn_list:
        dst_fn = join(dst_dir, basename(fn))
        if isfile(dst_fn): 
            os.unlink(dst_fn)
        copyfile(fn, dst_fn)

def clean_folder2(maindir):
    try:
        for subdir in os.listdir(maindir):
            path = join(maindir, subdir)
            if islink(path) or isfile(path):
                os.unlink(path)
            elif isdir(path): 
                for fn in os.listdir(path):
                    os.unlink(join(path, fn))
                os.rmdir(path)
        os.rmdir(maindir)
    except OSError as e:
        print('error while cleaning up dir {}: {}'.format(maindir, e))

def configread(config_fn, encoding='utf-8', cf=ConfigParser()):
    """read model configuration from file"""
    cf.optionxform=str # preserve capital letter
    with codecs.open(config_fn, 'r', encoding=encoding) as fp:
        cf.read_file(fp)
    return cf 

def config_to_dict(config_fn, encoding='utf-8',
                   cf=NamConfigParser()):
    "read config file to dictionary"
    cf.optionxform=str # preserve capital letter
    if isfile(config_fn):
        with codecs.open(config_fn, 'r', encoding=encoding) as fp:
            cf.read_file(fp)
    else:
        cf.read_string(config_fn)
    out_dict = OrderedDict((sec, OrderedDict((opt, cf.get(sec, opt))
                                    for opt in cf.options(sec)))
                            for sec in cf.sections())
    return out_dict

def dict_to_config(config, config_fn, encoding='utf-8',
                   cf=NamConfigParser(), **kwargs):
    "read config file to dictionary"
    if not isinstance(config, dict):
        raise ValueError("config argument should be of type dictionary")
    cf.read_dict(config)
    with codecs.open(config_fn, 'w', encoding=encoding) as fp:
        cf.write(fp)


if __name__ == "__main__":
    cli()