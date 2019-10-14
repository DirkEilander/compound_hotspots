#!/usr/bin/env python

import os
from os.path import join, isfile, isdir, abspath
import sys
from subprocess import Popen, PIPE, call
import click 
from datetime import timedelta

templates = {
'TORQUE': """\
#!/bin/bash
#PBS -S {path_list}
#PBS -N {jobname}
#PBS -d {workdir}
#PBS -o {stdout}
#PBS -e {stderr}
#PBS -lwalltime={lwalltime}
{args}
{modules}
{precommand}
{command}
{postcommand}""",

'TORQUE_simple': """\
#!/bin/bash
#PBS -S {path_list}
#PBS -N {jobname}
{args}
{modules}
{precommand}
{command}
{postcommand}""",

'SLURM': """\
#!/bin/bash
#SBATCH -t {lwalltime}
#SBATCH --job-name {jobname}
{args}
{modules}
cd "{workdir}"
{precommand}
{command}
{postcommand}"""
}

drm_symb = {
    'TORQUE': '#PBS',
    'TORQUE_simple': '#PBS',
    'SLURM': '#SBATCH'
}

drm_cmd = {
    'TORQUE': 'qsub',
    'TORQUE_simple': 'qsub',
    'SLURM': 'sbatch'
}

def fill_template(drm='TORQUE', **kwargs):
    jobscript = templates[drm].format(**kwargs)
    return jobscript

## MAIN
cwd = os.getcwd()
@click.command()
@click.argument('jobname')
@click.argument('command')
# @click.option('-o', '--stdout', default=cwd, help='the path to be used for the standard output stream of the batch job', show_default=True)
# @click.option('-e', '--stderr', default=cwd, help='the path to be used for the standard error stream of the batch job', show_default=True)
@click.option('-d', '--workdir', default=getattr(os.environ, 'HOMEDIR', cwd), help='the working directory path to be used for the job', show_default=True)
@click.option('--jobdir', default=cwd, help='directory to write job scripts, output and error files', show_default=True)
@click.option('-S', '--path_list', default='/bin/bash', help='the shell that interprets the job script', show_default=True)
@click.option('-V', is_flag=True, help="all environment variables in the qsub command's environment are exported to the batch job", show_default=True)
@click.option('--lwalltime', default=1., help='required walltime for job; unit = hours', show_default=True)
@click.option('--args', multiple=True, help='additional arguments to send the batch server; use "option=value" or "flag" syntax', show_default=True)
@click.option('--drm', default='TORQUE', type=click.Choice(['TORQUE']), help='the DRM to generate scripts for', show_default=True)
@click.option('--modules', multiple=True, help='modules to import at the start of the job script')
@click.option('--precommand', default='', help='command to excecute before the main command')
@click.option('--postcommand', default='', help='command to excecute after the main command')
@click.option('--submit', is_flag=True, help='submit the job to queue', show_default=True)
@click.option('--sargs', '--submit_args', multiple=True, help='additional arguments to use when submitting the job', show_default=True)
@click.option('--force-overwrite', is_flag=True, help='overwrite previous job script', show_default=True)
@click.option('--verbose', is_flag=True, help='enables verbose mode', show_default=True)
def cli(**kwargs):
    qsub(**kwargs)

def qsub(jobname, command, 
        workdir=getattr(os.environ, 'HOMEDIR', cwd), jobdir=cwd, path_list='/bin/bash',
        v=False, lwalltime=1, args=[], drm='TORQUE', modules=[], precommand='', postcommand='', 
        submit=False, submit_args=[], force_overwrite=False, verbose=False):
        """
        creates a job script to submit to a batch serve
        """
        # I/O
        if not isdir(jobdir): os.makedirs(jobdir)
        
        # parse additional arguments
        lwalltime = timedelta(hours=lwalltime).total_seconds()
        if drm == 'SLURM':
            lwalltime = lwalltime/60
        args_parsed = []
        prefix = drm_symb[drm]
        args = args + tuple('V') if v else args
        for arg in args:
            try:
                if '=' in arg:
                    opt = arg.split('=')[0]
                    value = '='.join(arg.split('=')[1:])
                    # if opt.startswith('l'):
                    #     args_parsed.append('{:s} -{:s}={:s}'.format(prefix, opt, value))
                    # else:
                    args_parsed.append('{:s} -{:s} {:s}'.format(prefix, opt, value))
                else:
                    args_parsed.append('{:s} -{:s}'.format(prefix, arg))
            except:
                raise ValueError('could not parse additional arguments {}. use "option=value" or "flag" syntax'.format(arg))

        # create job_script
        job_script = fill_template(
            drm=drm, 
            path_list=path_list, 
            jobname=jobname, 
            workdir = workdir,
            stdout = jobdir, 
            stderr = jobdir, 
            lwalltime= '{:.0f}'.format(lwalltime),
            args= '\n'.join(args_parsed),
            modules= '\n'.join(['module load {:s}'.format(mod) for mod in modules]),
            precommand= precommand, 
            command= command, 
            postcommand= postcommand
        )

        # write to file to file
        fn = join(jobdir, '{:s}.sh'.format(jobname))
        if not force_overwrite and os.path.isfile(fn):
            raise IOError('job script with same filename already exists. use --force-overwrite flag to overwrite old job script')
        elif force_overwrite and os.path.isfile(fn):
            os.unlink(fn)
        if verbose: click.echo('writing job script to {}'.format(fn))
        with open(fn, 'w') as f:
            f.write(job_script)

        # submit 
        qsub_cmd = [drm_cmd[drm]]
        if len(submit_args) != 0:
            sargs_parsed = []
            for arg in submit_args:
                try:
                    if '=' in arg:
                        opt, value = arg.split('=')
                        sargs_parsed.append('-{:s} {:s}'.format(opt, value))
                    else:
                        sargs_parsed.append('-{:s}'.format(arg))
                except:
                    raise ValueError('could not parse additional submit arguments {}. use "option=value" or "flag" syntax'.format(arg))
            qsub_cmd.extend(sargs_parsed)
        qsub_cmd.append(f.name)
        if submit:
            click.echo('submitting job: "{}"'.format(' '.join(qsub_cmd)))
            p = Popen(' '.join(qsub_cmd),shell=True,stdout=PIPE,cwd=jobdir)
            p.wait()
            stdout, stderr = p.communicate()
            click.echo(str(stdout))
            click.echo(str(stderr))
        else:
            click.echo('to submit the job run: "{}"'.format(' '.join(qsub_cmd)))

if __name__ == '__main__':
    cli()