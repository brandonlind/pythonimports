"""Python commands to interface with slurm queue and seff commands."""
import os
import time
import subprocess
import shutil
import copy
from tqdm import tqdm
from tqdm import trange as _trange
import matplotlib.pyplot as plt
import pandas as pd
from os import path as op
from collections import defaultdict, Counter
from typing import Union
from functools import partial
import warnings

import pythonimports as pyimp
import balance_queue as balq
import myfigs
from myclasses import MySeries

trange = partial(_trange, bar_format='{l_bar}{bar:15}{r_bar}')
pbar = partial(tqdm, bar_format='{l_bar}{bar:15}{r_bar}')


def getpid(out: str) -> str:
    """From an .out file with structure <anytext_JOBID.out>, return JOBID."""
    return out.split("_")[-1].replace(".out", "")


def get_seff(outs=None, pids=None, desc='executing seff commands', progress_bar=True, pids_as_keys=False):
    """From a list of outs or pids, get seff output.
    
    Parameters
    ----------
    outs : list
        .out files (ending in f'_{SLURM_JOB_ID}.out')
    pids : list
        a list of slurm_job_ids
    desc
        description for progress bar
    progress_bar : bool
        whether to use a progress bar when querying seff
    pids_as_keys : bool
        if outs is not None, retrieve pid from each outfile to use as the key
        
    Notes
    -----
    - assumes f'{job}_{slurm_job_id}.out' and f'{job}.sh' underly slurm jobs
    """
    jobs = outs if outs is not None else pids

    exception_text = None
    if outs is not None:
        if type(outs) in [dict().keys().__class__, dict().values().__class__]:
            outs = list(outs)
        if isinstance(outs[0], str) is False:
            exception_text = f'out is not a string: {type(outs[0]) = }'
        elif outs[0].endswith('.out') is False:
            exception_text = f'outs files must end with ".out": outs[0] = {outs[0]}'
        if exception_text is not None:
            raise(Exception(exception_text))

    if progress_bar is True and len(jobs) > 0:
        iterator = pbar(jobs, desc=desc)
    else:
        iterator = jobs

    seffs = {}
    for job in iterator:
        pid = getpid(job)

        if pids_as_keys is True:
            key = pid
        else:
            key = job

        seffs[key] = Seff(pid)

        if outs is not None:
            seffs[key].out = job
            seffs[key].job = '_'.join(op.basename(job).split("_")[:-1])
            seffs[key].sh = op.join(op.dirname(job), f'{seffs[key].job}.sh')
        else:
            seffs[key].out = None
            seffs[key].job = None
            seffs[key].sh = None

    return seffs


def get_mems(seffs: dict, units="MB", plot=True, **kwargs) -> list:
    """From output by `get_seff()`, extract mem in `units` units; histogram if `plot` is True.

    Parameters
    ----------
    - seffs : dict of any key with values of class Seff
    - units : passed to Seff._convert_mem(). options: GB, MB, KB

    """
    mems = []
    for info in seffs.values():
        if "running" in info.state().lower() or "pending" in info.state().lower():
            continue
        mems.append(info.mem(units=units, per_core=False))

    if plot is True:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax_box, ax_hist = myfigs.histo_box(mems,
                                           xlab=units,
                                           ylab='count',
                                           title=f'n_mems = {len(mems)}',
                                           ax=ax,
                                           **kwargs)
        plt.show()

    return mems


def clock_hrs(clock: str, unit="hrs") -> float:
    """From a clock (days-hrs:min:sec) extract hrs or days as float."""
    assert unit in ["hrs", "days", 'mins', 'minutes']
    hrs = 0
    if "-" in clock:
        days, clock = clock.split("-")
        hrs += 24 * float(days)
    h, m, s = clock.split(":")
    hrs += float(h)
    hrs += float(m) / 60
    hrs += float(s) / 3600
    if unit == "days":
        hrs /= 24
    elif unit == 'mins' or unit=='minutes':
        hrs *= 60
    return hrs


def get_times(seffs: dict, unit="hrs", plot=True) -> list:
    """From dict(seffs) [val = seff output], get times in hours.

    fix: add in other clock units"""
    times = []
    for key, info in seffs.items():
        if "running" in info.state().lower() or "pending" in info.state().lower():
            continue
        hrs = info.walltime(unit=unit)
        times.append(hrs)

    if plot is True:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax_box, ax_hist = myfigs.histo_box(times, xlab=unit, ylab='count', title=f'n_times = {len(times)}', ax=ax)
        plt.show()

    return times


def sbatch(shfiles: Union[str, list], sleep=0, printing=False, outdir=None, progress_bar=True) -> list:
    """From a list of .sh shfiles, sbatch them and return associated jobid in a list.

    Notes
    -----
    - assumes that the job name that appears in the queue is the basename of the .sh file
        - eg for job_187.sh, the job name is job_187
        - this convention is used to make sure a job isn't submitted twice
    - `sbatch` therefore assumes that each job has a unique job name
    """

    if isinstance(shfiles, list) is False:
        assert isinstance(shfiles, str)
        shfiles = [shfiles]

    if progress_bar is True:
        iterator = pbar(shfiles, desc='sbatching')
    else:
        iterator = shfiles

    pids = []
    for sh in iterator:
        # make sure job matches filename
        filejob = op.basename(sh).split(".")[0]
        sbatchflag = [line for line in pyimp.read(sh, lines=True) if '--job-name' in line][0].split("job-name=")[1]
        if filejob != sbatchflag:
            exceptiontext = pyimp.ColorText(
                "The .sh file's basename does not match the job name in the .sh file's SBATCH flags.\n"
            ).fail().bold().__str__() +\
            f"\tfile basename = {filejob}\n" +\
            f"\tsbatchflag = {sbatchflag}\n"
            raise Exception(exceptiontext)

        if outdir is None:
            os.chdir(os.path.dirname(sh))
        else:
            os.chdir(outdir)

        # try and sbatch file 10 times before giving up
        failcount = 0
        sbatched = False
        while sbatched is False:
            try:
                pid = (subprocess.check_output([shutil.which("sbatch"), sh])
                       .decode("utf-8")
                       .replace("\n", "")
                       .split()[-1])
                sbatched = True
            except subprocess.CalledProcessError:
                failcount += 1
                if failcount == 10:
                    print("!!!REACHED FAILCOUNT LIMIT OF 10!!!")
                    return pids
            # one more failsafe to ensure a job isn't submitted twice
            sq = Squeue(verbose=False)
            jobs = defaultdict(list)
            for _pid, q in sq.items():
                jobs[q.job].append(_pid)
            if filejob in list(jobs.keys()):
                sbatched = True
                filepids = jobs[filejob]
                if len(filepids) > 1:
                    # cancel all but the oldest job in the queue
                    sq.cancel(grepping=sorted(filepids)[1:])
                pid = sorted(filepids)[0]
                break
            else:
                sbatched = False
        if printing is True:
            print("sbatched %s" % sh)
        pids.append(pid)
        time.sleep(sleep)
        del pid

    return pids


def create_watcherfile(pids, directory, watcher_name="watcher", email="b.lind@northeastern.edu", time='0:00:01', ntasks=1, 
                       rem_flags=None, mem=25, end_alert=False, fail_alert=True, begin_alert=False, added_text=''):
    """From a list of dependency `pids`, sbatch a file that will not start until all `pids` have completed.
    
    Parameters
    ----------
    pids - list of SLURM job IDs
    directory - where to sbatch the watcher file
    watcher_name - basename for slurm job queue, and .sh and .outfiles
    email - where alerts will be sent
        requires at least of of the following to be True: end_alert, fail_alert, begin_alert
    time - time requested for job
    ntasks - number of tasks
    rem_flags - list of additional SBATCH flags to add (separate with \n)
        eg - rem_flags=['#SBATCH --cpus-per-task=5', '#SBATCH --nodes=1']
    mem - requested memory for job
        default is 25 bytes, but any string will work - eg mem='2500M'
    end_alert - bool
        use if wishing to receive an email when the job ends
    fail_alert - bool
        use if wishing to receive an email if the job fails
    begin_alert - bool
        use if wishing to receive an email when the job begins
    added text - any text to add within the body of the .sh file

    TODO
    ----
    - incorporate code to save mem and time info of `pids`
    """
    rem_flags = '\n'.join(rem_flags) if rem_flags is not None else ''
    end_text = '#SBATCH --mail-type=END' if end_alert is True else ''
    fail_text = '#SBATCH --mail-type=FAIL' if fail_alert is True else ''
    begin_text = '#SBATCH --mail-type=BEGIN' if begin_alert is True else ''
    email = f'#SBATCH --mail-user={email}' if any([end_alert, fail_alert, begin_alert]) else ''

    watcherfile = op.join(directory, f"{watcher_name}.sh")
    jpids = ",".join(pids)
    text = f"""#!/bin/bash
#SBATCH --job-name={watcher_name}
#SBATCH --time={time}
#SBATCH --ntasks={ntasks}
#SBATCH --mem={mem}
#SBATCH --output={watcher_name}_%j.out
#SBATCH --dependency=afterok:{jpids}
{email}
{fail_text}
{end_text}
{begin_text}
{rem_flags}

{added_text}

"""

    with open(watcherfile, "w") as o:
        o.write(text)

    print(sbatch(watcherfile))

    return watcherfile


class Seff:
    """Parse info output by `seff $SLURM_JOB_ID`.

    example output from os.popen __init__ call

    ['Job ID: 38771990',                                      0
    'Cluster: cedar',                                         1
    'User/Group: lindb/lindb',                                2
    'State: COMPLETED (exit code 0)',                         3
    'Nodes: 1',                                               4  # won't always show up
    'Cores per node: 48',                                     5 -6
    'CPU Utilized: 56-18:58:40',                              6 -5
    'CPU Efficiency: 88.71% of 64-00:26:24 core-walltime',    7 -4
    'Job Wall-clock time: 1-08:00:33',                        8 -3
    'Memory Utilized: 828.22 MB',                             9 -2
    'Memory Efficiency: 34.51% of 2.34 GB']                  10 -1

    """
    def __init__(self, slurm_job_id):
        """Get return from seff command."""
        info = os.popen("seff %s" % str(slurm_job_id)).read().split("\n")
        info.remove("")
        self.info = [i for i in info[:11] if 'WARNING' not in i]
        self.info[-2] = self.info[-2].split("(estimated")[0]
        self.slurm_job_id = str(slurm_job_id)
        if len(self.info) == 11:
            self.nodes = int(self.info[4].split()[-1])
            self.cpus = self.info[5].split()[-1]
        else:
            self.nodes = 1
            self.cpus = 1
        pass

    def __repr__(self):
        return repr(self.info)

    def pid(self) -> str:
        """Get SLURM_JOB_ID."""
        return self.slurm_job_id

    def state(self) -> str:
        """Get state of job (R, PD, COMPLETED, etc)."""
        return self.info[3]

    def cpu_u(self, unit="clock") -> str:
        """Get CPU time utilized by job (actual time CPUs were active across all cores)."""
        utilized = self.info[-5].split()[-1]
        if unit != "clock":
            utilized = clock_hrs(utilized, unit)
        return utilized

    def cpu_e(self) -> str:
        """Get CPU efficiency (cpu_u() / core_walltime())"""
        return self.info[-4].split()[2]

    def core_walltime(self, unit="clock") -> str:
        """Get time that CPUs were active (across all cores)."""
        walltime = self.info[-4].split()[-2]
        if unit != "clock":
            walltime = clock_hrs(walltime, unit)
        return walltime

    def walltime(self, unit="clock") -> str:
        """Get time that job ran after starting."""
        walltime = self.info[-3].split()[-1]
        if unit != "clock":
            walltime = clock_hrs(walltime, unit)
        return walltime

    def mem_req(self, units="MB") -> Union[float, int]:
        """Get the requested memory for job."""
        mem, mem_units = self.info[-1].split('(')[0].split()[-2:]
        return self._convert_mem(mem, mem_units, units)

    def mem_e(self) -> str:
        """Get the memory efficiency (~ mem / mem_req)"""
        return self.info[-1].split()[2]

    def _convert_mem(self, mem, mem_units, units="MB") -> float:
        """Convert between memory mem_units."""
        # first convert reported mem to MB
        if mem_units == "GB":
            mem = float(mem) * 1024
        elif mem_units == "KB":
            mem = float(mem) / 1024
        elif mem_units == "EB":
            mem = 0
        else:
            assert mem_units == "MB", ("info = ", mem_units, self.info)
            mem = float(mem)
        # then convert to requested mem
        if units == "GB":
            mem /= 1024
        elif units == "KB":
            mem *= 1024
        return mem

    def mem(self, units="MB", per_core=False) -> float:
        """Get memory unitilized by job (across all cores, or per core)."""
        mem, mem_units = self.info[-2].split()[-2:]
        mem = self._convert_mem(mem, mem_units, units)
        if per_core is True:
            mem = mem / float(self.info[-6].split()[-1])
        return mem

    def copy(self):
        seff = type(self).__new__(self.__class__)
        
        seff.info = self.info.copy()
        seff.slurm_job_id = str(self.slurm_job_id)
        seff.nodes = self.nodes
        seff.cpus = self.cpus
        for attr in ['job', 'out', 'sh']:
            if attr in self.__dict__:
                seff.__dict__[attr] = self.__dict__[attr]
            else:
                seff.__dict__[attr] = None
        
        return seff

    pass


class Seffs:
    """dict-like container with arbitrary keys and values for multiple `Seff` class objects.
    
    Notes
    -----
    - __isub__ and __iadd__ do not Seffs.check_shfiles for duplicates (but __add__ and __sub__ do)

    """
    def __init__(self, outs=None, seffs=None, pids=None, pids_as_keys=True, units="MB", unit="clock", plot=False, progress_bar=True):
        if any([outs is not None, seffs is not None, pids is not None]):
            if outs is not None or pids is not None:
                seffs = get_seff(outs=outs, pids=pids, pids_as_keys=pids_as_keys, progress_bar=progress_bar)
            else:
                for key, seff in seffs.items():
                    if 'out' not in seff.__dict__:
                        seff.out = None
                    if 'job' not in seff.__dict__:
                        seff.job = None
                    if 'sh' not in seff.__dict__:
                        seff.sh = None
        else:
            raise Exception('one of `outs` or `pids` or `seffs` kwargs needs to be provided.')

        self.seffs = seffs

        self.unit = unit

        self.units = units

        self.outs = [seff.out for seff in self.seffs.values()]

        self.jobs = [seff.job for seff in self.seffs.values()]

        self.shfiles = [seff.sh for seff in self.seffs.values()]

        self.pids = [seff.slurm_job_id for seff in self.seffs.values()]

        self.slurm_job_ids = self.pids

        self.states = MySeries([seff.state() for seff in self.seffs.values()],
                               index=pyimp.keys(seffs))

        self.mems = MySeries(
            get_mems(self.seffs, units=units, plot=plot),
            dtype=float,  # f "running" in info.state().lower() or "pending" in info.state().lower(
            name=units,
            index = [key for (key, seff) in self.items()
                     if 'running' not in seff.state().lower()
                     and 'pending' not in seff.state().lower()]
        )

        self.times = MySeries(
            get_times(self.seffs,
                      unit=unit if unit != 'clock' else 'hrs',
                      plot=plot),
            dtype=float,
            name=unit if unit != 'clock' else 'hrs',
            index = [key for (key, seff) in self.items()
                     if 'running' not in seff.state().lower()
                     and 'pending' not in seff.state().lower()]
        )

        self.walltimes = self.times

        self.cpu_us = MySeries([seff.cpu_u(unit=unit) for seff in seffs.values()],
                               index=pyimp.keys(seffs),
                               name=unit)

        self.cpu_es = MySeries([seff.cpu_e() for seff in seffs.values()],
                               index=pyimp.keys(seffs))

        self.core_walltimes = MySeries([seff.core_walltime(unit=unit) for seff in seffs.values()],
                                       index=pyimp.keys(seffs), name=unit)

        self.mem_reqs = MySeries([seff.mem_req(units=units) for seff in seffs.values()],
                                 index=pyimp.keys(seffs), name=units)

        self.mem_es = MySeries([seff.mem_e() for seff in seffs.values()],
                               index=pyimp.keys(seffs))
        
        self.cpus = MySeries([seff.cpus for seff in seffs.values()],
                             index=pyimp.keys(seffs))
        
        self.nodes = MySeries([seff.nodes for seff in seffs.values()],
                              index=pyimp.keys(seffs))
        
        Seffs.check_shfiles(self.shfiles)

        pass
    
    @classmethod
    def check_shfiles(cls, shfiles):
        if len(shfiles) != pyimp.luni(shfiles) and shfiles[0] is not None:
            # perhaps a given shfile was run more than once, or duplicate job names?
            text = 'There are multiple shfiles associated with outfiles.'
            text += f' len={len(shfiles)}'
            text += f' luni={pyimp.luni(shfiles)}'
            warnings.warn(text)
        pass

    def __repr__(self):
        return repr(self.seffs)

    def __add__(self, seffs2):
        assert isinstance(seffs2, Seffs)
        seffs = copy.deepcopy(self.seffs)
        seffs.update(seffs2.seffs)
        
        newseffs = Seffs(seffs=seffs, units=self.units, unit=self.unit)
        Seffs.check_shfiles(newseffs.shfiles)

        return newseffs

    def __iadd__(self, seffs2):
        assert isinstance(seffs2, Seffs)
        self.seffs.update(seffs2.seffs)

        self.outs = [seff.out for seff in self.seffs.values()]

        self.jobs = [seff.job for seff in self.seffs.values()]

        self.shfiles = [seff.sh for seff in self.seffs.values()]

        self.pids = [seff.slurm_job_id for seff in self.seffs.values()]

        self.slurm_job_ids = self.pids

        self.states = Seffs._update(self.states, seffs2.states)

        self.mems = Seffs._update(
            self.mems,
            MySeries(
                get_mems(seffs2.finished().seffs, units=self.units, plot=False),
                name=self.units,
                index=pyimp.keys(seffs2.finished().seffs)
            )  # TODO: add index?
        )

        self.times = Seffs._update(
            self.times,
            MySeries(
                get_times(seffs2.finished().seffs, unit=self.unit if self.unit != 'clock' else 'hrs', plot=False),
                index=pyimp.keys(seffs2.finished()),
                name=self.unit if self.unit != 'clock' else 'hrs'
            )
        )

        self.walltimes = self.times

        self.cpu_us = Seffs._update(
            self.cpu_us,
            MySeries([seff.cpu_u(unit=self.unit) for seff in seffs2.values()],
                     index=pyimp.keys(seffs2),
                     name=self.unit)
        )

        self.cpu_es = Seffs._update(self.cpu_es, seffs2.cpu_es)

        self.core_walltimes = Seffs._update(
            self.core_walltimes,
            MySeries([seff.core_walltime(unit=self.unit) for seff in seffs2.values()],
                     index=pyimp.keys(seffs2),
                     name=self.unit)
        )

        self.mem_reqs = Seffs._update(
            self.mem_reqs,
            MySeries([seff.mem_req(units=self.units) for seff in seffs2.values()],
                     index=pyimp.keys(seffs2),
                     name=self.units)
        )

        self.mem_es = Seffs._update(self.mem_es, seffs2.mem_es)
        
        self.cpus = Seffs._update(self.cpus, seffs2.cpus)
        
        self.nodes = Seffs._update(self.nodes, seffs2.nodes)

        return self

    def __sub__(self, seffs2):
        assert any([isinstance(seffs2, Seffs), isinstance(seffs2, list)])
        oldseffs = copy.deepcopy(self.seffs)
        for key in seffs2:
            try:
                oldseffs.pop(key)
            except KeyError as e:
                pass

        newseffs = Seffs(seffs=oldseffs)
        Seffs.check_shfiles(newseffs.shfiles)

        return newseffs

    def __isub__(self, seffs2):
        assert any([isinstance(seffs2, Seffs), isinstance(seffs2, list)])
        for key, seff in seffs2.items():
            for i in range(16):  # i want to do all of these things below even if any preceding fails
                try:
                    if i == 0:
                        self.seffs.pop(key)
                    elif i == 1:
                        self.pids.remove(seff.pid)  # self.slurm_job_ids is same as self.pids
                    elif i == 2:
                        self.states.pop(key)
                    elif i == 3:
                        self.mems.pop(key)
                    elif i == 4:
                        self.times.pop(key)  # self.walltimes is same as self.times
                    elif i == 5:
                        self.cpu_us.pop(key)
                    elif i == 6:
                        self.cpu_es.pop(key)
                    elif i == 7:
                        self.core_walltimes.pop(key)
                    elif i == 8:
                        self.mem_reqs.pop(key)
                    elif i == 9:
                        self.mem_es.pop(key)
                    elif i == 10:
                        self.outs.remove(seff.out)
                    elif i == 11:
                        self.jobs.remove(seff.job)
                    elif i == 12:
                        self.shfiles.remove(seff.sh)
                    elif i == 13:
                        self.nodes.remove(seff.nodes)
                    elif i == 14:
                        self.cpus.remove(seff.cpus)
                except KeyError as e:
                    pass
                except ValueError as e:
                    print(pid, len(self.pids), len(self.slurm_job_ids))
                    raise e

        return self                

    def __len__(self):
        return len(self.seffs)

    def __iter__(self):
        return iter(self.seffs.keys())

    def __getitem__(self, key):
        return self.seffs[key]

    def __setitem__(self, key, item):
        self.seffs[key] = item

    def __delitem__(self, key):
        del self.seffs[key]

    def __contains__(self, key):
        return True if key in pyimp.keys(self) else False

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    @staticmethod
    def _update(attr1, attr2):
        name = attr1.name
        attr1 = attr1.to_dict()
        attr1.update(attr2.to_dict())
        return MySeries(attr1, name=name)

    def plot_mems(self, **kwargs):
        _ = get_mems(self.seffs, **kwargs)
        pass

    def plot_times(self, **kwargs):
        _ = get_times(self.finished().seffs, **kwargs)
        pass

    def keys(self):
        return self.seffs.keys()

    def values(self):
        return self.seffs.values()

    def items(self):
        return self.seffs.items()

    def copy(self):
        return Seffs(seffs=self.seffs.copy(), unit=self.unit, units=self.units)
    
    def len(self):
        return len(self)

    def out_sh(self):
        """key = out, val = sh."""
        outdict = {}
        for key, seff in self.seffs.items():
            outdict[seff.out] = seff.sh

        return outdict
    
    @staticmethod
    def filter_states(seffs, state):
        newseffs = {}
        for key, seff in seffs.items():
            if state.lower() in seff.state().lower():
                newseffs[key] = seff.copy()
        return newseffs
    
    def running(self):
        """Return Seffs object for any running job."""
        seffs = Seffs.filter_states(self.seffs, 'running')
        return Seffs(seffs=seffs.copy(), unit=self.unit, units=self.units)
    
    def pending(self):
        """Return pending jobs."""
        seffs = Seffs.filter_states(self.seffs, 'pending')
        return Seffs(seffs=seffs.copy(), unit=self.unit, units=self.units)

    def completed(self):
        """Return Seffs object for any *successfully* completed job (compare to Seffs.finished())."""
        seffs = Seffs.filter_states(self.seffs, 'completed')
        return Seffs(seffs=seffs.copy(), unit=self.unit, units=self.units)

    def failed(self):
        """Return Seffs object for any failed job."""
        seffs = Seffs.filter_states(self.seffs, 'failed')
        return Seffs(seffs=seffs.copy(), unit=self.unit, units=self.units)

    def cancelled(self):
        """Return Seffs object for any cancelled job."""
        seffs = Seffs.filter_states(self.seffs, 'cancelled')
        return Seffs(seffs=seffs.copy(), unit=self.unit, units=self.units)
    
    def timeouts(self):
        """Return Seffs object for any timeout jobs."""
        seffs = Seffs.filter_states(self.seffs, 'timeout')
        return Seffs(seffs=seffs.copy(), unit=self.unit, units=self.units)
    
    def sh_outs(self, sh_as_key=True, internal=False):
        """key = sh, val = list of outfiles."""
        shdict = defaultdict(list)
        for seffkey, seff in self.items():
            if sh_as_key is True:
                shdict[seff.sh].append(seff.out)
            else:
                shdict[seffkey].append(seff.out)

        return shdict

    def sh_out(self, sh_as_key=True):
        """key = sh, val = most_recent outfile.

        TODO
        ----
        - add `remove` kwarg and pass to pyimp.getmostrecent
            - but do I care that any outfiles removed could have elements within the Seff?
                - eg a pid or out as a key
        """
        shdict = self.sh_outs(sh_as_key=sh_as_key, internal=True)

        for key in pyimp.keys(shdict):            
            shdict[key] = pyimp.getmostrecent(shdict[key])

        shdict = dict(shdict)

        return shdict
    
    def finished(self):
        """Return non-running and non-pending jobs."""
        seffs = {}
        for key, seff in self.items():
            if "running" in seff.state().lower() or "pending" in seff.state().lower():
                continue
            seffs[key] = seff.copy()
        
        return Seffs(seffs=seffs.copy(), unit=self.unit, units=self.units)
    
    def uncompleted(self):
        """Return Seffs object for any uncompleted job.
        
        Notes
        -----
        - if most recent .out failed but an early .out completed, this code will miss that
        """
        recent_outs = self.sh_out(sh_as_key=False)
        
        seffs = {}
        for key, out in recent_outs.items():
            seff = self[key].copy()
            state = seff.state().lower()
            if 'completed' not in state and 'pending' not in state and 'running' not in state:
                seffs[key] = seff
                assert 'sh' in seff.__dict__, seff.__dict__

        return Seffs(seffs=seffs.copy(), unit=self.unit, units=self.units)

    pass


class SQInfo:
    """Convert each line returned from `squeue -u $USER`.

    Assumed
    -------
    SQUEUE_FORMAT="%i %u %a %j %t %S %L %D %C %b %m %N (%r) %P"

    Notes
    -----
    - I realized that %N can be blank when pending, and then cause problems with .split()
        so the attrs now can handle this. But I'm leaving the methods for backwards compatibility.

    Example jobinfo    (index number of list)
    ---------------
    ('29068196',       0
     'b.lindb',        1
     'lotterhos',      2
     'batch_0583',     3
     'R',              4
     'N/A',            5
     '9:52:32',        6
     '1',              7
     '56',             8
     'N/A',            9
     '2000M',          10
     'd0036',          11
     '(Priority)')     12
     'short'           13
    """

    def __init__(self, jobinfo):
        self.info = list(jobinfo)
        
        (self.pid,        # 0
         self.user,       # 1
         self.account,    # 2
         self.job,        # 3
         self.state,      # 4
         self.start,      # 5
         self.time,       # 6
         self.nodes,      # 7
         self.cpus,       # 8
         self.tres_p,     # 9
         self.memory,     # 10
         *self.nodelist,  # 11
         self.status,     # 12
         self.partition   # 13
        ) = list(jobinfo)
        
        if len(self.nodelist) > 0 and self.nodelist[0].startswith('('):
            # sometimes status can have spaces in between parentheses which is passed to nodelist when nodelist is blank
                # nodelist is blank when eg job is not running
            self.status = ' '.join(self.nodelist) + ' ' + self.status
            self.nodelist = []

        pass

    def __repr__(self):
        return repr(dict((k, v) for (k, v) in self.__dict__.items() if k != 'info'))

    def __iter__(self):
        return iter(self.info)

    def mem(self, units="MB"):
        memory = self.memory
        if all([memory.endswith("G") is False, memory.endswith("M") is False]):
            print("Unexpected units found in memory: ", memory)
            raise Exception
        if memory.endswith("G") and units == "MB":
            memory = memory.replace("G", "")
            memory = int(memory) * 1024
        elif memory.endswith("M") and units == "GB":
            memory = memory.replace("M", "")
            memory = int(memory) / 1024
        else:
            memory = int(memory.replace("M", ""))
        return memory

    pass


sqinfo = SQInfo  # backwards compatibility


class Squeue:
    """dict-like container class for holding and updating slurm squeue information.

    Methods - most methods can be filtered by passing the kwargs from Squeue._filter_jobs to the method
    -------
    states - return a list of job states (eg running, pending)
    jobs - return a list of job names
    pids - return a list of pids (SLURM_JOB_IDs)
    accounts - return a list of accounts
    cancel - cancel entire queue or specific jobs for specific accounts (if user is True, cancel all jobs)
    update - update time, memory, or account for specifc jobs for specific accounts
    balance - balance jobs in queue across available slurm accounts
    summary - print counts of various categories within the queue
    hold - hold jobs
    release - release held jobs
    keys - returns list of all pids (SLURM_JOB_IDs) - cannot be filtered with Squeue._filter_jobs
    items - returns item tuples of (pid,SQInfo) - cannot be filtered with Squeue._filter_jobs
    values - returns list of SQInfos - cannot be filtered with Squeue._filter_jobs
    save_default_accounts - save default accounts to use in Squeue.balance, cannot be filtered with Squeue._filter_jobs


    TODO
    ----
    TODO: address Squeue.balance() TODOs
    TODO: update_job needs to handle skipping over jobs that started running or closed after class instantiation
    TODO: update_job needs to skip errors when eg trying to increase time of job beyond initial submission
            eg when initially scheduled for 1 day, but update tries to extend beyond 1 day
            (not allowed on compute canada)
    TODO: make it so it can return the queue for `grepping` without needing `user` (ie all users)
    TODO: address `user` kwarg potential conflict between _get_sq and _filter_jobs

    Examples
    --------
    Get squeue information (of class SQInfo) for each job in output (stdout line) containing 'batch' or 'gatk'.

    >>> sq = Squeue(grepping=['batch', 'gatk'])


    Get queue with "batch" in one of the columns (eg the NAME col).
    For theses jobs, only update jobs with "batch_001" for mem and time.

    >>> sq = Squeue(grepping='batch', states='PD')
    >>> sq.update(grepping='batch_001', minmemorynode=1000, timelimit=3-00:00:00)


    Cancel all jobs in queue.

    >>> sq = Squeue()
    >>> sq.cancel(user=True)
    # OR:
    >>> Squeue().cancel(user=True)


    Cancel jobs in queue with "batch" in one of the columns (eg the NAME col).
    For theses jobs only cancel job names containing "batch_001" and 3 day run time,
        but not the 'batch_0010' job.

    >>> sq = Squeue(grepping='batch', states='PD')
    >>> sq.cancel(grepping=['batch_001', '3-00:00:00'], exclude='batch_0010')


    Get jobs from a specific user.

    >>> sq = Squeue(user='some_user')

    Get a summary of queue.

    >>> sq = Squeue()
    >>> sq.summary()


    """
    # export SQUEUE_FORMAT
    # (JOBID USER ACCOUNT NAME ST START_TIME TIME_LEFT NODES CPUS TRES_PER_NODE MIN_MEMORY NODELIST (REASON)) PARTITION
    #    %i   %u    %a     %j  %t     %S        %L      %D    %C         %b        %m         %N      (%r)       %P
    os.environ["SQUEUE_FORMAT"] = "%i %u %a %j %t %S %L %D %C %b %m %N (%r) %P"
    
    def __init__(self, verbose=True, **kwargs):
        # get queue matching grepping
        self.sq = Squeue._getsq(**kwargs)
        # filter further with kwargs
        if len(self.sq) > 0:
            self.sq = self._filter_jobs(self, **kwargs)
            if len(self.sq) == 0 and verbose is True:
                print("\tno jobs in queue matching query")
        pass

    def __repr__(self):
        return repr(self.sq)

    def __add__(self, sq2):
        assert isinstance(sq2, Squeue)
        newself = copy.deepcopy(self)
        newself.sq.update(sq2)
        return newself

    def __iadd__(self, sq2):
        assert isinstance(sq2, Squeue)
        self.sq.update(sq2)
        return self

    def __sub__(self, sq2):
        assert any([isinstance(sq2, Squeue), isinstance(sq2, list)])
        newself = copy.deepcopy(self)
        for pid in sq2:
            try:
                newself.sq.pop(pid)
            except KeyError as e:
                pass
        return newself

    def __isub__(self, sq2):
        assert any([isinstance(sq2, Squeue), isinstance(sq2, list)])
        for pid in sq2:
            self.sq.pop(pid)
        return self                

    def __len__(self):
        return len(self.sq)

    def __iter__(self):
        return iter(self.sq.keys())

    def __getitem__(self, key):
        return self.sq[key]

    def __setitem__(self, key, item):
        self.sq[key] = item

    def __delitem__(self, key):
        del self.sq[key]

    def __contains__(self, pid):
        return True if pid in self.keys() else False

    def __cmp__(self, other):
        return cmp(self.sq, other.sq)

    @staticmethod
    def _grep_sq(sq, grepping):
        """Get jobs that have any match to anything in `grepping`."""
        if isinstance(grepping, str):
            # in case I pass a single str instead of a list of strings
            grepping = [grepping]
        elif grepping is None:
            grepping = [os.environ["USER"]]

        grepped = {}
        for q in sq:  # for each job in queue
            if q.__class__.__name__ == "SQInfo":  # when called from Squeue.balance()
                splits = q.info
            else:  # when called from Squeue._getsq()
                splits = q.split()
            if "CG" not in splits:  # grep -v 'CG' = skip jobs that are closing
                keepit = 0
                if len(grepping) > 0:  # see if all necessary greps are in the job
                    for grep in grepping:
                        for split in splits:
                            if grep.lower() in split.lower():
                                keepit += 1
                                break
                if keepit == len(grepping):  # and len(grepping) != 0
                    info = SQInfo(splits)
                    grepped[info.pid] = info
        return grepped

    @staticmethod
    def _getsq(grepping=None, states=[], user=None, partition=None, **kwargs):
        """Get and parse slurm queue according to criteria. kwargs is not used."""

        def _checksq(sq):
            """Make sure queue slurm command worked. Sometimes it doesn't.

            Positional arguments:
            sq - list of squeue slurm command jobs, each line is str.split()
            """
            exceptionneeded = False
            if not isinstance(sq, list):
                print("\ttype(sq) != list, exiting Squeue")
                exceptionneeded = True
            else:
                for s in sq:
                    if "socket" in s.lower():
                        print("\tsocket in sq return, exiting Squeue")
                        exceptionneeded = True
                    if not int(s.split()[0]) == float(
                        s.split()[0]
                    ):  # if the jobid isn't a float:
                        print("\tcould not assert int == float, %s" % (s[0]))
                        exceptionneeded = True
            if exceptionneeded is True:
                raise Exception(pyimp.ColorText("FAIL: slurm screwed something up for Squeue, lame").fail().bold())
            return sq

        if user is None:
            user = os.environ["USER"]
        if grepping is None:
            grepping = [user]

        # get the queue, without a header
        cmd = [shutil.which("squeue"), "-u", user, "-h"]
        if any(["running" in states, "R" in states, "r" in states]):
            cmd.extend(["-t", "RUNNING"])
        if any(["pending" in states, "PD" in states, "pd" in states]):
            cmd.extend(["-t", "PD"])
        if partition is not None:
            cmd.extend(["-p", partition])

        # execute command
        found = 0
        while found < 5:
            try:
                sqout = subprocess.check_output(cmd).decode("utf-8").split("\n")
                found = 10
            except subprocess.CalledProcessError:
                found += 1
                pass
        if found == 5:
            raise Exception(pyimp.ColorText("FAIL: Exceeded five subprocess.CalledProcessError errors.").fail().bold())

        sq = [s for s in sqout if s != ""]
        _checksq(sq)  # make sure slurm gave me something useful

        # look for the things I want to grep
        grepped = Squeue._grep_sq(sq, grepping)
        if len(grepped) == 0:
            print("\tno jobs in queue matching query")
        return grepped

    @staticmethod
    def _handle_mem(mem):
        """If the memory units are specified, remove."""
        mem = str(mem)
        if mem.endswith("M") or mem.endswith("G"):
            mem = mem[:-1]
        return mem

    @staticmethod
    def _handle_account(account):
        """Make sure account name is specified correctly."""
        if isinstance(account, str):
            if not account.endswith("_cpu"):
                account = f"{account}_cpu"
        else:
            for i, acct in enumerate(account):
                if not acct.endswith("_cpu"):
                    account[i] = f"{acct}_cpu"
        return account

    @staticmethod
    def _handle_clock(clock):
        """Create aesthetic clock."""
        *days, clock = clock.split("-")
        clock = ":".join([digits.zfill(2) for digits in clock.split(":")])
        return f"{days[0]}-{clock}" if len(days) > 0 else clock

    @staticmethod
    def _update_self(info, **kwargs):
        """After successfully updating jobid with scontrol update, update Squeue class object."""
        if "account" in kwargs:
            info[2] = Squeue._handle_account(kwargs["account"])
        if "minmemorynode" in kwargs:
            info[10] = Squeue._handle_mem(kwargs["minmemorynode"]) + "M"
        if "timelimit" in kwargs:
            info[6] = Squeue._handle_clock(kwargs["timelimit"])
        return info

    @staticmethod
    def _update_job(cmd, job=None, jobid=None):
        """Execute 'scontrol update' or 'scancel' command for jobid."""
        # add jobid to command
        if jobid is not None:
            cmd.append(f"jobid={jobid}" if "scancel" not in cmd else jobid)

        failcount = 0
        # try to execute commands three times before giving up (slurm is a jerk sometimes)
        while failcount < 3:
            try:
                out = (
                    subprocess.check_output(cmd)
                    .decode("utf-8")
                    .replace("\n", "")
                    .split()
                )
                return True
            except subprocess.CalledProcessError:
                # see if job is running or still in queue
                if all([job is None, jobid is None, "scancel" in "".join(cmd)]):
                    # for scancel -u $USER
                    sq = Squeue(verbose=False)
                    if len(sq) > 0:
                        failcount += 1
                    else:
                        return True
                else:
                    jobq = Squeue(grepping=jobid, verbose=False)
                    if len(jobq) == 0:
                        return "missing"
                    elif jobq[jobid].state == "R":
                        return "running"
                    # otherwise count as failure
                    failcount += 1
        print(pyimp.ColorText(f"FAIL: Update failed for cmd: {job} {jobid}").fail().bold())
        return False

    @staticmethod
    def _filter_jobs(sq, grepping=None, exclude=None, onaccount=None, priority=None, states=None, **kwargs):
        """Filter jobs in `Squeue` class object.
        Parameters
        ----------
        grepping - a string or list of strings to use as queries to select job from queue (case insensitive)
            - strings are converted to a single-element list internally
            - to keep a job from the queue, all elements of `grepping` must be found in >= of the squeue columns
                - the elements of `grepping` can be a subset of the column info (eg `squeue -u $USER | grep match`)
                    - to retrieve specific jobs, make sure elements of grep are not substrings of any other job info
                - note SQUEUE_FORMAT immediately after Squeue.__doc__
        exclude - a string or list of strings that should be excluded if found in any of the job info fields
            (case insensitive)
            - strings are converted to a single-element list internally
            - if any element of `exclude` is found in one of the columns, the job is not returned
                - jobs will be excluded if any element of exclude is a substring of any of the job info
            - note SQUEUE_FORMAT immediately after Squeue.__doc__
        onaccount - a string or list of strings (account names) to subset queue
            - strings are converted to lists internally
            - like `grepping` and `exclude`, if any element of onaccount is a substring of the job info, the job
                will be returned
            - onaccount gets appended to `grepping` and so account name should not be a substrint of other column
                info TODO: use SQInfo.account after queue query to filter jobs based on account
            - note SQUEUE_FORMAT immediately after Squeue.__doc__
        priority - if True, only balance jobs with priority status; if False, balance jobs without priority status,
            if None, balance all jobs regardless of priority status
        TODO
        ----
        - use SQInfo.account after queue query to filter jobs based on account
        """
        def _exclude_jobs(_sq, exclude=None):
            """Remove anything that needs to be excluded (including non-pending jobs)."""
            for pid in list(_sq.keys()):
                information = _sq[pid]
                remove = False
                if exclude is not None:
                    for _info in information:
                        for ex in exclude:
                            if ex.lower() in _info.lower():
                                remove = True
                #                 if 'pd' not in information.state.lower():  # if a job isn't pending, it can't be updated
                #                     remove = True
                if remove is True:
                    _sq.pop(pid)
            return _sq

        # set up exclude as list
        if exclude is None:
            exclude = []
        else:
            if isinstance(exclude, str):
                exclude = [exclude]
            else:
                assert isinstance(exclude, list), "Squeue.balance() only expects `exclude` as `str` or `list`."

        # set up grepping as list
        if grepping is None:
            grepping = []
        elif isinstance(grepping, str):
            grepping = [grepping]
        else:
            assert isinstance(grepping, list), "Squeue.balance() only expects `grepping` as `str` or `list`."

        # add wanted accounts to `grepping`
        if onaccount is not None:
            # add accounts to grepping
            if isinstance(onaccount, str):
                grepping.append(onaccount)
            elif isinstance(onaccount, list):
                grepping.extend(onaccount)
            else:
                raise Exception("Squeue.balance() only expects `onaccount` as `str` or `list`.")

        # determine whether to keep/ignore priority status jobs
        if priority is True:
            grepping.append("priority")
        elif priority is False:
            exclude.append("priority")

        # limit queue to keyword match
        _sq = Squeue._grep_sq(sq.values(), grepping)

        # remove anything that needs to be excluded (including non-pending jobs)
        _sq = _exclude_jobs(_sq, exclude)

        # remove state mismatch
        if states is not None:
            if isinstance(states, str):
                states = [states]
            for pid in list(_sq.keys()):
                info = _sq[pid]
                keep = False
                for state in states:
                    if state.lower() in info.state.lower():
                        keep = True
                        break
                if keep is False:
                    _sq.pop(pid)

        return _sq

    def _update_queue(self, cmd, desc, user=False, num_jobs=None, **kwargs):
        """Update jobs in queue and job info in Squeue class object."""

        if user is False:  # scontrol commands
            cmd = cmd.split()
        elif desc == "scancel" and user is True:  # cancel all jobs
            return Squeue._update_job(["scancel", "-u", os.environ["USER"]])
        # get subset of jobs returned from __init__()
        _sq = self._filter_jobs(self, **kwargs)
        # update each of the jobs
        if len(_sq) > 0:
            if num_jobs is None:
                num_jobs = len(_sq)
            for q in pbar(list(_sq.values())[:num_jobs], desc=desc):
                # if the job is updated successfully
                updated_result = Squeue._update_job(cmd, q.job, q.pid)
                if updated_result is True:
                    if "scancel" in cmd:
                        # remove job from Squeue container
                        self.__delitem__(q.pid)
                    else:
                        # update job info in Squeue container
                        self[q.pid].info = Squeue._update_self(q.info, **kwargs)
                elif updated_result == "running":
                    self[q.pid].info = updated_result[1]
                elif updated_result == "missing":
                    self.__delitem__(q.pid)
                else:
                    assert (
                        updated_result is False
                    ), '`updated_result` must be one of {True, False, "missing", "running"}'
        else:
            print(pyimp.ColorText("\tNone of the jobs in Squeue class passed criteria.").custom('lightyellow'))
        pass

    @staticmethod
    def _save_default_accounts(save_dir=os.environ["HOME"]):
        """Among accounts available, choose which to use during balancing.

        The chosen accounts will be saved as op.join(save_dir, 'accounts.pkl'), and will be used
            to balance accounts in the future when setting `parentdir` in Squeue.balance to `save_dir`.
        """
        balq.get_avail_accounts(parentdir=save_dir, save=True)

        pass

    def keys(self):
        return list(self.sq.keys())

    def values(self):
        return list(self.sq.values())

    def items(self):
        return self.sq.items()

    def copy(self):
        obj = type(self).__new__(self.__class__)
        obj.sq = self.sq.copy()
        return obj

    def running(self):
        obj = type(self).__new__(self.__class__)
        obj.sq = self._filter_jobs(self, states="R").copy()
        return obj

    def pending(self):
        obj = type(self).__new__(self.__class__)
        obj.sq = self._filter_jobs(self, states="PD").copy()
        return obj

    def states(self, **kwargs):
        """Get a list of job states."""
        _sq = self._filter_jobs(self, **kwargs)
        return [info.state for q, info in _sq.items()]

    def pids(self, **kwargs):
        """Get a list of pids, subset with kwargs."""
        _sq = self._filter_jobs(self, **kwargs)
        return [info.pid for q, info in _sq.items()]

    def jobs(self, **kwargs):
        """Get a list of job names, subset with kwargs."""
        _sq = self._filter_jobs(self, **kwargs)
        return [info.job for q, info in _sq.items()]

    def accounts(self, **kwargs):
        """Get a list of accounts, subset with kwargs."""
        _sq = self._filter_jobs(self, **kwargs)
        return [info.account for q, info in _sq.items()]
    
    def partitions(self):
        """Get counts of job states across partitions."""
        partitions = defaultdict(Counter)
        for pid, info in self.sq.items():
            partitions[info.partition][info.state] += 1

        return dict(partitions)

    def cancel(self, **kwargs):
        """Cancel jobs in slurm queue, remove job info from Squeue class."""
        # cancel all jobs if user is True
        self._update_queue(cmd="scancel", desc="scancel", **kwargs)

        pass

    def update(self, num_jobs=None, **kwargs):
        """Update jobs in slurm queue with scontrol, and update job info in Squeue class.

        kwargs - that control what can be updated (other kwargs go to Squeue._filter_jobs)
        ------
        account - the account to transfer jobs
        minmemorynode - total memory requested
        timelimit - total wall time requested
        """
        def _cmd(account=None, minmemorynode=None, timelimit=None, to_partition=None, to_reservation=None, **kwargs):
            """Create bash command for slurm scontrol update."""
            # base command
            cmd_ = "scontrol update"
            # update cmd by appending flags
            if account is not None:
                account = Squeue._handle_account(account)
                cmd_ = f"{cmd_} Account={account}"
            if minmemorynode is not None:
                minmemorynode = Squeue._handle_mem(minmemorynode)
                cmd_ = f"{cmd_} MinMemoryNode={minmemorynode}"
            if timelimit is not None:
                timelimit = Squeue._handle_clock(timelimit)
                cmd_ = f"{cmd_} TimeLimit={timelimit}"
            if to_partition is not None:
                cmd_ = f"{cmd_} partition={to_partition}"
            if to_reservation is not None:
                cmd_ = f"{cmd_} reservation={to_reservation}"
            return cmd_

        # get scontrol update command
        cmd = _cmd(**kwargs)

        # update each of the jobs
        Squeue._update_queue(self, cmd, "update", num_jobs=num_jobs, **kwargs)
        pass

    def balance(self, parentdir='HOME', **kwargs):
        """Evenly distribute pending jobs across available slurm sbatch accounts.

        Parameters
        ----------
        parentdir - used in balance_queue to look for `accounts.pkl`; see balance_queue.__doc__
            - `parentdir` can be set to 'choose' to manually choose slurm accounts from those available
            - if `parentdir` is set to `None`, then all available accounts will be used to balance
            - otherwise, `parentdir` can be set to `some_directory` that contains "accounts.pkl" saved from:
                balance_queue.get_avail_accounts(some_directory, save=True)
            - default = os.environ['HOME']
        kwargs - see Squeue._filter_jobs.__doc__

        Notes
        -----
        - printouts (and docstrings of balance_queue.py + functions) will refer to 'priority jobs', since this
            was the purpose of the app. However, Squeue.balance() can pass jobs that are not necessarily of
            priority status.

        TODO
        ----
        - balance even if all accounts have jobs
            As of now, if all accounts have jobs (whether filtering for priority status or not),
            Squeue.balance() will not balance. This behavior is inherited from balance_queue.py. The quickest
            work-around is to use `onaccount` to balance jobs on a specific account (eg the one with the most jobs).
        """
        # 🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁
        # balance_queue.py originated as part of the CoAdapTree project: github.com/CoAdapTree/varscan_pipeline
        # 🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁

        os.environ["SQUEUE_FORMAT"] = "%i %u %a %j %t %S %L %D %C %b %m %N (%r) %P"

        if parentdir == 'HOME':
            parentdir = os.environ["HOME"]

        if "priority" in kwargs.keys():
            priority = kwargs["priority"]
        else:
            priority = None

        print(pyimp.ColorText("\nStarting balance_queue.py").bold())

        # get jobs that match input parameters
        _sq = self._filter_jobs(self, **kwargs)
        for pid in list(_sq.keys()):
            # remove non-pending jobs, since these cannot be balanced
            info = _sq[pid]
            if info.state != "PD":
                _sq.pop(pid)
        if len(_sq) == 0 or _sq is None:
            print("\tno pending jobs in queue matching query")
            return

        # get accounts available for billing
        user_accts = balq.get_avail_accounts(parentdir)

        if len(user_accts) > 1:
            # get per-account lists of jobs in pending status, return if all accounts have jobs (no need to balance)
            accts, early_exit_decision = balq.getaccounts(_sq,
                                                          "",
                                                          user_accts)
            balq.announceacctlens(accts, early_exit_decision, priority=priority)
            if early_exit_decision is True:
                return

            # determine number of jobs to redistribute to each account
            balance = balq.getbalance(accts, len(user_accts))

            # redistribute
            balq.redistribute_jobs(accts, user_accts, balance)

            # announce final job counts
            time.sleep(2)  # give system a little time to update (sometimes it can print original job counts)
            if "priority" in kwargs:
                # if the job no longer has same priority status, it won't be found in new queue query
                kwargs.pop("priority")
            if "onaccount" in kwargs:
                # if the job is no longer on the queried account, it won't be found in new queue query
                kwargs.pop("onaccount")
            sq = self._filter_jobs(Squeue(), **kwargs)  # re-query the queue, filter
            balq.announceacctlens(*balq.getaccounts(sq,
                                                    "final",
                                                    user_accts),
                                  priority=priority)  # print updated counts
            # update self
            for pid, q in _sq.items():
                account = sq[pid].account
                self[q.pid].info = Squeue._update_self(q.info, account=account)
        else:
            print(f"\tthere is only one account ({user_accts[0]}), no more accounts to balance queue.")

        pass

    def summary(self, **kwargs):
        """Print counts of states and statuses of the queue."""
        _sq = self._filter_jobs(self, **kwargs)

        # count stuff
        stats = defaultdict(Counter)
        statuses = Counter()
        states = Counter()
        account_counts = Counter()
        for pid, q in _sq.items():
            stats[q.account][q.status] += 1
            stats[q.account][q.state] += 1
            statuses[q.status] += 1
            states[q.state] += 1
            account_counts[q.account] += 1

        # print account stats
        print(pyimp.ColorText("There are %s accounts with jobs matching search criteria." % len(stats.keys())).bold())
        for account, acct_stats in stats.items():
            print("\t", account, "has", account_counts[account], "total jobs, which include:")
            for stat, count in acct_stats.items():
                print("\t\t", count, "jobs with", "%s" % "state =" if stat in ["R", "PD"] else "status =", stat)
        # print status counts
        print(pyimp.ColorText("\nIn total, there are %s jobs:" % sum(statuses.values())).bold())
        for status, count in statuses.items():
            print("\t", count, "jobs with status =", status)
        # print state counts
        for state, count in states.items():
            print("\t", count, "jobs with state =", state)

        pass

    def hold(self, **kwargs):
        """Hold jobs. Parameters described in `Squeue._update_job.__doc__`."""
        _sq = self._filter_jobs(self, **kwargs)

        for pid, q in pbar(_sq.items()):
            if "pd" in q.state.lower():  # only pending jobs can be held
                updated_result = Squeue._update_job([shutil.which("scontrol"), "hold"], q.job, pid)
                if updated_result is True:
                    # update job info in Squeue container
                    self[q.pid].info = Squeue._update_self(q.info, **kwargs)
                elif updated_result == "running":
                    self[q.pid].info = updated_result[1]
                elif updated_result == "missing":
                    self.__delitem__(q.pid)
        pass

    def release(self, **kwargs):
        """Release held jobs. Parameters described in `Squeue._update_job.__doc__`."""
        _sq = self._filter_jobs(self, **kwargs)

        released = 0
        for pid, q in pbar(_sq.items()):
            if "held" in q.status.lower():  # JobHeldUser
                updated_result = Squeue._update_job([shutil.which("scontrol"), "release"], q.job, pid)
                if updated_result is True:
                    # update job info in Squeue container
                    self[q.pid].info = Squeue._update_self(q.info, **kwargs)
                elif updated_result == "running":
                    self[q.pid].info = updated_result[1]
                elif updated_result == "missing":
                    self.__delitem__(q.pid)
                released += 1
        print(pyimp.ColorText(f"\tReleased {released} jobs").gray())
        pass

    pass


getsq = Squeue._getsq  # backwards compatibility
