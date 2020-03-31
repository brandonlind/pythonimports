"""Python commands to interface with slurm queue and seff commands."""


import os
import time
import subprocess
import shutil
from tqdm import tqdm as nb
import matplotlib.pyplot as plt


def get_seff(outs:list):
    """From a list of .out files (ending in f'_{SLURM_JOB_ID}.out'), get seff output."""
    infos = {}
    for out in nb(outs):
        infos[out] = seff(getpid(out))
    return infos


def get_mems(infos:dict, units='MB', plot=True) -> list:
    """From dict(infos) [val = seff output], extract mem in MB.

    fix: add in other mem units
    """
    mems = []
    for key,info in infos.items():
        if 'running' in info.state().lower() or 'pending' in info.state().lower():
            continue
        mems.append(info.mem(units=units))

    if plot is True:
        plt.hist(mems)
        plt.xlabel(units)
        plt.show()

    return mems


def clock_hrs(clock:str, unit='hrs') -> float:
    """From a clock (days-hrs:min:sec) extract hrs or days as float."""
    assert unit in ['hrs', 'days']
    hrs = 0
    if '-' in clock:
        days, clock = clock.split("-")
        hrs += 24*float(days)
    h,m,s = clock.split(":")
    hrs += float(h)
    hrs += float(m)/60
    hrs += float(s)/3600
    if unit=='days':
        hrs = hrs/24
    return hrs


def get_times(infos:dict, unit='hrs', plot=True) -> list:
    """From dict(infos) [val = seff output], get times in hours.

    fix: add in other clock units"""
    times = []
    for key, info in infos.items():
        if 'running' in info.state().lower() or 'pending' in info.state().lower():
            continue
        hrs = info.walltime(unit=unit)
        times.append(hrs)

    if plot is True:
        plt.hist(times)
        plt.xlabel(unit.title() if unit=='days' else 'Hours')
        plt.show()
    return times


def sbatch(shfiles:list, sleep=0, printing=False) -> list:
    """From a list of .sh shfiles, sbatch them and return associated jobid in a list."""
    
    if isinstance(shfiles, list) is False:
        assert isinstance(shfiles, str)
        shfiles = [shfiles]
    pids = []
    failcount = 0
    for sh in nb(shfiles):
        os.chdir(os.path.dirname(sh))
        try:
            pid = subprocess.check_output([shutil.which('sbatch'), sh]).decode('utf-8').replace("\n", "").split()[-1]
        except subprocess.CalledProcessError as e:
            failcount += 1
            if failcount == 10:
                print('!!!REACHED FAILCOUNT LIMIT OF 10!!!')
                return pids
            continue
        if printing is True:
            print('sbatched %s' % sh)
        pids.append(pid)
        time.sleep(sleep)
    return pids


def getpids(user=os.environ['USER']) -> list:
    """From squeue -u $USER, return list of queue."""
    pids = [q[0] for q in getsq() if q[0] != '']
    if len(pids) != len(list(set(pids))):
        print('len !- luni pids')
    return pids


def getjobs(user=os.environ['USER']) -> list:
    """From squeue -u $USER, return list of job names, alert if len != unique."""
    jobs = [q[3] for q in getsq() if q[3] != '']
    if len(jobs) != len(list(set(jobs))):
        print('len != luni jobs')
    return jobs


def qaccounts(pd=False, user=os.environ['USER']) -> list:
    """From squeue -u $USER, return list of billing accounts."""
    if pd == False:
        accounts = [q[2] for q in getsq() if q != '']
    else:
        accounts = [q[2] for q in getsq(states=['PD']) if q != '']
    return accounts


class seff():
    """Parse info output by `seff $SLURM_JOB_ID`.
    
    example output from os.popen __init__ call
    
    ['Job ID: 38771990',
    'Cluster: cedar',
    'User/Group: lindb/lindb',
    'State: COMPLETED (exit code 0)',
    'Nodes: 1',
    'Cores per node: 48',
    'CPU Utilized: 56-18:58:40',
    'CPU Efficiency: 88.71% of 64-00:26:24 core-walltime',
    'Job Wall-clock time: 1-08:00:33',
    'Memory Utilized: 828.22 MB',
    'Memory Efficiency: 34.51% of 2.34 GB']
    
    """
    def __init__(self, slurm_job_id):
        # get info
        info = os.popen('seff %s' % str(slurm_job_id)).read().split("\n")
        info.remove('')
        self.info = info
        self.info[-2] = self.info[-2].split("(estimated")[0]
        self.slurm_job_id = str(slurm_job_id)
        

    def pid(self) -> str:
        """Get SLURM_JOB_ID."""
        return self.slurm_job_id
    
    def state(self) -> str:
        """Get state of job (R, PD, COMPLETED, etc)."""
        return self.info[3]
    
    def cpu_u(self, unit='clock') -> str:
        """Get CPU time utilized by job (actual time CPUs were active across all cores)."""
        utilized = self.info[6].split()[-1]
        if unit != 'clock':
            utilized = clock_hrs(utilized, unit)
        return utilized
    
    def cpu_e(self) -> str:
        """Get CPU efficiency (cpu_u() / core_walltime())"""
        return self.info[7].split()[2]
    
    def core_walltime(self, unit='clock') -> str:
        """Get time that CPUs were active (across all cores)."""
        walltime = self.info[7].split()[-2]
        if unit != 'clock':
            walltime = clock_hrs(walltime, unit)
        return walltime
    
    def walltime(self, unit='clock') -> str:
        """Get time that job ran after starting."""
        walltime = self.info[-3].split()[-1]
        if unit != 'clock':
            walltime = clock_hrs(walltime, unit)
        return walltime
    
    def mem_req(self, units='MB') -> str:
        """Get the requested memory for job."""
        mem, mem_units = self.info[-1].split()[-2:]
        return self._convert_mem(mem, mem_units, units)
    
    def mem_e(self) -> str:
        """Get the memory efficiency (~ mem / mem_req)"""
        return self.info[-1].split()[2]
    
    def _convert_mem(self, mem, units, unit='MB'):
        """Convert between memory units."""
        if units == 'GB':
            mem = float(mem)*1024
        elif units == 'EB':
            mem = 0
        else:
            try:
                assert units == 'MB'
            except AssertionError as e:
                print('info = ', self.info)
                raise e
            mem = float(mem)
        if unit == 'GB':
            mem = mem/1024
        return mem
    
    def mem(self, units='MB', per_core=False) -> str:
        """Get memory unitilized by job (across all cores, or per core)."""
        mem, mem_units = self.info[-2].split()[-2:]
        mem = self._convert_mem(mem, mem_units, units)
        if per_core is True:
            mem = mem / self.info[5].split()[-1]
        return mem
    pass


def getpid(out:str) -> list:
    """From an .out file with structure <anytext_JOBID.out>, return JOBID."""
    return out.split("_")[-1].replace('.out', '')


def checksq(sq):
    """Make sure queue slurm command worked. Sometimes it doesn't.
    
    Positional arguments:
    sq - list of squeue slurm command jobs, each line is str.split()
       - slurm_job_id is zeroth element of str.split()
    """
    exitneeded = False
    if not isinstance(sq, list):
        print("\ttype(sq) != list, exiting %(thisfile)s" % globals())
        exitneeded = True
    for s in sq:
        if 'socket' in s.lower():
            print("\tsocket in sq return, exiting %(thisfile)s" % globals())
            exitneeded = True
        if not int(s.split()[0]) == float(s.split()[0]):
            print("\tcould not assert int == float, %s" % (s[0]))
            exitneeded = True
    if exitneeded is True:
        print('\tslurm screwed something up for %(thisfile)s, lame' % globals())
        exit()
    else:
        return sq


def getsq_exit(balancing):
    """Determine if getsq is being used to balance priority jobs.

    Positional arguments:
    balancing - bool: True if using to balance priority jobs, else for other queue queries
    """
    print('\tno jobs in queue matching query')
    if balancing is True:
        print('\texiting balance_queue.py')
        exit()
    else:
        return []


def getsq(grepping=None, states=[], balancing=False, user=None):
    """
    Get jobs from squeue slurm command matching crieteria.

    Positional arguments:
    grepping - list of key words to look for in each column of job info
    states - list of states {pending, running} wanted in squeue jobs
    balancing - bool: True if using to balance priority jobs, else for other queue queries

    Returns:
    grepped - list of tuples where tuple elements are line.split() for each line of squeue \
slurm command that matched grepping queries
    """
    if user is None:
        user = os.environ['USER']
    if grepping is None:
        grepping = [user]
    if isinstance(grepping, str):
        # in case I pass a single str instead of a list of strings
        grepping = [grepping]

    # get the queue, without a header
    cmd = [shutil.which('squeue'),
           '-u',
           user,
           '-h']
    if 'running' in states:
        cmd.extend(['-t', 'RUNNING'])
    elif 'pending' in states:
        cmd.extend(['-t', 'PD'])
    sqout = subprocess.check_output(cmd).decode('utf-8').split('\n')

    sq = [s for s in sqout if s != '']
    checksq(sq)  # make sure slurm gave me something useful

    # look for the things I want to grep
    grepped = []
    if len(sq) > 0:
        for q in sq:  # for each job in queue
            splits = q.split()
            if 'CG' not in splits:  # grep -v 'CG' = skip jobs that are closing
                keepit = 0
                if len(grepping) > 0:  # see if all necessary greps are in the job
                    for grep in grepping:
                        for split in splits:
                            if grep.lower() in split.lower():
                                keepit += 1
                                break
                if keepit == len(grepping) and len(grepping) != 0:
                    grepped.append(tuple(splits))

        if len(grepped) > 0:
            return grepped
    return getsq_exit(balancing)


def adjustjob(acct, jobid):
    """Move job from one account to another."""
    subprocess.Popen([shutil.which('scontrol'),
                      'update',
                      'Account=%s_cpu' % acct,
                      'JobId=%s' % str(jobid)])

