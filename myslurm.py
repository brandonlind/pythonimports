"""Python commands to interface with slurm queue and seff commands."""


import os
import time
import subprocess
import shutil
import matplotlib.pyplot as plt


def get_mems(infos:dict, unit='MB', plot=True) -> list:
    """From dict(infos) [val = seff output], extract mem in MB.
    
    fix: add in other mem units
    """
    mems = []
    for key,info in infos.items():
        if 'running' in info[3].lower() or 'pending' in info[3].lower():
            continue
        info[-2] = info[-2].split("(estimated")[0]
        mem, units = info[-2].split()[-2:]
        if units == 'GB':
            mem = float(mem)*1024
        elif units == 'EB':
            mem = 0
        else:
            try:
                assert units == 'MB'
            except AssertionError:
                print('info = ', info)
            mem = float(mem)
        if unit == 'GB':
            mem = mem/1024
        mems.append(mem)
    
    if plot is True:
        plt.hist(mems)
        plt.xlabel(unit)
        plt.show()
    
    return mems


def clock_hrs(clock:str, unit='hrs') -> float:
    """from a clock (days-hrs:min:sec) extract hrs.
    
    fix: add in other clock units"""
    hrs = 0
    if '-' in clock:
        days, clock = clock.split("-")
        hrs += 24*float(days)
    h,m,s = clock.split(":")
    hrs += float(h)
    hrs += float(m)/60
    hrs += float(s)/3600
    return hrs


def get_times(infos:dict, unit='hrs', plot=True) -> list:
    """From dict(infos) [val = seff output], get times in hours.
    
    fix: add in other clock units"""
    times = []
    for key, info in infos.items():
        if 'running' in info[3].lower() or 'pending' in info[3].lower():
            continue
        clock = info[-3].split()[-1]
        hrs = clock_hrs(clock, unit=unit)
        times.append(hrs)
    
    if plot is True:
        plt.hist(times)
        plt.xlabel(unit)
        plt.show()
    return times


def sbatch(files:list, sleep=0) -> list:
    """From a list of .sh files, sbatch them and return associated jobid in a list."""
    if isinstance(files, list) is False:
        assert isinstance(files, str)
        files = [files]
    pids = []
    for file in files:
        os.chdir(os.path.dirname(file))
        pid = subprocess.check_output([shutil.which('sbatch'), file]).decode('utf-8').replace("\n", "").split()[-1]
        print('sbatched %s' % file)
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


def seff(pid:str) -> list:
    """Using jobid, get seff output from bash."""
    lst = os.popen('seff %s' % pid).read().split("\n")
    lst.remove('')
    return lst


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

