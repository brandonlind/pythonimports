"""Python commands to interface with slurm queue and seff commands."""


import os
import subprocess

def get_mems(infos:dict, unit='MB') -> list:
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


def get_times(infos:dict, unit='hrs') -> list:
    """From dict(infos) [val = seff output], get times in hours.
    
    fix: add in other clock units"""
    times = []
    for key, info in infos.items():
        if 'running' in info[3].lower() or 'pending' in info[3].lower():
            continue
        clock = info[-3].split()[-1]
        hrs = clock_hrs(clock, unit=unit)
        times.append(hrs)
    return times


def sbatch(files:list) -> list:
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
    return pids


def getpids(user=os.environ['USER']) -> list:
    """From squeue -u $USER, return list of queue."""
    pids = os.popen(f'squeue -u {user} -h -o "%i"').read().split("\n")
    pids = [p for p in pids if not p == '']
    if len(pids) != len(list(set(pids))):
        print('len !- luni pids')
    return pids


def getjobs(user=os.environ['USER']) -> list:
    """From squeue -u $USER, return list of job names, alert if len != unique."""
    jobs = os.popen(f'squeue -u {user} -h -o "%j"').read().split("\n")
    jobs = [j for j in jobs if not j == '']
    if len(jobs) != len(list(set(jobs))):
        print('len != luni jobs')
    return jobs


def getaccounts(pd=False, user=os.environ['USER']) -> list:
    """From squeue -u $USER, return list of billing accounts."""
    if pd == False:
        accounts = os.popen(f'squeue -u {user} -o "%a"').read().split("\n")
    else:
        accounts = os.popen(f'squeue -u {user} -t "pd" -o "%a"').read().split('\n')
    accounts = [a for a in accounts if not a in ['', 'ACCOUNT']]
    return accounts


def seff(pid:str) -> list:
    """Using jobid, get seff output from bash."""
    lst = os.popen('seff %s' % pid).read().split("\n")
    lst.remove('')
    return lst


def getpid(out:str) -> list:
    """From an .out file with structure <anytext_JOBID.out>, return JOBID."""
    return out.split("_")[-1].replace('.out', '')


def get_mems(infos:dict, unit='MB') -> list:
    """From dict(infos) [val = seff output], return list of mem in MB.
    
    fix: add in other mem units"""
    mems = []
    for key,info in infos.items():
        if 'running' in info[3].lower() or 'pending' in info[3].lower():
            continue
        info[-2] = info[-2].split("(estimated")[0]
        mem, units = info[-2].split()[-2:]
        if units == 'GB':
            mem = float(mem)*1024
        elif units == 'EB':
            pass
        else:
            try:
                assert units == 'MB'
            except AssertionError:
                print('info = ', info)
            mem = float(mem)
        if unit == 'GB':
            mem = mem/1024
        mems.append(mem)
    return mems
