import os


def get_mems(infos:dict, unit='MB') -> list:
    mems = []
    for key,info in infos.items():
        if 'running' in info[3].lower() or 'pending' in info[3].lower():
            continue
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


def clock_hrs(clock:str) -> float:
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
    times = []
    for key, info in infos.items():
        if 'running' in info[3].lower() or 'pending' in info[3].lower():
            continue
        clock = info[-3].split()[-1]
        hrs = clock_hrs(clock)
        times.append(hrs)
    return times


def sbatch(files):
    if not type(files) == list:
        files = [files]
    for f in files:
        os.chdir(op.dirname(f))
        os.system('sbatch %s' % f)


def getpids():
    pids = os.popen('squeue -u lindb -o "%i"').read().split("\n")
    pids = [p for p in pids if not p == '']
    if len(pids) != luni(pids):
        print('len !- luni pids')
    return pids[1:]


def getjobs(user='lindb'):
    jobs = os.popen(f'squeue -u {user} -o "%j"').read().split("\n")
    jobs = [j for j in jobs if not j == '']
    if len(jobs) != luni(jobs):
        print('len != luni jobs')
    return jobs[1:]


def getaccounts(pd=False, user='lindb'):
    if pd == False:
        accounts = os.popen(f'squeue -u {user} -o "%a"').read().split("\n")
    else:
        accounts = os.popen(f'squeue -u {user} -t "pd" -o "%a"').read().split('\n')
    accounts = [a for a in accounts if not a in ['', 'ACCOUNT']]
    return accounts


def seff(pid):
    lst = os.popen('seff %s' % pid).read().split("\n")
    lst.remove('')
    return lst


def getpid(out):
    return out.split("_")[-1].replace('.out', '')


def get_mems(infos:dict, unit='MB') -> list:
    mems = []
    for key,info in infos.items():
        if 'running' in info[3].lower() or 'pending' in info[3].lower():
            continue
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


def clock_hrs(clock:str) -> float:
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
    times = []
    for key, info in infos.items():
        if 'running' in info[3].lower() or 'pending' in info[3].lower():
            continue
        clock = info[-3].split()[-1]
        hrs = clock_hrs(clock)
        times.append(hrs)
    return times
