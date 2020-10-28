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
        infos[out] = Seff(getpid(out))
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
    pids = [q.pid() for q in getsq() if q.pid() != '']
    if len(pids) != len(list(set(pids))):
        print('len !- luni pids')
    return pids


def getjobs(user=os.environ['USER']) -> list:
    """From squeue -u $USER, return list of job names, alert if len != unique."""
    jobs = [q.job() for q in getsq() if q.job() != '']
    if len(jobs) != len(list(set(jobs))):
        print('len != luni jobs')
    return jobs


def qaccounts(pd=False, user=os.environ['USER']) -> list:
    """From squeue -u $USER, return list of billing accounts."""
    if pd == False:
        accounts = [q.account() for q in getsq() if q.account() != '']
    else:
        accounts = [q.account() for q in getsq(states=['PD']) if q.account() != '']
    return accounts


class Seff():
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
        """Get return from seff command."""
        info = os.popen('seff %s' % str(slurm_job_id)).read().split("\n")
        info.remove('')
        self.info = info
        self.info[-2] = self.info[-2].split("(estimated")[0]
        self.slurm_job_id = str(slurm_job_id)

    def __repr__(self):
        return repr(self.info)

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
        elif units == 'KB':
            mem = float(mem)/1024
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
    pass


class SQInfo():
    """Convert each line returned from `squeue -u $USER`.
    
    Assumes
    -------
    export SQUEUE_FORMAT="%i %u %a %j %t %S %L %D %C %b %m %N (%r)"
    # JOBID USER ACCOUNT NAME ST START_TIME TIME_LEFT NODES CPUS TRES_PER_NODE MIN_MEMORY NODELIST (REASON)
    #   %i   %u    %a     %j  %t     %S        %L      %D    %C       %b           %m        %N      (%r)
    
    Example jobinfo
    ---------------
    ('38768536',0
     'lindb',1
     'def-jonmee_cpu',2
     'batch_0583',3
     'PD',4
     'N/A',5
     '2-00:00:00',6
     '1',7
     '48',8
     'N/A',9
     '50M',0
     '(Priority)')
    """
    def __init__(self, jobinfo):
        self.info = list(jobinfo)
        pass
    
    def __repr__(self):
        return repr(self.info)
    
    def pid(self):
        return self.info[0]
    
    def user(self):
        return self.info[1]
    
    def account(self):
        return self.info[2]
    
    def job(self):
        return self.info[3]
    
    def state(self):
        return self.info[4]
    
    def start(self):
        """Job start time."""
        return self.info[5]
    
    def time(self):
        """Remaining time."""
        return self.info[6]
    
    def nodes(self):
        return self.info[7]
    
    def cpus(self):
        return self.info[8]
    
    def mem(self, units='MB'):
        memory = self.info[10]
        if all([memory.endswith('G') is False, memory.endswith('M') is False]):
            print('Unexpected units found in memory: ', memory)
            raise AssertionError
        if memory.endswith('G') and units=='MB':
            memory = memory.replace("G","")
            memory = int(memory) * 1024
        elif memory.endswith('M') and units=='GB':
            memory = memory.replace("M","")
            memory = int(memory) / 1024
        else:
            memory = int(memory.replace('M',''))
        return memory
    
    def status(self):
        return self.info[11]
    
    def reason(self):
        return self.status()
sqinfo = SQInfo


def getsq(grepping=None, states=[], balancing=False, user=None):
    """
    Get jobs from `squeue` slurm command matching criteria.

    Assumes
    -------
    export SQUEUE_FORMAT="%.8i %.8u %.15a %.68j %.3t %16S %.10L %.5D %.4C %.6b %.7m %N (%r)"

    Positional arguments
    --------------------
    grepping - list of key words to look for in each column of job info
    states - list of states {pending, running} wanted in squeue jobs
    balancing - bool: True if using to balance priority jobs, else for other queue queries
    user - user name to use to query `squeue -u {user}`

    Returns
    -------
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
                    grepped.append(sqinfo(tuple(splits)))

        if len(grepped) > 0:
            return grepped
    return getsq_exit(balancing)


def adjustjob(acct, jobid):
    """Move job from one account to another."""
    subprocess.Popen([shutil.which('scontrol'),
                      'update',
                      'Account=%s_cpu' % acct,
                      'JobId=%s' % str(jobid)])
    pass


class Squeue():
    """dict-like container class for holding and updating slurm squeue information.
    
    todo
    ----
    TODO: describe kwargs
    TODO: pass .cancel() and .update() a list of pids
    TODO: add balance method
    TODO: update_job needs to handle skipping over jobs that started running after class instantiation
    TODO: allow onaccount to accept list of accounts
    TODO: handle running/missing in _update_job()
    TODO: make it so it can return the queue for `grepping` without needing `user`
    
    
    kwargs
    ------
    
    
    Methods
    -------
    cancel - cancel entire queue or specific jobs for specific accounts
    update - update time, memory, or account for specifc jobs for specific accounts
    
    
    Examples
    --------
    Get squeue information (of class SQInfo) for each job in queue containing 'batch' or 'gatk'.
    
    >>> sq = Squeue(grepping=['batch', 'gatk'])
    
    
    Get queue with "batch" in one of the columns (eg the NAME col).
    For theses jobs, only update jobs with "batch_001" for mem and time.
    
    >>> sq = Squeue(grepping='batch', states='PD')
    >>> sq.update(match='batch_001', minmemorynode=1000, timelimit=3-00:00:00)
    
    
    Cancel all jobs in queue.
    
    >>> sq = Squeue()
    >>> sq.cancel(user=True)
    # OR:
    >>> Squeue().cancel(user=True)
    
    
    Cancel jobs in queue with "batch" in one of the columns (eg the NAME col).
    For theses jobs only cancel job names containing "batch_001" and 3 day run time,
        but not the 'batch_0010' job.
    
    >>> sq = Squeue(grepping='batch', states='PD')
    >>> sq.cancel(match=['batch_001', '3-00:00:00'], exclude='batch_0010')
    
    
    Get jobs from a specific user.
    
    >>> sq = Squeue(user='some_user')
    """
    # export SQUEUE_FORMAT
    # (JOBID USER ACCOUNT NAME ST START_TIME TIME_LEFT NODES CPUS TRES_PER_NODE MIN_MEMORY NODELIST (REASON))
    #    %i   %u    %a     %j  %t     %S        %L      %D    %C         %b        %m         %N      (%r)
    os.environ['SQUEUE_FORMAT'] = "%i %u %a %j %t %S %L %D %C %b %m %N (%r)"

    def __repr__(self):
        return repr(self.sq)

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

    def keys(self):
        return list(self.sq.keys())
    
    def items(self):
        return self.sq.items()

    def __contains__(self, item):
        return True if item in self.keys() else False

    def __init__(self, **kwargs):
        self.sq = Squeue._getsq(**kwargs)
        self.kwargs = kwargs

    def _getsq(grepping=None, states=[], user=None):
        """Get and parse slurm queue according to kwargs criteria."""
        def _checksq(sq):
            """Make sure queue slurm command worked. Sometimes it doesn't.

            Positional arguments:
            sq - list of squeue slurm command jobs, each line is str.split()
            """
            exitneeded = False
            if not isinstance(sq, list):
                print("\ttype(sq) != list, exiting %(thisfile)s" % globals())
                exitneeded = True
            for s in sq:
                if 'socket' in s.lower():
                    print("\tsocket in sq return, exiting %(thisfile)s" % globals())
                    exitneeded = True
                if not int(s.split()[0]) == float(s.split()[0]):  # if the jobid isn't a float:
                    print("\tcould not assert int == float, %s" % (s[0]))
                    exitneeded = True
            if exitneeded is True:
                print('\tslurm screwed something up for %(thisfile)s, lame' % globals())
                return None
            else:
                return sq

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
        if any(['running' in states,
                'R' in states,
                'r' in states]):
            cmd.extend(['-t', 'RUNNING'])
        if any(['pending' in states,
                'PD' in states,
                'pd' in states]):
            cmd.extend(['-t', 'PD'])

        # execute command
        found = 0 
        while found < 5:
            try:
                sqout = subprocess.check_output(cmd).decode('utf-8').split('\n')
                found = 10
            except subprocess.CalledProcessError as e:
                found += 1
                pass
        if found != 10:
            print(ColorText('FAIL: Exceeded five subprocess.CalledProcessError errors.').fail.bold())
            return []

        sq = [s for s in sqout if s != '']
        _checksq(sq)  # make sure slurm gave me something useful

        # look for the things I want to grep
        grepped = {}
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
                        info = SQInfo(splits)
                        grepped[info.pid()] = info

            if len(grepped) > 0:
                return grepped
        return getsq_exit(False)

    def _handle_mem(mem):
        """If the memory units are specified, remove."""
        mem = str(mem)
        if mem.endswith('M') or mem.endswith('G'):
            mem = mem[:-1]
        return mem

    def _handle_account(account):
        """Make sure account name is specified correctly."""
        if isinstance(account, str):
            if not account.endswith('_cpu'):
                account = f'{account}_cpu'
        else:
            for i,acct in enumerate(account):
                if not acct.endswith('_cpu'):
                    account[i] = f'{acct}_cpu'
        return account    

    def _handle_clock(clock):
        """Create aesthetic clock."""
        *days, clock = clock.split('-')
        clock = ':'.join([digits.zfill(2) for digits in clock.split(":")])
        return f'{days[0]}-{clock}' if len(days) > 0 else clock

    def _update_queue(self, cmd, desc, user=False, **kwargs):
        """Update jobs in queue and job info in Squeue class object."""
        from tqdm import tqdm as pbar
        def _update_self(info, **kwargs):
            """After successfully updating jobid with scontrol update, update Squeue class object."""
            if 'account' in kwargs:
                info[2] = Squeue._handle_account(kwargs['account'])
            if 'minmemorynode' in kwargs:
                info[10] = Squeue._handle_mem(kwargs['minmemorynode']) + 'M'
            if 'timelimit' in kwargs:
                info[6] = Squeue._handle_clock(kwargs['timelimit'])
            return info

        def _subset(sq, match=None, exclude=None, onaccount=None, **kwargs):
            """Return jobs in Squeue class according to kwarg criteria.
            Assumes first element of list is representative of entire list.
            """
            def _pid_or_job(match, exclude):
                """Determine if matching on pid or jobname."""
                def _ispid(flag):
                    """Determine if flag is representing jobIDs or jobnames."""
                    if isinstance(flag, list):
                        # no need to except TypeError here, list should not contain None
                        try:
                            assert float(flag[0]) == int(flag[0])
                            ispid = True
                        except ValueError:
                            # trying to float an alphanumeric string
                            ispid = False
                    else:
                        try:
                            assert float(flag) == int(flag)
                            ispid = True
                        except ValueError:
                            # trying to float an alphanumeric string
                            ispid = False
                        except TypeError:
                            assert flag is None
                            ispid = None
                    return ispid
                # first decide about match
                match_ispid = _ispid(match)
                # next decide about exclude
                exclude_ispid = _ispid(exclude)
                
                if sum([match_ispid is not None, exclude_ispid is not None]) == 2:
                    # if both are not none, make sure they are the same
                    try:
                        # make sure they are both pids or not pids
                        assert match_ispid == exclude_ispid
                    except AssertionError as e:
                        text = f'FAIL: match and exclude must both be either pids or job names, not a mixture.'
                        print(ColorText(text).fail().bold())
                        raise e
                # return True or False if one of the flags is not None, otherwise return None
                return match_ispid if match_ispid is not None else exclude_ispid
            
            def _handle_match(query, match_in):
                """Determine if query is the same as, or is in, match_in."""
                matches = False
                if isinstance(match_in, list):
                    if query in match_in:
                        matches = True
                else:
                    assert isinstance(match_in, str)
                    if query == match_in:
                        matches = True
                return matches
                                       
            _sq = dict(sq)
            # if iteration is not necessary, return input sq
            if any([match is not None, exclude is not None, onaccount is not None]):
                match_onpid = _pid_or_job(match, exclude)
                # determine which jobs I want to update:
                for pid,q in sq.items():
                    # if this job's account is not in the accounts I want, remove it
                    if onaccount is not None:
                        if _handle_match(q.account(), Squeue._handle_account(onaccount)) is False:
                            _sq.pop(pid)
                            continue                    
                    # determine what to query
                    if match_onpid is True:
                        query = pid
                    else:
                        query = q.job()                    
                    # if this job does not match the jobs I want, remove it
                    if match is not None:
                        if _handle_match(query, match) is False:
                            _sq.pop(q.pid())
                            continue
                    # if this job is one of the jobs I want to exclude, remove it
                    if exclude is not None:
                        if _handle_match(query, exclude):
                            _sq.pop(q.pid())
            return _sq

        def _update_job(cmd, job, jobid):
            """Execute 'scontrol update' or 'scancel' command for jobid."""
            # add jobid to command
            cmd.append(f'jobid={jobid}' if 'scancel' not in cmd else jobid)

            failcount = 0
            # try to execute commands three times before giving up (slurm is a jerk sometimes)
            while failcount < 3:
                try:
                    out = subprocess.check_output(cmd).decode('utf-8').replace("\n", "").split()
                    return True
                except subprocess.CalledProcessError as e:
                    # see if job is running or still in queue
                    jobq = Squeue._getsq(grepping=jobid)
                    if len(jobq) == 0:
                        return 'missing'
                    elif jobq[jobid].state() == 'R':
                        return ('running', jobq[jobid])
                    # otherwise count as failure
                    failcount += 1
            print(ColorText(f'FAIL: Update failed for cmd: {job} {jobid}').fail().bold())
            print(ColorText('FAIL: Exiting iteration.').bold().fail())
            return False
        
        if user is False:  # scontrol commands
            cmd = cmd.split()
        else:  # cancel all jobs
            return _update_job(['scancel', '-u', os.environ['USER']], None, None)
        # get subset of jobs returned from __init__()
        sq = _subset(self, **kwargs)
        # update each of the jobs
        if len(sq) > 0:
            for q in pbar(list(sq.values()), desc=desc):
                # if the job is updated successfully
                updated_result = _update_job(cmd, q.job(), q.pid())
                if updated_result is True:
                    if 'scancel' in cmd:
                        # remove job from Squeue container
                        self.__delitem__(q.pid())
                    else:
                        # update job info in Squeue container
                        self[q.pid()].info = _update_self(q.info, **kwargs)
                elif updated_result == 'running':
                    self[q.pid()].info = updated_result[1]
                elif updated_result == 'missing':
                    self.__delitem__(q.pid())
                else:
                    break
        else:
            print(ColorText('None of the jobs in Squeue class passed criteria.').warn())

    def cancel(self, **kwargs):
        """Cancel jobs in slurm queue, remove job info from Squeue class."""
        # cancel all jobs if user is True
#         self._update_queue(self.sq, cmd='scancel', desc='scancel', **kwargs)
        self._update_queue(cmd='scancel', desc='scancel', **kwargs)
        
        pass

    def update(self, **kwargs):
        """Update jobs in slurm queue and job info in Squeue class."""
        def _cmd(account=None, minmemorynode=None, timelimit=None, **kwargs):
            """Create bash command for slurm scontrol update."""
            # base command
            cmd = 'scontrol update'
            # update cmd by appending flags
            if account is not None:
                account = Squeue._handle_account(account)
                cmd = f'{cmd} Account={account}'
            if minmemorynode is not None:
                minmemorynode = Squeue._handle_mem(minmemorynode)
                cmd = f'{cmd} MinMemoryNode={minmemorynode}'
            if timelimit is not None:
                timelimit = Squeue._handle_clock(timelimit)
                cmd = f'{cmd} TimeLimit={timelimit}'
            return cmd
        
        # get scontrol update command
        cmd = _cmd(**kwargs)
        
        # update each of the jobs
        Squeue._update_queue(self.sq, cmd, 'update', **kwargs)
        
        pass
