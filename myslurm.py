"""Python commands to interface with slurm queue and seff commands."""


import os
import time
import subprocess
import shutil
from tqdm import tqdm as pbar
import matplotlib.pyplot as plt
from os import path as op
from collections import defaultdict, Counter
from tqdm import tqdm as pbar
from typing import Union

import pythonimports as pyimp


def get_seff(outs: list, desc=None):
    """From a list of .out files (ending in f'_{SLURM_JOB_ID}.out'), get seff output."""
    infos = {}
    for out in pbar(outs, desc=desc):
        infos[out] = Seff(getpid(out))
    return infos


def get_mems(infos: dict, units="MB", plot=True) -> list:
    """From output by `get_seff()`, extract mem in `units` units; histogram if `plot` is True.

    Parameters
    ----------
    - infos : dict of any key with values of class Seff
    - units : passed to Seff._convert_mem(). options: GB, MB, KB

    """
    mems = []
    for key, info in infos.items():
        if "running" in info.state().lower() or "pending" in info.state().lower():
            continue
        mems.append(info.mem(units=units, per_core=False))

    if plot is True:
        plt.hist(mems)
        plt.xlabel(units)
        plt.show()

    return mems


def clock_hrs(clock: str, unit="hrs") -> float:
    """From a clock (days-hrs:min:sec) extract hrs or days as float."""
    assert unit in ["hrs", "days"]
    hrs = 0
    if "-" in clock:
        days, clock = clock.split("-")
        hrs += 24 * float(days)
    h, m, s = clock.split(":")
    hrs += float(h)
    hrs += float(m) / 60
    hrs += float(s) / 3600
    if unit == "days":
        hrs = hrs / 24
    return hrs


def get_times(infos: dict, unit="hrs", plot=True) -> list:
    """From dict(infos) [val = seff output], get times in hours.

    fix: add in other clock units"""
    times = []
    for key, info in infos.items():
        if "running" in info.state().lower() or "pending" in info.state().lower():
            continue
        hrs = info.walltime(unit=unit)
        times.append(hrs)

    if plot is True:
        plt.hist(times)
        plt.xlabel(unit.title() if unit == "days" else "Hours")
        plt.show()
    return times


def sbatch(shfiles: Union[str, list], sleep=0, printing=False) -> list:
    """From a list of .sh shfiles, sbatch them and return associated jobid in a list.

    Notes
    -----
    - assumes that the job name that appears in the queue is the basename of the .sh file
        - eg for job_187.sh, the job name is job_187
        - this convention is used to make sure a job isn't submitted twice
    """

    if isinstance(shfiles, list) is False:
        assert isinstance(shfiles, str)
        shfiles = [shfiles]
    pids = []
    for sh in pbar(shfiles):
        job = op.basename(sh).split(".")[0]
        os.chdir(os.path.dirname(sh))
        # try and sbatch file 10 times before giving up
        failcount = 0
        sbatched = False
        while sbatched is False:
            try:
                pid = (
                    subprocess.check_output([shutil.which("sbatch"), sh])
                    .decode("utf-8")
                    .replace("\n", "")
                    .split()[-1]
                )
                sbatched = True
            except subprocess.CalledProcessError as e:
                failcount += 1
                if failcount == 10:
                    print("!!!REACHED FAILCOUNT LIMIT OF 10!!!")
                    return pids
            # one more failsafe to ensure a job isn't submitted twice
            sq = Squeue()
            jobs = defaultdict(list)
            for pid, q in sq.items():
                jobs[q.job()].append(q.pid())
            if job in list(jobs.keys()):
                sbatched = True
                pid = jobs[job]
                if len(jobs[job]) > 1:
                    sq.cancel(match=pid)
                break
        if printing is True:
            print("sbatched %s" % sh)
        pids.append(pid)
        time.sleep(sleep)
        del pid
    return pids


def getpids(user=os.environ["USER"]) -> list:
    """From squeue -u $USER, return list of queue."""
    pids = Squeue(user=user).pids()
    if len(pids) != len(list(set(pids))):
        print("len !- luni pids")
    return pids


def getjobs(user=os.environ["USER"]) -> list:
    """From squeue -u $USER, return list of job names, alert if len != unique."""
    jobs = Squeue(user=user).jobs()
    if len(jobs) != len(list(set(jobs))):
        print("len != luni jobs")
    return jobs


def qaccounts(pd=False, user=os.environ["USER"]) -> list:
    """From squeue -u $USER, return list of billing accounts."""
    if pd is False:
        accounts = Squeue(user=user).accounts()
    else:
        accounts = Squeue(states=["PD"], user=user).accounts()
    return accounts


class Seff:
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
        info = os.popen("seff %s" % str(slurm_job_id)).read().split("\n")
        info.remove("")
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

    def cpu_u(self, unit="clock") -> str:
        """Get CPU time utilized by job (actual time CPUs were active across all cores)."""
        utilized = self.info[6].split()[-1]
        if unit != "clock":
            utilized = clock_hrs(utilized, unit)
        return utilized

    def cpu_e(self) -> str:
        """Get CPU efficiency (cpu_u() / core_walltime())"""
        return self.info[7].split()[2]

    def core_walltime(self, unit="clock") -> str:
        """Get time that CPUs were active (across all cores)."""
        walltime = self.info[7].split()[-2]
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
        mem, mem_units = self.info[-1].split()[-2:]
        return self._convert_mem(mem, mem_units, units)

    def mem_e(self) -> str:
        """Get the memory efficiency (~ mem / mem_req)"""
        return self.info[-1].split()[2]

    def _convert_mem(self, mem, mem_units, units="MB"):
        """Convert between memory mem_units."""
        # first convert reported mem to MB
        if mem_units == "GB":
            mem = float(mem) * 1024
        elif mem_units == "KB":
            mem = float(mem) / 1024
        elif mem_units == "EB":
            mem = 0
        else:
            try:
                assert mem_units == "MB"
            except AssertionError as e:
                print("info = ", self.info)
                raise e
            mem = float(mem)
        # then convert to requested mem
        if units == "GB":
            mem /= 1024
        elif units == "KB":
            mem *= 1024
        return mem

    def mem(self, units="MB", per_core=False) -> str:
        """Get memory unitilized by job (across all cores, or per core)."""
        mem, mem_units = self.info[-2].split()[-2:]
        mem = self._convert_mem(mem, mem_units, units)
        if per_core is True:
            mem = mem / self.info[5].split()[-1]
        return mem

    pass


def getpid(out: str) -> str:
    """From an .out file with structure <anytext_JOBID.out>, return JOBID."""
    return out.split("_")[-1].replace(".out", "")


class SQInfo:
    """Convert each line returned from `squeue -u $USER`.

    Example jobinfo    (index number of list)
    ---------------
    ('38768536',       0
     'lindb',          1
     'def-jonmee_cpu', 2
     'batch_0583',     3
     'PD',             4
     'N/A',            5
     '2-00:00:00',     6
     '1',              7
     '48',             8
     'N/A',            9
     '50M',            10
     '(Priority)')     11
    """

    def __init__(self, jobinfo):
        self.info = list(jobinfo)
        pass

    def __repr__(self):
        return repr(self.info)

    def __iter__(self):
        return iter(self.info)

    def pid(self):
        """SLURM_JOB_ID."""
        return self.info[0]

    def user(self):
        return self.info[1]

    def account(self):
        return self.info[2]

    def job(self):
        """Job name."""
        return self.info[3]

    def state(self):
        """Job state - eg pending, closing, running, failed/completed + exit code."""
        return self.info[4]

    def start(self):
        """Job start time."""
        return self.info[5]

    def time(self):
        """Remaining time."""
        return self.info[6]

    def nodes(self):
        """Compute nodes."""
        return self.info[7]

    def cpus(self):
        return self.info[8]

    def mem(self, units="MB"):
        memory = self.info[10]
        if all([memory.endswith("G") is False, memory.endswith("M") is False]):
            print("Unexpected units found in memory: ", memory)
            raise AssertionError
        if memory.endswith("G") and units == "MB":
            memory = memory.replace("G", "")
            memory = int(memory) * 1024
        elif memory.endswith("M") and units == "GB":
            memory = memory.replace("M", "")
            memory = int(memory) / 1024
        else:
            memory = int(memory.replace("M", ""))
        return memory

    def status(self):
        return self.info[-1]

    def reason(self):
        return self.status()

    pass


sqinfo = SQInfo  # backwards compatibility


def adjustjob(acct, jobid):
    """Move job from one account to another."""
    acct = acct.replace("_cpu", "")
    subprocess.Popen(
        [shutil.which("scontrol"), "update", f"Account={acct}_cpu", f"JobId={jobid}"]
    )
    pass


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
    TODO: pass .cancel() and .update() a list of pids
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
    # (JOBID USER ACCOUNT NAME ST START_TIME TIME_LEFT NODES CPUS TRES_PER_NODE MIN_MEMORY NODELIST (REASON))
    #    %i   %u    %a     %j  %t     %S        %L      %D    %C         %b        %m         %N      (%r)
    os.environ["SQUEUE_FORMAT"] = "%i %u %a %j %t %S %L %D %C %b %m %N (%r)"

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

    def __contains__(self, pid):
        return True if pid in self.keys() else False

    def __init__(self, **kwargs):
        # get queue matching grepping
        self.sq = Squeue._getsq(**kwargs)
        # filter further with kwargs
        if len(self.sq) > 0:
            self.sq = Squeue._filter_jobs(self.sq, **kwargs)
            if len(self.sq) == 0:
                print("\tno jobs in queue matching query")
        pass

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
                    grepped[info.pid()] = info
        return grepped

    @staticmethod
    def _getsq(grepping=None, states=[], user=None, **kwargs):
        """Get and parse slurm queue according to kwargs criteria."""

        def _checksq(sq):
            """Make sure queue slurm command worked. Sometimes it doesn't.

            Positional arguments:
            sq - list of squeue slurm command jobs, each line is str.split()
            """
            exitneeded = False
            if not isinstance(sq, list):
                print("\ttype(sq) != list, exiting Squeue")
                exitneeded = True
            for s in sq:
                if "socket" in s.lower():
                    print("\tsocket in sq return, exiting Squeue")
                    exitneeded = True
                if not int(s.split()[0]) == float(
                    s.split()[0]
                ):  # if the jobid isn't a float:
                    print("\tcould not assert int == float, %s" % (s[0]))
                    exitneeded = True
            if exitneeded is True:
                print("\tslurm screwed something up for Squeue, lame")
                return None
            else:
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

        # execute command
        found = 0
        while found < 5:
            try:
                sqout = subprocess.check_output(cmd).decode("utf-8").split("\n")
                found = 10
            except subprocess.CalledProcessError:
                found += 1
                pass
        if found != 10:
            print(
                pyimp.ColorText(
                    "FAIL: Exceeded five subprocess.CalledProcessError errors."
                ).fail.bold()
            )
            return []

        sq = [s for s in sqout if s != ""]
        if _checksq(sq) is None:  # make sure slurm gave me something useful
            return None

        # look for the things I want to grep
        if len(sq) > 0:
            grepped = Squeue._grep_sq(sq, grepping)
            if len(grepped) > 0:
                return grepped
        print("\tno jobs in queue matching query")
        return []

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
    def _update_job(cmd, job, jobid):
        """Execute 'scontrol update' or 'scancel' command for jobid."""
        # add jobid to command
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
                jobq = Squeue(grepping=jobid)
                if len(jobq) == 0:
                    return "missing"
                elif jobq[jobid].state() == "R":
                    return "running", jobq[jobid]
                # otherwise count as failure
                failcount += 1
        print(
            pyimp.ColorText(f"FAIL: Update failed for cmd: {job} {jobid}").fail().bold()
        )
        return False

    @staticmethod
    def _filter_jobs(sq, grepping=None, exclude=None, onaccount=None, priority=None, states=None, **kwargs):
        """Filter jobs in `Squeue` class object.
        Parameters
        ----------
        grepping - a string or list of strings to use as queries to select job from queue (case insensitive)
            - strings are converted to a single-element list internally
            - to keep a job from the queue, all elements of `grepping` must be found in at least one of the squeue columns
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
                    for info in information:
                        for ex in exclude:
                            if ex.lower() in info.lower():
                                remove = True
                #                 if 'pd' not in information.state().lower():  # if a job isn't pending, it can't be updated
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
                assert isinstance(exclude, list), 'Squeue.balance() only expects `exclude` as `str` or `list`.'

        # set up grepping as list
        if grepping is None:
            grepping = []
        elif isinstance(grepping, str):
            grepping = [grepping]
        else:
            assert isinstance(grepping, list), 'Squeue.balance() only expects `grepping` as `str` or `list`.'

        # add wanted accounts to `grepping`
        if onaccount is not None:
            # add accounts to grepping
            if isinstance(onaccount, str):
                grepping.append(onaccount)
            elif isinstance(onaccount, list):
                grepping.extend(onaccount)
            else:
                raise Exception('Squeue.balance() only expects `onaccount` as `str` or `list`.')

        # determine whether to keep/ignore priority status jobs
        if priority is True:
            grepping.append('priority')
        elif priority is False:
            exclude.append('priority')

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
                    if state.lower() in info.state().lower():
                        keep = True
                        break
                if keep is False:
                    _sq.pop(pid)

        return _sq

    def _update_queue(self, cmd, desc, user=False, **kwargs):
        """Update jobs in queue and job info in Squeue class object."""

        if user is False:  # scontrol commands
            cmd = cmd.split()
        elif desc == "scancel" and user is True:  # cancel all jobs
            return Squeue._update_job(["scancel", "-u", os.environ["USER"]], None, None)
        # get subset of jobs returned from __init__()
        _sq = self._filter_jobs(self, **kwargs)
        # update each of the jobs
        if len(_sq) > 0:
            for q in pbar(list(_sq.values()), desc=desc):
                # if the job is updated successfully
                updated_result = Squeue._update_job(cmd, q.job(), q.pid())
                if updated_result is True:
                    if "scancel" in cmd:
                        # remove job from Squeue container
                        self.__delitem__(q.pid())
                    else:
                        # update job info in Squeue container
                        self[q.pid()].info = Squeue._update_self(q.info, **kwargs)
                elif updated_result == "running":
                    self[q.pid()].info = updated_result[1]
                elif updated_result == "missing":
                    self.__delitem__(q.pid())
                else:
                    assert (
                        updated_result is False
                    ), '`updated_result` must be one of {True, False, "missing", "running"}'
        else:
            print(
                pyimp.ColorText(
                    "None of the jobs in Squeue class passed criteria."
                ).warn()
            )
        pass

    @staticmethod
    def _save_default_accounts(save_dir=os.environ["HOME"]):
        """Among accounts available, choose which to use during balancing.

        The chosen accounts will be saved as op.join(save_dir, 'accounts.pkl'), and will be used
            to balance accounts in the future when setting `parentdir` in Squeue.balance to `save_dir`.
        """
        import balance_queue as balq

        balq.get_avail_accounts(parentdir=save_dir, save=True)

        pass

    def keys(self):
        return list(self.sq.keys())

    def values(self):
        return list(self.sq.values())

    def items(self):
        return self.sq.items()

    def running(self):
        return self._filter_jobs(self, states='R')

    def pending(self):
        return self._filter_jobs(self, states='PD')

    def states(self, **kwargs):
        """Get a list of job states."""
        _sq = self._filter_jobs(self, **kwargs)
        return [info.state() for q, info in _sq.items()]

    def pids(self, **kwargs):
        """Get a list of pids, subset with kwargs."""
        _sq = self._filter_jobs(self, **kwargs)
        return [info.pid() for q, info in _sq.items()]

    def jobs(self, **kwargs):
        """Get a list of job names, subset with kwargs."""
        _sq = self._filter_jobs(self, **kwargs)
        return [info.job() for q, info in _sq.items()]

    def accounts(self, **kwargs):
        """Get a list of accounts, subset with kwargs."""
        _sq = self._filter_jobs(self, **kwargs)
        return [info.account() for q, info in _sq.items()]

    def cancel(self, **kwargs):
        """Cancel jobs in slurm queue, remove job info from Squeue class."""
        # cancel all jobs if user is True
        self._update_queue(cmd="scancel", desc="scancel", **kwargs)

        pass

    def update(self, **kwargs):
        """Update jobs in slurm queue with scontrol, and update job info in Squeue class.

        kwargs - that control what can be updated (other kwargs go to Squeue._filter_jobs)
        ------
        account - the account to transfer jobs
        minmemorynode - total memory requested
        timelimit - total wall time requested
        """

        def _cmd(account=None, minmemorynode=None, timelimit=None, **kwargs):
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
            return cmd_

        # get scontrol update command
        cmd = _cmd(**kwargs)

        # update each of the jobs
        Squeue._update_queue(self, cmd, "update", **kwargs)

        pass

    def balance(self, parentdir=os.environ["HOME"], **kwargs):
        """Evenly distribute pending jobs across available slurm sbatch accounts.

        Parameters
        ----------
        parentdir - used in balance_queue to look for `accounts.pkl`; see balance_queue.__doc__
            - `parentdir` can be set to 'choose' to manually choose slurm accounts from those available
            - if `parentdir` is set to `None`, then all available accounts will be used to balance
            - otherwise, `parentdir` can be set to `some_directory` that contains "accounts.pkl" saved from:
                balance_queue.get_avail_accounts(some_directory, save=True)
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
        import balance_queue as balq

        os.environ["SQUEUE_FORMAT"] = "%i %u %a %j %t %S %L %D %C %b %m %N (%r)"

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
            if info.state() != "PD":
                _sq.pop(pid)
        if len(_sq) == 0 or _sq is None:
            print("\tno pending jobs in queue matching query")
            return

        # get accounts available for billing
        user_accts = balq.get_avail_accounts(parentdir)

        if len(user_accts) > 1:
            # get per-account lists of jobs in pending status, return if all accounts have jobs (no need to balance)
            accts, early_exit_decision = balq.getaccounts(
                [q.info for q in _sq.values() if "pd" in q.state().lower()],
                "",
                user_accts,
            )
            balq.announceacctlens(accts, early_exit_decision, priority=priority)
            if early_exit_decision is True:
                return

            # determine number of jobs to redistribute to each account
            balance = balq.getbalance(accts, len(user_accts))

            # redistribute
            balq.redistribute_jobs(accts, user_accts, balance)

            # announce final job counts
            time.sleep(
                2
            )  # give system a little time to update (sometimes it can print original job counts)
            if "priority" in kwargs:
                # if the job no longer has same priority status, it won't be found in new queue query
                kwargs.pop("priority")
            if "onaccount" in kwargs:
                # if the job is no longer on the queried account, it won't be found in new queue query
                kwargs.pop("onaccount")
            sq = self._filter_jobs(Squeue(), **kwargs)  # re-query the queue, filter
            balq.announceacctlens(
                *balq.getaccounts(
                    [q.info for q in sq.values() if "pd" in q.state().lower()],
                    "final",
                    user_accts,
                ),
                priority=priority,
            )  # print updated counts
            # update self
            for pid, q in _sq.items():
                account = sq[pid].account()
                self[q.pid()].info = Squeue._update_self(q.info, account=account)
        else:
            print(
                "\tthere is only one account (%s), no more accounts to balance queue."
                % user_accts[0]
            )

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
            stats[q.account()][q.status()] += 1
            stats[q.account()][q.state()] += 1
            statuses[q.status()] += 1
            states[q.state()] += 1
            account_counts[q.account()] += 1

        # print account stats
        print(
            pyimp.ColorText(
                "There are %s accounts with jobs matching search criteria."
                % len(stats.keys())
            ).bold()
        )
        for account, acct_stats in stats.items():
            print(
                "\t",
                account,
                "has",
                account_counts[account],
                "total jobs, which include:",
            )
            for stat, count in acct_stats.items():
                print(
                    "\t\t",
                    count,
                    "jobs with",
                    "%s" % "state =" if stat in ["R", "PD"] else "status =",
                    stat,
                )
        # print status counts
        print(
            pyimp.ColorText(
                "\nIn total, there are %s jobs:" % sum(statuses.values())
            ).bold()
        )
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
            if "pd" in q.state().lower():  # only pending jobs can be held
                updated_result = Squeue._update_job(
                    [shutil.which("scontrol"), "hold"], q.job(), pid
                )
                if updated_result is True:
                    # update job info in Squeue container
                    self[q.pid()].info = Squeue._update_self(q.info, **kwargs)
                elif updated_result == "running":
                    self[q.pid()].info = updated_result[1]
                elif updated_result == "missing":
                    self.__delitem__(q.pid())
        pass

    def release(self, **kwargs):
        """Release held jobs. Parameters described in `Squeue._update_job.__doc__`."""
        _sq = self._filter_jobs(self, **kwargs)

        released = 0
        for pid, q in pbar(_sq.items()):
            if "held" in q.status().lower():  # JobHeldUser
                updated_result = Squeue._update_job(
                    [shutil.which("scontrol"), "release"], q.job(), pid
                )
                if updated_result is True:
                    # update job info in Squeue container
                    self[q.pid()].info = Squeue._update_self(q.info, **kwargs)
                elif updated_result == "running":
                    self[q.pid()].info = updated_result[1]
                elif updated_result == "missing":
                    self.__delitem__(q.pid())
                released += 1
        print(pyimp.ColorText(f"\tReleased {released} jobs").gray())
        pass

    pass


getsq = Squeue._getsq  # backwards compatibility


def create_watcherfile(
    pids, directory, watcher_name="watcher", email="brandon.lind@ubc.ca"
):
    """From a list of dependency pids, sbatch a file that will email once pids are completed.

    TODO
    ----
    - incorporate code to save mem and time info
    """
    watcherfile = op.join(directory, f"{watcher_name}.sh")
    jpids = ",".join(pids)
    text = f"""#!/bin/bash
#SBATCH --job-name={watcher_name}
#SBATCH --time=0:00:01
#SBATCH --ntasks=1
#SBATCH --mem=25
#SBATCH --output=watcher_%j.out
#SBATCH --dependency=afterok:{jpids}
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
"""

    with open(watcherfile, "w") as o:
        o.write(text)

    print(sbatch(watcherfile))

    return watcherfile
