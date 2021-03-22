"""
Distribute priority jobs among accounts. To distribute non-priority jobs see myslurm.Squeue.balance.

###
# purpose: evenly redistributes jobs across available slurm accounts. Jobs are
#          found via searching for the keyword among the squeue output fields;
#          Helps speed up effective run time by spreading out the load.
###

###
# usage: python balance_queue.py [keyword] [parentdir]
#
# keyword is used to search across column data in queue
# parentdir is used to either find a previously saved list of accounts
#    or is set to 'choose' so the user can run from command line
#    and manually choose which accounts are used
#
# accounts are those slurm accounts with '_cpu' returned from the command:
#    sshare -U --user $USER --format=Account
#
# to manually balance queue using all available accounts:
#    python balance_queue.py
# to manually balance queue and choose among available accounts:
#    python balance_queue.py $USER choose
#    python balance_queue.py keyword choose
# as run in pipeline when balancing trim jobs from 01_trim.py:
#    this looks for accounts.pkl in parentdir to determine accounts saved in 00_start.py
#    python balance_queue.py trim /path/to/parentdir
#
# because of possible exit() commands in balance_queue, this should be run
#    as a main program, or as a subprocess when run inside another python
#    script.
###

### assumes
# export SQUEUE_FORMAT="%i %u %a %j %t %S %L %D %C %b %m %N (%r)"
###

# FUN FACTS
# 🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁
# balance_queue.py originated as part of the CoAdapTree project: github.com/CoAdapTree/varscan_pipeline
# 🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁🇨🇦🍁
"""

import os, shutil, sys, math, subprocess, time
from random import shuffle
from collections import Counter
import pythonimports as pyimp
import myslurm


def announceacctlens(accounts, fin, priority=True):
    """How many priority jobs does each account have?

    Positional arguments:
    accounts - dictionary with key = account_name, val = list of jobs (squeue output)
    fin - True if this is the final job announcement, otherwise the first announcement
    """
    print("\t%s job announcement" % ("final" if fin is True else "first"))
    if fin is True:
        time.sleep(1)
    for account in accounts:
        num_acct = len(accounts[account])
        status = "with Priority status " if priority is True else ""
        print(f"\t{num_acct} jobs {status}on {account}")
    pass


def adjustjob(acct, jobid):
    """Move job from one account to another."""
    subprocess.Popen([shutil.which("scontrol"),
                      "update",
                      "Account=%s_cpu" % acct,
                      "JobId=%s" % str(jobid)])
    pass


def getaccounts(sq, stage, user_accts):
    """
    Count the number of priority jobs assigned to each account.

    Positional arguments:
    sq - list of squeue slurm command jobs, each line is str.split()
       - slurm_job_id is zeroth element of str.split()
    stage - stage of pipeline, used as keyword to filter jobs in queue
    user_accts - list of slurm accounts to use in balancing
    """
    # get accounts with low priority
    accounts = {}
    for q in sq:
        pid = q[0]
        account = q[2].split("_")[0]
        if account not in accounts and account in user_accts:
            accounts[account] = {}
        accounts[account][pid] = q

    # if all user_accts have low priority, exit()
    if len(accounts.keys()) == len(user_accts) and stage != "final":
        print("\tall accounts have low priority, leaving queue as-is")
        early_exit_decision = True
    else:
        early_exit_decision = False
    if stage == "final":
        early_exit_decision = True
    return accounts, early_exit_decision


def getbalance(accounts, num):
    """Determine how many jobs should be given from one account to another.

    Positional arguments:
    accounts - dictionary with key = account_name, val = list of jobs (squeue output)
    num - number of accounts to balance among (this needs to be changed to object not number)
    """
    sums = 0
    for account in accounts:
        sums += len(accounts[account].keys())
    bal = math.ceil(sums / num)
    return bal


def choose_accounts(accts):
    print(
        pyimp.ColorText(
            "\nDetermining which slurm accounts are available for use by balance_queue.py"
        ).bold()
    )
    if len(accts) > 1:
        keep = []
        for acct in accts:
            while True:
                inp = input(pyimp.ColorText(
                    f"\tINPUT NEEDED: Do you want to use this account: {acct}? (yes | no): ").warn()
                            ).lower()
                if inp in ["yes", "no"]:
                    if inp == "yes":
                        print("\t\tkeeping %s" % acct)
                        keep.append(acct)
                    else:
                        print("\t\tignoring %s" % acct)
                    break
                else:
                    print(pyimp.ColorText("Please respond with 'yes' or 'no'").fail())
    else:
        # no need to ask if they want to use the only account that's available, duh
        keep = accts
    # make sure they've chosen at least one account
    while len(keep) == 0:
        print(pyimp.ColorText("FAIL: You need to specify at least one account. Revisiting accounts...").fail())
        keep = choose_accounts(accts)
    return keep


def get_avail_accounts(parentdir=None, save=False):
    """Query slurm with sshare command to determine accounts available.

    If called with parentdir=None, return all available accounts.
        - Meant to be called from command line outside of pipeline. See also sys.argv input.
    If called with parentdir='choose', allow user to choose accounts.
        - Meant to be called from command line outside of pipeline. See also sys.argv input.
    If called with save=True, confirm each account with user and save .pkl file in parentdir.
        - save=True is only called from 00_start.py

    Returns a list of accounts to balance queue.
    """

    if parentdir is not None and save is False:
        # if the accounts have already been chosen, just return them right away
        # keep 'save is False' so 00_start can overwrite previous pkl and skip here
        pkl = os.path.join(parentdir, "accounts.pkl")
        if os.path.exists(pkl):
            return pyimp.pklload(pkl)

    # get a list of all available accounts
    acctout = (subprocess.check_output([shutil.which("sshare"),
                                        "-U",
                                        "--user",
                                        os.environ["USER"],
                                        "--format=Account"]).decode("utf-8").split("\n"))
    accts = [acct.split()[0].split("_")[0] for acct in acctout if "_cpu" in acct]

    # for running outside of the pipeline:
    if parentdir is None:
        # to manually run on command line, using all accounts
        return accts
    elif parentdir == "choose":
        # to manually run on command line, choose accounts
        return choose_accounts(accts)

    # save if necessary
    if save is True:
        # called from 00_start.py
        keep = choose_accounts(accts)
        pyimp.pkldump(keep, os.path.join(parentdir, "accounts.pkl"))
        # no return necessary for 00_start.py
        return

    return accts


def redistribute_jobs(accts, user_accts, balance):
    """Redistribute priority jobs to other accounts without high priority.

    Positional arguments:
    accts - dict: key = account, value = dict with key = pid, value = squeue output
    user_accts - list of all available slurm accounts
    balance  - int; ceiling number of jobs each account should have after balancing
    """

    # which jobs can be moved, which accounts need jobs?
    moveable = []
    takers = []
    for account in user_accts:
        if account not in accts:  # if account has zero priority jobs
            takers.append(account)
        else:
            pids = list(accts[account].keys())
            # if account has excess, give away jobs
            if len(pids) > balance:
                numtomove = len(pids) - balance
                print("\t%s is giving up %s jobs" % (account, numtomove))
                moveable.extend(pids[-numtomove:])  # move newest jobs, hopefully old will schedule
            # keep track of accounts that need jobs
            elif len(pids) < balance:
                takers.append(account)
            elif len(pids) == 1 and balance == 1 and len(accts.keys()) < len(user_accts):
                # if numjobs and balance == 1 but not all accounts have low priority, give up the job
                moveable.append(pids[0])

    # shuffle list(takers) to avoid passing only to accounts that appear early in the list
    shuffle(takers)
    # redistribute jobs
    taken = Counter()
    while len(moveable) > 0:
        for taker in takers:
            # determine numtotake
            if taker not in accts:
                pids = []
            else:
                pids = accts[taker].keys()
            numtotake = balance - len(pids)
            if balance == 1 and len(pids) == 1:
                numtotake = 1
            # give numtotake to taker
            for pid in moveable[-numtotake:]:
                adjustjob(taker, pid)
                taken[taker] += 1  # needs to be above .remove because of while()
                moveable.remove(pid)
    for taker, count in taken.items():
        print("\t%s has taken %s jobs" % (taker, count))
    pass


def main(keyword, parentdir):
    print(pyimp.ColorText("\nStarting balance_queue.py").bold())
    # get accounts available for billing
    user_accts = get_avail_accounts(parentdir)

    # if only one account, skip balancing
    if len(user_accts) == 1:
        print("\tthere is only one account (%s), no more accounts to balance queue." % user_accts[0])
        print("\texiting balance_queue.py")
        exit()

    # get priority jobs from the queue
    sq = myslurm.Squeue(grepping=[keyword, "Priority"])
    if sq is None or len(sq) == 0:
        print("\texiting balance_queue.py")
        sys.exit(0)

    # get per-account lists of jobs in Priority pending status, exit if all accounts have low priority
    accts, early_exit_decision = getaccounts(sq, "", user_accts)
    announceacctlens(accts, early_exit_decision)  # TODO: announce all accounts, not just accts with priority jobs
    if early_exit_decision is True:
        sys.exit(0)

    # determine number of jobs to redistribute to each account
    balance = getbalance(accts, len(user_accts))

    # redistribute
    redistribute_jobs(accts, user_accts, balance)

    # announce final job counts
    announceacctlens(*getaccounts(myslurm.Squeue(grepping=[keyword, "Priority"]),
                                  "final",
                                  user_accts))
    pass


if __name__ == "__main__":
    # args
    if len(sys.argv) == 1:
        # so I can run from command line and balance full queue
        keyword = os.environ["USER"]
        parentdir = None
    elif len(sys.argv) == 2:
        # so I can run from command line without a parentdir
        thisfile, keyword = sys.argv
        parentdir = None
    else:
        thisfile, keyword, parentdir = sys.argv

    main(keyword, parentdir)
