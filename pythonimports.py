import os
import sys
import time
import math
import pickle
import random
import string
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from myslurm import *
from typing import Optional, Union
from IPython.display import clear_output
from collections import OrderedDict, Counter
from IPython.display import Markdown, display
from os import listdir
from os import path as op
from os import chdir as cd
from decimal import Decimal
from tqdm import tqdm as nb
from os import getcwd as cwd
from shutil import copy as cp
from shutil import move as mv
from ipyparallel import Client
from datetime import timedelta
from matplotlib import pyplot as pl
from datetime import datetime as dt
from tqdm import tqdm_notebook as tnb
from matplotlib.backends.backend_pdf import PdfPages


pd.set_option('display.max_columns', 100)


def ls(DIR:str) -> list:
    """Get a list of file basenames from DIR."""
    return sorted(listdir(DIR))


def fs(DIR:str, pattern='', endswith='', startswith='', exclude=None, dirs=None, bnames=False) -> list:
    """Get a list of full path names for files and/or directories in a DIR.

    pattern - pattern that file/dir basename must have to keep in return
    endswith - str that file/dir basename must have to keep
    startswith - str that file/dir basename must have to keep
    exclude - str that will eliminate file/dir from keep if in basename
    dirs - bool; True if keep only dirs, False if exclude dirs, None if keep files and dirs
    bnames - bool; True if return is file basenames, False if return is full file path
    """
    if isinstance(exclude, str):
        exclude = [exclude]
    if dirs is False:
        return sorted([f for f in fs(DIR,
                                     pattern=pattern,
                                     endswith=endswith,
                                     startswith=startswith,
                                     exclude=exclude,
                                     bnames=bnames)
                       if not op.isdir(f)])
    elif dirs is True:
        return sorted([d for d in fs(DIR,
                                     pattern=pattern,
                                     endswith=endswith,
                                     startswith=startswith,
                                     exclude=exclude,
                                     bnames=bnames)
                       if op.isdir(d)])
    elif dirs is None:
        if exclude is not None:
            return sorted([op.join(DIR, f) if bnames is False else f
                           for f in os.listdir(DIR)
                           if pattern in f
                           and f.endswith(endswith)
                           and f.startswith(startswith)
                           and all([excl not in f for excl in exclude])])
        else:
            return sorted([op.join(DIR, f) if bnames is False else f
                           for f in os.listdir(DIR)
                           if pattern in f
                           and f.endswith(endswith)
                           and f.startswith(startswith)])


def uni(mylist:list) -> list:
    """Return unique values from list."""
    mylist = list(mylist)
    return list(set(mylist))


def luni(mylist:list) -> list:
    """Return length of unique values from list."""
    return len(uni(mylist))


def suni(mylist:list) -> list:
    """Retrun sorted unique values from list."""
    return sorted(uni(mylist))


def nrow(df) -> int:
    """Return number of rows in pandas.DataFrame."""
    return len(df.index)


def ncol(df) -> int:
    """Return number of cols in pandas.DataFrame."""
    return len(df.columns)


def table(lst:list, exclude=[]) -> dict:
    """Count each item in a list.
    
    Return a dict with key for each item, val of count
    """
    c = Counter()
    for x in lst:
        c[x] += 1
    if len(exclude) > 0:
        for ex in exclude:
            if ex in c:
                c.pop(ex)
    return c


def pkldump(obj, f:str) -> None:
    """Save object to .pkl file."""
    with open(f, 'wb') as o:
        pickle.dump(obj, o, protocol=pickle.HIGHEST_PROTOCOL)


def head(df):
    """Return head of pandas.DataFame."""
    return df.head()


def update(args:list):
    """For jupyter notebook, clear printout and print something new.
    
    Good for for-loops etc.
    """
    clear_output(wait=True)
    [print(x) for x in args]


def keys(Dict:dict) -> list:
    """Get a list of keys in a dictionary."""
    return list(Dict.keys())


def values(Dict:dict) -> list:
    """Get a list of values in a dictionary."""
    return list(Dict.values())


def setindex(df, colname:str):
    """Set index of pandas.DataFrame to values in a column, remove col."""
    df.index = df[colname].tolist()
    df.index.names = ['']
    df = df[[c for c in df.columns if not colname == c]]
    return df


def pklload(path:str):
    """Load object from a .pkl file"""
    pkl = pickle.load(open(path, 'rb'))
    return pkl


def gettimestamp(f:str):
    """Get ctime from a file path."""
    return time.ctime(os.path.getmtime(f))


def getmostrecent(files:list, remove=False) -> Optional[str]:
    """From a list of files, determine most recent.
    
    Optional to delete non-most recent files.
    """
    if not isinstance(files, list):
        files = [files]
    if len(files) > 1:
        whichout = files[0]
        dt1 = dt.strptime(gettimestamp(whichout)[4:], "%b %d %H:%M:%S %Y")
        for o in files[1:]:
            dt2 = dt.strptime(gettimestamp(o)[4:], "%b %d %H:%M:%S %Y")
            if dt2 > dt1:
                whichout = o
                dt1 = dt2
        if remove == True:
            [os.remove(f) for f in files if not f == whichout]
        return whichout
    elif len(files) == 1:
        return files[0]
    else:
        return None


def formatclock(hrs:float) -> str:
    """For a given number of hours, format a clock: days-hours:mins:seconds."""
    # format the time
    TIME = dt(1, 1, 1) + timedelta(hours=hrs)
    # zero out the minutes, add an hour
    if TIME.minute > 0:
        TIME = TIME + timedelta(hours=1) - timedelta(minutes=TIME.minute)
    # round up to 7 days, zero out the hours
    if 3 < (TIME.day-1) < 7 or (3 <= TIME.day-1 and TIME.hour > 0):
        diff = 7 - (TIME.day-1)
        TIME = TIME + timedelta(days=diff) - timedelta(hours=TIME.hour)
    # round up to 3 days, zero out the hours
    if 1 < (TIME.day-1) < 3 or (1 <= TIME.day-1 and TIME.hour > 0):
        diff = 3 - (TIME.day-1)
        TIME = TIME + timedelta(days=diff) - timedelta(hours=TIME.hour)
    # round up to 24 hrs, zero out the hours
    if TIME.day == 1 and 12 < TIME.hour < 24:
        TIME = TIME + timedelta(days=1) - timedelta(hours=TIME.hour)
    # round up to 12 hrs 
    if TIME.day == 1 and 3 < TIME.hour < 12:
        diff = 12 - TIME.hour
        TIME = TIME + timedelta(hours=diff)
    # round up to 3 hrs 
    if TIME.day == 1 and TIME.hour < 3:
        diff = 3 - TIME.hour
        TIME = TIME + timedelta(hours=diff) 
    clock = "%s-%s:00:00"  % (TIME.day -1, str(TIME.hour).zfill(2))
    return clock


def printmd(string:str) -> None:
    """For jupyter notebook, print as markdown.
    
    Useful for for-loops, etc
    """
    string = str(string)
    display(Markdown(string))


def makedir(directory:str) -> str:
    """If directory doesn't exist, create it.
    
    Return directory.
    """    
    if not op.exists(directory):
        os.makedirs(directory)
    return directory


def getdirs(paths:Union[str, list]) -> list:
    """Recursively get a list of all subdirs from given path."""
    if isinstance(paths, str):
        print('converting to list')
        paths = [paths]
    newdirs = []
    for path in paths:
        if op.isdir(path):
            print(path)
            newdirs.append(path)
            newestdirs = getdirs(fs(path, dirs=True))
            newdirs.extend(newestdirs)
    return newdirs


def get_client(profile='default') -> tuple:
    """Get lview,dview from ipcluster."""
    rc = Client(profile=profile)
    dview = rc[:]
    lview = rc.load_balanced_view()
    print(len(lview),len(dview))
    return lview, dview


def make_jobs(inputs:list, cmd, lview) -> list:
    """Send each arg from inputs to a function command; async."""
    print(f"making jobs for {cmd.__name__}")
    jobs = []
    for arg in tnb(inputs):
        jobs.append(lview.apply_async(cmd, arg))
    return jobs


def send_chunks(fxn, elements, thresh):
    """Send a list of args from inputs to a function command; async."""
    jobs = []
    mylst = []
    for i,element in enumerate(elements):
        mylst.append(element)
        if len(mylst) == math.ceil(thresh) or (i+1)==len(elements):
            print('len(mylst) = ', len(mylst))
            jobs.append(lview.apply_async(fxn, mylst))
            mylst = []
    return jobs


def watch_async(jobs:list, phase=None) -> None:
    """Wait until jobs are done executing, get updates"""
    print(len(jobs))
    count = 0
    while count != len(jobs):
        time.sleep(5)
        count = 0
        for j in jobs:
            if j.ready():
                count += 1
        if phase is not None:
            update([phase, count, len(jobs)])
        else:
            update([count, len(jobs)])


def read(file:str, lines=True) -> Union[str, list]:
    """Read lines from a file.
    
    Return a list of lines, or one large string
    """
    with open(file, 'r') as o:
        text = o.read()
    if lines is True:
        return text.split("\n")
    else:
        return text
    
