import sys
import os
import pickle
import random
import string
import math
import shutil
import time
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as nb
from IPython.display import clear_output
from collections import OrderedDict, Counter
from IPython.display import Markdown, display
from os import path as op
from os import chdir as cd
from os import getcwd as cwd
from os import listdir
from shutil import copy as cp
from shutil import move as mv
from ipyparallel import Client
from decimal import Decimal
from datetime import timedelta
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as pl
import matplotlib.dates as mdates

pd.set_option('display.max_columns', 100)
def ls(DIR):
    return sorted(listdir(DIR))
def fs(DIR, dirs=None):
    if dirs == False:
        return sorted([f for f in fs(DIR) if not op.isdir(f)])
    elif dirs == True:
        return sorted([d for d in fs(DIR) if op.isdir(d)])
    elif dirs == None:
        return sorted([op.join(DIR, f) for f in ls(DIR)])
def uni(mylist):
    mylist = list(mylist)
    return list(set(mylist))
def luni(mylist):
    return len(uni(mylist))
def suni(mylist):
    return sorted(uni(mylist))
def nrow(df):
    return len(df.index)
def ncol(df):
    return len(df.columns)
def table(lst, exclude=[]):
    c = Counter()
    for x in lst:
        c[x] += 1
    if len(exclude) > 0:
        for ex in exclude:
            if ex in c:
                c.pop(ex)
    return c
def pkldump(obj, f):
    with open(f, 'wb') as o:
        pickle.dump(obj, o, protocol=pickle.HIGHEST_PROTOCOL)
def head(df):
    return df.head()
def update(args):
    clear_output(wait=True)
    [print(x) for x in args]
def keys(Dict):
    return list(Dict.keys())
def values(Dict):
    return list(Dict.values())
def setindex(df, colname):
    df.index = df[colname].tolist()
    df.index.names = ['']
    df = df[[c for c in df.columns if not colname == c]]
    return df
def pklload(path):
    pkl = pickle.load(open(path, 'rb'))
    return pkl
def gettimestamp(f):
    return time.ctime(os.path.getmtime(f))
def getmostrecent(files, remove=False):
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
def formatclock(hrs):
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
def printmd(string):
    string = str(string)
    display(Markdown(string))
def makedir(directory):
    if not op.exists(directory):
        os.makedirs(directory)
    return directory
def getdirs(paths):
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
def get_client(profile):
    rc = Client(profile=profile)
    dview = rc[:]
    lview = rc.load_balanced_view()
    print(len(lview),len(dview))
    return lview, dview

def make_jobs(inputs, cmd, lview):
    jobs = []
    for arg in inputs:
        jobs.append(lview.apply_async(cmd, arg))
    return jobs

def watch_async(jobs):
    """Wait until jobs are done executing, get updates"""
    print(len(jobs))
    count = 0
    while count != len(jobs):
        time.sleep(5)
        count = 0
        for j in jobs:
            if j.ready():
                count += 1
        update([count, len(jobs)])
