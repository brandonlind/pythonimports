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
from tqdm import trange
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
from tqdm.notebook import tqdm as tnb
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
        return sorted([op.basename(f) if bnames is True else f
                       for f in fs(DIR,
                                   pattern=pattern,
                                   endswith=endswith,
                                   startswith=startswith,
                                   exclude=exclude)
                       if not op.isdir(f)])
    elif dirs is True:
        return sorted([op.basename(d) if bnames is True else d
                       for d in fs(DIR,
                                   pattern=pattern,
                                   endswith=endswith,
                                   startswith=startswith,
                                   exclude=exclude)
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


def formatclock(hrs:float, exact=False) -> str:
    """For a given number of hours, format a clock: days-hours:mins:seconds.
    
    Parameters
    ----------
    exact - if False, return clock rounded up by partitions on Compute Canada
            if True, return clock with exactly days/min/hrs/seconds
    """
    # format the time
    TIME = dt(1, 1, 1) + timedelta(hours=hrs)
    if exact is False:
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
    else:
        clock = "%s-%s:%s:%s" % (TIME.day - 1,
                                 str(TIME.hour).zfill(2),
                                 str(TIME.minute).zfill(2),
                                 str(TIME.second).zfill(2))
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


def send_chunks(fxn, elements, thresh, lview, kwargs={}):
    """Send a list of args from inputs to a function command; async."""
    jobs = []
    mylst = []
    for i,element in enumerate(elements):
        mylst.append(element)
        if len(mylst) == math.ceil(thresh) or (i+1)==len(elements):
            jobs.append(lview.apply_async(fxn, mylst, **kwargs))
            mylst = []
    return jobs


def watch_async(jobs:list, phase=None) -> None:
    """Wait until jobs are done executing, show progress bar."""
    from tqdm import trange

    print(ColorText(f"\nWatching {len(jobs)} {f'{phase} ' if phase is not None else ''}jobs ...").bold())
    time.sleep(1)

    job_idx = list(range(len(jobs)))
    for i in trange(len(jobs), desc=phase):
        count = 0
        while count < (i+1):
            count = len(jobs) - len(job_idx)
            for j in job_idx:
                if jobs[j].ready():
                    count += 1
                    job_idx.remove(j)
    pass


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

class ColorText():
    """
    Use ANSI escape sequences to print colors +/- bold/underline to bash terminal.
    """
    def __init__(self, text:str):
        self.text = text
        self.ending = '\033[0m'
        self.colors = []

    def __str__(self):
        return self.text

    def bold(self):
        self.text = '\033[1m' + self.text + self.ending
        return self

    def underline(self):
        self.text = '\033[4m' + self.text + self.ending
        return self

    def green(self):
        self.text = '\033[92m' + self.text + self.ending
        self.colors.append('green')
        return self

    def purple(self):
        self.text = '\033[95m' + self.text + self.ending
        self.colors.append('purple')
        return self

    def blue(self):
        self.text = '\033[94m' + self.text + self.ending
        self.colors.append('blue')
        return self

    def warn(self):
        self.text = '\033[93m' + self.text + self.ending
        self.colors.append('yellow')
        return self

    def fail(self):
        self.text = '\033[91m' + self.text + self.ending
        self.colors.append('red')
        return self
    pass


def get_skipto_df(f, skipto, nrows, sep='\t', index_col=None, header='infer', **kwargs):
    """Retrieve dataframe in parallel so that all rows are captured when iterating.
    
    Parameters
    ----------
    f - filename to open
    skipto - row number to skip, read rows thereafter
    nrows - how many rows to read from f after skipto
    args - a list of functions to apply to df after reading in
    kwargs - kwargs for the functions in args
    
    Returns
    -------
    df - pandas.DataFrame
    """
    import pandas
    
    # read in the appropriate chunk of the file
    if skipto == 0:
        df = pandas.read_table(f,
                               sep=sep,
                               index_col=index_col,
                               header=header,
                               nrows=nrows-1)
    else:
        df = pandas.read_table(f,
                               sep=sep,
                               index_col=index_col,
                               header=header,
                               skiprows=range(1, skipto),
                               nrows=nrows)

    # do other stuff to the dataframe while in parallel
    if len(kwargs) > 0:
        for function,args in kwargs.items():  # for a list of functions
            func = globals()[function]
            print('function = ', func.__name__)
            if 'None' in args:
                df = func(df)
            else:
                df = func(df, *args)  # do stuff

    return df


def parallel_read(f:str, linenums=None, nrows=None, header=None, lview=None, **kwargs):
    """
    Read in a dataframe file in parallel with ipcluster.
    
    Parameters
    ----------
    f - filename to open
    linenums - the number of non-header lines in the txt/csv file
    nrows - the number of lines to read in each parallel process
    
    Returns
    -------
    jobs - a list of AsyncResult for each iteration (num iterations = linenums/nrows)
    """
    print(ColorText('parallel_read()').bold().__str__() + ' is:')
    
    # determine how many lines are in the file
    if linenums is None:
        print('\tdeterming line numbers for ', f, ' ...')
        linenums = int(subprocess.check_output(['wc', '-l', f]).decode('utf-8').replace("\n", "").split()[0])
        if header is not None:
            # if there is a header, subtract from line count
            linenums = linenums - 1

    # evenly distribute jobs across engines
    if nrows is None:
        print('\tdetermining chunksize (nrows) ...')
        nrows = math.ceil(linenums/len(lview))

    # load other functions to engines
    if len(kwargs) > 0:
        print('\tloading functions to engines ...')
        for func,args in kwargs.items():
            dview[func] = globals()[func]
#             print('args = ', args)
#             print('type(args) = ', type(args))
            if 'None' in args:
                continue
            for arg in args:
                dview[arg] = globals()[arg]
                time.sleep(1)
        time.sleep(5)

    # read-in in parallel
    print('\tsending jobs to engines ...')
    time.sleep(0.1)
    jobs = []
    for skipto in trange(0, linenums, nrows):
        jobs.append(lview.apply_async(get_skipto_df, *(f, skipto, nrows), **kwargs))
#         jobs.append(get_skipto_df(f, skipto, nrows, **kwargs))  # for testing
    
    return jobs


def makesweetgraph(x=None,y=None,cmap='jet',ylab=None,xlab=None,bins=100,saveloc=None,
                   figsize=(5,4),snsbins=60,title=None,xlim=None,ylim=None,vlim=(None,None)):
    """Make 2D histogram with marginal histograms for each axis.
    
    Parameters
    ----------
    x - data for x-axis
    y - data for y-axis (sample identity in same order as x)
    cmap - color map (eg 'jet', 'cool', etc)
    ylab,xlab - axes labels
    bins - bins for plt.hist2d - basically how thick points are in figure
    snsbins - bins for margin histograms
    saveloc - location to save PDF, or None to skip saving
    figsize - dimensions of figure in inches (x, y)
    title - text above figure
    xlim, ylim - tuple with min and max for each axis
    vlim - tuple with min and max for color bar (to standardize across figures)
    """
    import seaborn as sns
    from matplotlib.colors import LogNorm
    from matplotlib.backends.backend_pdf import PdfPages
    # plot data
    ax1 = sns.jointplot(x=x, y=y,marginal_kws=dict(bins=snsbins))
    ax1.fig.set_size_inches(figsize[0], figsize[1])
    ax1.ax_joint.cla()
    plt.sca(ax1.ax_joint)
    plt.hist2d(x,y,bins,norm=LogNorm(*vlim),cmap=cmap,
                       range=None if xlim is None else np.array([xlim, ylim]))
    # set title and axes labels
    if title is None:
        plt.title('%s\nvs\n%s\n' % (xlab,ylab),y=1.2,x=0.6)
    else:
        plt.title(title,y=1.2,x=0.6)
    plt.ylabel(ylab,fontsize=12)
    plt.xlabel(xlab,fontsize=12)
    # set up scale bar legend
    cbar_ax = ax1.fig.add_axes([1, 0.1, .03, .7])
    cb = plt.colorbar(cax=cbar_ax)
    cb.set_label(r'$\log_{10}$ density of points',fontsize=13)
    # save if prompted
    if saveloc is not None:
        with PdfPages(saveloc) as pdf:
            pdf.savefig(bbox_inches='tight')
        print(ColorText('Saved to: ').bold(), saveloc)
    plt.show()
    pass
