import os
import sys
import time
import math
import pickle
import random
import string
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from typing import Optional, Union
from collections import OrderedDict, Counter, defaultdict
from IPython.display import Markdown, display, clear_output
from tqdm import trange
from os import path as op
from os import chdir as cd
from decimal import Decimal
from tqdm import tqdm as pbar
from os import getcwd as cwd
from shutil import copy as cp
from shutil import move as mv
from ipyparallel import Client
from datetime import timedelta
from datetime import datetime as dt
from tqdm.notebook import tqdm as tnb
from matplotlib.backends.backend_pdf import PdfPages

from myutils import *
from myslurm import *

# backwards compatibility
nb = pbar
# /backwards compatibility


pd.set_option('display.max_columns', 100)

def latest_commit():
    """Print latest commit upon import."""
    cwd = os.getcwd()
    pypaths = os.environ['PYTHONPATH'].split(":")
    pyimportpath = [path for path in pypaths if 'pythonimports' in path][0]
    os.chdir(pyimportpath)
    gitout = subprocess.check_output([shutil.which('git'), 'log', '--pretty', '-n1', pyimportpath]).decode('utf-8')
    gitout = '\n'.join(gitout.split('\n')[:3])
    current_datetime = "Today:\t" + dt.now().strftime("%B %d, %Y - %H:%M:%S")
    hashes = '##################################################################\n'
    print(hashes + 'Current commit of pythonimports:\n' + gitout + '\n' + current_datetime + '\n' + hashes)
    os.chdir(cwd)
    pass


def ls(DIR:str) -> list:
    """Get a list of file basenames from DIR."""
    return sorted(os.listdir(DIR))


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
    c = Counter(lst)
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


def formatclock(hrs:Union[datetime.timedelta, float], exact=False) -> str:
    """For a given number of hours, format a clock: days-hours:mins:seconds.
    
    Parameters
    ----------
    hrs - either a float (in hours) or a datetime.timedelta object (which is converted to hours)
    exact - if False, return clock rounded up by partitions on Compute Canada
            if True, return clock with exactly days/min/hrs/seconds
    """    
    # format hours
    if isinstance(hrs, datetime.timedelta):
        hrs = hrs.total_seconds() / 3600
    else:
        assert isinstance(hrs, float), 'hrs object must either be of type `datetime.timedelta` or `float`'
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


def getdirs(paths:Union[str, list], verbose=False, **kwargs) -> list:
    """Recursively get a list of all subdirs from given path."""
    if isinstance(paths, str):
        print('converting to list')
        paths = [paths]
    newdirs = []
    for path in paths:
        if op.isdir(path):
            if verbose is True:
                print(path)
            newdirs.append(path)
            newestdirs = getdirs(fs(path, dirs=True, **kwargs), verbose=verbose, **kwargs)
            newdirs.extend(newestdirs)
    return newdirs


def get_client(profile='default', **kwargs) -> tuple:
    """Get lview,dview from ipcluster."""
    rc = Client(profile=profile, **kwargs)
    dview = rc[:]
    lview = rc.load_balanced_view()
    print(len(lview),len(dview))
    return lview, dview


def make_jobs(cmd, inputs:list, lview) -> list:
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


def watch_async(jobs:list, phase=None, desc=None) -> None:
    """Wait until jobs are done executing, show progress bar."""
    from tqdm import trange

    print(ColorText(f"\nWatching {len(jobs)} {f'{phase} ' if phase is not None else ''}jobs ...").bold())
    time.sleep(1)

    try:
        job_idx = list(range(len(jobs)))
        for i in trange(len(jobs), desc=phase if desc is None else desc):
            count = len(jobs) - len(job_idx)
            while count < (i+1):
                for j in job_idx:
                    if jobs[j].ready():
                        count += 1
                        job_idx.remove(j)
    except KeyboardInterrupt:
        time.sleep(0.2)
        print(ColorText(f'KeboardInterrupted').warn())
        
    pass


def read(file:str, lines=True, ignore_blank=True) -> Union[str, list]:
    """Read lines from a file.
    
    Return a list of lines, or one large string
    """
    with open(file, 'r') as o:
        text = o.read()
    if lines is True:
        text = text.split("\n")
        if ignore_blank is True:
            text = [line for line in text if line != '']
        return text
    else:
        return text


# class ColorText():
#     """
#     Use ANSI escape sequences to print colors +/- bold/underline to bash terminal.
    
#     Notes
#     -----
#     execute ColorText.demo() for a printout of colors.
#     """
#     def demo():
#         """Prints examples of all colors in normal, bold, underline, bold+underline."""
#         for color in dir(ColorText):
#             if all([color.startswith('_') is False,
#                    color not in ['bold', 'underline', 'demo'],
#                    callable(getattr(ColorText, color))]):
#                 print(getattr(ColorText(color), color)(),'\t',                
#                       getattr(ColorText(f'bold {color}').bold(), color)(),'\t',
#                       getattr(ColorText(f'underline {color}').underline(), color)(),'\t',
#                       getattr(ColorText(f'bold underline {color}').underline().bold(), color)())
#         pass

#     def __init__(self, text:str):
#         self.text = text
#         self.ending = '\033[0m'
#         self.colors = []
#         pass

#     def __repr__(self):
#         return self.text

#     def __str__(self):
#         return self.text

#     def bold(self):
#         self.text = '\033[1m' + self.text + self.ending
#         return self

#     def underline(self):
#         self.text = '\033[4m' + self.text + self.ending
#         return self

#     def green(self):
#         self.text = '\033[92m' + self.text + self.ending
#         self.colors.append('green')
#         return self

#     def purple(self):
#         self.text = '\033[95m' + self.text + self.ending
#         self.colors.append('purple')
#         return self

#     def blue(self):
#         self.text = '\033[94m' + self.text + self.ending
#         self.colors.append('blue')
#         return self

#     def ltblue(self):
#         self.text = '\033[34m' + self.text + self.ending
#         self.colors.append('lightblue')
#         return self

#     def pink(self):
#         self.text = '\033[35m' + self.text + self.ending
#         self.colors.append('pink')
#         return self
    
#     def gray(self):
#         self.text = '\033[30m' + self.text + self.ending
#         self.colors.append('gray')
#         return self

#     def ltgray(self):
#         self.text = '\033[37m' + self.text + self.ending
#         self.colors.append('ltgray')
#         return self

#     def warn(self):
#         self.text = '\033[93m' + self.text + self.ending
#         self.colors.append('yellow')
#         return self

#     def fail(self):
#         self.text = '\033[91m' + self.text + self.ending
#         self.colors.append('red')
#         return self

#     def ltred(self):
#         self.text = '\033[31m' + self.text + self.ending
#         self.colors.append('lightred')
#         return self

#     def cyan(self):
#         self.text = '\033[36m' + self.text + self.ending
#         self.colors.append('cyan')
#         return self

#     pass


def get_skipto_df(f, skipto, nrows, sep='\t', index_col=None, header='infer', **kwargs):
    """Retrieve dataframe in parallel so that all rows are captured when iterating.
    
    Parameters
    ----------
    f - filename to open
    skipto - row number to skip, read rows thereafter
    nrows - how many rows to read from f after skipto
    args - a list of functions to apply to df after reading in
    kwargs - kwargs for pandas.read_table;
             kwargs['functions'] are functions (keys) and args (values) to apply
                 to dataframe while in parallel
    
    Returns
    -------
    df - pandas.DataFrame
    """
    import pandas
    
    # isolate functions that are wanted to be applied to df chunk after being read in
    if 'functions' in list(kwargs.keys()):
        func_dict = kwargs.pop('functions')
    else:
        func_dict = {}
    
    # read in the appropriate chunk of the file
    if skipto == 0:
        df = pandas.read_table(f,
                               sep=sep,
                               index_col=index_col,
                               header=header,
                               nrows=nrows-1,
                               **kwargs)
    else:
        df = pandas.read_table(f,
                               sep=sep,
                               index_col=index_col,
                               header=header,
                               skiprows=range(1, skipto),
                               nrows=nrows,
                               **kwargs)

    # do other stuff to the dataframe while in parallel
    if len(func_dict) > 0:
        for function,func_info in func_dict.items():  # for a list of functions
            print(function)
            func = func_info[function]
            df = func(df, *list(func_info['args'].values()), **func_info['kwargs'])    # do stuff

    return df


def create_fundict(function, args={}, kwargs={}):
    """Create a fundict for `parallel_read()`."""
    fundict = OrderedDict({function.__name__: dict({function.__name__: function,
                                                    'args': args,
                                                    'kwargs': kwargs
                                                   })
                          })
    return fundict


def timer(func):
    """Decorator to report time to complete function `func`."""
    from functools import wraps
    
    @wraps(func)
    def call(*args, **kwargs):
        t1 = dt.now()
        result = func(*args, **kwargs)
        time.sleep(0.2)
        print(f'Function `{call.__name__}` completed after : %s' % formatclock(dt.now() - t1, exact=True))
        return result
    return call


@timer
def parallel_read(f:str, linenums=None, nrows=None, header=0, lview=None, dview=None, verbose=True, desc=None,
                  assert_rowcount=True, reset_index=True, maintain_dataframe=True, **kwargs):
    """
    Read in a dataframe file in parallel with ipcluster.
    
    Parameters
    ----------
    f - filename to open
    linenums - the number of non-header lines in the txt/csv file
    nrows - the number of lines to read in each parallel process
    header - passed to pd.read_csv(), used to infer proper line counts
    verbose - more printing
    desc - passed to watch_async(); if None, then uses basename of `f`
    reset_index - name index with range(len(df.index))
    assert_rowcount - if one of the functions passed in kwargs filters rows, 
        then set to False otherwise there will be an AssertionError when double checking it read them in
    maintain_dataframe - True if I pass functions that do return pd.DataFrame
        - set to False if the functions in kwargs['functions'] do not return pd.DataFrame's
    kwargs - passed to get_skipto_df(); see get_skipto_df() docstring for more info
    
    Returns
    -------
    jobs - a list of AsyncResult for each iteration (num iterations = linenums/nrows)
    
    
    Example
    -------
    # function_x will be executed while in parallel, and therefore cannot depend on the full dataframe
        # the parallel execution is only operating on a chunk of the dataframe - see get_skipto_df
        # so if this function reduces the number of rows that are read in, set `assert_rowcount` to `False`

    # step 1) start by importing and specify the file path
    >>> from somemodule import function_x  # you can also just define this function in line
    >>> from pythonimports import get_client, parallel_read
    >>>
    >>> lview,dview = get_client()  # get ipcluster engines
    >>> f = '/some/path.txt'  # path to really big file
    
    # step 2)
    # set up a dictionary that contains functions and their arguments that are availale in __main__.globals()
        # this is a multidimensional dict that in the upper most dimension has the function names as keys
        # each key points to another dictionary, with three keys:
                # the function name as a string, 'args', and 'kwargs'
            # the function name key points to the actual function in __main__.globals()
            # the value of 'args' and 'kwargs' will be applied to the function (while in parallel) as:
                # function(df, *args, **kwargs)
                # therefore the value for the `args` and `kwargs` key must have a .__len__() method

    >>> fundict = dict({'function_x': dict({'function_x': function_x,
                                            'args': [],
                                            'kwargs': {}
                                           })
                       })

    # step 3) run parallel_read()
    >>> df = parallel_read(f, lview=lview, dview=dview, assert_rowcount=False, **dict(functions=fundict))
    """
    from functools import partial
    
    kwargs.update({'header': header})
    
    if verbose:
        print(ColorText('parallel_read()').bold().__str__() + ' is:')
    
    # determine how many lines are in the file
    if linenums is None:
        if verbose:
            print('\tdeterming line numbers for ', ColorText(f).gray(), ' ...')
#         linenums = int(subprocess.check_output(['wc', '-l', f]).decode('utf-8').replace("\n", "").split()[0])
        lines = subprocess.check_output([shutil.which('awk'), '{print $1}', f]).split(b"\n")
        linenums = len([line for line in lines if line != b'']) # use this to avoid weird cases where file ends with '\n'
            

    # evenly distribute jobs across engines
    if nrows is None:
        if verbose:
            print('\tdetermining chunksize (nrows) ...')
        nrows = math.ceil(linenums/len(lview))

    # load other functions to engines
    if 'functions' in kwargs.keys():
        if verbose:
            print('\tloading functions to engines ...')
        for func,func_info in kwargs['functions'].items():
            if verbose:
                print('\t', '\t', func)
            dview[func] = func_info[func]
            if len(func_info['args']) > 0:
                for arg_name,arg in func_info['args'].items():
                    dview[arg_name] = arg
                    time.sleep(1)
            if len(func_info['kwargs']) > 0:
                for kwarg_name,kwarg in func_info['kwargs'].items():
                    if all([isinstance(kwarg, bool) is False, isinstance(kwarg, str) is False]):
                        # if the kwarg is an object that needs to be loaded to engines, load it.
                        dview[kwarg_name] = kwarg
            time.sleep(1)
        time.sleep(5)

    # set iterator function
    if verbose:
        print('\tsending jobs to engines ...')
        ranger = partial(trange, desc='sending jobs')
    else:
        ranger = range
    time.sleep(1)

    # read-in in parallel
    jobs = []
    for skipto in ranger(0, linenums, nrows):
        jobs.append(lview.apply_async(get_skipto_df, *(f, skipto, nrows), **kwargs))
#         jobs.append(get_skipto_df(f, skipto, nrows, **kwargs))  # for testing

    watch_async(jobs, phase='parallel_read()', desc=desc if desc is not None else op.basename(f))
    
    if maintain_dataframe is True:
        df = pd.concat([j.r for j in jobs])

        if reset_index is True:
            # avoid duplicated indices across jobs when no index was set
            df.index = range(len(df.index))

        if header is not None:
            # if there is a header, subtract from line count
            linenums = linenums - 1

        if assert_rowcount is True:
            assert nrow(df) == linenums, (nrow(df), linenums)
    else:
        # in case I want to execute functions that do not return dataframes
        df = [j.r for j in jobs]
    
    return df


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


def rsync(src, dst, different_basenames=False):
    """Execute rsync command; can execute via ipyparallel engines.

    Parameters
    ----------
    src - source file; assumes full file path is given
    dst - destination path, either a directory or full file path name
    different_basenames - bool; True if src and dst file paths differ in their basenames, False otherwise

    Notes
    -----
    - src and dst basenames can differ.

    TODO : add override kwarg to skip over hacky assertion of destination being a full filepath ...
           if the src and dst basenames do not match
    """
    import subprocess, shutil, os

    assertion_msg = 'Either the source or the destination should have a server \
in the name that includes a colon (":") that prepends the path.'
    assert any([':' in src, ':' in dst]), assertion_msg

    # so I can pass a directory or the actual destination path
    if dst.endswith(os.path.basename(src)) is False and different_basenames is False:
        # if dst is a directory and it's not expected that source or destination basenames vary
        dst = op.join(dst, os.path.basename(src))  # change dst to full destination file path
    elif dst.endswith(os.path.basename(src)) is False and different_basenames is True:
        # hacky way to ensure this is a file: assert that the basename has a '.' in it
        assert '.' in op.basename, 'it seems that dst is a directory'

    output = subprocess.check_output([shutil.which('rsync'), '-azv', src, dst]).decode('utf-8').split('\n')

    return output


def quick_write(df, dst, sep='\t', header=True, index=False):
    """Quickly write a pd.DataFrame to file, much faster than .to_csv for large files."""
    from tqdm import tqdm
    tqdm.pandas()
    
    if index is not False:
        raise AssertionError('as of now this function cannot write indices to file.')

    lines = []
    if header is True:
        lines = [sep.join(df.columns)]

    lines.extend(df.progress_apply(lambda line: sep.join(map(str, line)), axis=1).tolist())

    with open(dst, 'w') as o:
        o.write('\n'.join(lines))

    pass


def flatten(list_of_lists, unique=False):
    """Return a single list of values from each value in a list of lists.
    
    Parameters
    ----------
    - list_of_lists - a list where each element is a list
    - unique - bool; True to return only unique values, False to return all values.
    
    """
    assert list_of_lists.__class__.__name__ in ['list', 'dict_values', 'ndarray']
    vals = list(pd.core.common.flatten(list_of_lists))
    if unique is True:
        vals = uni(vals)
    return vals


def sleeping(counts:int, desc='sleeping', sleep=1):
    """Basically a sleep timer with a progress bar; counts up to `counts`, interval = 1sec."""
    try:
        for i in trange(counts, desc=desc):
            time.sleep(sleep)
    except KeyboardInterrupt:
        print(ColorText(f'KeyboardInterrupt after {i} seconds of sleep.').warn())
    pass
