import os
import sys
import time
import math
import pydoc
import pickle
import random
import string
import shutil
import datetime
import numpy as np
import pandas as pd
import session_info
import ipyparallel as ipp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from tqdm import trange as _trange
from os import path as op
from os import chdir as cd
from decimal import Decimal
from tqdm import tqdm
from shutil import copy as cp
from shutil import move as mv
from ipyparallel import Client
from datetime import timedelta
from typing import Optional, Union
from datetime import datetime as dt
from tqdm.notebook import tqdm as tnb
from collections import OrderedDict, Counter, defaultdict
from IPython.display import Markdown, display, clear_output
from functools import partial

from myslurm import *
from mymaps import *
from myclasses import ColorText
#from myfigs import *

# backwards compatibility
bar_format = '{l_bar}{bar:15}{r_bar}'
trange = partial(_trange, bar_format=bar_format)
pbar = partial(tqdm, bar_format=bar_format)
nb = pbar
sinfo = session_info.show
tqdm.pandas(bar_format=bar_format)
# /backwards compatibility

pd.set_option("display.max_columns", 100)

def ls(directory: str) -> list:
    """Get a list of file basenames from DIR."""
    return sorted(os.listdir(directory))


def fs(directory: str, pattern="", endswith="", startswith="", exclude=None, dirs=None, bnames=False) -> list:
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
        return sorted(
            [
                op.basename(f) if bnames is True else f
                for f in fs(directory,
                            pattern=pattern,
                            endswith=endswith,
                            startswith=startswith,
                            exclude=exclude)
                if not op.isdir(f)
            ]
        )
    elif dirs is True:
        return sorted(
            [
                op.basename(d) if bnames is True else d
                for d in fs(directory,
                            pattern=pattern,
                            endswith=endswith,
                            startswith=startswith,
                            exclude=exclude)
                if op.isdir(d)
            ]
        )
    elif dirs is None:
        if exclude is not None:
            return sorted([op.join(directory, f) if bnames is False else f
                           for f in os.listdir(directory)
                           if pattern in f
                           and f.endswith(endswith)
                           and f.startswith(startswith)
                           and all([excl not in f for excl in exclude])])
        else:
            return sorted([op.join(directory, f) if bnames is False else f
                           for f in os.listdir(directory)
                           if pattern in f
                           and f.endswith(endswith)
                           and f.startswith(startswith)])
    pass


def uni(mylist: list) -> list:
    """Return unique values from list."""
    mylist = list(mylist)
    return list(set(mylist))


def luni(mylist: list) -> int:
    """Return length of unique values from list."""
    return len(uni(mylist))


def suni(mylist: list) -> list:
    """Retrun sorted unique values from list."""
    return sorted(uni(mylist))


def nrow(df) -> int:
    """Return number of rows in pandas.DataFrame."""
    return len(df.index)


def ncol(df) -> int:
    """Return number of cols in pandas.DataFrame."""
    return len(df.columns)


def table(lst: list, exclude=[]) -> dict:
    """Count each item in a list.

    Return a dict with key for each item, val of count
    """
    c = Counter(lst)
    if len(exclude) > 0:
        for ex in exclude:
            if ex in c:
                c.pop(ex)
    return c


def pkldump(obj, f: str, protocol=pickle.HIGHEST_PROTOCOL) -> None:
    """Save object to .pkl file."""
    with open(f, "wb") as o:
        pickle.dump(obj, o, protocol=protocol)


def head(df: pd.DataFrame) -> pd.DataFrame:
    """Return head of pandas.DataFame."""
    return df.head()


def update(args: list) -> None:
    """For jupyter notebook, clear printout and print something new.

    Good for for-loops etc.
    """
    clear_output(wait=True)
    [print(x) for x in args]


def keys(dikt: dict) -> list:
    """Get a list of keys in a dictionary."""
    return list(dikt.keys())


def values(dikt: dict) -> list:
    """Get a list of values in a dictionary."""
    return list(dikt.values())


def setindex(df, colname: str) -> pd.DataFrame:
    """Set index of pandas.DataFrame to values in a column, remove col."""
    df.index = df[colname].tolist()
    df.index.names = [""]
    df = df[[c for c in df.columns if not colname == c]]
    return df


def pklload(path: str):
    """Load object from a .pkl file"""
    pkl = pickle.load(open(path, "rb"))
    return pkl


def gettimestamp(f: Union[str, list]) -> str:
    """Get ctime from a file path."""
    return time.ctime(os.path.getmtime(f))


def getmostrecent(files: list, remove=False) -> Optional[str]:
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
        if remove is True:
            [os.remove(f) for f in files if not f == whichout]
        return whichout
    elif len(files) == 1:
        return files[0]
    else:
        return None
    pass


def formatclock(hrs: Union[datetime.timedelta, float], exact=True) -> str:
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
        assert isinstance(
            hrs, float
        ), "hrs object must either be of type `datetime.timedelta` or `float`"
    # format the time
    _time = dt(1, 1, 1) + timedelta(hours=hrs)
    if exact is False:
        # zero out the minutes, add an hour
        if _time.minute > 0:
            _time = _time + timedelta(hours=1) - timedelta(minutes=_time.minute)
        # round up to 7 days, zero out the hours
        if 3 < (_time.day - 1) < 7 or (3 <= _time.day - 1 and _time.hour > 0):
            diff = 7 - (_time.day - 1)
            _time = _time + timedelta(days=diff) - timedelta(hours=_time.hour)
        # round up to 3 days, zero out the hours
        if 1 < (_time.day - 1) < 3 or (1 <= _time.day - 1 and _time.hour > 0):
            diff = 3 - (_time.day - 1)
            _time = _time + timedelta(days=diff) - timedelta(hours=_time.hour)
        # round up to 24 hrs, zero out the hours
        if _time.day == 1 and 12 < _time.hour < 24:
            _time = _time + timedelta(days=1) - timedelta(hours=_time.hour)
        # round up to 12 hrs
        if _time.day == 1 and 3 < _time.hour < 12:
            diff = 12 - _time.hour
            _time = _time + timedelta(hours=diff)
        # round up to 3 hrs
        if _time.day == 1 and _time.hour < 3:
            diff = 3 - _time.hour
            _time = _time + timedelta(hours=diff)
        clock = "%s-%s:00:00" % (_time.day - 1, str(_time.hour).zfill(2))
    else:
        clock = "%s-%s:%s:%s" % (
            _time.day - 1,
            str(_time.hour).zfill(2),
            str(_time.minute).zfill(2),
            str(_time.second).zfill(2),
        )
    return clock


def printmd(string: str) -> None:
    """For jupyter notebook, print as markdown.

    Useful for for-loops, etc
    """
    string = str(string)
    display(Markdown(string))


def makedir(directory: str) -> str:
    """If directory doesn't exist, create it.

    Return directory.
    """
#     if not op.exists(directory):
    os.makedirs(directory, exist_ok=True)
    
    return directory


def getdirs(paths: Union[str, list], verbose=False, exclude=None, **kwargs) -> list:
    """Recursively get a list of all subdirs from given path.
    
    Parameters
    ----------
    paths - a path (str) or list of paths to explore
    verbose - whether to print all directories when found
    kwargs - same kwargs used in `fs` to filter directories that are found    
    """
    if isinstance(paths, str):
        print("converting to list")
        paths = [paths]
        
    # get all subdirectories
    newdirs = []
    for path in paths:
        if op.isdir(path):
            if verbose is True:
                print(path)

            if exclude is not None and all([excl not in op.basename(path) for excl in exclude]):
                newdirs.append(path)
            elif exclude is None:
                newdirs.append(path)

            newestdirs = getdirs(
                fs(path, dirs=True, exclude=exclude),
                verbose=verbose,
                exclude=exclude
            )

            newdirs.extend(newestdirs)

    # filter for kwargs (faster than passing to `fs`)
    keeping = []
    for d in newdirs:
        basename = op.basename(d)
        
        keep = []
        if 'pattern' in kwargs:
            if kwargs['pattern'] in basename:
                keep.append(True)
            else:
                keep.append(False)
                
        if 'startswith' in kwargs:
            if basename.startswith(kwargs['startswith']):
                keep.append(True)
            else:
                keep.append(False)
                
        if 'endswith' in kwargs:
            if basename.endswith(kwargs['endswith']):
                keep.append(True)
            else:
                keep.append(False)
                
#         if exclude is not None:
#             if all([excl not in basename for excl in exclude]):
#                 keep.append(True)
#             else:
#                 keep.append(False)
                
        if all(keep):  # note `all([])` is True
            if 'bnames' in kwargs and kwargs['bnames'] is True:
                dname = basename
            else:
                dname = d
            keeping.append(dname)
            
    return keeping


def get_client(profile="default", targets=None, **kwargs) -> tuple:
    """Get lview,dview from ipcluster."""
    rc = Client(profile=profile, **kwargs)
    dview = rc.direct_view(targets=targets)
    lview = rc.load_balanced_view(targets=targets)
    print(len(lview), len(dview))
    return lview, dview


def start_engines(targets=None, cluster_id='', n=None, **kwargs):
    """Start ipcluster engines from within a python script.
    
    Notes
    -----
    For some reason, executing this function within a notebook causes fork errors on machines.
    """
    try:
        # see if engines exist
        cluster = ipp.Cluster.from_file(cluster_id=cluster_id)
    except FileNotFoundError as e:
        print('registering engines')
        # if not register engines
        cluster = ipp.Cluster(**kwargs)

    # start engines
    print(ColorText(str(cluster)).custom('gray'))
    c = cluster.start_and_connect_sync(n=n)
    
    # connect to engines, `c`, indirectly
    lview,dview = get_client(cluster_id=cluster.cluster_id, targets=targets, **kwargs)
    
    return lview, dview, cluster.cluster_id


def make_jobs(fxn, inputs: list, lview) -> list:
    """Send each arg from inputs to a function command; async."""
    print(f"making jobs for {fxn.__name__}")
    jobs = []
    for arg in tnb(inputs):
        jobs.append(lview.apply_async(fxn, arg))
    return jobs


def send_chunks(fxn, elements, thresh, lview, kwargs={}):
    """Send a list of args from inputs to a function command; async."""
    jobs = []
    mylst = []
    for i, element in enumerate(elements):
        mylst.append(element)
        if len(mylst) == math.ceil(thresh) or (i + 1) == len(elements):
            jobs.append(lview.apply_async(fxn, mylst, **kwargs))
            mylst = []
    return jobs


def watch_async(jobs: list, phase=None, desc=None, color=None, verbose=True) -> None:
    """Wait until all ipyparallel jobs `jobs` are done executing, show progress bar."""

    if verbose is True:
        print(
            ColorText(
                f"\nWatching {len(jobs)} {f'{phase} ' if phase is not None else ''}jobs ..."
            ).bold().custom(color)
        )
        time.sleep(1)
        
        iterator = partial(_trange, bar_format=bar_format, desc=phase if desc is None else desc)
    else:
        iterator = range

    try:
        job_idx = list(range(len(jobs)))
        for i in iterator(len(jobs)):
            count = len(jobs) - len(job_idx)
            while count < (i + 1):
                for j in job_idx:
                    if jobs[j].ready():
                        count += 1
                        job_idx.remove(j)
    except KeyboardInterrupt:
        time.sleep(0.2)
        print(ColorText(f"KeboardInterrupted").warn())

    pass


def read(file: str, lines=True, ignore_blank=False) -> Union[str, list]:
    """Read lines from a file.

    Return a list of lines, or one large string
    """
    with open(file, "r") as o:
        text = o.read()

    if lines is True:
        text = text.split("\n")
        if ignore_blank is True:
            text = [line for line in text if line != ""]

    return text


def get_skipto_df(f: str, skipto: int, nrows: int, sep="\t", index_col=None, header="infer", **kwargs) -> pd.DataFrame:
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
    import pandas._libs.lib as lib

    # isolate functions that are wanted to be applied to df chunk after being read in
    if "functions" in list(kwargs.keys()):
        func_dict = kwargs.pop("functions")
    else:
        func_dict = {}
    
    # avoid error
    if 'delim_whitespace' in kwargs.keys():
        sep=lib.no_default

    # read in the appropriate chunk of the file
    if skipto == 0:
        df = pandas.read_table(f, sep=sep, index_col=index_col, header=header, nrows=nrows - 1, **kwargs)
    else:
        df = pandas.read_table(f, sep=sep, index_col=index_col, header=header, skiprows=range(1, skipto),
                               nrows=nrows, **kwargs)

    # do other stuff to the dataframe while in parallel
    if len(func_dict) > 0:
        for function, func_info in func_dict.items():  # for a list of functions
            print(function)
            func = func_info[function]
            df = func(df, *list(func_info["args"].values()), **func_info["kwargs"])  # do stuff

    return df


def create_fundict(function, args={}, kwargs={}):
    """Create a fundict for `parallel_read()`.
    
    For each function `function` that is to be applied during parallel execution of `parallel_read()`, 
    use a dictionary for `args` and `kwargs` to specify args and kwargs, respectively. These are used
    to load objects to engines and when function is called.
    """
    fundict = OrderedDict({function.__name__: dict({function.__name__: function,
                                                    "args": args,
                                                    "kwargs": kwargs})})
    return fundict


def timer(func):
    """Decorator to report time to complete function `func`."""
    from functools import wraps

    @wraps(func)
    def call(*args, **kwargs):
        t1 = dt.now()
        result = func(*args, **kwargs)
        time.sleep(0.2)
        print(f"Function `{call.__name__}` completed after : %s" % formatclock(dt.now() - t1, exact=True))
        return result

    return call


@timer
def parallel_read(f: str, linenums=None, nrows=None, header=0, lview=None, dview=None, verbose=True,
                  desc=None, assert_rowcount=True, reset_index=True, maintain_dataframe=True, **kwargs
                  ) -> Union[list, pd.DataFrame]:
    """Read in a dataframe file in parallel with ipcluster.

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

    >>> fundict = create_fundict(function_x, args={}, kwargs={})

    # step 3) run parallel_read()
    >>> df = parallel_read(f, lview=lview, dview=dview, assert_rowcount=False, **dict(functions=fundict))
    """
    from functools import partial

    kwargs.update({"header": header})

    if verbose:
        print(ColorText("parallel_read()").bold().__str__() + " is:")

    # determine how many lines are in the file
    if linenums is None:
        if verbose:
            print("\tdeterming line numbers for ", ColorText(f).gray(), " ...")
        lines = subprocess.check_output([shutil.which("awk"), "{print $1}", f]).split(b"\n")
        linenums = len([line for line in lines if line != b""])  # use this 2avoid weird cases where file ends with '\n'

    # evenly distribute jobs across engines
    if nrows is None:
        if verbose:
            print("\tdetermining chunksize (nrows) ...")
        nrows = math.ceil(linenums / len(lview))

    # load other functions to engines
    if "functions" in kwargs.keys():
        if verbose:
            print("\tloading functions to engines ...")
        for func, func_info in kwargs["functions"].items():
            if verbose:
                print("\t", "\t", func)
            dview[func] = func_info[func]
            if len(func_info["args"]) > 0:
                for arg_name, arg in func_info["args"].items():
                    dview[arg_name] = arg
                    time.sleep(1)
            if len(func_info["kwargs"]) > 0:
                for kwarg_name, kwarg in func_info["kwargs"].items():
                    if all([isinstance(kwarg, bool) is False,
                            isinstance(kwarg, str) is False]):
                        # if the kwarg is an object that needs to be loaded to engines, load it.
                        dview[kwarg_name] = kwarg
            time.sleep(1)
        time.sleep(5)

    # set iterator function
    if verbose:
        print("\tsending jobs to engines ...")
        ranger = partial(trange, desc="sending jobs")
    else:
        ranger = range
    time.sleep(1)

    # read-in in parallel
    jobs = []
    for skipto in ranger(0, linenums, nrows):
        jobs.append(lview.apply_async(get_skipto_df, *(f, skipto, nrows), **kwargs))
    #         jobs.append(get_skipto_df(f, skipto, nrows, **kwargs))  # for testing

    watch_async(
        jobs, phase="parallel_read()", desc=desc if desc is not None else op.basename(f)
    )

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


def rsync(src, dst, options="-azv", different_basenames=False, assert_remote_transfer=True) -> list:
    """Execute rsync command; can execute via ipyparallel engines.

    Parameters
    ----------
    src - source file; assumes full file path is given
    dst - destination path, either a directory or full file path name
    options - the flags that should be used with the command; default -azv
    different_basenames - bool; True if src and dst file paths differ in their basenames, False otherwise
    assert_remote_transfer - bool; True if one of the paths should contain remote server info (eg `server:/path`)

    Notes
    -----
    - src and dst basenames can differ.
    """
    import subprocess
    import shutil
    import os

    if assert_remote_transfer is True:
        assertion_msg = 'Either the source or the destination should have a server \
    in the name that includes a colon (":") that prepends the path.'
        assert any([":" in src, ":" in dst]), assertion_msg

    # so I can pass a directory or the actual destination path
    if dst.endswith(os.path.basename(src)) is False and different_basenames is False:
        # if dst is a directory and it's not expected that source or destination basenames vary
        dst = op.join(dst, os.path.basename(src))  # change dst to full destination file path
    elif dst.endswith(os.path.basename(src)) is False and different_basenames is True:
        # hacky way to ensure this is a file: assert that the basename has a '.' in it
        assert "." in op.basename, "it seems that dst is a directory"

    output = (subprocess.check_output([shutil.which("rsync"), options, src, dst]).decode("utf-8").split("\n"))

    return output


def quick_write(df, dst, sep="\t", header=True, index=False) -> None:
    """Quickly write a pd.DataFrame to file, much faster than .to_csv for large files."""
    from tqdm import tqdm

    tqdm.pandas()

    if index is not False:
        raise AssertionError("as of now this function cannot write indices to file.")

    lines = []
    if header is True:
        lines = [sep.join(df.columns)]

    lines.extend(df.progress_apply(lambda line: sep.join(map(str, line)), axis=1).tolist())

    with open(dst, "w") as o:
        o.write("\n".join(lines))

    pass


def flatten(list_of_lists, unique=False) -> list:
    """Return a single list of values from each value in a list of lists.

    Parameters
    ----------
    - list_of_lists - a list where each element is a list
    - unique - bool; True to return only unique values, False to return all values.

    """
    assert list_of_lists.__class__.__name__ in ["list", "dict_values", "ndarray"]
    vals = list(pd.core.common.flatten(list_of_lists))
    if unique is True:
        vals = uni(vals)
    return vals


def sleeping(counts: int, desc="sleeping", sleep=1) -> None:
    """Basically a sleep timer with a progress bar; counts up to `counts`, interval = 1sec."""
    try:
        for i in trange(counts, desc=desc):
            time.sleep(sleep)
    except KeyboardInterrupt:
        print(ColorText(f"KeyboardInterrupt after {i} seconds of sleep.").warn())
    pass


def _git_pretty(repo, split_lines=False):
    """Get the latest commit hash, author, and date for the repo `repo`."""
    cwd = os.getcwd()
    os.chdir(repo)

    gitout = subprocess.check_output(
        [shutil.which("git"), "log", "--pretty", "-n1", repo]
    ).decode("utf-8").split("\n")[:3]

    if split_lines is False:
        gitout = "  \n".join(gitout) + "\n"

    os.chdir(cwd)

    return gitout


def _find_pythonimports():
    """Find the path to the repository directory for my pythonimports repo."""
    pypaths = os.environ["PYTHONPATH"].split(":")
    pyimportpath = [path for path in pypaths if "pythonimports" in path][0]

    return pyimportpath


def _update_pythonimports_README():
    """Print out the documentation for all .py files in pythonimports repo."""
    import balance_queue as balance_queue
    import myfigs as myfigs
    import mymaps as mymaps
    import myslurm as myslurm
    import pythonimports as pyimp
    import my_r as my_r
    import myclasses as myclasses

    # get commit hash
    pyimportpath = pyimp._find_pythonimports()
    commit_hash = pyimp._git_pretty(pyimportpath)
    
    # get help documentation
    docs = []
    for mod in [pyimp, mymaps, myfigs, balance_queue, myslurm, my_r, myclasses]:
        doc = pydoc.render_doc(mod, renderer=pydoc.plaintext).split('\n')[:-4]  # exclude file name
        doc[0] = '### ' + doc[0] + '\n```' # markdown header for each .py file, ticks to print in code block
        doc.append('```')
        doc.append('\n')
        docs.extend(doc)        

    # add hash to docs
    docs.insert(0, f'help documentation as of \n\n{commit_hash}\n----')

    file = op.join(pyimportpath, 'README.md')
    with open(file, 'w') as o:
        o.write('\n'.join(docs))

    return file


def latest_commit(repopath=None):
    """Print latest commit upon import for git repo in `repopath`, default `_find_pythonimports()`."""
    import pythonimports as pyimp

    if repopath is None:
        repopath = pyimp._find_pythonimports()

    try:
        env = 'conda env: %s\n' % os.environ['CONDA_DEFAULT_ENV']
    except KeyError as e:
        env = ''

    gitout = pyimp._git_pretty(repopath)
    current_datetime = "Today:\t" + time.strftime("%B %d, %Y - %H:%M:%S %Z") + "\n"
    version = "python version: " + sys.version.split()[0] + "\n"
    hashes = "##################################################################\n"

    print(
        hashes
        + current_datetime
        + version + f'{env}\n'
        + f"Current commit of {op.basename(repopath)}:\n"
        + gitout
        + hashes
    )

    pass


def wrap_defaultdict(instance, times=1):
    """Wrap an `instance` an arbitrary number of `times` to create nested defaultdict.
    
    Parameters
    ----------
    instance - e.g., list, dict, int, collections.Counter
    times - the number of nested keys above `instance`; if `times=3` dd[one][two][three] = instance
    
    Notes
    -----
    using `x.copy` allows pickling (loading to ipyparallel cluster or pkldump)
        - thanks https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
    """
    from collections import defaultdict

    def _dd(x):
        return defaultdict(x.copy)

    dd = defaultdict(instance)
    for i in range(times-1):
        dd = _dd(dd)

    return dd


def unwrap_dictionary(nested_dict, progress_bar=False):
    """Instead of iterating a nested dict, spit out all keys and the final value.
    
    Example
    -------
    # pack up a nested dictionary
    x = wrap_defaultdict(None, 3)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x[i][j][k] = random.random()
                
    # unwrap the pretty way
    for (i,j,k),val in upwrap_dictionary(x):
        # do stuff
        
    # unwrap the ugly way
    for i,jdict in x.items():
        for j,kdict in jdict.items():
            for k,val in kdict.items():
                # do stuffs

    Notes
    -----
    - thanks https://stackoverflow.com/questions/68322685/how-do-i-explode-a-nested-dictionary-for-assignment-iteration
    """
    if progress_bar is True:
        iterator = pbar(nested_dict.items())
    else:
        iterator = nested_dict.items()
    
    # iterate over the top-level dictionary
    for k, v in iterator:
        if isinstance(v, dict):
            # it's a nested dictionary, so recurse
            for ks, v2 in unwrap_dictionary(v):
                # ks is a tuple of keys, and we want to
                # prepend k, so we convert it into a tuple
                yield (k,)+ks, v2
        else:
            # make sure that in the base case
            # we're still yielding the keys as a tuple
            yield (k,), v
    pass
