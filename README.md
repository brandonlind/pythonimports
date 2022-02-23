help documentation as of 

commit 0944645b99e0199b3740ee7b83eab44936ee0829  
Author: Brandon Lind <lind.brandon.m@gmail.com>  
Date:   Wed Feb 23 13:59:57 2022 -0500

----
### Python Library Documentation: module pythonimports
```

NAME
    pythonimports

CLASSES
    builtins.object
        ColorText
    
    class ColorText(builtins.object)
     |  ColorText(text: str = '')
     |  
     |  Use ANSI escape sequences to print colors +/- bold/underline to bash terminal.
     |  
     |  Notes
     |  -----
     |  execute ColorText.demo() for a printout of colors.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, text: str = '')
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __repr__(self)
     |      Return repr(self).
     |  
     |  __str__(self)
     |      Return str(self).
     |  
     |  blue(self)
     |  
     |  bold(self)
     |  
     |  custom(self, *color_hex)
     |      Print in custom color, `color_hex` - either actual hex, or tuple(r,g,b)
     |  
     |  cyan(self)
     |  
     |  fail(self)
     |  
     |  gray(self)
     |  
     |  green(self)
     |  
     |  ltblue(self)
     |  
     |  ltgray(self)
     |  
     |  ltred(self)
     |  
     |  pink(self)
     |  
     |  purple(self)
     |  
     |  underline(self)
     |  
     |  warn(self)
     |  
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |  
     |  demo() from builtins.type
     |      Prints examples of all colors in normal, bold, underline, bold+underline.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    cd = chdir(path)
        Change the current working directory to the specified path.
        
        path may always be specified as a string.
        On some platforms, path may also be specified as an open file descriptor.
          If this functionality is unavailable, using it raises an exception.
    
    create_fundict(function, args={}, kwargs={})
        Create a fundict for `parallel_read()`.
        
        For each function `function` that is to be applied during parallel execution of `parallel_read()`, 
        use a dictionary for `args` and `kwargs` to specify args and kwargs, respectively. These are used
        to load objects to engines and when function is called.
    
    flatten(list_of_lists, unique=False) -> list
        Return a single list of values from each value in a list of lists.
        
        Parameters
        ----------
        - list_of_lists - a list where each element is a list
        - unique - bool; True to return only unique values, False to return all values.
    
    formatclock(hrs: Union[datetime.timedelta, float], exact=False) -> str
        For a given number of hours, format a clock: days-hours:mins:seconds.
        
        Parameters
        ----------
        hrs - either a float (in hours) or a datetime.timedelta object (which is converted to hours)
        exact - if False, return clock rounded up by partitions on Compute Canada
                if True, return clock with exactly days/min/hrs/seconds
    
    fs(directory: str, pattern='', endswith='', startswith='', exclude=None, dirs=None, bnames=False) -> list
        Get a list of full path names for files and/or directories in a DIR.
        
        pattern - pattern that file/dir basename must have to keep in return
        endswith - str that file/dir basename must have to keep
        startswith - str that file/dir basename must have to keep
        exclude - str that will eliminate file/dir from keep if in basename
        dirs - bool; True if keep only dirs, False if exclude dirs, None if keep files and dirs
        bnames - bool; True if return is file basenames, False if return is full file path
    
    get_client(profile='default', targets=None, **kwargs) -> tuple
        Get lview,dview from ipcluster.
    
    get_skipto_df(f: str, skipto: int, nrows: int, sep='\t', index_col=None, header='infer', **kwargs) -> pandas.core.frame.DataFrame
        Retrieve dataframe in parallel so that all rows are captured when iterating.
        
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
    
    getdirs(paths: Union[str, list], verbose=False, **kwargs) -> list
        Recursively get a list of all subdirs from given path.
    
    getmostrecent(files: list, remove=False) -> Union[str, NoneType]
        From a list of files, determine most recent.
        
        Optional to delete non-most recent files.
    
    gettimestamp(f: Union[str, list]) -> str
        Get ctime from a file path.
    
    head(df: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame
        Return head of pandas.DataFame.
    
    keys(dikt: dict) -> list
        Get a list of keys in a dictionary.
    
    latest_commit(repopath=None)
        Print latest commit upon import for git repo in `repopath`, default `_find_pythonimports()`.
    
    ls(directory: str) -> list
        Get a list of file basenames from DIR.
    
    luni(mylist: list) -> int
        Return length of unique values from list.
    
    make_jobs(fxn, inputs: list, lview) -> list
        Send each arg from inputs to a function command; async.
    
    makedir(directory: str) -> str
        If directory doesn't exist, create it.
        
        Return directory.
    
    ncol(df) -> int
        Return number of cols in pandas.DataFrame.
    
    nrow(df) -> int
        Return number of rows in pandas.DataFrame.
    
    parallel_read(f: str, linenums=None, nrows=None, header=0, lview=None, dview=None, verbose=True, desc=None, assert_rowcount=True, reset_index=True, maintain_dataframe=True, **kwargs) -> Union[list, pandas.core.frame.DataFrame]
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
        
        >>> fundict = create_fundict(function_x, args={}, kwargs={})
        
        # step 3) run parallel_read()
        >>> df = parallel_read(f, lview=lview, dview=dview, assert_rowcount=False, **dict(functions=fundict))
    
    pkldump(obj, f: str, protocol=5) -> None
        Save object to .pkl file.
    
    pklload(path: str)
        Load object from a .pkl file
    
    printmd(string: str) -> None
        For jupyter notebook, print as markdown.
        
        Useful for for-loops, etc
    
    quick_write(df, dst, sep='\t', header=True, index=False) -> None
        Quickly write a pd.DataFrame to file, much faster than .to_csv for large files.
    
    read(file: str, lines=True, ignore_blank=True) -> Union[str, list]
        Read lines from a file.
        
        Return a list of lines, or one large string
    
    rsync(src, dst, options='-azv', different_basenames=False, assert_remote_transfer=True) -> list
        Execute rsync command; can execute via ipyparallel engines.
        
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
    
    send_chunks(fxn, elements, thresh, lview, kwargs={})
        Send a list of args from inputs to a function command; async.
    
    setindex(df, colname: str) -> pandas.core.frame.DataFrame
        Set index of pandas.DataFrame to values in a column, remove col.
    
    sleeping(counts: int, desc='sleeping', sleep=1) -> None
        Basically a sleep timer with a progress bar; counts up to `counts`, interval = 1sec.
    
    suni(mylist: list) -> list
        Retrun sorted unique values from list.
    
    table(lst: list, exclude=[]) -> dict
        Count each item in a list.
        
        Return a dict with key for each item, val of count
    
    timer(func)
        Decorator to report time to complete function `func`.
    
    uni(mylist: list) -> list
        Return unique values from list.
    
    unwrap_dictionary(nested_dict)
        Instead of iterating a nested dict, spit out all keys and the final value.
        
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
    
    update(args: list) -> None
        For jupyter notebook, clear printout and print something new.
        
        Good for for-loops etc.
    
    values(dikt: dict) -> list
        Get a list of values in a dictionary.
    
    watch_async(jobs: list, phase=None, desc=None) -> None
        Wait until all ipyparallel jobs `jobs` are done executing, show progress bar.
    
    wrap_defaultdict(instance, times=1)
        Wrap an `instance` an arbitrary number of `times` to create nested defaultdict.
        
        Parameters
        ----------
        instance - e.g., list, dict, int, collections.Counter
        times - the number of nested keys above `instance`; if `times=3` dd[one][two][three] = instance
        
        Notes
        -----
        using `x.copy` allows pickling (loading to ipyparallel cluster or pkldump)
            - thanks https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict

DATA
    Optional = typing.Optional
    Union = typing.Union
    colorConverter = <matplotlib.colors.ColorConverter object>
    nb = functools.partial(<class 'tqdm.std.tqdm'>, bar_format='{l_bar}{ba...
    pbar = functools.partial(<class 'tqdm.std.tqdm'>, bar_format='{l_bar}{...
    trange = functools.partial(<function trange at 0x2ad91c20d310>, bar_fo...

```


### Python Library Documentation: module mymaps
```

NAME
    mymaps

FUNCTIONS
    basemap(extent, shapefiles=None, figsize=(8, 15))
        Build basemap +/- range shapefile.
        
        Parameters
        ----------
        - extent - the geographic extent to be displayed
        - shapefiles - a list of tuples, each tuple is (color, shapefile_path.shp)
        
        # douglas-fir shortcuts
        coastrange = '/data/projects/pool_seq/environemental_data/shapefiles/Coastal_DF.shp'
        intrange = '/data/projects/pool_seq/environemental_data/shapefiles/Interior_DF.shp'
            # df_shapfiles = zip(['lime', 'purple'], [coastrange, intrange])
            # extent=[-130, -112.5, 37.5, 55.5]
            # zoom out [-130, -108.5, 32.5, 55.5]
        
        # jack pine shortcuts
        extent=[-119.5, -58, 41, 60], figsize=(15,10),
        shapefiles=[('green', '/data/projects/pool_seq/environemental_data/shapefiles/jackpine.shp')]
    
    draw_pie_marker(ratios, xcoord, ycoord, sizes, colors, ax, edgecolors='black', slice_edgecolors='none', alpha=1, edge_linewidths=1.5, slice_linewidths=1.5, zorder=10, transform=False, label=None, edgefactor=1)
        Draw a pie chart at coordinates `[xcoord,ycoord]` on `ax`.
        
        Parameters
        ----------
        ratios : a list of ratios to create pie slices eg `[1,1]` would be 50% each slice, `[1,1,1]` would be 33% each slice
        xcoord, ycoord : coordinates to plot the pie graph marker
        sizes : the size of the circle containing the pie graph
        ax : `GeoAxesSubplot` class object. Use like `ax` from matplotlib
        edgecolors : color of the pie graph circumference
        slice_edgecolors : color of the lines that separate slices within the pie graph
        zorder : layer order
        transform : bool; transform coordinates to cylindrical projection
        label : str label for the pie graph at `[x,y]`
        
        TODO
        ----
        - offset label
        
        Notes
        -----
        - thanks stackoverflow! https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    
    plot_pie_freqs(locus, snpinfo, envinfo, saveloc=None, use_popnames=False, popcolors=None, bmargs={}, **kwargs)
        Create geographic map, overlay pie graphs (ref/alt allele freqs).

```


### Python Library Documentation: module myfigs
```

NAME
    myfigs - Personalized functions to build figures.

FUNCTIONS
    histo_box(data, xticks_by=10, title=None, xlab=None, ylab='count', col=None, fontsize=20, y_pad=1.3, histbins='auto', saveloc=None, rotation=0, **kwargs)
        Create histogram with boxplot in top margin.
        
        https://www.python-graph-gallery.com/24-histogram-with-a-boxplot-on-top-seaborn
    
    makesweetgraph(x=None, y=None, cmap='jet', ylab=None, xlab=None, bins=100, saveloc=None, figsize=(5, 4), snsbins=60, title=None, xlim=None, ylim=None, vlim=(None, None)) -> None
        Make 2D histogram with marginal histograms for each axis.
        
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
    
    save_pdf(saveloc)
        After creating a figure in jupyter notebooks, save as PDFs at `saveloc`.
    
    slope_graph(x, *y, labels=['x', 'y'], figsize=(3, 8), positive_color='black', negative_color='tomato', labeldict=None, saveloc=None, title=None, legloc='center', colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'], markers=None, addtolegend=None, ylabel='importance rank', ascending=False, legendcols=None, bbox_to_anchor=(0.5, -0.05))
        Visually display how rank order of .index changes between arbitrary number of pd.Series, `x` and *`y`.
        
        Parameters
        ----------
        x - pd.Series; shares index with all of `*y`
        *y - at least one pd.Series with which to visualize rank with `x`
        labels - list of length = len(y) + 1
        positive_color & negative_color - color of positive (≥0) rank change between series, color of negative slope
        labeldict - color of label, label is from pd.Series.index
        saveloc - location to save figure
        title - title of figure
        legloc - location of legend, passed to `ax.legend()`
        colors - list of colors to apply to each of the set {x, *y} in the order of x+*y
        markers - the marker shape to apply, one for each of the set {x, *y} in the order of x+*y
        addtolegend - tuple of (list of marker_type, list of marker_label_for_legend)
        ylabel - label for the y-axis
        ascending - bool; if False, lowest value gets lower rank (1 being high rank, and eg 20 being lower rank)
        
        Notes
        -----
        - thanks https://cduvallet.github.io/posts/2018/03/slopegraphs-in-python

DATA
    colorConverter = <matplotlib.colors.ColorConverter object>

```


### Python Library Documentation: module balance_queue
```

NAME
    balance_queue - Distribute priority jobs among accounts. To distribute non-priority jobs see myslurm.Squeue.balance.

DESCRIPTION
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

FUNCTIONS
    adjustjob(acct, jobid)
        Move job from one account to another.
    
    announceacctlens(accounts, fin, priority=True)
        How many priority jobs does each account have?
        
        Positional arguments:
        accounts - dictionary with key = account_name, val = list of jobs (squeue output)
        fin - True if this is the final job announcement, otherwise the first announcement
    
    choose_accounts(accts)
    
    get_avail_accounts(parentdir=None, save=False)
        Query slurm with sshare command to determine accounts available.
        
        If called with parentdir=None, return all available accounts.
            - Meant to be called from command line outside of pipeline. See also sys.argv input.
        If called with parentdir='choose', allow user to choose accounts.
            - Meant to be called from command line outside of pipeline. See also sys.argv input.
        If called with save=True, confirm each account with user and save .pkl file in parentdir.
            - save=True is only called from 00_start.py
        
        Returns a list of accounts to balance queue.
    
    getaccounts(sq, stage, user_accts)
        Count the number of priority jobs assigned to each account.
        
        Positional arguments:
        sq - list of squeue slurm command jobs, each line is str.split()
           - slurm_job_id is zeroth element of str.split()
        stage - stage of pipeline, used as keyword to filter jobs in queue
        user_accts - list of slurm accounts to use in balancing
    
    getbalance(accounts, num)
        Determine how many jobs should be given from one account to another.
        
        Positional arguments:
        accounts - dictionary with key = account_name, val = list of jobs (squeue output)
        num - number of accounts to balance among (this needs to be changed to object not number)
    
    main(keyword, parentdir)
    
    redistribute_jobs(accts, user_accts, balance)
        Redistribute priority jobs to other accounts without high priority.
        
        Positional arguments:
        accts - dict: key = account, value = dict with key = pid, value = squeue output
        user_accts - list of all available slurm accounts
        balance  - int; ceiling number of jobs each account should have after balancing

```


### Python Library Documentation: module myslurm
```

NAME
    myslurm - Python commands to interface with slurm queue and seff commands.

CLASSES
    builtins.object
        SQInfo
        Seff
        Squeue
    
    class SQInfo(builtins.object)
     |  SQInfo(jobinfo)
     |  
     |  Convert each line returned from `squeue -u $USER`.
     |  Example jobinfo    (index number of list)
     |  ---------------
     |  ('38768536',       0
     |   'lindb',          1
     |   'def-jonmee_cpu', 2
     |   'batch_0583',     3
     |   'PD',             4
     |   'N/A',            5
     |   '2-00:00:00',     6
     |   '1',              7
     |   '48',             8
     |   'N/A',            9
     |   '50M',            10
     |   '(Priority)')     11
     |  
     |  Methods defined here:
     |  
     |  __init__(self, jobinfo)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __repr__(self)
     |      Return repr(self).
     |  
     |  account(self)
     |  
     |  cpus(self)
     |  
     |  job(self)
     |      Job name.
     |  
     |  mem(self, units='MB')
     |  
     |  nodes(self)
     |      Compute nodes.
     |  
     |  pid(self)
     |      SLURM_JOB_ID.
     |  
     |  reason(self)
     |  
     |  start(self)
     |      Job start time.
     |  
     |  state(self)
     |      Job state - eg pending, closing, running, failed/completed + exit code.
     |  
     |  status(self)
     |  
     |  time(self)
     |      Remaining time.
     |  
     |  user(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Seff(builtins.object)
     |  Seff(slurm_job_id)
     |  
     |  Parse info output by `seff $SLURM_JOB_ID`.
     |  
     |  example output from os.popen __init__ call
     |  
     |  ['Job ID: 38771990',
     |  'Cluster: cedar',
     |  'User/Group: lindb/lindb',
     |  'State: COMPLETED (exit code 0)',
     |  'Nodes: 1',
     |  'Cores per node: 48',
     |  'CPU Utilized: 56-18:58:40',
     |  'CPU Efficiency: 88.71% of 64-00:26:24 core-walltime',
     |  'Job Wall-clock time: 1-08:00:33',
     |  'Memory Utilized: 828.22 MB',
     |  'Memory Efficiency: 34.51% of 2.34 GB']
     |  
     |  Methods defined here:
     |  
     |  __init__(self, slurm_job_id)
     |      Get return from seff command.
     |  
     |  __repr__(self)
     |      Return repr(self).
     |  
     |  core_walltime(self, unit='clock') -> str
     |      Get time that CPUs were active (across all cores).
     |  
     |  cpu_e(self) -> str
     |      Get CPU efficiency (cpu_u() / core_walltime())
     |  
     |  cpu_u(self, unit='clock') -> str
     |      Get CPU time utilized by job (actual time CPUs were active across all cores).
     |  
     |  mem(self, units='MB', per_core=False) -> float
     |      Get memory unitilized by job (across all cores, or per core).
     |  
     |  mem_e(self) -> str
     |      Get the memory efficiency (~ mem / mem_req)
     |  
     |  mem_req(self, units='MB') -> Union[float, int]
     |      Get the requested memory for job.
     |  
     |  pid(self) -> str
     |      Get SLURM_JOB_ID.
     |  
     |  state(self) -> str
     |      Get state of job (R, PD, COMPLETED, etc).
     |  
     |  walltime(self, unit='clock') -> str
     |      Get time that job ran after starting.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Squeue(builtins.object)
     |  Squeue(**kwargs)
     |  
     |  dict-like container class for holding and updating slurm squeue information.
     |  
     |  Methods - most methods can be filtered by passing the kwargs from Squeue._filter_jobs to the method
     |  -------
     |  states - return a list of job states (eg running, pending)
     |  jobs - return a list of job names
     |  pids - return a list of pids (SLURM_JOB_IDs)
     |  accounts - return a list of accounts
     |  cancel - cancel entire queue or specific jobs for specific accounts (if user is True, cancel all jobs)
     |  update - update time, memory, or account for specifc jobs for specific accounts
     |  balance - balance jobs in queue across available slurm accounts
     |  summary - print counts of various categories within the queue
     |  hold - hold jobs
     |  release - release held jobs
     |  keys - returns list of all pids (SLURM_JOB_IDs) - cannot be filtered with Squeue._filter_jobs
     |  items - returns item tuples of (pid,SQInfo) - cannot be filtered with Squeue._filter_jobs
     |  values - returns list of SQInfos - cannot be filtered with Squeue._filter_jobs
     |  save_default_accounts - save default accounts to use in Squeue.balance, cannot be filtered with Squeue._filter_jobs
     |  
     |  
     |  TODO
     |  ----
     |  TODO: address Squeue.balance() TODOs
     |  TODO: update_job needs to handle skipping over jobs that started running or closed after class instantiation
     |  TODO: update_job needs to skip errors when eg trying to increase time of job beyond initial submission
     |          eg when initially scheduled for 1 day, but update tries to extend beyond 1 day
     |          (not allowed on compute canada)
     |  TODO: make it so it can return the queue for `grepping` without needing `user` (ie all users)
     |  TODO: address `user` kwarg potential conflict between _get_sq and _filter_jobs
     |  
     |  Examples
     |  --------
     |  Get squeue information (of class SQInfo) for each job in output (stdout line) containing 'batch' or 'gatk'.
     |  
     |  >>> sq = Squeue(grepping=['batch', 'gatk'])
     |  
     |  
     |  Get queue with "batch" in one of the columns (eg the NAME col).
     |  For theses jobs, only update jobs with "batch_001" for mem and time.
     |  
     |  >>> sq = Squeue(grepping='batch', states='PD')
     |  >>> sq.update(grepping='batch_001', minmemorynode=1000, timelimit=3-00:00:00)
     |  
     |  
     |  Cancel all jobs in queue.
     |  
     |  >>> sq = Squeue()
     |  >>> sq.cancel(user=True)
     |  # OR:
     |  >>> Squeue().cancel(user=True)
     |  
     |  
     |  Cancel jobs in queue with "batch" in one of the columns (eg the NAME col).
     |  For theses jobs only cancel job names containing "batch_001" and 3 day run time,
     |      but not the 'batch_0010' job.
     |  
     |  >>> sq = Squeue(grepping='batch', states='PD')
     |  >>> sq.cancel(grepping=['batch_001', '3-00:00:00'], exclude='batch_0010')
     |  
     |  
     |  Get jobs from a specific user.
     |  
     |  >>> sq = Squeue(user='some_user')
     |  
     |  Get a summary of queue.
     |  
     |  >>> sq = Squeue()
     |  >>> sq.summary()
     |  
     |  Methods defined here:
     |  
     |  __cmp__(self, other)
     |  
     |  __contains__(self, pid)
     |  
     |  __delitem__(self, key)
     |  
     |  __getitem__(self, key)
     |  
     |  __init__(self, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __len__(self)
     |  
     |  __repr__(self)
     |      Return repr(self).
     |  
     |  __setitem__(self, key, item)
     |  
     |  accounts(self, **kwargs)
     |      Get a list of accounts, subset with kwargs.
     |  
     |  balance(self, parentdir='HOME', **kwargs)
     |      Evenly distribute pending jobs across available slurm sbatch accounts.
     |      
     |      Parameters
     |      ----------
     |      parentdir - used in balance_queue to look for `accounts.pkl`; see balance_queue.__doc__
     |          - `parentdir` can be set to 'choose' to manually choose slurm accounts from those available
     |          - if `parentdir` is set to `None`, then all available accounts will be used to balance
     |          - otherwise, `parentdir` can be set to `some_directory` that contains "accounts.pkl" saved from:
     |              balance_queue.get_avail_accounts(some_directory, save=True)
     |          - default = os.environ['HOME']
     |      kwargs - see Squeue._filter_jobs.__doc__
     |      
     |      Notes
     |      -----
     |      - printouts (and docstrings of balance_queue.py + functions) will refer to 'priority jobs', since this
     |          was the purpose of the app. However, Squeue.balance() can pass jobs that are not necessarily of
     |          priority status.
     |      
     |      TODO
     |      ----
     |      - balance even if all accounts have jobs
     |          As of now, if all accounts have jobs (whether filtering for priority status or not),
     |          Squeue.balance() will not balance. This behavior is inherited from balance_queue.py. The quickest
     |          work-around is to use `onaccount` to balance jobs on a specific account (eg the one with the most jobs).
     |  
     |  cancel(self, **kwargs)
     |      Cancel jobs in slurm queue, remove job info from Squeue class.
     |  
     |  copy(self)
     |  
     |  hold(self, **kwargs)
     |      Hold jobs. Parameters described in `Squeue._update_job.__doc__`.
     |  
     |  items(self)
     |  
     |  jobs(self, **kwargs)
     |      Get a list of job names, subset with kwargs.
     |  
     |  keys(self)
     |  
     |  pending(self)
     |  
     |  pids(self, **kwargs)
     |      Get a list of pids, subset with kwargs.
     |  
     |  release(self, **kwargs)
     |      Release held jobs. Parameters described in `Squeue._update_job.__doc__`.
     |  
     |  running(self)
     |  
     |  states(self, **kwargs)
     |      Get a list of job states.
     |  
     |  summary(self, **kwargs)
     |      Print counts of states and statuses of the queue.
     |  
     |  update(self, **kwargs)
     |      Update jobs in slurm queue with scontrol, and update job info in Squeue class.
     |      
     |      kwargs - that control what can be updated (other kwargs go to Squeue._filter_jobs)
     |      ------
     |      account - the account to transfer jobs
     |      minmemorynode - total memory requested
     |      timelimit - total wall time requested
     |  
     |  values(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    sqinfo = class SQInfo(builtins.object)
     |  sqinfo(jobinfo)
     |  
     |  Convert each line returned from `squeue -u $USER`.
     |  Example jobinfo    (index number of list)
     |  ---------------
     |  ('38768536',       0
     |   'lindb',          1
     |   'def-jonmee_cpu', 2
     |   'batch_0583',     3
     |   'PD',             4
     |   'N/A',            5
     |   '2-00:00:00',     6
     |   '1',              7
     |   '48',             8
     |   'N/A',            9
     |   '50M',            10
     |   '(Priority)')     11
     |  
     |  Methods defined here:
     |  
     |  __init__(self, jobinfo)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __repr__(self)
     |      Return repr(self).
     |  
     |  account(self)
     |  
     |  cpus(self)
     |  
     |  job(self)
     |      Job name.
     |  
     |  mem(self, units='MB')
     |  
     |  nodes(self)
     |      Compute nodes.
     |  
     |  pid(self)
     |      SLURM_JOB_ID.
     |  
     |  reason(self)
     |  
     |  start(self)
     |      Job start time.
     |  
     |  state(self)
     |      Job state - eg pending, closing, running, failed/completed + exit code.
     |  
     |  status(self)
     |  
     |  time(self)
     |      Remaining time.
     |  
     |  user(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    adjustjob(acct, jobid)
        Move job from one account to another.
    
    clock_hrs(clock: str, unit='hrs') -> float
        From a clock (days-hrs:min:sec) extract hrs or days as float.
    
    create_watcherfile(pids, directory, watcher_name='watcher', email='brandon.lind@ubc.ca')
        From a list of dependency pids, sbatch a file that will email once pids are completed.
        
        TODO
        ----
        - incorporate code to save mem and time info
    
    get_mems(seffs: dict, units='MB', plot=True) -> list
        From output by `get_seff()`, extract mem in `units` units; histogram if `plot` is True.
        
        Parameters
        ----------
        - seffs : dict of any key with values of class Seff
        - units : passed to Seff._convert_mem(). options: GB, MB, KB
    
    get_seff(outs: list, desc=None)
        From a list of .out files (ending in f'_{SLURM_JOB_ID}.out'), get seff output.
    
    get_times(seffs: dict, unit='hrs', plot=True) -> list
        From dict(seffs) [val = seff output], get times in hours.
        
        fix: add in other clock units
    
    getpid(out: str) -> str
        From an .out file with structure <anytext_JOBID.out>, return JOBID.
    
    getsq = _getsq(grepping=None, states=[], user=None, **kwargs)
        Get and parse slurm queue according to kwargs criteria.
    
    sbatch(shfiles: Union[str, list], sleep=0, printing=False, outdir=None) -> list
        From a list of .sh shfiles, sbatch them and return associated jobid in a list.
        
        Notes
        -----
        - assumes that the job name that appears in the queue is the basename of the .sh file
            - eg for job_187.sh, the job name is job_187
            - this convention is used to make sure a job isn't submitted twice
        - `sbatch` therefore assumes that each job has a unique job name

DATA
    Union = typing.Union

```


### Python Library Documentation: module my_r
```

NAME
    my_r - Personalized functions to interact with R.

CLASSES
    builtins.object
        SetupR
    
    class SetupR(builtins.object)
     |  SetupR(home=None, ld_library_path='')
     |  
     |  Set up rpy2 class object for interacting with R.
     |  
     |  Methods defined here:
     |  
     |  __call__(self, arg)
     |      Call self as a function.
     |  
     |  __cmp__(self, other)
     |  
     |  __getattr__(self, attr)
     |  
     |  __init__(self, home=None, ld_library_path='')
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __repr__(self)
     |      Return repr(self).
     |  
     |  data(self, dataname, _return=False)
     |      Load data object, `dataname` into R namespace.
     |  
     |  library(self, lib)
     |      Load library into R namespace.
     |  
     |  remove(self, *args, all=False)
     |      Delete objects or functions from namespace.
     |  
     |  session_info(self)
     |      Get R session info.
     |  
     |  source(self, path)
     |      Source R script.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

```

