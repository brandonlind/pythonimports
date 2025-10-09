help documentation as of 

commit f447ba2c8bba8b2d996589427d424907a41a8e4a

Author: Brandon Lind <lind.brandon.m@gmail.com>

Date: Tue Oct 7 21:56:21 2025 -400

----
### Python Library Documentation: module pythonimports
```

NAME
    pythonimports

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

    formatclock(hrs: Union[datetime.timedelta, float], exact=True) -> str
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

    getdirs(paths: Union[str, list], verbose=False, exclude=None, **kwargs) -> list
        Recursively get a list of all subdirs from given path.

        Parameters
        ----------
        paths - a path (str) or list of paths to explore
        verbose - whether to print all directories when found
        kwargs - same kwargs used in `fs` to filter directories that are found

    getmostrecent(files: list, remove=False) -> Optional[str]
        From a list of files, determine most recent.

        Optional to delete non-most recent files.

    gettimestamp(f: Union[str, list]) -> str
        Get ctime from a file path.

    head(df: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame
        Return head of pandas.DataFame.

    keys(dikt: dict) -> list
        Get a list of keys in a dictionary.

    latest_commit(repopath=None, html=True)
        Display latest commit upon import for git repo in `repopath`, make repo clickable.

        Parameters
        ----------
        repopath : str
            path to directory with .git - default `_find_pythonimports()`
        html : bool
            whether to display output with HTML (for jupyter notebooks) or to print output (for command line)

    lower_tri(df)
        Get values from the lower triangle of a pandas dataframe.

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

    print_directory_tree(root_dir, print_files=False, _indent='')
        Print the directory tree for everything nested in `root_dir`.

        Parameters
        ----------
        root_dir : str | Path
            a directory; must satisfy `os.path.isdir(root_dir)`
        print_files : bool
            whether to print filenames in addition to directory names within the tree
        _indend : str
            used internally

    printmd(string: str) -> None
        For jupyter notebook, print as markdown.

        Useful for for-loops, etc

    quick_write(df, dst, sep='\t', header=True, index=False) -> None
        Quickly write a pd.DataFrame to file, much faster than .to_csv for large files.

    read(file: str, lines=True, ignore_blank=False) -> Union[str, list]
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

    sleeping(counts: int, desc='sleeping', sleep=1, raise_e=False) -> None
        Basically a sleep timer with a progress bar; counts up to `counts`, interval = 1sec.

    start_engines(targets=None, cluster_id='', n=None, **kwargs)
        Start ipcluster engines from within a python script.

        Notes
        -----
        For some reason, executing this function within a notebook causes fork errors on machines.

    suni(mylist: list) -> list
        Retrun sorted unique values from list.

    table(lst: list, exclude=[]) -> dict
        Count each item in a list.

        Return a dict with key for each item, val of count

    timer(func)
        Decorator to report time to complete function `func`.

    uni(mylist: list) -> list
        Return unique values from list.

    unwrap_dictionary(nested_dict, progress_bar=False)
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

    watch_async(jobs: list, phase=None, desc=None, color=None, verbose=True) -> None
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
        Optional[X] is equivalent to Union[X, None].

    Union = typing.Union
        Union type; Union[X, Y] means either X or Y.

        On Python 3.10 and higher, the | operator
        can also be used to denote unions;
        X | Y means the same thing to the type checker as Union[X, Y].

        To define a union, use e.g. Union[int, str]. Details:
        - The arguments must be types and there must be at least one.
        - None as an argument is a special case and is replaced by
          type(None).
        - Unions of unions are flattened, e.g.::

            assert Union[Union[int, str], float] == Union[int, str, float]

        - Unions of a single argument vanish, e.g.::

            assert Union[int] == int  # The constructor actually returns int

        - Redundant arguments are skipped, e.g.::

            assert Union[int, str, int] == Union[int, str]

        - When comparing unions, the argument order is ignored, e.g.::

            assert Union[int, str] == Union[str, int]

        - You cannot subclass or instantiate a union.
        - You can use Optional[X] as a shorthand for Union[X, None].

    bar_format = '{l_bar}{bar:15}{r_bar}'
    colorConverter = <matplotlib.colors.ColorConverter object>
    nb = functools.partial(<class 'tqdm.std.tqdm'>, bar_format='{l_bar}{ba...
    pbar = functools.partial(<class 'tqdm.std.tqdm'>, bar_format='{l_bar}{...
    trange = functools.partial(<function trange at 0x7f2c0a058360>, bar_fo...

```


### Python Library Documentation: module mymaps
```

NAME
    mymaps - Functions for mapping / GIS.

CLASSES
    cartopy.io.img_tiles.GoogleTiles(cartopy.io.img_tiles.GoogleWTS)
        ESRIShadedReliefTint

    class ESRIShadedReliefTint(cartopy.io.img_tiles.GoogleTiles)
     |  ESRIShadedReliefTint(color=(70, 110, 160), strength=0.4, desaturate=0.8, **kwargs)
     |
     |  ESRI World Shaded Relief (pre-colored) with a customizable color tint.
     |
     |  Parameters
     |  ----------
     |  color : (R, G, B)
     |      The target tint color, e.g., (40, 100, 170) for a cool blue.
     |  strength : float in [0, 1]
     |      How strongly to push toward the tint color (0=no tint, 1=full tint).
     |  desaturate : float
     |      Factor to desaturate the original tile before tinting (1.0 = unchanged).
     |
     |  Method resolution order:
     |      ESRIShadedReliefTint
     |      cartopy.io.img_tiles.GoogleTiles
     |      cartopy.io.img_tiles.GoogleWTS
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __init__(self, color=(70, 110, 160), strength=0.4, desaturate=0.8, **kwargs)
     |      Parameters
     |      ----------
     |      desired_tile_form: optional
     |          Defaults to 'RGB'.
     |      style: optional
     |          The style for the Google Maps tiles.  One of 'street',
     |          'satellite', 'terrain', and 'only_streets'.  Defaults to 'street'.
     |      url: optional
     |          URL pointing to a tile source and containing {x}, {y}, and {z}.
     |          Such as: ``'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg'``
     |
     |  get_image(self, tile)
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __abstractmethods__ = frozenset()
     |
     |  ----------------------------------------------------------------------
     |  Methods inherited from cartopy.io.img_tiles.GoogleWTS:
     |
     |  find_images = _find_images(self, target_domain, target_z, start_tile=(0, 0, 0))
     |
     |  image_for_domain(self, target_domain, target_z)
     |
     |  subtiles(self, x_y_z)
     |
     |  tile_bbox(self, x, y, z, y0_at_north_pole=True)
     |      Return the ``(x0, x1), (y0, y1)`` bounding box for the given x, y, z
     |      tile position.
     |
     |      Parameters
     |      ----------
     |      x
     |          The x tile coordinate in the Google tile numbering system.
     |      y
     |          The y tile coordinate in the Google tile numbering system.
     |      z
     |          The z tile coordinate in the Google tile numbering system.
     |
     |      y0_at_north_pole: optional
     |          Boolean representing whether the numbering of the y coordinate
     |          starts at the north pole (as is the convention for Google tiles)
     |          or not (in which case it will start at the south pole, as is the
     |          convention for TMS). Defaults to True.
     |
     |  tileextent(self, x_y_z)
     |      Return extent tuple ``(x0,x1,y0,y1)`` in Mercator coordinates.
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from cartopy.io.img_tiles.GoogleWTS:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object

FUNCTIONS
    add_north_arrow(ax, location_xy=(0.92, 0.08), length_km=None, color='k', lw=2.5, mutation_scale=18, text='N', font_size=12, box_alpha=1, zorder=1000)
        Add a north-pointing arrow that is geodetically correct at its placement location.

        Parameters
        ----------
        ax : Cartopy GeoAxes
        location_xy : (x, y)
            Placement in axes fraction coords (0-1). E.g., (0.92, 0.08) ~ bottom-right.
        length_km : float or None
            Arrow length in km. If None, picks a nice value ~1/8 of the map height.
        color : str
            Arrow and text color.
        lw : float
            Arrow line width.
        mutation_scale : float
            Controls arrowhead size for FancyArrowPatch.
        text : str
            Label for the arrow (usually 'N').
        font_size : int
            Font size for the label.
        box_alpha : float
            Background alpha for the label for readability.

    add_scalebar(ax: matplotlib.axes._axes.Axes, length_km=None, location_xy=(0.1, 0.05), ndiv: int = 2, height_frac: float = 0.01, linewidth: float = 2.5, color: str = 'k', font_size: int = 9, box_alpha: float = 1, units: str = 'km', arrow_location=(0.15, 0.08), zorder=1000) -> None
        Add a scale bar that is accurate at its placement location for the current map.
        Distances computed on WGS84 using a rhumb line (constant latitude for east-west bar).

        Parameters
        ----------
        ax : plt.Axes (Cartopy axes)
        length_km : float or None
            If None, choose ~1/4 of map width at the bar's latitude, rounded to {1,2,5}*10^n.
        location_xy : tuple
            Placement in axes fraction coordinates (x,y). (0,0)=bottom-left.
        ndiv : int
            Number of subdivisions (e.g., 2 → 0, mid, end ticks).
        height_frac : float
            Bar height as a fraction of the map's latitude span (visual thickness).
        linewidth : float
            Line width for the bar.
        color : str
            Color of the bar and text.
        font_size : int
            Font size for labels.
        box_alpha : float
            Opacity for the background text box.
        units : str
            Display units ("km" recommended).

    add_shapefiles_to_map(ax, shapefiles=None, face_alpha=0.1, edge_alpha=1, zorder=20, progress_bar=False, geo_kws={}, **kwargs)
        Add shapefiles to `ax`.

        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxes
        shapefiles
            - a list of tuples, each tuple is (color, /path/to/shapefile.shp)
        kwargs : dict
            - passed to cut_shapes - now only for `epsg` kwarg

        Assumes
        -------
        - assumed dependent files associated with .shp
        - generally assumes epsg 4326

    basemap(extent, projection=None, centralize_projection_on_extent=True, shaded_relief: str = 'stock', stamen_zoom=None, figsize=(10, 8), add_rivers=True, gridline_kwargs=None, gridline_width=0.5, scalebar_kwargs={}, scalebar_zorder=1000, shapefiles=None, cut_extent=None, scalebar=True, x_interval=None, y_interval=None, ticklabels=True, **kwargs)
        Create a Cartopy map for the given extent with shaded relief and an accurate scale bar.

        Parameters
        ----------
        extent : [min_lon, max_lon, min_lat, max_lat]
            Geographic extent in degrees.
        projection : ccrs.Projection or None
            If None, choose based on extent (recenter PlateCarree or Lambert Azimuthal for high-lat).
        centralize_projection_on_extent : bool
            If True and projection is None, center the projection at the extent's center longitude (and
            latitude for azimuthal).
        shaded_relief : str
            "stock" uses ax.stock_img() (low-res but offline).
            "stamen-terrain" / "stamen-toner-lite" use tile servers (requires internet).
            "none" disables a background.
        stamen_zoom : int or None
            If None, approximates a zoom from the extent width.
        figsize : (w, h)
            Matplotlib figure size in inches.
        add_rivers : bool
            whether to add river and lake centerlines from natural earth feature
        gridline_kwargs : dict
            Customization for gridlines (e.g., {"linewidth":0.5, "color":"gray"}).
        scalebar_kwargs : dict
            Args passed to add_scalebar (e.g., {"length_km":100, "location_xy":(0.1,0.08)}).
        scalebar_zorder : int
            zorder for scalebar and north arrow
        shapefiles : list
            a list of tuples, each tuple is (color, shapefile_path.shp)
        cut_extent : list
            different order than `extent`, the extent to cut shapefiles (default is an internally re-ordered `extent`)
        scalebar : bool
            whether to add a scalebar
        kwargs : dict
            passed to add_shapefiles_to_map and cut_shapes
        x_interval : [int, float]
            manually set interval of x-axis (in degrees)
        y_interval : [int, float]
            manually set interval of y-axis (in degrees)
        ticklables : bool
            whether to keep x- and y- ticklabels

        Returns
        -------
        fig, ax

    cut_shapes(shapefile, cut=True, epsg=4326, cut_extent=None, query=None)
        Cut out overlapping polygons from non-overlapping polygons.

        Notes
        -----
        `cut_extent` is useful when original shapefile is large

    draw_pie_marker(ratios, xcoord, ycoord, sizes, colors, ax, edgecolors='black', slice_edgecolors='none', alpha=1, edge_linewidths=1.5, slice_linewidths=1.5, zorder=10, transform=False, label=None, edgefactor=1, label_kws={})
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
        label_kws : dict, passed to ax.annotate

        TODO
        ----
        - offset label

        Notes
        -----
        - thanks stackoverflow! https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib

    draw_rectangle(ax, extent, facecolor='gray', alpha=0.5, linewidth=3, zorder=1500)

    gdalwarp(infile, netcdf_outfile, proj, gdalwarp_exe=None)
        Convert `infile` (eg .tif or .nc) to WGS84 `netcdf_outfile` (.nc).

        Notes
        -----
        conda install -c conda-forge gdal

        Notes
        -----
        to retrieve proj string from a file:
            `gdalsrsinfo <filename>`
            - thanks https://gis.stackexchange.com/questions/55196/how-can-i-get-the-proj4-string-or-epsg-code-from-a-shapefile-prj-file

        Parameters
        ----------
        infile
            path to input GIS file
        netcdf_outfile
            path to where to save converted file in netcdf (.nc) format
        proj
            proj string of `infile` - eg
                proj = '+proj=laea +lat_1=49.0 +lat_2=77.0 +lat_0=45             +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84             +units=m +no_defs'
        gdalwarp_exe
            path to gdalwarp executable - eg one created with conda
                ~/anaconda3/envs/gdal_env/bin/gdalwarp

    inset_map(extent, map_extent=None, shapes=[], shapefiles=None, figsize=(8, 15), projection=None, **kwargs)
        Create an black and white inset map for placing within a larger map.

        Parameters
        ----------
        extent - list
            extent of inset map - [minlong, maxlong, minlat, maxlat]
        map_extent - list
            extent of larger map for drawing box within inset map

    overlaps(polygon, polygons)
        Determine if `polygon` overlaps with any of the `polygons`.

    plot_pie_freqs(locus, snpinfo, envinfo, saveloc=None, use_popnames=False, popcolors=None, bmargs={}, **kwargs)
        Create geographic map, overlay pie graphs (ref/alt allele freqs).

    read_geofile(geofile, epsg='epsg:4326', x_dim='latitude', y_dim='longitude', debug=False)
        Read in netcdf file.

DATA
    colorConverter = <matplotlib.colors.ColorConverter object>

```


### Python Library Documentation: module myfigs
```

NAME
    myfigs - Personalized functions to build figures.

CLASSES
    builtins.object
        SeabornFig2Grid

    class SeabornFig2Grid(builtins.object)
     |  SeabornFig2Grid(seaborngrid, fig, subplot_spec)
     |
     |  Allow seaborn figure-level figs to be suplots.
     |
     |  thanks - https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot/47664533#47664533
     |
     |  Methods defined here:
     |
     |  __init__(self, seaborngrid, fig, subplot_spec)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object

FUNCTIONS
    add_legend(ldict, markers='s', legendmarkerfacecolor='fill', markeredgecolor=None, title=None, ax=None, loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=11, ncol=1, add_handles=[], face_alpha=1, edge_alpha=1)
        Add a legend to `ax`.

        Parameters
        ----------
        ldict : dict
            key = value = label used in legend labels, value = color
        markers : dict | str
            if dict, use keys in ldict. If str, specify marker for all ldict.keys()
        legendmarkerfacecolor : 'fill' | str | tuple(R, G, B, [alpha])
            if 'fill', use color in ldict. Otherwise, specify another color. use with `face_alpha`.
        markeredgecolor : str | dict
            use with `edge_alpha`. if dict, keys are ldict.keys()
        title : str
            title for legend
        ax : e.g., matplotlib.axes._axes.Axes | cartopy.mpl.geoaxes.GeoAxes
        loc : str
            location of legend with respect to bbox_to_anchor location
        bbox_to_anchor : tuple
            Box that is used to position the legend in conjunction with *loc*.
        fontsize : int
            fontsize of legend. legend title is fontsize+1
        ncol : int
            the number of columns in the legend
        face_alpha : int | dict
            opacity of facecolor. if dict, keys are ldict.keys()
        edge_alpha : int | dict
            opacity of edgecolor. if dict, keys are ldict.keys()

    adjust_box_widths(axes, fac=0.9)
        Adjust the widths of a seaborn-generated boxplot or boxenplot.

        Notes
        -----
        - thanks https://github.com/mwaskom/seaborn/issues/1076

    create_cmap(list_of_colors, name=None, grain=500)
        Create a custom color map with fine-grain transition.

    draw_xy(ax, lims=None, alpha=1, zorder=5, linewidth=0.5, color='k', linestyle='-', equal_aspect=False)
        Draw x=y line on a matplotlib ax.

    gradient_image(ax, direction=0.3, cmap_range=(0, 1), extent=(0, 1, 0, 1), **kwargs)
        Draw a gradient image based on a colormap.

        Parameters
        ----------
        ax : Axes
            The axes to draw on.
        extent
            The extent of the image as (xmin, xmax, ymin, ymax).
            By default, this is in Axes coordinates but may be
            changed using the *transform* kwarg.
        direction : float
            The direction of the gradient. This is a number in
            range 0 (=vertical) to 1 (=horizontal).
        cmap_range : float, float
            The fraction (cmin, cmax) of the colormap that should be
            used for the gradient, where the complete colormap is (0, 1).
        **kwargs
            Other parameters are passed on to `.Axes.imshow()`.
            In particular useful is *cmap*.

        Example
        -------
        >> x = # some data
        >> y = # some data
        >> fig, ax = plt.subplots()
        >> ax.scatter(x, y)
        >> _cmap = create_cmap(['white', 'blue'], grain=1000)
        >> gradient_image(ax, direction=0.5, transform=ax.transAxes, extent=(0,1,0,1),
                          cmap=_cmap, cmap_range=(0.0, 0.2))

        Notes
        -----
        thanks https://matplotlib.org/3.2.0/gallery/lines_bars_and_markers/gradient_bar.html

    histo_box(data, xticks_by=None, title=None, xlab=None, ylab=None, col=None, fontsize=12, y_pad=1.3, histbins='auto', saveloc=None, rotation=0, ax=None, jitter=True, jit=0.15, marker='.', markersize=8, zorder=0, markerfacecolor='gray', alpha=0.5, markeredgewidth=0.0, histplot_kws={}, boxplot_kws={}, height_ratios=(0.15, 0.85), xlim=None, ylim=None, ticksize=None, **kwargs)
        Create histogram with boxplot in top margin.

        Parameters
        ----------
        data
            data from which histogram and boxplot are to be made
        xticks_by - [float, int]
            interval by which xticks are made on x-axis
        title - str
            figure/ax title
        xlab - str
            label for x-axis
        ylab - str
            label for y-axis
        col - str
            column name if pandas.DataFrame is passed to `data`
        fontsize - int
            fontsize of ax text
        y_pad - [float, int]
            padding for title
        histobins - int
            number of bins for histogram
        saveloc - str
            where to save - note if `ax` is not None, then `ax` will be saved to `saveloc`
        rotation - int
            rotation for x-axis tick labels
        ax - [matplotlib.axes.Axes, matplotlib.axes._subplots.AxesSubplot]
            axis canvas upon which to create the histo boxplot
        boxplot_kws - (markersize, zorder, markerfacecolor, alpha, markeredgewidth)
            kwargs passed to seaborn.boxplot
        histplot_kws - dict
            kwargs passed to seaborn.histplot
        kwargs
            passed to plt.subplots

        Notes
        ------
            thanks https://www.python-graph-gallery.com/24-histogram-with-a-boxplot-on-top-seaborn

    jitter_fliers(g=None, axes=None, jitter_axis='x', jit=0.05)
        Add jitter to boxplot fliers.

        Parameters
        ----------
        g : seaborn.axisgrid.FacetGrid
            eg returned from seaborn.catplot etc
        axes : list of matplotlib.axes._subplots.AxesSubplot

        Notes
        -----
        Thanks - https://stackoverflow.com/questions/61638303/how-to-jitter-the-outliers-of-a-boxplot

    makesweetgraph = scatter2d(x=None, y=None, cmap='jet', ylab=None, xlab=None, bins=100, saveloc=None, figsize=(5, 4), snsbins=60, title=None, xlim=None, ylim=None, vlim=(None, None), marginal_kws={}, title_kws={}, return_cbar=False, normalization='lognorm', cbar_label=None) -> seaborn.axisgrid.JointGrid
        Make 2D scatterplot with marginal histograms for each axis.

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

    pdf_to_png(pdf, page=0, outfile=None, outdir=None)
        Convert the first page of a pdf document to a png.

        Parameters
        ----------
        pdf : str
            path to .pdf file
        page : int
            page index of pdf to save as png
        outfile : str
            path to save output png; default is '/path/to/pdf'.replace('.pdf', '.png')
        outdir : str
            if `outfile` is None, save as /path/to/outdir/basename.png

    save_pdf(saveloc)
        After creating a figure in jupyter notebooks, save as PDFs at `saveloc`.

    scatter2d(x=None, y=None, cmap='jet', ylab=None, xlab=None, bins=100, saveloc=None, figsize=(5, 4), snsbins=60, title=None, xlim=None, ylim=None, vlim=(None, None), marginal_kws={}, title_kws={}, return_cbar=False, normalization='lognorm', cbar_label=None) -> seaborn.axisgrid.JointGrid
        Make 2D scatterplot with marginal histograms for each axis.

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

    slope_graph(x, *y, labels=['x', 'y'], figsize=(3, 8), positive_color='black', negative_color='tomato', labeldict=None, shape_color=None, saveloc=None, title=None, legloc='center', colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'], markers=None, addtolegend=None, ylabel='importance rank', ascending=False, legendcols=None, bbox_to_anchor=(0.5, -0.05), ax=None, marker_size=80, all_yticks=False)
        Visually display how rank order of .index changes between arbitrary number of pd.Series, `x` and *`y`.

        Parameters
        ----------
        x - pd.Series; shares index with all of `*y`
        *y - at least one pd.Series with which to visualize rank with `x`
        labels - list of length = len(y) + 1
        positive_color & negative_color - color of positive (≥0) rank change between series, color of negative slope
        labeldict - color of label, label is from pd.Series.index
        shape_color - dict; each .index label of x and *y is key, val is color (overrides kwarg `colors`)
        saveloc - location to save figure
        title - title of figure
        legloc - location of legend, passed to `ax.legend()`
        colors - list of colors of len(*y)+1 to apply to all .index labels for each {x, *y} in the order of x+*y (if shape_color is None)
        markers - the marker shape to apply, one for each of the set {x, *y} in the order of x+*y
        addtolegend - tuple of (list of marker_type, list of marker_label_for_legend)
        ylabel - label for the y-axis
        ascending - bool; if False, lowest value gets lower rank (1 being high rank, and eg 20 being lower rank)
        ax - matplotlib.axes._subplots.AxesSubplot; in case I want a slope graph on an ax within a plt.subplot
        all_yticks - whether to show all y-axis tick labels, otherwise only every 5th tick label is shown

        Notes
        -----
        - thanks https://cduvallet.github.io/posts/2018/03/slopegraphs-in-python

    stacked_histo_box(data, x=None, y=None, height_ratios=(0.15, 0.85), figsize=(9, 6), ax=None, box_linewidth=1.5, boxplot_kws={}, hist_bins=None, histplot_kws={}, **kwargs)
        Create stacked histogram with multi-category boxplot in top margin.

        Parameters
        ----------
        data : pd.DataFrame.astype({x : float | int, y : str | object})
            dataframe with columns `x` and `y`, where `y` is the categories used for plotting
        height_ratios : tuple
            height (y-axis) ratios of boxplot and histogram
        figsize : tuple
            figure size
        ax : None | matplotlib.axes.Axes | matplotlib.axes._subplots.AxesSubplot
            axis for plotting
        box_linewidth : float
            edge width of boxplot, median line, whiskers, and caps
        boxplot_kws : dict
            passed to seaborn.boxplot
        histplot_kws : dict
            passed to AxesSubplot.hist
        kwargs : dict
            passed to plt.subplots

        Returns
        -------
        (ax_box, ax_hist)
            AxesSubplot's for histogram and

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
    # accounts are those slurm accounts that do not end with '_gpu' returned from the command:
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
        accounts - dictionary with key = account_name, val = dict with key = pid and val = job info (squeue output)
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
        sq - Squeue class object, dict-like: keys for slurm_job_ids, values=info
        stage - stage of pipeline, used as keyword to filter jobs in queue
        user_accts - list of slurm accounts to use in balancing

    getbalance(accounts, num)
        Determine how many jobs should be given from one account to another.

        Positional arguments:
        accounts - dictionary with key = account_name, val = dict with key = pid and val = job info (squeue output)
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
        Seffs
        Squeue

    class SQInfo(builtins.object)
     |  SQInfo(jobinfo)
     |
     |  Convert each line returned from `squeue -u $USER`.
     |
     |  Assumed
     |  -------
     |  SQUEUE_FORMAT="%i %u %a %j %t %S %L %D %C %b %m %N (%r) %P"
     |                  0  1  2  3  4  5  6  7  8  9 10 11  12  13
     |
     |  Example jobinfo    index
     |  ---------------    -----
     |  ('29068196',         0
     |   'b.lindb',          1
     |   'lotterhos',        2
     |   'batch_0583',       3
     |   'R',                4
     |   'N/A',              5
     |   '9:52:32',          6
     |   '1',                7
     |   '56',               8
     |   'N/A',              9
     |   '2000M',           10
     |   'd0036',           11
     |   '(Priority)')      12
     |   'short'            13
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
     |  cancel(self, verbose=True)
     |      Cancel this job.
     |
     |  mem(self, units='MB')
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object

    class Seff(builtins.object)
     |  Seff(slurm_job_id)
     |
     |  Parse info output by `seff $SLURM_JOB_ID`.
     |
     |  example output from os.popen __init__ call
     |
     |  ['Job ID: 38771990',                                      0
     |  'Cluster: cedar',                                         1
     |  'User/Group: lindb/lindb',                                2
     |  'State: COMPLETED (exit code 0)',                         3
     |  'Nodes: 1',                                               4  # won't always show up
     |  'Cores per node: 48',                                     5 -6
     |  'CPU Utilized: 56-18:58:40',                              6 -5
     |  'CPU Efficiency: 88.71% of 64-00:26:24 core-walltime',    7 -4
     |  'Job Wall-clock time: 1-08:00:33',                        8 -3
     |  'Memory Utilized: 828.22 MB',                             9 -2
     |  'Memory Efficiency: 34.51% of 2.34 GB']                  10 -1
     |
     |  Methods defined here:
     |
     |  __init__(self, slurm_job_id)
     |      Get return from seff command.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  copy(self)
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
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object

    class Seffs(builtins.object)
     |  Seffs(outs=None, seffs=None, pids=None, pids_as_keys=True, units='MB', unit='clock', plot=False, progress_bar=True)
     |
     |  dict-like container with arbitrary keys and values for multiple `Seff` class objects.
     |
     |  Notes
     |  -----
     |  - __isub__ and __iadd__ do not Seffs.check_shfiles for duplicates (but __add__ and __sub__ do)
     |
     |  Methods defined here:
     |
     |  __add__(self, seffs2)
     |
     |  __contains__(self, key)
     |
     |  __delitem__(self, key)
     |
     |  __getitem__(self, key)
     |
     |  __iadd__(self, seffs2)
     |
     |  __init__(self, outs=None, seffs=None, pids=None, pids_as_keys=True, units='MB', unit='clock', plot=False, progress_bar=True)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __isub__(self, seffs2)
     |
     |  __iter__(self)
     |
     |  __len__(self)
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  __setattr__(self, key, value)
     |      Implement setattr(self, name, value).
     |
     |  __setitem__(self, key, item)
     |
     |  __sub__(self, seffs2)
     |
     |  cancelled(self)
     |      Return Seffs object for any cancelled job.
     |
     |  completed(self)
     |      Return Seffs object for any *successfully* completed job (compare to Seffs.finished()).
     |
     |  copy(self)
     |
     |  describe(self, cols=['walltime_hrs', 'core_walltime_hrs', 'memory_used_MB'], **kwargs)
     |      Print out quantile info for walltime hours and memory used.
     |
     |  failed(self)
     |      Return Seffs object for any failed job.
     |
     |  finished(self)
     |      Return non-running and non-pending jobs.
     |
     |  items(self)
     |
     |  keys(self)
     |
     |  len(self)
     |
     |  most_recent(self)
     |      From the shfiles inferred from outs, pair most recent out with sh.
     |
     |  oom(self)
     |      Return Seffs object for any out of memory jobs.
     |
     |  out_sh(self)
     |      key = out, val = sh.
     |
     |  pending(self)
     |      Return pending jobs.
     |
     |  plot_mems(self, **kwargs)
     |      Plot myfigs.histo_box of mem usage for all jobs in self.
     |
     |  plot_times(self, **kwargs)
     |      Plot myfigs.histo_box of times usage for all jobs in self.
     |
     |  running(self)
     |      Return Seffs object for any running job.
     |
     |  sh_out(self, sh_as_key=True)
     |      key = sh, val = most_recent outfile.
     |
     |      TODO
     |      ----
     |      - add `remove` kwarg and pass to pyimp.getmostrecent
     |          - but do I care that any outfiles removed could have elements within the Seff?
     |              - eg a pid or out as a key
     |
     |  sh_outs(self, sh_as_key=True, internal=False)
     |      key = sh, val = list of outfiles.
     |
     |  timeouts(self)
     |      Return Seffs object for any timeout jobs.
     |
     |  to_csv(self, path_or_buf, df_kwargs={}, *args, **kwargs)
     |      Write Seffs.to_dataframe to csv.
     |
     |      Parameters
     |      ----------
     |      df_kwargs : dict
     |          passed to Seffs.to_dataframe (e.g., time_units='hrs', mem_units='MB')
     |          default unless overwritten : kwargs = dict(index=False, header=True, sep='  ')
     |      path_or_buf, args, kwargs
     |          all passed to pd.DataFrame
     |
     |      Returns
     |      -------
     |      None
     |
     |  to_dataframe(self, time_units='hrs', mem_units='MB')
     |      Convert seff info in Seffs to a dataframe, one row for each key (ie slurm_job_id) in self.
     |
     |  uncompleted(self)
     |      Return Seffs object for any uncompleted job.
     |
     |      Notes
     |      -----
     |      - if most recent .out failed but an early .out completed, this code will miss that
     |
     |  values(self)
     |
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |
     |  check_shfiles(shfiles)
     |
     |  filter_states(seffs, state)
     |
     |  parallel(lview, outs=None, pids=None, units='MB', unit='clock', verbose=True)
     |      Execute Seffs in parallel using `lview`.
     |
     |      lview = ipyparallel.client.view.LoadBalancedView
     |
     |  remote(hostname='login.hpc.cam.uchc.edu', outs=None, user=None, python=None)
     |      Connect to remote host and execute seff commands.
     |
     |      Parameters
     |      ----------
     |      hostname : str
     |          host address
     |      outs : list | str
     |          list of .out files
     |      user : str
     |          bash $USER
     |      python : path
     |          path to python executable
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object

    class Squeue(builtins.object)
     |  Squeue(verbose=True, **kwargs)
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
     |  __add__(self, sq2)
     |
     |  __cmp__(self, other)
     |
     |  __contains__(self, pid)
     |
     |  __delitem__(self, key)
     |
     |  __getitem__(self, key)
     |
     |  __iadd__(self, sq2)
     |
     |  __init__(self, verbose=True, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __isub__(self, sq2)
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
     |  __sub__(self, sq2)
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
     |  closing(self)
     |
     |  copy(self)
     |
     |  cpus(self, **kwargs)
     |      Get a list of CPUs.
     |
     |  hold(self, num_jobs=None, **kwargs)
     |      Hold jobs. Parameters described in `Squeue._update_job.__doc__`.
     |
     |  items(self)
     |
     |  jobs(self, **kwargs)
     |      Get a list of job names, subset with kwargs.
     |
     |  keys(self)
     |
     |  mems(self, units='MB', **kwargs)
     |      Get a list of memory requests.
     |
     |  nodelists(self, **kwargs)
     |
     |  nodes(self, **kwargs)
     |
     |  partitions(self)
     |      Get counts of job states across partitions.
     |
     |  pending(self)
     |
     |  pids(self, **kwargs)
     |      Get a list of pids, subset with kwargs.
     |
     |  print(self)
     |      Print out the squeue info similar to how it would appear in command line.
     |
     |  release(self, num_jobs=None, **kwargs)
     |      Release held jobs. Parameters described in `Squeue._update_job.__doc__`.
     |
     |  running(self)
     |
     |  states(self, **kwargs)
     |      Get a list of job states.
     |
     |  statuses(self, **kwargs)
     |
     |  summary(self, **kwargs)
     |      Print counts of states and statuses of the queue.
     |
     |  times(self, unit='clock', **kwargs)
     |
     |  to_dataframe(self)
     |      Convert the squeue info into a pd.DataFrame.
     |
     |  update(self, num_jobs=None, **kwargs)
     |      Update jobs in slurm queue with scontrol, and update job info in Squeue class.
     |
     |      kwargs - that control what can be updated (other kwargs go to Squeue._filter_jobs)
     |      ------
     |      account - the account to transfer jobs
     |      minmemorynode - total memory requested
     |      timelimit - total wall time requested
     |
     |  users(self, **kwargs)
     |
     |  values(self)
     |
     |  watch(self, sleep=5, progress_bar=True)
     |      Refresh __repr__ every `sleep` seconds, clear previous printout.
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object

    sqinfo = class SQInfo(builtins.object)
     |  sqinfo(jobinfo)
     |
     |  Convert each line returned from `squeue -u $USER`.
     |
     |  Assumed
     |  -------
     |  SQUEUE_FORMAT="%i %u %a %j %t %S %L %D %C %b %m %N (%r) %P"
     |                  0  1  2  3  4  5  6  7  8  9 10 11  12  13
     |
     |  Example jobinfo    index
     |  ---------------    -----
     |  ('29068196',         0
     |   'b.lindb',          1
     |   'lotterhos',        2
     |   'batch_0583',       3
     |   'R',                4
     |   'N/A',              5
     |   '9:52:32',          6
     |   '1',                7
     |   '56',               8
     |   'N/A',              9
     |   '2000M',           10
     |   'd0036',           11
     |   '(Priority)')      12
     |   'short'            13
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
     |  cancel(self, verbose=True)
     |      Cancel this job.
     |
     |  mem(self, units='MB')
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object

FUNCTIONS
    clock_hrs(clock: str, unit='hrs') -> float
        From a clock (days-hrs:min:sec) extract hrs or days as float.

    create_watcherfile(pids, directory, watcher_name='watcher', email=None, time='0:00:01', ntasks=1, rem_flags=None, mem=25, end_alert=False, fail_alert=True, begin_alert=False, added_text='', verbose=True)
        From a list of dependency `pids`, sbatch a file that will not start until all `pids` have completed.

            Parameters
            ----------
            pids - list of SLURM job IDs
            directory - where to sbatch the watcher file
            watcher_name - basename for slurm job queue, and .sh and .outfiles
            email - where alerts will be sent
                requires at least of of the following to be True: end_alert, fail_alert, begin_alert
            time - time requested for job
            ntasks - number of tasks
            rem_flags - list of additional SBATCH flags to add (separate with
        )
                eg - rem_flags=['#SBATCH --cpus-per-task=5', '#SBATCH --nodes=1']
            mem - requested memory for job
                default is 25 bytes, but any string will work - eg mem='2500M'
            end_alert - bool
                use if wishing to receive an email when the job ends
            fail_alert - bool
                use if wishing to receive an email if the job fails
            begin_alert - bool
                use if wishing to receive an email when the job begins
            added text - any text to add within the body of the .sh file
            verbose - bool
                whether to print the watcherfile's slurm job ID, `watcher_pid`

            TODO
            ----
            - incorporate code to save mem and time info of `pids`

    get_mems(seffs: dict, units='MB', plot=True, **kwargs) -> list
        From output by `get_seff()`, extract mem in `units` units; histogram if `plot` is True.

        Parameters
        ----------
        - seffs : dict of any key with values of class Seff
        - units : passed to Seff._convert_mem(). options: GB, MB, KB

    get_seff(outs=None, pids=None, desc='executing seff commands', progress_bar=True, pids_as_keys=False)
        From a list of outs or pids, get seff output.

        Parameters
        ----------
        outs : list
            .out files (ending in f'_{SLURM_JOB_ID}.out')
        pids : list
            a list of slurm_job_ids
        desc
            description for progress bar
        progress_bar : bool
            whether to use a progress bar when querying seff
        pids_as_keys : bool
            if outs is not None, retrieve pid from each outfile to use as the key

        Notes
        -----
        - assumes f'{job}_{slurm_job_id}.out' and f'{job}.sh' underly slurm jobs

    get_times(seffs: dict, unit='hrs', plot=True, **kwargs) -> list
        From dict(seffs) [val = seff output], get times in hours.

        fix: add in other clock units

    getpid(out: str) -> str
        From an .out file with structure <anytext_JOBID.out>, return JOBID.

    getsq = _getsq(grepping=None, states=[], user=None, partition=None, aflag=False, p=None, **kwargs)
        Get and parse slurm queue according to criteria. kwargs is not used.

    sbatch(shfiles: Union[str, list], sleep=0, printing=False, outdir=None, progress_bar=True) -> list
        From a list of .sh shfiles, sbatch them and return associated jobid in a list.

        Notes
        -----
        - assumes that the job name that appears in the queue is the basename of the .sh file
            - eg for job_187.sh, the job name is job_187
            - this convention is used to make sure a job isn't submitted twice
        - `sbatch` therefore assumes that each job has a unique job name
        - if the job's job name `filejob` is already in the queue, all jobs in the queue except the oldest
            job are cancelled. the pid returned for the shfile will be the pid of the job that remains in
            the queue.

DATA
    Union = typing.Union
        Union type; Union[X, Y] means either X or Y.

        On Python 3.10 and higher, the | operator
        can also be used to denote unions;
        X | Y means the same thing to the type checker as Union[X, Y].

        To define a union, use e.g. Union[int, str]. Details:
        - The arguments must be types and there must be at least one.
        - None as an argument is a special case and is replaced by
          type(None).
        - Unions of unions are flattened, e.g.::

            assert Union[Union[int, str], float] == Union[int, str, float]

        - Unions of a single argument vanish, e.g.::

            assert Union[int] == int  # The constructor actually returns int

        - Redundant arguments are skipped, e.g.::

            assert Union[int, str, int] == Union[int, str]

        - When comparing unions, the argument order is ignored, e.g.::

            assert Union[int, str] == Union[str, int]

        - You cannot subclass or instantiate a union.
        - You can use Optional[X] as a shorthand for Union[X, None].

    pbar = functools.partial(<class 'tqdm.std.tqdm'>, bar_format='{l_bar}{...
    trange = functools.partial(<function trange at 0x7f2c0a058360>, bar_fo...

```


### Python Library Documentation: module my_r
```

NAME
    my_r - Personalized functions to interact with R.

DESCRIPTION
    TODO
    ----
    - R.plot save/show pdf - right now it will save/show png etc but if pdf only can save but not show

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
     |  help(self, obj)
     |      Get help documentation for an object, `obj`.
     |
     |  library(self, lib)
     |      Load library into R namespace.
     |
     |  plot(self, obj, kind='plot', width=600, height=600, dpi=100, saveloc=None, **kwargs)
     |      Plot in line with jupyter notebook.
     |
     |      Parameters
     |      ----------
     |      obj - rpy2 object, or string - if string then look for unstrung string in R environment
     |      kind - eg plot, barplot, boxplot
     |      width/height/dpi - figure qualities; dpi applies to non-pdf
     |      saveloc - path to save figure - must contain suffix (eg .png, .jpeg, .pdf)
     |
     |      Notes
     |      -----
     |      - this is a bit hacky (tmp files, inner()) so that I can save and show pdfs in the same process
     |          - without this, I can create and display non-pdfs
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
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object

```


### Python Library Documentation: module myclasses
```

NAME
    myclasses - Some of my custom general-purpose classes.

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
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object

DATA
    colorConverter = <matplotlib.colors.ColorConverter object>

```

