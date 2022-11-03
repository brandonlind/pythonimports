"""Personalized functions to build figures."""
from matplotlib.patches import PathPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import rgb2hex, colorConverter, LinearSegmentedColormap
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import pandas as pd
import seaborn as sns
import pythonimports as pyimp
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pythonimports as pyimp
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_cmap(list_of_colors, name=None, grain=500):
    """Create a custom color map with fine-grain transition."""
    return LinearSegmentedColormap.from_list(name, list_of_colors, N=grain)


def histo_box(data, xticks_by=None, title=None, xlab=None, ylab=None, col=None, fontsize=12,
              y_pad=1.3, histbins='auto', saveloc=None, rotation=0, ax=None, histplot_kws={},
              boxplot_kws={}, **kwargs):
    """Create histogram with boxplot in top margin.
    
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
    boxplot_kws - dict
        kwargs pass to seaborn.boxplot
    histplot_kws - dict
        kwargs passed to seaborn.histplot
    kwargs
        passed to plt.subplots
    
    Notes
    ------
        thanks https://www.python-graph-gallery.com/24-histogram-with-a-boxplot-on-top-seaborn
    """
#     col = 'data' if col is None else col
    if 'name' in dir(data):
        col = data.name
    elif 'columns' in dir(data):
        col = data.columns[0]
    elif col is None:
        col = 'data'

    if isinstance(data, pd.DataFrame) is False:
        data = pd.DataFrame(data, columns=[col])
    
    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    if ax is None:
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, **kwargs)
    else:
        ax_hist = ax
        divider = make_axes_locatable(ax_hist)
        ax_box = divider.append_axes("top", size="15%", pad=0.1, sharex=ax_hist)
        
    # assigning a graph to each ax
    sns.boxplot(x=data[col], ax=ax_box, **boxplot_kws)
    sns.histplot(data=data, x=col, ax=ax_hist, bins=histbins, **histplot_kws)

    # Remove xlabel and xticks from the boxplot
    xticklabels = ax_hist.get_xticklabels()
    ax_box.tick_params(labelbottom=False)
    ax_box.set_xlabel(None)
    
    if title is not None:
        ax_hist.set_title(title, y=y_pad, fontdict=dict(fontsize=fontsize))
        
    if xticks_by is not None:
        ax_hist.set_xticks(np.arange(0, max(data[col]), xticks_by), rotation=rotation)

    ax_hist.set_xlabel(xlab, fontsize=fontsize)
        
    ax_hist.set_ylabel(ylab, fontsize=fontsize)
    
    if saveloc is not None:
        save_pdf(saveloc)
    
    if ax is None:
        plt.show()
        
    return ax_box, ax_hist


import matplotlib.lines as mlines
import matplotlib.colors as mcolors
# differs from commit because I've moved `marker_size` from being hard-coded (as `size`) to it being a kwarg
def slope_graph(x, *y, labels=['x', 'y'], figsize=(3,8), positive_color='black', negative_color='tomato',
                labeldict=None, shape_color=None, saveloc=None, title=None, legloc='center',
                colors=list(mcolors.TABLEAU_COLORS), markers=None, addtolegend=None, ylabel='importance rank',
                ascending=False, legendcols=None, bbox_to_anchor=(0.5, -0.05), ax=None, marker_size=80):
    """Visually display how rank order of .index changes between arbitrary number of pd.Series, `x` and *`y`.
    
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
    
    Notes
    -----
    - thanks https://cduvallet.github.io/posts/2018/03/slopegraphs-in-python
    """    
    # line colors
    poshex = rgb2hex(colorConverter.to_rgb(positive_color))
    neghex = rgb2hex(colorConverter.to_rgb(negative_color))
    alpha = 0.8
    x1 = 0.85
    x2 = 1.15
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    xranks = x.rank(ascending=ascending)
    xcolor = colors[0]
    xname = labels[0]
    xmarker = 'o' if markers is None else markers[0]
    
    for i,other in enumerate(y):
        yranks = other.rank(ascending=ascending)
        ycolor = colors[i+1]
        yname = labels[i+1]
        ymarker = 'o' if markers is None else markers[i+1]
        for idx in xranks.index:
            # get the values for each index label from each pd.Series
            xrank = xranks[idx]
            yrank = yranks[idx]

            if ascending is False:
                # if large numbers get low ranks (largest is ranked #1)
                line_color = neghex if yrank > xrank else poshex
            else:
                # if large numbers get high ranks (smallest is ranked #1)
                line_color = neghex if yrank < xrank else poshex

            # Plot the lines connecting the dots
            ax.plot([x1, x2], [xrank, yrank], c=line_color, alpha=alpha, zorder=1)

            # annotate index labels (next to circles)
            if i == 0:
                plt.annotate(idx, (x1-0.1, xrank+0.13), ha='right', c='k' if labeldict is None else labeldict[idx])
            if (i+1)==len(y):
                plt.annotate(idx, (x2+0.13, yrank+0.13), ha='left', c='k' if labeldict is None else labeldict[idx])

            # plot the points
            ax.scatter([x1], xrank, c=xcolor if shape_color is None else shape_color[idx], s=marker_size, label=xname,
                       edgecolors='k', marker=xmarker, zorder=2)
            ax.scatter([x2], yrank, c=ycolor if shape_color is None else shape_color[idx], s=marker_size, label=yname, 
                       edgecolors='k', marker=ymarker, zorder=2)
        xranks = yranks.copy()
        xcolor = ycolor
        xname = yname
        xmarker = ymarker
        x1 += 0.3
        x2 += 0.3
    # fix the axes and labels
    ax.set_xticks([0])
    _ = ax.set_xticklabels([None], fontsize='x-large')
    plt.yticks(np.arange(1, pyimp.nrow(x), 5))
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title)

    # Add legend and fix it to show only the first two elements
    handles, labels = ax.get_legend_handles_labels()
    keep_labels = []
    keep_handles = []
    for i,label in enumerate(labels):
        if label not in keep_labels:
            keep_labels.append(label)
            keep_handles.append(handles[i])

    if shape_color is not None:  # if color depends on rank level, just use shape with white fill
        if markers is None:
            # if markers is None then shapes for x and *y are all circles ('o')
            raise Exception('combination of shape_color and marker kwargs not allowed')
        keep_handles = []
        for i,marker in enumerate(markers):
            keep_handles.append(
                mlines.Line2D([],[], color='None', marker=marker, linestyle='None',
                              label=labels[i], markeredgecolor='k')
            )

    if addtolegend is not None:
        #mpatches.Patch(color='grey', label='manual patch')   
        keep_handles = [mpatches.Patch(color=handle.get_facecolor()[0]) for handle in keep_handles]
        keep_handles.extend(addtolegend[0])
        keep_labels.extend(addtolegend[1])
    lgd = ax.legend(keep_handles,
                    keep_labels,
                    fontsize='large',
                    loc=legloc,
                    borderaxespad=0.5,
                    bbox_to_anchor=bbox_to_anchor,
                    ncol=len(y)+1 if legendcols is None else legendcols,
                    scatterpoints=1)
    lgd.legendHandles[0]._sizes = [marker_size]
    lgd.legendHandles[1]._sizes = [marker_size]
    low,hi = ax.get_xlim()
    plt.xlim(0, hi + 0.85)
    ax.invert_yaxis()
    
    if saveloc is not None:
        save_pdf(saveloc)
    
    if ax is None:
        plt.show()
        return None

    return ax


def save_pdf(saveloc):
    """After creating a figure in jupyter notebooks, save as PDFs at `saveloc`."""
    with PdfPages(saveloc) as pdf:
        pdf.savefig(bbox_inches="tight")
    print(pyimp.ColorText("Saved to: ").bold(), saveloc)
    pass


def makesweetgraph(x=None, y=None, cmap="jet", ylab=None, xlab=None, bins=100, saveloc=None, figsize=(5, 4), snsbins=60,
                   title=None, xlim=None, ylim=None, vlim=(None, None)) -> None:
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
    # plot data
    ax1 = sns.jointplot(x=x, y=y, marginal_kws=dict(bins=snsbins))
    ax1.fig.set_size_inches(figsize[0], figsize[1])
    ax1.ax_joint.cla()
    plt.sca(ax1.ax_joint)
    plt.hist2d(x, y, bins, norm=mcolors.LogNorm(*vlim), cmap=cmap, range=None if xlim is None else np.array([xlim, ylim]))
    # set title and axes labels
    if title is None:
        plt.title("%s\nvs\n%s\n" % (xlab, ylab), y=1.2, x=0.6)
    else:
        plt.title(title, y=1.2, x=0.6)
    plt.ylabel(ylab, fontsize=12)
    plt.xlabel(xlab, fontsize=12)
    # set up scale bar legend
    cbar_ax = ax1.fig.add_axes([1, 0.1, 0.03, 0.7])
    cb = plt.colorbar(cax=cbar_ax)
    cb.set_label(r"$\log_{10}$ density of points", fontsize=13)
    # save if prompted
    if saveloc is not None:
        save_pdf(saveloc)
    plt.show()
    pass


def gradient_image(ax, direction=0.3, cmap_range=(0, 1), extent=(0, 1, 0, 1), **kwargs):
    """
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
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, interpolation='bicubic', extent=extent,
                   vmin=0, vmax=1, **kwargs)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('auto')
    
    return im


def adjust_box_widths(axes, fac=0.9):
    """
    Adjust the widths of a seaborn-generated boxplot.
    
    Notes
    -----
    - thanks https://github.com/mwaskom/seaborn/issues/1076
    """
    if isinstance(axes, list) is False:
        axes = [axes]
    
    # iterating through Axes instances
    for ax in axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    try:
                        if np.all(l.get_xdata() == [xmin, xmax]):
                            l.set_xdata([xmin_new, xmax_new])
                    except:
                        # /tmp/ipykernel_138835/916607433.py:32: DeprecationWarning: elementwise comparison failed;
                            # this will raise an error in the future.
                                # if np.all(l.get_xdata() == [xmin, xmax]):
                        pass
    pass
