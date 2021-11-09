"""Personalized functions to build figures."""
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import rgb2hex, colorConverter
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import pandas as pd
import seaborn as sns
import pythonimports as pyimp
import numpy as np
from matplotlib.colors import LogNorm

def venn_diagram(a, b, c, set_labels=['A', 'B', 'C'], title=''):
    """Create Venn diagram with three groups.
    
    Parameters
    ----------
    - a,b,c - each are lists of loci to use for overlap calculations
    
    Notes
    -----
    - thanks stackoverflow! https://stackoverflow.com/questions/19841535/python-matplotlib-venn-diagram
    """
    a = list(set(a))
    b = list(set(b))
    c = list(set(c))
    
    only_a = len(set(a) - set(b+c))
    only_b = len(set(b) - set(a+c))
    only_c = len(set(c) - set(a+b))

    a_b = len(set(a).intersection(b))
    a_c = len(set(a).intersection(c))
    b_c = len(set(b).intersection(c))
    
    a_b_c = len((set(a).intersection(b)).intersection(c))
    
    venn_out = venn3(subsets=(only_a, only_b, a_b, only_c, a_c, b_c, a_b_c), set_labels=set_labels)
    for text in venn_out.set_labels:
        text.set_fontsize(17)
    for text in venn_out.subset_labels:
        text.set_fontsize(10)
    plt.title(title, fontdict=dict(fontsize=16))
    plt.show()
    
    pass


def histo_box(data, xticks_by=10, title=None, xlab=None, ylab='count', col=None, fontsize=20,
              y_pad=1.3, histbins='auto', saveloc=None, rotation=0, **kwargs):
    """Create histogram with boxplot in top margin.
    
    https://www.python-graph-gallery.com/24-histogram-with-a-boxplot-on-top-seaborn"""
    col = 'data' if col is None else col
    if isinstance(data, pd.DataFrame) is False:
        data = pd.DataFrame(data, columns=[col])

    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, **kwargs)

    # assigning a graph to each ax
    sns.boxplot(x=data[col], ax=ax_box)
    sns.histplot(data=data, x=col, ax=ax_hist, bins=histbins)

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    plt.title(title, y=y_pad, fontdict=dict(fontsize=fontsize))
    plt.xticks(np.arange(0, max(data[col]), xticks_by), rotation=rotation)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    
    if saveloc is not None:
        save_pdf(saveloc)
    
    plt.show()
    pass


def slope_graph(x, *y, labels=['x', 'y'], figsize=(3,8), positive_color='black', negative_color='tomato',
                labeldict=None, saveloc=None, title=None, legloc='center',
                colors=list(mcolors.TABLEAU_COLORS), markers=None, addtolegend=None, ylabel='importance rank',
                ascending=False):
    """Visually display how rank order of .index changes between arbitrary number of pd.Series, `x` and *`y`.
    
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
    """
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    
    # line colors
    poshex = rgb2hex(colorConverter.to_rgb(positive_color))
    neghex = rgb2hex(colorConverter.to_rgb(negative_color))
    alpha = 0.8
    x1 = 0.85
    x2 = 1.15
    size = 80
    fig, ax = plt.subplots(figsize=figsize)
    
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

            line_color = neghex if yrank > xrank else poshex

            # Plot the lines connecting the dots
            ax.plot([x1, x2], [xrank, yrank], c=line_color, alpha=alpha, zorder=0)

            # annotate index labels (next to circles)
            if i == 0:
                plt.annotate(idx, (x1-0.1, xrank+0.13), ha='right', c='k' if labeldict is None else labeldict[idx])
            if (i+1)==len(y):
                plt.annotate(idx, (x2+0.13, yrank+0.13), ha='left', c='k' if labeldict is None else labeldict[idx])

            # plot the points
            ax.scatter([x1], xrank, c=xcolor, s=size, label=xname, edgecolors='k', marker=xmarker)
            ax.scatter([x2], yrank, c=ycolor, s=size, label=yname, edgecolors='k', marker=ymarker)
        xranks = yranks.copy()
        xcolor = ycolor
        xname = yname
        xmarker = ymarker
        x1 += 0.3
        x2 += 0.3
    # fix the axes and labels
    ax.set_xticks([0])
    _ = ax.set_xticklabels([None], fontsize='x-large')
    plt.yticks(np.arange(1, nrow(x), 5))
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
    if addtolegend is not None:
        #mpatches.Patch(color='grey', label='manual patch')   
        keep_handles = [mpatches.Patch(color=handle.get_facecolor()[0]) for handle in keep_handles]
        keep_handles.extend(addtolegend[0])
        keep_labels.extend(addtolegend[1])
    lgd = ax.legend(keep_handles,
                    keep_labels,
                    fontsize='large',
                    loc=legloc,
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=len(y)+1,
                    scatterpoints=1)
    lgd.legendHandles[0]._sizes = [size]
    lgd.legendHandles[1]._sizes = [size]
    low,hi = ax.get_xlim()
    plt.xlim(0, hi + 0.85)
    ax.invert_yaxis()
    
    if saveloc is not None:
        save_pdf(saveloc)
    
    plt.show()

    return None


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
    plt.hist2d(x, y, bins, norm=LogNorm(*vlim), cmap=cmap, range=None if xlim is None else np.array([xlim, ylim]))
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
