"""Personalized functions to build figures."""
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import rgb2hex, colorConverter
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import pandas as pd
import seaborn as sns

def venn_diagram(a, b, c, set_labels=['A', 'B', 'C'], title=''):
    """Create Venn diagram with three groups.
    
    Parameters
    ----------
    - a,b,c - each are lists of loci to use for overlap calculations
    
    Notes
    -----
    - thanks stackoverflow! https://stackoverflow.com/questions/19841535/python-matplotlib-venn-diagram
    """s
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
              y_pad=1.3, histbins='auto', saveloc=None, **kwargs):
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
    plt.xticks(np.arange(0, max(data[col]), xticks_by))
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    
    if saveloc is not None:
        save_pdf(saveloc)
    
    plt.show()
    pass


def slope_graph(x, y, xname, yname, figsize=(3,8), positive_color='black', negative_color='tomato', labeldict=None,
                saveloc=None):
    """Visually display how rank order of .index changes between two pd.Series, `x` and `y`.
    
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
    size = 80
    fig, ax = plt.subplots(figsize=figsize)
    
    xranks = x.rank(ascending=False)
    yranks = y.rank(ascending=False)
    for idx in xranks.index:
        # get the values for each index label from each pd.Series
        xrank = xranks[idx]
        yrank = yranks[idx]

        line_color = neghex if yrank > xrank else poshex

        # Plot the lines connecting the dots
        ax.plot([x1, x2], [xrank, yrank], c=line_color, alpha=alpha, zorder=0)

        # annotate
        plt.annotate(idx, (x1-0.05, xrank), ha='right', c='k' if labeldict is None else labeldict[idx])
        plt.annotate(idx, (x2+0.05, yrank), ha='left', c='k' if labeldict is None else labeldict[idx])

        # plot the points
        ax.scatter([x1-0.01], xrank, c='royalblue',
                   s=size, label=xname, edgecolors='k')
        ax.scatter([x2+0.01], yrank, c='lightblue',
                   s=size, label=yname, edgecolors='k')
    # fix the axes and labels
    ax.set_xticks([0])
    _ = ax.set_xticklabels([None], fontsize='x-large')
    plt.yticks(np.arange(1, nrow(x), 5))
    plt.ylabel('importance rank', fontsize=15)


    # Add legend and fix it to show only the first two elements
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles[0:2], labels[0:2],   
                    fontsize='large',
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.1),
                    ncol=2,
                    scatterpoints=1)
    lgd.legendHandles[0]._sizes = [size]
    lgd.legendHandles[1]._sizes = [size]
    plt.xlim(0.5,1.5)
    ax.invert_yaxis()
    
    if saveloc is not None:
        save_pdf(saveloc)
    
    plt.show()
    return None


def save_pdf(saveloc):
    """After creating a figure in jupyter notebooks, save as PDFs at `saveloc`."""
    with PdfPages(saveloc) as pdf:
        pdf.savefig(bbox_inches="tight")
    print(ColorText("Saved to: ").bold(), saveloc)
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

