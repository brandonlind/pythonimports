import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import pythonimports as pyimp


def draw_pie_marker(ratios, xcoord, ycoord, sizes, colors, ax, edgecolors="black", slice_edgecolors="none", alpha=1,
                    edge_linewidths=1.5, slice_linewidths=1.5, zorder=10, transform=False, label=None, edgefactor=1):
    """Draw a pie chart at coordinates `[xcoord,ycoord]` on `ax`.

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
    """
    from functools import partial

    ratios = [ratio / sum(ratios) for ratio in ratios]

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        x = np.cos(np.linspace(previous, this, 100)).tolist()
        y = np.sin(np.linspace(previous, this, 100)).tolist()
        if len(ratios) > 1 and 1 not in ratios:
            # commenting this section out will convert pie chart to cross-sectional shading
            # adding the [0] to x and y without a condition will result in an interior edge when circle should
                # be completely filled
            x = [0] + x + [0]
            y = [0] + y + [0]

        xy = np.column_stack([x, y])
        previous = this
        markers.append({"marker": xy,
                        "s": np.abs(xy).max() ** 2 * np.array(sizes),
                        "alpha": alpha,
                        "facecolor": color,
                        "edgecolors": slice_edgecolors,
                        "linewidths": slice_linewidths})
    markers.append({"marker": np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 100)).tolist(),
                                               np.sin(np.linspace(0, 2 * np.pi, 100)).tolist()]),
                    "s": np.abs(xy).max() ** 2 * np.array(sizes) * edgefactor,
                    "facecolor": "none",
                    "edgecolors": edgecolors,
                    "linewidths": edge_linewidths,
                    "alpha": alpha})

    # scatter each of the pie pieces to create pies
    scatter = (ax.scatter if transform is False else partial(ax.scatter, transform=ccrs.PlateCarree()))
    annotate = (ax.annotate if transform is False
                else partial(ax.annotate, xycoords=ccrs.PlateCarree()._as_mpl_transform(ax)))
    for marker in markers:
        scatter(xcoord, ycoord, zorder=zorder, **marker)
    if label is not None:
        annotate(label, (xcoord, ycoord), zorder=zorder + 10, color="white", weight="bold")

    pass


def basemap(extent, shapefiles=None, figsize=(8, 15)):
    """Build basemap +/- range shapefile.

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
    """
    from cartopy.io.img_tiles import GoogleTiles

    class _ShadedReliefESRI(GoogleTiles):
        """https://stackoverflow.com/questions/37423997/cartopy-shaded-relief"""

        # shaded relief
        def _image_url(self, tile):
            x, y, z = tile
            url = (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
            ).format(z=z, y=y, x=x)
            return url

        pass

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=_ShadedReliefESRI().crs)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(_ShadedReliefESRI(), 7)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.2)

    land_50m = cfeature.NaturalEarthFeature("physical", "land", "50m", edgecolor="face")
    ax.add_feature(land_50m, edgecolor="black", facecolor="gray", alpha=0.4)
    states_provinces = cfeature.NaturalEarthFeature(category="cultural",
                                                    name="admin_1_states_provinces_lines",
                                                    scale="50m",
                                                    facecolor="none")
    ax.add_feature(states_provinces, edgecolor="black")

    if shapefiles is not None:
        for color, shape in shapefiles:
            ax.add_geometries(Reader(shape).geometries(),
                              ccrs.PlateCarree(),
                              facecolor=color,
                              alpha=0.1,
                              edgecolor="none",
                              zorder=2)
            ax.add_geometries(Reader(shape).geometries(),
                              ccrs.PlateCarree(),
                              facecolor="none",
                              edgecolor=color,
                              alpha=0.8,
                              zorder=3)
    ax.coastlines(resolution="10m", zorder=4)
    ax.add_feature(cfeature.BORDERS)

    bathym = cfeature.NaturalEarthFeature(name="bathymetry_J_1000", scale="10m", category="physical")
    ax.add_feature(bathym, edgecolor="none", facecolor="gray", alpha=0.1)

    return ax


def plot_pie_freqs(locus, snpinfo, envinfo, saveloc=None, use_popnames=False, popcolors=None, bmargs=[], **kwargs):
    """Create geographic map, overlay pie graphs (ref/alt allele freqs)."""
    freqcols = [col for col in snpinfo.columns if "FREQ" in col]
    snpdata = (snpinfo.loc[locus, freqcols].str.replace("%", "").astype(float))  # eg change 97.5% to 97.5
    print(len(snpdata))

    ax = basemap(**bmargs)

    # plot the pops
    for pop in envinfo.index:
        color = "black" if popcolors is None else popcolors[pop]
        long, lat = envinfo.loc[pop, ["LONG", "LAT"]]
        #         try:
        try:
            af = round(snpdata[f"{pop}.FREQ"])  # ALT freq
        except ValueError as e:
            print("passing ", pop)
            continue
        rf = 100 - af  # REF freq
        draw_pie_marker([af, rf],
                        long,
                        lat,
                        150,
                        ax=ax,
                        colors=["blue", "orange"],
                        transform=True,
                        label=None if use_popnames is False else pop,
                        edgecolors=color,
                        **kwargs)
    # save
    if saveloc is not None:
        with PdfPages(saveloc) as pdf:
            pdf.savefig(bbox_inches="tight")
        print(pyimp.ColorText("Saved to: ").bold(), saveloc)
    plt.show()

    pass
