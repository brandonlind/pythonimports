"""Functions for mapping / GIS."""
from scalebar import add_scalebar

from os import path as op
import os
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from functools import partial
from cartopy.io.img_tiles import Stamen
from cartopy.io.shapereader import Reader
from cartopy.io.img_tiles import GoogleTiles
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon, box
import geopandas as gpd
from matplotlib.colors import colorConverter

import pythonimports as pyimp


def gdalwarp(infile, netcdf_outfile, proj, gdalwarp_exe=None):
    """Convert `infile` (eg .tif or .nc) to WGS84 `netcdf_outfile` (.nc).
    
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
            proj = '+proj=laea +lat_1=49.0 +lat_2=77.0 +lat_0=45 \
            +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 \
            +units=m +no_defs'
    gdalwarp_exe
        path to gdalwarp executable - eg one created with conda
            ~/anaconda3/envs/gdal_env/bin/gdalwarp
    """
    import subprocess, shutil
    
    if gdalwarp_exe is None:
        gdalwarp_exe = op.join(os.environ['HOME'], 'anaconda3/envs/gdal_env/bin/gdalwarp')
    
    # gdalwarp -s_srs <proj> -t_srs <proj> -of netCDF /path/to/infile /path/to/outfile -overwrite
    
    output = subprocess.check_output([gdalwarp_exe,
                                      '-s_srs', proj,
                                      '-t_srs',  '+proj=longlat +ellps=WGS84',
                                      '-of', 'netCDF',
                                      infile,
                                      netcdf_outfile,
                                      '-overwrite']).decode('utf-8').split('\n')
    return output


def draw_pie_marker(ratios, xcoord, ycoord, sizes, colors, ax, edgecolors="black", slice_edgecolors="none", alpha=1,
                    edge_linewidths=1.5, slice_linewidths=1.5, zorder=10, transform=False, label=None, edgefactor=1,
                    label_kws={}):
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
    label_kws : dict, passed to ax.annotate

    TODO
    ----
    - offset label

    Notes
    -----
    - thanks stackoverflow! https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    """
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
    
    objects = []
    for marker in markers:
        objects.append(
            scatter(xcoord, ycoord, zorder=zorder, **marker)
        )
    if label is not None:
        objects.append(
            annotate(label, (xcoord, ycoord), zorder=zorder + 10, **label_kws)
        )

    return pyimp.flatten(objects)


def overlaps(polygon, polygons):
    """Determine if `polygon` overlaps with any of the `polygons`."""
    for poly in polygons:
        if polygon.equals(poly):
            continue
        if polygon.intersects(poly):
            return True

    return False


def cut_shapes(shapefile, cut=True, epsg=4326, cut_extent=None, query=None):
    """Cut out overlapping polygons from non-overlapping polygons.
    
    Notes
    -----
    `cut_extent` is useful when original shapefile is large
    """
    geodf = gpd.read_file(shapefile)
    # geodf = geodf.loc[geodf.AREA.sort_values(ascending=False).index]  # sort polygons from largest to smallest area

    if cut_extent is not None:
        geodf = geodf.clip(box(*cut_extent))

    if query is not None:
        geodf.query(query, inplace=True)
    
    if geodf.crs is None:
        geodf.set_crs(epsg=epsg, inplace=True)
    elif geodf.crs.to_epsg() != epsg:
        geodf.to_crs(epsg=epsg, inplace=True)

    if cut is False:
        return [geom for geom in geodf.geometry]

    # Iterate through each geometry
    non_overlapping_geoms = []
    overlapping_geoms = []
    for geom in geodf.geometry:
        if not overlaps(geom, non_overlapping_geoms):
            non_overlapping_geoms.append(geom)
        else:
            overlapping_geoms.append(geom)

    # cut out overlapping from non overlapping
    polygons = []
    for geom in pyimp.pbar(non_overlapping_geoms, desc=f'cutting polygons from {op.basename(shapefile)}'):
        polygons.append(
            Polygon(geom, holes=overlapping_geoms)# if isinstance(overlapping_geoms, list) else None)
        )

    return polygons


def add_shapefiles_to_map(ax, shapefiles=None, face_alpha=0.1, edge_alpha=1, zorder=20, progress_bar=False, geo_kws={}, **kwargs):
    """Add shapefiles to `ax`.
    
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
    """
    if shapefiles is not None:
        if progress_bar is False:
            iterator = shapefiles
        else:
            iterator = pyimp.pbar(shapefiles, desc='adding shapes')

        for color, shape in iterator:
            ax.add_geometries(cut_shapes(shape, **kwargs),
                              ccrs.PlateCarree(),
                              facecolor=(*colorConverter.to_rgb(color), face_alpha),
                              edgecolor=(*colorConverter.to_rgb(color), edge_alpha),
                              zorder=zorder,
                              **geo_kws
                             )
    pass


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


def basemap(extent, shapefiles=None, coastlines=0.6, add_bathym=True, cut_extent=None, figsize=(8, 15), **kwargs):
    """Build basemap +/- range shapefile.

    Parameters
    ----------
    extent : list
        - the geographic extent to be displayed eg [lon_min, lon_max, lat_min, lat_max]
    shapefiles 
        - a list of tuples, each tuple is (color, shapefile_path.shp)
    coastlines : int
        - linewidth of coastlines
    add_bathym : bool
        - whether or not to add bathymetry data (eg if adding shapefiles on top of oceans)
    cut_extent : list
        - different order than `extent`, the extent to cut shapefiles (default is an internally re-ordered `extent`)
    kwargs : dict
        - passed to add_shapefiles_to_map and cut_shapes

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
    # if cut_extent is None:
    #     cut_extent = [extent[0], extent[2], extent[1], extent[3]]
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=_ShadedReliefESRI().crs)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(_ShadedReliefESRI(), 7)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.2, zorder=10)

    land_50m = cfeature.NaturalEarthFeature("physical", "land", "50m", edgecolor="face")
    ax.add_feature(land_50m, edgecolor="black", facecolor="gray", alpha=0.4)

    ax.add_feature(cfeature.RIVERS, edgecolor="lightsteelblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightsteelblue")
    
    states_provinces = cfeature.NaturalEarthFeature(category="cultural",
                                                    name="admin_1_states_provinces_lines",
                                                    scale="50m",
                                                    facecolor="none")
    ax.add_feature(states_provinces, edgecolor="black")

    add_shapefiles_to_map(ax, shapefiles, cut_extent=cut_extent, **kwargs)
            
    ax.coastlines(resolution="10m", zorder=4, linewidth=coastlines)
    ax.add_feature(cfeature.BORDERS)

    if add_bathym is True:
        bathym = cfeature.NaturalEarthFeature(name="bathymetry_J_1000", scale="10m", category="physical")
        ax.add_feature(bathym, edgecolor="none", facecolor="gray", alpha=0.1)

    return ax


def inset_map(extent, map_extent=None, shapes=[], shapefiles=None, figsize=(8, 15), **kwargs):
    """Create an black and white inset map for placing within a larger map.
    
    Parameters
    ----------
    extent - list
        extent of inset map - [minlong, maxlong, minlat, maxlat]
    map_extent - list
        extent of larger map for drawing box within inset map
    """
    fig = plt.figure(figsize=figsize)

    # create a map
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.coastlines(resolution="10m", zorder=4, linewidth=0.6)
    
    land_50m = cfeature.NaturalEarthFeature("physical", "land", "50m", edgecolor="face")
    ax.add_feature(land_50m, edgecolor="black", facecolor="gray", alpha=0.4)

    states_provinces = cfeature.NaturalEarthFeature(category="cultural",
                                                    name="admin_1_states_provinces_lines",
                                                    scale="50m",
                                                    facecolor="none")
    ax.add_feature(states_provinces, edgecolor="black")

    ax.add_feature(cfeature.BORDERS)
    
    add_shapefiles_to_map(ax, shapefiles, **kwargs)

    # add box for domain of larger map
    if map_extent is not None:
        lonmin, lonmax, latmin, latmax = map_extent
        xy = (lonmin, latmin)
        width = abs(lonmax - lonmin)
        height = abs(latmax - latmin)
        ax.add_patch(Rectangle(xy, width, height, facecolor='gray', alpha=0.5, linewidth=3))
        ax.add_patch(Rectangle(xy, width, height, fill=False, edgecolor='k', linewidth=3))
    
    # add any other shapes to the map
    for shape in shapes:
        ax.add_patch(shape)

    return ax


def plot_pie_freqs(locus, snpinfo, envinfo, saveloc=None, use_popnames=False, popcolors=None, bmargs={}, **kwargs):
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
        except ValueError:
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


def read_geofile(geofile, epsg="epsg:4326", x_dim="latitude", y_dim="longitude", debug=False):
    """Read in netcdf file."""
    import xarray as xr
    import rioxarray

    ds = xr.open_dataset(geofile)
    if debug:
        print(f'{ds.variables = }')
    ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
    ds.rio.write_crs(epsg, inplace=True)
    
    layers = [var for var in list(ds.variables) if var not in ['crs', x_dim, y_dim]]
    assert len(layers) == 1, layers
    layer = layers[0]
    vals = ds[layer][:,:]
    lons = ds[layer][x_dim]
    lats = ds[layer][y_dim]
    
    return ds, layer, vals, lons, lats
