"""Functions for mapping / GIS."""
from scalebar import add_scalebar

from os import path as op
import os
import math
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from functools import partial
from cartopy.io.img_tiles import Stamen
from cartopy.io.shapereader import Reader
from cartopy.io.img_tiles import GoogleTiles
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.ticker as mticker
from shapely.geometry import Polygon, box
import geopandas as gpd
from matplotlib.colors import colorConverter
from PIL import Image, ImageEnhance, ImageChops
from pyproj import Geod

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


def _nice_round_km(target_km: float) -> float:
    """Choose a 'nice' round number near target_km from {1,2,5} * 10^n."""
    if target_km <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(target_km))
    for m in [1, 2, 5]:
        val = m * mag
        if val >= target_km:
            return val
    return 10 * mag


def add_north_arrow(ax,
                    location_xy=(0.92, 0.08),   # axes fraction (x,y)
                    length_km=None,             # if None, ~1/8 of map height
                    color="k",
                    lw=2.5,
                    mutation_scale=18,          # arrowhead size
                    text="N",
                    font_size=12,
                    box_alpha=1,
                    zorder=1000
                   ):
    """
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
    """
    geod = Geod(ellps="WGS84")

    # Get current extent in geodetic coordinates
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    # Start point (lon0, lat0) from axes-fraction placement
    lon0 = lon_min + location_xy[0] * lon_span
    lat0 = lat_min + location_xy[1] * lat_span

    # Choose a default arrow length (~1/32 of map height along a meridian at lon0)
    if length_km is None:
        _, _, map_h_m = geod.inv(lon0, lat_min, lon0, lat_max)
        length_km = _nice_round_km(0.03125 * map_h_m / 1000.0)

    # Compute endpoint by moving north (bearing 0°) along the ellipsoid
    lon1, lat1, _ = geod.fwd(lon0, lat0, 0.0, length_km * 1000.0)

    # Draw arrow in geodetic coords (transform from PlateCarree)
    trans_pc = ccrs.PlateCarree()._as_mpl_transform(ax)  # transform lon/lat -> display
    arrow = FancyArrowPatch((lon0, lat0), (lon1, lat1),
                            transform=trans_pc,
                            arrowstyle='-|>',
                            color=color,
                            lw=lw,
                            zorder=zorder,
                            mutation_scale=mutation_scale)
    ax.add_patch(arrow)

    # Put the 'N' label slightly beyond the arrow tip
    lon_t, lat_t, _ = geod.fwd(lon1, lat1, 0.0, 0.12 * length_km * 1000.0)
    ax.text(lon_t, lat_t, text,
            transform=ccrs.PlateCarree(),
            ha="center", va="bottom",
            fontsize=font_size, color=color, zorder=zorder,
            bbox=dict(facecolor="white", edgecolor="none", alpha=box_alpha, pad=0.4))
    pass


class ESRIShadedReliefTint(GoogleTiles):
    """
    ESRI World Shaded Relief (pre-colored) with a customizable color tint.
    
    Parameters
    ----------
    color : (R, G, B)
        The target tint color, e.g., (40, 100, 170) for a cool blue.
    strength : float in [0, 1]
        How strongly to push toward the tint color (0=no tint, 1=full tint).
    desaturate : float
        Factor to desaturate the original tile before tinting (1.0 = unchanged).
    """
    def __init__(self, color=(70, 110, 160), strength=0.4, desaturate=0.8, **kwargs):
        super().__init__(**kwargs)
        self.color = tuple(color)
        self.strength = float(strength)
        self.desaturate = float(desaturate)

    def _image_url(self, tile):
        x, y, z = tile
        return (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
        ).format(z=z, y=y, x=x)

    def get_image(self, tile):
        # Fetch the base tile using the parent implementation
        img, extent, origin = super().get_image(tile)
        img = img.convert("RGB")

        # Optional: reduce original saturation so the tint dominates
        if self.desaturate != 1.0:
            img = ImageEnhance.Color(img).enhance(self.desaturate)

        # Create a tint layer and multiply it over the base
        tint = Image.new("RGB", img.size, self.color)
        white = Image.new("RGB", img.size, (255, 255, 255))
        # Mix white → tint so strength controls tint intensity
        tint_mix = Image.blend(white, tint, self.strength)
        out = ImageChops.multiply(img, tint_mix)

        return out, extent, origin

    pass


def _normalize_lon(lon: float) -> float:
    """Wrap longitude to [-180, 180)."""
    return ((lon + 180) % 360) - 180


def _center_of_longitudes(lon_min: float, lon_max: float) -> float:
    """
    Compute the longitudinal center robustly even if the interval crosses the antimeridian.
    Returns center_lon in [-180, 180).
    """
    a = _normalize_lon(lon_min)
    b = _normalize_lon(lon_max)
    # If interval is 'reversed', it likely crosses the dateline—shift one side by 360 and average.
    if (b - a) % 360 > 180:
        if a < 0:
            a += 360
        else:
            b += 360
    center = (a + b) / 2.0
    return _normalize_lon(center)


def _nice_round_km(target_km: float) -> float:
    """
    Choose a 'nice' round number near target_km from {1,2,5} * 10^n.
    """
    if target_km <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(target_km))
    for m in [1, 2, 5]:
        val = m * mag
        if val >= target_km:
            return val
    return 10 * mag


def _pick_projection_from_extent(extent, centralize=True, force_projection=None) -> ccrs.Projection:
    """
    Decide a projection based on the extent. Defaults to PlateCarree but recenters if requested.
    For very high-latitude extents, choose a Lambert Azimuthal Equal Area centered on the bbox.
    """
    if force_projection is not None:
        return force_projection

    lon_min, lon_max, lat_min, lat_max = extent
    lat_c = 0.5 * (lat_min + lat_max)
    lon_c = _center_of_longitudes(lon_min, lon_max) if centralize else 0.0

    lat_span = abs(lat_max - lat_min)

    # Heuristic: if we're mostly polar or very high-latitude, use an azimuthal equal-area centered.
    if (abs(lat_c) > 60) or (lat_span > 60):
        return ccrs.LambertAzimuthalEqualArea(central_longitude=lon_c, central_latitude=lat_c)

    # Otherwise, a re-centered PlateCarree keeps things simple and avoids antimeridian issues.
    return ccrs.PlateCarree(central_longitude=lon_c)


def _auto_stamen_zoom(extent) -> int:
    """
    Approximate a Stamen/OSM-like zoom from longitudinal span.
    """
    lon_min, lon_max, _, _ = extent
    # Compute the smaller absolute span accounting for wrap
    span = abs((_normalize_lon(lon_max) - _normalize_lon(lon_min)))
    span = min(span, 360 - span)
    span = max(span, 1e-6)  # avoid log issues
    zoom = int(np.clip(np.round(np.log2(360.0 / span)), 2, 12))
    return zoom


def add_scalebar(ax: plt.Axes,
                 length_km=None,
                 location_xy=(0.1, 0.05),
                 ndiv: int = 2,
                 height_frac: float = 0.01,
                 linewidth: float = 2.5,
                 color: str = "k",
                 font_size: int = 9,
                 box_alpha: float = 1,
                 units: str = "km",
                 arrow_location=(0.15, 0.08),
                 zorder=1000
                ) -> None:
    """
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
    """
    geod = Geod(ellps="WGS84")
    # Get current displayed extent in geodetic coords:
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    # Compute the bar's reference position in geodetic coordinates:
    bar_lat = lat_min + location_xy[1] * lat_span
    bar_lon_start = lon_min + location_xy[0] * lon_span

    # Determine available width at this latitude (rhumb distance along the parallel)
    try:
        _, _, map_width_m = geod.rhumb_inv(lon_min, bar_lat, lon_max, bar_lat)
        rhumb_supported = True
    except Exception:
        # Fallback to geodesic distance (close for small segments)
        _, _, map_width_m = geod.inv(lon_min, bar_lat, lon_max, bar_lat)
        rhumb_supported = False

    if length_km is None:
        target_m = 0.25 * map_width_m  # Aim ~1/4 of map width
        length_km = _nice_round_km(target_m / 1000.0)

    length_m = float(length_km) * 1000.0

    # Compute the end lon of the bar along the parallel:
    if rhumb_supported:
        bar_lon_end, bar_lat_end, _ = geod.rhumb_fwd(bar_lon_start, bar_lat, 90.0, length_m)
    else:
        # Geodesic fallback (bearing 90 deg). Latitude will not be exactly constant, but close.
        bar_lon_end, bar_lat_end, _ = geod.fwd(bar_lon_start, bar_lat, 90.0, length_m)

    # Visual bar height in degrees (visual thickness only, not used for distance)
    dh = height_frac * lat_span

    # Draw the main bar (as a rectangle-like symbol made of top/bottom plus fill)
    # We'll draw the central line and short ticks; Matplotlib/Cartopy will project it appropriately.
    ax.plot([bar_lon_start, bar_lon_end], [bar_lat, bar_lat],
            transform=ccrs.PlateCarree(), color=color, lw=linewidth, zorder=zorder)

    # Subdivision ticks and labels
    for i in range(ndiv + 1):
        frac = i / ndiv if ndiv > 0 else 1.0
        dist_m_i = frac * length_m
        if rhumb_supported:
            lon_i, lat_i, _ = geod.rhumb_fwd(bar_lon_start, bar_lat, 90.0, dist_m_i)
        else:
            lon_i, lat_i, _ = geod.fwd(bar_lon_start, bar_lat, 90.0, dist_m_i)

        # Tick
        ax.plot([lon_i, lon_i], [lat_i - 0.8 * dh, lat_i + 0.8 * dh],
                transform=ccrs.PlateCarree(), color=color, lw=linewidth, zorder=zorder)

        # Tick labels: 0 at start, middle value in the middle, full length at end
        label_km = (length_km * frac)
        label_txt = f"{int(label_km) if label_km.is_integer() else round(label_km, 1)}"
        va = "bottom"
        # Slight offset above the bar
        ax.text(lon_i, lat_i + 1.1 * dh, label_txt,
                transform=ccrs.PlateCarree(), ha="center", va=va,
                fontsize=font_size, color=color, zorder=zorder,
                bbox=dict(facecolor="white", edgecolor="none", alpha=box_alpha, pad=1.0))

    # Units centered under the bar
    mid_lon = (bar_lon_start + bar_lon_end) / 2.0
    mid_lat = (bar_lat + bar_lat_end) / 2.0
    ax.text(mid_lon, mid_lat - 1.8 * dh, units,
            transform=ccrs.PlateCarree(), ha="center", va="top",
            fontsize=font_size, color=color, zorder=zorder,
            bbox=dict(facecolor="white", edgecolor="none", alpha=box_alpha, pad=0.6))

    add_north_arrow(ax, location_xy=arrow_location)
    
    pass


def basemap(
    extent,
    projection=None,
    centralize_projection_on_extent=True,
    shaded_relief: str = "stock",  # "stock" | "stamen-terrain" | "stamen-toner-lite" | "none"
    stamen_zoom=None,
    figsize=(10, 8),
    add_rivers=True,
    gridline_kwargs=None,
    gridline_width=0.5,
    scalebar_kwargs={},
    scalebar_zorder=1000,
    shapefiles=None,
    cut_extent=None,
    scalebar=True,
    x_interval=None,
    y_interval=None,
    ticklabels=True,
    **kwargs
):
    """
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
    """
    extent = list(extent)
    if len(extent) != 4:
        raise ValueError("extent must be [min_lon, max_lon, min_lat, max_lat]")

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    
    # Choose projection if not provided
    if projection is None:
        proj = _pick_projection_from_extent(extent,
                                            centralize=centralize_projection_on_extent,
                                            force_projection=projection)
        ax = plt.axes(projection=proj)
    else:
        proj = projection
        ax = plt.axes(projection=_ShadedReliefESRI().crs)

    # Set the displayed extent using geodetic PlateCarree coordinates
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # ax.add_image(_ShadedReliefESRI(), 7)
    ax.add_image(ESRIShadedReliefTint(color=(255, 255, 255), strength=0.35, desaturate=0.7), 7)
    
    # Optional map adornments
    ax.coastlines(resolution="10m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=99)
    if add_rivers:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="physical",
                name="rivers_lake_centerlines",
                scale="10m",
                facecolor="none",
                edgecolor="lightsteelblue"
            ),
            zorder=98,
            edgecolor='lightsteelblue'
        )

    ax.add_feature(cfeature.LAKES, facecolor="lightsteelblue", zorder=98)

    bathym = cfeature.NaturalEarthFeature(name="bathymetry_J_1000", scale="10m", category="physical")
    ax.add_feature(bathym, edgecolor="none", facecolor="gray", alpha=0.1)
    
    states_provinces = cfeature.NaturalEarthFeature(category="cultural",
                                                    name="admin_1_states_provinces_lines",
                                                    scale="50m",
                                                    facecolor="none")
    ax.add_feature(states_provinces, edgecolor="black", zorder=99)

    add_shapefiles_to_map(ax, shapefiles, cut_extent=cut_extent, zorder=100, **kwargs)

    # add gridlines
    xmin, xmax, ymin, ymax = ax.get_extent(crs=ccrs.PlateCarree())
    gl_kwargs = dict(draw_labels=True, linewidth=gridline_width, color="gray", alpha=0.5, linestyle="--", zorder=100)
    gl = ax.gridlines(**gl_kwargs)
    ax._gridliner = gl

    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    gl.top_labels = False
    gl.right_labels = False

    if ticklabels is False:
        gl.bottom_labels = False
        gl.left_labels = False

    if gridline_kwargs:
        for key, value in gridline_kwargs.items():
            setattr(gl, key, value)

    # Set locators dynamically
    if x_interval is not None:
        gl.xlocator = mticker.FixedLocator(np.arange(np.floor(xmin), np.ceil(xmax) + x_interval, x_interval))
    if y_interval is not None:
        gl.ylocator = mticker.FixedLocator(np.arange(np.floor(ymin), np.ceil(ymax) + y_interval, y_interval))

    if scalebar is True:
        add_scalebar(ax, zorder=scalebar_zorder, **scalebar_kwargs)

    return fig, ax


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


# def basemap(extent, shapefiles=None, coastlines=0.6, add_bathym=True, cut_extent=None, figsize=(8, 15), **kwargs):
#     """Build basemap +/- range shapefile.

#     Parameters
#     ----------
#     extent : list
#         - the geographic extent to be displayed eg [lon_min, lon_max, lat_min, lat_max]
#     shapefiles 
#         - a list of tuples, each tuple is (color, shapefile_path.shp)
#     coastlines : int
#         - linewidth of coastlines
#     add_bathym : bool
#         - whether or not to add bathymetry data (eg if adding shapefiles on top of oceans)
#     cut_extent : list
#         - different order than `extent`, the extent to cut shapefiles (default is an internally re-ordered `extent`)
#     kwargs : dict
#         - passed to add_shapefiles_to_map and cut_shapes

#     # douglas-fir shortcuts
#     coastrange = '/data/projects/pool_seq/environemental_data/shapefiles/Coastal_DF.shp'
#     intrange = '/data/projects/pool_seq/environemental_data/shapefiles/Interior_DF.shp'
#         # df_shapfiles = zip(['lime', 'purple'], [coastrange, intrange])
#         # extent=[-130, -112.5, 37.5, 55.5]
#         # zoom out [-130, -108.5, 32.5, 55.5]

#     # jack pine shortcuts
#     extent=[-119.5, -58, 41, 60], figsize=(15,10),
#     shapefiles=[('green', '/data/projects/pool_seq/environemental_data/shapefiles/jackpine.shp')]
#     """
#     # if cut_extent is None:
#     #     cut_extent = [extent[0], extent[2], extent[1], extent[3]]
    
#     fig = plt.figure(figsize=figsize)
#     ax = plt.axes(projection=_ShadedReliefESRI().crs)
#     ax.set_extent(extent, crs=ccrs.PlateCarree())
#     ax.add_image(_ShadedReliefESRI(), 7)
#     ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.2, zorder=10)

#     land_50m = cfeature.NaturalEarthFeature("physical", "land", "50m", edgecolor="face")
#     ax.add_feature(land_50m, edgecolor="black", facecolor="gray", alpha=0.4)

#     ax.add_feature(cfeature.RIVERS, edgecolor="lightsteelblue")
#     ax.add_feature(cfeature.LAKES, facecolor="lightsteelblue")
    
#     states_provinces = cfeature.NaturalEarthFeature(category="cultural",
#                                                     name="admin_1_states_provinces_lines",
#                                                     scale="50m",
#                                                     facecolor="none")
#     ax.add_feature(states_provinces, edgecolor="black")

#     add_shapefiles_to_map(ax, shapefiles, cut_extent=cut_extent, **kwargs)
            
#     ax.coastlines(resolution="10m", zorder=4, linewidth=coastlines)
#     ax.add_feature(cfeature.BORDERS)

#     if add_bathym is True:
#         bathym = cfeature.NaturalEarthFeature(name="bathymetry_J_1000", scale="10m", category="physical")
#         ax.add_feature(bathym, edgecolor="none", facecolor="gray", alpha=0.1)

#     return ax

def draw_rectangle(ax, extent, facecolor='gray', alpha=0.5, linewidth=3, zorder=1500):
    lonmin, lonmax, latmin, latmax = extent
    xy = (lonmin, latmin)
    width = abs(lonmax - lonmin)
    height = abs(latmax - latmin)
    ax.add_patch(
        Rectangle(xy, width, height, transform=ccrs.PlateCarree(), facecolor=facecolor, alpha=alpha, linewidth=linewidth, edgecolor='none',
                  zorder=1500)
    )
    ax.add_patch(
        Rectangle(xy, width, height, transform=ccrs.PlateCarree(), fill=False, edgecolor=(0, 0, 0, 1), linewidth=linewidth, zorder=zorder)
    )
    pass


def inset_map(extent, map_extent=None, shapes=[], shapefiles=None, figsize=(8, 15), projection=None, **kwargs):
    """Create an black and white inset map for placing within a larger map.
    
    Parameters
    ----------
    extent - list
        extent of inset map - [minlong, maxlong, minlat, maxlat]
    map_extent - list
        extent of larger map for drawing box within inset map
    """
    fig = plt.figure(figsize=figsize)
    
    if projection is None:
        projection = ccrs.PlateCarree()

    # create a map
    # ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plt.axes(projection=projection)
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
        draw_rectangle(ax, map_extent)
        # lonmin, lonmax, latmin, latmax = map_extent
        # xy = (lonmin, latmin)
        # width = abs(lonmax - lonmin)
        # height = abs(latmax - latmin)
        # ax.add_patch(Rectangle(xy, width, height, facecolor='gray', alpha=0.5, linewidth=3))
        # ax.add_patch(Rectangle(xy, width, height, fill=False, edgecolor='k', linewidth=3))
    
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
