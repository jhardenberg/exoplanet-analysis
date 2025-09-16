import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from util import dv2uv, sp2gp


def plotmap(da, fig=None, subplot=(1, 1, 1), title=None, extent=None, 
            coast=True, proj="lonlat", cmap="viridis", cline=None,
            clabel="off", grid=[30,30], national=False, cdir="vertical", 
            figsize=(12, 8), add_colorbar=True, terminator=False, ice=None, **kwargs):
    """
    Plot a map using cartopy and xarray.
    
    Parameters
    ----------
    da : xarray.DataArray
        Data to plot.
    fig : matplotlib.figure.Figure, optional
        Figure to plot the map. If None, a new figure is created.
    subplot : tuple, optional
        Subplot to plot the map.
    title : str, optional
        Title of the plot.
    extent : list, optional
        Extent of the map [lon_min, lon_max, lat_min, lat_max].
    coast : bool, optional
        Add coastlines to the plot.
    proj : str, optional    
        Projection of the map. Options are 'lonlat', 'orthographic', 'mercator', 
        'gnomic', 'lambert', 'transverse', 'azimuthal', 'albers', 'stereographic',
        'robinson', 'mollweide', 'polar_north', 'polar_south'.
    cmap : str, optional
        Colormap of the plot.
    cline : list, optional
        Contour lines to plot.
    clabel : str, optional
        Label of the colorbar. By default the label is the variable name.
    grid : list, optional
        Grid lines to plot [lon_spacing, lat_spacing].
    terminator: bool, optional
        Add terminator to the plot.
    national : bool, optional
        Add national borders to the plot.
    cdir : str, optional
        Direction of the colorbar. Options are 'vertical' or 'horizontal'.
    figsize : tuple, optional
        Size of the figure.
    add_colorbar : bool, optional
        Add colorbar to the plot.
    ice : xarray.DataArray, optional
        Ice data to plot. If None, no ice data is plotted.
    **kwargs : optional 
        Additional arguments to pass to the xarray plot function.

    Returns     
    ------- 
    im : matplotlib.collections.QuadMesh
        Image of the plot.

    """

    if proj == "lonlat":
        projection = ccrs.PlateCarree()
    elif proj == "orthographic":
        projection = ccrs.Orthographic()
    elif proj == "mercator":
        projection = ccrs.Mercator()
    elif proj == "gnomic":
        projection = ccrs.Gnomonic()
    elif proj == "lambert":
        projection = ccrs.LambertConformal()
    elif proj == "transverse":
        projection = ccrs.TransverseMercator()
    elif proj == "azimuthal":
        projection = ccrs.AzimuthalEquidistant()
    elif proj == "albers":
        projection = ccrs.AlbersEqualArea()
    elif proj == "stereographic":
        projection = ccrs.Stereographic()
    elif proj == "robinson":
        projection = ccrs.Robinson()
    elif proj == "mollweide":
        projection = ccrs.Mollweide()
    elif proj == "polar_north":
        projection = ccrs.NorthPolarStereo()
        if not extent:
            extent = [-180, 180, 45, 90]
    elif proj == "polar_south":
        projection = ccrs.SouthPolarStereo()
        if not extent:
            extent = [-180, 180, -90, -45]
    else:
        projection = ccrs.PlateCarree() # default

    if not fig:
        fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(subplot[0], subplot[1], subplot[2], projection=projection)

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    if 'nsp' in da.dims:
        da = sp2gp(da)
    
    if clabel == "off":
        units = da.attrs.get('units', None)
        longname = da.attrs.get('long_name', None)
        if longname:
            name = longname
        else:
            name = da.name
        if units:
            clabel = f'{name} [{units}]'
        else:
            clabel = name
            
    # Plot the data
    if add_colorbar:
        cbar_kwargs = {'shrink': 0.6, 'label': clabel, 'orientation': cdir}
    else:
        cbar_kwargs = None
    
    im = da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, 
            cbar_kwargs=cbar_kwargs, add_colorbar=add_colorbar,
            **kwargs)

    if ice is not None:
        # ice = ice.where(ice >= 0.5)
        mask = (ice >= 0.5).compute()

        lon2d, lat2d = np.meshgrid(ice.lon, ice.lat)
        lon_masked = lon2d[mask]
        lat_masked = lat2d[mask]

        ax.plot(lon_masked, lat_masked, 'k.', markersize=0.5, transform=ccrs.PlateCarree())

    if cline is not None:
        levels = np.array(cline)
        contour = ax.contour(da.lon, da.lat, da, levels=levels, colors='k', linewidths=0.5, transform=ccrs.PlateCarree())
        #ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

    if terminator:
        db = da.copy()
        db.values = np.zeros_like(db.values)
        db.values[:, (da.lon >= 270) | (da.lon <= 90)] = 2
        levels = np.array([1])
        _ = ax.contour(db.lon, db.lat, db, levels=levels, colors='w', linewidths=3, transform=ccrs.PlateCarree())

    if coast:
        ax.coastlines()
    if title:
        plt.title(title)
    if national:
        ax.add_feature(cfeature.BORDERS, linewidth=1, linestyle=':')

    if grid:
        gl = ax.gridlines(draw_labels=True, dms=True, 
                          x_inline=False, y_inline=False, 
                          linestyle='--', linewidth=0.5, color='gray')
        gl.top_labels = False
        gl.right_labels = False

        gl.xlocator = mticker.FixedLocator(range(-180, 181, grid[0]))
        gl.ylocator = mticker.FixedLocator(range(-90, 91, grid[1]))

        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}

    plt.tight_layout()
    return im



def plotmap_quiver(duv, fig=None, subplot=(1, 1, 1), title=None, extent=None, 
            coast=True, proj="lonlat", cmap="viridis", cline=None,
            clabel="off", grid=[30,30], national=False, cdir="vertical", 
            figsize=(12, 8), add_colorbar=True, scale=1000, **kwargs):
    """
    Plot a map using cartopy and xarray.
    
    Parameters
    ----------
    duv : xarray.Dataset
        Data to plot (contains u and v).
    fig : matplotlib.figure.Figure, optional
        Figure to plot the map. If None, a new figure is created.
    subplot : tuple, optional
        Subplot to plot the map.
    title : str, optional
        Title of the plot.
    extent : list, optional
        Extent of the map [lon_min, lon_max, lat_min, lat_max].
    coast : bool, optional
        Add coastlines to the plot.
    proj : str, optional    
        Projection of the map. Options are 'lonlat', 'orthographic', 'mercator', 
        'gnomic', 'lambert', 'transverse', 'azimuthal', 'albers', 'stereographic',
        'robinson', 'mollweide', 'polar_north', 'polar_south'.
    cmap : str, optional
        Colormap of the plot.
    cline : list, optional
        Contour lines to plot.
    clabel : str, optional
        Label of the colorbar. By default the label is the variable name.
    grid : list, optional
        Grid lines to plot [lon_spacing, lat_spacing].
    national : bool, optional
        Add national borders to the plot.
    cdir : str, optional
        Direction of the colorbar. Options are 'vertical' or 'horizontal'.
    figsize : tuple, optional
        Size of the figure.
    add_colorbar : bool, optional
        Add colorbar to the plot.
    scale : float, optional
        Scale of the quiver plot.
    **kwargs : optional 
        Additional arguments to pass to the xarray plot function.

    Returns     
    ------- 
    im : matplotlib.collections.QuadMesh
        Image of the plot.

    """

    if proj == "lonlat":
        projection = ccrs.PlateCarree()
    elif proj == "orthographic":
        projection = ccrs.Orthographic()
    elif proj == "mercator":
        projection = ccrs.Mercator()
    elif proj == "gnomic":
        projection = ccrs.Gnomonic()
    elif proj == "lambert":
        projection = ccrs.LambertConformal()
    elif proj == "transverse":
        projection = ccrs.TransverseMercator()
    elif proj == "azimuthal":
        projection = ccrs.AzimuthalEquidistant()
    elif proj == "albers":
        projection = ccrs.AlbersEqualArea()
    elif proj == "stereographic":
        projection = ccrs.Stereographic()
    elif proj == "robinson":
        projection = ccrs.Robinson()
    elif proj == "mollweide":
        projection = ccrs.Mollweide()
    elif proj == "polar_north":
        projection = ccrs.NorthPolarStereo()
        if not extent:
            extent = [-180, 180, 45, 90]
    elif proj == "polar_south":
        projection = ccrs.SouthPolarStereo()
        if not extent:
            extent = [-180, 180, -90, -45]
    else:
        projection = ccrs.PlateCarree() # default

    if not fig:
        fig = plt.figure(figsize=figsize)

    da = np.sqrt(duv.u**2 + duv.v**2)

    ax = fig.add_subplot(subplot[0], subplot[1], subplot[2], projection=projection)

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
    if clabel == "off":
        units = da.attrs.get('units', None)
        longname = da.attrs.get('long_name', None)
        if longname:
            name = longname
        else:
            name = da.name
        if units:
            clabel = f'{name} [{units}]'
        else:
            clabel = name
            
    # Plot the data
    if add_colorbar:
        cbar_kwargs = {'shrink': 0.6, 'label': clabel, 'orientation': cdir}
    else:
        cbar_kwargs = None
    
    im = da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, 
            cbar_kwargs=cbar_kwargs, add_colorbar=add_colorbar,
            **kwargs)
    
    # Skip every second arrow
    skip = (slice(None, None, 2), slice(None, None, 2))
    duvs = duv.isel(lat=skip[0], lon=skip[1])
    duvs.plot.quiver(ax=ax, x='lon', y='lat', u='u', v='v', 
                     transform=ccrs.PlateCarree(), scale=scale, add_guide=False)
    #ax.quiver(du.lon, du.lat, du, dv, scale=scale, transform=ccrs.PlateCarree())
    #ax.quiver(-7.1, 35.1, 10.0, 0., scale=scale, transform=ccrs.PlateCarree(), color='green')

    # qv_key = ax.quiverkey(q, 0.7, 0.7, 5, r'a5 m/s',labelpos='E', labelsep =0.05, 
    #                       color='black', coordinates='figure')

    if cline is not None:
        levels = np.array(cline)
        contour = ax.contour(da.lon, da.lat, da, levels=levels, colors='k', linewidths=0.5, transform=ccrs.PlateCarree())
        #ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

    if coast:
        ax.coastlines()
    if title:
        plt.title(title)
    if national:
        ax.add_feature(cfeature.BORDERS, linewidth=1, linestyle=':')

    if grid:
        gl = ax.gridlines(draw_labels=True, dms=True, 
                          x_inline=False, y_inline=False, 
                          linestyle='--', linewidth=0.5, color='gray')
        gl.top_labels = False
        gl.right_labels = False

        gl.xlocator = mticker.FixedLocator(range(-180, 181, grid[0]))
        gl.ylocator = mticker.FixedLocator(range(-90, 91, grid[1]))

        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}

    plt.tight_layout()
    return im

def make_stamps(exp_file, exp_name, var='tas', label=None, cline=None, level=None,
                cmin=160, cmax=380, outfile=None, cmap='jet', factor=1, 
                subplot=(5, 5), figsize=(12, 8), terminator=False, ice=False):
    """
    Make a set of stamps for a given variable and experiments.
    
    Parameters
    ----------
    
    exp_file : list
        List of filenames with the data to plot for each experiment.
    
    exp_name : list
        List of names of the experiments.
        
    var : str or function, optional
        Variable to plot. If a function is passed, it should take a xarray dataset
        as input and return a xarray dataarray. Default is 'tas'.

    label : str, optional
        Label of the colorbar. By default the label is the variable name.

    cline : list, optional  
        Contour lines to plot.

    cmin : float, optional
        Minimum value of the colorbar.

    cmax : float, optional
        Maximum value of the colorbar.

    outfile : str, optional
        Name of the output file.

    cmap : str, optional
        Colormap of the plot.

    terminator: bool, optional
        Add terminator to the plot.

    factor : float, optional
        Factor to multiply the variable. Default is 1.    

    ice : bool, optional
        Add ice data to the plot. If None, no ice data is plotted.

    level : int, optional
        Level to plot. If None, the variable is plotted as is.
    """

    fig = plt.figure(figsize=figsize)

    for i, exp in enumerate(exp_name):
        print(f"Processing {exp} ({i+1}/{len(exp_name)})")
        data = xr.open_mfdataset(exp_file[i])
        data = data.isel(time=slice(-12, None)).mean(dim='time')

        if isinstance(var, str):  # var is a variable name
            da = data[var]*factor
        else:  # assuming var is a function
            da = var(data)*factor

        if level is not None:
            da = da.isel(sfc=level)

        if ice:
            iceda = data['sic']
        else:
            iceda = None
             
        im = plotmap(da, fig=fig, subplot=(subplot[0], subplot[1], i+1), proj="robinson", 
                coast=False, vmin=cmin, vmax=cmax, clabel=var, cmap=cmap, terminator=terminator,
                title=exp, cdir='vertical', add_colorbar=False, cline=cline, ice=iceda)

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    fig.colorbar(im, cax=cbar_ax, label=label, orientation='vertical')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    if not outfile:
        outfile = var

    fig.savefig(outfile+'.pdf', format='pdf', dpi=300)
    fig.savefig(outfile+'.png', format='png', dpi=300)

def make_stamps_wind(exp_file, exp_name, label=None, cline=None, scale=1000,
                cmin=160, cmax=380, outfile=None, cmap='jet', factor=1, level=2,
                subplot=(5, 5), figsize=(12, 8)):
    """
    Make a set of stamps for a given variable and experiments.
    
    Parameters
    ----------
    
    exp_file : list
        List of filenames with the data to plot for each experiment.
    
    exp_name : list
        List of names of the experiments.

    label : str, optional
        Label of the colorbar. By default the label is the variable name.

    cline : list, optional  
        Contour lines to plot.

    cmin : float, optional
        Minimum value of the colorbar.

    cmax : float, optional
        Maximum value of the colorbar.

    outfile : str, optional
        Name of the output file.

    cmap : str, optional
        Colormap of the plot.

    factor : float, optional
        Factor to multiply the variable. Default is 1.    
    """

    fig = plt.figure(figsize=figsize)

    for i, exp in enumerate(exp_name):
        data = xr.open_mfdataset(exp_file[i])
        data = data.isel(time=slice(-12, None))
        data = dv2uv(data)
        data = data.mean(dim='time').isel(sfc=level)
             
        im = plotmap_quiver(data, fig=fig, subplot=(subplot[0], subplot[1], i+1), proj="robinson", 
                coast=False, vmin=cmin, vmax=cmax, clabel='wind speed', cmap=cmap, 
                title=exp, cdir='vertical', add_colorbar=False, cline=cline, scale=scale)

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    fig.colorbar(im, cax=cbar_ax, label=label, orientation='vertical')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    if not outfile:
        outfile = var

    fig.savefig(outfile+'.pdf', format='pdf', dpi=300)
    fig.savefig(outfile+'.png', format='png', dpi=300)


def plotvert(da, fig=None, subplot=(1, 1, 1), title=None, cmap="viridis",
             levels=20, clabel="off", figsize=(12, 8), add_colorbar=True,
             roll=True, invert_yaxis=True, cline=None, **kwargs):
    """
    Plot a vertical section using xarray.

    Parameters
    ----------
    da : xarray.DataArray
        Data to plot. Must be 2D.
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None, a new figure is created.
    subplot : tuple, optional
        Subplot to plot on.
    title : str, optional
        Title of the plot.
    cmap : str, optional
        Colormap of the plot.
    levels : int or list, optional
        Number of contour levels or a list of contour levels.
    clabel : str, optional
        Label of the colorbar. By default the label is the variable name.
    figsize : tuple, optional
        Size of the figure.
    add_colorbar : bool, optional
        Add colorbar to the plot.
    invert_yaxis : bool, optional
        Invert the y-axis. Default is True.
    roll: bool, optional
        If True, roll the longitude to center the plot around the dateline.
    cline : list, optional
        Contour lines to plot.
    **kwargs : optional
        Additional arguments to pass to the xarray plot function.

    Returns
    -------
    im : matplotlib.collections.QuadMesh
        Image of the plot.
    """

    if not fig:
        fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(subplot[0], subplot[1], subplot[2])

    if clabel == "off":
        units = da.attrs.get('units', None)
        longname = da.attrs.get('long_name', None)
        if longname:
            name = longname
        else:
            name = da.name
        if units:
            clabel = f'{name} [{units}]'
        else:
            clabel = name

    # Plot the data
    if add_colorbar:
        cbar_kwargs = {'label': clabel}
    else:
        cbar_kwargs = None

    if 'sfc' in da.dims:
        da = da.assign_coords(sfc=da.sfc / 10)
        da.sfc.attrs['standard_name'] = r'$\sigma$'

    if roll and 'lon' in da.dims:
        # Shift the longitude
        n_lon = len(da.lon)
        da = da.roll(lon=n_lon // 2, roll_coords=True)
        attrs = da.lon.attrs
        da.coords['lon'] = (da.coords['lon'] + 180) % 360 - 180
        da.lon.attrs = attrs

    im = da.plot.contourf(ax=ax, cmap=cmap, levels=levels,
                           add_colorbar=add_colorbar, cbar_kwargs=cbar_kwargs,
                           **kwargs)
    
    if cline is not None:
        levels = np.array(cline)
        da.plot.contour(ax=ax, levels=levels, linewidths=0.5)
    
    if invert_yaxis:
        ax.invert_yaxis()

    if title:
        plt.title(title)

    plt.tight_layout()
    return im

def make_stamps_vert(exp_file, exp_name, var='ta', mean_dim='lon', lon=None, lat=None, label=None,
                     levels=40, cmin=None, cmax=None, outfile=None, cmap='jet',
                     factor=1, subplot=(5, 5), figsize=(12, 8), cline=None):
    """
    Make a set of vertical section stamps for a given variable and experiments.

    Parameters
    ----------
    exp_file : list
        List of filenames with the data to plot for each experiment.
    exp_name : list
        List of names of the experiments.
    var : str or function, optional
        Variable to plot. If a function is passed, it should take an xarray dataset
        as input and return an xarray dataarray. Default is 'ta'.
    mean_dim : str, optional
        Dimension to average over to create the 2D vertical section.
        Default is 'lon' for a zonal mean.
    label : str, optional
        Label of the colorbar. By default, the label is the variable name.
    levels : int or list, optional
        Number of contour levels or a list of contour levels.
    cmin : float, optional
        Minimum value of the colorbar.
    cmax : float, optional
        Maximum value of the colorbar.
    outfile : str, optional
        Name of the output file.
    cmap : str, optional
        Colormap of the plot.
    factor : float, optional
        Factor to multiply the variable. Default is 1.
    subplot : tuple, optional
        Grid dimensions for the subplots.
    figsize : tuple, optional
        Size of the figure.
    cline : list, optional
        Contour lines to plot.
    """

    fig = plt.figure(figsize=figsize)

    if lon or lat:
        mean_dim = None  # Override mean_dim if lon or lat is specified

    for i, exp in enumerate(exp_name):
        print(f"Processing {exp} ({i+1}/{len(exp_name)})")
        data = xr.open_mfdataset(exp_file[i])
        data = data.isel(time=slice(-12, None)).mean(dim='time')

        if isinstance(var, str):  # var is a variable name
            da = data[var]
        else:  # assuming var is a function
            da = var(data)

        # Convert from spectral to grid if necessary
        if 'nsp' in da.dims:
            da = sp2gp(da)

        # Create 2D data by averaging over the specified dimension
        if mean_dim:
            if mean_dim in da.dims:
                if mean_dim == 'lat':
                    weights = np.cos(np.deg2rad(da.lat))
                    da =  da.weighted(weights).mean(dim="lat")
                else:
                    da = da.mean(dim=mean_dim)
            else:
                raise ValueError(f"Dimension '{mean_dim}' not found in data array.")
        else:
            if lon is not None:
                da = da.sel(lon=lon, method='nearest').squeeze()
            if lat is not None:
                da = da.sel(lat=lat, method='nearest').squeeze()

        da = da * factor

        im = plotvert(da, fig=fig, subplot=(subplot[0], subplot[1], i + 1),
                      cmap=cmap, levels=levels, vmin=cmin, vmax=cmax,
                      title=exp, add_colorbar=False, cline=cline)

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    if label is None:
        if isinstance(var, str):
            label = var
        else:
            label = da.name
    fig.colorbar(im, cax=cbar_ax, label=label, orientation='vertical')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    if not outfile:
        if isinstance(var, str):
            outfile = f"{var}_vert"
        else:
            outfile = "vertical_section"

    fig.savefig(outfile + '.pdf', format='pdf', dpi=300)
    fig.savefig(outfile + '.png', format='png', dpi=300)