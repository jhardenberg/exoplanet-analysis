import os
import xarray as xr
import numpy as np

def dv2uv(ds):
    """Convert divergence and vorticity to u and v using CDO"""
    
    ds[['d', 'zeta']].to_netcdf('temp_spectral.nc')
    if os.path.exists('temp_grid.nc'):
        os.remove('temp_grid.nc')
    os.system("cdo -s dv2uv temp_spectral.nc temp_grid.nc")
    uv = xr.open_dataset('temp_grid.nc')
    return uv

def sp2gp(da):
    """Convert datarray from spectral to grid point using CDO"""
    
    var = da.name
    da.attrs['CDI_grid_type'] = 'spectral'  # Make sure it is spectral data adding CDI_grid_type :spectral
    da.to_netcdf('temp_spectral.nc')
    if os.path.exists('temp_grid.nc'):
        os.remove('temp_grid.nc')
    os.system("cdo -s sp2gp temp_spectral.nc temp_grid.nc")
    gp = xr.open_dataset('temp_grid.nc')
    gp = gp[var].squeeze()

    if 'CDI_grid_type' in gp.attrs:
        del gp.attrs['CDI_grid_type']
    
    if 'CDI_grid_num_LPE' in gp.attrs: 
        del gp.attrs['CDI_grid_num_LPE']

    # if sfc is in the coordinates and not in the dimension drop it
    if 'sfc' in gp.coords and 'sfc' not in gp.dims:
        gp = gp.drop_vars('sfc')
    
    return gp

def fldmean(data):
    """Simple function to provide a weighted mean."""
    weights = np.cos(np.deg2rad(data.lat))
    if 'lon' in data.coords:
        data_mean = data.weighted(weights).mean(dim=['lat', 'lon'])
    else:
        data_mean = data.weighted(weights).mean(dim=['lat'])
    return data_mean
