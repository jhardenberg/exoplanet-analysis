import xarray as xr
from util import fldmean, dv2uv
import pandas as pd
import numpy as np
import yaml

def compute_stats(exp_name, exp_file, flux=None):

    stats = {}
    for i, exp in enumerate(exp_name):
            data = xr.open_mfdataset(exp_file[i])
            data = data.isel(time=slice(-12, None)).mean(dim='time')
            print(i, exp,':')

            stats[exp] = {}

            tas = float(fldmean(data.tas).values)
            stats[exp]['tas'] = tas
            print(f"   tas: {tas}")

            pr =  float((fldmean(data.prl) + fldmean(data.prc) + fldmean(data.prsn)).values)
            stats[exp]['pr'] = pr
            print(f"   pr: {pr}")

            albedo = float((-fldmean(data.rsut) / (fldmean(data.rst) - fldmean(data.rsut))).values)
            stats[exp]['albedo'] = albedo
            print(f"   albedo: {albedo}")

            rsut = float((-fldmean(data.rsut)).values)
            stats[exp]['rsut'] = rsut
            print(f"   rsut: {rsut}")

            rst = float(fldmean(data.rst).values)
            stats[exp]['rst'] = rst
            print(f"   rst: {rst}")

            olr = float((-fldmean(data.rlut)).values)
            stats[exp]['olr'] = olr
            print(f"   olr: {olr}")

            clt = float(fldmean(data.clt).values)
            stats[exp]['clt'] = clt
            print(f"   clt: {clt}")

            sic = float(fldmean(data.sic).values)
            stats[exp]['sic'] = sic
            print(f"   sic: {sic}")

            # habitability 
            h100 = float(fldmean(data.tas.where( data.tas >= 273.15).where(data.tas < 373.15) > 0).values)
            stats[exp]['h100'] = h100
            print(f"   h100: {h100}")

            h50 = float(fldmean(data.tas.where( data.tas >= 273.15).where(data.tas < 323.15) > 0).values)
            stats[exp]['h50'] = h50
            print(f"   h50: {h50}")

            deltatas = float((data.tas.sel(lat=0, lon=0, method='nearest') - data.tas.sel(lat=0, lon=180, method='nearest')).values)
            stats[exp]['deltatas'] = deltatas
            print(f"   deltatas: {deltatas}")

            prw = float(fldmean(data.prw).values)
            stats[exp]['prw'] = prw
            print(f"   prw: {prw}")

            if flux:
                stats[exp]['flux'] = flux[i]
                print(f"   flux: {flux[i]}")
    return stats

def save_stats(stats, basename='stats', fmt='csv'):
        """Saves statistics to files.

        Always saves a yaml file. Also saves a csv or xlsx file.

        Args:
            stats (dict): The statistics dictionary.
            basename (str, optional): The base name for the output files. Defaults to 'stats'.
            fmt (str, optional): The format for the table file ('csv' or 'xlsx'). Defaults to 'csv'.
        """
        with open(f'{basename}.yaml', 'w') as file:
            yaml.dump(stats, file)

        stats_df = pd.DataFrame.from_dict(stats, orient='index')

        if fmt == 'csv':
            stats_df.to_csv(f'{basename}.csv')
        elif fmt == 'xlsx':
            stats_df.to_excel(f'{basename}.xlsx')
        else:
            raise ValueError(f"Unsupported format: {fmt}. Please choose 'csv' or 'xlsx'.")
        
        print(f"Statistics saved to {basename}.yaml and {basename}.{fmt}")
        return stats_df



def compute_planetary_stats(exp_name, exp_file, flux=None):

    stats = {}
    for i, exp in enumerate(exp_name):
            data = xr.open_mfdataset(exp_file[i]).isel(time=slice(-12, None))
            datau = dv2uv(data)
            datau = datau.mean(dim='time')
            data = data.mean(dim='time')

            tas = float(fldmean(data.tas).values)
            stats[exp]['tas'] = tas
            print(f"   tas: {tas}")

            uscale =  np.sqrt(float(fldmean(data.u**2 + data.v**2)))
            stats[exp]['uscale'] = uscale
            print(f"  uscale: {uscale}")

    return stats