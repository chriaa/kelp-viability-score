import xarray as xr
import pandas as pd
import requests
from bs4 import BeautifulSoup
import sys

# Constants
BASE_HTTPS_URL = "https://www.ncei.noaa.gov/data/oceans/ghrsst/L4/GLOB/JPL/MUR/"
DODS_URL = "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ghrsst/L4/GLOB/JPL/MUR/"
CA_BOUNDS = {'lat': slice(32, 42), 'lon': slice(-126, -116)}

def get_links(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, 'html.parser')
        return [a.get('href') for a in soup.find_all('a') if a.get('href')]
    except:
        return []

def run_seasonal_pull(start_year=2010, end_year=2024):
    # --- Fail Fast Checks ---
    if end_year < start_year:
        print(f"ERROR: End year ({end_year}) is before start year ({start_year}).")
        sys.exit(1)

    print(f"--> Scanning NCEI Server from {start_year} to {end_year}...")
    
    all_hrefs = get_links(BASE_HTTPS_URL)
    # Filter years based on user input
    years = [y.strip('/') for y in all_hrefs if y.strip('/').isdigit()]
    years = [y for y in years if start_year <= int(y) <= end_year]
    
    if not years:
        print(f"CRITICAL ERROR: No data found for range {start_year}-{end_year}."); sys.exit(1)

    all_data = []

    # --- Data Collection ---
    for year in sorted(years):
        print(f"Processing Year: {year}...")
        days = [d.strip('/') for d in get_links(f"{BASE_HTTPS_URL}{year}/") if d.strip('/').isdigit()]
        
        # Monthly snapshots (approx first day of each month)
        targets = ["001", "032", "060", "091", "121", "152", "182", "213", "244", "274", "305", "335"]
        sample_days = [d for d in days if d in targets]

        for day in sample_days:
            files = [f for f in get_links(f"{BASE_HTTPS_URL}{year}/{day}/") if f.endswith('.nc')]
            if not files: continue
            
            opendap_path = f"{DODS_URL}{year}/{day}/{files[0]}"
            
            try:
                with xr.open_dataset(opendap_path) as ds:
                    # Spatial subset + load to memory
                    subset = ds['analysed_sst'].sel(**CA_BOUNDS).load()
                    all_data.append(subset)
            except Exception as e:
                print(f"  Skipping {year}-{day}: {e}")

    if not all_data:
        print("CRITICAL ERROR: Data list is empty."); sys.exit(1)

    # --- Processing ---
    ds_full = xr.concat(all_data, dim='time')
    ds_full = ds_full - 273.15 # Kelvin to Celsius
    
    print("\n--> Calculating Seasonal SSTA...")
    # 1. Resample to Seasonal Mean (DJF, MAM, JJA, SON)
    seasonal_mean = ds_full.resample(time='QS-DEC').mean()

    # 2. Calculate Anomaly (Season value - Long-term mean of that specific season)
    climatology = seasonal_mean.groupby('time.season').mean('time')
    ssta_seasonal = seasonal_mean.groupby('time.season') - climatology

    # --- Output ---
    output_name = f"CA_SSTA_{start_year}_{end_year}.nc"
    ssta_seasonal.to_netcdf(output_name)
    print(f"\nSUCCESS: {output_name} saved.")
    print(f"Total seasonal steps: {len(ssta_seasonal.time)}")

if __name__ == "__main__":
    # You can now specify your start and terminal dates here:
    run_seasonal_pull(start_year=2000, end_year=2010)