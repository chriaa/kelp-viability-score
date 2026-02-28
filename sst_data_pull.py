import xarray as xr
from siphon.catalog import TDSCatalog
import sys

def get_ca_ssta(start_year=2010):
    base_url = "https://www.ncei.noaa.gov/thredds-ocean/catalog/ghrsst/L4/GLOB/JPL/MUR/catalog.html"
    print(f"--> Connecting to: {base_url}")
    
    try:
        cat = TDSCatalog(base_url)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not connect to NCEI. Check internet/URL.\n{e}")
        sys.exit(1)

    # 1. Filter Years (Fail fast if no years found)
    year_refs = [r for r in cat.catalog_refs if r[0].isdigit() and int(r.strip('/')) >= start_year]
    if not year_refs:
        print(f"ERROR: No folders found for year >= {start_year}. Check the catalog manually.")
        sys.exit(1)
    
    print(f"Found {len(year_refs)} year directories. Starting data pull...")

    all_data = []
    # California Bounding Box
    lat_range, lon_range = slice(32, 42), slice(-126, -116)

    for year_ref in sorted(year_refs):
        print(f"Processing {year_ref}...")
        year_cat = cat.catalog_refs[year_ref].follow()
        
        # MUR uses Day-of-Year subfolders (001, 002... 365)
        # To get "Monthly-ish" data fast, we grab the first day of each month
        # which corresponds to days roughly: 1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335
        target_days = ["001", "032", "060", "091", "121", "152", "182", "213", "244", "274", "305", "335"]
        
        day_refs = [d for d in year_cat.catalog_refs if d.strip('/') in target_days]
        
        for day_ref in day_refs:
            day_cat = year_cat.catalog_refs[day_ref].follow()
            
            # Get the first .nc file in this day folder
            if not day_cat.datasets:
                continue
                
            ds_name = list(day_cat.datasets.keys())[0]
            print(f"  Fetching: {ds_name}")
            
            try:
                # Access via OPeNDAP
                url = day_cat.datasets[ds_name].access_urls['OPENDAP']
                with xr.open_dataset(url, chunks={}) as ds:
                    # Select SST and crop to CA immediately to keep memory low
                    subset = ds['analysed_sst'].sel(lat=lat_range, lon=lon_range).load()
                    all_data.append(subset)
            except Exception as e:
                print(f"  Warning: Failed to pull {ds_name}: {e}")

    # 2. Final Check before processing
    if not all_data:
        print("FATAL ERROR: Script finished but 'all_data' is empty. No files were downloaded.")
        sys.exit(1)

    # 3. Combine and calculate Anomaly
    print("\nProcessing Anomaly...")
    full_ds = xr.concat(all_data, dim='time')
    full_ds = full_ds - 273.15  # Kelvin to Celsius
    
    # SSTA = Monthly Value - Long-term Mean
    ssta = full_ds - full_ds.mean(dim='time')
    
    ssta.to_netcdf("CA_SSTA_2010_Present.nc")
    print("SUCCESS: File saved as 'CA_SSTA_2010_Present.nc'")

if __name__ == "__main__":
    get_ca_ssta(2010)