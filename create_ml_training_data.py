"""
ML Training Dataset Creation Script
====================================
This script creates a CSV of points for ML training by integrating:
1. Kelp bed observations (2000 onwards) with estimated centroids
2. CUTI (Coastal Upwelling Transport Index) data by latitude band matching
3. CA SSTA (Sea Surface Temperature Anomaly) data for available dates

METHODOLOGY NOTES:
- Kelp bed centroids: Estimated using simple lat/lon bounds from administrative data
  (See METHODOLOGY.md for details on coordinate estimation)
- CUTI matching: Each kelp bed is matched to the nearest available latitude band
  (e.g., 31N, 32N, etc.) based on estimated latitude
- SSTA integration: Only 2 timesteps available (2009-03-01, 2009-06-01)
  Values are matched to nearest available date
- Target variable: Set to 1 where kelp is present (TOTAL > 0), 0 otherwise
"""

import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from scipy.spatial.distance import cdist

# ============================================================================
# SECTION 1: LOAD AND PREPARE DATA
# ============================================================================

print("[1] Loading datasets...")

# Load kelp data (2000 onwards)
kelp_df = pd.read_csv('data/final_data/Updated_kelp_data_2000_onwards.csv')
print(f"  Loaded {len(kelp_df)} kelp observations")

# Load administrative data for bed geometry
admin_df = pd.read_csv('data/Administrative_Kelp_Beds_-_R7_-_CDFW_[ds3135].csv')
print(f"  Loaded {len(admin_df)} administrative records")

# Load CUTI quarterly data
cuti_df = pd.read_csv('data/final_data/Updated_final_input_CUTI_quarterly_averages.csv')
print(f"  Loaded {len(cuti_df)} CUTI quarterly records")

# Load SSTA netCDF
ssta_ds = xr.open_dataset('data/final_data/CA_SSTA_2000_2010.nc')
print(f"  Loaded SSTA data: {dict(ssta_ds.dims)}")

# ============================================================================
# SECTION 2: CREATE SYNTHETIC KELP BED CENTROIDS
# ============================================================================

print("\n[2] Generating kelp bed centroids...")

# Create a simple synthetic centroid mapping based on bed number location patterns
# This uses a rough geographic distribution of California kelp beds
# NOTE: These are ESTIMATED centroids until actual polygon data is available

def estimate_kelp_bed_centroid(bed_num):
    """
    Estimates lat/lon for a kelp bed number based on known geographic distribution.
    Beds 1-30: Southern California (around 32-34N, -117 to -120W)
    Beds 30-50: Central Coast (around 35-37N, -120 to -122W)
    Beds 50+: Northern regions (around 37-39N, -121 to -123W)
    
    This is a temporary solution. Real coordinates should come from polygon centroids.
    """
    if bed_num <= 30:
        # Southern CA: beds 1-30
        lat = 32.5 + (bed_num / 30) * 1.5
        lon = -119.5 - (bed_num / 30) * 1.0
    elif bed_num <= 50:
        # Central CA: beds 31-50
        lat = 34.0 + ((bed_num - 30) / 20) * 2.0
        lon = -120.5 - ((bed_num - 30) / 20) * 1.5
    else:
        # Northern CA: beds 51+
        lat = 36.0 + ((bed_num - 50) / 30) * 2.5
        lon = -121.5 - ((bed_num - 50) / 30) * 1.5
    
    return lat, lon

# Create centroid mapping
bed_centroids = {}
for bed in admin_df['KelpBed'].unique():
    try:
        bed_num = int(bed) if bed.isdigit() else int(bed.split()[0])
        lat, lon = estimate_kelp_bed_centroid(bed_num)
        bed_centroids[str(bed)] = {'lat': lat, 'lon': lon}
    except:
        # Fallback for non-standard bed numbering (e.g., "107A")
        base_num = int(''.join(c for c in str(bed) if c.isdigit()))
        lat, lon = estimate_kelp_bed_centroid(base_num)
        bed_centroids[str(bed)] = {'lat': lat, 'lon': lon}

print(f"  Generated centroids for {len(bed_centroids)} kelp beds")

# ============================================================================
# SECTION 3: MAP KELP DATA WITH COORDINATES
# ============================================================================

print("\n[3] Preparing kelp data with coordinates...")

# Convert date and numeric columns
kelp_df['DATE_OF_SURVEY'] = pd.to_datetime(kelp_df['DATE_OF_SURVEY'], errors='coerce')
kelp_df['TOTAL'] = pd.to_numeric(kelp_df['TOTAL'], errors='coerce')
kelp_df['AVAILABLE'] = pd.to_numeric(kelp_df['AVAILABLE'], errors='coerce')
kelp_df['HARVESTED'] = pd.to_numeric(kelp_df['HARVESTED'], errors='coerce')
kelp_df['YEAR'] = pd.to_numeric(kelp_df['YEAR'], errors='coerce')
kelp_df['MONTH'] = pd.to_numeric(kelp_df['MONTH'], errors='coerce')

# Add coordinates from centroids
kelp_df['BED'] = kelp_df['BED'].astype(str)
kelp_df['lat'] = kelp_df['BED'].map(lambda x: bed_centroids.get(x, {}).get('lat', np.nan))
kelp_df['lon'] = kelp_df['BED'].map(lambda x: bed_centroids.get(x, {}).get('lon', np.nan))

# Remove rows without coordinate mapping
kelp_df = kelp_df.dropna(subset=['lat', 'lon'])
print(f"  {len(kelp_df)} observations have coordinate mappings")

# ============================================================================
# SECTION 4: MATCH CUTI DATA BY NEAREST LATITUDE BAND
# ============================================================================

print("\n[4] Integrating CUTI data...")

# Extract CUTI latitude bands (31N through 47N)
cuti_lats = [int(col[:-1]) for col in cuti_df.columns if col.endswith('N')]
cuti_lats_sorted = sorted(cuti_lats)
print(f"  Available CUTI latitude bands: {cuti_lats_sorted[0]}N to {cuti_lats_sorted[-1]}N")

def get_nearest_cuti_lat(lat):
    """Find nearest CUTI latitude band for a given latitude."""
    return min(cuti_lats_sorted, key=lambda x: abs(x - lat))

# Add CUTI latitude band mapping
kelp_df['cuti_lat_band'] = kelp_df['lat'].map(get_nearest_cuti_lat)
kelp_df['cuti_lat_str'] = kelp_df['cuti_lat_band'].astype(str) + 'N'

# Merge CUTI data: match year, quarter, and latitude band
kelp_df['quarter'] = ((kelp_df['MONTH'] - 1) // 3 + 1).astype(int)

cuti_df['year'] = pd.to_numeric(cuti_df['year'], errors='coerce')
cuti_df['quarter'] = pd.to_numeric(cuti_df['quarter'], errors='coerce')

kelp_df = kelp_df.merge(
    cuti_df[['year', 'quarter']].merge(
        cuti_df,
        on=['year', 'quarter']
    ).rename(columns={'year': 'YEAR', 'quarter': 'quarter'}),
    on=['YEAR', 'quarter'],
    how='left'
)

# Extract CUTI value for the matched latitude band
def extract_cuti_value(row):
    lat_col = row['cuti_lat_str']
    if pd.isna(row[lat_col]):
        return np.nan
    return row[lat_col]

kelp_df['CUTI'] = kelp_df.apply(extract_cuti_value, axis=1)
kelp_df['CUTI'] = pd.to_numeric(kelp_df['CUTI'], errors='coerce')

print(f"  Matched {kelp_df['CUTI'].notna().sum()} observations with CUTI data")

# ============================================================================
# SECTION 5: INTEGRATE SSTA DATA
# ============================================================================

print("\n[5] Integrating SSTA data...")

# Extract SSTA times and create time to date mapping
ssta_times = ssta_ds['time'].values
ssta_dates = pd.to_datetime(ssta_times).date

print(f"  Available SSTA dates: {ssta_dates}")

# Get SSTA values at nearest spatial point for each observation
ssta_lat = ssta_ds['lat'].values
ssta_lon = ssta_ds['lon'].values

def get_ssta_at_location(lat, lon, date_to_match):
    """
    Extract SSTA value at nearest grid point for a given lat/lon and date.
    """
    # Find nearest grid cell
    distances = cdist(
        [[lat, lon]],
        np.column_stack([ssta_lat.ravel(), ssta_lon.ravel()])
    )[0]
    nearest_idx = np.argmin(distances)
    
    # Convert flat index to 2D coordinates
    row = nearest_idx // len(ssta_lon)
    col = nearest_idx % len(ssta_lon)
    
    # Find nearest available SSTA date
    date_diffs = [abs((pd.to_datetime(d).date() - date_to_match).days) for d in ssta_times]
    nearest_time_idx = np.argmin(date_diffs)
    
    # Extract SSTA value
    ssta_val = ssta_ds['analysed_sst'].isel(time=nearest_time_idx).values[row, col]
    
    return float(ssta_val) if not np.isnan(ssta_val) else np.nan

# Only process observations with SSTA date coverage (within range of available dates)
kelp_df['DATE_DATE'] = kelp_df['DATE_OF_SURVEY'].dt.date
ssta_date_range = (pd.to_datetime(min(ssta_dates)), pd.to_datetime(max(ssta_dates)))

print(f"  SSTA date range: {ssta_date_range[0].date()} to {ssta_date_range[1].date()}")
print(f"  Note: SSTA data is limited to 2 timesteps. Only observations within this range will have SSTA.")

kelp_df['SSTA'] = np.nan

# Apply SSTA extraction only to observations within date range
within_range = (
    (kelp_df['DATE_OF_SURVEY'] >= ssta_date_range[0]) &
    (kelp_df['DATE_OF_SURVEY'] <= ssta_date_range[1])
)

if within_range.sum() > 0:
    for idx in kelp_df[within_range].index:
        try:
            lat = kelp_df.loc[idx, 'lat']
            lon = kelp_df.loc[idx, 'lon']
            date = kelp_df.loc[idx, 'DATE_DATE']
            ssta_val = get_ssta_at_location(lat, lon, date)
            kelp_df.loc[idx, 'SSTA'] = ssta_val
        except Exception as e:
            pass

print(f"  Matched {kelp_df['SSTA'].notna().sum()} observations with SSTA data")

# ============================================================================
# SECTION 6: CREATE TARGET VARIABLE AND FINALIZE DATASET
# ============================================================================

print("\n[6] Creating ML training dataset...")

# Create target variable: 1 if kelp is present (TOTAL > 0), 0 otherwise
kelp_df['target'] = (kelp_df['TOTAL'] > 0).astype(int)

# Select relevant columns for ML training
output_cols = [
    'DATE_OF_SURVEY',
    'YEAR',
    'MONTH',
    'lat',
    'lon',
    'BED',
    'BED_NAME',
    'REGION',
    'TOTAL',
    'AVAILABLE',
    'HARVESTED',
    'Shape__Area',
    'Shape__Length',
    'cuti_lat_band',
    'CUTI',
    'SSTA',
    'target'
]

ml_df = kelp_df[[col for col in output_cols if col in kelp_df.columns]].copy()

# Sort by date
ml_df = ml_df.sort_values('DATE_OF_SURVEY')

# Write output
output_path = 'data/ml_training_data.csv'
ml_df.to_csv(output_path, index=False)

print(f"\n✓ ML training dataset created: {output_path}")
print(f"  Shape: {ml_df.shape}")
print(f"  Date range: {ml_df['DATE_OF_SURVEY'].min()} to {ml_df['DATE_OF_SURVEY'].max()}")
print(f"  Rows with complete features: {ml_df.dropna().shape[0]}")
print(f"\nDataset summary:")
print(ml_df.describe())

# ============================================================================
# SECTION 7: SAVE METHODOLOGY NOTES
# ============================================================================

methodology_notes = """
ML TRAINING DATA CREATION - METHODOLOGY NOTES
==============================================

DATE CREATED: {date}

1. KELP BED COORDINATES (Temporary Solution)
   - Since actual polygon centroid data is not available in the current dataset,
     synthetic centroids were generated based on bed numbering patterns and 
     known geographic distribution of California kelp beds.
   - Southern CA beds (1-30): Estimated around 32-34°N, 119-120°W
   - Central Coast beds (31-50): Estimated around 34-36°N, 120-121°W  
   - Northern CA beds (51+): Estimated around 36-39°N, 121-123°W
   - These coordinates are ESTIMATES and should be replaced with actual 
     polygon centroids when available.
   
   ACTION REQUIRED: Obtain polygon geometry data for all kelp beds and 
   calculate true centroids for improved accuracy.

2. CUTI INTEGRATION (Coastal Upwelling Transport Index)
   - Each kelp bed observation is matched to the nearest available CUTI 
     latitude band (31N, 32N, ..., 47N) based on estimated latitude.
   - Matching uses the quarter of the observation year.
   - If a kelp observation doesn't have a matching CUTI entry, the value is NaN.
   
   APPROACH: Nearest-neighbor latitude band matching
   NOTES: CUTI data is quarterly. Observations are assigned to the quarter
          they fall in (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)

3. SSTA INTEGRATION (Sea Surface Temperature Anomaly)
   - Only 2 timesteps of SSTA data are available: 2009-03-01 and 2009-06-01
   - For observations within this date range, SSTA values are extracted at 
     the nearest grid cell to the estimated kelp bed location.
   - For observations outside this date range, SSTA is NaN.
   
   APPROACH: Nearest grid cell spatial matching for available dates
   LIMITATION: Very limited temporal coverage (only 2 dates in 10-year span)
   
   ACTION REQUIRED: Obtain complete SSTA dataset covering 2000-2010 
   for better temporal integration.

4. TARGET VARIABLE
   - Set to 1 if TOTAL kelp biomass > 0 (kelp is present)
   - Set to 0 if TOTAL kelp biomass = 0 (no kelp observed)
   - Can be used as classification target: presence/absence of kelp

5. DATA QUALITY NOTES
   - {total_rows} total observations in 2000+ kelp dataset
   - {mapped_rows} observations have valid coordinates and are included
   - {cuti_match} observations matched with CUTI data ({cuti_pct}%)
   - {ssta_match} observations matched with SSTA data ({ssta_pct}%)
   - {complete_rows} observations have all features (no NaN values)

6. MISSING VALUES
   - CUTI: Missing when no matching year/quarter found in CUTI dataset
   - SSTA: Missing for observations outside 2009-03-01 to 2009-06-01 date range
           and for grid cells with invalid/missing SSTA values
   - Handle missing values in ML preprocessing (imputation, removal, or as feature)

7. RECOMMENDATIONS FOR NEXT STEPS
   a) Obtain polygon geometry and compute true centroids for all kelp beds
   b) Acquire complete SSTA daily or monthly data for 2000-2010
   c) Validate coordinate estimates by cross-referencing with regional maps
   d) Consider other environmental variables (temperature, nutrients, etc.)
   e) Evaluate feature importance to identify key drivers of kelp presence/absence
""".format(
    date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    total_rows=len(kelp_df),
    mapped_rows=len(kelp_df.dropna(subset=['lat', 'lon'])),
    cuti_match=kelp_df['CUTI'].notna().sum(),
    cuti_pct=round(100 * kelp_df['CUTI'].notna().sum() / len(kelp_df), 1),
    ssta_match=kelp_df['SSTA'].notna().sum(),
    ssta_pct=round(100 * kelp_df['SSTA'].notna().sum() / len(kelp_df), 1),
    complete_rows=ml_df.dropna().shape[0]
)

notes_path = 'data/ML_TRAINING_DATA_METHODOLOGY.txt'
with open(notes_path, 'w') as f:
    f.write(methodology_notes)

print(f"\n✓ Methodology notes saved: {notes_path}")
print("\nSetup complete! Review the methodology notes for details on data handling.")
