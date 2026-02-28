import pandas as pd

# read administrative kelp beds with all columns
admin_path = "data/Administrative_Kelp_Beds_-_R7_-_CDFW_[ds3135].csv"
hist_path = "data/Historical_Kelp_Data.csv"

# load datasets
admin = pd.read_csv(admin_path, dtype=str)
hist = pd.read_csv(hist_path, dtype=str)

# ensure bed number columns are strings and stripped
admin['KelpBed'] = admin['KelpBed'].astype(str).str.strip()
hist['BED'] = hist['BED'].astype(str).str.strip()

# perform merge: left merge historical with administrative details on bed number
merged = hist.merge(admin, how='left', left_on='BED', right_on='KelpBed')

# output merged results to a new CSV
output_path = "data/merged_historical_with_admin.csv"
merged.to_csv(output_path, index=False)

print(f"Merged {len(merged)} rows. Output written to {output_path}.")

# optionally display some rows
print(merged.head())

# --- validation step ---------------------------------------------------
# check that the BED_NAME values are consistent for each BED value in the
# historical data.  this provides a simple way to validate that the area
# description (name) aligns with the bed number and can flag any anomalies.

# create a smaller dataframe of unique bed/name combinations
combo = merged[['BED', 'BED_NAME']].drop_duplicates()
beds_with_multiple_names = (
	combo.groupby('BED')['BED_NAME']
		 .nunique()
		 .loc[lambda x: x > 1]
)

if not beds_with_multiple_names.empty:
	print(f"\nFound {len(beds_with_multiple_names)} bed(s) with more than one name:")
	for bed in beds_with_multiple_names.index:
		names = combo.loc[combo['BED'] == bed, 'BED_NAME'].tolist()
		print(f"  BED {bed}: {names}")
	# write a small report csv as well
	report_path = "data/bed_name_conflicts.csv"
	combo[combo['BED'].isin(beds_with_multiple_names.index)].to_csv(report_path, index=False)
	print(f"Conflict details written to {report_path}")
else:
	print("\nAll beds have a unique name in the historical dataset.")
