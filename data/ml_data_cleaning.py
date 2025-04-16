import numpy as np
import pandas as pd

# Load the CSV
df = pd.read_csv("./data/Crime_Data_from_2020_to_Present.csv")

# Transform the 'part' column to 'serious_crime'
df['serious_crime'] = df['part'].apply(lambda x: 1 if x == 1 else 0)
df = df.drop(columns=['part'])

# Remove rows where victim_age, victim_sex, or victim_descent is blank or null.
required_victim_cols = ['victim_age', 'victim_sex', 'victim_descent']
df = df.dropna(subset=required_victim_cols)
df = df[df['victim_sex'].str.strip() != '']
df = df[df['victim_descent'].str.strip() != '']

# Remove rows where victim_age is 0 or victim_sex/victim_descent equal "X"
df = df[df['victim_age'] != 0]
df = df[(df['victim_sex'] != 'X') & (df['victim_descent'] != 'X')]

# df = df.dropna(subset=['weapon_code', 'premise_code'])

# Filter to only include the top 10 most common crime_code values.
top10_crime_codes = df['crime_code'].value_counts().nlargest(10).index
df = df[df['crime_code'].isin(top10_crime_codes)]

# Filter to keep only rows with the top 10 most frequent weapon_code, top 10 premise_code, and top 10 area values.
top10_weapon_codes = df['weapon_code'].value_counts().nlargest(10).index
top10_premise_codes = df['premise_code'].value_counts().nlargest(10).index
df = df[df['weapon_code'].isin(top10_weapon_codes) & df['premise_code'].isin(top10_premise_codes)]
top10_areas = df['area'].value_counts().nlargest(10).index
df = df[df['area'].isin(top10_areas)]

# Drop unnecessary columns.
cols_to_drop = [
    'division_number', 'date_reported', 'date_occurred', 'modus_operandi',
    'status', 'status_description', 'crime_code_1', 'crime_code_2',
    'crime_code_3', 'crime_code_4', 'location', 'cross_street', 'reporting_district'
]
df = df.drop(columns=cols_to_drop)

# Store original mappings of categorical variables.
weapon_map_orig = df[['weapon_code', 'weapon_description']].drop_duplicates()
premise_map_orig = df[['premise_code', 'premise_description']].drop_duplicates()
area_map_orig    = df[['area', 'area_name']].drop_duplicates()
crime_map_orig   = df[['crime_code', 'crime_description']].drop_duplicates()

# Coordinates of the center of the city of Los Angeles.
center_lat = 34.055913  
center_lon = -118.242575 

# Method that calculates distance between two coordinates.
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth's radius in kilometers
    return c * r

# Feature engineered variable: distance from center.
df['dist_from_center'] = haversine(df['latitude'], df['longitude'], center_lat, center_lon)

# Remap the categorical variables to continuous integers.
def create_mapping(series):
    # Create a mapping from each unique value to an integer starting at 0.
    unique_vals = sorted(series.dropna().unique())
    return {val: int(i) for i, val in enumerate(unique_vals)}

crime_code_dict   = create_mapping(df['crime_code'])
weapon_code_dict  = create_mapping(df['weapon_code'])
premise_code_dict = create_mapping(df['premise_code'])
area_dict         = create_mapping(df['area'])

# Apply the new mappings.
df['crime_code']   = df['crime_code'].map(crime_code_dict).astype(int)
df['weapon_code']  = df['weapon_code'].map(weapon_code_dict).astype(int)
df['premise_code'] = df['premise_code'].map(premise_code_dict).astype(int)
df['area']         = df['area'].map(area_dict).astype(int)

# Change the original codes to the new continuous integer codes.

# For crime:
crime_mapping = crime_map_orig.drop_duplicates(subset=['crime_code']).copy()
crime_mapping['crime_code'] = crime_mapping['crime_code'].map(crime_code_dict)
crime_mapping = crime_mapping[['crime_code', 'crime_description']]
crime_mapping = crime_mapping.dropna(subset=['crime_code', 'crime_description'])
crime_mapping['crime_code'] = crime_mapping['crime_code'].astype(int)

# Combine simple assault and theft from vehicle into one group.
simple_assault_group = ["BATTERY - SIMPLE ASSAULT", "INTIMATE PARTNER - SIMPLE ASSAULT"]

# Combine simple assault and theft from vehicle into one group.
theft_vehicle_group = ["THEFT FROM MOTOR VEHICLE - GRAND ($950.01 AND OVER)", "BURGLARY FROM VEHICLE"]

# Identify old crime codes (after remapping) for each group.
simple_assault_codes = crime_mapping[crime_mapping['crime_description'].isin(simple_assault_group)]['crime_code'].unique().tolist()
theft_vehicle_codes = crime_mapping[crime_mapping['crime_description'].isin(theft_vehicle_group)]['crime_code'].unique().tolist()

# Choose a unified crime code for each group; here, we take the minimum code in each group.
if simple_assault_codes:
    unified_simple_assault_code = min(simple_assault_codes)
if theft_vehicle_codes:
    unified_theft_vehicle_code = min(theft_vehicle_codes)

# Update the DataFrame: reassign crime_code for rows in these groups.
df.loc[df['crime_code'].isin(simple_assault_codes), 'crime_code'] = unified_simple_assault_code
df.loc[df['crime_code'].isin(theft_vehicle_codes), 'crime_code'] = unified_theft_vehicle_code

# Remove the old mappings that belong to these groups.
crime_mapping = crime_mapping[~crime_mapping['crime_code'].isin(simple_assault_codes + theft_vehicle_codes)]
# Append new unified mapping rows.
if simple_assault_codes:
    crime_mapping = pd.concat([crime_mapping, pd.DataFrame({
        'crime_code': [unified_simple_assault_code],
        'crime_description': ['SIMPLE ASSAULT']
    })], ignore_index=True)
if theft_vehicle_codes:
    crime_mapping = pd.concat([crime_mapping, pd.DataFrame({
        'crime_code': [unified_theft_vehicle_code],
        'crime_description': ['THEFT FROM VEHICLE']
    })], ignore_index=True)


# Make mappings for categorical variables:

# For weapon:
weapon_mapping = weapon_map_orig.drop_duplicates(subset=['weapon_code']).copy()
weapon_mapping['weapon_code'] = weapon_mapping['weapon_code'].map(weapon_code_dict)
weapon_mapping = weapon_mapping[['weapon_code', 'weapon_description']]
weapon_mapping = weapon_mapping.dropna(subset=['weapon_code', 'weapon_description'])
weapon_mapping['weapon_code'] = weapon_mapping['weapon_code'].astype(int)

# For premise:
premise_mapping = premise_map_orig.drop_duplicates(subset=['premise_code']).copy()
premise_mapping['premise_code'] = premise_mapping['premise_code'].map(premise_code_dict)
premise_mapping = premise_mapping[['premise_code', 'premise_description']]
premise_mapping = premise_mapping.dropna(subset=['premise_code', 'premise_description'])
premise_mapping['premise_code'] = premise_mapping['premise_code'].astype(int)

# For area:
area_mapping = area_map_orig.drop_duplicates(subset=['area']).copy()
area_mapping['area_code'] = area_mapping['area'].map(area_dict)
area_mapping = area_mapping[['area_code', 'area_name']]
area_mapping = area_mapping.dropna(subset=['area_code', 'area_name'])
area_mapping['area_code'] = area_mapping['area_code'].astype(int)

# One-hot encode the categorical columns.
# Ensure weapon code is an integer before one-hot encoding.
df['weapon_code'] = df['weapon_code'].astype(int)

# One-hot encode the categorical columns.
df = pd.get_dummies(df, columns=['weapon_code', 'premise_code', 'area'],
                            prefix=['weapon', 'premise', 'area'])

# Remove outliers in the'dist_from_center' column using the IQR method.
Q1 = df['dist_from_center'].quantile(0.25)
Q3 = df['dist_from_center'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['dist_from_center'] >= lower_bound) & (df['dist_from_center'] <= upper_bound)]

# Normalize the distance to range [0, 1].
min_distance = df['dist_from_center'].min()
max_distance = df['dist_from_center'].max()
df['norm_distance'] = (df['dist_from_center'] - min_distance) / (max_distance - min_distance)

# Compute the angle from the center.
df['delta_lat'] = df['latitude'] - center_lat
df['delta_lon'] = df['longitude'] - center_lon
df['angle'] = np.arctan2(df['delta_lon'], df['delta_lat'])
# Shift angle from [-pi, pi] to [0, 2pi] and then normalize.
df['norm_angle'] = (df['angle'] + np.pi) / (2 * np.pi)

# Drop unnecessary coordinate columns.
df = df.drop(columns=['latitude', 'longitude', 'delta_lat', 'delta_lon', 'angle', 'dist_from_center'])

# Normalize victim_age to range [0, 1].
min_age = df['victim_age'].min()
max_age = df['victim_age'].max()
df['victim_age'] = (df['victim_age'] - min_age) / (max_age - min_age)

# Map victim_sex to binary: "M" to 1 and "F" to 0.
df['victim_sex'] = df['victim_sex'].map({'M': 1, 'F': 0})
df = df.dropna(subset=['victim_sex'])
df['victim_sex'] = df['victim_sex'].astype(int)

# Filter to include only the allowed victim_descent codes (These are the ones that constitute the majority of the data).
allowed_descent = ['A', 'B', 'H', 'O', 'W']
df = df[df['victim_descent'].isin(allowed_descent)]

# Create a mapping for the allowed descent codes and their full names.
descent_map = {
    'A': 'Asian',
    'B': 'Black',
    'H': 'Hispanic',
    'O': 'Other',
    'W': 'White'
}

# Save the descent mapping as a CSV.
descent_mapping = pd.DataFrame({
    'descent_code': list(descent_map.keys()),
    'descent_description': list(descent_map.values())
})
descent_mapping.to_csv("./data/descent_mapping.csv", index=False)

# One-hot encode the victim_descent column.
df = pd.get_dummies(df, columns=['victim_descent'], prefix='victim_descent')

# Drop description columns as they are now in the mapping CSVs.
df = df.drop(columns=['weapon_description', 'premise_description', 'area_name', 'crime_description'])

# Convert boolean columns to integers.
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

# Remove crime codes with less than 500 occurrences.
crime_freq = df['crime_code'].value_counts()
valid_crime_codes = crime_freq[crime_freq >= 500].index.tolist()
df = df[df['crime_code'].isin(valid_crime_codes)]
# Update crime_mapping accordingly.
crime_mapping = crime_mapping[crime_mapping['crime_code'].isin(valid_crime_codes)]

# Filter out categorical columns with less than 150 occurrences.
def filter_dummy_columns(df, prefix, mapping_df, mapping_col):
    # Get list of dummy column names with this prefix.
    dummy_cols = [col for col in df.columns if col.startswith(prefix + "_")]
    # Sum up the ones in each column.
    counts = df[dummy_cols].sum()
    # Keep only the columns with count >= 150.
    valid_cols = counts[counts >= 150].index.tolist()
    # Drop the columns that are not valid.
    to_drop = set(dummy_cols) - set(valid_cols)
    df = df.drop(columns=list(to_drop))
    # Update mapping_df: extract valid numeric code from valid_cols.
    valid_codes = [int(col.split('_')[1]) for col in valid_cols]
    mapping_df = mapping_df[mapping_df[mapping_col].isin(valid_codes)]
    return df, mapping_df

df, weapon_mapping = filter_dummy_columns(df, "weapon", weapon_mapping, "weapon_code")
df, premise_mapping = filter_dummy_columns(df, "premise", premise_mapping, "premise_code")
df, area_mapping = filter_dummy_columns(df, "area", area_mapping, "area_code")

# Limit the number of samples for each crime code to 3000 to keep balance.
samples_list = []
for code, group in df.groupby('crime_code'):
    if len(group) >= 3000:
        sample = group.sample(n=3000, random_state=42)
    else:
        sample = group
    print(f"Crime code {code} sample size after combining: {len(sample)}")
    samples_list.append(sample)
df = pd.concat(samples_list)
# Now we assign this final DataFrame to df.
df = df.copy()

# Write all dataframes to files.
df.to_csv("./data/cleaned_crime_data_stratified.csv", index=False)
weapon_mapping.to_csv("./data/weapon_mapping.csv", index=False)
premise_mapping.to_csv("./data/premise_mapping.csv", index=False)
area_mapping.to_csv("./data/area_mapping.csv", index=False)
crime_mapping.to_csv("./data/crime_mapping.csv", index=False)

total_rows = df.shape[0]
print(f"Total filtered rows: {total_rows}")
print(f"Unique crime codes (after remapping, combining, sampling, and filtering): {df['crime_code'].nunique()}")
print(f"Unique weapon codes (after remapping): {len(weapon_code_dict)}")
print(f"Unique premise codes (after remapping): {len(premise_code_dict)}")
print(f"Unique area codes (after remapping): {len(area_dict)}")

print("\nNumber of rows per crime code in the final dataset:")
crime_counts = df['crime_code'].value_counts().sort_index()
for code, count in crime_counts.items():
    print(f"Crime code {code}: {count} rows")