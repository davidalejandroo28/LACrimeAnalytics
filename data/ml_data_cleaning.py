import numpy as np
import pandas as pd

# Load the CSV
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

# Transform the 'part' column to 'serious_crime'
df['serious_crime'] = df['part'].apply(lambda x: 1 if x == 1 else 0)
df = df.drop(columns=['part'])

# Remove rows where victim_age, victim_sex, or victim_descent is blank or null.
required_victim_cols = ['victim_age', 'vi ctim_sex', 'victim_descent']
df = df.dropna(subset=required_victim_cols)
df = df[df['victim_sex'].str.strip() != '']
df = df[df['victim_descent'].str.strip() != '']

# Remove rows where victim_age is 0 or victim_sex/victim_descent equal "X"
df = df[df['victim_age'] != 0]
df = df[(df['victim_sex'] != 'X') & (df['victim_descent'] != 'X')]

# Drop rows that are null for weapon_code and premise_code.
df = df.dropna(subset=['weapon_code', 'premise_code'])

# Filter to only include the top 10 most common crime_code values.
top10_crime_codes = df['crime_code'].value_counts().nlargest(10).index
df = df[df['crime_code'].isin(top10_crime_codes)]

# Further filter rows to keep only those with the top 10 most frequent weapon_code and top 10 premise_code.
top10_weapon_codes = df['weapon_code'].value_counts().nlargest(10).index
top10_premise_codes = df['premise_code'].value_counts().nlargest(10).index
df = df[df['weapon_code'].isin(top10_weapon_codes) & df['premise_code'].isin(top10_premise_codes)]

# Drop unnecessary columns.
cols_to_drop = [
    'division_number', 'date_reported', 'date_occurred', 'modus_operandi',
    'status', 'status_description', 'crime_code_1', 'crime_code_2',
    'crime_code_3', 'crime_code_4', 'location', 'cross_street', 'reporting_district'
]
df = df.drop(columns=cols_to_drop)

# Store original mappings of categorical variables
weapon_map_orig = df[['weapon_code', 'weapon_description']].drop_duplicates()
premise_map_orig = df[['premise_code', 'premise_description']].drop_duplicates()
area_map_orig    = df[['area', 'area_name']].drop_duplicates()
crime_map_orig   = df[['crime_code', 'crime_description']].drop_duplicates()

# Define the center (using city center coordinates)
center_lat = 34.055913  
center_lon = -118.242575 

# method that calculates distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth's radius in kilometers
    return c * r

# feature engineered variable
df['dist_from_center'] = haversine(df['latitude'], df['longitude'], center_lat, center_lon)

# Keep only rows from the top 10 most common areas.
top10_areas = df['area'].value_counts().nlargest(10).index
df = df[df['area'].isin(top10_areas)]

# For each of the top 10 crime codes, randomly sample up to 3000 rows if available.
samples_list = []
for code, group in df.groupby('crime_code'):
    if len(group) >= 3000:
        sample = group.sample(n=3000, random_state=42)
    else:
        sample = group
    print(f"Crime code {code} sample size: {len(sample)}")
    samples_list.append(sample)
df_sampled = pd.concat(samples_list)

# Remap the categorical variables to continuous integers.
def create_mapping(series):
    # Create a mapping from each unique value to an integer starting at 0.
    unique_vals = sorted(series.dropna().unique())
    return {val: int(i) for i, val in enumerate(unique_vals)}

crime_code_dict   = create_mapping(df_sampled['crime_code'])
weapon_code_dict  = create_mapping(df_sampled['weapon_code'])
premise_code_dict = create_mapping(df_sampled['premise_code'])
area_dict         = create_mapping(df_sampled['area'])

df_sampled['crime_code']   = df_sampled['crime_code'].map(crime_code_dict).astype(int)
df_sampled['weapon_code']  = df_sampled['weapon_code'].map(weapon_code_dict).astype(int)
df_sampled['premise_code'] = df_sampled['premise_code'].map(premise_code_dict).astype(int)
df_sampled['area']         = df_sampled['area'].map(area_dict).astype(int)

# Change the original codes to the new continuouis integer codes.

# For crime:
crime_mapping = crime_map_orig.drop_duplicates(subset=['crime_code']).copy()
crime_mapping['crime_code'] = crime_mapping['crime_code'].map(crime_code_dict)
crime_mapping = crime_mapping[['crime_code', 'crime_description']]
crime_mapping = crime_mapping.dropna(subset=['crime_code', 'crime_description'])
crime_mapping['crime_code'] = crime_mapping['crime_code'].astype(int)

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
df_sampled = pd.get_dummies(df_sampled, columns=['weapon_code', 'premise_code', 'area'],
                            prefix=['weapon', 'premise', 'area'])

# Remove outliers based on 'dist_from_center' using the IQR method.
Q1 = df_sampled['dist_from_center'].quantile(0.25)
Q3 = df_sampled['dist_from_center'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_sampled = df_sampled[(df_sampled['dist_from_center'] >= lower_bound) & (df_sampled['dist_from_center'] <= upper_bound)]

# Normalize the distance to range [0, 1].
min_distance = df_sampled['dist_from_center'].min()
max_distance = df_sampled['dist_from_center'].max()
df_sampled['norm_distance'] = (df_sampled['dist_from_center'] - min_distance) / (max_distance - min_distance)

# Compute the angle from the center.
df_sampled['delta_lat'] = df_sampled['latitude'] - center_lat
df_sampled['delta_lon'] = df_sampled['longitude'] - center_lon
df_sampled['angle'] = np.arctan2(df_sampled['delta_lon'], df_sampled['delta_lat'])
# Shift angle from [-pi, pi] to [0, 2pi] and then normalize.
df_sampled['norm_angle'] = (df_sampled['angle'] + np.pi) / (2 * np.pi)

# Drop unnecessary coordinate columns.
df_sampled = df_sampled.drop(columns=['latitude', 'longitude', 'delta_lat', 'delta_lon', 'angle', 'dist_from_center'])

# Normalize victim_age to range [0, 1].
min_age = df_sampled['victim_age'].min()
max_age = df_sampled['victim_age'].max()
df_sampled['victim_age'] = (df_sampled['victim_age'] - min_age) / (max_age - min_age)

# Map victim_sex to binary: "M" to 1 and "F" to 0.
df_sampled['victim_sex'] = df_sampled['victim_sex'].map({'M': 1, 'F': 0})
df_sampled = df_sampled.dropna(subset=['victim_sex'])
df_sampled['victim_sex'] = df_sampled['victim_sex'].astype(int)

# Remove uncommon victim_descent codes.
exclusion_set = set(['C', 'F', 'G', 'I', 'J', 'K', 'V'])
df_sampled = df_sampled[~df_sampled['victim_descent'].isin(exclusion_set)]

# Dictionary for the remaining descent codes and their full names.
descent_map = {
    'A': 'Asian',
    'B': 'Black',
    'H': 'Hispanic',
    'O': 'Other',
    'W': 'White'
}

# Create a mapping DataFrame for victim_descent and save it as a CSV.
descent_mapping = pd.DataFrame({
    'descent_code': list(descent_map.keys()),
    'descent_description': list(descent_map.values())
})
descent_mapping.to_csv("descent_mapping.csv", index=False)

# One-hot encode the victim_descent column.
df_sampled = pd.get_dummies(df_sampled, columns=['victim_descent'], prefix='victim_descent')

# Drop description columns as they are now in the mapping csvs.
df_sampled = df_sampled.drop(columns=['weapon_description', 'premise_description', 'area_name', 'crime_description'])

# Convert boolean columns to integers.
for col in df_sampled.columns:
    if df_sampled[col].dtype == 'bool':
        df_sampled[col] = df_sampled[col].astype(int)

# Write all dataframes to files.
df_sampled.to_csv("cleaned_crime_data_stratified.csv", index=False)
weapon_mapping.to_csv("weapon_mapping.csv", index=False)
premise_mapping.to_csv("premise_mapping.csv", index=False)
area_mapping.to_csv("area_mapping.csv", index=False)
crime_mapping.to_csv("crime_mapping.csv", index=False)

total_rows = df_sampled.shape[0]
print(f"Total filtered rows: {total_rows}")
print(f"Unique crime codes (after remapping): {len(crime_code_dict)}")
print(f"Unique weapon codes (after remapping): {len(weapon_code_dict)}")
print(f"Unique premise codes (after remapping): {len(premise_code_dict)}")
print(f"Unique area codes (after remapping): {len(area_dict)}")

print("Cleaning, filtering, remapping, one-hot encoding, new feature engineering, and stratified sampling complete.")
print("Final dataset is saved as 'cleaned_crime_data_stratified.csv'.")
print("Mapping CSVs are saved as 'weapon_mapping.csv', 'premise_mapping.csv', 'area_mapping.csv', 'crime_mapping.csv', and 'descent_mapping.csv'.")