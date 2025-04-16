import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset.
data_path = "./data/cleaned_crime_data_stratified.csv"
directory = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(data_path)

# Create a bar plot for the distribution of crime_code.
crime_counts = df['crime_code'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
crime_counts.plot(kind='bar', color='skyblue')
plt.xlabel("Crime Code")
plt.ylabel("Number of Occurrences")
plt.title("Distribution of Crime Codes")
plt.tight_layout()
plt.savefig(f"{directory}/crime_code_distribution.png")
plt.close()

# For the one-hot encoded features, we first filter the columns by prefix.
# Then we sum the values (since they are 0/1, sum indicates the count of 1s per column).

# Distribution for weapon.
weapon_cols = [col for col in df.columns if col.startswith("weapon_")]
weapon_counts = df[weapon_cols].sum().sort_index()
plt.figure(figsize=(8, 6))
weapon_counts.plot(kind='bar', color='salmon')
plt.xlabel("Weapon Code")
plt.ylabel("Count of 1s")
plt.title("Distribution of Weapon Codes (One-Hot Encoded)")
plt.tight_layout()
plt.savefig(f"{directory}/weapon_distribution.png")
plt.close()

# Distribution for area.
area_cols = [col for col in df.columns if col.startswith("area_")]
area_counts = df[area_cols].sum().sort_index()
plt.figure(figsize=(8, 6))
area_counts.plot(kind='bar', color='lightgreen')
plt.xlabel("Area Code")
plt.ylabel("Count of 1s")
plt.title("Distribution of Area Codes (One-Hot Encoded)")
plt.tight_layout()
plt.savefig(f"{directory}/area_distribution.png")
plt.close()

# Distribution for premise.
premise_cols = [col for col in df.columns if col.startswith("premise_")]
premise_counts = df[premise_cols].sum().sort_index()
plt.figure(figsize=(8, 6))
premise_counts.plot(kind='bar', color='plum')
plt.xlabel("Premise Code")
plt.ylabel("Count of 1s")
plt.title("Distribution of Premise Codes (One-Hot Encoded)")
plt.tight_layout()
plt.savefig(f"{directory}/premise_distribution.png")
plt.close()

print("Plots saved: crime_code_distribution.png, weapon_distribution.png, area_distribution.png, premise_distribution.png")
