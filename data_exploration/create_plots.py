import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

# Distributino for descent.
descent_cols = [col for col in df.columns if col.startswith("victim_descent_")]
descent_counts = df[descent_cols].sum().sort_index()
plt.figure(figsize=(8, 6))
descent_counts.plot(kind='bar', color='plum')
plt.xlabel("Victim Descent Codes")
plt.ylabel("Count of 1s")
plt.title("Distribution of Descent Codes (One-Hot Encoded)")
plt.tight_layout()
plt.savefig(f"{directory}/descent_distribution.png")
plt.close()

r     = df['norm_distance']
theta = df['norm_angle'] * 2 * np.pi

fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection':'polar'})
# optional: set zero at top, clockwise direction
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

sc = ax.scatter(theta, r, 
                c=r,             # you can color‐map by r or any other column
                cmap='viridis', 
                alpha=0.7, 
                s=20)

ax.set_rlim(0, 1)
ax.set_title("Unit‑Circle Plot (Polar) of norm_distance & norm_angle", va='bottom')

cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('norm_distance')

plt.tight_layout()
plt.savefig(f"{directory}/polar_plot.png")
plt.close()

prefixes = {
    'victim_descent_':   'Descent',
    'weapon_':           'Weapon',
    'premise_':          'Premise',
}

for prefix, label in prefixes.items():
    # 1) find all the one‑hot columns
    cols = [c for c in df.columns if c.startswith(prefix)]

    # 2) collapse into a single categorical column
    cat_col = prefix.rstrip('_')
    df[cat_col] = (
        df[cols]
        .idxmax(axis=1)                 # e.g. "weapon_12"
        .str.replace(prefix, '', regex=False)  # -> "12"
        .astype(str)                    # or str, if codes aren't numeric
    )

    # 3) compute percent distribution of crime_code across this categoryxs
    pct = (
        pd.crosstab(df['crime_code'], df[cat_col], normalize='index') 
        * 100
    ).round(2)

    # 4) plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pct,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={'label': '% of incidents'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f'Percentage of {label} by Crime Code (rows = 100%)')
    plt.xlabel(label)
    plt.ylabel('Crime Code')
    plt.tight_layout()

    # 5) save
    out_file = os.path.join(directory, f'crime_code_vs_{cat_col}_heatmap.png')
    plt.savefig(out_file)
    plt.close()


df['victim_sex_label'] = df['victim_sex'].map({0: 'Female', 1: 'Male'})
df['serious_crime_label'] = df['serious_crime'].map({0: 'Non‑Felony', 1: 'Felony'})

plt.figure(figsize=(6, 4))
sns.countplot(x='victim_sex_label', data=df)
plt.title('Distribution of Victim Sex')
plt.ylabel('Number of Incidents')
plt.xlabel('Victim Sex')
plt.tight_layout()
plt.savefig(os.path.join(directory, 'victim_sex_distribution.png'))
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(x='serious_crime_label', data=df)
plt.title('Distribution of Serious Crime (Felony vs Non‑Felony)')
plt.ylabel('Number of Incidents')
plt.xlabel('Crime Severity')
plt.tight_layout()
plt.savefig(os.path.join(directory, 'serious_crime_distribution.png'))
plt.close()



df['victim_sex_label'] = df['victim_sex'].map({0: 'Female', 1: 'Male'})

pct_table = (
    pd.crosstab(df['crime_code'], df['victim_sex_label'], normalize='index')
    * 100
).round(2)

crime_codes = pct_table.index.tolist()
female_pct = pct_table['Female'].values
male_pct = pct_table['Male'].values

x = np.arange(len(crime_codes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, female_pct, width, label='Female')
ax.bar(x + width/2, male_pct, width, label='Male')

ax.set_xticks(x)
ax.set_xticklabels(crime_codes)
ax.set_xlabel('Crime Code')
ax.set_ylabel('Percentage of incidents')
ax.set_title('Gender Distribution by Crime Code (pairs sum to 100%)')
ax.legend()

plt.tight_layout()

output_path = os.path.join(directory, 'crime_code_gender_distribution.png')
plt.savefig(output_path)
plt.close()