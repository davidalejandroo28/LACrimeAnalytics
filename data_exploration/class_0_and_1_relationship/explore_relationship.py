import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style.
sns.set(style="whitegrid")

# Define file path for cleaned data.
data_file = "./data/cleaned_crime_data_stratified.csv"
directory = os.path.dirname(os.path.abspath(__file__))

# Load the cleaned dataset.
df = pd.read_csv(data_file)
print("Dataset shape:", df.shape)

# Display overall summary.
print("\nOverall dataset summary:")
print(df.describe())

# --- Subset Data for Classes 0 and 1 ---
df_sub = df[df['crime_code'].isin([0, 1])]
print("\nSubset of data for crime codes 0 and 1:")
print(df_sub['crime_code'].value_counts())

# --- Summary Statistics for Numeric Features ---
numeric_features = ['victim_age', 'norm_distance', 'norm_angle']
for feature in numeric_features:
    print(f"\nSummary statistics for {feature}:")
    print(df_sub.groupby('crime_code')[feature].describe())

# --- Percentage Histograms of Numeric Features ---
for feature in numeric_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df_sub, x=feature, hue="crime_code", stat="percent",
        common_norm=False, multiple="layer", bins=30
    )
    plt.title(f"Percentage Histogram of {feature} for Crime Codes 0 vs 1")
    plt.xlabel(feature)
    plt.ylabel("Percentage")
    plt.tight_layout()
    plt.savefig(f"{directory}/{feature}_percentage_histogram_classes_0_1.png")
    plt.close()

# --- Boxplots of Numeric Features ---
for feature in numeric_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_sub, x="crime_code", y=feature)
    plt.title(f"Boxplot of {feature} by Crime Code (0 vs 1)")
    plt.xlabel("Crime Code")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(f"{directory}/{feature}_boxplot_classes_0_1.png")
    plt.close()

# --- Categorical One‑Hot Distributions ---
def save_onehot_dist(prefix):
    cols = [c for c in df_sub.columns if c.startswith(prefix)]
    if not cols:
        print(f"No {prefix} one-hot columns found.")
        return
    counts = df_sub.groupby('crime_code')[cols].sum().T
    totals = df_sub.groupby('crime_code').size()
    pct = counts.divide(totals, axis=1) * 100
    pct.plot(kind='bar', figsize=(10, 6))
    plt.title(f"Distribution of {prefix.rstrip('_')} across Crime Codes 0 and 1 (%)")
    plt.xlabel(prefix.rstrip('_'))
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{directory}/{prefix.rstrip('_')}_distribution_percent_classes_0_1.png")
    plt.close()

save_onehot_dist("victim_descent_")
save_onehot_dist("weapon_")
save_onehot_dist("premise_")

# --- Correlation of Quantile Summaries for Continuous Features ---
continuous_features = ['victim_age', 'norm_distance', 'norm_angle']
quantile_levels = np.linspace(0.1, 0.9, 9)

# Build quantile series for each class
quantiles = {cls: {} for cls in [0, 1]}
for cls in [0, 1]:
    dfc = df_sub[df_sub['crime_code'] == cls]
    for feature in continuous_features:
        quantiles[cls][feature] = dfc[feature].quantile(quantile_levels).values

# Compute correlation matrix
corr_matrix = pd.DataFrame(index=continuous_features,
                           columns=continuous_features, dtype=float)
for f1 in continuous_features:
    for f2 in continuous_features:
        corr = np.corrcoef(quantiles[0][f1], quantiles[1][f2])[0, 1]
        corr_matrix.loc[f1, f2] = corr

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix.astype(float), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Quantile Summaries of Class 0 and Class 1 Features")
plt.xlabel("Feature (Class 1)")
plt.ylabel("Feature (Class 0)")
plt.tight_layout()
plt.savefig(f"{directory}/correlation_between_class0_and_class1_continuous_features.png")
plt.close()
