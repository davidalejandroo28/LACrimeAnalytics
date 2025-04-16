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

# --- Subset Data for Classes 1 and 2 ---
# (Assume that 1 and 2 are the crime_code values in question.)
df_sub = df[df['crime_code'].isin([1, 2])]
print("\nSubset of data for crime codes 1 and 2:")
print(df_sub['crime_code'].value_counts())

# --- Summary Statistics for Numeric Features ---
numeric_features = ['victim_age', 'norm_distance', 'norm_angle']
for feature in numeric_features:
    print(f"\nSummary statistics for {feature}:")
    print(df_sub.groupby('crime_code')[feature].describe())

# --- Percentage Histograms of Numeric Features ---
# Instead of plotting frequency, we plot the percentage each bin represents within its crime_code.
for feature in numeric_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_sub, x=feature, hue="crime_code", stat="percent",
                 common_norm=False, multiple="dodge", bins=30)
    plt.title(f"Percentage Histogram of {feature} for Crime Codes 1 vs 2")
    plt.xlabel(feature)
    plt.ylabel("Percentage")
    plt.tight_layout()
    plt.savefig(f"{directory}/{feature}_percentage_histogram_classes_1_2.png")
    plt.close()

# --- Boxplots of Numeric Features ---
for feature in numeric_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_sub, x="crime_code", y=feature)
    plt.title(f"Boxplot of {feature} by Crime Code")
    plt.xlabel("Crime Code")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(f"{directory}/{feature}_boxplot_classes_1_2.png")
    plt.close()

victim_descent_cols = [col for col in df_sub.columns if col.startswith("victim_descent_")]
if victim_descent_cols:
    # Group by crime code and sum the one-hot columns.
    descent_counts = df_sub.groupby('crime_code')[victim_descent_cols].sum().T
    # Get total number of rows per crime_code.
    crime_totals = df_sub.groupby('crime_code').size()
    # Compute percentage per class.
    descent_percent = descent_counts.divide(crime_totals, axis=1) * 100
    descent_percent.plot(kind='bar', figsize=(10, 6))
    plt.title("Distribution of victim_descent (one-hot) across Crime Codes 1 and 2 (%)")
    plt.xlabel("Victim Descent Code")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{directory}/victim_descent_distribution_percent_classes_1_2.png")
    plt.close()
else:
    print("No victim_descent one-hot columns found.")

# Similarly, compute percentages for weapon columns.
weapon_cols = [col for col in df_sub.columns if col.startswith("weapon_")]
if weapon_cols:
    weapon_counts = df_sub.groupby('crime_code')[weapon_cols].sum().T
    crime_totals = df_sub.groupby('crime_code').size()
    weapon_percent = weapon_counts.divide(crime_totals, axis=1) * 100
    weapon_percent.plot(kind='bar', figsize=(10, 6))
    plt.title("Distribution of weapon codes (one-hot) across Crime Codes 1 and 2 (%)")
    plt.xlabel("Weapon Code")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{directory}/weapon_distribution_percent_classes_1_2.png")
    plt.close()
else:
    print("No weapon one-hot columns found.")


# Define the continuous features to compare.
continuous_features = ['victim_age', 'norm_distance', 'norm_angle']

# Define a set of quantile levels to summarize each feature.
quantile_levels = np.linspace(0.1, 0.9, 9)  # 9 decile values

# Compute quantile summaries for each continuous feature for class 1 and class 2.
quantiles_class1 = {}
quantiles_class2 = {}

for feature in continuous_features:
    quantiles_class1[feature] = df_sub[df_sub['crime_code'] == 1][feature].quantile(quantile_levels).values
    quantiles_class2[feature] = df_sub[df_sub['crime_code'] == 2][feature].quantile(quantile_levels).values

# Create a correlation matrix where:
#   - rows correspond to continuous features computed on class 1,
#   - columns correspond to continuous features computed on class 2.
# Each entry is the Pearson correlation between the quantile series for the row feature (from class 1)
# and the column feature (from class 2).
corr_matrix = pd.DataFrame(index=continuous_features, columns=continuous_features, dtype=float)

for feat1 in continuous_features:
    for feat2 in continuous_features:
        # Compute Pearson correlation between the two quantile series.
        # These series have length = len(quantile_levels)
        corr_coef = np.corrcoef(quantiles_class1[feat1], quantiles_class2[feat2])[0, 1]
        corr_matrix.loc[feat1, feat2] = corr_coef

# Plot the resulting correlation matrix as a heatmap.
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix.astype(float), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Quantile Summaries of Class 1 and Class 2 Features")
plt.xlabel("Feature (Class 2)")
plt.ylabel("Feature (Class 1)")
plt.tight_layout()
plt.savefig(f"{directory}/correlation_between_class1_and_class2_continuous_features.png")
plt.close()
