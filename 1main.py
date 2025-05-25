# Drug Recommendation System - End to End

import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("drug_clean.csv")

# Step 2: Explore the dataset
print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 3: Data Cleaning (if needed)
# In this dataset, let's drop rows with missing critical info (if any)
df_clean = df.dropna(subset=['Condition', 'Drug', 'Satisfaction', 'EaseOfUse', 'Effective'])

# Step 4: Feature Engineering - Composite Score
# Weights can be adjusted based on importance of each metric
weight_satisfaction = 0.5
weight_ease_of_use = 0.3
weight_effective = 0.2

df_clean['CompositeScore'] = (
    df_clean['Satisfaction'] * weight_satisfaction +
    df_clean['EaseOfUse'] * weight_ease_of_use +
    df_clean['Effective'] * weight_effective
)

# Step 5: Define Recommender Function
def recommend_drugs(condition, top_n=5):
    """
    Recommend top N drugs for a given medical condition based on composite score.

    Parameters:
    - condition (str): Medical condition to filter drugs.
    - top_n (int): Number of top recommendations to return.

    Returns:
    - pandas.DataFrame: Top N recommended drugs sorted by CompositeScore.
    """
    # Filter dataset by condition (case insensitive match)
    condition = condition.lower()
    filtered_df = df_clean[df_clean['Condition'].str.lower() == condition]

    if filtered_df.empty:
        return f"No drugs found for condition: {condition}"

    # Sort by CompositeScore descending
    recommendations = filtered_df.sort_values(by='CompositeScore', ascending=False)

    # Select relevant columns to display
    return recommendations[['Drug', 'CompositeScore', 'Satisfaction', 'EaseOfUse', 'Effective', 'Price']].head(top_n)

# Step 6: Example Usage
if __name__ == "__main__":
    user_condition = "Acute Bacterial Sinusitis"
    print(f"\nTop drug recommendations for condition: {user_condition}\n")
    print(recommend_drugs(user_condition, top_n=5))
