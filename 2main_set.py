import pandas as pd

# Load your dataset
df_clean = pd.read_csv('drug_clean.csv')

# Calculate Composite Score (example weights)
df_clean['CompositeScore'] = (
    0.5 * df_clean['Satisfaction'] +
    0.3 * df_clean['EaseOfUse'] +
    0.2 * df_clean['Effective']
)

def recommend_drugs(condition, top_n=5):
    """
    Recommend top N drugs for a given medical condition based on composite score.
    Highlights most effective drug and shows usage details.

    Parameters:
    - condition (str): Medical condition to filter drugs.
    - top_n (int): Number of top recommendations to return.

    Returns:
    - pandas.DataFrame or str: Top N recommended drugs with details or message if none found.
    """
    condition_lower = condition.lower()
    filtered_df = df_clean[df_clean['Condition'].str.lower() == condition_lower]

    if filtered_df.empty:
        return f"No drugs found for condition: {condition}"

    # Sort by composite score descending
    recommendations = filtered_df.sort_values(by='CompositeScore', ascending=False)

    # Identify the drug(s) with the highest effectiveness score
    max_effective = recommendations['Effective'].max()
    recommendations['MostEffective'] = recommendations['Effective'] == max_effective

    # Select relevant columns for output
    cols = [
        'Drug', 'Condition', 'Indication', 'CompositeScore',
        'Satisfaction', 'EaseOfUse', 'Effective', 'MostEffective', 'Price'
    ]
    
    return recommendations[cols].head(top_n)


# Example usage
condition_to_query = 'Acute Bacterial Sinusitis'
top_drugs = recommend_drugs(condition_to_query, top_n=5)

print(f"Top {len(top_drugs)} drug recommendations for condition: {condition_to_query}\n")
print(top_drugs.to_string(index=False))
