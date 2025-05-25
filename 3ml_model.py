import pandas as pd

# Load your dataset
df_clean = pd.read_csv('drug_clean.csv')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Prepare features and target
features = df_clean[['Satisfaction', 'EaseOfUse', 'Price']]
target = df_clean['Effective']
features = features.fillna(features.mean())
drug_names = df_clean['Drug']

# Split data
X_train, X_test, y_train, y_test, drugs_train, drugs_test = train_test_split(
    features, target, drug_names, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Scatter plot: Actual vs Predicted Effectiveness
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Effectiveness')
plt.ylabel('Predicted Effectiveness')
plt.title('Actual vs Predicted Drug Effectiveness')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.show()

# Feature importance plot
importances = model.feature_importances_
feature_names = features.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances for Effectiveness Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# # Display few predictions with drug names
# print("\nSample Drug effectiveness predictions:")
# for drug, actual, pred in zip(drugs_test[:10], y_test[:10], y_pred[:10]):
#     print(f"Drug: {drug}, Actual: {actual:.2f}, Predicted: {pred:.2f}")

# # print("\nSample Drug Effectiveness Predictions:")

# # Loop through first 10 test samples
# for i in range(10):
#     drug = drugs_test.iloc[i]       # Get the drug name at position i
#     actual = y_test.iloc[i]          # Get the actual effectiveness value at position i
#     predicted = y_pred[i]            # Get the predicted effectiveness value at position i

#     # Print the drug name and the actual vs predicted values nicely formatted
#     print("Drug:", drug)
#     print("  Actual Effectiveness: {:.2f}".format(actual))
#     print("  Predicted Effectiveness: {:.2f}".format(predicted))
#     print()  # Blank line for readability


#To print All Drug Effectiveness Predictions
print("\nAll Drug Effectiveness Predictions:")

for i in range(len(drugs_test)):
    drug = drugs_test.iloc[i]
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    print(f"Drug: {drug}, Actual: {actual:.2f}, Predicted: {predicted:.2f}")