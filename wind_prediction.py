import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
n_samples = 10000
data = {}

# Features
data['wind_speed'] = np.random.normal(10, 3, n_samples)  # Mean 10 m/s, std 3
data['temperature'] = np.random.normal(15, 5, n_samples)  # Mean 15°C, std 5
data['humidity'] = np.random.uniform(30, 80, n_samples)  # 30-80%
data['vibration'] = np.random.exponential(0.5, n_samples)  # Exponential for positive skew
data['rotor_speed'] = np.random.normal(1500, 200, n_samples)  # RPM
data['power_output'] = np.random.normal(2000, 500, n_samples)  # kW
data['turbine_age'] = np.random.uniform(1, 20, n_samples)  # Years

# Create target: Failure (1) or No Failure (0)
# Simulate correlations: Higher failure risk with extreme values
failure_prob = (
    0.1 +  # Base probability
    0.05 * (data['wind_speed'] > 15).astype(int) +  # High wind
    0.05 * (data['temperature'] < 0).astype(int) +  # Low temp
    0.1 * (data['vibration'] > 1.5).astype(int) +    # High vibration
    0.05 * (data['turbine_age'] > 15).astype(int)   # Old turbine
)
data['failure'] = np.random.binomial(1, failure_prob, n_samples)

# Create DataFrame
df = pd.DataFrame(data)

# Balance the dataset (roughly 50/50 for simplicity)
failure_count = df['failure'].sum()
no_failure_count = len(df) - failure_count
print(f"Dataset: {failure_count} failures, {no_failure_count} no failures")

# Split into features and target
X = df.drop('failure', axis=1)
y = df['failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance (without plotting)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print("Feature Importances:")
print(feature_importances.sort_values(ascending=False))

# Save dataset to CSV
df.to_csv('wind_turbine_synthetic_dataset.csv', index=False)

# Save the model
joblib.dump(model, 'wind_turbine_model.pkl')
print("Model saved as 'wind_turbine_model.pkl'")
print("Dataset saved as 'wind_turbine_synthetic_dataset.csv'")

# Sample of the dataset (first 10 rows)
print("\nSample Dataset (first 10 rows):")
print(df.head(10))

