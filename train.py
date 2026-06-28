from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import joblib

# Load dataset
data = fetch_california_housing()

# Create feature DataFrame
X = pd.DataFrame(data.data, columns=data.feature_names)

# Target variable
y = data.target

print("Total records:", X.shape[0])
print("Total features:", X.shape[1])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=10,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Average Error: ${mae * 100000:,.0f}")
print(f"R² Score: {r2:.4f}")

# Save model and feature names
joblib.dump(model, "house_model.joblib")
joblib.dump(list(X.columns), "house_features.joblib")

print("Model saved successfully!")