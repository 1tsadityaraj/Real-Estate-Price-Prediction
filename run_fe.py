import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data/processed/geocoded_property_data.csv')

# 1. Target and Feature identification
target = 'Price_INR'
features_removed = ['price_per_sqft'] # Target leakage
df = df.drop(columns=features_removed)

# Drop any remaining NaNs (we handled them before, but double check)
df = df.dropna()

# 2. Outlier handling
initial_rows = len(df)
# Remove properties with zero or negative price/area
df = df[(df['Price_INR'] > 0) & (df['Area_sqft'] > 0)]
# Additional reasonable bounds
df = df[(df['BHK'] > 0) & (df['Bathrooms'] > 0)]
rows_removed = initial_rows - len(df)

# 3. Categorical encoding
categorical_features = ['Location', 'Property_Type']
numerical_features = ['Area_sqft', 'BHK', 'Bathrooms', 'Latitude', 'Longitude']

# Creating a ColumnTransformer for the pipeline (saved for app.py)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Fit the preprocessor
X_raw = df[categorical_features + numerical_features]
y = df[target]
preprocessor.fit(X_raw)

# Save the preprocessor
os.makedirs('model', exist_ok=True)
with open('model/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# For ml_ready_data.csv, we'll use pandas get_dummies for readability in the CSV
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# 4. Save ml_ready_data
os.makedirs('data/processed', exist_ok=True)
df_encoded.to_csv('data/processed/ml_ready_data.csv', index=False)

# 5. Train-Test Split
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

summary = {
    "selected_features": categorical_features + numerical_features,
    "removed_features": features_removed,
    "encoding_method": "One-Hot Encoding",
    "outlier_strategy": "Removed negative/zero values and strictly enforced domain bounds",
    "train_test_ratio": "80/20",
    "final_dataset_shape": df_encoded.shape,
    "feature_count": X.shape[1],
    "rows_before_outlier": initial_rows,
    "rows_removed": rows_removed,
    "rows_remaining": len(df),
    "x_train_shape": X_train.shape,
    "x_test_shape": X_test.shape
}

with open('fe_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("Feature Engineering completed successfully.")
