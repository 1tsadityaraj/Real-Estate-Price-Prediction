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

df = df.dropna()

initial_rows = len(df)
df = df[(df['Price_INR'] > 0) & (df['Area_sqft'] > 0)]
df = df[(df['BHK'] > 0) & (df['Bathrooms'] > 0)]
rows_removed = initial_rows - len(df)

categorical_features = ['Location', 'Property_Type']
numerical_features = ['Area_sqft', 'BHK', 'Bathrooms', 'Latitude', 'Longitude']

# Drop first to match get_dummies behaviour and avoid collinearity
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

X_raw = df[categorical_features + numerical_features]
y = df[target]

# Fit and transform
X_encoded = preprocessor.fit_transform(X_raw)

# Get feature names if possible (scikit-learn >= 1.0)
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    # fallback
    feature_names = [f'feature_{i}' for i in range(X_encoded.shape[1])]
    
df_encoded = pd.DataFrame(X_encoded, columns=feature_names)
df_encoded[target] = y.values

os.makedirs('model', exist_ok=True)
with open('model/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

os.makedirs('data/processed', exist_ok=True)
df_encoded.to_csv('data/processed/ml_ready_data.csv', index=False)

# Train Test Split
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
    "feature_count": X.shape[1]
}

with open('fe_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("Feature Engineering completed successfully.")
