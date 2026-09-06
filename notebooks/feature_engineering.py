#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ## 1. Load Geocoded Data

# In[2]:


df = pd.read_csv('../data/processed/geocoded_mumbai_indore_property_data.csv')

print(f"Initial dataset shape: {df.shape}")
print(f"Columns:\n{list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

print(df.head())


# In[3]:


if 'price_per_sqft' in df.columns:
    df = df.drop('price_per_sqft', axis=1, errors='ignore')
    print("Dropped price_per_sqft to prevent data leakage")


# ## 2. Define Features and Handle Missing Values via Pipeline

# In[4]:


target = 'Price_INR'

# Define final features
categorical_features = ['City', 'Location', 'Property_Type']
numerical_features = ['Area_sqft', 'BHK', 'Bathrooms', 'Latitude', 'Longitude']
print(f"Final Categorical Features: {categorical_features}")
print(f"Final Numerical Features: {numerical_features}")


# ## 3. Handle Domain Outliers

# In[5]:


initial_rows = len(df)

# Domain-based outlier/invalid value removal
df = df[(df['Price_INR'] > 0) & (df['Area_sqft'] > 0)]
df = df[(df['BHK'] > 0) & (df['Bathrooms'] > 0)]

rows_removed = initial_rows - len(df)
print(f"Rows before outlier handling: {initial_rows}")
print(f"Rows removed (domain checks): {rows_removed}")
print(f"Rows remaining: {len(df)}")


# ## 4. Construct Scikit-Learn Pipeline and Save Preprocessor

# In[6]:


# Define unified preprocessing steps
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='passthrough'
)

X_raw = df[categorical_features + numerical_features]

# We must ensure X_raw order is correctly fed: [categorical_features] + [numerical_features]
# The preprocessor output will be num_features_transformed followed by cat_features_transformed.
X_encoded = preprocessor.fit_transform(X_raw)

os.makedirs('../model', exist_ok=True)
with open('../model/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("Saved preprocessor to: model/preprocessor.pkl")

# Create the explicit ML-ready dataset CSV for direct model training/inspection
try:
    encoded_columns = preprocessor.get_feature_names_out()
except AttributeError:
    encoded_columns = [f"f_{i}" for i in range(X_encoded.shape[1])]

df_encoded = pd.DataFrame(X_encoded, columns=encoded_columns)
df_encoded[target] = df[target].values

os.makedirs('../data/processed', exist_ok=True)
output_path = '../data/processed/ml_ready_mumbai_indore_data.csv'
df_encoded.to_csv(output_path, index=False)
print(f"Saved ml-ready dataset to: {output_path}")


# ## 5. Train-Test Split

# In[7]:


X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# In[8]:


print("--- Final Validation ---")
print(f"Final feature count: {X.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Missing values in X_train: {X_train.isnull().sum().sum()}")

