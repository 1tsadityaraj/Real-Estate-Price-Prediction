import pandas as pd
import numpy as np

# 1. Load and inspect the dataset
print("--- 1. Load and Inspect ---")
df = pd.read_csv('data/raw/property_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Column names: {list(df.columns)}")
print(f"\\nData types:\\n{df.dtypes}")
print(f"\\nFirst 5 rows:\\n{df.head()}")
print(f"\\nMissing values count:\\n{df.isnull().sum()}")
print(f"\\nDuplicate count: {df.duplicated().sum()}")
print(f"\\nBasic statistical summary:\\n{df.describe(include='all')}")

# 2. Handle duplicate records
print("\\n--- 2. Handle Duplicate Records ---")
before_dup = len(df)
df = df.drop_duplicates()
after_dup = len(df)
print(f"Before duplicates: {before_dup}")
print(f"After duplicates: {after_dup}")
print(f"Duplicates removed: {before_dup - after_dup}")

# 3. Handle missing values
print("\\n--- 3. Handle Missing Values ---")
# Area_sqft missing values: We can fill with median of similar property types and BHK
df['Area_sqft'] = df.groupby(['Property_Type', 'BHK'])['Area_sqft'].transform(lambda x: x.fillna(x.median()))
# For remaining, use overall median
df['Area_sqft'] = df['Area_sqft'].fillna(df['Area_sqft'].median())

# Bathrooms missing values: Fill with median bathrooms for that BHK
df['Bathrooms'] = df.groupby('BHK')['Bathrooms'].transform(lambda x: x.fillna(x.median()))
df['Bathrooms'] = df['Bathrooms'].fillna(df['BHK'] + 1) # Fallback

print("Missing values handled using group medians (Area based on Property_Type+BHK, Bathrooms based on BHK).")

# 4. Clean numerical columns
print("\\n--- 4. Clean Numerical Columns ---")
# Defensive cleaning for Price_INR and Area_sqft assuming they could be strings
if df['Price_INR'].dtype == object:
    df['Price_INR'] = df['Price_INR'].astype(str).str.replace(',', '').str.extract('(\d+\.?\d*)').astype(float)
if df['Area_sqft'].dtype == object:
    df['Area_sqft'] = df['Area_sqft'].astype(str).str.replace(',', '').str.extract('(\d+\.?\d*)').astype(float)

# 5. Clean BHK/bedroom information
print("\\n--- 5. Clean BHK ---")
if df['BHK'].dtype == object:
    df['BHK'] = df['BHK'].astype(str).str.extract('(\d+)').astype(float)

# 6. Clean location data
print("\\n--- 6. Clean Location Data ---")
df['Location'] = df['Location'].astype(str).str.strip().str.title()

# 7. Clean property-type information
print("\\n--- 7. Clean Property Type ---")
df['Property_Type'] = df['Property_Type'].astype(str).str.strip().str.title()

# 8. Outlier analysis & Handling
print("\\n--- 8. Outlier Analysis ---")
# Let's remove properties with Area < 100 sqft or Bathrooms > 10 (unrealistic)
# Also remove properties where Price is extremely high or low
initial_len = len(df)
df = df[df['Area_sqft'] >= 100]
df = df[df['Bathrooms'] <= 10]
# Remove price per sqft outliers (outside 1st-99th percentile)
temp_price_sqft = df['Price_INR'] / df['Area_sqft']
lower_bound = temp_price_sqft.quantile(0.01)
upper_bound = temp_price_sqft.quantile(0.99)
df = df[(temp_price_sqft >= lower_bound) & (temp_price_sqft <= upper_bound)]
print(f"Removed {initial_len - len(df)} rows due to outlier thresholds (Area < 100, Bathrooms > 10, or extreme price/sqft).")

# 9. Feature creation
print("\\n--- 9. Feature Creation ---")
df['price_per_sqft'] = (df['Price_INR'] / df['Area_sqft']).round(2)
print("Created 'price_per_sqft' feature.")

# 10. Remove irrelevant columns
print("\\n--- 10. Remove Irrelevant Columns ---")
# 'Listing_URL' and 'Property_Name' are not useful for ML
cols_to_drop = ['Listing_URL', 'Property_Name']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
print(f"Dropped columns: {cols_to_drop} (Not useful for predictive modeling).")

# 11. Final dataset validation
print("\\n--- 11. Final Dataset Validation ---")
print(f"Final dataset shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")
print(f"Final data types:\\n{df.dtypes}")
print(f"Remaining missing values:\\n{df.isnull().sum()}")

# 12. Save the cleaned dataset
import os
os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/cleaned_property_data.csv'
df.to_csv(output_path, index=False)
print(f"\\nSaved cleaned dataset to: {output_path}")
