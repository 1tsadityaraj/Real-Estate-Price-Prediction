import pandas as pd
import numpy as np
import json

df = pd.read_csv('data/processed/cleaned_property_data.csv')

summary = {
    'shape': df.shape,
    'columns': list(df.columns),
    'num_locations': df['Location'].nunique(),
    'top_location': df['Location'].value_counts().index[0],
    'top_location_count': int(df['Location'].value_counts().iloc[0]),
    'min_price': df['Price_INR'].min(),
    'max_price': df['Price_INR'].max(),
    'mean_price': df['Price_INR'].mean(),
    'median_price': df['Price_INR'].median(),
    'min_area': df['Area_sqft'].min(),
    'max_area': df['Area_sqft'].max(),
    'mean_area': df['Area_sqft'].mean(),
    'bhk_most_common': int(df['BHK'].mode()[0]),
    'property_type_common': df['Property_Type'].mode()[0]
}

# Correlation
numeric_cols = df.select_dtypes(include=[np.number])
corr = numeric_cols.corr()
summary['correlation_price_area'] = corr.loc['Price_INR', 'Area_sqft']
summary['correlation_price_bhk'] = corr.loc['Price_INR', 'BHK']
summary['correlation_price_bathrooms'] = corr.loc['Price_INR', 'Bathrooms']

# Groupby location
loc_stats = df.groupby('Location').agg({
    'Price_INR': 'mean',
    'price_per_sqft': 'mean',
    'Location': 'count'
}).rename(columns={'Location': 'Count'}).sort_values('Price_INR', ascending=False)
summary['most_expensive_loc'] = loc_stats.index[0]
summary['most_expensive_loc_price'] = loc_stats.iloc[0]['Price_INR']
summary['highest_price_sqft_loc'] = loc_stats.sort_values('price_per_sqft', ascending=False).index[0]

with open('eda_summary.json', 'w') as f:
    json.dump(summary, f)

print("Summary generated")
