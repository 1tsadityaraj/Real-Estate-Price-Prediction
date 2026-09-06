import pandas as pd
import numpy as np

# Setting a random seed for reproducibility
np.random.seed(42)

# Simulating property data for Mumbai
num_properties = 200

# Lists of sample data for Mumbai
locations = ['Andheri West', 'Bandra East', 'Juhu', 'Powai', 'Borivali West', 'Goregaon East', 'Malad West', 'Kandivali East', 'Worli', 'Lower Parel']
property_types = ['Apartment', 'Villa', 'Independent House', 'Studio']
bhk_options = [1, 2, 3, 4, 5]

data = {
    'Property_Name': [f"Luxury {np.random.choice(bhk_options)} BHK in {np.random.choice(locations)}" for _ in range(num_properties)],
    'Location': np.random.choice(locations, num_properties),
    'Property_Type': np.random.choice(property_types, num_properties, p=[0.8, 0.05, 0.05, 0.1]),
    'Price_INR': np.random.uniform(50_00_000, 5_00_00_000, num_properties).round(-5), # Prices between 50L and 5Cr
    'Area_sqft': np.random.uniform(400, 3500, num_properties).round(0),
    'BHK': np.random.choice(bhk_options, num_properties, p=[0.2, 0.4, 0.3, 0.08, 0.02]),
    'Bathrooms': [],
    'Listing_URL': [f"https://www.sample-realestate.com/property/{i}" for i in range(num_properties)]
}

# Assign bathrooms based on BHK (usually 1 or 2 more/less than BHK)
for bhk in data['BHK']:
    data['Bathrooms'].append(max(1, bhk + np.random.choice([-1, 0, 1], p=[0.1, 0.7, 0.2])))

df = pd.DataFrame(data)

# Introduce some missing values to simulate raw scraped data
df.loc[np.random.choice(df.index, 10), 'Area_sqft'] = np.nan
df.loc[np.random.choice(df.index, 5), 'Bathrooms'] = np.nan

# Save to CSV
df.to_csv('data/raw/property_data.csv', index=False)
print(f"Dataset generated with shape {df.shape}")
print(f"Columns: {list(df.columns)}")
