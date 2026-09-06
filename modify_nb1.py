import nbformat as nbf
import os

nb_path = 'notebooks/web_scraping.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Find the cell that saves data/raw/property_data.csv and modify it to add City column
# and merge both datasets

new_code = """import pandas as pd
import os

# Load Mumbai Data
mumbai_df = pd.read_csv('../data/raw/property_data.csv')
mumbai_df['City'] = 'Mumbai'

# Load Indore Data (Simulated offline source)
indore_df = pd.read_csv('../data/raw/indore_property_data.csv')
indore_df['City'] = 'Indore'

# Standardize column names and combine
# Both generated datasets share the same schema: 
# ['Property_Name', 'Location', 'Property_Type', 'Price_INR', 'Area_sqft', 'BHK', 'Bathrooms', 'Listing_URL']
combined_df = pd.concat([mumbai_df, indore_df], ignore_index=True)

# Save combined dataset
os.makedirs('../data/raw', exist_ok=True)
combined_df.to_csv('../data/raw/mumbai_indore_property_data.csv', index=False)

print(f"Combined dataset saved with shape: {combined_df.shape}")
print(combined_df['City'].value_counts())
"""

nb.cells.append(nbf.v4.new_markdown_cell("## Combining Mumbai and Indore Datasets\nSince the data is collected independently, we now combine the raw offline data into a standardized combined dataset."))
nb.cells.append(nbf.v4.new_code_cell(new_code))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

