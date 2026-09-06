import nbformat as nbf

nb_path = 'notebooks/feature_engineering.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # update input file
        if 'geocoded_property_data.csv' in cell.source:
            cell.source = cell.source.replace('geocoded_property_data.csv', 'geocoded_mumbai_indore_property_data.csv')
            
        # update categorical columns to include 'City'
        if "categorical_features = ['Location', 'Property_Type']" in cell.source:
            cell.source = cell.source.replace("categorical_features = ['Location', 'Property_Type']", "categorical_features = ['City', 'Location', 'Property_Type']")

        # drop price_per_sqft if it exists in the drop logic
        if "X = df.drop" in cell.source and "price_per_sqft" not in cell.source:
            # We will handle price_per_sqft specifically if it's there, but actually 
            # I can just do a general drop for price_per_sqft if it exists
            pass

        # update output file
        if 'ml_ready_data.csv' in cell.source:
            cell.source = cell.source.replace('ml_ready_data.csv', 'ml_ready_mumbai_indore_data.csv')

# add explicit logic to ensure price_per_sqft is dropped
drop_code = """
if 'price_per_sqft' in df.columns:
    df.drop('price_per_sqft', axis=1, inplace=True)
    print("Dropped price_per_sqft to prevent data leakage")
"""
nb.cells.insert(3, nbf.v4.new_code_cell(drop_code))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

