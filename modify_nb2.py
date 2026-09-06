import nbformat as nbf

nb_path = 'notebooks/data_cleaning.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# We want to find the cell reading property_data.csv and change it to mumbai_indore_property_data.csv
for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'property_data.csv' in cell.source and 'read_csv' in cell.source:
            cell.source = cell.source.replace('property_data.csv', 'mumbai_indore_property_data.csv')
        # update output path
        if 'cleaned_property_data.csv' in cell.source:
            cell.source = cell.source.replace('cleaned_property_data.csv', 'cleaned_mumbai_indore_property_data.csv')

# Add validation check
nb.cells.append(nbf.v4.new_markdown_cell("## City Validation Check"))
validation_code = """
print(f"Total Records: {df.shape[0]}")
print("City Distribution:")
print(df['City'].value_counts())
print("\nUnique Locations:")
print(df['Location'].nunique())
"""
nb.cells.append(nbf.v4.new_code_cell(validation_code))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

