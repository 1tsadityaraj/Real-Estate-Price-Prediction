import nbformat as nbf

nb_path = 'notebooks/feature_engineering.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Delete the bad cell
bad_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "if 'price_per_sqft' in df.columns:" in cell.source:
        bad_idx = i
        break

if bad_idx is not None:
    del nb.cells[bad_idx]

# Insert after read_csv
insert_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "pd.read_csv" in cell.source:
        insert_idx = i
        break

if insert_idx is not None:
    drop_code = """
if 'price_per_sqft' in df.columns:
    df.drop('price_per_sqft', axis=1, inplace=True)
    print("Dropped price_per_sqft to prevent data leakage")
"""
    nb.cells.insert(insert_idx + 1, nbf.v4.new_code_cell(drop_code))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

