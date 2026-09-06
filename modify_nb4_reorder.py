import nbformat as nbf

nb_path = 'notebooks/geocoders_maps.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Find the df loading cell and move it to the first code cell
df_load_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'pd.read_csv' in cell.source:
        df_load_idx = i
        break

if df_load_idx is not None:
    # also we need the import pandas cell which might be cell 0
    import_idx = None
    for i, cell in enumerate(nb.cells):
         if cell.cell_type == 'code' and 'import pandas as pd' in cell.source:
             import_idx = i
             break

    # Reconstruct cells
    new_cells = []
    # put markdown titles
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and '##' in cell.source:
            pass # just to keep it simple, let's just make sure imports and df reading happen first.

# The easiest way is to just inject an import and df reading cell at the very beginning.
new_import_cell = nbf.v4.new_code_cell("import pandas as pd\ndf = pd.read_csv('../data/processed/cleaned_mumbai_indore_property_data.csv')")
nb.cells.insert(0, new_import_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

