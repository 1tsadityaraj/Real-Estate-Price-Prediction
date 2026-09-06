import nbformat as nbf

nb_path = 'notebooks/feature_engineering.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "df = df.drop(columns=features_removed)" in cell.source:
            cell.source = cell.source.replace("df = df.drop(columns=features_removed)", "df = df.drop(columns=features_removed, errors='ignore')")

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
