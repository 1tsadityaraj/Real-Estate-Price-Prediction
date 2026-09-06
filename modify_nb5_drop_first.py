import nbformat as nbf

nb_path = 'notebooks/feature_engineering.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "drop_first=True" in cell.source:
            cell.source = cell.source.replace("drop_first=True", "drop_first=False")

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
