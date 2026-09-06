import nbformat as nbf

nb_path = 'notebooks/model_building.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'ml_ready_data.csv' in cell.source:
            cell.source = cell.source.replace('ml_ready_data.csv', 'ml_ready_mumbai_indore_data.csv')

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

