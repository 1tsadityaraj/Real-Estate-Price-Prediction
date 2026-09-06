import nbformat as nbf

nb_path = 'notebooks/feature_engineering.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "X_encoded = preprocessor.fit_transform(X_raw)" in cell.source:
            cell.source = cell.source.replace("X_encoded = preprocessor.fit_transform(X_raw)", "X_encoded = preprocessor.fit_transform(X_raw)\nif hasattr(X_encoded, 'toarray'):\n    X_encoded = X_encoded.toarray()")

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
