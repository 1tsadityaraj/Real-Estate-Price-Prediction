import nbformat as nbf

nb_path = 'notebooks/data_cleaning.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

nb.cells[-1].source = """
print(f"Total Records: {df.shape[0]}")
print("City Distribution:")
print(df['City'].value_counts())
print("\\nUnique Locations:")
print(df['Location'].nunique())
"""

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

