import nbformat as nbf

nb_path = 'notebooks/feature_engineering.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

new_cell_source = """
# 1. Create a deployable Pipeline preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

X_raw = df[categorical_features + numerical_features]
X_encoded = preprocessor.fit_transform(X_raw)

os.makedirs('../model', exist_ok=True)
with open('../model/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("Saved preprocessor to: model/preprocessor.pkl")

# 2. Create the explicit ML-ready dataset CSV for direct model training/inspection
# Get feature names if possible
try:
    encoded_columns = preprocessor.get_feature_names_out()
except AttributeError:
    # For older scikit-learn
    encoded_columns = [f"f_{i}" for i in range(X_encoded.shape[1])]

df_encoded = pd.DataFrame(X_encoded, columns=encoded_columns)
df_encoded[target] = df[target].values

os.makedirs('../data/processed', exist_ok=True)
output_path = '../data/processed/ml_ready_mumbai_indore_data.csv'
df_encoded.to_csv(output_path, index=False)
print(f"Saved ml-ready dataset to: {output_path}")
"""

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "preprocessor = ColumnTransformer(" in cell.source:
            cell.source = new_cell_source

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
