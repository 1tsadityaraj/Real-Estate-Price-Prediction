import nbformat as nbf
import json

nb_path = 'notebooks/EDA.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Fix the hue injections
        cell.source = cell.source.replace("sns.histplot(hue='City', df['Price_INR']", "sns.histplot(data=df, x='Price_INR', hue='City'")
        cell.source = cell.source.replace("sns.histplot(hue='City', df['Area_sqft']", "sns.histplot(data=df, x='Area_sqft', hue='City'")
        cell.source = cell.source.replace("sns.boxplot(hue='City', data=df", "sns.boxplot(data=df, hue='City'")
        cell.source = cell.source.replace("sns.scatterplot(hue='City', data=df", "sns.scatterplot(data=df, hue='City'")
        cell.source = cell.source.replace("sns.countplot(hue='City', data=df", "sns.countplot(data=df, hue='City'")
        
        # fix the location vs price plot (which is probably a barplot or boxplot)
        if "hue='City'" in cell.source and "y='Location'" in cell.source:
            pass # we want location comparison but keeping it simple

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

