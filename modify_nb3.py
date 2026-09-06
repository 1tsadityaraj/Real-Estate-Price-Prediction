import nbformat as nbf
import json

nb_path = 'notebooks/EDA.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Change dataset
        if 'cleaned_property_data.csv' in cell.source:
            cell.source = cell.source.replace('cleaned_property_data.csv', 'cleaned_mumbai_indore_property_data.csv')
        
        # Add hue='City' to sns plots if possible
        if 'sns.histplot' in cell.source and 'hue' not in cell.source:
            cell.source = cell.source.replace('sns.histplot(', "sns.histplot(hue='City', ")
        
        if 'sns.boxplot' in cell.source and 'hue' not in cell.source:
            cell.source = cell.source.replace('sns.boxplot(', "sns.boxplot(hue='City', ")

        if 'sns.scatterplot' in cell.source and 'hue' not in cell.source:
            cell.source = cell.source.replace('sns.scatterplot(', "sns.scatterplot(hue='City', ")

        if 'sns.countplot' in cell.source and 'hue' not in cell.source:
            cell.source = cell.source.replace('sns.countplot(', "sns.countplot(hue='City', ")

# Add a specific city comparison section
comparison_md = "## City Comparison Analysis\nLet's analyze the difference in median prices and area between Mumbai and Indore."
comparison_code = """
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='City', y='Price_INR', estimator='median', ci=None)
plt.title('Median Property Price: Mumbai vs Indore')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='City', y='Area_sqft')
plt.title('Property Area Distribution: Mumbai vs Indore')
plt.show()

print(df.groupby('City')[['Price_INR', 'Area_sqft']].median())
"""
nb.cells.append(nbf.v4.new_markdown_cell(comparison_md))
nb.cells.append(nbf.v4.new_code_cell(comparison_code))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

