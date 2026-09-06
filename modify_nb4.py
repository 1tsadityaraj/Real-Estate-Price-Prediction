import nbformat as nbf
import json

nb_path = 'notebooks/geocoders_maps.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # update input file
        if 'cleaned_property_data.csv' in cell.source:
            cell.source = cell.source.replace('cleaned_property_data.csv', 'cleaned_mumbai_indore_property_data.csv')
            
        # update output file
        if 'geocoded_property_data.csv' in cell.source:
            cell.source = cell.source.replace('geocoded_property_data.csv', 'geocoded_mumbai_indore_property_data.csv')
            
        # update geocoding query to include City
        if 'query = f"{loc}, Mumbai, India"' in cell.source:
            cell.source = cell.source.replace('query = f"{loc}, Mumbai, India"', 'city_name = df[df["Location"] == loc]["City"].iloc[0]\n    query = f"{loc}, {city_name}, India"')

        # remove any simple map creation that relies on a hardcoded center if it exists. 
        # (Though folium usually centers on the mean of the coordinates).
        # We will add a new cell at the end for mapping both cities.

mapping_code = """
import folium

# Filter out rows with missing coordinates
map_df = df.dropna(subset=['Latitude', 'Longitude'])

if not map_df.empty:
    mumbai_data = map_df[map_df['City'] == 'Mumbai']
    indore_data = map_df[map_df['City'] == 'Indore']
    
    # Create Mumbai Map
    if not mumbai_data.empty:
        print("Mapping Mumbai Properties")
        mumbai_map = folium.Map(location=[mumbai_data['Latitude'].mean(), mumbai_data['Longitude'].mean()], zoom_start=11)
        for _, row in mumbai_data.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=f"{row['Location']} - ₹{row['Price_INR']}",
                color='blue',
                fill=True
            ).add_to(mumbai_map)
        display(mumbai_map)

    # Create Indore Map
    if not indore_data.empty:
        print("Mapping Indore Properties")
        indore_map = folium.Map(location=[indore_data['Latitude'].mean(), indore_data['Longitude'].mean()], zoom_start=12)
        for _, row in indore_data.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=f"{row['Location']} - ₹{row['Price_INR']}",
                color='green',
                fill=True
            ).add_to(indore_map)
        display(indore_map)
"""
nb.cells.append(nbf.v4.new_markdown_cell("## City-Specific Property Maps"))
nb.cells.append(nbf.v4.new_code_cell(mapping_code))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

