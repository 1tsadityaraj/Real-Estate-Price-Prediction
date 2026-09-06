import pandas as pd

# Hardcoded coordinates for our generated data locations to bypass SSL issues
LOCATION_COORDS = {
    # Mumbai
    'Andheri West': (19.1363, 72.8277),
    'Bandra East': (19.0600, 72.8544),
    'Juhu': (19.1048, 72.8267),
    'Powai': (19.1176, 72.9060),
    'Borivali West': (19.2372, 72.8441),
    'Goregaon East': (19.1693, 72.8596),
    'Malad West': (19.1843, 72.8347),
    'Kandivali East': (19.2066, 72.8687),
    'Worli': (19.0069, 72.8156),
    'Lower Parel': (18.9953, 72.8300),
    
    # Indore
    'Vijay Nagar': (22.7533, 75.8937),
    'Palasia': (22.7244, 75.8856),
    'Bhawarkua': (22.6916, 75.8672),
    'Nipania': (22.7667, 75.9167),
    'Rau': (22.6333, 75.8167),
    'AB Road': (22.7196, 75.8577),
    'Sudama Nagar': (22.6953, 75.8333),
    'LIG Colony': (22.7369, 75.8858),
    'Bypass Road': (22.7230, 75.9320),
    'Super Corridor': (22.7589, 75.8364)
}

df = pd.read_csv('data/processed/cleaned_mumbai_indore_property_data.csv')
unique_locations = df['Location'].dropna().unique()
print(f"Found {len(unique_locations)} unique locations to geocode.")

df['Latitude'] = df['Location'].map(lambda x: LOCATION_COORDS.get(x, (None, None))[0])
df['Longitude'] = df['Location'].map(lambda x: LOCATION_COORDS.get(x, (None, None))[1])

df.to_csv('data/processed/geocoded_mumbai_indore_property_data.csv', index=False)
print("Saved geocoded_mumbai_indore_property_data.csv")

# Generate the maps logic
import folium

map_df = df.dropna(subset=['Latitude', 'Longitude'])
if not map_df.empty:
    mumbai_data = map_df[map_df['City'] == 'Mumbai']
    indore_data = map_df[map_df['City'] == 'Indore']
    
    # Create Mumbai Map
    if not mumbai_data.empty:
        print("Mapping Mumbai Properties")
        mumbai_map = folium.Map(location=[mumbai_data['Latitude'].mean(), mumbai_data['Longitude'].mean()], zoom_start=11)
        mumbai_map.save("screenshots/mumbai_map.html")

    # Create Indore Map
    if not indore_data.empty:
        print("Mapping Indore Properties")
        indore_map = folium.Map(location=[indore_data['Latitude'].mean(), indore_data['Longitude'].mean()], zoom_start=12)
        indore_map.save("screenshots/indore_map.html")
