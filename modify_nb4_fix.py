import nbformat as nbf
import json

nb_path = 'notebooks/geocoders_maps.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# I will replace the entire geocoding cell logic
# I need to find the cell that contains Nominatim

new_cell_content = """unique_locations = df['Location'].dropna().unique()
print(f"Found {len(unique_locations)} unique locations to geocode.\\n")

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="real_estate_agent")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

location_dict = {}
successful = 0
failed = 0

for loc in unique_locations:
    # get the city name from df
    city_name = df[df["Location"] == loc]["City"].iloc[0]
    query = f"{loc}, {city_name}, India"
    try:
        location = geocode(query)
        if location:
            location_dict[loc] = (location.latitude, location.longitude)
            successful += 1
        else:
            # Try a broader search without location if it failed completely
            location_dict[loc] = (None, None)
            failed += 1
    except Exception as e:
        print(f"Error geocoding {loc}: {e}")
        location_dict[loc] = (None, None)
        failed += 1

print(f"Geocoding complete! Successful: {successful} | Failed: {failed} | Success Rate: {(successful/(successful+failed))*100:.2f}%")
"""

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'Nominatim' in cell.source and 'RateLimiter' in cell.source:
            cell.source = new_cell_content

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

