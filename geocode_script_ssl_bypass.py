import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import ssl
import geopy.geocoders

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
geopy.geocoders.options.default_ssl_context = ctx

df = pd.read_csv('data/processed/cleaned_mumbai_indore_property_data.csv')
unique_locations = df['Location'].dropna().unique()
print(f"Found {len(unique_locations)} unique locations to geocode.")

geolocator = Nominatim(user_agent="mumbai_indore_property_agent_v2")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

location_dict = {}

for loc in unique_locations:
    city_name = df[df["Location"] == loc]["City"].iloc[0]
    query = f"{loc}, {city_name}, India"
    try:
        location = geocode(query)
        if location:
            location_dict[loc] = (location.latitude, location.longitude)
        else:
            print(f"Failed to geocode: {query}")
            location_dict[loc] = (None, None)
    except Exception as e:
        print(f"Error for {query}: {e}")
        location_dict[loc] = (None, None)

df['Latitude'] = df['Location'].map(lambda x: location_dict.get(x, (None, None))[0])
df['Longitude'] = df['Location'].map(lambda x: location_dict.get(x, (None, None))[1])

df.to_csv('data/processed/geocoded_mumbai_indore_property_data.csv', index=False)
print("Saved geocoded_mumbai_indore_property_data.csv")
