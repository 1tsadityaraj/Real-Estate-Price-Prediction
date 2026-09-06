import pandas as pd
import numpy as np
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import json
import ssl
import certifi
import geopy.geocoders

# SSL Bypass to prevent macOS cert verification failures
ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx
ssl._create_default_https_context = ssl._create_unverified_context

df = pd.read_csv('data/processed/cleaned_property_data.csv')

unique_locations = df['Location'].dropna().unique()

geolocator = Nominatim(user_agent="mumbai_real_estate_agent")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

location_dict = {}
successful = 0
failed = 0

for loc in unique_locations:
    query = f"{loc}, Mumbai, Maharashtra, India"
    try:
        location = geocode(query)
        if location:
            location_dict[loc] = (location.latitude, location.longitude)
            successful += 1
        else:
            fallback_query = f"{loc}, Mumbai, India"
            location = geocode(fallback_query)
            if location:
                location_dict[loc] = (location.latitude, location.longitude)
                successful += 1
            else:
                location_dict[loc] = (None, None)
                failed += 1
    except Exception as e:
        print(f"Error geocoding {loc}: {e}")
        location_dict[loc] = (None, None)
        failed += 1

df['Latitude'] = df['Location'].map(lambda x: location_dict.get(x, (None, None))[0])
df['Longitude'] = df['Location'].map(lambda x: location_dict.get(x, (None, None))[1])

output_path = 'data/processed/geocoded_property_data.csv'
df.to_csv(output_path, index=False)

missing_coords = df['Latitude'].isnull().sum()
success_rate = (successful / (successful + failed)) * 100 if (successful + failed) > 0 else 0

summary = {
    'total_unique': len(unique_locations),
    'successful': successful,
    'failed': failed,
    'success_rate': success_rate,
    'missing_records': int(missing_coords),
    'location_density': df['Location'].value_counts().head(5).to_dict(),
}

valid_df = df.dropna(subset=['Latitude', 'Longitude'])
if not valid_df.empty:
    loc_price = valid_df.groupby('Location').agg({
        'Price_INR': ['mean', 'median'],
        'price_per_sqft': 'mean'
    })
    loc_price.columns = ['Mean_Price', 'Median_Price', 'Mean_Price_Sqft']
    loc_price = loc_price.sort_values(by='Mean_Price', ascending=False)
    summary['highest_price_location'] = loc_price.index[0]
    summary['lowest_price_location'] = loc_price.index[-1]
    summary['highest_price_sqft_location'] = loc_price.sort_values(by='Mean_Price_Sqft', ascending=False).index[0]
else:
    summary['highest_price_location'] = "N/A"
    summary['lowest_price_location'] = "N/A"
    summary['highest_price_sqft_location'] = "N/A"

with open('geocode_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
    
print("Geocoding complete and saved to geocode_summary.json")
