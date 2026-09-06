import pandas as pd
import joblib

def test_prediction(city, location, p_type, area, bhk, bath):
    try:
        model = joblib.load('model/best_model.pkl')
        preprocessor = joblib.load('model/preprocessor.pkl')
        df = pd.read_csv('data/processed/geocoded_mumbai_indore_property_data.csv')
        
        city_df = df[df['City'] == city]
        if location not in city_df['Location'].values:
            return "Error: Invalid Location for City"
            
        coord_map = city_df.drop_duplicates(subset=['Location']).set_index('Location')[['Latitude', 'Longitude']].to_dict('index')
        lat = coord_map.get(location, {}).get('Latitude', 0)
        lon = coord_map.get(location, {}).get('Longitude', 0)
        
        input_data = pd.DataFrame([{
            'City': city,
            'Location': location,
            'Property_Type': p_type,
            'Area_sqft': area,
            'BHK': bhk,
            'Bathrooms': bath,
            'Latitude': lat,
            'Longitude': lon
        }])
        
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)[0]
        return prediction
    except Exception as e:
        return f"Exception: {e}"

print("Test 1 - Mumbai Prediction:")
print(test_prediction('Mumbai', 'Andheri West', 'Apartment', 1500, 3, 2))

print("\nTest 2 - Indore Prediction:")
print(test_prediction('Indore', 'Vijay Nagar', 'Apartment', 1500, 3, 2))

print("\nTest 3 - Invalid Input (City Mismatch):")
print(test_prediction('Indore', 'Andheri West', 'Apartment', 1500, 3, 2))

print("\nTest 4 - Consistency (Run Mumbai again):")
print(test_prediction('Mumbai', 'Andheri West', 'Apartment', 1500, 3, 2))

