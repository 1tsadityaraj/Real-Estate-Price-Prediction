import pandas as pd
import joblib

preprocessor = joblib.load('model/preprocessor.pkl')
model = joblib.load('model/best_model.pkl')

print("Testing Mumbai Prediction...")
mumbai_input = pd.DataFrame([{
    'City': 'Mumbai',
    'Location': 'Andheri West',
    'Property_Type': 'Apartment',
    'Area_sqft': 1500,
    'BHK': 3,
    'Bathrooms': 2,
    'Latitude': 19.1363,
    'Longitude': 72.8277
}])
m_trans = preprocessor.transform(mumbai_input)
m_pred = model.predict(m_trans)[0]
print(f"Mumbai Estimated Price: ₹ {m_pred:,.2f}")

print("\nTesting Indore Prediction...")
indore_input = pd.DataFrame([{
    'City': 'Indore',
    'Location': 'Vijay Nagar',
    'Property_Type': 'Apartment',
    'Area_sqft': 1500,
    'BHK': 3,
    'Bathrooms': 2,
    'Latitude': 22.7533,
    'Longitude': 75.8937
}])
i_trans = preprocessor.transform(indore_input)
i_pred = model.predict(i_trans)[0]
print(f"Indore Estimated Price: ₹ {i_pred:,.2f}")

