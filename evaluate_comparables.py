import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.live_data.provider import ComparableAnalyzer

def evaluate_comparables():
    df = pd.read_csv('data/processed/cleaned_mumbai_indore_property_data.csv')
    
    # We must ensure price_per_sqft exists for the ComparableAnalyzer
    if 'price_per_sqft' not in df.columns:
        df['price_per_sqft'] = df['Price_INR'] / df['Area_sqft']
        
    # Standardize column names for the Analyzer
    eval_df = df.rename(columns={
        'City': 'city',
        'Location': 'location',
        'Property_Type': 'property_type',
        'BHK': 'bhk',
        'Area_sqft': 'area_sqft',
        'Price_INR': 'price_inr'
    })
    
    results = []
    
    print("Starting Offline Validation of Comparable Market Estimates...")
    
    for idx, row in eval_df.iterrows():
        # Hold out the current property (hide actual price)
        holdout_df = eval_df.drop(idx)
        analyzer = ComparableAnalyzer(holdout_df)
        
        comp_result = analyzer.find_comparables(
            city=row['city'],
            location=row['location'],
            property_type=row['property_type'],
            bhk=row['bhk'],
            area=row['area_sqft']
        )
        
        if comp_result['status'] == 'SUCCESS':
            results.append({
                'actual_price': row['price_inr'],
                'estimated_price': comp_result['estimate'],
                'match_level': comp_result['match_level'],
                'city': row['city']
            })
            
    res_df = pd.DataFrame(results)
    
    if res_df.empty:
        print("No successful comparables found.")
        return
        
    print(f"\\nEvaluated {len(res_df)} properties out of {len(eval_df)}")
    
    mae = np.mean(np.abs(res_df['actual_price'] - res_df['estimated_price']))
    
    # Calculate R-squared
    ss_res = np.sum((res_df['actual_price'] - res_df['estimated_price']) ** 2)
    ss_tot = np.sum((res_df['actual_price'] - res_df['actual_price'].mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"Comparable-Only Model MAE: ₹ {mae:,.2f}")
    print(f"Comparable-Only Model R²: {r2:.4f}")
    
    print("\\nResults by City:")
    for city in res_df['city'].unique():
        city_df = res_df[res_df['city'] == city]
        c_mae = np.mean(np.abs(city_df['actual_price'] - city_df['estimated_price']))
        print(f"{city} MAE: ₹ {c_mae:,.2f} ({len(city_df)} properties)")

if __name__ == "__main__":
    evaluate_comparables()
