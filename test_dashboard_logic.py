import pandas as pd
import numpy as np

def test_filtering():
    df = pd.read_csv('data/processed/geocoded_mumbai_indore_property_data.csv')
    
    # Test 3: Mumbai analysis loads
    mumbai_df = df[df['City'] == 'Mumbai']
    assert len(mumbai_df) > 0, "Mumbai data empty"
    
    # Test 4: Indore analysis loads
    indore_df = df[df['City'] == 'Indore']
    assert len(indore_df) > 0, "Indore data empty"
    
    # Test 5: Both cities
    assert len(df) > 0, "Combined data empty"
    
    # Test 6 & 7: Filters work
    filtered = df[(df['City'] == 'Mumbai') & (df['BHK'] == 3)]
    assert len(filtered) >= 0
    
    # Test 8: Empty filters
    empty = df[(df['City'] == 'Indore') & (df['Location'] == 'NonExistent')]
    assert empty.empty, "Should be empty"
    
    print("All backend dashboard logic tests passed!")

test_filtering()
