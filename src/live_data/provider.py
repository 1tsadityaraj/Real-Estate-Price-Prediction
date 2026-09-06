import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

class PropertyDataProvider(ABC):
    @abstractmethod
    def fetch_listings(self, city: str, location: str, property_type: str, bhk: int) -> pd.DataFrame:
        """Fetch current listings from the live data source."""
        pass

class ConfiguredAPIDataProvider(PropertyDataProvider):
    """
    A real-world implementation would contain the API keys and endpoints for a commercial data provider.
    Since web-scraping and bypassing anti-bot protections is prohibited, and we do not have an authorized API key,
    this provider gracefully returns an empty DataFrame to trigger the required fallback behavior.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.is_configured = False # Set to False to prove missing API key

    def fetch_listings(self, city: str, location: str, property_type: str, bhk: int) -> pd.DataFrame:
        if not self.is_configured:
            return pd.DataFrame()
            
        # If it were configured, it would return columns:
        # ['listing_id', 'city', 'location', 'property_type', 'bhk', 'area_sqft', 'price_inr', 'price_per_sqft', 'bathrooms', 'source', 'retrieved_at']
        return pd.DataFrame()

class ComparableAnalyzer:
    """Analyzes a set of listings to find comparables based on hierarchical matching."""
    
    def __init__(self, listings_df: pd.DataFrame):
        self.df = listings_df
        
    def find_comparables(self, city: str, location: str, property_type: str, bhk: int, area: float) -> dict:
        """Finds comparable properties and calculates market estimates."""
        if self.df.empty:
            return {
                "status": "LIVE DATA PROVIDER NOT CONFIGURED",
                "count": 0,
                "estimate": None,
                "median_sqft_price": None,
                "match_level": "None"
            }
            
        # Try Level 1 Match: Exact Location, Type, BHK, and Area within 20%
        level1 = self.df[
            (self.df['city'].str.lower() == city.lower()) &
            (self.df['location'].str.lower() == location.lower()) &
            (self.df['property_type'].str.lower() == property_type.lower()) &
            (self.df['bhk'] == bhk) &
            (self.df['area_sqft'] >= area * 0.8) &
            (self.df['area_sqft'] <= area * 1.2)
        ]
        
        if len(level1) >= 3:
            return self._compute_stats(level1, area, "Level 1 (Very Similar)")
            
        # Try Level 2 Match: Exact Location, Type, and Area within 30% (relax BHK)
        level2 = self.df[
            (self.df['city'].str.lower() == city.lower()) &
            (self.df['location'].str.lower() == location.lower()) &
            (self.df['property_type'].str.lower() == property_type.lower()) &
            (self.df['area_sqft'] >= area * 0.7) &
            (self.df['area_sqft'] <= area * 1.3)
        ]
        
        if len(level2) >= 3:
            return self._compute_stats(level2, area, "Level 2 (Similar Location/Area)")
            
        # Try Level 3 Match: Broader City, Type (relax Location)
        level3 = self.df[
            (self.df['city'].str.lower() == city.lower()) &
            (self.df['property_type'].str.lower() == property_type.lower()) &
            (self.df['area_sqft'] >= area * 0.7) &
            (self.df['area_sqft'] <= area * 1.3)
        ]
        
        if len(level3) > 0:
            return self._compute_stats(level3, area, "Level 3 (Broader City/Type Match)")
            
        return {
            "status": "INSUFFICIENT COMPARABLES",
            "count": 0,
            "estimate": None,
            "median_sqft_price": None,
            "match_level": "None"
        }
        
    def _compute_stats(self, matched_df: pd.DataFrame, target_area: float, match_level: str) -> dict:
        median_sqft_price = matched_df['price_per_sqft'].median()
        estimated_price = target_area * median_sqft_price
        
        return {
            "status": "SUCCESS",
            "count": len(matched_df),
            "estimate": estimated_price,
            "median_sqft_price": median_sqft_price,
            "match_level": match_level,
            "listings": matched_df.to_dict('records')
        }
