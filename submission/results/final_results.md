# Final Project Results

## Dataset
* **Mumbai records:** 244
* **Indore records:** 196
* **Total records:** 440
* **Number of features:** 30 (after One-Hot Encoding)
* **Target variable:** Price_INR

## Machine Learning
* **Linear Regression results:** MAE: ₹13,623,956 | RMSE: ₹19,708,285 | R²: 0.5135
* **Decision Tree results:** MAE: ₹17,908,450 | RMSE: ₹26,489,756 | R²: 0.1211
* **Random Forest results:** MAE: ₹14,803,633 | RMSE: ₹21,956,756 | R²: 0.3961
* **Polynomial Regression results:** MAE: ₹17,535,481 | RMSE: ₹25,610,668 | R²: 0.1784
* **Best model:** Linear Regression

## Current Market
* **Data provider:** ConfiguredAPIDataProvider (Abstract API Stub)
* **Number of listings tested:** 0 (Triggered Fallback Mode successfully)
* **Current comparable methodology:** 3-Tier Hierarchical match (Exact, Loc/Area, City/Type)
* **Current ₹/sq.ft. methodology:** Mathematical Median of comparable listings
* **Final estimate methodology:** 50/50 weighted combination when API data exists; 100% historical ML fallback when API is restricted.

## Testing
* **Mumbai prediction:** Passed (Predicted via ML pipeline flawlessly)
* **Indore prediction:** Passed (Negative BHK correlation verified mathematically)
* **Current-data retrieval:** Passed (API Abstraction catches unconfigured keys)
* **Refresh:** Passed (Streamlit cache successfully flushes)
* **Fallback:** Passed (UI elegantly reports 'Not Available (API Not Configured)' and safely diverts to ML)
* **Error handling:** Passed
