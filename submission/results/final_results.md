# Final Project Results

## Data Source
* **Mumbai historical data:** 250 property listings (`data/raw/property_data.csv`)
* **Indore historical data:** 200 property listings (`data/raw/indore_property_data.csv`)
* **Total records (after cleaning):** 440
* **Number of features:** 30 (after One-Hot Encoding)
* **Target variable:** Price_INR
* **Data type:** Historical real-estate property listing data

## Machine Learning
* **Linear Regression results:** MAE: ₹15,111,250 | RMSE: ₹19,416,551 | R²: 0.4885
* **Decision Tree results:** MAE: ₹19,188,640 | RMSE: ₹26,599,997 | R²: 0.0398
* **Random Forest results:** MAE: ₹14,862,030 | RMSE: ₹19,568,949 | R²: 0.4803
* **Polynomial Regression results:** MAE: ₹17,519,460 | RMSE: ₹23,285,561 | R²: 0.2642
* **Best model:** Linear Regression

## Testing
* **Mumbai prediction:** Passed (Predicted via ML pipeline successfully)
* **Indore prediction:** Passed (Predicted via ML pipeline successfully)
* **City switching:** Passed (Location dropdown correctly filters by selected city)
* **Market analysis:** Passed (Historical charts and KPIs render correctly)
* **Error handling:** Passed (Streamlit gracefully handles invalid inputs)

## Notes
* No live market API or real-time property listing feed is used in the final system.
* The predicted price is an ML-based estimate derived from historical property data.
