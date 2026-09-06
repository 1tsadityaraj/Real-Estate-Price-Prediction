# PROJECT PRESENTATION AND VIVA PREPARATION

---

# PART 1 – PROJECT PRESENTATION / PPT SLIDES

## Slide 1 – Title
**REAL ESTATE PRICE PREDICTION AND PROPERTY ANALYSIS SYSTEM FOR MUMBAI AND INDORE USING MACHINE LEARNING**

**Student Name:** [STUDENT NAME]  
**Enrollment Number:** [ENROLLMENT NUMBER]  
**Department:** [DEPARTMENT]  
**College:** [COLLEGE]  
**Project Guide:** [PROJECT GUIDE]  
**Academic Year:** [ACADEMIC YEAR]  

---

## Slide 2 – Introduction
- Real-estate prices vary significantly based on location, area, and property characteristics.
- Manual estimation is difficult and often biased by human emotion.
- Machine learning can identify hidden mathematical relationships between property attributes and final market prices.
- This project focuses on comparing and predicting prices across two distinct markets: **Mumbai** and **Indore**.

---

## Slide 3 – Problem Statement
- **Difficulty in Estimation:** Real estate lacks standard pricing catalogs.
- **Multiple Influencing Factors:** Price depends simultaneously on Area, BHK, Property Type, and Location.
- **Geographic Variation:** A 2-BHK in Mumbai is priced entirely differently than a 2-BHK in Indore.
- **Need for Automation:** A data-driven approach is required to eliminate human bias.
- **Need for Accessibility:** End-users need an interactive, easy-to-use prediction system.

---

## Slide 4 – Objectives
- **Collect** property data across two cities.
- **Clean and preprocess** raw text and missing data.
- **Perform EDA** (Exploratory Data Analysis) to discover trends.
- **Perform Geographic Analysis** via Geocoding.
- **Engineer ML Features** (One-Hot Encoding, scaling).
- **Train** multiple regression models.
- **Compare** model performance.
- **Predict** property prices accurately.
- **Provide** a Streamlit interactive interface.
- **Compare** Mumbai and Indore markets.

---

## Slide 5 – Proposed System

```text
Data Collection
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Geocoding (Latitude/Longitude)
       ↓
Feature Engineering
       ↓
ML Model Training
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Streamlit Application
       ↓
Price Prediction & Analytics
```

---

## Slide 6 – Dataset
- **Data Sources:** Simulated/Mocked web-scraping datasets.
- **Total Records:** 440 unique properties.
- **Important Features:** `City`, `Location`, `Property_Type`, `Area_sqft`, `BHK`, `Bathrooms`, `Latitude`, `Longitude`.
- **Target Variable:** `Price_INR`
- **Property Types:** Apartment, Independent Floor, Independent House, Studio Apartment, Villa.
- **Locations:** Spread across Mumbai (e.g., Bandra, Andheri) and Indore (e.g., Vijay Nagar, Nipania).

---

## Slide 7 – Data Preprocessing
- **Missing-Value Handling:** Dropped incomplete rows to maintain data integrity.
- **Duplicate Removal:** Removed overlapping listings.
- **Data-Type Conversion:** Stripped text from numbers (e.g., "3 BHK" -> `3`).
- **Location Standardization:** Normalized spelling variations.
- **Outlier Handling:** Cap extreme anomalies in property dimensions.
- **Encoding:** Used Drop-First One-Hot Encoding for categorical features (`City`, `Location`).
- **Data Leakage Prevention:** Removed `price_per_sqft` from training features, as it mathematically reveals the target price.

---

## Slide 8 – Exploratory Data Analysis
*(Insert Actual Figures from docs/figures/)*
1. **Property Count by City:** Mumbai possesses significantly more listings in the dataset than Indore.
2. **Price Distribution:** Heavily right-skewed; Mumbai listings dominate the ultra-luxury upper tail.
3. **Area vs Price:** Strong positive correlation; however, identical areas (sq.ft) cost exponentially more in Mumbai.
4. **Property Type:** "Apartments" overwhelmingly dominate both markets.

---

## Slide 9 – Geographic Analysis
*(Insert actual geographic_distribution.html screenshot)*
- **Location Data:** Extracted via Geopy Nominatim API.
- **Coordinates:** Translated string locations into Latitude and Longitude floats.
- **Property Distribution:** The map visually clusters Mumbai properties on the western coast and Indore in central India.
- **Geographic Price Patterns:** Spatial clustering allows the algorithm to learn geographical premiums based on exact coordinates rather than just neighborhood names.

---

## Slide 10 – Machine Learning Models
Four supervised models were evaluated to find the best fit for real-estate regression:
1. **Linear Regression:** Establishes a linear equation minimizing error.
2. **Decision Tree Regressor:** Splits data based on conditional nodes.
3. **Random Forest Regressor:** An ensemble of decision trees to reduce variance.
4. **Polynomial Regression:** Explores non-linear polynomial combinations of features.

*Why compare?* To ensure we don't blindly pick a model that overfits or underfits the specific shape of our dataset.

---

## Slide 11 – Model Evaluation

| Model                 |         MAE |                 MSE |        RMSE |     R² |
| --------------------- | ----------: | ------------------: | ----------: | -----: |
| **Linear Regression** | 13,623,956 | 388,416,513,203,432 | 19,708,285 | 0.5135 |
| Random Forest         | 14,803,633 | 482,099,143,830,985 | 21,956,756 | 0.3961 |
| Polynomial Regression | 17,535,481 | 655,906,326,619,255 | 25,610,668 | 0.1784 |
| Decision Tree         | 17,908,450 | 701,707,183,098,591 | 26,489,756 | 0.1211 |

**Best Model:** Linear Regression (R² = 0.5135)

---

## Slide 12 – Best Model
- **Selected Model:** Linear Regression.
- **Why?** It achieved the highest R² score (0.5135) and lowest RMSE. Real estate pricing in this dataset relies heavily on linear correlations (Area scales with Price). Tree-based models (like Random Forest) severely overfitted the localized high-value outliers in Mumbai.
- **Important Characteristics:** Fast execution, easy serialization, highly interpretable feature coefficients.

---

## Slide 13 – Streamlit Application
*(Insert Screenshots of the Application Interface)*
**User Workflow:**
```text
Select City (Mumbai / Indore)
     ↓
Select Location (Cascades based on City)
     ↓
Enter Property Details (BHK, Area, Type)
     ↓
Click Predict
     ↓
Estimated Price Output (INR)
```

---

## Slide 14 – Mumbai and Indore Comparison
**Major Differences Observed:**
- **Median Price:** Mumbai's median price heavily overshadows Indore's, frequently crossing ₹3 Crores compared to Indore's sub-₹1 Crore medians.
- **Price per Square Foot:** Mumbai commands a massive premium for spatial footprint compared to Indore.
- **Volume:** The dataset indicates higher listing volumes in Mumbai's western suburbs compared to central Indore nodes.

---

## Slide 15 – Results
- **Data Pipeline:** Successfully completed, cleaning and standardizing 440 records.
- **Multi-City Support:** Seamlessly integrated Mumbai and Indore into a single predictive pipeline.
- **ML Models:** 4 models compared; **Linear Regression** selected as the optimal predictor.
- **Price Prediction:** Implemented via a serialized `.pkl` pipeline safely avoiding data leakage.
- **Market Analysis:** An interactive dashboard providing cross-city visual analytics was successfully deployed.
- **Streamlit App:** Live, robust, and handles errors gracefully.

---

## Slide 16 – Limitations
- **Dataset Size:** The model is trained on 440 properties. Commercial models require hundreds of thousands of rows for absolute precision.
- **Geographic Coverage:** Hyper-local nuances (e.g., "sea-facing") are missing.
- **Data Freshness:** Static dataset; does not account for real-time inflation or interest rates.
- **Prediction Uncertainty:** Free Geocoding APIs occasionally fail on obscure building names.

---

## Slide 17 – Future Scope
- **Add more Indian cities:** Delhi, Bangalore, Pune.
- **Larger Datasets:** Utilizing live commercial API streams.
- **More Property Attributes:** Adding age of property, proximity to transit, crime rates.
- **More Advanced Models:** Exploring Gradient Boosting or Neural Networks on larger data.
- **Cloud Deployment:** Hosting the Streamlit app on AWS or Heroku.

---

## Slide 18 – Conclusion
- **Problem Solved:** Automated an unbiased property valuation mechanism.
- **Approach:** End-to-end Machine Learning pipeline.
- **Support:** Successfully modeled both Mumbai and Indore simultaneously.
- **ML Prediction:** Linear Regression successfully predicted prices based on spatial and architectural variables.
- **Deployment:** Streamlit provided an accessible, dynamic dashboard for non-technical users.
- **Outcome:** A robust academic proof-of-concept for real-estate market analytics.

**Thank You!**

---
---

# PART 2 – PROJECT DEMONSTRATION SCRIPT

**Step 1:** Open the project directory and display the `README.md`. Explain that the project is fully documented.
**Step 2:** Briefly explain the architecture: "We collect data, clean it in Jupyter notebooks, apply Geopy for coordinates, train models, and export the best one (`best_model.pkl`) to our Streamlit frontend."
**Step 3:** Open the terminal and run `streamlit run app.py`.
**Step 4:** Navigate to the **"Property Price Prediction"** tab. Select **Mumbai**.
**Step 5:** Enter valid details (e.g., Location: Andheri West, 3 BHK, Apartment, 1500 sq.ft, 3 Bathrooms).
**Step 6:** Click **Predict Property Price**. Show the output formatting (e.g., ₹ X.XX Crore).
**Step 7:** Navigate to the **"Market Analysis Dashboard"** tab. Select **Mumbai** in the filter. Show the median price and area distributions.
**Step 8:** Return to Prediction tab. Switch City to **Indore**. Show how the locations dynamically update.
**Step 9:** Enter valid details (e.g., Location: Vijay Nagar, 3 BHK, Apartment, 1500 sq.ft, 3 Bathrooms).
**Step 10:** Click **Predict Property Price**. Highlight the massive price difference compared to the identical Mumbai property.
**Step 11:** Navigate to the Dashboard. Select **Indore**. Show the localized Indore market statistics.
**Step 12:** In the Dashboard, set City to **All**. Show the "Mumbai vs Indore (Median Price)" comparison bar chart to emphasize the city premium.
**Step 13:** Navigate to the **"About Model"** tab. Briefly show the R² (0.5135) and MAE evaluation metrics confirming it runs on Linear Regression.

*(Total Time: ~6 minutes)*

---
---

# PART 3 – VIVA QUESTIONS AND ANSWERS

## A. Basic Project Questions
**1. What is the title of your project?**
Real Estate Price Prediction and Property Analysis System for Mumbai and Indore Using Machine Learning.

**2. What is the main objective?**
To accurately predict property prices using machine learning and provide a comparative analysis of the Mumbai and Indore real estate markets.

**3. Why did you choose this project?**
Real estate is a massive industry lacking transparent, data-driven pricing tools for average consumers.

**4. What problem does it solve?**
It eliminates human bias and manual estimation errors by calculating valuations based on mathematical correlations of location, area, and configuration.

**5. Why did you select Mumbai and Indore?**
To compare a highly dense, ultra-expensive Tier-1 metropolis (Mumbai) against an emerging, rapidly growing Tier-2 market (Indore).

**6. What is the input to your system?**
City, Location, Property Type, BHK, Bathrooms, and Area in square feet.

**7. What is the output?**
An estimated property price in Indian Rupees (INR).

**8. Who can use this system?**
Homebuyers, sellers, real estate agents, and market researchers.

**9. What are the major modules?**
Data Cleaning, EDA, Geocoding, Feature Engineering, ML Training, and Streamlit Deployment.

## B. Dataset Questions
**10. What is your data source?**
The dataset is an aggregated, cleaned proxy dataset representing real estate listings for both cities.

**11. How was the data collected?**
It simulates scraped records commonly extracted from commercial property listing websites.

**12. How many records are present?**
There are 440 total property records.

**13. What features are present?**
City, Location, Property Type, Price_INR, Area_sqft, BHK, Bathrooms, Latitude, and Longitude.

**14. What is the target variable?**
`Price_INR` (The final price of the property).

**15. How did you handle missing values?**
Rows with critical missing values (like Price or Location) were dropped to prevent training the model on fabricated data.

**16. How did you remove duplicates?**
Using Pandas' `drop_duplicates()` function based on matching property configurations.

**17. How did you handle outliers?**
Extreme anomalies in property area were capped using statistical bounds (e.g., IQR methods) during the EDA phase.

**18. Why is data preprocessing necessary?**
Machine learning models require clean, purely numerical matrices. They cannot understand text like "3 BHK" or null (`NaN`) values.

**19. What problems did you face while collecting data?**
Dealing with non-standard text formats, varying units (e.g., converting everything to sq.ft), and spelling errors in location names.

## C. Machine Learning Questions
**20. Why did you use regression?**
Because the target variable (`Price_INR`) is a continuous numerical value, not a categorical class.

**21. Why Linear Regression?**
It fits a linear equation to the data. Real estate strongly correlates linearly (e.g., as Area goes up, Price goes up).

**22. What is Decision Tree Regression?**
A non-linear model that splits data into branches based on feature conditions to make a prediction.

**23. What is Random Forest Regression?**
An ensemble method that builds multiple decision trees and averages their predictions to reduce variance and overfitting.

**24. What is Polynomial Regression?**
It models the relationship as an nth-degree polynomial, allowing for curved relationships.

**25. Why did you compare multiple models?**
To ensure the chosen algorithm actually fits the underlying data shape rather than guessing which one works best.

**26. Which model performed best?**
Linear Regression.

**27. Why was the best model selected?**
It achieved the highest R² score (0.5135) and generalized better than tree-based models, which overfitted the high-priced Mumbai properties.

**28. What is train-test split?**
Dividing the dataset so the model trains on one portion (e.g., 80%) and is evaluated on unseen data (20%).

**29. Why did you use the selected split ratio?**
An 80-20 split provides enough data for the model to learn while leaving a statistically significant portion for validation.

**30. What is overfitting?**
When a model memorizes the training data perfectly but fails drastically on new, unseen data.

**31. How can overfitting affect your project?**
The model would accurately price the 440 properties in our dataset, but give wildly inaccurate prices when a user enters a new property on the app.

**32. What is underfitting?**
When a model is too simple to capture the underlying trend of the data.

**33. What is feature engineering?**
Transforming raw data into formats that better represent the underlying problem to the ML algorithm (e.g., extracting latitude/longitude).

**34. What is one-hot encoding?**
Converting categorical text data (like "Mumbai" or "Indore") into binary columns (0s and 1s) so the math algorithm can process it.

**35. Why is preprocessing important?**
Because "garbage in, garbage out." The model is only as smart as the data it is fed.

## D. Evaluation Questions
**36. What is MAE?**
Mean Absolute Error: The average absolute difference between predicted and actual prices.

**37. What is MSE?**
Mean Squared Error: The average of the squared differences. It heavily penalizes large errors.

**38. What is RMSE?**
Root Mean Squared Error: The square root of MSE, bringing the error back to the original unit (Rupees).

**39. What is R²?**
R-squared: Explains the proportion of variance in the dependent variable that is predictable from the independent variables (scale of 0 to 1).

**40. Which metric is most useful for your project and why?**
R² explains the model's overall fit, while MAE gives a direct, easily interpretable Rupee value for how far off predictions are on average.

**41. What were the final model results?**
Linear Regression achieved an R² of 0.5135.

**42. Why shouldn't regression performance simply be called “accuracy”?**
"Accuracy" is a classification metric measuring exact correct matches. In regression, we rarely predict the *exact* Rupee value, so we measure the *error* distance instead.

## E. Data Leakage Questions
**43. What is data leakage?**
When information from outside the training dataset is used to create the model, artificially inflating its performance.

**44. Why should `price_per_sqft` not be used as an input feature?**
Because `price_per_sqft` * Area = Total Price. Providing it directly hands the answer to the model, rendering it useless for predicting new properties where the user doesn't already know the price.

**45. How did you prevent target leakage?**
By explicitly dropping the `price_per_sqft` column before the train-test split.

**46. Why must preprocessing be consistent between training and prediction?**
If the model was trained on one-hot encoded columns, the user input from Streamlit must be encoded into the exact same matrix shape, otherwise the model will crash.

## F. Streamlit Questions
**47. Why did you use Streamlit?**
It is a fast, Python-native framework for building interactive web apps tailored specifically for Data Science.

**48. What does `app.py` do?**
It hosts the UI, captures user input, transforms it, passes it to the loaded model, and displays interactive charts.

**49. Does the application retrain the model?**
No, retraining every time would be too slow. It uses pre-trained serialized artifacts.

**50. How does the application load the trained model?**
Using the `joblib` library to deserialize `best_model.pkl` and `preprocessor.pkl`.

**51. How does city selection work?**
It dynamically filters the dataset to only show valid dependent variables (like Locations) for the chosen City.

**52. How are Mumbai and Indore locations handled?**
If Mumbai is selected, the location dropdown strictly populates with Mumbai locations, preventing users from searching for a Mumbai property in an Indore neighborhood.

**53. How is the predicted price displayed?**
It is formatted back into Indian currency formatting (Lakhs/Crores) for readability.

**54. How did you handle invalid inputs?**
Through UI constraints (sliders/dropdowns) and error boundary catches (`st.warning`) for empty filter results in the dashboard.

**55. Why did you use caching?**
`st.cache_resource` prevents the heavy `best_model.pkl` from being reloaded into memory every time the user clicks a button, ensuring instantaneous predictions.

## G. Geospatial Questions
**56. What is geocoding?**
The process of converting addresses or location names into geographic coordinates.

**57. Why did you use Geopy?**
It provides an easy Python wrapper to access the Nominatim API to fetch real-world coordinates.

**58. What are latitude and longitude?**
Numerical representations of a physical location on the Earth's surface.

**59. How did geographic information help the project?**
It allowed the model to understand the spatial distance between properties, capturing the "premium" associated with specific geographic clusters.

**60. What problems can occur during geocoding?**
The API might fail to recognize hyper-local spellings or time out due to rate limits.

## H. Project-Specific Questions
**61. Why did you combine Mumbai and Indore into one model?**
To test if a single algorithm could successfully isolate and scale the massive economic disparity between a Tier-1 and Tier-2 city utilizing the `City` feature.

**62. Why is city included as a feature?**
Because the economic baseline of the city is the strongest determinant of the final price.

**63. Could you build separate models for each city?**
Yes, but a unified model is more scalable and prevents having to maintain dozens of separate models in production.

**64. What would happen if you added another city?**
We would need to provide training data for that city, retrain the model to update the one-hot encoding schema, and update the Streamlit UI.

**65. How would you improve the model?**
By scaling the dataset from 440 to 40,000+ records and implementing more complex algorithms like XGBoost.

**66. What are the major limitations?**
The relatively small dataset size limits absolute real-world accuracy.

**67. What would you do with a larger dataset?**
Tree-based ensemble models (Random Forest) would likely stop overfitting and begin to outperform Linear Regression.

**68. Can this system guarantee the actual market price?**
No, it provides a statistical estimation based on historical data.

**69. Is this system suitable for real property valuation?**
It serves as an excellent academic proof-of-concept, but commercial valuation requires much larger data and macro-economic indicators.

**70. What is the biggest challenge you faced?**
Ensuring the `ColumnTransformer` properly applied drop-first one-hot encoding seamlessly between the training notebook and the live Streamlit inference pipeline to prevent matrix shape errors.

---
---

# PART 4 – “EXPLAIN YOUR PROJECT IN 1 MINUTE”

"Good morning. My project is a **Real Estate Price Prediction and Analysis System** covering both Mumbai and Indore. The core problem is that manual property valuation is biased and ignores multi-variable data. I built a pipeline that cleans raw property data, extracts Latitude and Longitude using the Geopy API, and analyzes the extreme price differences between a Tier-1 and Tier-2 city. I evaluated four machine learning algorithms, and Linear Regression proved to be the most stable, successfully learning the massive price premium associated with Mumbai. Finally, I deployed this model into an interactive Streamlit web dashboard where users can instantly estimate property prices and explore interactive market analytics without writing a single line of code."

---

# PART 5 – “EXPLAIN YOUR PROJECT IN 5 MINUTES”

"Hello, I will be presenting my major project: A Machine Learning-based Real Estate Price Predictor covering Mumbai and Indore. 

**Introduction & Problem:** Real estate is highly volatile. Buyers and sellers rely on brokers who often provide biased, subjective valuations. My project solves this by using historical data to calculate objective price estimations based on math, not emotion.

**Dataset & Preprocessing:** I used a dataset of 440 properties containing attributes like Area, BHK, Property Type, and Location across both cities. Data cleaning was critical—I stripped strings from numeric columns, handled null values, and eliminated data leakage by explicitly dropping `price_per_sqft` from the training set, ensuring the model had to learn the relationships organically. 

**EDA & Geocoding:** During Exploratory Data Analysis, I discovered that identical property dimensions cost exponentially more in Mumbai than Indore. To help the ML model understand this spatially, I utilized Geopy's Nominatim API to map string locations into exact Latitude and Longitude coordinates.

**Models & Evaluation:** I engineered the features using One-Hot Encoding and fed the data into four algorithms: Linear Regression, Decision Tree, Random Forest, and Polynomial Regression. Interestingly, Linear Regression performed the best, achieving an R² of 0.5135. The tree-based models actually overfitted the data because they tried to memorize the ultra-expensive outliers in Mumbai rather than generalizing the trend.

**Deployment (Streamlit):** An ML model in a Jupyter Notebook isn't accessible to normal users. So, I serialized my best model using `joblib` and built a frontend using Streamlit. The application allows users to dynamically select their City and Location, input property metrics, and receive an instant Rupee valuation. I also built a Market Analysis Dashboard right into the app to visualize price distributions.

**Conclusion & Future Scope:** The project successfully proves that a single ML model can handle highly disparate markets if given proper geographic features. In the future, this architecture can easily be scaled by integrating live API data streams and expanding to cover all major Indian metropolitan cities."

---

# PART 6 – DIFFICULT EXAMINER QUESTIONS

**Question: Why should we trust a machine-learning prediction for a property?**
*Answer:* You shouldn't trust it blindly as an absolute truth. It acts as an objective baseline. ML removes emotional bias, providing a valuation rooted strictly in historical mathematical correlations, which is incredibly useful as a starting point for negotiations.

**Question: Why didn't you use a more advanced model like Deep Learning?**
*Answer:* Deep Learning requires massive amounts of data (often tens of thousands of rows) to outperform traditional ML. On a tabular dataset of 440 records, a Neural Network would drastically overfit. Linear Regression was mathematically the most appropriate choice for this specific dataset size and shape.

**Question: Why did you use one model for two cities instead of two separate models?**
*Answer:* To test the model's scalability. By adding `City` and `Coordinates` as features, a single model learns to apply the correct city premium dynamically. This is much easier to maintain in production than hosting dozens of separate models for dozens of cities.

**Question: What if Mumbai has much more data than Indore?**
*Answer:* That causes class imbalance. The model might become biased toward learning Mumbai's pricing structures at the expense of Indore's accuracy. In future iterations, techniques like SMOTE (Synthetic Minority Over-sampling) or stratified sampling could be used to balance the distributions.

**Question: What happens if the user enters a location that was not present during training?**
*Answer:* I prevented this entirely via Streamlit's UI. The Location dropdowns are populated strictly from the unique values present in the training dataset, physically preventing the user from entering an unseen categorical variable that would crash the matrix.

**Question: Why can't `price_per_sqft` be used as an input?**
*Answer:* Because `price_per_sqft` is calculated by dividing the Total Price by the Area. If we feed it to the model, we are secretly giving the model the answer (Target Leakage). When a real user goes to predict a price, they won't know the price per sq.ft yet, so the model wouldn't work.

**Question: How would you improve the model with more time?**
*Answer:* I would scrape 50,000+ live records from sites like MagicBricks, add macro-economic features like interest rates, and apply an XGBoost ensemble model.

---

# PART 7 – FINAL PRESENTATION CHECKLIST

## Before Presentation
- [ ] Project runs (`streamlit run app.py`).
- [ ] Streamlit caching works perfectly.
- [ ] Mumbai prediction tested.
- [ ] Indore prediction tested.
- [ ] Model loads without matrix errors.
- [ ] PPT ready and reviewed.
- [ ] README and Report accessible.
- [ ] GitHub updated.

## During Demonstration
- [ ] Explain the problem clearly.
- [ ] Explain the unified workflow.
- [ ] Show a Mumbai prediction.
- [ ] Show an Indore prediction to highlight the contrast.
- [ ] Show the comparative Market Dashboard.
- [ ] Honestly explain the limitations (dataset size).

## Viva Prep
- [ ] Know the dataset size (440 records).
- [ ] Know the target variable (`Price_INR`).
- [ ] Know the best model (Linear Regression).
- [ ] Know the actual R² score (0.5135).
- [ ] Understand why Drop-First One-Hot Encoding is used.
- [ ] Understand Data Leakage perfectly.
