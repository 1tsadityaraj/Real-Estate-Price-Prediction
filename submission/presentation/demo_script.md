# Real Estate Prediction Project Demo Script

### 0:00–0:45: Introduction
"Hello, my project is a Real Estate Price Prediction and Property Analysis System for Mumbai and Indore. The goal is to use historical real-estate property listing data and machine learning to generate data-driven property price estimates."

### 0:45–1:30: Pipeline Overview
"The architecture follows a strict pipeline: We start with historical property data from Mumbai and Indore. The data is cleaned and analyzed through exploratory data analysis. It is then geocoded using the Nominatim geocoding service to fetch coordinates, and structurally feature-engineered using a Scikit-Learn Pipeline with SimpleImputers and OneHotEncoders. The best-performing Linear Regression model is saved and served via this Streamlit application."

### 1:30–2:30: Demonstrate Mumbai
"Let's predict a property in Mumbai. I select 'Mumbai', 'Andheri West', 'Apartment', 2 BHK, and 1000 sq.ft. I click predict. The model processes the coordinates and features through the saved preprocessing pipeline and gives us an estimated property price."

### 2:30–3:30: Demonstrate Indore
"Now for Indore. I select 'Indore', 'Ab Road', 'Apartment', 2 BHK, 1000 sq.ft. Notice the significant price difference compared to Mumbai. Our exploratory data analysis revealed that identical property dimensions cost exponentially more in Mumbai than in Indore, and the ML model accurately learned this geographic price disparity."

### 3:30–4:15: Show Market Analysis
"In the Market Analysis tab, you can explore historical price distributions, property type comparisons, and cross-city analytics. The dashboard provides interactive charts filtered by City and Property Type to visualize the insights from our historical dataset."

### 4:15–4:45: Show Model Evaluation
"In the 'About Model' tab, you can see our Linear Regression model outperformed Decision Trees and Random Forests, achieving an R² of 0.4885. The model successfully learned the massive geographic price premium between Mumbai and Indore."

### 4:45–5:00: Limitations and Conclusion
"The primary limitation is that our model is trained on a historical dataset of 440 records. The predicted price is an ML-based estimate and should not be treated as an official property valuation. In the future, this architecture can be scaled by integrating larger historical datasets and expanding to cover more Indian cities. Thank you."
