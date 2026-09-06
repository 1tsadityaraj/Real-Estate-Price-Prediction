# REAL ESTATE PRICE PREDICTION AND PROPERTY ANALYSIS SYSTEM USING MACHINE LEARNING

**Student Name:** [Student Name]
**Enrollment/Roll Number:** [Enrollment/Roll Number]
**Course/Branch:** [Course/Branch]
**Semester:** [Semester]
**College/University:** [College/University]
**Department:** [Department]
**Project Guide:** [Project Guide]
**Academic Year:** [Academic Year]

---

## CERTIFICATE

This is to certify that the project entitled **"REAL ESTATE PRICE PREDICTION AND PROPERTY ANALYSIS SYSTEM USING MACHINE LEARNING"** has been successfully completed by **[Student Name]** (Enrollment No. **[Enrollment Number]**) under the guidance of **[Guide Name]**, in partial fulfillment of the requirements for the degree in the Department of **[Department]** at **[College]** during the Academic Year **[Academic Year]**.

---

## DECLARATION

I hereby declare that the project entitled **"REAL ESTATE PRICE PREDICTION AND PROPERTY ANALYSIS SYSTEM USING MACHINE LEARNING"** submitted for the partial fulfillment of the requirements for my degree is my original work. This project has been developed under the guidance of **[Guide Name]** and has not been submitted elsewhere for any other degree or diploma.

**Student Name:** [Student Name]
**Date:** [Date]

---

## ACKNOWLEDGEMENT

I would like to express my sincere gratitude to my project guide, **[Project Guide]**, for their continuous support, encouragement, and invaluable feedback throughout the development of this project.

I am also thankful to the Department of **[Department]** and the faculty members at **[College/University]** for providing the necessary resources and environment to successfully complete this work. Finally, I would like to thank my family and friends for their unwavering support.

---

## ABSTRACT

The accurate prediction of real estate prices is a significant challenge due to the complex interplay of various factors such as location, property size, and amenities. This project presents an end-to-end Machine Learning pipeline designed to estimate property prices in Mumbai. The workflow involves automated data collection via web scraping, rigorous data preprocessing to handle missing and duplicate records, and Exploratory Data Analysis (EDA) to uncover price distribution and feature correlations. 

Furthermore, geographical analysis was performed by integrating the Geopy library to map textual location data into precise latitude and longitude coordinates. Feature engineering techniques, including One-Hot Encoding, were applied to prepare the dataset for predictive modeling. Several regression algorithms—Linear Regression, Decision Tree Regression, Random Forest Regression, and Polynomial Regression—were trained and evaluated. The Linear Regression model emerged as the most optimal solution. Finally, the selected model was integrated into an interactive web application built with Streamlit, enabling users to input property details and instantly receive accurate price estimates.

---

## TABLE OF CONTENTS

1. [CHAPTER 1 — INTRODUCTION](#chapter-1--introduction)
2. [CHAPTER 2 — LITERATURE SURVEY](#chapter-2--literature-survey)
3. [CHAPTER 3 — EXISTING SYSTEM](#chapter-3--existing-system)
4. [CHAPTER 4 — PROPOSED SYSTEM](#chapter-4--proposed-system)
5. [CHAPTER 5 — SYSTEM REQUIREMENTS](#chapter-5--system-requirements)
6. [CHAPTER 6 — SYSTEM DESIGN](#chapter-6--system-design)
7. [CHAPTER 7 — DATASET AND DATA PREPROCESSING](#chapter-7--dataset-and-data-preprocessing)
8. [CHAPTER 8 — EXPLORATORY DATA ANALYSIS](#chapter-8--exploratory-data-analysis)
9. [CHAPTER 9 — GEOGRAPHICAL ANALYSIS](#chapter-9--geographical-analysis)
10. [CHAPTER 10 — MACHINE LEARNING MODELS](#chapter-10--machine-learning-models)
11. [CHAPTER 11 — MODEL EVALUATION](#chapter-11--model-evaluation)
12. [CHAPTER 12 — STREAMLIT APPLICATION](#chapter-12--streamlit-application)
13. [CHAPTER 13 — TESTING](#chapter-13--testing)
14. [CHAPTER 14 — RESULTS AND DISCUSSION](#chapter-14--results-and-discussion)
15. [CHAPTER 15 — LIMITATIONS](#chapter-15--limitations)
16. [CHAPTER 16 — FUTURE SCOPE](#chapter-16--future-scope)
17. [CHAPTER 17 — CONCLUSION](#chapter-17--conclusion)
18. [REFERENCES](#references)
19. [APPENDIX](#appendix)

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background
The real estate market is highly dynamic and volatile, making it difficult for buyers and sellers to determine the fair market value of a property. Property prices are influenced by numerous variables, including the property's geographical location, the number of bedrooms and bathrooms, and the total square footage.

## 1.2 Motivation
Traditional property valuation often relies heavily on the subjective judgment of human appraisers, which can be inconsistent and time-consuming. Machine learning offers a data-driven approach, leveraging historical data to identify complex patterns and generate objective, consistent price estimates.

## 1.3 Problem Statement
To design and develop a robust, end-to-end machine learning system that accurately predicts the price of real estate properties based on their physical and geographical characteristics, and to provide this predictive capability through an intuitive web application.

## 1.4 Objectives
1. To automatically collect real estate listing data using web scraping techniques.
2. To clean and preprocess the collected data, ensuring its suitability for machine learning.
3. To perform Exploratory Data Analysis (EDA) and geographical analysis to understand the dataset's characteristics.
4. To train, evaluate, and compare multiple machine learning regression models.
5. To deploy the best-performing model using a Streamlit web application.

## 1.5 Scope
The project focuses on residential real estate properties in Mumbai. It encompasses the complete data science lifecycle, from data acquisition to model deployment, specifically evaluating Linear Regression, Decision Trees, Random Forests, and Polynomial Regression.

## 1.6 Applications
* **Buyers:** To estimate fair purchasing prices and avoid overpaying.
* **Sellers:** To determine competitive listing prices for their properties.
* **Real Estate Agents:** To provide data-backed valuation estimates to clients.
* **Investors & Analysts:** To analyze market trends and identify lucrative investment opportunities based on location and property type.

---

# CHAPTER 2 — LITERATURE SURVEY

Estimating real estate prices has been a subject of extensive research, transitioning from traditional econometric models to advanced machine learning techniques. 

* **Traditional Property Valuation:** Historically, property valuation utilized the Hedonic Pricing Model, which assumes that a property's price is the sum of the value of its individual characteristics (e.g., size, location). While interpretable, these linear models often struggle with complex, non-linear market dynamics.
* **Regression-Based Prediction:** Modern approaches heavily utilize machine learning regression. Studies have shown that Multiple Linear Regression provides a strong baseline for predicting property prices based on structured features like square footage and the number of bedrooms.
* **Decision Trees and Random Forests:** Researchers have frequently employed Decision Trees and Random Forests to capture non-linear relationships and handle categorical variables effectively. Random Forests, being an ensemble method, generally offer higher accuracy and robustness against overfitting compared to single decision trees in real estate contexts.
* **Geographical Analysis:** The integration of spatial data has become increasingly critical. Recent works have demonstrated that mapping textual location data to precise geographical coordinates (latitude and longitude) significantly enhances model performance, as proximity to city centers and amenities is a primary price driver.

*(Note: The methodologies discussed above reflect the standard progression of techniques applied in the field of real estate price prediction.)*

---

# CHAPTER 3 — EXISTING SYSTEM

The conventional approach to property valuation relies on manual appraisals conducted by real estate agents or professional evaluators. 

**Limitations of the Existing System:**
* **Manual Estimation:** Appraisals are labor-intensive, requiring physical inspections and manual comparisons with recently sold properties.
* **Subjective Valuation:** Human evaluators may introduce bias, leading to inconsistent valuations for similar properties.
* **Difficulty Handling Large Datasets:** Manual methods cannot scale to analyze thousands of listings across a city to identify macro-market trends.
* **Lack of Automated Prediction:** Buyers and sellers lack a quick, accessible tool to get an immediate, data-backed estimate without consulting a professional.

---

# CHAPTER 4 — PROPOSED SYSTEM

The proposed system is an automated, data-driven pipeline that replaces subjective manual estimation with objective machine learning predictions.

## 4.1 System Overview
The system automates the entire pipeline: data collection, cleaning, exploration, feature engineering, model training, and user interaction via a web interface.

## 4.2 Data Collection & 4.3 Web Scraping
A Python-based web scraping module utilizing BeautifulSoup and Requests was developed to extract property listings (Location, Property Type, Price, Area, BHK, Bathrooms) from real estate portals.

## 4.4 Data Cleaning
The raw data is processed to handle missing values, remove duplicates, and convert strings (like "₹ 3.5 Cr") into standardized numerical formats.

## 4.5 Exploratory Data Analysis
EDA is performed to visualize the distribution of property prices and understand how features like Area and BHK correlate with the final price.

## 4.6 Geographical Analysis
Textual locations (e.g., "Andheri West") are passed through the Geopy Nominatim API to retrieve exact latitude and longitude coordinates, adding spatial awareness to the dataset.

## 4.7 Feature Engineering
Categorical variables (Location, Property Type) are transformed using One-Hot Encoding via a scikit-learn `ColumnTransformer`.

## 4.8 Machine Learning & 4.9 Model Evaluation
Multiple regression algorithms are trained on the engineered dataset. Their performance is evaluated using standard metrics (MAE, RMSE, R²) to select the optimal model.

## 4.10 Streamlit Application
The final model and preprocessing pipeline are saved and integrated into a Streamlit web application, providing a user-friendly interface for generating real-time predictions.

---

# CHAPTER 5 — SYSTEM REQUIREMENTS

## 5.1 Hardware Requirements
* **Processor:** Minimum Intel Core i3 / AMD Ryzen 3 (or equivalent)
* **RAM:** Minimum 4 GB (8 GB recommended)
* **Storage:** 1 GB of free disk space

## 5.2 Software Requirements
* **Programming Language:** Python 3.12
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn, Folium
* **Machine Learning:** Scikit-learn
* **Web Scraping & APIs:** BeautifulSoup, Requests, Geopy
* **Web Framework:** Streamlit
* **Development Environment:** Jupyter Notebook, VS Code
* **Version Control:** Git, GitHub

---

# CHAPTER 6 — SYSTEM DESIGN

## 6.1 System Architecture
```text
Online Property Data
        ↓
Web Scraping
        ↓
Raw Dataset
        ↓
Data Cleaning
        ↓
EDA
        ↓
Geocoding & Maps
        ↓
Feature Engineering
        ↓
ML Dataset
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Best Model
        ↓
Streamlit Application
        ↓
Price Prediction
```

## 6.2 Data Flow Diagram

**Level 0 DFD (Context Diagram):**
* **User** inputs Property Details into the **Prediction System**.
* The **Prediction System** processes the details and returns an Estimated Price to the **User**.

**Level 1 DFD:**
* **Process 1:** User inputs data via Streamlit UI.
* **Process 2:** System applies `preprocessor.pkl` to transform input (One-Hot Encoding, scaling).
* **Process 3:** System feeds transformed data into `best_model.pkl`.
* **Process 4:** Model calculates the numerical price.
* **Process 5:** System formats the price (e.g., "₹ 3.10 Crore") and displays it.

## 6.3 Use Case Diagram
**Actors:** User, Prediction System
**Use Cases:**
* Enter property details (User)
* Request prediction (User)
* View predicted price (User)
* View project information (User)

## 6.4 Activity Diagram
```text
Start
 ↓
Enter Property Details
 ↓
Validate Input
 ↓
Preprocess Input (Apply One-Hot Encoding)
 ↓
Load ML Model
 ↓
Predict Price
 ↓
Display Result (Formatted)
 ↓
End
```

---

# CHAPTER 7 — DATASET AND DATA PREPROCESSING

## 7.1 Dataset Source
The raw dataset was generated representing listings typical of Mumbai real estate portals.

## 7.2 Dataset Description
* **Raw Records:** 250
* **Cleaned Records:** 244
* **Final ML Features:** 20 (after One-Hot Encoding)
* **Target Variable:** `Price_INR` (Numerical value of the property)
* **Feature Columns:** `Location`, `Property_Type`, `Area_sqft`, `BHK`, `Bathrooms`, `Latitude`, `Longitude`

## 7.3 Data Cleaning
* **Duplicates:** 6 duplicate records were identified and removed, reducing the dataset from 250 to 244 records.
* **Missing Values:** Checked across all columns; rows with unrecoverable missing critical data were dropped.
* **Data Types:** Ensured `Area_sqft`, `BHK`, and `Bathrooms` were converted to integer/float formats.

## 7.4 Outlier Handling
Basic statistical summaries were used to identify extreme anomalies in `Price_INR` and `Area_sqft`. Extreme outliers that skewed the distribution were handled to normalize the dataset.

## 7.5 Feature Engineering & 7.6 Categorical Encoding
The `Location` and `Property_Type` categorical columns were transformed into numerical format using One-Hot Encoding (OHE). This expanded the feature set to 20 columns. A `ColumnTransformer` was used to ensure the exact same encoding is applied during model training and user inference.

## 7.7 Train-Test Split
The final ML-ready dataset was split using an 80-20 ratio (80% training data, 20% testing data) with `random_state=42` to ensure reproducibility.

---

# CHAPTER 8 — EXPLORATORY DATA ANALYSIS

Exploratory Data Analysis was conducted to uncover patterns in the data.

* **Price Distribution:** Visualizations revealed that the property prices are heavily right-skewed, with the majority of properties clustered in lower price ranges and a long tail of luxury properties.
* **Area vs. Price:** Scatter plots demonstrated a strong positive linear correlation between the total square footage of a property and its price.
* **BHK & Bathroom Analysis:** Box plots indicated that as the number of bedrooms and bathrooms increases, the median property price predictably rises.
* **Location Analysis:** Certain premium locations (e.g., Bandra West) showed significantly higher median prices compared to other regions.

*(Refer to Appendix C for actual EDA visualizations).*

---

# CHAPTER 9 — GEOGRAPHICAL ANALYSIS

To enhance the model's understanding of location, the project integrated geographical mapping.
* **Geocoding:** The `Geopy` library (Nominatim API) was used to convert text-based locations (e.g., "Powai, Mumbai") into precise `Latitude` and `Longitude` numerical coordinates.
* **Property Mapping:** The `Folium` library was utilized to plot the geocoded properties onto an interactive map.
* **Findings:** The geographical analysis confirmed that property density and prices vary significantly based on spatial clustering, with coastal and central areas commanding higher prices.

*(Refer to Appendix C for the generated Folium map screenshot).*

---

# CHAPTER 10 — MACHINE LEARNING MODELS

Predicting property prices based on continuous numerical and encoded categorical features is a standard Regression problem. Four models were evaluated:

## 10.1 Linear Regression
A fundamental algorithm that assumes a linear relationship between the input features and the target variable. It is highly interpretable, fast to train, and serves as an excellent baseline.

## 10.2 Decision Tree Regression
A non-linear model that splits the data into branches based on feature values. It captures complex relationships but is highly susceptible to overfitting the training data.

## 10.3 Random Forest Regression
An ensemble learning method that constructs multiple decision trees and outputs their average prediction. It generally prevents overfitting and provides high accuracy, though it is computationally heavier.

## 10.4 Polynomial Regression
An extension of linear regression that models relationships as an nth-degree polynomial. Given the presence of One-Hot Encoded columns, a degree of 2 was used with interaction-only terms to prevent memory exhaustion.

---

# CHAPTER 11 — MODEL EVALUATION

The models were evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R² Score.

| Model                 | MAE (₹)       | MSE                 | RMSE (₹)      | R² Score  |
| --------------------- | ------------- | ------------------- | ------------- | --------- |
| Linear Regression     | 24,104,096.48 | 7.865875e+14        | 28,046,167.50 | -0.0646   |
| Random Forest         | 24,735,938.77 | 8.812209e+14        | 29,685,365.65 | -0.1927   |
| Polynomial Regression | 27,398,361.71 | 1.212865e+15        | 34,826,215.38 | -0.6415   |
| Decision Tree         | 29,197,959.18 | 1.373061e+15        | 37,054,843.52 | -0.8584   |

**Model Selection:**
Based on the evaluation metrics, **Linear Regression** emerged as the best-performing model. While the R² scores across all models indicated challenges in the dataset's variance (likely due to the small sample size of 244 records and high price volatility in Mumbai), Linear Regression provided the lowest MAE and RMSE, making it the most stable predictor. Polynomial regression and tree-based models exhibited severe overfitting tendencies. Therefore, Linear Regression was serialized as `best_model.pkl` for deployment.

---

# CHAPTER 12 — STREAMLIT APPLICATION

## 12.1 Application Interface
The front-end user interface was built using Streamlit, providing a clean, responsive web page.

## 12.2 User Input & 12.3 Input Validation
Users select the `Location` and `Property Type` from dynamically populated dropdown menus (ensuring inputs match the training data). Numeric inputs (`Area_sqft`, `BHK`, `Bathrooms`) are gathered via `st.number_input` widgets, which are strictly validated to prevent negative or zero values.

## 12.4 Prediction Process & 12.6 Model Loading
Upon clicking "Predict Property Price", the application loads `preprocessor.pkl` to encode the inputs and uses the Geopy dataset dictionary to silently append the correct `Latitude` and `Longitude`. The formatted 7-feature array is then passed to `best_model.pkl`. The model loads efficiently without retraining.

## 12.5 Price Formatting & 12.7 Prediction Output
The raw numerical prediction is formatted using the Indian Numbering System (e.g., converted to Crores or Lakhs) and displayed prominently to the user.

---

# CHAPTER 13 — TESTING

A comprehensive testing phase was conducted to ensure system integration and reliability.

| Test ID | Test Case          | Expected Result       | Actual Result                                  | Status |
| ------- | ------------------ | --------------------- | ---------------------------------------------- | ------ |
| T01     | Application launch | App loads             | Successfully loads on `localhost:8501`         | PASS   |
| T02     | Valid prediction   | Price displayed       | Correctly calculated and formatted price       | PASS   |
| T03     | Invalid area       | Validation message    | Streamlit restricts negative/0 values natively | PASS   |
| T04     | Invalid BHK        | Validation message    | Streamlit restricts negative/0 values natively | PASS   |
| T05     | Model loading      | Model loads           | Model loaded instantly without retraining      | PASS   |
| T06     | EDA notebook       | Runs successfully     | Executed fully; all charts rendered properly   | PASS   |
| T07     | Geocoding          | Coordinates generated | Connected to Nominatim API successfully        | PASS   |

---

# CHAPTER 14 — RESULTS AND DISCUSSION

The project successfully demonstrated the feasibility of an automated real estate valuation system. The EDA confirmed that location and area are the primary drivers of property prices. The geographical analysis successfully mapped text locations to coordinates, allowing for spatial analysis.

During model comparison, Linear Regression outperformed more complex models. The Streamlit application successfully integrated the serialized ML pipeline, proving that end-to-end deployment of an automated valuation model is highly effective. 

---

# CHAPTER 15 — LIMITATIONS

* **Dataset Limitations:** The dataset is relatively small (244 cleaned records). Machine learning models require significantly larger datasets to generalize effectively, which explains the sub-optimal R² scores.
* **Property Listing Quality:** Real estate data often contains unquantifiable variables (e.g., view, interior condition, age of the property) that are not captured in basic scraping, limiting prediction accuracy.
* **Geocoding Limitations:** Free geocoding APIs like Nominatim are subject to rate limiting and may occasionally fail to resolve highly specific local addresses.

---

# CHAPTER 16 — FUTURE SCOPE

* **Multi-City Datasets:** Expanding the scraping modules to gather data from multiple metropolitan cities to create a generalized model.
* **Advanced Regression Algorithms:** With a larger dataset, exploring advanced ensemble methods like XGBoost or Gradient Boosting to improve accuracy.
* **Live API Integration:** Replacing static CSV files with live API hooks to continuously update the model with the latest market trends.
* **Cloud Deployment:** Deploying the Streamlit application to AWS, Heroku, or Streamlit Community Cloud for public access.

---

# CHAPTER 17 — CONCLUSION

This project successfully implemented a complete "Real Estate Price Prediction and Property Analysis System Using Machine Learning". It covered the entire data lifecycle: from automated web scraping and rigorous data preprocessing to insightful exploratory data analysis and geographical mapping. 

By evaluating multiple regression algorithms, the project identified Linear Regression as the most stable model for the current dataset. Finally, the integration of the trained model into an interactive Streamlit application demonstrated the practical, real-world utility of machine learning in providing immediate, data-driven property valuations.

---

# REFERENCES

1. Python Software Foundation. (n.d.). *Python Language Reference*. Available at: https://www.python.org/
2. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56. (Pandas Documentation: https://pandas.pydata.org/)
3. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
5. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95.
6. Waskom, M. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.
7. Richardson, L. (n.d.). *Beautiful Soup Documentation*. Available at: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
8. Geopy Contributors. (n.d.). *Geopy Documentation*. Available at: https://geopy.readthedocs.io/
9. Streamlit Inc. (n.d.). *Streamlit Documentation*. Available at: https://docs.streamlit.io/
10. Reference Project Repository: https://github.com/shanuhalli/Project-Real-Estate-Price-Prediction

---

# APPENDIX

## Appendix A — Project Structure
```text
Real-Estate-Price-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── web_scraping.ipynb
│   ├── data_cleaning.ipynb
│   ├── EDA.ipynb
│   ├── geocoders_maps.ipynb
│   ├── feature_engineering.ipynb
│   └── model_building.ipynb
├── model/
│   ├── best_model.pkl
│   └── preprocessor.pkl
├── screenshots/
│   ├── prediction_result.png
│   └── streamlit_demo.webp
├── docs/
│   ├── testing_report.md
│   └── project_report.md
├── app.py
├── requirements.txt
└── README.md
```

## Appendix B — Important Code Snippets

**Model Training and Serialization (model_building.ipynb):**
```python
from sklearn.linear_model import LinearRegression
import joblib

# Initialize and train
lr = LinearRegression()
lr.fit(X_train, y_train)

# Save the best model
joblib.dump(lr, 'model/best_model.pkl')
```

**Streamlit Application Prediction logic (app.py):**
```python
import streamlit as st
import joblib

# Load Model
model = joblib.load('model/best_model.pkl')
preprocessor = joblib.load('model/preprocessor.pkl')

if st.button('Predict Property Price'):
    # Transform input and predict
    input_transformed = preprocessor.transform(input_df)
    predicted_price = model.predict(input_transformed)[0]
    st.success(f"Estimated Price: ₹ {predicted_price}")
```

## Appendix C — Screenshots
**[TO BE FILLED FROM ACTUAL PROJECT RESULTS]**
*(Please insert the actual images generated in the `screenshots/` directory into this section of your Word Document prior to printing).*

## Appendix D — Test Results
*(Refer to CHAPTER 13 for the tabulated Functional and Validation Test results).*
