# Real Estate Price Prediction and Property Analysis System Using Machine Learning

## Objective
This project is a college major project aimed at building an end-to-end Machine Learning pipeline to predict real estate prices and provide an interactive property analysis system. It covers data collection (web scraping), data cleaning, exploratory data analysis (EDA), feature engineering, model building, and deployment using a web interface.

## Technologies Used
- **Python**: Core programming language.
- **Jupyter Notebook**: For interactive data exploration and model building.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning algorithms and evaluation.
- **BeautifulSoup & Requests**: For web scraping property data.
- **Geopy**: For geographical data processing (geocoding).
- **Streamlit**: For building the interactive web application.
- **Git/GitHub**: For version control.

## Planned Workflow
1. **Web Scraping (`notebooks/web_scraping.ipynb`)**: Collect real estate property data from websites.
2. **Data Cleaning (`notebooks/data_cleaning.ipynb`)**: Handle missing values, outliers, and incorrect data types.
3. **Exploratory Data Analysis (`notebooks/EDA.ipynb`)**: Analyze the data distribution, uncover patterns, and find correlations.
4. **Geocoders & Maps (`notebooks/geocoders_maps.ipynb`)**: Extract geographical features and visualize properties on maps.
5. **Model Building (`notebooks/model_building.ipynb`)**: Train and evaluate machine learning models for price prediction.
6. **Deployment (`app.py`)**: Create a Streamlit web app to interact with the trained model and view property analysis.

## Folder Structure
- `data/raw/`: Contains the original, unmodified scraped data.
- `data/processed/`: Contains the cleaned and transformed data ready for modeling.
- `notebooks/`: Contains Jupyter Notebooks for each step of the pipeline (scraping, cleaning, EDA, geocoding, modeling).
- `model/`: Stores the trained machine learning models (e.g., pickle files).
- `app.py`: The main script for the Streamlit web application.
- `requirements.txt`: Lists all Python dependencies required to run the project.
- `README.md`: Project documentation.
- `.gitignore`: Specifies intentionally untracked files to ignore (like data files, virtual environments, and model files).
