#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Model Building
# In this step, we train and evaluate multiple regression algorithms on our engineered real estate dataset to determine which model is best suited for predicting house prices.

# ### Why Regression?
# House price prediction is fundamentally a regression problem because our target variable (`Price_INR`) is a continuous numerical value rather than a discrete category.
# 
# ### Why Multiple Models?
# No single algorithm works best for every dataset (the 'No Free Lunch' theorem). We compare Linear models against Tree-based models to see which captures the underlying geographic and structural patterns more effectively.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# ## 1, 2, & 3. Load Prepared Data and Split X/y
# We load the clean, one-hot encoded dataset.

# In[2]:


df = pd.read_csv('../data/processed/ml_ready_mumbai_indore_data.csv')

target = 'Price_INR'
X = df.drop(columns=[target])
y = df[target]

print(f"Target variable: {target}")
print(f"Features used: {list(X.columns)}")
print(f"Dataset shape: {df.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")


# ## 4. Linear Regression

# In[3]:


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)


# ## 5. Decision Tree Regression

# In[4]:


dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)


# ## 6. Random Forest Regression

# In[5]:


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# ## 7. Polynomial Regression
# Using degree=2. To avoid massive memory explosion caused by squaring One-Hot Encoded categorical variables, we restrict the expansion to interaction terms.

# In[6]:


poly = make_pipeline(PolynomialFeatures(degree=2, interaction_only=True), LinearRegression())
poly.fit(X_train, y_train)
poly_pred = poly.predict(X_test)


# ## 8 & 13. Model Evaluation and Cross-Validation
# We calculate standard error metrics (MAE, RMSE) and R2 (Accuracy) across all models, supplemented by 5-fold cross-validation.

# In[7]:


def eval_model(model_name, y_true, y_pred, model_obj):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    cv_scores = cross_val_score(model_obj, X, y, cv=5, scoring='r2')
    return {'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'CV_R2_Mean': cv_scores.mean()}

results = []
results.append(eval_model('Linear Regression', y_test, lr_pred, lr))
results.append(eval_model('Decision Tree', y_test, dt_pred, dt))
results.append(eval_model('Random Forest', y_test, rf_pred, rf))
results.append(eval_model('Polynomial Regression', y_test, poly_pred, poly))

df_results = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print(df_results)


# ### Note on Results & Limitations:
# *(Since our dataset is comprised of completely synthetic, randomized patterns for demonstration purposes, the variables share zero true mathematical correlation with the target price. This naturally results in negative R2 scores for all algorithms. In a real-world scenario with legitimate housing data, tree-based models typically exhibit high positive R2 scores here).* 

# ## 9. Model Comparison Visualizations

# In[8]:


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# R2 Comparison
sns.barplot(data=df_results, x='R2', y='Model', ax=axes[0], palette='Blues_r')
axes[0].set_title('R² Score Comparison (Higher is Better)')
axes[0].set_xlabel('R² Score')

# RMSE Comparison
sns.barplot(data=df_results, x='RMSE', y='Model', ax=axes[1], palette='Reds_r')
axes[1].set_title('RMSE Comparison (Lower is Better)')
axes[1].set_xlabel('Root Mean Squared Error')

# plt.tight_layout()
# plt.show()


# ## 10. Actual Model Selection
# **Best Model:** Linear Regression  
# **Reason:** Based purely on empirical evidence from our synthetic dataset execution, Linear Regression yielded the highest R2 score and lowest RMSE/MAE compared to the tree-based alternatives.

# ## 11. Prediction vs Actual Visualization (Best Model)

# In[9]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=lr_pred, alpha=0.7, color='indigo')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
# plt.show()


# ## 12. Residual Analysis
# Analyzing prediction errors for our best model.

# In[10]:


residuals = y_test - lr_pred

plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Distribution of Residuals')
plt.xlabel('Error (Actual - Predicted)')
# plt.show()


# **Insight:** The residuals are broadly normally distributed around 0, which suggests the Linear model is not systematically overestimating or underestimating, despite the high total variance.

# ## 14 & 15. Save and Test the Best Model

# In[11]:


best_model_obj = lr
model_path = '../model/best_model.pkl'
joblib.dump(best_model_obj, model_path)
print(f"Best model saved to: {model_path}")

# Reload and Test
loaded_model = joblib.load(model_path)
sample_input = X_test.iloc[[0]]
prediction = loaded_model.predict(sample_input)

print(f"Test successful. Loaded model predicted: ₹{prediction[0]:,.2f}")


# ## 16. Final Model Performance Summary
# | Model                 |    MAE |   RMSE |     R² |
# | --------------------- | -----: | -----: | -----: |
# | Linear Regression     | 1.21e7 | 1.41e7 | -0.237 |
# | Random Forest         | 1.27e7 | 1.49e7 | -0.384 |
# | Decision Tree         | 1.39e7 | 1.73e7 | -0.852 |
# | Polynomial Regression | 1.54e7 | 1.91e7 | -1.273 |
