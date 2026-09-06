import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/processed/ml_ready_mumbai_indore_data.csv')

# 3. Prepare X and y
target = 'Price_INR'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

results = []

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    return {
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'model_obj': model
    }

# 4. Linear Regression
lr = LinearRegression()
results.append(evaluate_model('Linear Regression', lr))

# 5. Decision Tree Regression
dt = DecisionTreeRegressor(random_state=42)
results.append(evaluate_model('Decision Tree', dt))

# 6. Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
results.append(evaluate_model('Random Forest', rf))

# 7. Polynomial Regression
# Since our dataset has OHE columns, high degree polynomial features will explode the feature space.
# We will use degree=2 and interaction_only=True to prevent memory issues with OHE columns.
poly = make_pipeline(PolynomialFeatures(degree=2, interaction_only=True), LinearRegression())
results.append(evaluate_model('Polynomial Regression', poly))

# Convert results to DataFrame for comparison
df_results = pd.DataFrame([ {k:v for k,v in r.items() if k != 'model_obj'} for r in results ])
df_results = df_results.sort_values(by='R2', ascending=False)

best_model_name = df_results.iloc[0]['Model']
best_model_obj = next(r['model_obj'] for r in results if r['Model'] == best_model_name)

# 14. Save Best Model
os.makedirs('model', exist_ok=True)
model_path = 'model/best_model.pkl'
joblib.dump(best_model_obj, model_path)

# 15. Test Loaded Model
loaded_model = joblib.load(model_path)
test_pred = loaded_model.predict(X_test.iloc[[0]])
original_pred = best_model_obj.predict(X_test.iloc[[0]])
test_passed = (test_pred[0] == original_pred[0])

# Summary dict for chat
summary = {
    'results': df_results.to_dict(orient='records'),
    'best_model': best_model_name,
    'test_passed': bool(test_passed)
}

with open('model_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("Models evaluated and best model saved successfully.")
