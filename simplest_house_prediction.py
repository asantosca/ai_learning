import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    'Bedrooms': [3, 4, 1, 4, 4, 2, 3, 1, 3, 5, 1, 2, 2, 3, 4, 3, 1, 5, 3, 3],
    'Age':[31,  3,  9, 42, 49, 27, 33,  7, 47, 22, 30, 17, 41, 11, 35, 15,  7, 9, 29, 10],
    'GarageCars': [1,      3,      2, 4,      4,      4,      4,      2,      2,      4,      3,      4,      1,      1,      3,      4,      4,      4,      1,      3],
    'Price': [190762, 309103, 268307, 428540, 414703, 433675, 479893, 265504, 257242, 496057, 368060, 465254, 155207, 124690, 309461, 473884, 477587, 431282, 196605, 373674]
}

print(data)
df = pd.DataFrame(data)

# Step 1: Define features and target
X = df.drop(columns=['Price'])  # Features
y = df['Price']  # Target

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = linear_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Performance:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Step 5: Display Feature Contributions
print("\nFeature Contributions (Coefficients):")
feature_contributions = {}
for feature, coef in zip(X.columns, linear_model.coef_):
    feature_contributions[feature] = coef
    print(f"{feature}: {coef}")

# Step 6: Calculate Individual Contributions for Each Test Instance
contributions_df = X_test.copy()
for feature in X.columns:
    contributions_df[f"{feature}_contribution"] = contributions_df[feature] * feature_contributions[feature]

# Add the intercept to contributions
contributions_df['Intercept'] = linear_model.intercept_

# Calculate Total Predicted Price for verification
contributions_df['Predicted_Price'] = contributions_df[[f"{feature}_contribution" for feature in X.columns]].sum(axis=1) + contributions_df['Intercept']

print("\nIndividual Feature Contributions for Test Data:")
print(contributions_df.head())

# Step 7: Optionally Train Ridge or Lasso for comparison
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

print("\nRidge and Lasso Feature Contributions:")
print("Ridge Coefficients:", ridge_model.coef_)
print("Lasso Coefficients:", lasso_model.coef_)
