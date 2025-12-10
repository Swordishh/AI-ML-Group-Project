from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Features and target
X = merged_df[['avg_temp_anomaly', 'avg_enso_index']]
y = merged_df['tornado_count']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial transform (degree=2; adjust as needed)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict and evaluate
y_pred_train = model.predict(X_train_poly)
y_pred_test = model.predict(X_test_poly)

print("Train MSE:", mean_squared_error(y_train, y_pred_train))
print("Train R2:", r2_score(y_train, y_pred_train))
print("Test MSE:", mean_squared_error(y_test, y_pred_test))
print("Test R2:", r2_score(y_test, y_pred_test))