import pandas as pd




# ==============================================================
# 1. Load and process the tornado dataset (1950-2024)
# ==============================================================
tornado_df = pd.read_csv('1950-2024_all_tornadoes.csv')   # ← your exact file

# Convert date and extract year
tornado_df['date'] = pd.to_datetime(tornado_df['date'], errors='coerce')
tornado_df['year'] = tornado_df['date'].dt.year

# Clean magnitude (-9 = unknown → treat as missing)
tornado_df['mag_clean'] = tornado_df['mag'].replace(-9, pd.NA)

# Annual aggregation
annual_tornadoes = tornado_df.groupby('year').agg(
    tornado_count=('mag', 'size'),                    # total tornadoes (including -9)
    tornadoes_reported=('mag_clean', 'count'),        # excludes unknown magnitude
    total_intensity=('mag_clean', 'sum'),             # sum of F/EF ratings
    mean_intensity=('mag_clean', 'mean')
).reset_index()

print("Tornado years processed:", annual_tornadoes['year'].min(), "–", annual_tornadoes['year'].max())
print(annual_tornadoes.tail(6))   # shows 2019–2024


# ==============================================================
# 2. Load and create annual ENSO (ONI) from your ENSO.csv
# ==============================================================
enso_df = pd.read_csv('ENSO.csv')

# The ONI column is already the official 3-month running mean of Niño 3.4
# We'll use the **Mar–May (MAM)** ONI → best predictor for U.S. spring tornado season
mam_oni = enso_df[enso_df['Season (3-Month)'] == 'MAM'].copy()
mam_oni = mam_oni[['Year', 'ONI']].rename(columns={'Year': 'year', 'ONI': 'oni_mam'})

# Convert to numeric and drop any bad rows
mam_oni['oni_mam'] = pd.to_numeric(mam_oni['oni_mam'], errors='coerce')

print("\nMAM ONI sample:")
print(mam_oni.tail(8))


# ==============================================================
# 3. Load global temperature anomalies (from your ENSO.csv – already there!)
# ==============================================================
# Your ENSO.csv already has monthly global temp anomalies → just average per year
annual_temp = enso_df.groupby('Year')['Global Temperature Anomalies'].mean().reset_index()
annual_temp = annual_temp.rename(columns={'Year': 'year', 'Global Temperature Anomalies': 'global_temp_anomaly'})


# ==============================================================
# 4. Merge everything on 'year'
# ==============================================================
merged_df = (annual_tornadoes
             .merge(mam_oni, on='year', how='inner')
             .merge(annual_temp, on='year', how='inner'))

# Drop any accidental missing years (shouldn’t happen 1950-2024)
merged_df = merged_df.dropna().reset_index(drop=True)

print("\nFinal merged dataset (1950–2024):")
print(f"Rows: {len(merged_df)} → years {merged_df['year'].min()}–{merged_df['year'].max()}")
print(merged_df[['year', 'tornado_count', 'oni_mam', 'global_temp_anomaly']].tail(10))

# ==============================================================
# 5. Save clean dataset for polynomial regression
# ==============================================================
merged_df.to_csv('tornado_enso_temp_1950_2024_ready.csv', index=False)
print("\nSaved → tornado_enso_temp_1950_2024_ready.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Features and target
X = merged_df[['global_temp_anomaly', 'oni_mam']]
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

print("Features:", poly.get_feature_names_out(['global_temp_anomaly', 'oni_mam']))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

new_data = pd.DataFrame({'global_temp_anomaly': [1.48], 'oni_mam': [-0.2]})
new_poly = poly.transform(new_data)
prediction = model.predict(new_poly)
print("Predicted tornado count:", prediction[0])

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred_test)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Your existing plot
plt.scatter(y_test, y_pred_test)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')

# Add regression line through the scatter points
z = np.polyfit(y_test, y_pred_test, 1)  # Fit 1st degree polynomial (line)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "b-", linewidth=2, label='Regression Line')

plt.xlabel('Actual Tornado Count')
plt.ylabel('Predicted Tornado Count')
plt.legend()
plt.title('Model Predictions vs Actual Values')
plt.show()