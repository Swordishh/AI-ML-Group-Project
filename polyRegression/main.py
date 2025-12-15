import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==============================================================
# 1. Load and process the tornado dataset (1950-2024)
# ==============================================================
tornado_df = pd.read_csv('1950-2024_all_tornadoes.csv')

# Convert date and extract year and month
tornado_df['date'] = pd.to_datetime(tornado_df['date'], errors='coerce')
tornado_df['year'] = tornado_df['date'].dt.year
tornado_df['month'] = tornado_df['date'].dt.month

# Clean magnitude (-9 = unknown → treat as missing)
tornado_df['mag_clean'] = tornado_df['mag'].replace(-9, pd.NA)

# Filter for MAM (March-April-May) season - peak tornado months
mam_tornadoes = tornado_df[tornado_df['month'].isin([3, 4, 5])].copy()

# Annual aggregation for full year (for comparison)
annual_tornadoes = tornado_df.groupby('year').agg(
    tornado_count_annual=('mag', 'size'),
    tornadoes_reported_annual=('mag_clean', 'count'),
    total_intensity_annual=('mag_clean', 'sum'),
    mean_intensity_annual=('mag_clean', 'mean')
).reset_index()

# MAM season aggregation (primary target)
mam_aggregated = mam_tornadoes.groupby('year').agg(
    tornado_count_mam=('mag', 'size'),
    tornadoes_reported_mam=('mag_clean', 'count'),
    total_intensity_mam=('mag_clean', 'sum'),
    mean_intensity_mam=('mag_clean', 'mean')
).reset_index()

print("Tornado years processed:", tornado_df['year'].min(), "–", tornado_df['year'].max())
print("\nMAM Season Tornado Statistics:")
print(mam_aggregated.tail(6))

# ==============================================================
# 2. Load and create seasonal ENSO (ONI) from ENSO.csv
# ==============================================================
enso_df = pd.read_csv('ENSO.csv')

# Use MAR-MAY (MAM) ONI → best predictor for U.S. spring tornado season
mam_oni = enso_df[enso_df['Season (3-Month)'] == 'MAM'].copy()
mam_oni = mam_oni[['Year', 'ONI']].rename(columns={'Year': 'year', 'ONI': 'oni_mam'})
mam_oni['oni_mam'] = pd.to_numeric(mam_oni['oni_mam'], errors='coerce')

print("\nMAM ONI sample:")
print(mam_oni.tail(8))

# ==============================================================
# 3. Load seasonal temperature anomalies (MAM focus)
# ==============================================================
# Extract MAM temperature anomalies from your ENSO.csv
mam_temp = enso_df[enso_df['Season (3-Month)'] == 'MAM'].copy()
mam_temp = mam_temp[['Year', 'Global Temperature Anomalies']].rename(
    columns={'Year': 'year', 'Global Temperature Anomalies': 'temp_anomaly_mam'}
)
mam_temp['temp_anomaly_mam'] = pd.to_numeric(mam_temp['temp_anomaly_mam'], errors='coerce')

print("\nMAM Temperature Anomaly sample:")
print(mam_temp.tail(8))

# ==============================================================
# 4. Load and process Precipitable Water data
# ==============================================================
precip_water_df = pd.read_csv('file1iCLNidsarg.csv')

# Clean column name (remove extra spaces)
precip_water_df.columns = precip_water_df.columns.str.strip()

# Parse date and extract year and month
precip_water_df['date'] = pd.to_datetime(precip_water_df['Date'])
precip_water_df['year'] = precip_water_df['date'].dt.year
precip_water_df['month'] = precip_water_df['date'].dt.month

# Rename the precipitable water column for easier handling
precip_water_df = precip_water_df.rename(columns={
    'NCEP/NCAR R1 Precipitable Water (kg/m^2) 30N-40N;-95E--105E': 'precip_water'
})

# Filter for MAM months (March=3, April=4, May=5)
mam_precip = precip_water_df[precip_water_df['month'].isin([3, 4, 5])].copy()

# Calculate MAM average precipitable water for each year
mam_precip_avg = mam_precip.groupby('year').agg(
    precip_water_mam=('precip_water', 'mean')
).reset_index()

print("\nMAM Precipitable Water sample:")
print(mam_precip_avg.tail(8))

# ==============================================================
# 5. Merge everything on 'year'
# ==============================================================
merged_df = (mam_aggregated
             .merge(annual_tornadoes, on='year', how='inner')
             .merge(mam_oni, on='year', how='inner')
             .merge(mam_temp, on='year', how='inner')
             .merge(mam_precip_avg, on='year', how='inner'))

merged_df = merged_df.dropna().reset_index(drop=True)

print("\nFinal merged dataset (1950–2024):")
print(f"Rows: {len(merged_df)} → years {merged_df['year'].min()}–{merged_df['year'].max()}")
print("\nKey variables for MAM season:")
print(merged_df[['year', 'tornado_count_mam', 'oni_mam', 'temp_anomaly_mam', 'precip_water_mam']].tail(10))

# ==============================================================
# 6. Save clean dataset
# ==============================================================
merged_df.to_csv('tornado_mam_seasonal_with_precip_1950_2024.csv', index=False)
print("\nSaved → tornado_mam_seasonal_with_precip_1950_2024.csv")

# ==============================================================
# 7. Build Linear and Polynomial Regression Models
# ==============================================================
# Features and target (NOW INCLUDING PRECIPITABLE WATER)
X = merged_df[['temp_anomaly_mam', 'oni_mam', 'precip_water_mam']]
y = merged_df['tornado_count_mam']  # Predicting MAM tornado counts

# Time-based split: Test on 2010-2023, train on all years before 2010
# Reserve 2024 for final prediction accuracy test
test_start_year = 2010
test_end_year = 2023

train_mask = merged_df['year'] < test_start_year
test_mask = (merged_df['year'] >= test_start_year) & (merged_df['year'] <= test_end_year)

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

train_years = merged_df.loc[train_mask, 'year']
test_years = merged_df.loc[test_mask, 'year']

print(f"\nTrain period: {train_years.min()}-{train_years.max()} ({len(X_train)} years)")
print(f"Test period: {test_years.min()}-{test_years.max()} ({len(X_test)} years)")
print(f"Prediction year: 2024 (held out for accuracy assessment)")
print(f"\nPredicting: MAM (March-April-May) tornado counts")
print(f"Predictor variables: Temperature Anomaly, ONI, Precipitable Water")

# --- LINEAR REGRESSION MODEL ---
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear_train = linear_model.predict(X_train)
y_pred_linear_test = linear_model.predict(X_test)

# --- POLYNOMIAL REGRESSION MODEL (degree=2) ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly_train = poly_model.predict(X_train_poly)
y_pred_poly_test = poly_model.predict(X_test_poly)

# --- MODEL COMPARISON ---
print("\n" + "=" * 60)
print("LINEAR vs POLYNOMIAL REGRESSION COMPARISON (MAM Season)")
print("=" * 60)

print("\nLINEAR MODEL:")
print("-" * 40)
print("  Train MSE:", mean_squared_error(y_train, y_pred_linear_train))
print("  Train R²:", r2_score(y_train, y_pred_linear_train))
print("  Test MSE:", mean_squared_error(y_test, y_pred_linear_test))
print("  Test R²:", r2_score(y_test, y_pred_linear_test))
print("  Coefficients:", linear_model.coef_)
print("    [temp_anomaly_mam, oni_mam, precip_water_mam]")
print("  Intercept:", linear_model.intercept_)

print("\nPOLYNOMIAL MODEL (degree=2):")
print("-" * 40)
print("  Train MSE:", mean_squared_error(y_train, y_pred_poly_train))
print("  Train R²:", r2_score(y_train, y_pred_poly_train))
print("  Test MSE:", mean_squared_error(y_test, y_pred_poly_test))
print("  Test R²:", r2_score(y_test, y_pred_poly_test))
print("  Features:", poly.get_feature_names_out(['temp_anomaly_mam', 'oni_mam', 'precip_water_mam']))
print("  Number of coefficients:", len(poly_model.coef_))
print("  Intercept:", poly_model.intercept_)

# Determine which model is better
r2_diff = r2_score(y_test, y_pred_poly_test) - r2_score(y_test, y_pred_linear_test)
print("\n" + "=" * 60)
print(f"R² Improvement (Polynomial - Linear): {r2_diff:.4f}")
if r2_diff > 0.05:
    print("✓ POLYNOMIAL model is significantly better")
    use_polynomial = True
    y_pred_train = y_pred_poly_train
    y_pred_test = y_pred_poly_test
elif r2_diff < -0.05:
    print("✓ LINEAR model is significantly better")
    use_polynomial = False
    y_pred_train = y_pred_linear_train
    y_pred_test = y_pred_linear_test
else:
    print("→ Models perform similarly - using POLYNOMIAL for flexibility")
    use_polynomial = True
    y_pred_train = y_pred_poly_train
    y_pred_test = y_pred_poly_test
print("=" * 60)

# ==============================================================
# 8. 2024, 2025 & 2026 PREDICTIONS
# ==============================================================

print("\n" + "=" * 60)
print("2024, 2025 & 2026 MAM SEASON PREDICTIONS")
print("=" * 60)

# Variables to store predictions and actuals
predicted_2024 = None
actual_2024 = None
predicted_2025 = None
actual_2025 = None
predicted_2026 = None
actual_2026 = None
model_used = "Polynomial" if use_polynomial else "Linear"

# 2024 Prediction
if 2024 in merged_df['year'].values:
    actual_2024 = merged_df.loc[merged_df['year'] == 2024, 'tornado_count_mam'].values[0]
    temp_2024 = merged_df.loc[merged_df['year'] == 2024, 'temp_anomaly_mam'].values[0]
    oni_2024 = merged_df.loc[merged_df['year'] == 2024, 'oni_mam'].values[0]
    precip_2024 = merged_df.loc[merged_df['year'] == 2024, 'precip_water_mam'].values[0]

    data_2024 = pd.DataFrame({
        'temp_anomaly_mam': [temp_2024],
        'oni_mam': [oni_2024],
        'precip_water_mam': [precip_2024]
    })

    if use_polynomial:
        poly_2024 = poly.transform(data_2024)
        predicted_2024 = poly_model.predict(poly_2024)[0]
    else:
        predicted_2024 = linear_model.predict(data_2024)[0]

    error = predicted_2024 - actual_2024
    percent_error = (error / actual_2024) * 100
    accuracy = 100 - abs(percent_error)

    print(f"\n2024 MAM Season Results (Out-of-Sample):")
    print(f"  Climate Inputs:")
    print(f"    - Temperature Anomaly: {temp_2024:.3f}°C")
    print(f"    - ONI: {oni_2024:.3f}")
    print(f"    - Precipitable Water: {precip_2024:.3f} kg/m²")
    print(f"\n  Model Used: {model_used}")
    print(f"  Predicted MAM Tornadoes: {predicted_2024:.1f}")
    print(f"  Actual MAM Tornadoes: {actual_2024:.0f}")
    print(f"  Difference: {error:+.1f} tornadoes ({percent_error:+.1f}%)")
    print(f"  Accuracy: {accuracy:.1f}%")

# 2025 Prediction with ACTUAL tornado data
# Actual 2025 MAM tornado count: 245 (Mar) + 318 (Apr) + 286 (May) = 849
actual_2025 = 245 + 318 + 286

# Get 2025 climate data from ENSO file
if 2025 in mam_oni['year'].values and 2025 in mam_temp['year'].values:
    temp_2025 = mam_temp.loc[mam_temp['year'] == 2025, 'temp_anomaly_mam'].values[0]
    oni_2025 = mam_oni.loc[mam_oni['year'] == 2025, 'oni_mam'].values[0]
    # Use 2024 precipitable water as proxy for 2025
    precip_2025 = precip_2024 if 2024 in merged_df['year'].values else 0.5

    data_2025 = pd.DataFrame({
        'temp_anomaly_mam': [temp_2025],
        'oni_mam': [oni_2025],
        'precip_water_mam': [precip_2025]
    })

    if use_polynomial:
        poly_2025 = poly.transform(data_2025)
        predicted_2025 = poly_model.predict(poly_2025)[0]
    else:
        predicted_2025 = linear_model.predict(data_2025)[0]

    error_2025 = predicted_2025 - actual_2025
    percent_error_2025 = (error_2025 / actual_2025) * 100
    accuracy_2025 = 100 - abs(percent_error_2025)

    print(f"\n2025 MAM Season Results:")
    print(f"  Climate Inputs:")
    print(f"    - Temperature Anomaly: {temp_2025:.3f}°C")
    print(f"    - ONI: {oni_2025:.3f}")
    print(f"    - Precipitable Water: {precip_2025:.3f} kg/m² (2024 proxy)")
    print(f"\n  Model Used: {model_used}")
    print(f"  Predicted MAM Tornadoes: {predicted_2025:.1f}")
    print(f"  Actual MAM Tornadoes: {actual_2025:.0f}")
    print(f"    - March 2025: 245 tornadoes")
    print(f"    - April 2025: 318 tornadoes")
    print(f"    - May 2025: 286 tornadoes")
    print(f"  Difference: {error_2025:+.1f} tornadoes ({percent_error_2025:+.1f}%)")
    print(f"  Accuracy: {accuracy_2025:.1f}%")

    if abs(percent_error_2025) < 10:
        print(f"\n  ✓ Excellent prediction (within 10%)")
    elif abs(percent_error_2025) < 20:
        print(f"\n  → Good prediction (within 20%)")
    elif abs(percent_error_2025) < 30:
        print(f"\n  ~ Moderate prediction (within 30%)")
    else:
        print(f"\n  ✗ Prediction needs improvement (>30% error)")

# 2026 Forecast
if 2026 in mam_oni['year'].values and 2026 in mam_temp['year'].values:
    temp_2026 = mam_temp.loc[mam_temp['year'] == 2026, 'temp_anomaly_mam'].values[0]
    oni_2026 = mam_oni.loc[mam_oni['year'] == 2026, 'oni_mam'].values[0]
    # Use 2024 precipitable water as proxy for 2026
    precip_2026 = precip_2024 if 2024 in merged_df['year'].values else 0.5

    data_2026 = pd.DataFrame({
        'temp_anomaly_mam': [temp_2026],
        'oni_mam': [oni_2026],
        'precip_water_mam': [precip_2026]
    })

    if use_polynomial:
        poly_2026 = poly.transform(data_2026)
        predicted_2026 = poly_model.predict(poly_2026)[0]
    else:
        predicted_2026 = linear_model.predict(data_2026)[0]

    print(f"\n2026 MAM Season Forecast:")
    print(f"  Climate Inputs:")
    print(f"    - Temperature Anomaly: {temp_2026:.3f}°C")
    print(f"    - ONI: {oni_2026:.3f}")
    print(f"    - Precipitable Water: {precip_2026:.3f} kg/m² (2024 proxy)")
    print(f"\n  Model Used: {model_used}")
    print(f"  Predicted MAM Tornadoes: {predicted_2026:.1f}")
    print(f"  Actual: (Future - not yet observed)")

# ==============================================================
# 9. LINEAR MODEL VISUALIZATION
# ==============================================================

print("\n" + "=" * 60)
print("LINEAR MODEL VISUALIZATIONS (MAM Season)")
print("=" * 60)

sorted_years = test_years.values
sorted_actual = y_test.values
sorted_predicted_linear = y_pred_linear_test

# LINEAR Plot: Overlay comparison
plt.figure(figsize=(12, 6))
plt.plot(sorted_years, sorted_actual, marker='o', linewidth=2, markersize=6,
         color='red', label='Actual', alpha=0.7)
plt.plot(sorted_years, sorted_predicted_linear, marker='s', linewidth=2, markersize=6,
         color='green', label='Linear Predicted', alpha=0.7)
plt.xlabel('Year', fontsize=12)
plt.ylabel('MAM Tornado Count', fontsize=12)
plt.title('LINEAR Model: Actual vs Predicted MAM Tornado Counts (Test Data)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================================================
# 10. POLYNOMIAL MODEL VISUALIZATION
# ==============================================================

print("\n" + "=" * 60)
print("POLYNOMIAL MODEL VISUALIZATIONS (MAM Season)")
print("=" * 60)

sorted_predicted_poly = y_pred_poly_test

# POLYNOMIAL Plot: Overlay comparison
plt.figure(figsize=(12, 6))
plt.plot(sorted_years, sorted_actual, marker='o', linewidth=2, markersize=6,
         color='red', label='Actual', alpha=0.7)
plt.plot(sorted_years, sorted_predicted_poly, marker='s', linewidth=2, markersize=6,
         color='purple', label='Polynomial Predicted', alpha=0.7)
plt.xlabel('Year', fontsize=12)
plt.ylabel('MAM Tornado Count', fontsize=12)
plt.title('POLYNOMIAL Model: Actual vs Predicted MAM Tornado Counts (Test Data)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================================================
# 11. SIDE-BY-SIDE MODEL COMPARISON
# ==============================================================

print("\n" + "=" * 60)
print("SIDE-BY-SIDE COMPARISON (MAM Season)")
print("=" * 60)

# Compare both models on the same plot (test period only)
plt.figure(figsize=(14, 7))

# Plot test period actual data
plt.plot(sorted_years, sorted_actual, marker='o', linewidth=2.5, markersize=7,
         color='red', label='Actual MAM', alpha=0.9, zorder=3)

# Plot model predictions (only for test period)
plt.plot(sorted_years, sorted_predicted_linear, marker='s', linewidth=2, markersize=6,
         color='green', label='Linear Model', alpha=0.7, linestyle='--', zorder=2)
plt.plot(sorted_years, sorted_predicted_poly, marker='^', linewidth=2, markersize=6,
         color='purple', label='Polynomial Model', alpha=0.7, linestyle=':', zorder=2)

plt.xlabel('Year', fontsize=12)
plt.ylabel('MAM Tornado Count', fontsize=12)
plt.title('Model Comparison: Linear vs Polynomial vs Actual (MAM Season)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================================================
# 12. FEATURE IMPORTANCE VISUALIZATION
# ==============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Linear Model Coefficients)")
print("=" * 60)

feature_names = ['Temperature\nAnomaly', 'ONI\n(ENSO)', 'Precipitable\nWater']
coefficients = linear_model.coef_

plt.figure(figsize=(10, 6))
colors = ['#FF6B6B' if c > 0 else '#4ECDC4' for c in coefficients]
bars = plt.bar(feature_names, coefficients, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Linear Model: Feature Importance (MAM Tornado Prediction)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, coef in zip(bars, coefficients):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{coef:.1f}',
             ha='center', va='bottom' if coef > 0 else 'top', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nCoefficient Interpretation:")
print(f"  Temperature Anomaly: {coefficients[0]:.2f} (tornadoes per °C)")
print(f"  ONI (ENSO): {coefficients[1]:.2f} (tornadoes per ONI unit)")
print(f"  Precipitable Water: {coefficients[2]:.2f} (tornadoes per kg/m²)")

# ==============================================================
# 12. REGRESSION FORMULAS
# ==============================================================

print("\n" + "=" * 60)
print("REGRESSION FORMULAS")
print("=" * 60)

# Linear Model Formula
print("\nLINEAR REGRESSION FORMULA:")
print("-" * 60)
print(f"Tornado_Count = {linear_model.intercept_:.2f}")
print(f"              + ({linear_model.coef_[0]:.2f} × Temperature_Anomaly)")
print(f"              + ({linear_model.coef_[1]:.2f} × ONI)")
print(f"              + ({linear_model.coef_[2]:.2f} × Precipitable_Water)")
print("\nSimplified:")
print(
    f"y = {linear_model.intercept_:.2f} + {linear_model.coef_[0]:.2f}x₁ + {linear_model.coef_[1]:.2f}x₂ + {linear_model.coef_[2]:.2f}x₃")
print(f"\nWhere:")
print(f"  y  = Predicted MAM Tornado Count")
print(f"  x₁ = Temperature Anomaly (°C)")
print(f"  x₂ = ONI (Oceanic Niño Index)")
print(f"  x₃ = Precipitable Water (kg/m²)")

# Polynomial Model Formula
print("\n" + "=" * 60)
print("\nPOLYNOMIAL REGRESSION FORMULA (degree=2):")
print("-" * 60)
poly_features = poly.get_feature_names_out(['temp_anomaly_mam', 'oni_mam', 'precip_water_mam'])
print(f"Tornado_Count = {poly_model.intercept_:.2f}")
for i, (feat, coef) in enumerate(zip(poly_features, poly_model.coef_)):
    # Replace feature names for readability
    feat_readable = (feat.replace('temp_anomaly_mam', 'Temp')
                     .replace('oni_mam', 'ONI')
                     .replace('precip_water_mam', 'PrecipWater')
                     .replace('^2', '²')
                     .replace(' ', '×'))
    print(f"              + ({coef:.2f} × {feat_readable})")

print("\nSimplified:")
print(f"y = {poly_model.intercept_:.2f}")
for i, (feat, coef) in enumerate(zip(poly_features, poly_model.coef_)):
    feat_simple = (feat.replace('temp_anomaly_mam', 'x₁')
                   .replace('oni_mam', 'x₂')
                   .replace('precip_water_mam', 'x₃')
                   .replace('^2', '²')
                   .replace(' ', '×'))
    sign = '+' if coef >= 0 else ''
    print(f"    {sign}{coef:.2f}({feat_simple})", end='')
    if (i + 1) % 3 == 0:
        print()
    else:
        print(" ", end='')
print()

print(f"\nWhere:")
print(f"  y  = Predicted MAM Tornado Count")
print(f"  x₁ = Temperature Anomaly (°C)")
print(f"  x₂ = ONI (Oceanic Niño Index)")
print(f"  x₃ = Precipitable Water (kg/m²)")
print(f"\nThe polynomial model includes:")
print(f"  - Linear terms: x₁, x₂, x₃")
print(f"  - Quadratic terms: x₁², x₂², x₃²")
print(f"  - Interaction terms: x₁×x₂, x₁×x₃, x₂×x₃")

# Example Calculation
print("\n" + "=" * 60)
print("EXAMPLE CALCULATION (2024 MAM Season):")
print("-" * 60)
if 2024 in merged_df['year'].values:
    temp_ex = merged_df.loc[merged_df['year'] == 2024, 'temp_anomaly_mam'].values[0]
    oni_ex = merged_df.loc[merged_df['year'] == 2024, 'oni_mam'].values[0]
    precip_ex = merged_df.loc[merged_df['year'] == 2024, 'precip_water_mam'].values[0]

    print(f"\nInput values:")
    print(f"  Temperature Anomaly = {temp_ex:.3f}°C")
    print(f"  ONI = {oni_ex:.3f}")
    print(f"  Precipitable Water = {precip_ex:.3f} kg/m²")

    # Linear calculation
    linear_calc = linear_model.intercept_ + (linear_model.coef_[0] * temp_ex) + (linear_model.coef_[1] * oni_ex) + (
                linear_model.coef_[2] * precip_ex)
    print(f"\nLinear Model Calculation:")
    print(
        f"  = {linear_model.intercept_:.2f} + ({linear_model.coef_[0]:.2f} × {temp_ex:.3f}) + ({linear_model.coef_[1]:.2f} × {oni_ex:.3f}) + ({linear_model.coef_[2]:.2f} × {precip_ex:.3f})")
    print(
        f"  = {linear_model.intercept_:.2f} + {linear_model.coef_[0] * temp_ex:.2f} + {linear_model.coef_[1] * oni_ex:.2f} + {linear_model.coef_[2] * precip_ex:.2f}")
    print(f"  = {linear_calc:.1f} tornadoes")

    # Polynomial calculation (show just the result due to complexity)
    data_ex = pd.DataFrame({'temp_anomaly_mam': [temp_ex], 'oni_mam': [oni_ex], 'precip_water_mam': [precip_ex]})
    poly_ex = poly.transform(data_ex)
    poly_calc = poly_model.predict(poly_ex)[0]
    print(f"\nPolynomial Model Calculation:")
    print(f"  (includes quadratic and interaction terms)")
    print(f"  = {poly_calc:.1f} tornadoes")

    if actual_2024 is not None:
        print(f"\nActual 2024 MAM Tornadoes: {actual_2024:.0f}")
        print(
            f"Linear Model Error: {linear_calc - actual_2024:+.1f} ({(linear_calc - actual_2024) / actual_2024 * 100:+.1f}%)")
        print(
            f"Polynomial Model Error: {poly_calc - actual_2024:+.1f} ({(poly_calc - actual_2024) / actual_2024 * 100:+.1f}%)")

# ==============================================================
# 13. CONSOLIDATED VISUALIZATION WITH 2024, 2025 & 2026
# ==============================================================

print("\n" + "=" * 60)
print("CONSOLIDATED GRAPH WITH 2024, 2025 & 2026")
print("=" * 60)

# Test years
sorted_years = test_years.values
sorted_actual = y_test.values

if use_polynomial:
    sorted_predicted = y_pred_poly_test
else:
    sorted_predicted = y_pred_linear_test

plt.figure(figsize=(16, 8))

# Plot test period actual data (2010-2023)
plt.plot(sorted_years, sorted_actual, marker='o', linewidth=2.5, markersize=7,
         color='red', label='Actual MAM (2010-2023)', alpha=0.9, zorder=3)

# Plot model predictions (only for test period)
model_label = 'Polynomial Model' if use_polynomial else 'Linear Model'
model_color = 'purple' if use_polynomial else 'green'
plt.plot(sorted_years, sorted_predicted, marker='s', linewidth=2, markersize=6,
         color=model_color, label=f'{model_label} Predictions', alpha=0.7, linestyle='--', zorder=2)

# Prepare lists to connect predictions and actuals through 2026
prediction_years = list(sorted_years)
prediction_values = list(sorted_predicted)
actual_years = list(sorted_years)
actual_values = list(sorted_actual)

# Add 2024 if available
if predicted_2024 is not None and actual_2024 is not None:
    plt.scatter([2024], [actual_2024], s=250, color='darkred', marker='o',
                edgecolors='black', linewidths=2.5, label=f'2024 Actual: {actual_2024:.0f}', zorder=5)
    plt.scatter([2024], [predicted_2024], s=250, color='gold', marker='*',
                edgecolors='black', linewidths=2.5,
                label=f'2024 Predicted: {predicted_2024:.1f} ({accuracy:.1f}% accurate)', zorder=5)
    plt.plot([2024, 2024], [actual_2024, predicted_2024], 'k--', linewidth=2, alpha=0.5)

    # Add to connection lists
    prediction_years.append(2024)
    prediction_values.append(predicted_2024)
    actual_years.append(2024)
    actual_values.append(actual_2024)

# Add 2025 if available
if predicted_2025 is not None and actual_2025 is not None:
    plt.scatter([2025], [actual_2025], s=250, color='darkred', marker='o',
                edgecolors='black', linewidths=2.5, label=f'2025 Actual: {actual_2025:.0f}', zorder=5)
    plt.scatter([2025], [predicted_2025], s=250, color='orange', marker='*',
                edgecolors='black', linewidths=2.5,
                label=f'2025 Predicted: {predicted_2025:.1f} ({accuracy_2025:.1f}% accurate)', zorder=5)
    plt.plot([2025, 2025], [actual_2025, predicted_2025], 'k--', linewidth=2, alpha=0.5)

    # Add to connection lists
    prediction_years.append(2025)
    prediction_values.append(predicted_2025)
    actual_years.append(2025)
    actual_values.append(actual_2025)

# Add 2026 forecast if available
if predicted_2026 is not None:
    plt.scatter([2026], [predicted_2026], s=300, color='yellow', marker='*',
                edgecolors='black', linewidths=2.5,
                label=f'2026 Forecast: {predicted_2026:.1f} tornadoes', zorder=5)

    # Add to prediction connection list
    prediction_years.append(2026)
    prediction_values.append(predicted_2026)

# Draw continuous connection lines from 2010-2026
# Connect all predictions with dotted line
if len(prediction_years) > len(sorted_years):
    plt.plot(prediction_years, prediction_values, color=model_color,
             linestyle=':', linewidth=2, alpha=0.5, zorder=1)

# Connect all actuals with dotted line (only if we have 2024+ actuals)
if len(actual_years) > len(sorted_years):
    plt.plot(actual_years, actual_values, color='red',
             linestyle=':', linewidth=2, alpha=0.5, zorder=1)

plt.xlabel('Year', fontsize=13, fontweight='bold')
plt.ylabel('MAM Tornado Count', fontsize=13, fontweight='bold')
plt.title('Tornado Prediction Model: 2010-2026 Performance & Outlook',
          fontsize=15, fontweight='bold')
plt.legend(fontsize=9, loc='best', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)