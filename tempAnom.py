import pandas as pd

# Download from NOAA Climate at a Glance first
# https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series
# Select: Temperature, 3-Month, MAM, 1990-2023, Contiguous US

# Load the data
df = pd.read_csv('temp.csv', skiprows=2)
df.columns = ['date', 'temp']

# Parse year
df['year'] = df['date'].astype(str).str[:4].astype(int)

# Filter to your time period
df = df[(df['year'] >= 1990) & (df['year'] <= 2023)]

# Calculate anomaly (deviation from 1991-2020 baseline, standard practice)
baseline_years = (df['year'] >= 1991) & (df['year'] <= 2020)
baseline_mean = df.loc[baseline_years, 'temp'].mean()

df['temp_anomaly'] = df['temp'] - baseline_mean

# Final data
temp_data = df[['year', 'temp_anomaly']].copy()

print(f"Baseline (1991-2020 average): {baseline_mean:.2f}°F")
print(f"\nTemperature Anomalies for Tornado Season Prediction:")
print(temp_data)

# Save it
temp_data.to_csv('temperature_anomalies.csv', index=False)