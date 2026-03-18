# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: RAKSHATHA S A
RegisterNumber: 212225220079
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("C:/Users/acer/Downloads/weather-station-eee-block_2024_07_13.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert time column
df['time'] = pd.to_datetime(df['time'])

# Sort data by time
df = df.sort_values('time').reset_index(drop=True)

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
cols_to_fill = [
    'tem', 'pm2_5', 'tsr', 'hum',
    'pressure', 'wind_speed', 'illumination', 'co2'
]

for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear', limit=10)

# -----------------------------
# 3. Time Feature Engineering
# -----------------------------
df['hour'] = df['time'].dt.hour

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# -----------------------------
# 4. Lag Feature Creation
# -----------------------------
targets = ['tem', 'pm2_5', 'tsr']

for t in targets:
    df[f'{t}_lag1'] = df[t].shift(1)
    df[f'{t}_lag2'] = df[t].shift(2)

# Remove rows with missing lag values
processed_df = df.dropna(
    subset=['tem_lag2', 'pm2_5_lag2', 'tsr_lag2', 'hum', 'pressure']
).reset_index(drop=True)

# Save processed dataset
processed_df.to_csv("combined_processed_weather_data.csv", index=False)

# -----------------------------
# 5. Feature Selection
# -----------------------------
features = [
    'hum', 'pressure', 'wind_speed', 'illumination', 'co2',
    'hour_sin', 'hour_cos',
    'tem_lag1', 'pm2_5_lag1', 'tsr_lag1'
]

print("\n--- Feature Engineering Summary ---")
print("Original rows:", len(df))
print("Processed rows:", len(processed_df))
print("Final features:", features)

# -----------------------------
# 6. Train-Test Split
# -----------------------------
split_idx = int(len(processed_df) * 0.8)

train = processed_df.iloc[:split_idx]
test = processed_df.iloc[split_idx:]

X_train = train[features]
X_test = test[features]

# -----------------------------
# 7. Train Models
# -----------------------------
models = {}
results = {}

target_meta = {
    'tem': ('Temperature', '°C', 'red'),
    'pm2_5': ('Pollution (PM2.5)', 'µg/m³', 'green'),
    'tsr': ('Energy (Solar Radiation)', 'W/m²', 'orange')
}

for target in targets:

    y_train = train[target]
    y_test = test[target]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    models[target] = model

    results[target] = {
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds),
        'preds': preds,
        'actual': y_test.values
    }

# -----------------------------
# 8. Visualization
# -----------------------------
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, target in enumerate(targets):

    label, unit, color = target_meta[target]
    res = results[target]

    # Actual vs Predicted
    axes[i, 0].plot(
        res['actual'][-150:],
        label="Actual",
        color='black',
        alpha=0.4,
        linewidth=2
    )

    axes[i, 0].plot(
        res['preds'][-150:],
        label="Predicted",
        color=color,
        linestyle='--',
        linewidth=2
    )

    axes[i, 0].set_title(
        f"{label}: Actual vs Predicted\n"
        f"R²: {res['r2']:.3f} | MAE: {res['mae']:.2f}"
    )

    axes[i, 0].set_ylabel(unit)
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)

    # Feature Importance
    importances = pd.Series(
        models[target].feature_importances_,
        index=features
    ).sort_values()

    importances.plot(
        kind='barh',
        ax=axes[i, 1],
        color=color,
        alpha=0.7
    )

    axes[i, 1].set_title(f"Key Drivers: {label}")

plt.tight_layout()
plt.show()

# -----------------------------
# 9. Predict Next Step
# -----------------------------
last_row = processed_df.iloc[-1]

latest_data = pd.DataFrame([{
    'hum': last_row['hum'],
    'pressure': last_row['pressure'],
    'wind_speed': last_row['wind_speed'],
    'illumination': last_row['illumination'],
    'co2': last_row['co2'],
    'hour_sin': last_row['hour_sin'],
    'hour_cos': last_row['hour_cos'],
    'tem_lag1': last_row['tem'],
    'pm2_5_lag1': last_row['pm2_5'],
    'tsr_lag1': last_row['tsr']
}])

print("\n--- NEXT STEP PREDICTIONS (Using Latest Data) ---")

for target in targets:

    pred_val = models[target].predict(latest_data)[0]

    print(
        f"Predicted {target_meta[target][0]}: "
        f"{pred_val:.2f} {target_meta[target][1]}"
    )
~~~
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Jessica
RegisterNumber:  212225220044
*/
```

## Output:
<img width="1010" height="1070" alt="image" src="https://github.com/user-attachments/assets/13140b45-b1c8-41f8-a35f-8daf1fe5df3f" />



## Result:
