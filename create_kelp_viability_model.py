"""
Kelp Viability Model
====================
A machine learning model that predicts kelp viability (0-1 continuous score)
given CUTI index and geographic coordinates of a coastal region.

The model uses regression to output a continuous viability score where:
- 0 = Very low viability (kelp unlikely to be present/abundant)
- 1 = Very high viability (kelp very likely to be present/abundant)
- 0.5 = Moderate viability
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
from datetime import datetime

# ============================================================================
# SECTION 1: LOAD AND PREPARE TRAINING DATA
# ============================================================================

print("[1] Loading training data...")

# Load ML training dataset
ml_df = pd.read_csv('data/ml_training_data.csv')
print(f"  Loaded {len(ml_df)} training observations")

# Convert target to continuous viability score (0-1)
# Use the proportion of kelp observations (TOTAL) normalized by location
kelp_df = pd.read_csv('data/final_data/Updated_kelp_data_2000_onwards.csv')
kelp_df['TOTAL'] = pd.to_numeric(kelp_df['TOTAL'], errors='coerce')

# Calculate normalized viability as a continuous value
# Viability = observed biomass / maximum observed biomass across all observations
max_biomass = kelp_df['TOTAL'].max()
print(f"  Maximum observed kelp biomass: {max_biomass}")

# Create viability score by merging with kelp data
ml_df_with_viability = ml_df.copy()

# For continuous scores, use TOTAL biomass normalized to 0-1 range
# Add small epsilon to avoid dividing by zero
ml_df_with_viability['viability_score'] = ml_df_with_viability['TOTAL'] / max_biomass
ml_df_with_viability['viability_score'] = ml_df_with_viability['viability_score'].clip(0, 1)

print(f"  Viability score range: {ml_df_with_viability['viability_score'].min()} to {ml_df_with_viability['viability_score'].max()}")

# ============================================================================
# SECTION 2: PREPARE FEATURES FOR MODEL
# ============================================================================

print("\n[2] Preparing features...")

# Select features: CUTI, latitude, longitude
feature_columns = ['CUTI', 'lat', 'lon']
X = ml_df_with_viability[feature_columns].copy()
y = ml_df_with_viability['viability_score'].copy()

# Remove rows with NaN values
valid_idx = ~(X.isna().any(axis=1) | y.isna())
X = X[valid_idx].reset_index(drop=True)
y = y[valid_idx].reset_index(drop=True)

print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Removed {(~valid_idx).sum()} rows with missing values")
print(f"\nFeature statistics:")
print(X.describe())
print(f"\nTarget (viability) statistics:")
print(y.describe())

# ============================================================================
# SECTION 3: SPLIT DATA AND SCALE FEATURES
# ============================================================================

print("\n[3] Train-test split and feature scaling...")

# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# Scale features to mean=0, std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# SECTION 4: TRAIN GRADIENT BOOSTING REGRESSION MODEL
# ============================================================================

print("\n[4] Training Gradient Boosting Regressor...")

# Initialize and train model
# Using Gradient Boosting for better handling of non-linear relationships
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42,
    verbose=0
)

model.fit(X_train_scaled, y_train)
print("  Model training complete")

# ============================================================================
# SECTION 5: EVALUATE MODEL
# ============================================================================

print("\n[5] Model evaluation...")

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"  Training R²: {train_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Training RMSE: {train_rmse:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")
print(f"  Training MAE: {train_mae:.4f}")
print(f"  Test MAE: {test_mae:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature importance:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# SECTION 6: SAVE MODEL AND SCALER
# ============================================================================

print("\n[6] Saving model artifacts...")

# Save model
model_path = 'models/kelp_viability_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"  Model saved: {model_path}")

# Save scaler
scaler_path = 'models/kelp_viability_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  Scaler saved: {scaler_path}")

# Save feature columns and model metadata
metadata = {
    'feature_columns': feature_columns,
    'feature_importance': feature_importance.set_index('feature')['importance'].to_dict(),
    'training_date': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'model_type': 'GradientBoostingRegressor',
    'output_range': [0, 1],
    'output_description': 'Continuous kelp viability score (0=low, 1=high)'
}

metadata_path = 'models/kelp_viability_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  Metadata saved: {metadata_path}")

# ============================================================================
# SECTION 7: PREDICTION FUNCTION
# ============================================================================

print("\n[7] Creating prediction interface...")

def predict_kelp_viability(cuti_index, latitude, longitude):
    """
    Predict kelp viability score for given coordinates and CUTI index.
    
    Parameters:
    -----------
    cuti_index : float
        Coastal Upwelling Transport Index value
    latitude : float
        Latitude of location (degrees N, typically 31-47 for California)
    longitude : float
        Longitude of location (degrees W, typically -116 to -126)
    
    Returns:
    --------
    viability_score : float
        Kelp viability score between 0 and 1
        0 = very low viability
        1 = very high viability
    """
    # Create input array
    input_data = np.array([[cuti_index, latitude, longitude]])
    
    # Scale using the same scaler
    input_scaled = scaler.transform(input_data)
    
    # Predict
    viability = model.predict(input_scaled)[0]
    
    # Ensure output is in [0, 1]
    viability = np.clip(viability, 0, 1)
    
    return float(viability)

# Save prediction function
predictor_code = '''
import pickle
import numpy as np

# Load model and scaler
with open('models/kelp_viability_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/kelp_viability_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_kelp_viability(cuti_index, latitude, longitude):
    """
    Predict kelp viability score for given coordinates and CUTI index.
    
    Parameters:
    -----------
    cuti_index : float
        Coastal Upwelling Transport Index value
    latitude : float
        Latitude of location (degrees N, typically 31-47 for California)
    longitude : float
        Longitude of location (degrees W, typically -116 to -126)
    
    Returns:
    --------
    viability_score : float
        Kelp viability score between 0 and 1
        0 = very low viability
        1 = very high viability
    """
    # Create input array
    input_data = np.array([[cuti_index, latitude, longitude]])
    
    # Scale using the same scaler
    input_scaled = scaler.transform(input_data)
    
    # Predict
    viability = model.predict(input_scaled)[0]
    
    # Ensure output is in [0, 1]
    viability = np.clip(viability, 0, 1)
    
    return float(viability)

if __name__ == "__main__":
    # Example usage
    test_cases = [
        (0.5, 34.0, -120.5),   # Central CA coast, moderate upwelling
        (-0.2, 32.5, -119.5),  # Southern CA coast, weak upwelling
        (0.8, 36.0, -121.5),   # Northern CA coast, strong upwelling
    ]
    
    print("Kelp Viability Predictions")
    print("=" * 60)
    for cuti, lat, lon in test_cases:
        viability = predict_kelp_viability(cuti, lat, lon)
        print(f"CUTI={cuti:6.2f}, Lat={lat:6.2f}°N, Lon={lon:7.2f}°W → Viability={viability:.3f}")
'''

predictor_path = 'models/predict_kelp_viability.py'
with open(predictor_path, 'w') as f:
    f.write(predictor_code)
print(f"  Prediction script saved: {predictor_path}")

# ============================================================================
# SECTION 8: TEST PREDICTIONS
# ============================================================================

print("\n[8] Testing predictions with example locations...")

test_locations = [
    (0.5, 34.0, -120.5, "Central CA coast, moderate upwelling"),
    (-0.2, 32.5, -119.5, "Southern CA coast, weak upwelling"),
    (0.8, 36.0, -121.5, "Northern CA coast, strong upwelling"),
    (0.0, 35.0, -120.0, "Central CA coast, neutral upwelling"),
]

print("\nExample predictions:")
print("-" * 80)
for cuti, lat, lon, description in test_locations:
    viability = predict_kelp_viability(cuti, lat, lon)
    print(f"  {description}")
    print(f"    CUTI={cuti:6.2f}, Lat={lat:6.2f}°N, Lon={lon:7.2f}°W → Viability={viability:.3f}")

# ============================================================================
# SECTION 9: SAVE SUMMARY REPORT
# ============================================================================

summary_report = f"""
KELP VIABILITY MODEL - TRAINING SUMMARY REPORT
===============================================

Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL SPECIFICATION
-------------------
Model Type: Gradient Boosting Regressor
Output Type: Continuous score (0 to 1)
- 0 = Very low viability (kelp unlikely to be present/abundant)
- 1 = Very high viability (kelp very likely to be present/abundant)

TRAINING DATA
-------------
Total observations: {len(ml_df_with_viability)}
Training set: {len(X_train)} samples
Test set: {len(X_test)} samples
Valid samples (no missing values): {len(X)}
Removed samples: {len(ml_df_with_viability) - len(X)}

TARGET VARIABLE (Kelp Viability Score)
---------------------------------------
Calculation: TOTAL kelp biomass / maximum observed biomass
Range: 0 to 1
Mean: {y.mean():.4f}
Std Dev: {y.std():.4f}
Min: {y.min():.4f}
Max: {y.max():.4f}

INPUT FEATURES
--------------
1. CUTI (Coastal Upwelling Transport Index)
   - Range: {X['CUTI'].min():.4f} to {X['CUTI'].max():.4f}
   - Mean: {X['CUTI'].mean():.4f}
   - Std Dev: {X['CUTI'].std():.4f}

2. Latitude (degrees North)
   - Range: {X['lat'].min():.2f}°N to {X['lat'].max():.2f}°N
   - Mean: {X['lat'].mean():.2f}°N
   - Std Dev: {X['lat'].std():.2f}°

3. Longitude (degrees West)
   - Range: {X['lon'].min():.2f}°W to {X['lon'].max():.2f}°W
   - Mean: {X['lon'].mean():.2f}°W
   - Std Dev: {X['lon'].std():.2f}°

FEATURE IMPORTANCE
------------------
{chr(10).join([f"{row['feature']:15s}: {row['importance']:.4f}" for _, row in feature_importance.iterrows()])}

MODEL PERFORMANCE
-----------------
Training Set:
  R² Score: {train_r2:.4f}
  RMSE: {train_rmse:.4f}
  MAE: {train_mae:.4f}

Test Set:
  R² Score: {test_r2:.4f}
  RMSE: {test_rmse:.4f}
  MAE: {test_mae:.4f}

Interpretation:
  R² Score: Proportion of variance explained (0.0-1.0, higher is better)
  RMSE: Root mean squared error in viability score units
  MAE: Mean absolute error in viability score units

MODEL FILES
-----------
Model: models/kelp_viability_model.pkl
Scaler: models/kelp_viability_scaler.pkl
Metadata: models/kelp_viability_metadata.json
Predictor Script: models/predict_kelp_viability.py

USAGE
-----
Python API:
  from create_kelp_viability_model import predict_kelp_viability
  viability = predict_kelp_viability(cuti=0.5, latitude=34.0, longitude=-120.5)

Command Line:
  python models/predict_kelp_viability.py

EXAMPLE PREDICTIONS
-------------------
Central CA coast (CUTI=0.5, Lat=34.0°N, Lon=-120.5°W):
  Viability: {predict_kelp_viability(0.5, 34.0, -120.5):.3f}

Southern CA coast (CUTI=-0.2, Lat=32.5°N, Lon=-119.5°W):
  Viability: {predict_kelp_viability(-0.2, 32.5, -119.5):.3f}

Northern CA coast (CUTI=0.8, Lat=36.0°N, Lon=-121.5°W):
  Viability: {predict_kelp_viability(0.8, 36.0, -121.5):.3f}

NOTES
-----
- The model outputs a continuous score between 0 and 1, not binary classification
- CUTI (upwelling) is typically the strongest predictor of kelp viability
- Geographic location (latitude) also plays an important role
- Model should be retrained when new data becomes available
- For production use, consider ensemble methods or cross-validation results
"""

report_path = 'models/KELP_VIABILITY_MODEL_SUMMARY.txt'
with open(report_path, 'w') as f:
    f.write(summary_report)
print(f"\n✓ Summary report saved: {report_path}")

print("\n" + "=" * 80)
print("KELP VIABILITY MODEL TRAINING COMPLETE")
print("=" * 80)
print(f"\nModel successfully trained and saved!")
print(f"Output folder: models/")
print(f"Use models/predict_kelp_viability.py for predictions")
