
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
