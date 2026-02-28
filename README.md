## **Can we train a predictive model on historical kelp presence and environmental drivers correlated with kelp growth and health (SSTA, bathymetry, and seasonal CUTI averages) to accurately assign habitat suitability scores to coastal regions to identify areas of high restoration potential?**

---

# Kelp Viability Model: Optimal Growth Conditions & Data Sources

This project focuses on identifying and modeling the environmental drivers that sustain healthy kelp forests. By integrating historical biomass data with physical oceanographic variables, the model assesses the viability of restoration and farming sites.

## 🌊 Ideal Environmental Conditions

For canopy-forming kelp (specifically *Macrocystis pyrifera* and  *Nereocystis luetkeana* ), the following parameters are generally required:

* **Temperature:** **$5^{\circ}C$** to **$14^{\circ}C$**. High temperatures (above **$18^{\circ}C$** - **$20^{\circ}C$**) lead to thermal stress and high mortality.
* **Bathymetry:**  **$5m$ to **$30m$**** .
* **Substrate:** Hard, stable surfaces (Basalt, Granite, Limestone) are required for holdfast attachment.
* **Water Chemistry:** High salinity and nutrient-rich (Nitrate/Phosphate) environments.
* **Energy:** Moderate to high wave energy (necessary for nutrient exchange) but below destructive thresholds.

---

## 📊 Data Inputs & Model Features

The following datasets are used to train the model and establish "truly viable" historical baselines.

| **Feature**             | **Description**                                                                  | **Source**                                                                                        | **Temporal Range** |
| ----------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------ |
| **Historical Biomass**  | 30x30m pixel canopy area (kg) derived from Landsat 5, 7, 8, & 9.                       | [Environmental Data Initiative](https://edirepository.org/)                                                | 1987 – Present          |
| **SSTA (unsused)**      | Sea Surface Temperature Anomalies (Seasonal averages with 30-year climatology).        | [NOAA MUR](https://www.google.com/search?q=https://www.ncei.noaa.gov/products/mur-sea-surface-temperature) | 1992 – Present          |
| **CUTI (Upwelling)**    | Coastal Upwelling Transport Index; indicates nutrient delivery from deep, cold waters. | [Jacox Upwelling Indices](https://mjacox.com/upwelling-indices/)                                           | 1988 – 2025             |
| **Bathymetry (unused)** | Depth and seafloor topography (essential for light and pressure limits).               | USGS / GEBCO                                                                                            | Static                   |

### Data Processing Notes

* **Biomass:** Includes data for Baja California, California, Oregon, and Washington.
* **Anomalies:** SSTA is calculated against a 30-year baseline to identify extreme heat events (e.g., Marine Heatwaves).
* **Upwelling:** CUTI data is averaged quarterly to align with seasonal growth cycles of the kelp canopy.

## 🌊 CUTI (Coastal Upwelling Transport Index) - Output Interpretation

### What is CUTI?

The **Coastal Upwelling Transport Index (CUTI)** measures the intensity of coastal upwelling—the process where deep, cold, nutrient-rich water is driven to the ocean surface. This is **critically important for kelp** because:

- **Nutrient delivery:** Deep water is rich in nitrogen and phosphorus, which kelp requires for growth
- **Temperature regulation:** Cold upwelled water helps maintain optimal kelp temperatures (5–14°C)
- **Oxygen supply:** Enhanced water mixing increases dissolved oxygen

### CUTI Output Values & Interpretation

CUTI is a **continuous index** that can range from negative to positive values. It's calculated quarterly and represents the magnitude and direction of upwelling.

| **CUTI Range** | **Interpretation** | **Kelp Impact** |
| --- | --- | --- |
| **< -0.5** | Strong downwelling (water sinking) | ❌ Poor conditions: Warm, nutrient-poor water |
| **-0.5 to 0** | Weak/negative upwelling | ⚠️ Marginal: Limited nutrient delivery |
| **0 to 0.5** | Weak/moderate upwelling | ✅ Good: Adequate nutrient delivery |
| **0.5 to 1.0** | Strong upwelling | ✅✅ Excellent: Robust nutrient delivery |
| **> 1.0** | Intense upwelling | ✅✅✅ Optimal: Maximum nutrient-rich conditions |

### Typical California CUTI Values

Based on the training dataset (1988–2025):
- **Mean CUTI:** 0.429 (indicating generally favorable upwelling conditions)
- **Range:** -0.335 to 1.847
- **Seasonal patterns:** Strongest upwelling typically occurs in spring/summer (Q2–Q3)

### How CUTI Drives the Kelp Viability Model

In the trained kelp viability model:
- **CUTI accounts for 55.3% of the model's predictive power**—the strongest single driver of kelp viability
- Higher CUTI values → higher predicted viability scores (0–1)
- Example relationship:
  - CUTI = 0.80 (strong upwelling) → Predicted viability ≈ 0.076–0.093
  - CUTI = 0.00 (neutral) → Predicted viability ≈ 0.064
  - CUTI = -0.20 (weak downwelling) → Predicted viability ≈ 0.000

### CUTI Data Source & Availability

- **Source:** [Jacox Upwelling Indices](https://mjacox.com/upwelling-indices/)
- **Temporal coverage:** 1988–2025
- **Spatial resolution:** Latitude bands (31N–47N for California coast)
- **Temporal resolution:** Quarterly averages
- **Status in project:** ✅ Fully integrated with 100% coverage in ML training dataset

---

---

## 🔧 Project Workflow & Scripts

### 1. **Data Merging & Alignment** (`merge_kelp.py`)

Aligns historical kelp observations with administrative kelp bed data and validates field consistency.

**Output:**

- `data/merged_historical_with_admin.csv` - 23,785 rows with spatial details from administrative data
- `data/bed_name_conflicts.csv` - Report of 15 kelp beds with inconsistent naming

**Process:**

- Joins historical kelp data by bed number with administrative polygon boundaries
- Validates bed name consistency across observations

---

### 2. **ML Training Data Creation** (`create_ml_training_data.py`)

Integrates kelp observations with CUTI and SSTA environmental data to create a machine learning dataset.

**Outputs:**

- `data/ml_training_data.csv` - 1,500 observations × 17 features
- `data/ML_TRAINING_DATA_METHODOLOGY.txt` - Detailed methodology notes

**Data Integration:**

- **Kelp bed coordinates:** Estimated centroids (temporary solution pending polygon geometry)
- **CUTI integration:** Matched to nearest latitude band (31N–47N) by observation date
- **SSTA integration:** Values extracted for 2009-03-01 and 2009-06-01 (limited temporal coverage)
- **Target variable:** Binary presence/absence (1 if TOTAL > 0, else 0)

**Feature Completeness:**

- All 1,500 observations have valid coordinates and CUTI values
- 0 observations matched with SSTA (kelp data is 2000–2006, outside 2009 SSTA range)
- Kelp presence rate: 74.2%

**Methodology Notes:**

- Kelp bed coordinates are **estimates** based on bed numbering patterns—awaiting actual polygon centroids
- CUTI matching uses **nearest-neighbor latitude band** approach
- SSTA coverage is **extremely limited** (2 timesteps); should be replaced with complete dataset
- Document includes recommendations for acquiring full coordinate and SSTA data

---

### 3. **Kelp Viability Model** (`create_kelp_viability_model.py`)

Trains a continuous kelp viability prediction model using Gradient Boosting Regressor.

**Outputs:**

- `models/kelp_viability_model.pkl` - Trained model (648 KB)
- `models/kelp_viability_scaler.pkl` - Feature scaler
- `models/kelp_viability_metadata.json` - Model configuration & performance metrics
- `models/KELP_VIABILITY_MODEL_SUMMARY.txt` - Training summary report
- `models/predict_kelp_viability.py` - Standalone prediction script

**Model Specification:**

- **Type:** Gradient Boosting Regressor (tree‑based ensemble used for regression)
- **Output:** Continuous viability score (0.0 to 1.0, NOT binary)
  - 0 = Very low viability (kelp unlikely to be present/abundant)
  - 0.5 = Moderate viability
  - 1.0 = Very high viability (kelp very likely to be present/abundant)
- **Features:** CUTI index, Latitude, Longitude
- **Training Data:** 1,200 samples | **Test Data:** 300 samples

**Model Performance:**

| Metric              | Training | Test  |
| ------------------- | -------- | ----- |
| **R² Score** | 0.758    | 0.307 |
| **RMSE**      | 0.057    | 0.113 |
| **MAE**       | 0.032    | 0.055 |

**Feature Importance:**

| Feature | Importance |
| ------- | ---------- |
| CUTI    | 55.3%      |
|         |            |

> The model computes importance by summing the reduction in squared error
> contributed by splits on each variable and normalizing to 100 %.  In a
> three‑feature model that sum is 1.0, so CUTI’s score of 0.553 tells us that
> **more than half of the model’s explanatory power comes from CUTI alone**.
> That quantitative dominance is why we say CUTI is the strongest driver: if
> all features were equally weighted they would each be ~33 %, so 55 % is a
> clear outlier and indicates the model relies heavily on upwelling intensity
> to predict kelp viability.

**Example Predictions:**

```
Central CA coast   (CUTI= 0.50, Lat=34.0°N, Lon=-120.5°W) → Viability: 0.019
Southern CA coast  (CUTI=-0.20, Lat=32.5°N, Lon=-119.5°W) → Viability: 0.000
Northern CA coast  (CUTI= 0.80, Lat=36.0°N, Lon=-121.5°W) → Viability: 0.076
Neutral upwelling  (CUTI= 0.00, Lat=35.0°N, Lon=-120.0°W) → Viability: 0.064
```

**Usage:**

Python API:

```python
from create_kelp_viability_model import predict_kelp_viability
viability = predict_kelp_viability(cuti_index=0.5, latitude=34.0, longitude=-120.5)
```

Command Line:

```bash
python models/predict_kelp_viability.py
```

---

## 🛠 Future Considerations

* **Coordinate Refinement:** Replace estimated bed centroids with actual polygon centroid calculations from administrative geometry
* **SSTA Data Expansion:** Obtain complete SSTA daily/monthly dataset for 2000–2010 to improve temporal integration
* **Dashboard Integration:** Development of an interactive tool to assess restoration viability based on real-time environmental context
* **Success/Failure Analysis:** Incorporating a secondary dataset of previous human-led restoration projects to refine model weights
* **Substrate Fluidity:** While substrate is a "hard limit" for natural beds, the model should allow for flexibility if evaluating suspended kelp farms (longlines)
* **Model Refinement:** Test alternative algorithms (Random Forest, XGBoost, Neural Networks) and cross-validation strategies
* **Feature Engineering:** Investigate additional environmental features (dissolved oxygen, light penetration, wave energy)

---

## 📖 Terminology

* **Upwelling (CUTI):** The process where deep, cold, nutrient-rich water rises to the surface.
* **SST (Sea Surface Temperature):** The water temperature close to the ocean's surface.
* **SSTA (Sea Surface Temperature Anomaly):** Temperature deviation from a long-term climatological baseline, used to identify extreme heat events.
* **Climatology:** In this context, the average of an oceanographic variable over a long period (usually 30 years) used as a baseline.
* **Bathymetry:** The measurement of depth of water in oceans, seas, or lakes.
* **Viability Score:** A continuous prediction (0–1) of kelp presence likelihood; not a binary classification.
* **Centroid:** The geometric center of a polygon (kelp bed), used as a representative location for spatial analysis.
