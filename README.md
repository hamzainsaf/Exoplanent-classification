# Kepler Exoplanet Classification 🪐

Machine learning system for classifying Kepler Objects of Interest (KOI) into **CONFIRMED planets**, **CANDIDATES**, or **FALSE POSITIVES** using Gradient Boosting with 89.2% macro F1 score.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project builds a production-grade ML pipeline to classify transit signals from NASA's Kepler Space Telescope. The system distinguishes real exoplanets from astrophysical false positives (eclipsing binaries, stellar variability, background stars) using 55 features including:

- **Vetting Flags**: 4 false-positive indicators
- **Orbital Parameters**: Period, duration, depth, radius
- **Stellar Properties**: Temperature, mass, radius, metallicity
- **Centroid Diagnostics**: Astrometric motion signatures

## Key Features

✨ **High Performance**: 89.2% macro F1 score, 86.2% recall on confirmed planets  
📊 **Scientific Visualizations**: 7 publication-quality EDA plots following dataviz best practices  
🚀 **Production API**: FastAPI service with health checks, metrics, and CSV batch processing  
🔍 **Explainability**: Feature importance ranking and local prediction explanations  
🎯 **Triage Scoring**: Prioritizes candidates for telescope follow-up (0-100 scale)  
🧪 **Test Coverage**: 15 automated integration tests

---

## Dataset

**NASA Exoplanet Archive - Kepler KOI Cumulative Table**
- **9,564 KOIs** across 3 disposition classes
- **55 numerical features** (transit, stellar, centroid)
- **Group-stratified splits** by star system (`kepid`) to prevent data leakage

### Class Distribution

![Class Distribution](models/eda/eda_01_class_distribution.png)

| Disposition | Count | Percentage |
|-------------|-------|------------|
| FALSE POSITIVE | 4,841 | 50.6% |
| CANDIDATE | 2,420 | 25.3% |
| CONFIRMED | 2,303 | 24.1% |

---

## Model Performance

### Results Summary

| Model | Macro F1 | CONFIRMED Recall | Improvement |
|-------|----------|------------------|-------------|
| **Gradient Boosting** | **0.8924** | **86.2%** | **Best** |
| Random Forest | 0.8784 | 84.9% | +59% vs baseline |
| Logistic Regression | 0.8088 | 76.9% | +46% vs baseline |
| FP-Flag Heuristic | 0.5527 | — | Baseline |
| Majority Class | 0.2256 | — | Trivial |

**Key Insight**: The model achieves 61% improvement over the simple false-positive flag heuristic by learning complex interactions between transit geometry, stellar characteristics, and vetting diagnostics.

---

## Exploratory Data Analysis

### 1. Vetting Flags Effectiveness

The 4 Kepler vetting flags are highly predictive of false positives:

![Vetting Flags](models/eda/eda_02_vetting_flags_impact.png)

- **Centroid Offset (CO)**: 98% of flagged KOIs are false positives
- **Stellar Eclipse (SS)**: 95% false positive rate
- **Not Transit-Like (NT)**: 90% false positive rate
- **Ephemeris Match (EC)**: 85% false positive rate

### 2. Exoplanet Population Structure

![Population](models/eda/eda_03_exoplanet_population_radius_period.png)

**Key Observations:**
- **Radius Valley** visible at ~1.8 R⊕ separating terrestrial planets from sub-Neptunes
- **Hot Jupiters** cluster at short periods (<10 days) and large radii (>10 R⊕)
- **Confirmed planets** concentrate in well-characterized parameter space
- Earth analog marked at 365 days, 1.0 R⊕

### 3. Habitability Zone Analysis

![Habitability](models/eda/eda_04_habitability_temperature_radius.png)

Equilibrium temperature determines potential for liquid water:
- **Habitable Zone**: 180K - 320K (green shaded region)
- **Earth Reference**: 255K, 1.0 R⊕
- Most confirmed planets are hot (>500K) due to detection bias

### 4. Transit Signal Quality

![SNR Distribution](models/eda/eda_05_transit_snr_distribution.png)

- **Kepler Detection Threshold**: 7.1σ (dashed red line)
- **Confirmed Planets**: Median SNR = 35.8
- **False Positives**: Lower SNR, often near threshold
- SNR is the 3rd most important feature in the model

### 5. Host Star Properties

![Kiel Diagram](models/eda/eda_06_stellar_host_kiel_diagram.png)

Kiel diagram (Teff vs. log g) shows:
- Most Kepler targets are **main-sequence stars** (4.0 < log g < 4.8)
- Sun-like stars (G-type) dominate the sample
- Few evolved giants or hot A-stars

### 6. Feature Importance

![Feature Importance](models/eda/eda_07_feature_importances_top20.png)

**Top Predictive Features:**
1. **koi_fpflag_co** (Centroid Offset) - 18.2%
2. **koi_fpflag_nt** (Not Transit-Like) - 14.5%
3. **koi_model_snr** (Transit SNR) - 9.8%
4. **koi_fpflag_ss** (Stellar Eclipse) - 8.1%
5. **koi_fpflag_ec** (Ephemeris Match) - 6.9%

**Insight**: Vetting flags dominate (47.7% combined importance), but transit signal quality and orbital geometry provide crucial complementary information.

---

## Installation & Setup

### Requirements

- Python 3.12+
- scikit-learn 1.9
- pandas 3.0
- matplotlib 3.11
- MLflow 3.16
- FastAPI 0.141 (optional, for API)
- Streamlit 1.64 (optional, for dashboard)

### Quick Start

```bash
# Clone repository
git clone https://github.com/hamzainsaf/Exoplanent-classification.git
cd Exoplanent-classification

# Install dependencies
pip install -r requirements.txt

# Train model with EDA
python train.py

# Run tests
pytest tests/test_api.py -v

# Launch API (optional)
uvicorn API_main:app --reload --port 8000

# Launch dashboard (optional)
streamlit run app/dashboard.py
```

---

## Project Structure

```
Exoplanent-classification/
├── train.py                    # Training pipeline with integrated EDA
├── data/
│   └── kepler_koi.csv         # NASA Kepler dataset (9,564 KOIs)
├── models/
│   ├── best_model.joblib      # Trained Gradient Boosting model
│   ├── feature_columns.json   # 55 feature names
│   ├── class_labels.json      # ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
│   ├── baseline_scores.txt    # Baseline F1 benchmarks
│   └── eda/                   # 7 publication-quality visualizations
├── tests/
│   └── test_api.py            # 15 automated integration tests
├── NASA-SPACE-APP/
│   └── Problem Spec.md        # Classification task specification
└── README.md
```

---

## Training Pipeline

The `train.py` script executes:

1. **Data Loading & Cleaning**
   - NASA archive CSV parsing with comment line handling
   - Constant column removal
   - High-missingness feature filtering (>90% null)

2. **Group-Stratified Splitting**
   - Splits by star system (`kepid`) to prevent data leakage
   - Train: 70%, Validation: 10%, Test: 20%
   - Stratified by majority disposition per system

3. **Preprocessing Pipeline**
   - Median imputation for missing values
   - Standard scaling (zero mean, unit variance)
   - ColumnTransformer for numerical features

4. **Model Training**
   - 3 candidate models: Logistic Regression, Random Forest, Gradient Boosting
   - Class-balanced weighting
   - MLflow experiment tracking
   - Best model selection by macro F1

5. **EDA Generation**
   - 7 scientific visualizations using validated color palettes
   - Accessible design (colorblind-safe, high contrast)
   - Domain annotations (habitable zone, radius valley, thresholds)

**Run Training:**
```bash
python train.py
# Outputs: models/*.joblib, models/eda/*.png (7 figures)
```

---

## Model Details

### Gradient Boosting Classifier

**Hyperparameters:**
- Estimators: 200 trees
- Max Depth: 3 (prevents overfitting)
- Learning Rate: 0.1
- Loss: Deviance (logistic regression loss)

**Preprocessing:**
- Median imputation
- Standard scaling
- 55 numerical features

**Performance:**
- Macro F1: **0.8924**
- CONFIRMED Recall: **86.2%**
- CANDIDATE Recall: 82.4%
- FALSE POSITIVE Recall: 99.8%

---

## API Endpoints (Optional)

If you run the FastAPI service:

```bash
uvicorn API_main:app --reload --port 8000
```

### Available Endpoints

#### Predictions
- `POST /predict` - Classify single KOI
- `POST /predict-batch` - Batch JSON inference
- `POST /predict/csv` - Upload CSV file for bulk processing

#### Model Info
- `GET /model-info` - Model metadata and features
- `GET /model/features/importance` - Feature importance ranking
- `GET /model/confusion-matrix` - Confusion matrix image

#### Health & Metrics
- `GET /health` - Service health check
- `GET /ready` - Kubernetes readiness probe
- `GET /metrics` - Real-time performance metrics

**Interactive Docs**: http://localhost:8000/docs

---

## Streamlit Dashboard (Optional)

Interactive web application with 5 pages:

```bash
streamlit run app/dashboard.py
# Opens: http://localhost:8501
```

**Features:**
- 📊 Data Overview - Dataset statistics and previews
- 🔬 Exoplanet Population - Interactive scatter plots with filters
- ☀️ Stellar Hosts - Kiel diagram and metallicity analysis
- 🤖 ML Predictions - Single/batch/CSV prediction interface
- 📈 Model Insights - Feature importance and performance metrics

---

## Scientific Context

### The Kepler Mission

NASA's Kepler Space Telescope (2009-2018) discovered thousands of exoplanets using the **transit method**: detecting periodic brightness dips when planets cross in front of their host stars.

### Classification Challenge

Not all transit-like signals are planets. **False positives** include:
- **Eclipsing Binaries**: Two stars orbiting each other
- **Background Stars**: Blended light from unresolved sources
- **Stellar Variability**: Star spots, flares, pulsations
- **Instrumental Artifacts**: Cosmic rays, thruster firings

### Vetting Process

Kepler's automated pipeline flags suspicious signals using:
1. **Transit Shape**: Non-planetary light curve morphology
2. **Secondary Eclipse**: Brightness increase at phase 0.5 (stellar eclipse)
3. **Centroid Motion**: Source offset from target star
4. **Ephemeris Match**: Period/epoch matches known variable star

This project's ML model learns to combine these diagnostics with physical parameters to achieve 89% accuracy.

---

## Results & Insights

### What The Model Learned

1. **Vetting flags are crucial** but not sufficient alone (baseline: 55.3% F1)
2. **Transit SNR** separates marginal detections from robust signals
3. **Planetary radius** helps distinguish blended binaries (giant "planets" are suspicious)
4. **Stellar metallicity** correlates with giant planet occurrence
5. **Multi-transiting systems** have higher confirmation rates (dynamical validation)

### Remaining Challenges

- **Class Imbalance**: False positives outnumber planets 2:1
- **Detection Bias**: Hot, large planets overrepresented
- **Missing Data**: 15-30% null rate in some features
- **Candidate Ambiguity**: Many true status unknown pending follow-up

---

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Deep learning model (CNN on light curves)
- [ ] SHAP explainability integration
- [ ] Additional features (photometry, spectroscopy)
- [ ] Ensemble stacking with baseline models
- [ ] Hyperparameter optimization (Optuna)
- [ ] Deployment guide (Docker, Kubernetes)

---

## Citation

If you use this work, please cite:

```bibtex
@software{kepler_exoplanet_classifier,
  author = {Hamza Insaf},
  title = {Kepler Exoplanet Classification: ML Pipeline for KOI Disposition Prediction},
  year = {2026},
  url = {https://github.com/hamzainsaf/Exoplanent-classification}
}
```

**Dataset Citation:**
```
NASA Exoplanet Archive: Kepler Objects of Interest Cumulative Table
https://exoplanetarchive.ipac.caltech.edu/
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- **NASA Kepler Mission** - Dataset and scientific context
- **NASA Exoplanet Archive** - Data hosting and documentation
- **scikit-learn** - Machine learning framework
- **MLflow** - Experiment tracking

---

## Contact

**Hamza Insaf**  
📧 muhammadhamzainsaf@gmail.com  
🐙 [@hamzainsaf](https://github.com/hamzainsaf)

**Project**: [Exoplanet Classification](https://github.com/hamzainsaf/Exoplanent-classification)

---

*Built for NASA Space Apps Challenge 2026* 🚀
#   E x o p l a n e n t - c l a s s i f i c a t i o n  
 