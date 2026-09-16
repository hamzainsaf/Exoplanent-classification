# NASA Kepler Exoplanet Classification Project - Complete Redesign

## 🚀 Project Overview

A production-grade machine learning system for classifying Kepler Objects of Interest (KOI) into **CONFIRMED planets**, **CANDIDATES**, or **FALSE POSITIVES** using scientifically validated visualizations and an interactive web dashboard.

---

## 📊 What Was Built

### 1. **Modernized FastAPI Inference Service** (`API_main.py`)
- **Modern Async Lifespan**: Migrated from deprecated `@app.on_event` to `@asynccontextmanager`
- **Modular Architecture**: Separated concerns into `app/config.py`, `app/schemas.py`, `app/inference.py`, `app/metrics.py`, `app/routes/`
- **55 Strongly-Typed Features**: Each astronomical parameter documented with units and descriptions
- **Production Endpoints**:
  - `/health`, `/ready`, `/metrics` - Observability & monitoring
  - `/model-info`, `/model/features/importance`, `/model/confusion-matrix` - Model inspection
  - `/predict`, `/predict-batch`, `/predict/csv` - Single, batch, and CSV file inference
- **Explainability**: Local feature contributions and global importance rankings
- **Astronomical Triage Scoring**: 0-100 priority score for telescope follow-up observations
- **CSV Upload**: Direct ingestion of NASA Exoplanet Archive tables with metadata preservation

**Run the API**:
```bash
uvicorn API_main:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive Swagger documentation
```

---

### 2. **Publication-Quality EDA Visualizations** (`app/eda.py`)
Accessible, colorblind-safe scientific figures following strict dataviz standards from the `dataviz` skill:

**Generated Plots** (saved to `models/eda/`):
1. **Class Distribution** - Target disposition counts with percentages
2. **Vetting Flags Impact** - 4-panel breakdown showing false-positive flag effectiveness
3. **Exoplanet Population** - Radius vs. Period (log-log) with Earth reference and regime annotations
4. **Habitability Zone** - Equilibrium temperature vs. radius with habitable zone band
5. **Transit SNR Distribution** - Signal quality across dispositions with 7.1σ threshold
6. **Stellar Kiel Diagram** - Host star Teff vs. log(g) with Sun reference
7. **Feature Importances** - Top 20 predictive features by category

**Color Validation**: Uses validated accessible palette (CVD Delta E ≥ 8, contrast-checked)

**Integrated into Training**: `train.py` now automatically generates all EDA plots during model training.

---

### 3. **Interactive Streamlit Dashboard** (`app/dashboard.py`)
A complete web application with 5 pages of rich interactivity:

#### **Page 1: 📊 Data Overview**
- Key metrics tiles (Total KOIs, Confirmed, Candidates, False Positives)
- Interactive class distribution bar chart
- Dataset preview table with key features

#### **Page 2: 🔬 Exoplanet Population**
- **Interactive Filters**: Disposition, radius range, period range
- **Radius vs. Period Scatter**: Log-log plot with Earth reference and regime annotations
- **Habitability Plot**: Temperature vs. radius with habitable zone highlighting

#### **Page 3: ☀️ Stellar Hosts**
- **Kiel Diagram**: Inverted Teff vs. log(g) with Sun marker
- **Metallicity Distribution**: Histogram by disposition
- **Mass-Radius Relation**: Stellar parameter scatter plot

#### **Page 4: 🤖 ML Predictions**
- **Tab 1 - Single Prediction**: Interactive form with 14 key parameters
  - Real-time ML inference with confidence scores
  - Class probability distribution
  - Triage score and top contributing features
- **Tab 2 - Batch CSV Upload**: Drag-and-drop CSV processing
  - Bulk inference with summary statistics
  - Downloadable enriched results
- **Tab 3 - Sample Explorer**: Browse real KOI samples with predictions

#### **Page 5: 📈 Model Insights**
- Global feature importance ranking (top 25 features)
- Model performance metrics comparison
- Confusion matrix visualization
- Baseline model comparison table

**Run the Dashboard**:
```bash
streamlit run app/dashboard.py
# Automatically opens http://localhost:8501
```

---

## 🎨 Design Standards Applied

All visualizations follow the **dataviz skill** methodology:
- ✅ **Validated Color Palettes**: CVD Delta E ≥ 8, contrast-checked
- ✅ **Thin Marks**: 2px lines, 4px rounded data ends, recessive gridlines
- ✅ **Direct Labels**: Selective annotations instead of overwhelming every point
- ✅ **Accessible Typography**: High-contrast text tokens, never colored text for data
- ✅ **Domain Context**: Astronomical regime annotations (habitable zone, radius valley, stellar main sequence)

---

## 🧪 Testing & Verification

### **15 Automated Tests** (`tests/test_api.py`)
```bash
pytest tests/test_api.py -v
# ======================= 15 passed, 2 warnings in 4.46s =======================
```

**Coverage**:
- Health, readiness, and metrics endpoints
- Model metadata and feature importance
- Single and batch predictions
- CSV file upload (JSON and CSV output modes)
- Explainability and local feature contributions
- Vetting flag detection
- Sparse input imputation
- Error handling (400, 422, 503)

### **Training with EDA** (`train.py`)
```bash
python train.py
# Generates:
# - 3 trained models (Logistic, Random Forest, Gradient Boosting)
# - 6 EDA scientific figures in models/eda/
# - 1 feature importance plot
# - Best model artifacts in models/
```

**Performance**:
- **Gradient Boosting** (Best): Macro F1 = 0.8924, CONFIRMED Recall = 86.2%
- **Improvement over Baseline**: +0.34 F1 points

---

## 📁 Project Structure

```
NASA-Project/
├── API_main.py                    # FastAPI entrypoint (modernized)
├── train.py                       # Training pipeline with integrated EDA
├── app/
│   ├── __init__.py
│   ├── config.py                  # Centralized paths, limits, feature descriptions
│   ├── schemas.py                 # Pydantic v2 models with astronomical units
│   ├── inference.py               # Model manager, triage scoring, explainability
│   ├── metrics.py                 # Thread-safe telemetry tracker
│   ├── eda.py                     # Publication-quality scientific visualizations
│   ├── dashboard.py               # Interactive Streamlit web app (NEW)
│   └── routes/
│       ├── health.py              # /health, /ready, /metrics
│       ├── model.py               # /model-info, /model/features/importance
│       └── predict.py             # /predict, /predict-batch, /predict/csv
├── tests/
│   └── test_api.py                # 15 automated integration tests
├── models/
│   ├── best_model.joblib          # Trained Gradient Boosting pipeline
│   ├── feature_columns.json       # 55 feature names
│   ├── class_labels.json          # ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
│   ├── baseline_scores.txt        # Baseline F1 benchmarks
│   ├── confusion_matrix_*.png     # Model evaluation matrices
│   └── eda/                       # 7 publication-quality EDA figures (NEW)
└── data/
    └── kepler_koi.csv             # 9,564 KOIs from NASA Exoplanet Archive
```

---

## 🎯 Key Features

### **Astronomical Domain Integration**
- **Habitability Indicators**: Equilibrium temperature bands for liquid water
- **Physical Regimes**: Terrestrial (0.8-1.6 R⊕), Super-Earth (1.6-4.0 R⊕), Sub-Neptune
- **Stellar Context**: Kiel diagram with main sequence, Sun reference marker
- **Vetting Diagnostics**: 4 false-positive flags with interpretations

### **Explainability & Interpretability**
- **Global Feature Importance**: Tree-based weights by category
- **Local Explanations**: Top contributing features per prediction
- **Triage Scoring**: 0-100 follow-up priority combining probabilities and vetting flags
- **Physical Validation**: Flag audit with readable descriptions

### **Production Readiness**
- **Modern FastAPI Standards**: Async lifespan, typed schemas, CORS, middleware
- **Observability**: Real-time metrics (request counts, latency percentiles, class distribution)
- **Robust Error Handling**: Global exception handlers with structured JSON responses
- **CSV Streaming**: NASA header comment stripping, identifier preservation, chunked processing
- **Test Coverage**: 15 integration tests covering all endpoints and edge cases

---

## 🚦 Quick Start

### **1. Train the Model with EDA**
```bash
python train.py
# Outputs: models/*.joblib, models/eda/*.png (7 scientific figures)
```

### **2. Launch the FastAPI Service**
```bash
uvicorn API_main:app --reload --port 8000
# Visit: http://localhost:8000/docs
```

### **3. Run the Interactive Dashboard**
```bash
streamlit run app/dashboard.py
# Auto-opens: http://localhost:8501
```

### **4. Run Tests**
```bash
pytest tests/test_api.py -v
```

---

## 📊 Model Performance Summary

| Model | Macro F1 | CONFIRMED Recall | Notes |
|-------|----------|------------------|-------|
| **Gradient Boosting** | **0.8924** | **86.2%** | ✅ Best model |
| Random Forest | 0.8784 | 84.9% | Strong performance |
| Logistic Regression | 0.8088 | 76.9% | Simple baseline |
| FP-Flag Heuristic | 0.5527 | — | 4-flag-only model |
| Majority Class | 0.2256 | — | Trivial baseline |

**Improvement**: +61% macro F1 over heuristic baseline (+0.34 absolute)

---

## 🎨 Visualization Highlights

All plots use **scientifically validated accessible colors**:
- **CONFIRMED**: Sky Blue (#0284c7)
- **CANDIDATE**: Warm Amber (#d97706)
- **FALSE POSITIVE**: Rose Red (#e11d48)

**Domain Annotations**:
- Earth reference markers (365.25d, 1.0 R⊕, 255K, 5778K)
- Habitable zone temperature bands (180K - 320K)
- Radius valley (1.6-1.8 R⊕)
- Kepler 7.1σ detection threshold

---

## 🔬 Scientific Context

**Kepler Mission**: NASA space telescope that discovered thousands of exoplanets by detecting periodic brightness dips (transits) when planets cross in front of their host stars.

**Classification Challenge**: Distinguishing real planets from astrophysical false positives (eclipsing binaries, stellar variability, background stars) using transit shape, centroid motion, and multi-wavelength photometry.

**Vetting Flags** (4 Boolean indicators):
1. **Not Transit-Like**: Instrument artifacts or stellar variability
2. **Stellar Eclipse**: Secondary eclipse or odd-even depth asymmetry
3. **Centroid Offset**: Transit source offset from target star (background contaminant)
4. **Ephemeris Match**: Period/epoch matches known variable star

---

## 📚 Technologies Used

- **ML Framework**: scikit-learn (Gradient Boosting, Random Forest, Logistic Regression)
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI 0.141.1 (modern async lifespan)
- **Validation**: Pydantic v2 (strongly-typed astronomical schemas)
- **Visualization**: Matplotlib (EDA), Plotly (interactive dashboard)
- **Web Dashboard**: Streamlit 1.64.0
- **Testing**: Pytest with FastAPI TestClient
- **Data Processing**: Pandas, NumPy

---

## 🎓 Credits

**NASA Space Apps Challenge** - Kepler Exoplanet Candidate Classification
**Dataset**: NASA Exoplanet Archive Kepler KOI Cumulative Table
**Design Standards**: Claude Code `dataviz` skill methodology
