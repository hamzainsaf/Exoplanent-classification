"""Unit and integration tests for Kepler Exoplanet KOI Inference API."""

import io
import pytest
from fastapi.testclient import TestClient

from API_main import app
from app.inference import inference_engine


@pytest.fixture(scope="session", autouse=True)
def initialize_engine():
    """Ensure inference engine is initialized before running tests."""
    inference_engine.load()


@pytest.fixture
def client():
    """Create a FastAPI TestClient using context manager to trigger lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_koi_payload():
    """Sample valid KOI candidate data matching Kepler Catalog parameters."""
    return {
        "koi_fpflag_nt": 0,
        "koi_fpflag_ss": 0,
        "koi_fpflag_co": 0,
        "koi_fpflag_ec": 0,
        "koi_period": 9.48803557,
        "koi_time0bk": 170.53875,
        "koi_time0": 2455003.53875,
        "koi_impact": 0.146,
        "koi_duration": 2.9575,
        "koi_depth": 615.8,
        "koi_ror": 0.02234,
        "koi_srho": 3.2079,
        "koi_prad": 2.26,
        "koi_sma": 0.0853,
        "koi_incl": 89.66,
        "koi_teq": 793.0,
        "koi_insol": 93.59,
        "koi_dor": 24.81,
        "koi_ldm_coeff2": 0.2291,
        "koi_ldm_coeff1": 0.4044,
        "koi_max_sngle_ev": 5.149,
        "koi_max_mult_ev": 28.47,
        "koi_model_snr": 35.8,
        "koi_count": 1,
        "koi_num_transits": 142,
        "koi_tce_plnt_num": 1,
        "koi_bin_oedp_sig": 0.3315,
        "koi_steff": 5455.0,
        "koi_slogg": 4.467,
        "koi_smet": 0.14,
        "koi_srad": 0.927,
        "koi_smass": 0.919,
        "ra": 291.93423,
        "dec": 48.141651,
        "koi_kepmag": 15.347,
        "koi_gmag": 15.89,
        "koi_rmag": 15.27,
        "koi_imag": 15.114,
        "koi_zmag": 15.006,
        "koi_jmag": 14.082,
        "koi_hmag": 13.751,
        "koi_kmag": 13.648,
        "koi_fwm_stat_sig": 0.002,
        "koi_fwm_sra": 0.00047,
        "koi_fwm_sdec": -0.00021,
        "koi_fwm_srao": 0.0003,
        "koi_fwm_sdeco": -0.0002,
        "koi_fwm_prao": 0.0002,
        "koi_fwm_pdeco": -0.0001,
        "koi_dicco_mra": -0.01,
        "koi_dicco_mdec": 0.02,
        "koi_dicco_msky": 0.022,
        "koi_dikco_mra": -0.005,
        "koi_dikco_mdec": 0.015,
        "koi_dikco_msky": 0.016,
    }


@pytest.fixture
def false_positive_koi_payload(sample_koi_payload):
    """Sample KOI with active centroid offset and secondary eclipse flags."""
    fp_payload = sample_koi_payload.copy()
    fp_payload["koi_fpflag_co"] = 1
    fp_payload["koi_fpflag_ss"] = 1
    return fp_payload


# ---------------------------------------------------------------------------
# Test Health, Readiness, and Observability
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data
    assert "gradient_boosting" in data["model_name"]


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "total_predictions" in data
    assert "class_distribution" in data
    assert "avg_latency_ms" in data


# ---------------------------------------------------------------------------
# Test Model Metadata and Inspection
# ---------------------------------------------------------------------------

def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["n_features"] == 55
    assert len(data["features"]) == 55
    assert set(data["classes"]) == {"CANDIDATE", "CONFIRMED", "FALSE POSITIVE"}
    assert data["baseline_comparison"] is not None


def test_feature_importance_endpoint(client):
    response = client.get("/model/features/importance")
    assert response.status_code == 200
    data = response.json()
    assert data["n_features"] == 55
    assert len(data["importances"]) == 55
    # Ensure importances sum to approximately 1.0
    total_imp = sum(item["importance"] for item in data["importances"])
    assert 0.99 <= total_imp <= 1.01
    # Check top feature has valid descriptions
    top_feature = data["importances"][0]
    assert "feature" in top_feature
    assert "importance" in top_feature
    assert "description" in top_feature
    assert "category" in top_feature


def test_confusion_matrix_endpoint(client):
    response = client.get("/model/confusion-matrix")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# Test Single Prediction Endpoint
# ---------------------------------------------------------------------------

def test_predict_single_confirmed(client, sample_koi_payload):
    response = client.post("/predict", json=sample_koi_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}
    assert "probabilities" in data
    assert set(data["probabilities"].keys()) == {"CANDIDATE", "CONFIRMED", "FALSE POSITIVE"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["triage_score"] <= 100.0
    assert data["is_vetting_flagged"] is False
    assert data["inference_time_ms"] >= 0.0
    assert "X-Process-Time-Ms" in response.headers


def test_predict_single_with_explainability(client, sample_koi_payload):
    response = client.post("/predict?explain=true&top_k=5", json=sample_koi_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["top_contributing_features"] is not None
    assert len(data["top_contributing_features"]) == 5
    first_feat = data["top_contributing_features"][0]
    assert first_feat["importance_rank"] == 1
    assert "feature" in first_feat
    assert "description" in first_feat


def test_predict_single_false_positive_vetting(client, false_positive_koi_payload):
    response = client.post("/predict", json=false_positive_koi_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] == "FALSE POSITIVE"
    assert data["is_vetting_flagged"] is True
    assert len(data["flag_summary"]) >= 2
    # Flagged false positives should receive a low triage score
    assert data["triage_score"] <= 15.0


def test_predict_sparse_inputs_imputation(client):
    """Test that submitting only a few features triggers successful median imputation."""
    sparse_payload = {
        "koi_period": 10.5,
        "koi_prad": 1.8,
        "koi_steff": 5700.0,
    }
    response = client.post("/predict", json=sparse_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}


# ---------------------------------------------------------------------------
# Test Batch Prediction Endpoint
# ---------------------------------------------------------------------------

def test_predict_batch_array(client, sample_koi_payload, false_positive_koi_payload):
    batch = [sample_koi_payload, false_positive_koi_payload, {"koi_period": 3.2}]
    response = client.post("/predict-batch", json=batch)
    assert response.status_code == 200
    data = response.json()

    assert data["total_samples"] == 3
    assert len(data["results"]) == 3
    assert sum(data["summary_counts"].values()) == 3
    assert "high_priority_candidates" in data


def test_predict_batch_empty_fails(client):
    response = client.post("/predict-batch", json=[])
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test CSV File Ingestion Endpoint
# ---------------------------------------------------------------------------

def test_predict_csv_json_output(client):
    csv_content = (
        "# NASA Exoplanet Archive Kepler KOI Table\n"
        "# Column definitions...\n"
        "kepid,kepoi_name,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec,koi_period,koi_prad,koi_teq\n"
        "10797460,K00752.01,0,0,0,0,9.488,2.26,793.0\n"
        "10811496,K00755.01,1,0,0,0,2.525,0.78,1406.0\n"
    )

    files = {"file": ("kepler_sample.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/predict/csv?output_format=json", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["total_samples"] == 2
    assert len(data["results"]) == 2


def test_predict_csv_downloadable_csv_output(client):
    csv_content = (
        "kepid,kepoi_name,koi_period,koi_prad\n"
        "10797460,K00752.01,9.488,2.26\n"
        "10811496,K00755.01,2.525,0.78\n"
    )

    files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/predict/csv?output_format=csv", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "predicted_disposition" in response.text
    assert "triage_score" in response.text
    assert "prob_confirmed" in response.text


def test_predict_csv_empty_file_rejected(client):
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    response = client.post("/predict/csv", files=files)
    assert response.status_code == 400
