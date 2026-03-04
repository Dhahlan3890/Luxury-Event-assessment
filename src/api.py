import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_processing import preprocess_single, TARGET_MAP_INV
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_DIR    = Path(os.getenv("MODEL_DIR",    "models"))
FINAL_PATH   = MODEL_DIR / "final_model.pkl"
LOGREG_PATH  = MODEL_DIR / "logreg_model.pkl"
SCALER_PATH  = MODEL_DIR / os.getenv("SCALER_PATH", "models/scaler.pkl").split("/")[-1]
METRICS_PATH = MODEL_DIR / "metrics.json"


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts telecom customer churn status: Churned / Joined / Stayed.\n\n"
        "Two models available:\n"
        "- **Gradient Boosting** (final, better accuracy)\n"
        "- **Logistic Regression** (baseline, interpretable)\n\n"
        "Models are loaded from AWS S3 automatically on startup."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_final_model:  Optional[object] = None
_logreg_model: Optional[object] = None
_metrics:      Dict             = {}
_start_time                     = time.time()



def _download_from_s3(filename: str, local_path: Path):
    """
    Download a single file from S3 bucket into local_path.
    Reads bucket name and region from environment variables.
    Does nothing if AWS_S3_BUCKET is not set.
    """
    import boto3
    from botocore.exceptions import ClientError

    bucket = os.getenv("AWS_S3_BUCKET")
    region = os.getenv("AWS_REGION", "us-east-1")

    if not bucket:
        logger.warning(
            f"AWS_S3_BUCKET not set — cannot download {filename} from S3. "
            "Make sure the file exists locally or set AWS_S3_BUCKET in .env"
        )
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_key = f"models/{filename}"

    logger.info(f"Downloading s3://{bucket}/{s3_key} → {local_path} ...")

    try:
        s3 = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        s3.download_file(bucket, s3_key, str(local_path))
        logger.info(f"✓ Downloaded {filename} from S3 successfully")

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logger.error(f"File not found in S3: s3://{bucket}/{s3_key}")
        elif error_code == "403":
            logger.error(f"Access denied to S3. Check your AWS credentials.")
        else:
            logger.error(f"S3 download failed: {e}")
        raise



def get_final():
    """
    Load the Gradient Boosting model.
    If not found locally, download it from S3 first.
    """
    global _final_model
    if _final_model is None:
        if not FINAL_PATH.exists():
            logger.info("final_model.pkl not found locally — downloading from S3...")
            _download_from_s3("final_model.pkl", FINAL_PATH)

        if not FINAL_PATH.exists():
            raise RuntimeError(
                "final_model.pkl not found locally and could not be downloaded from S3. "
                "Run: python src/train.py --upload-s3"
            )

        _final_model = joblib.load(FINAL_PATH)
        logger.info(f"Gradient Boosting model loaded from {FINAL_PATH}")

    return _final_model


def get_logreg():
    """
    Load the Logistic Regression baseline model.
    If not found locally, download it from S3 first.
    """
    global _logreg_model
    if _logreg_model is None:
        if not LOGREG_PATH.exists():
            logger.info("logreg_model.pkl not found locally — downloading from S3...")
            _download_from_s3("logreg_model.pkl", LOGREG_PATH)

        if not LOGREG_PATH.exists():
            raise RuntimeError(
                "logreg_model.pkl not found locally and could not be downloaded from S3."
            )

        _logreg_model = joblib.load(LOGREG_PATH)
        logger.info(f"Logistic Regression model loaded from {LOGREG_PATH}")

    return _logreg_model


def get_metrics():
    """Load training metrics JSON (for /info and /model-comparison endpoints)."""
    global _metrics
    if not _metrics:
        if not METRICS_PATH.exists():
            _download_from_s3("metrics.json", METRICS_PATH)
        if METRICS_PATH.exists():
            _metrics = json.loads(METRICS_PATH.read_text())
    return _metrics


@app.on_event("startup")
async def startup_event():
    """
    On startup:
    1. Download scaler.pkl from S3 if not local (needed by data_processing.py)
    2. Download metrics.json from S3 if not local
    3. Pre-load both models into memory so first request is instant
    """
    logger.info("Starting up — checking model files...")

    scaler_local = Path(os.getenv("SCALER_PATH", "models/scaler.pkl"))
    if not scaler_local.exists():
        logger.info("scaler.pkl not found locally — downloading from S3...")
        _download_from_s3("scaler.pkl", scaler_local)

    if not METRICS_PATH.exists():
        logger.info("metrics.json not found locally — downloading from S3...")
        _download_from_s3("metrics.json", METRICS_PATH)
    try:
        get_final()
        get_logreg()
        get_metrics()
        logger.info("✓ API startup complete — all models loaded and ready")
    except Exception as e:
        logger.warning(f"Startup warning (API still running): {e}")


class CustomerRecord(BaseModel):
    Gender:                             str   = Field(..., example="Male")
    Age:                                int   = Field(..., example=45)
    Married:                            str   = Field(..., example="Yes")
    Number_of_Dependents:               int   = Field(..., alias="Number of Dependents", example=0)
    City:                               str   = Field(..., example="Los Angeles")
    Number_of_Referrals:                int   = Field(..., alias="Number of Referrals", example=2)
    Tenure_in_Months:                   int   = Field(..., alias="Tenure in Months", example=4)
    Offer:                              str   = Field(..., example="Offer A")
    Phone_Service:                      str   = Field(..., alias="Phone Service", example="Yes")
    Avg_Monthly_Long_Distance_Charges:  float = Field(..., alias="Avg Monthly Long Distance Charges", example=25.5)
    Multiple_Lines:                     str   = Field(..., alias="Multiple Lines", example="No")
    Internet_Service:                   str   = Field(..., alias="Internet Service", example="Yes")
    Internet_Type:                      str   = Field(..., alias="Internet Type", example="Fiber Optic")
    Avg_Monthly_GB_Download:            float = Field(..., alias="Avg Monthly GB Download", example=50.0)
    Online_Security:                    str   = Field(..., alias="Online Security", example="No")
    Online_Backup:                      str   = Field(..., alias="Online Backup", example="Yes")
    Device_Protection_Plan:             str   = Field(..., alias="Device Protection Plan", example="No")
    Premium_Tech_Support:               str   = Field(..., alias="Premium Tech Support", example="Yes")
    Streaming_TV:                       str   = Field(..., alias="Streaming TV", example="No")
    Streaming_Movies:                   str   = Field(..., alias="Streaming Movies", example="No")
    Streaming_Music:                    str   = Field(..., alias="Streaming Music", example="No")
    Unlimited_Data:                     str   = Field(..., alias="Unlimited Data", example="Yes")
    Contract:                           str   = Field(..., example="Month-to-Month")
    Paperless_Billing:                  str   = Field(..., alias="Paperless Billing", example="Yes")
    Payment_Method:                     str   = Field(..., alias="Payment Method", example="Credit Card")
    Monthly_Charge:                     float = Field(..., alias="Monthly Charge", example=75.0)
    Total_Charges:                      float = Field(..., alias="Total Charges", example=300.0)
    Total_Extra_Data_Charges:           float = Field(..., alias="Total Extra Data Charges", example=0.0)
    Total_Long_Distance_Charges:        float = Field(..., alias="Total Long Distance Charges", example=102.0)
    Total_Revenue:                      float = Field(..., alias="Total Revenue", example=402.0)

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    prediction:      str
    prediction_code: int
    probabilities:   Dict[str, float]
    model_used:      str
    model_version:   str = "1.0.0"


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count:       int


def _prepare_features(record: CustomerRecord) -> pd.DataFrame:
    """Convert a CustomerRecord into a preprocessed feature DataFrame."""
    raw = record.dict(by_alias=True)
    try:
        X = preprocess_single(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {e}")

    feature_names = get_metrics().get("feature_names", [])
    if feature_names:
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

    return X


def _make_response(model, X: pd.DataFrame, model_name: str) -> PredictionResponse:
    """Run predict_proba and build a PredictionResponse."""
    proba     = model.predict_proba(X)[0]
    pred_code = int(np.argmax(proba))

    return PredictionResponse(
        prediction=TARGET_MAP_INV[pred_code],
        prediction_code=pred_code,
        probabilities={
            "Churned": round(float(proba[0]), 4),
            "Joined":  round(float(proba[1]), 4),
            "Stayed":  round(float(proba[2]), 4),
        },
        model_used=model_name,
    )


@app.get("/health", tags=["Meta"])
def health():
    """
    Liveness check.
    Returns API uptime and whether each model is loaded into memory.
    """
    return {
        "status":        "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "final_model_ready":  _final_model  is not None,
        "logreg_model_ready": _logreg_model is not None,
        "s3_bucket":     os.getenv("AWS_S3_BUCKET", "not configured"),
    }


@app.get("/info", tags=["Meta"])
def info():
    """
    Returns model metadata: feature names, class labels, train/test sizes.
    """
    m = get_metrics()
    return {
        "final_model":    m.get("final_model"),
        "target_classes": ["Churned", "Joined", "Stayed"],
        "train_size":     m.get("train_size"),
        "test_size":      m.get("test_size"),
        "feature_count":  len(m.get("feature_names", [])),
        "feature_names":  m.get("feature_names", []),
    }


@app.get("/model-comparison", tags=["Meta"])
def model_comparison():
    """
    Side-by-side performance comparison of both trained models:
    Logistic Regression (baseline) vs Gradient Boosting (final).
    """
    m = get_metrics()
    if not m.get("models"):
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run: python src/train.py --upload-s3"
        )

    return {
        "comparison": [
            {
                "model":              r["model"],
                "accuracy":           r["accuracy"],
                "f1_weighted":        r["f1_weighted"],
                "precision_weighted": r["precision_weighted"],
                "recall_weighted":    r["recall_weighted"],
                "cv_mean":            r.get("cv_mean"),
                "cv_std":             r.get("cv_std"),
            }
            for r in m["models"]
        ],
        "selected_model": m.get("final_model"),
        "reasoning": (
            "Final model selected based on highest weighted F1 score. "
            "Gradient Boosting captures non-linear interactions between "
            "features like tenure, contract type, and monthly charge that "
            "Logistic Regression cannot model, and provides better recall "
            "on the minority Joined class."
        ),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(record: CustomerRecord):
    """
    Predict churn for a **single** customer using the FINAL model (Gradient Boosting).

    Returns:
    - **prediction**: Churned / Joined / Stayed
    - **prediction_code**: 0 (Churned) / 1 (Joined) / 2 (Stayed)
    - **probabilities**: confidence score for each class
    - **model_used**: which model made the prediction
    """
    return _make_response(
        get_final(),
        _prepare_features(record),
        "Gradient Boosting"
    )


@app.post("/predict/logreg", response_model=PredictionResponse, tags=["Prediction"])
def predict_logreg(record: CustomerRecord):
    """
    Predict churn using the BASELINE model (Logistic Regression).
    Send the same record to both /predict and /predict/logreg to compare outputs.
    """
    return _make_response(
        get_logreg(),
        _prepare_features(record),
        "Logistic Regression"
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(records: List[CustomerRecord]):
    """
    Predict churn for a **batch** of customers using Gradient Boosting.

    - Accepts a JSON array of CustomerRecord objects
    - Maximum 50,000 records per request
    - Uses vectorised pandas processing — efficient at scale
    """
    if not records:
        raise HTTPException(status_code=400, detail="Empty batch — send at least 1 record.")
    if len(records) > 50_000:
        raise HTTPException(
            status_code=400,
            detail="Batch limit is 50,000 records per request. Split into multiple calls."
        )

    model         = get_final()
    feature_names = get_metrics().get("feature_names", [])

    # Vectorised preprocessing
    try:
        frames  = [preprocess_single(r.dict(by_alias=True)) for r in records]
        X_batch = pd.concat(frames, ignore_index=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {e}")

    # Align feature columns
    if feature_names:
        for col in feature_names:
            if col not in X_batch.columns:
                X_batch[col] = 0
        X_batch = X_batch[feature_names]

    probas     = model.predict_proba(X_batch)
    pred_codes = np.argmax(probas, axis=1)

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                prediction=TARGET_MAP_INV[int(c)],
                prediction_code=int(c),
                probabilities={
                    "Churned": round(float(p[0]), 4),
                    "Joined":  round(float(p[1]), 4),
                    "Stayed":  round(float(p[2]), 4),
                },
                model_used="Gradient Boosting",
            )
            for c, p in zip(pred_codes, probas)
        ],
        count=len(records),
    )