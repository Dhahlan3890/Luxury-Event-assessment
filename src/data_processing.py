"""
Complete preprocessing pipeline:
  - Missing value imputation (median for numerics, mode for categoricals)
  - Feature engineering (3 new features)
  - Label / ordinal encoding
  - StandardScaler for numeric features
  - Train / test split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DROP_COLS = [
    "Customer ID", "Zip Code", "Latitude", "Longitude",
    "Total Refunds", "Churn Category", "Churn Reason",
]

NUMERIC_COLS = [
    "Age", "Number of Dependents", "Number of Referrals", "Tenure in Months",
    "Avg Monthly Long Distance Charges", "Avg Monthly GB Download",
    "Monthly Charge", "Total Charges", "Total Extra Data Charges",
    "Total Long Distance Charges", "Total Revenue",
]

TARGET_COL     = "Customer Status"
TARGET_MAP     = {"Churned": 0, "Joined": 1, "Stayed": 2}
TARGET_MAP_INV = {v: k for k, v in TARGET_MAP.items()}

SCALER_PATH = Path(os.getenv("SCALER_PATH", "models/scaler.pkl"))


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def preprocess(df: pd.DataFrame, fit_scaler: bool = True):
    """
    Full preprocessing pipeline.
    Returns (df_clean, scaler).
    fit_scaler=True during training, False during inference.
    """
    df = df.copy()

    # Step 1: Drop leaky / irrelevant columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    logger.info("Step 1: Dropped leaky/irrelevant columns")

    # Step 2: Rename target
    if TARGET_COL in df.columns:
        df.rename(columns={TARGET_COL: "Customer_Status"}, inplace=True)

    # Step 3: Handle missing values
    # Numeric  -> median (robust to outliers)
    # Categorical -> mode (most frequent class)
    before = df.isnull().sum().sum()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any() and col != "Customer_Status":
            df[col] = df[col].fillna(df[col].mode()[0])
    after = df.isnull().sum().sum()
    logger.info(f"Step 3: Missing values {before} -> {after}")

    # Step 4: Feature engineering
    # FE-1: Service_Diversity - count of active add-on services (stickiness proxy)
    svc_cols = ["Streaming TV", "Streaming Movies", "Streaming Music",
                "Online Security", "Online Backup",
                "Device Protection Plan", "Premium Tech Support"]
    present_svc = [c for c in svc_cols if c in df.columns]
    df["Service_Diversity"] = df[present_svc].isin(["Yes", 1]).sum(axis=1)

    # FE-2: Avg_Revenue_Per_Tenure - revenue efficiency metric
    tenure  = df.get("Tenure in Months", pd.Series(np.ones(len(df))))
    revenue = df.get("Total Revenue",    pd.Series(np.zeros(len(df))))
    df["Avg_Revenue_Per_Tenure"] = (revenue / (tenure + 1e-9)).round(4)

    # FE-3: Is_High_Value - binary flag for high monthly charge customers
    monthly   = df.get("Monthly Charge", pd.Series(np.zeros(len(df))))
    threshold = monthly.quantile(0.75) if fit_scaler else 75.0
    df["Is_High_Value"] = (monthly > threshold).astype(int)

    logger.info("Step 4: Engineered 3 new features")

    # Step 5: Binary Yes/No encoding
    df.replace({"No": 0, "Yes": 1}, inplace=True)

    # Step 6: Gender encoding
    df.replace({"Gender": {"Female": 0, "Male": 1}}, inplace=True)

    # Step 7: Label-encode remaining object columns
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        if col == "Customer_Status":
            continue
        df[col] = le.fit_transform(df[col].astype(str))
    logger.info("Step 7: Label-encoded remaining categoricals")

    # Step 8: Target encoding
    if "Customer_Status" in df.columns:
        df["Customer_Status"] = df["Customer_Status"].map(TARGET_MAP)
        df.dropna(subset=["Customer_Status"], inplace=True)
        df["Customer_Status"] = df["Customer_Status"].astype(int)

    # Step 9: StandardScaler on numeric features
    # Needed for fair model comparison (Logistic Regression is scale-sensitive).
    # XGBoost is scale-invariant but we apply it so both models use same feature space.
    num_cols_present = [c for c in NUMERIC_COLS + ["Avg_Revenue_Per_Tenure"]
                        if c in df.columns]
    if fit_scaler:
        scaler = StandardScaler()
        df[num_cols_present] = scaler.fit_transform(df[num_cols_present])
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"Step 9: StandardScaler fit+saved -> {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        df[num_cols_present] = scaler.transform(df[num_cols_present])
        logger.info("Step 9: StandardScaler transform applied")

    logger.info(f"Preprocessing complete - shape: {df.shape}")
    return df, scaler


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["Customer_Status"])
    y = df["Customer_Status"]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def preprocess_single(record: dict) -> pd.DataFrame:
    df = pd.DataFrame([record])
    drop = DROP_COLS + ["Customer_Status", "Customer Status"]
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True, errors="ignore")

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(0, inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna("Unknown", inplace=True)

    svc_cols = ["Streaming TV", "Streaming Movies", "Streaming Music",
                "Online Security", "Online Backup",
                "Device Protection Plan", "Premium Tech Support"]
    present = [c for c in svc_cols if c in df.columns]
    df["Service_Diversity"] = df[present].isin(["Yes", 1]).sum(axis=1)

    tenure  = float(df.get("Tenure in Months", pd.Series([1])).iloc[0])
    revenue = float(df.get("Total Revenue",    pd.Series([0])).iloc[0])
    df["Avg_Revenue_Per_Tenure"] = revenue / (tenure + 1e-9)

    monthly = float(df.get("Monthly Charge", pd.Series([0])).iloc[0])
    df["Is_High_Value"] = int(monthly > 75.0)

    df.replace({"No": 0, "Yes": 1}, inplace=True)
    df.replace({"Gender": {"Female": 0, "Male": 1}}, inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        num_cols = [c for c in NUMERIC_COLS + ["Avg_Revenue_Per_Tenure"]
                    if c in df.columns]
        df[num_cols] = scaler.transform(df[num_cols])

    return df
