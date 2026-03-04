import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score

from data_processing import load_data, preprocess, split_data
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR     = Path(os.getenv("MODEL_DIR", "models"))
FINAL_PATH    = MODEL_DIR / "final_model.pkl"
LOGREG_PATH   = MODEL_DIR / "logreg_model.pkl"
METRICS_PATH  = Path(os.getenv("METRICS_PATH", "models/metrics.json"))
CM_PLOT_PATH  = MODEL_DIR / "confusion_matrices.png"
CLASS_NAMES   = ["Churned", "Joined", "Stayed"]


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_w   = f1_score(y_test, y_pred, average="weighted")
    prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"\n{'='*55}")
    logger.info(f"  {model_name}")
    logger.info(f"  Accuracy         : {acc:.4f}")
    logger.info(f"  F1 (weighted)    : {f1_w:.4f}")
    logger.info(f"  Precision (w)    : {prec_w:.4f}")
    logger.info(f"  Recall (w)       : {rec_w:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")

    return {
        "model":              model_name,
        "accuracy":           round(acc, 4),
        "f1_weighted":        round(f1_w, 4),
        "precision_weighted": round(prec_w, 4),
        "recall_weighted":    round(rec_w, 4),
        "confusion_matrix":   cm,
        "classification_report": report,
    }


def plot_confusion_matrices(results: list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison — Confusion Matrices", fontsize=14, fontweight="bold")
    for ax, res in zip(axes, results):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f"{res['model']}\nAcc={res['accuracy']:.3f}  F1={res['f1_weighted']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    CM_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CM_PLOT_PATH, dpi=120)
    plt.close()
    logger.info(f"Confusion matrix plot saved -> {CM_PLOT_PATH}")


def train(data_path: str, upload_s3: bool = False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + preprocess ─────────────────────────────────────────────
    df_raw = load_data(data_path)
    df, _  = preprocess(df_raw, fit_scaler=True)

    # ── 2. Split ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    logger.info(f"Class distribution (train):\n{y_train.value_counts().to_string()}")

    results = []

    # ══════════════════════════════════════════════════════════════════════
    # MODEL 1: Logistic Regression (Baseline)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\nTraining Model 1: Logistic Regression (baseline)...")
    logreg = LogisticRegression(
        max_iter=1000,
        solver="lbfgs", random_state=42, C=1.0
    )
    lr_cv = cross_val_score(logreg, X_train, y_train, cv=5, scoring="accuracy")
    logger.info(f"LR CV Accuracy: {lr_cv.mean():.4f} +/- {lr_cv.std():.4f}")
    logreg.fit(X_train, y_train)
    lr_result = evaluate(logreg, X_test, y_test, "Logistic Regression")
    lr_result["cv_mean"] = round(float(lr_cv.mean()), 4)
    lr_result["cv_std"]  = round(float(lr_cv.std()),  4)
    results.append(lr_result)
    joblib.dump(logreg, LOGREG_PATH)
    logger.info(f"Logistic Regression saved -> {LOGREG_PATH}")

    # ══════════════════════════════════════════════════════════════════════
    # MODEL 2: Gradient Boosting Classifier (Final Model)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\nTraining Model 2: Gradient Boosting Classifier (final)...")
    gbt = GradientBoostingClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.1, subsample=0.8,
        random_state=42
    )
    gbt_cv = cross_val_score(gbt, X_train, y_train, cv=5, scoring="accuracy")
    logger.info(f"GBT CV Accuracy: {gbt_cv.mean():.4f} +/- {gbt_cv.std():.4f}")
    gbt.fit(X_train, y_train)
    gbt_result = evaluate(gbt, X_test, y_test, "Gradient Boosting")
    gbt_result["cv_mean"] = round(float(gbt_cv.mean()), 4)
    gbt_result["cv_std"]  = round(float(gbt_cv.std()),  4)
    results.append(gbt_result)
    joblib.dump(gbt, FINAL_PATH)
    logger.info(f"Gradient Boosting model saved -> {FINAL_PATH}")

    # ── 3. Comparison summary ─────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("  MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"  {'Model':<28} {'Accuracy':>9} {'F1 (w)':>9} {'CV Mean':>9}")
    logger.info(f"  {'-'*57}")
    for r in results:
        logger.info(f"  {r['model']:<28} {r['accuracy']:>9.4f} "
                    f"{r['f1_weighted']:>9.4f} {r['cv_mean']:>9.4f}")
    logger.info("="*60)
    winner = max(results, key=lambda r: r["f1_weighted"])
    logger.info(f"\n  SELECTED: {winner['model']} (F1={winner['f1_weighted']:.4f})")

    # ── 4. Confusion matrix plot ──────────────────────────────────────────
    plot_confusion_matrices(results)

    # ── 5. Save metrics ───────────────────────────────────────────────────
    metrics = {
        "models":        results,
        "final_model":   winner["model"],
        "feature_names": list(X_train.columns),
        "train_size":    len(y_train),
        "test_size":     len(y_test),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved -> {METRICS_PATH}")

    # ── 6. Optional S3 upload ─────────────────────────────────────────────
    if upload_s3:
        for p in [FINAL_PATH, LOGREG_PATH, METRICS_PATH.parent/"scaler.pkl"]:
            _upload_to_s3(str(p))

    return gbt, logreg, metrics


def _upload_to_s3(local_path: str):
    import boto3
    bucket = os.getenv("AWS_S3_BUCKET")
    region = os.getenv("AWS_REGION", "us-east-1")
    if not bucket:
        logger.warning("AWS_S3_BUCKET not set - skipping S3 upload"); return
    key = f"models/{Path(local_path).name}"
    boto3.client("s3", region_name=region).upload_file(local_path, bucket, key)
    logger.info(f"Uploaded -> s3://{bucket}/{key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telecom_customer_churn.csv")
    parser.add_argument("--upload-s3", action="store_true")
    args = parser.parse_args()
    train(data_path=args.data, upload_s3=args.upload_s3)
