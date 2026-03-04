from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Optional
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Luxury Event assessment API",
    description="Predicts probability of customer churn using XGBoost model",
    version="1.0.0"
)

class CustomerInput(BaseModel):
    tenure_in_months: int = Field(..., ge=0, description="Tenure in months")
    monthly_charge: float = Field(..., gt=0, description="Monthly charge amount")
    total_charges: float = Field(..., ge=0, description="Total charges so far")
    contract: str = Field(..., pattern="^(Month-to-Month|One Year|Two Year)$",
                          description="Contract type")
    internet_service: str = Field(..., pattern="^(Yes|No)$",
                                  description="Has internet service?")
    internet_type: Optional[str] = Field(None, pattern="^(Fiber Optic|DSL|None|)$")
    online_security: str = Field(..., pattern="^(Yes|No)$")
    online_backup: str = Field(..., pattern="^(Yes|No)$")
    device_protection_plan: str = Field(..., pattern="^(Yes|No)$")
    premium_tech_support: str = Field(..., pattern="^(Yes|No)$")
    streaming_tv: str = Field(..., pattern="^(Yes|No)$")
    streaming_movies: str = Field(..., pattern="^(Yes|No)$")
    paperless_billing: str = Field(..., pattern="^(Yes|No)$")
    payment_method: str = Field(..., pattern="^(Bank Withdrawal|Credit Card|)$")
    number_of_referrals: int = Field(0, ge=0)
    avg_monthly_long_distance_charges: float = Field(0.0, ge=0)
    avg_monthly_gb_download: Optional[float] = Field(None, ge=0)

    age: Optional[int] = Field(None, ge=18)
    married: Optional[str] = Field(None, pattern="^(Yes|No)$")
    number_of_dependents: Optional[int] = Field(None, ge=0)

    class Config:
        schema_extra = {
            "example": {
                "tenure_in_months": 12,
                "monthly_charge": 84.95,
                "total_charges": 1021.95,
                "contract": "Month-to-Month",
                "internet_service": "Yes",
                "internet_type": "Fiber Optic",
                "online_security": "No",
                "online_backup": "No",
                "device_protection_plan": "Yes",
                "premium_tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "Yes",
                "paperless_billing": "Yes",
                "payment_method": "Bank Withdrawal",
                "number_of_referrals": 0,
                "avg_monthly_long_distance_charges": 25.4,
                "avg_monthly_gb_download": 45.0
            }
        }


model = None


@app.on_event("startup")
async def load_model():
    global model
    try:
        model = xgb.XGBClassifier()
        model.load_model("xgb_model.json")
        print("XGBoost model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Failed to load XGBoost model")


def preprocess_input(data: CustomerInput) -> pd.DataFrame:
    row = {
        'Tenure in Months': data.tenure_in_months,
        'Monthly Charge': data.monthly_charge,
        'Total Charges': data.total_charges,
        'Contract': data.contract,
        'Internet Service': data.internet_service,
        'Internet Type': data.internet_type or "None",
        'Online Security': data.online_security,
        'Online Backup': data.online_backup,
        'Device Protection Plan': data.device_protection_plan,
        'Premium Tech Support': data.premium_tech_support,
        'Streaming TV': data.streaming_tv,
        'Streaming Movies': data.streaming_movies,
        'Paperless Billing': data.paperless_billing,
        'Payment Method': data.payment_method,
        'Number of Referrals': data.number_of_referrals,
        'Avg Monthly Long Distance Charges': data.avg_monthly_long_distance_charges,
        'Avg Monthly GB Download': data.avg_monthly_gb_download or 0.0,
    }

    if data.age is not None:
        row['Age'] = data.age
    if data.married is not None:
        row['Married'] = data.married
    if data.number_of_dependents is not None:
        row['Number of Dependents'] = data.number_of_dependents

    df = pd.DataFrame([row])

    contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    yes_no_map = {"Yes": 1, "No": 0}

    df['Contract'] = df['Contract'].map(contract_map)
    for col in [
        'Internet Service', 'Online Security', 'Online Backup',
        'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
        'Streaming Movies', 'Paperless Billing'
    ]:
        df[col] = df[col].map(yes_no_map)

    internet_types = ['DSL', 'Fiber Optic', 'None']
    for t in internet_types:
        df[f'Internet Type_{t}'] = (df['Internet Type'] == t).astype(int)
    df = df.drop(columns=['Internet Type'], errors='ignore')

    payment_map = {"Bank Withdrawal": 0, "Credit Card": 1, "Mailed Check": 2}
    df['Payment Method'] = df['Payment Method'].map(payment_map).fillna(0)

    expected_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]

    return df


@app.post("/predict", response_class=JSONResponse)
async def predict_churn(customer: CustomerInput):
    """
    Predicts churn probability for a single customer
    Returns probability and binary decision (churn / stay)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = preprocess_input(customer)

        prob = model.predict_proba(X)[0][1]          
        prediction = model.predict(X)[0]             

        return {
            "churn_probability": round(float(prob), 4),
            "will_churn": bool(prediction),
            "customer_input": customer.dict(),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)