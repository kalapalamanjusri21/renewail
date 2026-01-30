"""
RenewAI - FastAPI backend: renewal probability prediction with explainability + portfolio analytics.
"""

import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_PATH = ROOT_DIR / "data" / "saas_renewal_data.csv"
OUTPUT_PATH = ROOT_DIR / "outputs" / "renewal_portfolio_predictions.csv"

app = FastAPI(
    title="RenewAI",
    description="SaaS subscription renewal probability API with explainability",
    version="1.0.0",
)

# Load model at startup
_model_artifact = None


def get_model():
    global _model_artifact
    if _model_artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run: python model/train_model.py")
        _model_artifact = joblib.load(MODEL_PATH)
    return _model_artifact


# Request/response schemas
FEATURE_ORDER = [
    "avg_weekly_logins",
    "feature_adoption_pct",
    "support_tickets",
    "avg_ticket_sentiment",
    "contract_age_days",
    "renewal_window_days",
]


class PredictRequest(BaseModel):
    avg_weekly_logins: int = Field(..., ge=0, description="Average weekly logins")
    feature_adoption_pct: float = Field(..., ge=0, le=1, description="Feature adoption 0-1")
    support_tickets: int = Field(..., ge=0, description="Number of support tickets")
    avg_ticket_sentiment: float = Field(..., ge=-1, le=1, description="Avg ticket sentiment -1 to 1")
    contract_age_days: int = Field(..., ge=0, description="Days since contract start")
    renewal_window_days: int = Field(..., ge=0, description="Days until renewal")


class FactorExplanation(BaseModel):
    feature: str
    value: float
    coefficient: float
    contribution: float
    direction: str


class PredictResponse(BaseModel):
    renewal_probability: float
    risk_label: str
    top_factors: List[FactorExplanation]


class PortfolioSummaryResponse(BaseModel):
    total_customers: int
    will_renew: int
    at_risk: int
    likely_to_churn: int
    avg_renewal_probability: float
    portfolio_health_score: float
    generated_at: str
    will_renew_pct: float = 0.0
    at_risk_pct: float = 0.0
    churn_pct: float = 0.0
    segments_by_size: Dict[str, Dict] = {}
    segments_by_industry: Dict[str, Dict] = {}
    segments_by_region: Dict[str, Dict] = {}


class HighRiskCustomerResponse(BaseModel):
    account_id: str
    renewal_probability_pct: float
    risk_label: str
    avg_weekly_logins: int
    feature_adoption_pct: float
    support_tickets: int
    avg_ticket_sentiment: float
    contract_age_days: int
    renewal_window_days: int


class ChurnReasonResponse(BaseModel):
    reasons: Dict[str, float]
    primary_reason: str
    risk_score: float


class InterventionResponse(BaseModel):
    account_id: str
    risk_label: str
    interventions: List[Dict[str, Any]]
    estimated_impact: float



def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "Low"
    if prob >= 0.4:
        return "Medium"
    return "High"


# Portfolio data loading cache
_portfolio_cache = None
_portfolio_cache_time = None


def load_portfolio_data():
    """Load cached portfolio predictions; refresh if file exists and is newer."""
    global _portfolio_cache, _portfolio_cache_time
    
    if OUTPUT_PATH.exists():
        file_mtime = OUTPUT_PATH.stat().st_mtime
        if _portfolio_cache is not None and _portfolio_cache_time == file_mtime:
            return _portfolio_cache  # Return cached data
        
        df = pd.read_csv(OUTPUT_PATH)
        _portfolio_cache = df
        _portfolio_cache_time = file_mtime
        return df
    
    return None  # Portfolio not yet generated


def get_portfolio_metrics() -> Dict[str, Any]:
    """Compute portfolio health metrics from cached predictions."""
    df = load_portfolio_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Portfolio predictions not yet generated. Run: python analytics/batch_predict.py")
    
    total = len(df)
    will_renew = (df["risk_label"] == "Will Renew").sum()
    at_risk = (df["risk_label"] == "At Risk").sum()
    likely_churn = (df["risk_label"] == "Likely to Churn").sum()
    avg_prob = df["renewal_probability"].mean()
    health_score = (will_renew / total * 100) if total > 0 else 0.0
    
    # Segment analysis
    segments_by_size = {}
    segments_by_industry = {}
    segments_by_region = {}
    
    if "company_size" in df.columns:
        for size in df["company_size"].unique():
            size_df = df[df["company_size"] == size]
            segments_by_size[size] = {
                "total": int(len(size_df)),
                "churn_pct": round((size_df["risk_label"] == "Likely to Churn").sum() / len(size_df) * 100, 1)
            }
    
    if "industry" in df.columns:
        for ind in df["industry"].unique():
            ind_df = df[df["industry"] == ind]
            segments_by_industry[ind] = {
                "total": int(len(ind_df)),
                "churn_pct": round((ind_df["risk_label"] == "Likely to Churn").sum() / len(ind_df) * 100, 1)
            }
    
    if "region" in df.columns:
        for reg in df["region"].unique():
            reg_df = df[df["region"] == reg]
            segments_by_region[reg] = {
                "total": int(len(reg_df)),
                "churn_pct": round((reg_df["risk_label"] == "Likely to Churn").sum() / len(reg_df) * 100, 1)
            }
    
    return {
        "total_customers": int(total),
        "will_renew": int(will_renew),
        "at_risk": int(at_risk),
        "likely_to_churn": int(likely_churn),
        "will_renew_pct": round(will_renew / total * 100, 1),
        "at_risk_pct": round(at_risk / total * 100, 1),
        "churn_pct": round(likely_churn / total * 100, 1),
        "avg_renewal_probability": round(avg_prob, 4),
        "portfolio_health_score": round(health_score, 2),
        "generated_at": datetime.utcnow().isoformat(),
        "segments_by_size": segments_by_size,
        "segments_by_industry": segments_by_industry,
        "segments_by_region": segments_by_region,
    }


def get_high_risk_customers(top_n: int = 10) -> List[Dict[str, Any]]:
    """Get top N customers with lowest renewal probability (high risk)."""
    df = load_portfolio_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Portfolio predictions not yet generated. Run: python analytics/batch_predict.py")
    
    high_risk = df.nsmallest(top_n, "renewal_probability")
    
    result = []
    for _, row in high_risk.iterrows():
        result.append({
            "account_id": row["account_id"],
            "renewal_probability_pct": float(row["renewal_probability_pct"]),
            "risk_label": row["risk_label"],
            "avg_weekly_logins": int(row["avg_weekly_logins"]),
            "feature_adoption_pct": float(row["feature_adoption_pct"]),
            "support_tickets": int(row["support_tickets"]),
            "avg_ticket_sentiment": float(row["avg_ticket_sentiment"]),
            "contract_age_days": int(row["contract_age_days"]),
            "renewal_window_days": int(row["renewal_window_days"]),
        })
    
    return result


def parse_churn_reasons_from_csv(reasons_str: str) -> Dict[str, float]:
    """Parse churn reasons from CSV string format"""
    try:
        import ast
        return ast.literal_eval(reasons_str) if reasons_str and isinstance(reasons_str, str) else {}
    except:
        return {}


def get_churn_analysis(account_id: str) -> Dict[str, Any]:
    """Get detailed churn analysis for a specific customer"""
    df = load_portfolio_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Portfolio predictions not yet generated")
    
    customer = df[df["account_id"] == account_id]
    if len(customer) == 0:
        raise HTTPException(status_code=404, detail=f"Customer {account_id} not found")
    
    row = customer.iloc[0]
    churn_reasons = parse_churn_reasons_from_csv(row.get("churn_reasons", "{}"))
    
    primary_reason = list(churn_reasons.keys())[0] if churn_reasons else "unknown"
    risk_score = max(churn_reasons.values()) if churn_reasons else 0.0
    
    return {
        "account_id": row["account_id"],
        "reasons": churn_reasons,
        "primary_reason": primary_reason,
        "risk_score": risk_score
    }


def get_interventions_for_customer(account_id: str) -> Dict[str, Any]:
    """Get recommended interventions for a specific customer"""
    df = load_portfolio_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Portfolio predictions not yet generated")
    
    customer = df[df["account_id"] == account_id]
    if len(customer) == 0:
        raise HTTPException(status_code=404, detail=f"Customer {account_id} not found")
    
    row = customer.iloc[0]
    interventions_str = row.get("interventions", "[]")
    
    try:
        import ast
        interventions = ast.literal_eval(interventions_str) if isinstance(interventions_str, str) else []
    except:
        interventions = []
    
    # Calculate estimated impact (average of all intervention impacts)
    estimated_impact = np.mean([i[2] for i in interventions]) if interventions else 0.0
    
    return {
        "account_id": row["account_id"],
        "risk_label": row["risk_label"],
        "interventions": [
            {
                "action": i[0],
                "priority": i[1],
                "estimated_impact": i[2]
            } for i in interventions
        ],
        "estimated_impact": estimated_impact
    }



def explain_prediction_lr(model, feature_names, x_raw: np.ndarray, x_scaled: np.ndarray, top_k: int = 3) -> list[dict]:
    """Logistic Regression: top-k by coefficient * scaled value."""
    coefs = model.coef_[0]
    contribs = coefs * x_scaled[0]
    order = np.argsort(np.abs(contribs))[::-1]
    out = []
    for i in order[:top_k]:
        c = contribs[i]
        direction = "increases" if c > 0 else "decreases"
        out.append({
            "feature": feature_names[i],
            "value": float(x_raw[0, i]),
            "coefficient": float(coefs[i]),
            "contribution": float(c),
            "direction": direction,
        })
    return out


def explain_prediction_rf(
    feature_names: list,
    x_raw: np.ndarray,
    feature_importances_: np.ndarray,
    feature_means_: np.ndarray,
    feature_directions_: np.ndarray,
    top_k: int = 3,
) -> list[dict]:
    """Random Forest: top-k by importance; direction from (value - mean) * feature_direction."""
    order = np.argsort(feature_importances_)[::-1]
    out = []
    for i in order[:top_k]:
        val = float(x_raw[0, i])
        mean_i = float(feature_means_[i])
        direction_mult = (val - mean_i) * feature_directions_[i]
        direction = "increases" if direction_mult > 0 else "decreases"
        imp = float(feature_importances_[i])
        out.append({
            "feature": feature_names[i],
            "value": val,
            "coefficient": imp,
            "contribution": imp * (1 if direction_mult > 0 else -1),
            "direction": direction,
        })
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict renewal probability and return risk label + top 3 factors."""
    try:
        artifact = get_model()
        model = artifact["model"]
        feature_names = artifact["feature_names"]
        scaler = artifact.get("scaler")
        feature_importances_ = artifact.get("feature_importances_")
        feature_means_ = artifact.get("feature_means_")
        feature_directions_ = artifact.get("feature_directions_")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    x_raw = np.array([[
        req.avg_weekly_logins,
        req.feature_adoption_pct,
        req.support_tickets,
        req.avg_ticket_sentiment,
        req.contract_age_days,
        req.renewal_window_days,
    ]], dtype=np.float64)

    if feature_importances_ is not None:
        # Random Forest: no scaling
        x = x_raw
        factors = explain_prediction_rf(
            feature_names, x_raw,
            np.array(feature_importances_),
            np.array(feature_means_),
            np.array(feature_directions_),
            top_k=3,
        )
    else:
        # Logistic Regression: scale then predict and explain
        x = scaler.transform(x_raw) if scaler is not None else x_raw
        factors = explain_prediction_lr(model, feature_names, x_raw, x, top_k=3)

    prob = float(model.predict_proba(x)[0, 1])
    pct = round(prob * 100, 2)
    label = risk_label(prob)
    return PredictResponse(
        renewal_probability=pct,
        risk_label=label,
        top_factors=[FactorExplanation(**f) for f in factors],
    )


@app.get("/portfolio-summary", response_model=PortfolioSummaryResponse)
def portfolio_summary():
    """
    Get renewal portfolio health summary.
    
    Returns aggregated metrics across all customers:
    - Total customers
    - Count of Will Renew / At Risk / Likely to Churn
    - Average renewal probability
    - Portfolio health score (% that will renew)
    """
    metrics = get_portfolio_metrics()
    return PortfolioSummaryResponse(**metrics)


@app.get("/high-risk-customers", response_model=List[HighRiskCustomerResponse])
def high_risk_customers(top_n: int = 10):
    """
    Get top N customers with lowest renewal probability.
    
    Useful for customer success teams to prioritize interventions.
    
    Query parameters:
    - top_n: Number of high-risk customers to return (default: 10)
    """
    if top_n < 1 or top_n > 500:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 500")
    
    customers = get_high_risk_customers(top_n=top_n)
    return [HighRiskCustomerResponse(**c) for c in customers]


@app.get("/churn-analysis/{account_id}", response_model=ChurnReasonResponse)
def churn_analysis(account_id: str):
    """
    Get detailed churn analysis for a specific customer.
    
    Returns:
    - reasons: Dictionary of churn drivers with impact scores (0-1)
    - primary_reason: Most significant churn driver
    - risk_score: Highest impact score among all reasons
    
    Path parameters:
    - account_id: Customer account identifier
    """
    analysis = get_churn_analysis(account_id)
    return ChurnReasonResponse(
        reasons=analysis["reasons"],
        primary_reason=analysis["primary_reason"],
        risk_score=analysis["risk_score"]
    )


@app.get("/interventions/{account_id}", response_model=InterventionResponse)
def interventions(account_id: str):
    """
    Get recommended interventions for a specific customer.
    
    Returns:
    - interventions: List of recommended actions with priority and estimated impact
    - estimated_impact: Average effectiveness of recommended actions
    - risk_label: Customer risk category
    
    Path parameters:
    - account_id: Customer account identifier
    """
    intervention_data = get_interventions_for_customer(account_id)
    return InterventionResponse(
        account_id=intervention_data["account_id"],
        risk_label=intervention_data["risk_label"],
        interventions=intervention_data["interventions"],
        estimated_impact=intervention_data["estimated_impact"]
    )



# Serve frontend (explicit routes so /predict and /health stay API-only)
if FRONTEND_DIR.exists():
    @app.get("/")
    def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/style.css")
    def serve_css():
        return FileResponse(FRONTEND_DIR / "style.css")

    @app.get("/app.js")
    def serve_js():
        return FileResponse(FRONTEND_DIR / "app.js")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
