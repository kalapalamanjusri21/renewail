"""
RenewAI - Batch Portfolio Prediction & Analytics Engine (Enhanced)

Features:
- Batch predictions for all customers
- Risk categorization (Will Renew / At Risk / Likely to Churn)
- Churn reason analysis
- Intervention recommendations
- Portfolio-level metrics
- BigQuery-ready export

Outputs:
- renewal_portfolio_predictions.csv (primary)
- Portfolio metrics aggregations
- High-risk customer lists with interventions
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List, Tuple

# Paths
ANALYTICS_DIR = Path(__file__).resolve().parent
ROOT_DIR = ANALYTICS_DIR.parent
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
DATA_PATH = ROOT_DIR / "data" / "saas_renewal_data.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "renewal_portfolio_predictions.csv"

# Feature columns (must match training)
FEATURE_COLUMNS = [
    "avg_weekly_logins",
    "feature_adoption_pct",
    "support_tickets",
    "avg_ticket_sentiment",
    "contract_age_days",
    "renewal_window_days",
    "logins_trend",
    "days_since_last_login",
    "adoption_trend",
]

CATEGORICAL_COLUMNS = [
    "contract_stage",
    "company_size",
    "industry",
    "region",
    "payment_status",
]


def load_model_and_scaler():
    """Load the trained model artifact"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run: python model/train_model.py")
    
    artifact = joblib.load(MODEL_PATH)
    return artifact


def load_customer_data(path=None):
    """Load all customer records from dataset."""
    path = path or DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}. Run: python data/generate_data.py")
    
    df = pd.read_csv(path)
    return df


def analyze_churn_reasons(customer_row: pd.Series, renewal_prob: float) -> Dict[str, float]:
    """
    Analyze which factors contribute to churn risk.
    Returns dict with risk drivers and their scores (0-1).
    """
    
    reasons = {}
    
    # Engagement factors
    if customer_row["avg_weekly_logins"] < 2:
        reasons["low_engagement"] = 0.8
    elif customer_row["avg_weekly_logins"] < 5:
        reasons["low_engagement"] = 0.4
    else:
        reasons["low_engagement"] = 0.0
    
    # Feature adoption (biggest predictor)
    if customer_row["feature_adoption_pct"] < 20:
        reasons["low_adoption"] = 0.9
    elif customer_row["feature_adoption_pct"] < 40:
        reasons["low_adoption"] = 0.6
    elif customer_row["feature_adoption_pct"] < 60:
        reasons["low_adoption"] = 0.3
    else:
        reasons["low_adoption"] = 0.0
    
    # Support sentiment
    if customer_row["avg_ticket_sentiment"] < -0.3:
        reasons["negative_support_sentiment"] = 0.8
    elif customer_row["avg_ticket_sentiment"] < 0:
        reasons["negative_support_sentiment"] = 0.4
    else:
        reasons["negative_support_sentiment"] = 0.0
    
    # Support volume (high tickets = problems)
    if customer_row["support_tickets"] > 20:
        reasons["high_support_volume"] = 0.7
    elif customer_row["support_tickets"] > 10:
        reasons["high_support_volume"] = 0.3
    else:
        reasons["high_support_volume"] = 0.0
    
    # Contract lifecycle
    contract_stage = customer_row.get("contract_stage", "Growth")
    if contract_stage == "Legacy":
        reasons["legacy_contract"] = 0.5
    elif contract_stage == "Mature":
        reasons["mature_contract"] = 0.2
    else:
        reasons["legacy_contract"] = 0.0
        reasons["mature_contract"] = 0.0
    
    # Adoption trend
    if customer_row.get("adoption_trend", 0) < -0.1:
        reasons["declining_adoption"] = 0.8
    elif customer_row.get("adoption_trend", 0) < 0:
        reasons["declining_adoption"] = 0.3
    else:
        reasons["declining_adoption"] = 0.0
    
    # Login trend
    if customer_row.get("logins_trend", 0) < -0.2:
        reasons["declining_engagement"] = 0.8
    elif customer_row.get("logins_trend", 0) < 0:
        reasons["declining_engagement"] = 0.3
    else:
        reasons["declining_engagement"] = 0.0
    
    # Days since last login
    days_since_login = customer_row.get("days_since_last_login", 0)
    if days_since_login > 30:
        reasons["dormant_account"] = 0.9
    elif days_since_login > 14:
        reasons["dormant_account"] = 0.6
    elif days_since_login > 7:
        reasons["dormant_account"] = 0.2
    else:
        reasons["dormant_account"] = 0.0
    
    # Payment status risk
    payment_status = customer_row.get("payment_status", "Active")
    if payment_status == "Past_Due":
        reasons["payment_overdue"] = 0.9
    elif payment_status == "At_Risk":
        reasons["payment_risk"] = 0.5
    else:
        reasons["payment_overdue"] = 0.0
        reasons["payment_risk"] = 0.0
    
    # Company size risk (SMBs churn more)
    company_size = customer_row.get("company_size", "SMB")
    if company_size == "SMB":
        reasons["smb_customer"] = 0.3
    else:
        reasons["smb_customer"] = 0.0
    
    # Keep only non-zero reasons, sorted by impact
    reasons = {k: v for k, v in reasons.items() if v > 0}
    return dict(sorted(reasons.items(), key=lambda x: -x[1]))


def recommend_interventions(
    risk_label: str,
    churn_reasons: Dict[str, float],
    customer_row: pd.Series
) -> List[Tuple[str, str, float]]:
    """
    Generate intervention recommendations based on risk profile.
    Returns list of (action, priority, estimated_impact) tuples.
    """
    
    interventions = []
    
    # LIKELY TO CHURN - Critical interventions
    if risk_label == "Likely to Churn":
        if "low_adoption" in churn_reasons:
            interventions.append(("Feature Training Session", "CRITICAL", 0.6))
            interventions.append(("Dedicated CSM Call", "CRITICAL", 0.5))
        
        if "negative_support_sentiment" in churn_reasons:
            interventions.append(("Support Escalation", "CRITICAL", 0.7))
            interventions.append(("Executive Check-in", "HIGH", 0.4))
        
        if "dormant_account" in churn_reasons:
            interventions.append(("Re-engagement Campaign", "CRITICAL", 0.8))
            interventions.append(("Product Demo Update", "HIGH", 0.5))
        
        if "payment_overdue" in churn_reasons:
            interventions.append(("Payment Recovery", "CRITICAL", 0.9))
            interventions.append(("Flexible Pricing Discussion", "HIGH", 0.6))
        
        if "declining_adoption" in churn_reasons or "declining_engagement" in churn_reasons:
            interventions.append(("Onboarding Review", "HIGH", 0.5))
            interventions.append(("Success Metrics Review", "HIGH", 0.4))
    
    # AT RISK - Preventive interventions
    elif risk_label == "At Risk":
        if "low_adoption" in churn_reasons:
            interventions.append(("Onboarding Session", "HIGH", 0.5))
            interventions.append(("Use Case Training", "MEDIUM", 0.4))
        
        if "high_support_volume" in churn_reasons:
            interventions.append(("Support Review Meeting", "MEDIUM", 0.4))
        
        if "legacy_contract" in churn_reasons:
            interventions.append(("Contract Renewal Discussion", "MEDIUM", 0.3))
        
        if "dormant_account" in churn_reasons:
            interventions.append(("Email Campaign", "MEDIUM", 0.3))
        
        if "smb_customer" in churn_reasons:
            interventions.append(("SMB Success Program", "MEDIUM", 0.2))
    
    # WILL RENEW - Nurture & upsell
    elif risk_label == "Will Renew":
        if customer_row["feature_adoption_pct"] > 70:
            interventions.append(("Upsell Conversation", "LOW", 0.3))
            interventions.append(("Advanced Features Demo", "LOW", 0.2))
        
        interventions.append(("Renewal Celebration", "LOW", 0.1))
    
    # Remove duplicates and sort by priority
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    interventions = list(set(interventions))
    interventions.sort(key=lambda x: (priority_order[x[1]], -x[2]))
    
    return interventions[:3]  # Top 3 interventions


def categorize_risk(renewal_probability: float) -> str:
    """Assign risk label based on renewal probability"""
    if renewal_probability >= 0.70:
        return "Will Renew"
    elif renewal_probability >= 0.40:
        return "At Risk"
    else:
        return "Likely to Churn"


def batch_predict_portfolio(artifact: dict, df_customers: pd.DataFrame) -> pd.DataFrame:
    """Predict renewal probability for entire portfolio"""
    
    # Prepare features with one-hot encoding
    df_encoded = pd.get_dummies(
        df_customers[FEATURE_COLUMNS + CATEGORICAL_COLUMNS],
        columns=CATEGORICAL_COLUMNS,
        drop_first=True
    )
    
    X = df_encoded.values
    model = artifact["model"]
    scaler = artifact.get("scaler")
    optimal_threshold = artifact.get("optimal_threshold", 0.5)
    
    if scaler:
        X = scaler.transform(X)
    
    renewal_probs = model.predict_proba(X)[:, 1]
    
    # Categorize risks
    risk_labels = [categorize_risk(p) for p in renewal_probs]
    
    # Analyze churn reasons and interventions
    churn_analysis = []
    for idx, row in df_customers.iterrows():
        churn_reasons = analyze_churn_reasons(row, renewal_probs[idx])
        interventions = recommend_interventions(risk_labels[idx], churn_reasons, row)
        churn_analysis.append({
            "churn_reasons": churn_reasons,
            "interventions": interventions
        })
    
    # Build results DataFrame
    result = pd.DataFrame({
        "account_id": df_customers["account_id"],
        "renewal_probability": renewal_probs,
        "renewal_probability_pct": (renewal_probs * 100).round(2),
        "risk_label": risk_labels,
        "churn_reasons": [str(a["churn_reasons"]) for a in churn_analysis],
        "interventions": [str(a["interventions"]) for a in churn_analysis],
    })
    
    return result


def compute_portfolio_metrics(predictions: pd.DataFrame, df_customers: pd.DataFrame = None) -> dict:
    """Calculate portfolio-level aggregations with segment analysis"""
    
    total = len(predictions)
    will_renew = (predictions["risk_label"] == "Will Renew").sum()
    at_risk = (predictions["risk_label"] == "At Risk").sum()
    likely_churn = (predictions["risk_label"] == "Likely to Churn").sum()
    
    # Merge with customer data for segment analysis
    if df_customers is not None:
        merged = predictions.merge(df_customers, on="account_id")
    else:
        merged = predictions
    
    # Segment breakdown
    segments_by_size = {}
    segments_by_industry = {}
    segments_by_region = {}
    
    if "company_size" in merged.columns:
        for size in merged["company_size"].unique():
            size_df = merged[merged["company_size"] == size]
            segments_by_size[size] = {
                "total": len(size_df),
                "churn_pct": round((size_df["risk_label"] == "Likely to Churn").sum() / len(size_df) * 100, 1)
            }
    
    if "industry" in merged.columns:
        for ind in merged["industry"].unique():
            ind_df = merged[merged["industry"] == ind]
            segments_by_industry[ind] = {
                "total": len(ind_df),
                "churn_pct": round((ind_df["risk_label"] == "Likely to Churn").sum() / len(ind_df) * 100, 1)
            }
    
    if "region" in merged.columns:
        for reg in merged["region"].unique():
            reg_df = merged[merged["region"] == reg]
            segments_by_region[reg] = {
                "total": len(reg_df),
                "churn_pct": round((reg_df["risk_label"] == "Likely to Churn").sum() / len(reg_df) * 100, 1)
            }
    
    return {
        "total_customers": int(total),
        "will_renew": int(will_renew),
        "at_risk": int(at_risk),
        "likely_to_churn": int(likely_churn),
        "will_renew_pct": round(will_renew / total * 100, 2) if total > 0 else 0,
        "at_risk_pct": round(at_risk / total * 100, 2) if total > 0 else 0,
        "churn_pct": round(likely_churn / total * 100, 2) if total > 0 else 0,
        "avg_renewal_probability": round(predictions["renewal_probability"].mean(), 4),
        "health_score": round(will_renew / total * 100, 2) if total > 0 else 0,
        "segments_by_size": segments_by_size,
        "segments_by_industry": segments_by_industry,
        "segments_by_region": segments_by_region,
    }


def get_high_risk_customers(predictions: pd.DataFrame, df_customers: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Get highest-risk customers for intervention"""
    
    enriched = predictions.merge(df_customers, on="account_id")
    high_risk = enriched.nsmallest(top_n, "renewal_probability")
    
    return high_risk[[
        "account_id",
        "renewal_probability_pct",
        "risk_label",
        "avg_weekly_logins",
        "feature_adoption_pct",
        "support_tickets",
        "avg_ticket_sentiment",
        "interventions"
    ]]


def save_portfolio_predictions(predictions: pd.DataFrame, df_customers: pd.DataFrame, output_path=None):
    """Save enterprise-ready portfolio predictions"""
    
    output_path = output_path or OUTPUT_PATH
    
    # Merge predictions with original features
    result = predictions.merge(df_customers, on="account_id")
    result["prediction_timestamp"] = datetime.utcnow().isoformat()
    
    # Column order
    column_order = [
        "account_id",
        "renewal_probability",
        "renewal_probability_pct",
        "risk_label",
        "churn_reasons",
        "interventions",
        "prediction_timestamp",
    ] + FEATURE_COLUMNS + ["renewed"]
    
    result = result[column_order]
    result.to_csv(output_path, index=False)
    
    return result


def print_portfolio_summary(metrics: dict):
    """Print professional portfolio health summary"""
    print("\n" + "="*70)
    print("  RENEWAL PORTFOLIO HEALTH SUMMARY")
    print("="*70)
    print(f"\nTotal Customers:           {metrics['total_customers']:,}")
    print(f"Will Renew:                {metrics['will_renew']:,} ({metrics['will_renew_pct']:.1f}%)")
    print(f"At Risk:                   {metrics['at_risk']:,} ({metrics['at_risk_pct']:.1f}%)")
    print(f"Likely to Churn:           {metrics['likely_to_churn']:,} ({metrics['churn_pct']:.1f}%)")
    print(f"\nPortfolio Health Score:    {metrics['health_score']:.1f}%")
    print(f"Avg Renewal Probability:   {metrics['avg_renewal_probability']:.1%}")
    print("="*70 + "\n")


def main():
    """Main batch prediction pipeline"""
    print("\n>>> Loading model and data...")
    artifact = load_model_and_scaler()
    df_customers = load_customer_data()
    print(f"✓ Loaded {len(df_customers)} customer records")
    
    print("\n>>> Running batch predictions...")
    predictions = batch_predict_portfolio(artifact, df_customers)
    print(f"✓ Predicted renewal probability for all customers")
    
    print("\n>>> Computing portfolio metrics...")
    metrics = compute_portfolio_metrics(predictions, df_customers)
    print_portfolio_summary(metrics)
    
    # Print segment analysis
    print("\n📊 SEGMENT BREAKDOWN (Churn Risk by Category):")
    print("\nBy Company Size:")
    for size, data in metrics.get("segments_by_size", {}).items():
        print(f"  {size:15s}: {data['total']:4d} customers, {data['churn_pct']:5.1f}% churn")
    
    print("\nBy Industry:")
    for ind, data in sorted(metrics.get("segments_by_industry", {}).items(), key=lambda x: -x[1]['churn_pct'])[:5]:
        print(f"  {ind:20s}: {data['total']:4d} customers, {data['churn_pct']:5.1f}% churn")
    
    print("\nBy Region:")
    for reg, data in metrics.get("segments_by_region", {}).items():
        print(f"  {reg:20s}: {data['total']:4d} customers, {data['churn_pct']:5.1f}% churn")
    
    print("\n>>> Identifying high-risk customers (top 10)...")
    high_risk = get_high_risk_customers(predictions, df_customers, top_n=10)
    print(high_risk.to_string(index=False))
    
    print("\n>>> Saving enterprise-ready portfolio output...")
    saved_df = save_portfolio_predictions(predictions, df_customers)
    print(f"✓ Saved {len(saved_df)} predictions to {OUTPUT_PATH}")
    
    print("\n✓ BATCH PREDICTION PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
