"""
RenewAI - Enhanced Synthetic SaaS Dataset Generator
Now includes:
- Temporal features (login trend, days since login, adoption acceleration)
- Contract lifecycle stages (Onboarding/Growth/Mature/Legacy)
- Business context (company size, industry, geo, payment status)
- Churn reasons (categorized signals)
Generates realistic SaaS churn scenarios across 4,000 customers
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Reproducible randomness
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_RECORDS = 4000

# New parameters
COMPANY_SIZES = ["SMB", "Mid-Market", "Enterprise"]
INDUSTRIES = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]
REGIONS = ["North America", "Europe", "APAC", "LATAM"]
PAYMENT_STATUSES = ["On-Time", "Late (5-14d)", "Late (15-29d)", "Failed"]


def generate_enhanced_renewal_data(n: int = N_RECORDS) -> pd.DataFrame:
    """
    Generate enhanced synthetic SaaS customer data with:
    - Temporal features (trends)
    - Contract lifecycle
    - Business context
    - Churn reason signals
    """
    account_ids = [f"ACC-{i:05d}" for i in range(1, n + 1)]
    
    # Base temporal features
    contract_age_days = np.random.randint(30, 1095, size=n)
    renewal_window_days = np.random.randint(7, 90, size=n)
    
    # Current week metrics
    avg_weekly_logins = np.random.randint(1, 80, size=n)
    feature_adoption_pct = np.random.uniform(0.02, 0.98, size=n)
    support_tickets = np.random.poisson(3, size=n).clip(0, 20)
    avg_ticket_sentiment = np.random.uniform(-1.0, 1.0, size=n)
    
    # ========== TIER 1: TEMPORAL FEATURES ==========
    # Trend in logins (increasing/decreasing)
    logins_trend = np.random.normal(0, 5, size=n)  # -5 to +5 per week
    days_since_last_login = np.random.randint(0, 60, size=n)
    
    # Adoption acceleration (is adoption speeding up or slowing down?)
    adoption_trend = np.random.normal(0, 0.05, size=n).clip(-0.1, 0.1)
    
    # ========== TIER 2: CONTRACT LIFECYCLE STAGES ==========
    def get_contract_stage(age):
        if age < 30: return "Onboarding"
        elif age < 90: return "Growth"
        elif age < 365: return "Mature"
        else: return "Legacy"
    
    contract_stage = [get_contract_stage(age) for age in contract_age_days]
    
    # ========== TIER 3: BUSINESS CONTEXT ==========
    company_size = np.random.choice(COMPANY_SIZES, size=n)
    industry = np.random.choice(INDUSTRIES, size=n)
    region = np.random.choice(REGIONS, size=n)
    payment_status = np.random.choice(PAYMENT_STATUSES, size=n)
    
    # ========== CHURN REASON CLASSIFICATION ==========
    def classify_churn_reason(row_idx):
        """Determine primary churn reason based on feature patterns"""
        reasons = []
        
        # Low engagement
        if avg_weekly_logins[row_idx] < 5:
            reasons.append("Low Engagement")
        
        # Days since login
        if days_since_last_login[row_idx] > 30:
            reasons.append("Inactive User")
        
        # Poor support experience
        if avg_ticket_sentiment[row_idx] < -0.5:
            reasons.append("Support Issues")
        
        # Feature adoption gap
        if feature_adoption_pct[row_idx] < 0.2:
            reasons.append("Low Feature Adoption")
        
        # Declining adoption
        if adoption_trend[row_idx] < -0.05:
            reasons.append("Adoption Declining")
        
        # Payment issues
        if payment_status[row_idx] != "On-Time":
            reasons.append("Payment Problems")
        
        # Many support tickets (friction)
        if support_tickets[row_idx] > 10:
            reasons.append("High Support Load")
        
        return reasons[0] if reasons else "Other"
    
    churn_reason = [classify_churn_reason(i) for i in range(n)]
    
    # ========== RENEWAL PROBABILITY (Enhanced) ==========
    # Stronger correlation with new features
    logit_renewal = (
        # Original features (weighted)
        3.5 * (feature_adoption_pct - 0.5)
        + 1.8 * avg_ticket_sentiment
        - 0.25 * support_tickets
        + 0.03 * (avg_weekly_logins - 30)
        - 0.005 * renewal_window_days
        + 0.002 * contract_age_days
        # NEW: Temporal features (strong signals)
        + 0.1 * logins_trend          # Increasing logins = renewal
        - 0.05 * days_since_last_login / 10  # No login = churn
        + 2.0 * adoption_trend        # Adoption acceleration = renewal
        # NEW: Lifecycle stage adjustment
        + (0.5 if get_contract_stage(contract_age_days[n-1]) == "Onboarding" else 
           -0.3 if get_contract_stage(contract_age_days[n-1]) == "Legacy" else 0)
        # NEW: Business context
        + (0.3 if company_size[n-1] == "Enterprise" else
           -0.1 if company_size[n-1] == "SMB" else 0)
        - (0.4 if payment_status[n-1] != "On-Time" else 0)
    )
    
    prob_renewal = 1 / (1 + np.exp(-np.clip(logit_renewal, -10, 10)))
    renewed = (prob_renewal > 0.5).astype(int)
    
    # Flip 18% for realistic noise
    n_flip = int(n * 0.18)
    flip_idx = np.random.choice(n, size=n_flip, replace=False)
    renewed[flip_idx] = 1 - renewed[flip_idx]
    
    # ========== BUILD DATAFRAME ==========
    df = pd.DataFrame({
        # Original features
        "account_id": account_ids,
        "avg_weekly_logins": avg_weekly_logins,
        "feature_adoption_pct": np.round(feature_adoption_pct, 4),
        "support_tickets": support_tickets,
        "avg_ticket_sentiment": np.round(avg_ticket_sentiment, 4),
        "contract_age_days": contract_age_days,
        "renewal_window_days": renewal_window_days,
        # NEW: Temporal features
        "logins_trend": np.round(logins_trend, 2),
        "days_since_last_login": days_since_last_login,
        "adoption_trend": np.round(adoption_trend, 4),
        # NEW: Contract lifecycle
        "contract_stage": contract_stage,
        # NEW: Business context
        "company_size": company_size,
        "industry": industry,
        "region": region,
        "payment_status": payment_status,
        # NEW: Churn reason
        "primary_churn_reason": churn_reason,
        # Target
        "renewed": renewed,
    })
    
    return df


def main():
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / "saas_renewal_data.csv"
    
    df = generate_enhanced_renewal_data(N_RECORDS)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"Generated {len(df)} enhanced customer records")
    print(f"Output: {output_path}")
    print(f"{'='*70}")
    print(f"\nDataset Statistics:")
    print(f"  Renewal rate: {df['renewed'].mean():.2%}")
    print(f"\n  Contract Stages:")
    print(df['contract_stage'].value_counts().to_string())
    print(f"\n  Company Sizes:")
    print(df['company_size'].value_counts().to_string())
    print(f"\n  Top Churn Reasons:")
    print(df['primary_churn_reason'].value_counts().head(10).to_string())
    print(f"\n  Columns: {len(df.columns)} (was 8, now {len(df.columns)})")
    print(f"{'='*70}\n")
    
    return df


if __name__ == "__main__":
    main()

