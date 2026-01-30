"""
RenewAI - Load SaaS renewal data and portfolio predictions into Google BigQuery.
Tables:
- renewal_raw_data: Original customer features and ground truth
- customer_renewals: Batch predictions with risk labels (portfolio-ready)

Authentication: Service account JSON (GOOGLE_APPLICATION_CREDENTIALS).
"""

import os
from pathlib import Path
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
DATASET_ID = "saas_analytics"


def get_raw_data_csv_path():
    """Path to original saas_renewal_data.csv."""
    return Path(__file__).resolve().parent.parent / "data" / "saas_renewal_data.csv"


def get_portfolio_predictions_csv_path():
    """Path to batch portfolio predictions CSV."""
    return Path(__file__).resolve().parent.parent / "outputs" / "renewal_portfolio_predictions.csv"


def get_raw_data_schema():
    """Schema for original renewal data table."""
    return [
        bigquery.SchemaField("account_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("avg_weekly_logins", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("feature_adoption_pct", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("support_tickets", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("avg_ticket_sentiment", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("contract_age_days", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("renewal_window_days", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("renewed", "INTEGER", mode="NULLABLE"),
    ]


def get_portfolio_predictions_schema():
    """Schema for portfolio predictions (BigQuery/Looker Studio ready)."""
    return [
        bigquery.SchemaField("account_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("renewal_probability", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("renewal_probability_pct", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("risk_label", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("prediction_timestamp", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("avg_weekly_logins", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("feature_adoption_pct", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("support_tickets", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("avg_ticket_sentiment", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("contract_age_days", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("renewal_window_days", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("renewed", "INTEGER", mode="NULLABLE"),
    ]





def load_raw_data_to_bigquery(csv_path=None, project_id=None, dataset_id=DATASET_ID, create_dataset=True):
    """Load original customer data into renewal_raw_data table."""
    csv_path = csv_path or get_raw_data_csv_path()
    project_id = project_id or PROJECT_ID
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Data not found: {csv_path}. Run: python data/generate_data.py")
    
    client = bigquery.Client(project=project_id)
    
    if create_dataset:
        dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
        dataset_ref.location = "US"
        try:
            client.create_dataset(dataset_ref, exists_ok=True)
        except Exception as e:
            print(f"Dataset create/skip: {e}")
    
    df = pd.read_csv(csv_path)
    table_ref = f"{project_id}.{dataset_id}.renewal_raw_data"
    job_config = bigquery.LoadJobConfig(
        schema=get_raw_data_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    table = client.get_table(table_ref)
    print(f"✓ Loaded {table.num_rows} rows into {table_ref}")
    return table.num_rows


def load_portfolio_predictions_to_bigquery(csv_path=None, project_id=None, dataset_id=DATASET_ID, create_dataset=True):
    """Load batch portfolio predictions into customer_renewals table (Looker Studio ready)."""
    csv_path = csv_path or get_portfolio_predictions_csv_path()
    project_id = project_id or PROJECT_ID
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Portfolio predictions not found: {csv_path}. Run: python analytics/batch_predict.py")
    
    client = bigquery.Client(project=project_id)
    
    if create_dataset:
        dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
        dataset_ref.location = "US"
        try:
            client.create_dataset(dataset_ref, exists_ok=True)
        except Exception as e:
            print(f"Dataset create/skip: {e}")
    
    df = pd.read_csv(csv_path)
    table_ref = f"{project_id}.{dataset_id}.customer_renewals"
    job_config = bigquery.LoadJobConfig(
        schema=get_portfolio_predictions_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    table = client.get_table(table_ref)
    print(f"✓ Loaded {table.num_rows} portfolio predictions into {table_ref}")
    return table.num_rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load RenewAI data into BigQuery")
    parser.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id"))
    parser.add_argument("--no-create-dataset", action="store_true")
    parser.add_argument("--portfolio-only", action="store_true", help="Load only portfolio predictions")
    parser.add_argument("--raw-data-only", action="store_true", help="Load only raw customer data")
    
    args = parser.parse_args()
    
    print("\n>>> RenewAI BigQuery Loader")
    print("="*60)
    
    try:
        if not args.portfolio_only:
            print("\n1. Loading raw customer data...")
            load_raw_data_to_bigquery(project_id=args.project, create_dataset=not args.no_create_dataset)
        
        if not args.raw_data_only:
            print("\n2. Loading portfolio predictions...")
            load_portfolio_predictions_to_bigquery(project_id=args.project, create_dataset=not args.no_create_dataset)
        
        print("\n" + "="*60)
        print("✓ Data loading complete!")
        print(f"  Project: {args.project}")
        print(f"  Dataset: {DATASET_ID}")
        print("  Tables: renewal_raw_data, customer_renewals")
        print("\nNext: Create Looker Studio dashboard using gcp/looker_queries.sql")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("  Ensure GOOGLE_APPLICATION_CREDENTIALS is set to service account JSON")


if __name__ == "__main__":
    main()

