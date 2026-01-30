# RenewAI

**Predict SaaS subscription renewal probability** using synthetic data and Google Cloud. Built for hackathon demos: explainable ML, BigQuery integration, and a FastAPI backend.

---

## Problem Statement

SaaS businesses lose revenue when customers churn at renewal. RenewAI predicts **renewal probability** per customer so teams can prioritize outreach, improve feature adoption, and fix support issues before contracts end. The system uses **explainable ML** so stakeholders see *why* a customer is at risk (e.g., low feature adoption, negative support sentiment).

---

## Architecture Overview

```
renewai/
  data/           Synthetic dataset (500 rows) -> saas_renewal_data.csv
  model/          Logistic Regression -> model.pkl, feature coefficients
  backend/        FastAPI /predict with risk label + top 3 factors; serves frontend
  frontend/       Web UI: customer inputs, renewal probability, risk label, top 3 factors
  gcp/            BigQuery loader (saas_analytics.renewal_data) + sample SQL
```

- **Data**: `data/generate_data.py` creates 500 customer records with realistic correlations (low adoption + negative sentiment -> low renewal).
- **ML**: `model/train_model.py` trains Logistic Regression, prints accuracy and coefficients, saves `model.pkl`.
- **API**: `backend/main.py` exposes `POST /predict` with renewal probability, risk label (Low/Medium/High), and top 3 contributing factors; serves the frontend at `/`.
- **Frontend**: `frontend/` is a single-page app: enter customer health metrics (or use presets), see renewal probability, risk badge, and top 3 factors—aligned with the SaaS renewal problem.
- **GCP**: `gcp/bigquery_loader.py` loads the CSV into BigQuery; `gcp/looker_sample_queries.sql` provides dashboard-ready queries.

---

## Google Technologies Used

| Technology | Use |
|------------|-----|
| **BigQuery** (mandatory) | Store and query renewal data; dataset `saas_analytics`, table `renewal_data`. Schema is dashboard-friendly for Looker Studio. |
| **Vertex AI** (optional) | Can be added later for training or hosting the model in GCP. |

Authentication: set `GOOGLE_APPLICATION_CREDENTIALS` to your service account JSON path and `GCP_PROJECT_ID` to your project ID.

---

## How to Run Locally

### 1. Create a virtual environment (recommended)

```bash
cd renewai
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python data/generate_data.py
```

Output: `data/saas_renewal_data.csv` (500 rows).

### 4. Train the model

```bash
python model/train_model.py
```

Output: accuracy and feature coefficients in the console; `model/model.pkl` on disk.

### 5. Start the API

```bash
python backend/main.py
# or: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- Docs: http://localhost:8000/docs  
- Example prediction:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"avg_weekly_logins\": 25, \"feature_adoption_pct\": 0.6, \"support_tickets\": 2, \"avg_ticket_sentiment\": 0.3, \"contract_age_days\": 180, \"renewal_window_days\": 30}"
```

### 6. (Optional) Load data into BigQuery

```bash
set GOOGLE_APPLICATION_CREDENTIALS=path\to\service-account.json
set GCP_PROJECT_ID=your-project-id
python gcp/bigquery_loader.py --project your-project-id
```

Use `gcp/looker_sample_queries.sql` in Looker Studio or BigQuery for renewal risk, feature adoption vs renewal, and high-risk accounts near renewal.

---

## How This Improves Existing SaaS Churn Systems

- **Proactive risk score**: Predict renewal probability per account instead of only post-churn analysis.
- **Explainability**: Top 3 factors (e.g., feature adoption, sentiment, tickets) make the model actionable for CS and product.
- **Single pipeline**: Synthetic data -> train -> API -> BigQuery, so you can plug in real data and add Vertex AI later without changing the interface.
- **Dashboard-ready**: BigQuery schema and sample SQL are designed for Looker Studio (renewal by segment, adoption vs renewal, at-risk list).

---

## Hackathon Demo Instructions

1. **Setup**: `pip install -r requirements.txt` then `python data/generate_data.py` then `python model/train_model.py`
2. **Start server**: `python backend/main.py`
3. **Open frontend**: Go to http://localhost:8000 — use **At-risk customer** preset and click **Predict** to show High risk and low renewal %; use **Healthy customer** to show Low risk and high renewal %.
4. **Explainability**: Point to the "Top 3 factors" section (which features increase or decrease renewal likelihood).
5. **API**: Open http://localhost:8000/docs to show `POST /predict` and the response shape.
6. **Optional**: Load CSV to BigQuery and open Looker Studio with the sample SQL for renewal and risk views.

---

## License

MIT.
