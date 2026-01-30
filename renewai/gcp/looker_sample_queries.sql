-- RenewAI - Sample SQL for Google Looker Studio (BigQuery)
-- Dataset: saas_analytics, Table: renewal_data
-- Use these in Looker Studio custom queries or BigQuery saved views.

-- 1) Renewal risk by account (for segmentation)
SELECT
  account_id,
  avg_weekly_logins,
  feature_adoption_pct,
  support_tickets,
  avg_ticket_sentiment,
  contract_age_days,
  renewal_window_days,
  renewed,
  CASE
    WHEN renewed = 1 THEN 'Renewed'
    ELSE 'Churned'
  END AS renewal_status
FROM `saas_analytics.renewal_data`
ORDER BY renewal_window_days ASC, feature_adoption_pct DESC;

-- 2) Feature adoption vs renewal (for dashboards)
SELECT
  ROUND(feature_adoption_pct, 2) AS adoption_bucket,
  COUNT(*) AS accounts,
  SUM(renewed) AS renewed_count,
  SAFE_DIVIDE(SUM(renewed), COUNT(*)) AS renewal_rate
FROM `saas_analytics.renewal_data`
GROUP BY adoption_bucket
ORDER BY adoption_bucket;

-- 3) High-risk customers near renewal window (action list)
SELECT
  account_id,
  avg_weekly_logins,
  feature_adoption_pct,
  support_tickets,
  avg_ticket_sentiment,
  renewal_window_days,
  renewed
FROM `saas_analytics.renewal_data`
WHERE renewed = 0
  AND renewal_window_days <= 30
  AND (feature_adoption_pct < 0.5 OR avg_ticket_sentiment < 0)
ORDER BY renewal_window_days ASC
LIMIT 100;
