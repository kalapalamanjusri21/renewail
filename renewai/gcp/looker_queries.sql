-- RenewAI: BigQuery & Looker Studio SQL Examples
-- These queries prepare renewal portfolio data for Google Cloud dashboards
-- Upload renewal_portfolio_predictions.csv to BigQuery as table: `renewai.customer_renewals`

-- ============================================================================
-- QUERY 1: Renewal Risk Distribution (Portfolio Overview)
-- ============================================================================
-- Use this to create a pie chart or bar chart in Looker Studio
SELECT
  risk_label,
  COUNT(*) as customer_count,
  ROUND(COUNT(*) / SUM(COUNT(*)) OVER () * 100, 2) as percentage,
  ROUND(AVG(renewal_probability), 4) as avg_renewal_probability
FROM
  `renewai.customer_renewals`
GROUP BY
  risk_label
ORDER BY
  CASE 
    WHEN risk_label = 'Will Renew' THEN 1
    WHEN risk_label = 'At Risk' THEN 2
    ELSE 3
  END;


-- ============================================================================
-- QUERY 2: High-Risk Customers Nearing Renewal Window
-- ============================================================================
-- Identify customers with low renewal probability + short renewal window
-- These need immediate customer success intervention
SELECT
  account_id,
  renewal_probability_pct,
  risk_label,
  avg_weekly_logins,
  feature_adoption_pct,
  support_tickets,
  avg_ticket_sentiment,
  contract_age_days,
  renewal_window_days
FROM
  `renewai.customer_renewals`
WHERE
  risk_label IN ('At Risk', 'Likely to Churn')
  AND renewal_window_days <= 30
ORDER BY
  renewal_probability ASC
LIMIT 100;


-- ============================================================================
-- QUERY 3: Feature Analysis by Risk Segment
-- ============================================================================
-- Understand what differentiates each risk category
-- Use for segment-specific strategies
SELECT
  risk_label,
  ROUND(AVG(avg_weekly_logins), 2) as avg_logins,
  ROUND(AVG(feature_adoption_pct), 4) as avg_adoption,
  ROUND(AVG(support_tickets), 2) as avg_tickets,
  ROUND(AVG(avg_ticket_sentiment), 4) as avg_sentiment,
  ROUND(AVG(contract_age_days), 1) as avg_contract_age,
  COUNT(*) as segment_size
FROM
  `renewai.customer_renewals`
GROUP BY
  risk_label
ORDER BY
  CASE 
    WHEN risk_label = 'Will Renew' THEN 1
    WHEN risk_label = 'At Risk' THEN 2
    ELSE 3
  END;


-- ============================================================================
-- QUERY 4: Feature Adoption Impact on Renewal
-- ============================================================================
-- Show correlation between feature adoption and renewal probability
-- Create a scatter plot or bubble chart
SELECT
  CASE
    WHEN feature_adoption_pct < 0.2 THEN 'Low (0-20%)'
    WHEN feature_adoption_pct < 0.5 THEN 'Medium (20-50%)'
    WHEN feature_adoption_pct < 0.8 THEN 'High (50-80%)'
    ELSE 'Very High (80%+)'
  END as adoption_tier,
  COUNT(*) as customer_count,
  ROUND(AVG(renewal_probability_pct), 2) as avg_renewal_probability_pct,
  ROUND(STDDEV(renewal_probability_pct), 2) as stddev_renewal
FROM
  `renewai.customer_renewals`
GROUP BY
  adoption_tier
ORDER BY
  CASE
    WHEN adoption_tier = 'Low (0-20%)' THEN 1
    WHEN adoption_tier = 'Medium (20-50%)' THEN 2
    WHEN adoption_tier = 'High (50-80%)' THEN 3
    ELSE 4
  END;


-- ============================================================================
-- QUERY 5: Customer Support Sentiment vs Renewal
-- ============================================================================
-- Analyze relationship between support experience and renewal likelihood
SELECT
  CASE
    WHEN avg_ticket_sentiment < -0.5 THEN 'Very Negative'
    WHEN avg_ticket_sentiment < 0 THEN 'Negative'
    WHEN avg_ticket_sentiment < 0.5 THEN 'Positive'
    ELSE 'Very Positive'
  END as sentiment_category,
  COUNT(*) as customer_count,
  ROUND(AVG(renewal_probability_pct), 2) as avg_renewal_probability_pct,
  ROUND(SUM(CASE WHEN risk_label = 'Will Renew' THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) as pct_will_renew
FROM
  `renewai.customer_renewals`
GROUP BY
  sentiment_category
ORDER BY
  CASE
    WHEN sentiment_category = 'Very Negative' THEN 1
    WHEN sentiment_category = 'Negative' THEN 2
    WHEN sentiment_category = 'Positive' THEN 3
    ELSE 4
  END;


-- ============================================================================
-- QUERY 6: Customer Engagement Patterns
-- ============================================================================
-- Weekly login frequency as engagement proxy
SELECT
  CASE
    WHEN avg_weekly_logins <= 5 THEN 'Low Engagement (≤5)'
    WHEN avg_weekly_logins <= 15 THEN 'Medium Engagement (6-15)'
    WHEN avg_weekly_logins <= 30 THEN 'High Engagement (16-30)'
    ELSE 'Very High Engagement (>30)'
  END as engagement_level,
  COUNT(*) as customer_count,
  ROUND(AVG(renewal_probability_pct), 2) as avg_renewal_probability_pct,
  ROUND(AVG(support_tickets), 2) as avg_support_tickets
FROM
  `renewai.customer_renewals`
GROUP BY
  engagement_level
ORDER BY
  CASE
    WHEN engagement_level = 'Low Engagement (≤5)' THEN 1
    WHEN engagement_level = 'Medium Engagement (6-15)' THEN 2
    WHEN engagement_level = 'High Engagement (16-30)' THEN 3
    ELSE 4
  END;


-- ============================================================================
-- QUERY 7: Portfolio Health Trend (if table has historical data)
-- ============================================================================
-- Track portfolio health over time - useful once predictions run regularly
-- Requires adding prediction_timestamp to grouping
SELECT
  DATE(PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', prediction_timestamp)) as prediction_date,
  COUNT(*) as total_customers,
  SUM(CASE WHEN risk_label = 'Will Renew' THEN 1 ELSE 0 END) as will_renew,
  SUM(CASE WHEN risk_label = 'At Risk' THEN 1 ELSE 0 END) as at_risk,
  SUM(CASE WHEN risk_label = 'Likely to Churn' THEN 1 ELSE 0 END) as likely_to_churn,
  ROUND(AVG(renewal_probability_pct), 2) as avg_renewal_probability_pct
FROM
  `renewai.customer_renewals`
GROUP BY
  prediction_date
ORDER BY
  prediction_date DESC;


-- ============================================================================
-- BigQuery Setup Instructions
-- ============================================================================
-- 1. Create dataset in BigQuery:
--    bq mk --dataset --location=US renewai
--
-- 2. Load CSV to BigQuery:
--    bq load --autodetect --source_format=CSV \
--      renewai.customer_renewals \
--      outputs/renewal_portfolio_predictions.csv
--
-- 3. Create Looker Studio dashboard:
--    - Connect to BigQuery project
--    - Select renewai.customer_renewals table
--    - Create scorecard for total_customers, will_renew, at_risk, likely_to_churn
--    - Add pie chart from Query 1
--    - Add table from Query 2 for high-risk customers
--    - Add bar chart from Query 3 for feature analysis
--
-- ============================================================================
