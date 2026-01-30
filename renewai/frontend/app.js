/**
 * RenewAI frontend — calls /predict and displays renewal probability + top factors.
 */

(function () {
  const API_BASE = '';

  const presets = {
    'healthy': {
      avg_weekly_logins: 45,
      feature_adoption_pct: 85,
      support_tickets: 1,
      avg_ticket_sentiment: 0.7,
      contract_age_days: 365,
      renewal_window_days: 60,
    },
    'at-risk': {
      avg_weekly_logins: 8,
      feature_adoption_pct: 25,
      support_tickets: 12,
      avg_ticket_sentiment: -0.6,
      contract_age_days: 90,
      renewal_window_days: 14,
    },
    'churn': {
      avg_weekly_logins: 2,
      feature_adoption_pct: 10,
      support_tickets: 8,
      avg_ticket_sentiment: -0.8,
      contract_age_days: 300,
      renewal_window_days: 7,
    },
  };

  const featureLabels = {
    avg_weekly_logins: 'Avg weekly logins',
    feature_adoption_pct: 'Feature adoption %',
    support_tickets: 'Support tickets',
    avg_ticket_sentiment: 'Avg ticket sentiment',
    contract_age_days: 'Contract age (days)',
    renewal_window_days: 'Days until renewal',
  };

  function formatFactorValue(feature, value) {
    if (feature === 'feature_adoption_pct') return (value * 100).toFixed(1) + '%';
    if (feature === 'avg_ticket_sentiment') return value.toFixed(2) + ' (−1 to 1)';
    return Number(value) === value && value % 1 !== 0 ? value.toFixed(2) : value;
  }

  function getFormData() {
    const pct = document.getElementById('feature_adoption_pct');
    const rawPct = parseFloat(pct.value);
    const featureAdoptionPct = pct.max === '100' ? rawPct / 100 : rawPct;
    return {
      avg_weekly_logins: parseInt(document.getElementById('avg_weekly_logins').value, 10),
      feature_adoption_pct: featureAdoptionPct,
      support_tickets: parseInt(document.getElementById('support_tickets').value, 10),
      avg_ticket_sentiment: parseFloat(document.getElementById('avg_ticket_sentiment').value),
      contract_age_days: parseInt(document.getElementById('contract_age_days').value, 10),
      renewal_window_days: parseInt(document.getElementById('renewal_window_days').value, 10),
    };
  }

  function setFormData(data) {
    const pctInput = document.getElementById('feature_adoption_pct');
    const pctVal = data.feature_adoption_pct <= 1 ? data.feature_adoption_pct * 100 : data.feature_adoption_pct;
    pctInput.value = pctVal;
    document.getElementById('avg_weekly_logins').value = data.avg_weekly_logins;
    document.getElementById('support_tickets').value = data.support_tickets;
    document.getElementById('avg_ticket_sentiment').value = data.avg_ticket_sentiment;
    document.getElementById('contract_age_days').value = data.contract_age_days;
    document.getElementById('renewal_window_days').value = data.renewal_window_days;
  }

  function setLoading(loading) {
    const form = document.getElementById('predict-form');
    const btn = form.querySelector('button[type="submit"]');
    btn.disabled = loading;
    btn.innerHTML = loading 
      ? '<span class="btn-loading">Predicting…</span>'
      : '<span class="btn-text">Predict Renewal Probability</span>';
  }

  function showPlaceholder() {
    document.getElementById('result-section').style.display = 'none';
  }

  function showError(message) {
    const resultSection = document.getElementById('result-section');
    resultSection.style.display = 'block';
    const card = resultSection.querySelector('.result-card');
    card.innerHTML = '<h2>Prediction Error</h2><div class="error-message">' + escapeHtml(message) + '</div>';
  }

  function showResult(data) {
    const resultSection = document.getElementById('result-section');
    resultSection.style.display = 'block';

    // Update probability circle (data.renewal_probability is already a percentage 0-100)
    const probValue = typeof data.renewal_probability === 'string' 
      ? parseFloat(data.renewal_probability).toFixed(1) 
      : data.renewal_probability.toFixed(1);
    document.getElementById('probability-value').textContent = probValue;

    // Update label
    const label = data.risk_label === 'Will Renew' ? '✅ Will Renew' :
                  data.risk_label === 'At Risk' ? '⚠️ At Risk' :
                  '🚨 Likely to Churn';
    document.getElementById('probability-label').textContent = label;

    // Update factors
    const list = document.getElementById('factors-list');
    list.innerHTML = '';
    if (data.top_factors && data.top_factors.length > 0) {
      data.top_factors.forEach(function (f) {
        const li = document.createElement('li');
        const name = featureLabels[f.feature] || f.feature;
        const valueStr = formatFactorValue(f.feature, f.value);
        const impact = f.direction === 'increases' ? '📈 increases' : '📉 decreases';
        li.innerHTML =
          '<span class="factor-name">' + escapeHtml(name) + '</span>' +
          '<span class="factor-desc">Value: ' + escapeHtml(String(valueStr)) +
          ' — ' + impact + ' renewal likelihood</span>';
        list.appendChild(li);
      });
    }
  }

  function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  async function submitPredict() {
    setLoading(true);

    const body = getFormData();
    if (body.feature_adoption_pct > 1) body.feature_adoption_pct = body.feature_adoption_pct / 100;

    // Validate logical constraints
    const totalDays = body.contract_age_days + body.renewal_window_days;
    if (totalDays > 1095) { // ~3 years is reasonable max
      showError('Contract age + renewal days should not exceed ~3 years (1095 days). A customer cannot have been with you 3+ years AND have 3+ years until renewal.');
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(API_BASE + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        showError(data.detail || res.statusText || 'Request failed');
        return;
      }
      showResult(data);
    } catch (e) {
      showError('Cannot reach the API. Is the server running at ' + (window.location.origin || 'this host') + '?');
    } finally {
      setLoading(false);
    }
  }

  // Portfolio functions
  async function loadPortfolioData() {
    try {
      const response = await fetch(API_BASE + '/portfolio-summary');
      if (!response.ok) throw new Error('Failed to load portfolio data');
      const data = await response.json();
      displayPortfolioMetrics(data);
    } catch (error) {
      const errEl = document.getElementById('portfolio-error');
      errEl.textContent = 'Error loading portfolio: ' + error.message;
      errEl.classList.remove('hidden');
      document.getElementById('portfolio-loading').classList.add('hidden');
    }
  }

  function getChurnClass(churnPct) {
    if (churnPct >= 50) return 'segment-churn-high';
    if (churnPct >= 35) return 'segment-churn-medium';
    return 'segment-churn-low';
  }

  function displaySegmentData(type, segments, selector) {
    const container = document.querySelector(selector);
    container.innerHTML = '';
    
    // Sort segments by churn percentage (highest first)
    const sorted = Object.entries(segments).sort((a, b) => b[1].churn_pct - a[1].churn_pct);
    
    sorted.forEach(function([name, data]) {
      const item = document.createElement('div');
      item.className = 'segment-item';
      const churnClass = getChurnClass(data.churn_pct);
      item.innerHTML =
        '<span class="segment-name">' + escapeHtml(name) + '</span>' +
        '<span class="segment-churn ' + churnClass + '">' + data.churn_pct.toFixed(1) + '% churn</span>';
      container.appendChild(item);
    });
  }

  function displayPortfolioMetrics(data) {
    // Update main metrics
    document.getElementById('portfolio-total').textContent = data.total_customers.toLocaleString();
    document.getElementById('portfolio-will-renew').textContent = data.will_renew.toLocaleString();
    document.getElementById('portfolio-at-risk').textContent = data.at_risk.toLocaleString();
    document.getElementById('portfolio-churn').textContent = data.likely_to_churn.toLocaleString();
    
    // Update percentages
    const total = data.total_customers || 1;
    const willRenewPct = ((data.will_renew / total) * 100).toFixed(1);
    const atRiskPct = ((data.at_risk / total) * 100).toFixed(1);
    const churnPct = ((data.likely_to_churn / total) * 100).toFixed(1);
    
    document.getElementById('portfolio-will-renew-pct').textContent = willRenewPct + '%';
    document.getElementById('portfolio-at-risk-pct').textContent = atRiskPct + '%';
    document.getElementById('portfolio-churn-pct').textContent = churnPct + '%';
    
    document.getElementById('portfolio-health-score').textContent = data.portfolio_health_score.toFixed(1) + '%';
    document.getElementById('churn-percentage').textContent = churnPct + '%';
    
    // Display segments
    if (data.segments_by_size) {
      displaySegmentData('size', data.segments_by_size, '#segments-size');
    }
    if (data.segments_by_industry) {
      displaySegmentData('industry', data.segments_by_industry, '#segments-industry');
    }
    if (data.segments_by_region) {
      displaySegmentData('region', data.segments_by_region, '#segments-region');
    }
    
    document.getElementById('portfolio-loading').classList.add('hidden');
    document.getElementById('portfolio-content').classList.remove('hidden');
  }

  async function loadHighRiskCustomers() {
    try {
      const response = await fetch(API_BASE + '/high-risk-customers?top_n=10');
      if (!response.ok) throw new Error('Failed to load high-risk customers');
      const customers = await response.json();
      displayHighRiskCustomers(customers);
    } catch (error) {
      const errEl = document.getElementById('risk-error');
      errEl.textContent = 'Error loading high-risk customers: ' + error.message;
      errEl.classList.remove('hidden');
      document.getElementById('risk-loading').classList.add('hidden');
    }
  }

  function displayHighRiskCustomers(customers) {
    const tbody = document.getElementById('risk-tbody');
    tbody.innerHTML = '';
    
    customers.forEach(function (cust) {
      const row = document.createElement('tr');
      const riskClass = cust.risk_label.toLowerCase().replace(/\s+/g, '-');
      const adoption = (cust.feature_adoption_pct * 100).toFixed(0);
      
      row.innerHTML =
        '<td class="account-id">' + escapeHtml(cust.account_id) + '</td>' +
        '<td>' + (cust.renewal_probability_pct || 0).toFixed(1) + '%</td>' +
        '<td><span class="risk-label ' + riskClass + '">' + escapeHtml(cust.risk_label) + '</span></td>' +
        '<td>' + escapeHtml(cust.company_size || '—') + '</td>' +
        '<td>' + escapeHtml(cust.industry || '—') + '</td>' +
        '<td>' + (cust.avg_weekly_logins || 0) + '</td>' +
        '<td>' + adoption + '%</td>' +
        '<td><button class="action-btn" onclick="alert(\'Intervention: Contact customer for engagement review\')">Contact</button></td>';
      tbody.appendChild(row);
    });
    
    document.getElementById('risk-loading').classList.add('hidden');
    document.getElementById('risk-content').classList.remove('hidden');
  }

  // Load portfolio data on page load
  document.addEventListener('DOMContentLoaded', function () {
    loadPortfolioData();
    loadHighRiskCustomers();

    // Attach predict form
    const form = document.getElementById('predict-form');
    if (form) {
      form.addEventListener('submit', function (e) {
        e.preventDefault();
        submitPredict();
      });
    }

    // Attach preset buttons
    document.querySelectorAll('.btn-preset').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.preventDefault();
        const preset = presets[this.getAttribute('data-preset')];
        if (preset) setFormData(preset);
      });
    });
  });
})();

