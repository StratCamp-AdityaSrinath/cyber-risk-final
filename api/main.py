from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import StringIO
import sys

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- DATA: This is the new, complete data string ---
CYBER_DATA_STRING = """
NAICS,Event_Code,Service_Code,Event_Freq,Uptake_Prob,Cost
52,EVT001,SVC001,0.088333,0.850000,77500
52,EVT001,SVC002,0.088333,0.700000,122500
52,EVT002,SVC003,0.055000,0.925000,255000
52,EVT002,SVC004,0.055000,0.525000,155000
54,EVT001,SVC001,0.120000,0.850000,70000
54,EVT001,SVC002,0.120000,0.650000,110000
54,EVT003,SVC005,0.080000,0.700000,300000
51,EVT002,SVC003,0.070000,0.900000,270000
51,EVT005,SVC007,0.030000,0.880000,550000
62,EVT002,SVC003,0.060000,0.950000,260000
62,EVT002,SVC004,0.060000,0.550000,160000
62,EVT005,SVC007,0.020000,0.900000,500000
22,EVT003,SVC005,0.090000,0.750000,320000
22,EVT006,SVC008,0.030000,0.850000,450000
23,EVT001,SVC001,0.070000,0.700000,60000
23,EVT006,SVC008,0.040000,0.800000,400000
61,EVT001,SVC001,0.080000,0.750000,65000
61,EVT004,SVC006,0.100000,0.800000,180000
72,EVT001,SVC001,0.068889,0.793889,240000
72,EVT001,SVC002,0.068889,0.793889,240000
72,EVT002,SVC003,0.068889,0.793889,240000
72,EVT002,SVC004,0.068889,0.793889,240000
72,EVT003,SVC005,0.068889,0.793889,240000
72,EVT004,SVC006,0.068889,0.793889,240000
72,EVT005,SVC007,0.068889,0.793889,240000
72,EVT006,SVC008,0.068889,0.793889,240000
"""

EMPLOYEE_SIZE_SCALING_FACTORS = {
    "<5": 0.1, "5-9": 0.2, "10-14": 0.35, "15-19": 0.5, "20-24": 0.65,
    "25-29": 0.8, "30-34": 1.0, "35-39": 1.2, "40-49": 1.5, "50-74": 1.9,
    "75-99": 2.4, "100-149": 3.0, "150-199": 3.7, "200-299": 4.5,
    "300-399": 5.4, "400-499": 6.4, "500-749": 7.9, "750-999": 9.9,
    "1,000-1,499": 12.0, "1,500-1,999": 14.0, "2,000-2,499": 16.0,
    "2,500-4,999": 18.0, "5,000+": 21.0
}

CATASTROPHIC_FREQUENCY = 0.15
CATASTROPHIC_SEVERITY_SHAPE = 1.6
CATASTROPHIC_SEVERITY_SCALE = 0.75

def sample_catastrophic_load():
    if np.random.binomial(1, CATASTROPHIC_FREQUENCY):
        return (np.random.pareto(a=CATASTROPHIC_SEVERITY_SHAPE) + 1) * CATASTROPHIC_SEVERITY_SCALE
    return 0.05

def compute_lognormal_params(mean_val, min_val, max_val):
    if pd.isna(mean_val) or mean_val <= 0: return 0, 0
    if min_val < 0: min_val = 0
    if min_val >= mean_val or max_val <= mean_val or min_val >= max_val:
        return np.log(mean_val) if mean_val > 0 else 0, 0.3
    try:
        cv_approx = (max_val - min_val) / (6 * mean_val)
        cv_approx = max(cv_approx, 0.2)
        sigma = np.sqrt(np.log(1 + cv_approx**2))
        mu = np.log(mean_val) - sigma**2 / 2
        if sigma <= 0 or not np.isfinite(mu) or not np.isfinite(sigma):
            return np.log(mean_val), 0.3
        return mu, sigma
    except (ValueError, ZeroDivisionError, TypeError):
        return np.log(mean_val) if mean_val > 0 else 0, 0.3

def compute_beta_params(mean_val, min_val, max_val):
    if (pd.isna(mean_val) or pd.isna(min_val) or pd.isna(max_val) or
        not (0 <= mean_val <= 1) or min_val >= max_val or (max_val - min_val) < 1e-9):
        return 2.0, 2.0
    mean_val = np.clip(mean_val, min_val, max_val)
    mean_normalized = (mean_val - min_val) / (max_val - min_val)
    if not (0 < mean_normalized < 1):
        return 2.0, 2.0
    variance_of_standard_beta = max((mean_normalized * (1 - mean_normalized))/6, 0.01)
    nu = (mean_normalized * (1 - mean_normalized) / variance_of_standard_beta) - 1
    if nu <= 0: return 2.0, 2.0
    alpha = mean_normalized * nu
    beta = (1 - mean_normalized) * nu
    alpha = max(alpha, 0.5); beta = max(beta, 0.5)
    return alpha, beta

@app.route('/api/main', methods=['POST'])
def handle_simulation():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. Please send a JSON body."}), 400

    naics = data.get('naics')
    employee_size = data.get('employee_size')
    deductible = data.get('deductible')
    selected_services = data.get('selected_services')

    if not all([naics, employee_size, deductible is not None, selected_services]):
        return jsonify({"error": "Missing one or more required parameters"}), 400

    try:
        naics_int = int(naics)  # Convert naics to integer safely
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid NAICS code. Must be a valid integer."}), 400

    N_ITERATIONS = 1000
    cyber_data = pd.read_csv(StringIO(CYBER_DATA_STRING))

    filtered_data = cyber_data[
        (cyber_data['NAICS'] == naics_int) &
        (cyber_data['Service_Code'].isin(selected_services))
    ].copy()

    if filtered_data.empty:
        return jsonify({"error": "No data available for the selected industry and services."}), 400

    for metric in ['Event_Freq', 'Cost']:
        filtered_data[f'{metric}_Min'] = filtered_data[metric] * (0.7 if metric == 'Cost' else 0.6)
        filtered_data[f'{metric}_Max'] = filtered_data[metric] * (2.0 if metric == 'Cost' else 1.4)
    filtered_data['Uptake_Prob_Min'] = filtered_data['Uptake_Prob'].clip(0, 1) * 0.6
    filtered_data['Uptake_Prob_Max'] = filtered_data['Uptake_Prob'].clip(0, 1) * 1.4

    for metric in ['Event_Freq', 'Cost']:
        params = np.array([compute_lognormal_params(r[f'{metric}'], r[f'{metric}_Min'], r[f'{metric}_Max']) for _, r in filtered_data.iterrows()])
        if params.size > 0:
            filtered_data[f'{metric}_mu'] = params[:, 0]
            filtered_data[f'{metric}_sigma'] = params[:, 1]

    # Fix: Compute beta parameters for Uptake_Prob explicitly
    beta_params = np.array([compute_beta_params(r['Uptake_Prob'], r['Uptake_Prob_Min'], r['Uptake_Prob_Max']) for _, r in filtered_data.iterrows()])
    if beta_params.size > 0:
        filtered_data['Uptake_Prob_alpha'] = beta_params[:, 0]
        filtered_data['Uptake_Prob_beta'] = beta_params[:, 1]

    size_scaling_factor = EMPLOYEE_SIZE_SCALING_FACTORS.get(employee_size, 1.0)
    simulated_loss_per_firm = np.zeros(N_ITERATIONS)

    for _, service_params in filtered_data.iterrows():
        scaled_freq_mean = service_params['Event_Freq'] * size_scaling_factor
        n_events = np.random.poisson(scaled_freq_mean, size=N_ITERATIONS)
        beta_sample = np.random.beta(service_params['Uptake_Prob_alpha'], service_params['Uptake_Prob_beta'], size=N_ITERATIONS)
        sampled_uptake_prob = service_params['Uptake_Prob_Min'] + beta_sample * (service_params['Uptake_Prob_Max'] - service_params['Uptake_Prob_Min'])
        sampled_cost = np.random.lognormal(service_params['Cost_mu'], service_params['Cost_sigma'], size=N_ITERATIONS)
        service_loss = n_events * sampled_uptake_prob * sampled_cost
        simulated_loss_per_firm += service_loss

    catastrophic_loads = np.array([sample_catastrophic_load() for _ in range(N_ITERATIONS)])
    loaded_simulated_loss = simulated_loss_per_firm * (1 + catastrophic_loads)
    simulated_premium = np.maximum(0, loaded_simulated_loss - int(deductible))

    mean_premium = simulated_premium.mean()
    std_dev_premium = simulated_premium.std(ddof=1)
    max_premium = simulated_premium.max()
    cv = (std_dev_premium / mean_premium) if mean_premium > 0 else 0
    max_mean_ratio = (max_premium / mean_premium) if mean_premium > 0 else 0

    results = {
        "mean_premium": f"${mean_premium:,.2f}",
        "volatility_cv": f"{cv:.1%}",
        "max_to_mean_ratio": f"{max_mean_ratio:.2f}x"
    }
    return jsonify(results)