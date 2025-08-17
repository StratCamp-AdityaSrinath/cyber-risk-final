from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import StringIO
import json
import os # Import os to create a reliable file path

app = Flask(__name__)
CORS(app)

# --- CORRECTED DATA HANDLING ---
# Build a reliable path to the data file within the same directory
# This ensures Vercel can find the file.
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'attributes.csv')

def load_and_clean_data():
    """Loads and cleans the source CSV data."""
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        df.columns = df.columns.str.strip()
        df = df[pd.to_numeric(df['Industry NAICS'], errors='coerce').notna()]
        df['Industry NAICS'] = df['Industry NAICS'].astype(int)
        return df
    except Exception as e:
        print(f"Error loading data file: {e}")
        return None

def create_ui_data_structure(df):
    """Creates the nested dictionary for the dynamic frontend."""
    if df is None:
        return {"error": "Source data could not be loaded."}
    
    # Dynamically create Event Codes from Event Names for internal use
    unique_events = df['Event Name'].unique()
    event_map = {name: f"EVT{i+1:03d}" for i, name in enumerate(unique_events)}
    df['Event_Code'] = df['Event Name'].map(event_map)

    frontend_naics_info = {
        52: "Finance and Insurance", 54: "Professional, Scientific, and Technical Services",
        51: "Information", 62: "Health Care and Social Assistance", 22: "Utilities",
        23: "Construction", 61: "Educational Services", 72: "Accommodation and Food Services"
    }

    ui_structure = {}
    for naics_code, naics_name in frontend_naics_info.items():
        naics_df = df[df['Industry NAICS'] == naics_code]
        # Always add the NAICS to the structure, even if it has no events
        ui_structure[str(naics_code)] = {"name": naics_name, "events": {}}
        if not naics_df.empty:
            for event_name, group in naics_df.groupby('Event Name'):
                event_code = group['Event_Code'].iloc[0]
                ui_structure[str(naics_code)]["events"][event_code] = {
                    "name": event_name,
                    "services": dict(zip(group['Code'], group['Remedial Service Type']))
                }
    return ui_structure

# --- GLOBAL DATA STRUCTURES ---
SOURCE_DF = load_and_clean_data()
UI_DATA_STRUCTURE = create_ui_data_structure(SOURCE_DF.copy() if SOURCE_DF is not None else None)

# --- HELPER FUNCTIONS (FULL SIMULATION LOGIC RESTORED) ---
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
        return mu, sigma
    except (ValueError, ZeroDivisionError, TypeError):
        return np.log(mean_val) if mean_val > 0 else 0, 0.3

def compute_beta_params(mean_val, min_val, max_val):
    if (pd.isna(mean_val) or not (0 <= mean_val <= 1) or min_val >= max_val):
        return 2.0, 2.0
    mean_val = np.clip(mean_val, min_val, max_val)
    if (max_val - min_val) < 1e-9: return 2.0, 2.0
    mean_normalized = (mean_val - min_val) / (max_val - min_val)
    if not (0 < mean_normalized < 1): return 2.0, 2.0
    variance_of_standard_beta = max((mean_normalized * (1 - mean_normalized))/6, 0.01)
    nu = (mean_normalized * (1 - mean_normalized) / variance_of_standard_beta) - 1
    if nu <= 0: return 2.0, 2.0
    alpha = mean_normalized * nu
    beta = (1 - mean_normalized) * nu
    return max(alpha, 0.5), max(beta, 0.5)

# --- API ENDPOINTS ---

@app.route('/api/data', methods=['GET'])
def get_ui_data():
    """Provides the structured data needed to build the dynamic UI."""
    if "error" in UI_DATA_STRUCTURE:
        return jsonify(UI_DATA_STRUCTURE), 500
    return jsonify(UI_DATA_STRUCTURE)

@app.route('/api/main', methods=['POST'])
def handle_simulation():
    if SOURCE_DF is None:
        return jsonify({"error": "Server-side data file could not be loaded."}), 500

    data = request.get_json()
    if not data: return jsonify({"error": "Invalid JSON request"}), 400
    
    naics = data.get('naics')
    employee_size = data.get('employee_size')
    deductible = data.get('deductible')
    selected_services = data.get('selected_services')
    if not all([naics, employee_size, deductible is not None, selected_services]):
        return jsonify({"error": "Missing required parameters"}), 400

    N_ITERATIONS = 1000
    cyber_data = SOURCE_DF.copy()

    filtered_data = cyber_data[
        (cyber_data['Industry NAICS'] == int(naics)) &
        (cyber_data['Code'].isin(selected_services))
    ].copy()

    if filtered_data.empty:
        return jsonify({"error": "No data available for the selected services in this industry."}), 400
    
    # Use 2026 data as the baseline for the simulation
    filtered_data['Event_Freq'] = filtered_data['Event_Freq_2026']
    filtered_data['Uptake_Prob'] = filtered_data['Uptake_Prob_2026']
    filtered_data['Cost'] = filtered_data['Cost_2026']

    # --- FULL SIMULATION LOGIC RESTORED ---
    for metric in ['Event_Freq', 'Uptake_Prob', 'Cost']:
        filtered_data[f'{metric}_Min'] = filtered_data[metric] * 0.7
        filtered_data[f'{metric}_Max'] = filtered_data[metric] * 1.3
    filtered_data['Uptake_Prob_Min'] = filtered_data['Uptake_Prob_Min'].clip(0, 1)
    filtered_data['Uptake_Prob_Max'] = filtered_data['Uptake_Prob_Max'].clip(0, 1)

    for metric in ['Event_Freq', 'Cost']:
        params = np.array([compute_lognormal_params(r[f'{metric}'], r[f'{metric}_Min'], r[f'{metric}_Max']) for _, r in filtered_data.iterrows()])
        if params.size > 0:
            filtered_data[f'{metric}_mu'] = params[:, 0]
            filtered_data[f'{metric}_sigma'] = params[:, 1]

    beta_params = np.array([compute_beta_params(r['Uptake_Prob'], r['Uptake_Prob_Min'], r['Uptake_Prob_Max']) for _, r in filtered_data.iterrows()])
    if beta_params.size > 0:
        filtered_data[f'{metric}_alpha'] = beta_params[:, 0]
        filtered_data[f'{metric}_beta'] = beta_params[:, 1]
    
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