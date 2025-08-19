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
51,CM.01,Forensics,0.099875,0.840947,1236194
51,CM.01,Legal Support,0.063804,0.755109,1102232
51,CM.01,Negotiation,0.064879,0.539190,1004232
51,CM.01,Recovery,0.064500,0.933181,3764551
51,ZD.01,Forensics,0.012267,0.839891,1517848
51,ZD.01,Legal Support,0.012627,0.791395,1079793
51,ZD.01,Negotiation,0.017700,0.584782,1187969
51,ZD.01,Recovery,0.012856,0.968716,5214239
51,FI.01,Forensics,0.015002,0.864842,1492517
51,FI.01,Legal Support,0.019987,0.776606,1183664
51,FI.01,Negotiation,0.018102,0.511590,1518824
51,FI.01,Recovery,0.012998,0.907718,4843962
51,EC.01,Forensics,0.025858,0.870633,2016638
51,EC.01,Legal Support,0.027063,0.882606,1366228
51,EC.01,Negotiation,0.018875,0.519152,1009315
51,EC.01,Recovery,0.018790,0.947130,5068851
51,SB.01,Forensics,0.001897,0.945000,2597141
51,SB.01,Legal Support,0.002125,0.888304,2231356
51,SB.01,Negotiation,0.002547,0.459714,2024259
51,SB.01,Recovery,0.001745,0.932563,8028469
51,VR.01,Forensics,0.001812,0.918642,1410297
51,VR.01,Legal Support,0.001225,0.733838,1004406
51,VR.01,Negotiation,0.002216,0.474860,1341001
51,VR.01,Recovery,0.002047,0.966778,4120691
52,AI.01,Forensics,0.015699,0.890668,1931080
52,AI.01,Legal Support,0.011901,0.814358,1317219
52,AI.01,Negotiation,0.018332,0.489247,1699474
52,AI.01,Recovery,0.010120,0.943700,5073681
52,QC.01,Forensics,0.000146,0.835327,4564115
52,QC.01,Legal Support,0.000661,0.738403,3692484
52,QC.01,Negotiation,0.000742,0.414057,2120983
52,QC.01,Recovery,0.000491,0.967052,13093367
52,BC.01,Forensics,0.024164,0.899533,1540441
52,BC.01,Legal Support,0.015190,0.828945,2692702
52,BC.01,Negotiation,0.015188,0.450785,1776904
52,BC.01,Recovery,0.014276,0.945869,4013270
52,FI.01,Forensics,0.006969,0.917508,1660181
52,FI.01,Legal Support,0.005433,0.704190,1205326
52,FI.01,Negotiation,0.009877,0.504318,1274103
52,FI.01,Recovery,0.009065,0.976960,5613728
52,BD.01,Forensics,0.007680,0.818690,2383953
52,BD.01,Legal Support,0.006423,0.891174,1385921
52,BD.01,Negotiation,0.005092,0.472989,1736065
52,BD.01,Recovery,0.008135,0.906624,6709273
54,AM.01,Forensics,0.005016,0.801431,2938989
54,AM.01,Legal Support,0.007303,0.727295,1724462
54,AM.01,Negotiation,0.008654,0.406448,1540228
54,AM.01,Recovery,0.007061,0.955676,7985966
54,DF.01,Forensics,0.007887,0.936062,963297
54,DF.01,Legal Support,0.006437,0.712893,856577
54,DF.01,Negotiation,0.008366,0.484150,712322
54,DF.01,Recovery,0.008377,0.913934,2988888
54,VR.01,Forensics,0.002765,0.944308,1783138
54,VR.01,Legal Support,0.002058,0.815760,1153583
54,VR.01,Negotiation,0.002124,0.541147,1397211
54,VR.01,Recovery,0.001647,0.947865,3852413
22,CP.01,Forensics,0.006035,0.898147,3380758
22,CP.01,Legal Support,0.006372,0.864525,2453948
22,CP.01,Negotiation,0.009101,0.418457,3023402
22,CP.01,Recovery,0.009792,0.974022,9404621
22,SB.01,Forensics,0.005905,0.882767,2156031
22,SB.01,Legal Support,0.005993,0.864031,1694812
22,SB.01,Negotiation,0.006680,0.419754,2229757
22,SB.01,Recovery,0.009372,0.954811,8557681
22,AP.01,Forensics,0.024894,0.845910,2869936
22,AP.01,Legal Support,0.025882,0.774137,1955599
22,AP.01,Negotiation,0.018758,0.460015,1737580
22,AP.01,Recovery,0.019511,0.967091,5872811
23,SC.01,Forensics,0.041429,0.922353,2854051
23,SC.01,Legal Support,0.048277,0.713576,2324196
23,SC.01,Negotiation,0.038605,0.460605,2394434
23,SC.01,Recovery,0.031512,0.923769,10441476
23,CM.01,Forensics,0.083717,0.890511,1334624
23,CM.01,Legal Support,0.061828,0.713567,1314588
23,CM.01,Negotiation,0.094193,0.534489,965479
23,CM.01,Recovery,0.068078,0.918430,4013268
23,ZD.01,Forensics,0.017152,0.934873,1735446
23,ZD.01,Legal Support,0.016212,0.892294,1394103
23,ZD.01,Negotiation,0.013013,0.503077,1030857
23,ZD.01,Recovery,0.014622,0.965479,4898346
23,BC.01,Forensics,0.027914,0.813520,2481560
23,BC.01,Legal Support,0.024936,0.770821,1371474
23,BC.01,Negotiation,0.027530,0.450438,3832345
23,BC.01,Recovery,0.010820,0.916614,6839615
61,AM.01,Forensics,0.008569,0.825898,1748791
61,AM.01,Legal Support,0.006449,0.793807,1961049
61,AM.01,Negotiation,0.007293,0.445340,1121666
61,AM.01,Recovery,0.007102,0.976063,5688002
62,RN.01,Forensics,0.092674,0.826598,1420101
62,RN.01,Legal Support,0.066336,0.882278,1650661
62,RN.01,Negotiation,0.089039,0.421123,1780137
62,RN.01,Recovery,0.052182,0.927362,4840344
62,BD.01,Forensics,0.005979,0.885193,2181427
62,BD.01,Legal Support,0.005761,0.728183,1582787
62,BD.01,Negotiation,0.005360,0.465187,2131667
62,BD.01,Recovery,0.009088,0.909013,5904185
71,AI.01,Forensics,0.017513,0.937812,1795388
71,AI.01,Legal Support,0.011472,0.780039,964273
71,AI.01,Negotiation,0.019523,0.874631,1395408
71,AI.01,Recovery,0.016361,0.936542,4419901
71,AM.01,Forensics,0.009545,0.841061,2840534
71,AM.01,Legal Support,0.005164,0.817691,1612265
71,AM.01,Negotiation,0.009868,0.848550,1034399
71,AM.01,Recovery,0.005168,0.936129,7155360
71,VR.01,Forensics,0.006978,0.816684,1610366
71,VR.01,Legal Support,0.007430,0.800377,1434268
71,VR.01,Negotiation,0.007030,0.872976,1194398
71,VR.01,Recovery,0.008777,0.952551,3803870
72,IO.01,Forensics,0.035602,0.815525,1227735
72,IO.01,Legal Support,0.043332,0.869952,1564752
72,IO.01,Negotiation,0.043541,0.897856,1520173
72,IO.01,Recovery,0.034970,0.919185,4296951
72,DF.01,Forensics,0.016833,0.836516,1100062
72,DF.01,Legal Support,0.015682,0.730012,1245512
72,DF.01,Negotiation,0.016798,0.853111,1824478
72,DF.01,Recovery,0.017891,0.959337,3261197
81,AP.01,Forensics,0.024876,0.830705,2246587
81,AP.01,Legal Support,0.022374,0.768097,1643094
81,AP.01,Negotiation,0.016762,0.892990,2152494
81,AP.01,Recovery,0.022499,0.964582,5849685
92,FI.01,Forensics,0.019192,0.926973,1259898
92,FI.01,Legal Support,0.010861,0.706868,2405494
92,FI.01,Negotiation,0.013164,0.817410,2058386
92,FI.01,Recovery,0.017958,0.936885,4030932
92,BD.01,Forensics,0.007289,0.800347,2196584
92,BD.01,Legal Support,0.006814,0.801638,2383934
92,BD.01,Negotiation,0.006078,0.888263,2849805
92,BD.01,Recovery,0.006246,0.975828,6111013
"""

# --- Configuration & Helper Functions ---
EMPLOYEE_SIZE_SCALING_FACTORS = {
    "<5": 0.1, "5-9": 0.2, "10-14": 0.35, "15-19": 0.5, "20-24": 0.65,
    "25-29": 0.8, "30-34": 1.0, "35-39": 1.2, "40-49": 1.5, "50-74": 1.9,
    "75-99": 2.4, "100-149": 3.0, "150-199": 3.7, "200-299": 4.5,
    "300-399": 5.4, "400-499": 6.4, "500-749": 7.9, "750-999": 9.9,
    "1,000-1,499": 12.0, "1,500-1,999": 14.0, "2,000-2,499": 16.0,
    "2,500-4,999": 18.0, "5,000+": 21.0
}
CATASTROPHE_PARAMETER_SETS = {
    "1": {"freq_min":0.01,"freq_max":0.04,"shape_min":3.01,"shape_max":4.0,"scale_min":0.40,"scale_max":0.80,"beta_alpha":5,"beta_beta":5},
    "2": {"freq_min":0.05,"freq_max":0.15,"shape_min":2.5,"shape_max":3.0,"scale_min":0.81,"scale_max":1.00,"beta_alpha":5,"beta_beta":5},
    "3": {"freq_min":0.16,"freq_max":0.25,"shape_min":2.01,"shape_max":2.5,"scale_min":1.01,"scale_max":1.50,"beta_alpha":5,"beta_beta":5}
}
def sample_catastrophic_load(params):
    freq_beta_sample = np.random.beta(params["beta_alpha"], params["beta_beta"])
    sampled_frequency = params["freq_min"] + freq_beta_sample * (params["freq_max"] - params["freq_min"])
    if np.random.binomial(1, sampled_frequency):
        shape_beta_sample = np.random.beta(params["beta_alpha"], params["beta_beta"])
        sampled_shape = params["shape_min"] + shape_beta_sample * (params["shape_max"] - params["shape_min"])
        scale_beta_sample = np.random.beta(params["beta_alpha"], params["beta_beta"])
        sampled_scale = params["scale_min"] + scale_beta_sample * (params["scale_max"] - params["scale_min"])
        severity_multiplier = (np.random.pareto(a=sampled_shape) + 1) * sampled_scale
        return severity_multiplier
    else:
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
    if not (min_val < mean_val < max_val):
        return 2.0, 2.0
    mean_normalized = (mean_val - min_val) / (max_val - min_val)
    variance_of_standard_beta = max((mean_normalized * (1 - mean_normalized))/6, 0.01)
    nu = (mean_normalized * (1 - mean_normalized) / variance_of_standard_beta) - 1
    if nu <= 0: return 2.0, 2.0
    alpha = mean_normalized * nu
    beta = (1 - mean_normalized) * nu
    alpha = max(alpha, 0.5); beta = max(beta, 0.5)
    return alpha, beta

# --- API Endpoint ---
@app.route('/api/main', methods=['POST'])
def handle_simulation():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. Please send a JSON body."}), 400

    naics = data.get('naics')
    employee_size = data.get('employee_size')
    deductible = data.get('deductible')
    selected_services = data.get('selected_services')
    catastrophe_outlook = data.get('catastrophe_outlook', '2')
    gross_up_percentage = data.get('gross_up_percentage', 55) # NEW

    if not all([naics, employee_size, deductible is not None, selected_services]):
        return jsonify({"error": "Missing one or more required parameters"}), 400
    
    cat_params = CATASTROPHE_PARAMETER_SETS.get(str(catastrophe_outlook))
    if not cat_params:
        return jsonify({"error": "Invalid catastrophe outlook value."}), 400

    try:
        naics_int = int(naics)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid NAICS code. Must be a valid integer."}), 400

    N_ITERATIONS = 10000
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

    catastrophic_loads = np.array([sample_catastrophic_load(cat_params) for _ in range(N_ITERATIONS)])
    loaded_simulated_loss = simulated_loss_per_firm * (1 + catastrophic_loads)
    simulated_pure_premium_dist = np.maximum(0, loaded_simulated_loss - int(deductible))

    pure_premium = simulated_pure_premium_dist.mean()
    std_dev_premium = simulated_pure_premium_dist.std(ddof=1)
    max_premium = simulated_pure_premium_dist.max()
    cv = (std_dev_premium / pure_premium) if pure_premium > 0 else 0
    max_mean_ratio = (max_premium / pure_premium) if pure_premium > 0 else 0
    
    # --- NEW CALCULATIONS ---
    # 1. Premium Gross-Up
    loss_ratio_factor = gross_up_percentage / 100.0
    final_premium = pure_premium / loss_ratio_factor if loss_ratio_factor > 0 else pure_premium
    total_load = final_premium - pure_premium
    # Assume a 2:1 split for Expense vs. Profit from the total load
    expense_load = total_load * (2/3)
    profit_load = total_load * (1/3)

    # 2. Conditional Loss
    losses_when_event_occurs = simulated_pure_premium_dist[simulated_pure_premium_dist > 0]
    conditional_loss = losses_when_event_occurs.mean() if len(losses_when_event_occurs) > 0 else 0.0

    # 3. VaR Percentiles
    var_95 = np.percentile(simulated_pure_premium_dist, 95)
    var_99 = np.percentile(simulated_pure_premium_dist, 99)
    var_99_9 = np.percentile(simulated_pure_premium_dist, 99.9)

    # UPDATED results dictionary
    results = {
        "premium": f"${final_premium:,.2f}",
        "pure_premium": f"${pure_premium:,.2f}",
        "expense_load": f"${expense_load:,.2f}",
        "profit_load": f"${profit_load:,.2f}",
        "conditional_loss": f"${conditional_loss:,.2f}",
        "var_95": f"${var_95:,.2f}",
        "var_99": f"${var_99:,.2f}",
        "var_99_9": f"${var_99_9:,.2f}",
        "volatility_cv": f"{cv:.1%}",
        "max_to_mean_ratio": f"{max_mean_ratio:.2f}x"
    }
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)