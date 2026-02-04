# Functions.py 

from brian2 import prefs
prefs.codegen.target = 'numpy'
import numpy as np 
import pickle 
import os 
import matplotlib.pyplot as plt 
import pandas as pd 
from brian2 import ms, Hz
import datetime
from Network_set import build_network 

def DataGeneration( 
    params, 
    num_trial=100, 
    evidence_strength=0.0, 
    threshold=60.0,     # Decision threshold (Hz) consistent with GUI 
    trial_len=2000.0,   # Total stimulus duration per trial (ms) 
    BG_len=300.0,       # Background activity duration (ms) 
    input_amp=0.00156   # Sensory input amplitude (matches GUI default) 
): 
    """ 
    Generate multiple trials of data using the Brian2 network defined by build_network(params). 
    Parameters 
    ---------- 
    params : dict 
        Dictionary of network parameters (gamma_ee, gamma_ie, gamma_ei, gamma_ii, etc.) 
        as defined in 'Network_set.py'. 
    num_trial : int 
        Number of trials to run and generate. 
    strength : float 
        Fraction or ratio that biases input to E_R vs. E_L (0.0 for equal input). 
        e.g., if strength=0.2, then E_R gets input_amp*(1+0.2), E_L gets input_amp*(1-0.2). 
    threshold : float 
        Decision threshold in Hz. If E_L or E_R exceed this rate, the network is considered to have made a choice. 
    trial_len : float 
        Duration (in ms) of the stimulus phase per trial. 
    BG_len : float 
        Duration (in ms) of the background phase before stimulus onset. 
    input_amp : float 
        Base amplitude of the input current to excitatory populations. 
    Returns 
    ------- 
    Result_list : dict 
        Keys are trial indices (0..num_trial-1). Each value is either None (no decision) 
        or a list/tuple with structure: 
        [ER_flag, EL_flag, ER_RT, EL_RT, times, difference_ER_EL, rateER, rateEL, rateIL, rateIR] 
        where 'ER_flag' / 'EL_flag' are 0/1 indicating which side fired, 
        'ER_RT' / 'EL_RT' are reaction times, 
        'difference_ER_EL' is the array of (ER - EL) at each time, etc. 
    total_EL_RT : list of floats 
        Reaction times for E_L decisions across all trials. 
    total_ER_RT : list of floats 
        Reaction times for E_R decisions across all trials. 
    NO_decision : np.ndarray of int (0 or 1) 
        1 if no decision was made for that trial, 0 otherwise. 
    EL_firing : np.ndarray of int (0 or 1) 
        1 if E_L reached threshold unambiguously in that trial, 0 otherwise. 
    ER_firing : np.ndarray of int (0 or 1) 
        1 if E_R reached threshold unambiguously in that trial, 0 otherwise. 
    """ 
    # 1) Build the network (make sure build_network is actually imported from Network_set) 
    net, eqs, neuron_groups, synapses, monitors = build_network(params) 
    # 2) Store the initial state for later resetting 
    net.store('default') 
    # 3) Create containers for results 
    total_EL_RT = [] 
    total_ER_RT = [] 
    NO_decision = np.zeros(num_trial, dtype=int) 
    EL_firing   = np.zeros(num_trial, dtype=int) 
    ER_firing   = np.zeros(num_trial, dtype=int) 
    Result_list = {} 
    EL_RT = None 
    ER_RT = None 
    # Define a "low threshold" (for partial checks) 
    low_threshold = threshold * 0.5 
    # Loop over trials 
    for k in range(num_trial): 
        net.restore('default') 
        # run background activity 
        neuron_groups['E_L'].I   = 0 
        neuron_groups['E_R'].I   = 0 
        neuron_groups['NSE'].I   = 0 
        neuron_groups['Inh_L'].I = 0 
        neuron_groups['Inh_R'].I = 0 
        net.run(BG_len * ms) 
        # sensory input 
        E_R_input = input_amp * (1.0 + evidence_strength/100) 
        E_L_input = input_amp * (1.0 - evidence_strength/100) 
        neuron_groups['E_R'].I = E_R_input 
        neuron_groups['E_L'].I = E_L_input 
        # trial phase 
        stim_elapsed = 0*ms 
        decision_made = False 
        chunk = 10*ms 
        while stim_elapsed < trial_len*ms and not decision_made: 
            remaining = trial_len*ms - stim_elapsed 
            this_step = chunk if remaining > chunk else remaining 
            net.run(this_step) 
            stim_elapsed += this_step 
            rateEL = monitors['rate_EL'].smooth_rate(width=10*ms)/Hz 
            rateER = monitors['rate_ER'].smooth_rate(width=10*ms)/Hz 
            times  = monitors['rate_EL'].t / ms 
            idx_high_EL = np.where(rateEL >= threshold)[0] 
            idx_high_ER = np.where(rateER >= threshold)[0] 
            idx_low_EL  = np.where(rateEL >= low_threshold)[0] 
            idx_low_ER  = np.where(rateER >= low_threshold)[0] 
            # for any >= threshold, turn off stimuli, run more 150ms 
            if (len(idx_high_EL) > 0) or (len(idx_high_ER) > 0): 
                net.run(150 * ms) 
                # --- for E_R win clearly --- 
                if (len(idx_high_ER) > 0 and 
                    (len(idx_low_EL) == 0 or 
                    float(len(list(set(idx_high_ER).intersection(idx_low_EL)))/ len(idx_high_ER)) < 0.6)): 
                     
                    neuron_groups['E_L'].I   = 0 
                    neuron_groups['E_R'].I   = 0 
                    net.run(200 * ms) 
                     
                    first_spike_idx = idx_high_ER[0] 
                    ER_RT = first_spike_idx / 10.0 
                    ER_firing[k] = 1 
                    total_ER_RT.append(ER_RT) 
                    difference_ER_EL = rateER - rateEL 
                    Result_list[k] = [ 
                        1, 0, ER_RT, None, 
                        list(times), 
                        list(difference_ER_EL), 
                        list(rateER), 
                        list(rateEL), 
                        list(monitors['rate_IL'].smooth_rate(width=20*ms)/Hz), 
                        list(monitors['rate_IR'].smooth_rate(width=20*ms)/Hz) 
                    ] 
                    decision_made = True 
                # --- for E_L win clearly -- 
                elif (len(idx_high_EL) > 0 and 
                    (len(idx_low_ER) == 0 or 
                    float(len(list(set(idx_high_EL).intersection(idx_low_ER)))/ len(idx_high_EL)) < 0.6)): 
         
                    neuron_groups['E_L'].I   = 0 
                    neuron_groups['E_R'].I   = 0 
                    net.run(200 * ms) 
                     
                    first_spike_idx = idx_high_EL[0] 
                    EL_RT = first_spike_idx / 10.0 
                    EL_firing[k] = 1 
                    total_EL_RT.append(EL_RT) 
                    difference_ER_EL = rateER - rateEL 
                    Result_list[k] = [ 
                        0, 1, None, EL_RT, 
                        list(times), 
                        list(difference_ER_EL), 
                        list(rateER), 
                        list(rateEL), 
                        list(monitors['rate_IL'].smooth_rate(width=20*ms)/Hz), 
                        list(monitors['rate_IR'].smooth_rate(width=20*ms)/Hz) 
                    ] 
                    decision_made = True 
                else: 
                    pass 
        if not decision_made: 
            NO_decision[k] = 1 
            Result_list[k] = None 
            ER_RT = None 
            EL_RT = None 
        print(f"Trial {k+1}/{num_trial}: EL_firing={EL_firing[k]}, " 
            f"ER_firing={ER_firing[k]}, NO_decision={NO_decision[k]}, Trial_RT={EL_RT if EL_firing[k] else ER_RT}") 
    # Return all results 
    return Result_list, total_EL_RT, total_ER_RT, NO_decision, EL_firing, ER_firing 

def save_result_with_metadata(
    file_path,
    params,
    evidence_strength,
    num_trial,
    threshold,
    trial_len,
    BG_len,
    input_amp,
    result_list
):
    """
    Save the result_list and metadata to a .pkl file with conditional top_down info.
    """
    meta = {
        "gamma_ee": params.get('gamma_ee', 0.0),
        "gamma_ei": params.get('gamma_ei', 0.0),
        "gamma_ie": params.get('gamma_ie', 0.0),
        "gamma_ii": params.get('gamma_ii', 0.0),
        "evidence_strength": evidence_strength,
        "num_trial": num_trial,
        "threshold": threshold,
        "trial_len": trial_len,
        "BG_len": BG_len,
        "input_amp": input_amp,
        "timestamp": datetime.datetime.now().isoformat(),
        "top_down": params.get('top_down', False) 
    }

    if meta["top_down"]:
        meta['S'] = params.get('S')
        meta['R'] = params.get('R')

    data_to_save = {
        "metadata": meta,
        "result_list": result_list
    }

    with open(file_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"[save_result_with_metadata] Save done: {file_path}") 

def run_experiment(
    params,
    evidence_strength_list=[0, 3.2, 6.4, 12.8, 25.6, 51.2],
    evidence_strength_index=1,
    num_trial=50,
    threshold=40,
    trial_len=2000,
    BG_len=300,
    input_amp=0.00156,
    trial_n='0'
):
    """
    Run DataGeneration for one strength from the list and save a single .pkl with metadata.
    """
    evidence_strength = evidence_strength_list[evidence_strength_index]

    (Result_list, total_EL_RT, total_ER_RT, NO_decision, EL_firing, ER_firing) = DataGeneration(
        params=params, num_trial=num_trial, evidence_strength=evidence_strength, threshold=threshold,
        trial_len=trial_len, BG_len=BG_len, input_amp=input_amp
    )

    # 1. 建立 metadata
    meta = {
        "gamma_ee":          params.get('gamma_ee', 0.0),
        "gamma_ei":          params.get('gamma_ei', 0.0),
        "gamma_ie":          params.get('gamma_ie', 0.0),
        "gamma_ii":          params.get('gamma_ii', 0.0),
        "evidence_strength": evidence_strength,
        "num_trial":         num_trial,
        "threshold":         threshold,
        "trial_len":         trial_len,
        "BG_len":            BG_len,
        "input_amp":         input_amp,
        "timestamp":         datetime.datetime.now().isoformat(),
        "top_down":          params.get('top_down', False)
    }
    filename_base = f"EE{meta['gamma_ee']}_EI{meta['gamma_ei']}_IE{meta['gamma_ie']}_II{meta['gamma_ii']}"
    
    if meta["top_down"]:
        meta['S'] = params.get('S')
        meta['R'] = params.get('R')
        filename_base += f"_S{meta['S']}_R{meta['R']}"

    folder_name = "data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    filename = os.path.join(
        folder_name,
        f"{filename_base}_strength{evidence_strength}_trialCount{num_trial}_trialN{trial_n}.pkl"
    )
    data_to_save = {
        "metadata": meta,
        "result_list": Result_list
    }

    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Saved data with metadata to {filename}")
    return (Result_list, total_EL_RT, total_ER_RT, NO_decision, EL_firing, ER_firing)

def load_and_merge_data_with_metadata(file_paths, initial_offset=0, step=100):
    """
    Load several .pkl files,
    check metadata match (only: evidence_strength / top_down / S,R),
    and merge trials into one dict (reindexing keys with an offset).
    """
    if not file_paths:
        raise ValueError("No file paths given.")

    merged_dict = {}
    key_offset = initial_offset
    base_metadata = None

    for i, path in enumerate(file_paths):
        with open(path, 'rb') as f:
            loaded_data = pickle.load(f)

        if "metadata" not in loaded_data or "result_list" not in loaded_data:
            raise ValueError(f"File {path} does not contain 'metadata' or 'result_list'")

        meta = loaded_data["metadata"]
        res  = loaded_data["result_list"]

        # ---- strict requirement: must have evidence_strength (no downward compatibility) ----
        if "evidence_strength" not in meta:
            raise ValueError(f"File {path} metadata missing required key: 'evidence_strength'")

        if base_metadata is None:
            base_metadata = meta
            # also enforce base has evidence_strength
            if "evidence_strength" not in base_metadata:
                raise ValueError(f"Base file {path} metadata missing required key: 'evidence_strength'")
        else:
            base_td = base_metadata.get('top_down', False)
            meta_td = meta.get('top_down', False)

            error_msg = ""
            # FIX: compare evidence_strength, not "strength"
            if base_metadata["evidence_strength"] != meta["evidence_strength"]:
                error_msg = "Evidence Strength mismatch"
            elif base_td != meta_td:
                error_msg = "top_down flag mismatch"
            elif base_td and (base_metadata.get('S') != meta.get('S') or base_metadata.get('R') != meta.get('R')):
                error_msg = "S or R values mismatch"

            if error_msg:
                raise ValueError(
                    f"Metadata mismatch in file {path}: {error_msg}.\n"
                    f"Base: {base_metadata}\n"
                    f"Current: {meta}"
                )

        # Merge data
        shifted = {(k + key_offset): v for k, v in res.items()}
        merged_dict.update(shifted)
        key_offset += step

    return merged_dict, base_metadata

def _strength_to_log10(strength_list, zero_replacement=1e-3):
    """
    Convert strength (%) to log10(x) for plotting.
    - x==0 -> zero_replacement (avoid log10(0))
    Return: x_log10, x_safe
    """
    x = np.asarray(strength_list, dtype=float)
    if np.any(x < 0):
        raise ValueError(f"evidence_strength contains negative values: {x[x < 0]}")
    x_safe = np.where(x == 0, float(zero_replacement), x)
    x_log10 = np.log10(x_safe)
    return x_log10, x_safe

def _apply_log10_ticks_from_strengths(ax, strength_list, zero_replacement=1e-3):
    """
    Put xticks at log10(strength), but label them using original strength values.
    Ensures 0 is shown as '0' on axis (mapped to log10(zero_replacement)).
    """
    xs = np.asarray(strength_list, dtype=float)

    # unique sorted ticks
    ticks = np.unique(xs)
    ticks_safe = np.where(ticks == 0, float(zero_replacement), ticks)
    ax.set_xticks(np.log10(ticks_safe))

    def _fmt(v):
        if abs(v) < 1e-12:
            return "0"
        # keep 3.2 / 12.8 etc
        s = f"{v:.6g}"
        return s

    ax.set_xticklabels([_fmt(v) for v in ticks])

def _fit_psychometric_logistic_log10(x, y, n_alpha=200, n_beta=200):
    """
    Fit y ~ A + (B-A) * sigmoid((log10(x) - alpha)/beta)
    Return (A, B, alpha, beta)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # avoid log issues
    x = np.clip(x, 1e-6, None)
    logx = np.log10(x)

    # search ranges (夠用、穩定，不會太慢)
    a_min, a_max = float(logx.min()) - 0.5, float(logx.max()) + 0.5
    b_min, b_max = 0.05, 1.5

    alphas = np.linspace(a_min, a_max, n_alpha)
    betas  = np.linspace(b_min, b_max, n_beta)

    best = None
    best_sse = np.inf

    # grid search alpha,beta; given them, A,C are linear => least squares
    for alpha in alphas:
        # vectorize beta loop a bit
        for beta in betas:
            f = 1.0 / (1.0 + np.exp(-(logx - alpha) / beta))  # sigmoid

            # y = A + C*f ; solve A,C via least squares
            X = np.column_stack([np.ones_like(f), f])
            A, C = np.linalg.lstsq(X, y, rcond=None)[0]
            B = A + C

            # optional: clamp to [0,1] if you want nice psychometric bounds
            A2 = np.clip(A, 0.0, 1.0)
            B2 = np.clip(B, 0.0, 1.0)
            C2 = B2 - A2

            yhat = A2 + C2 * f
            sse = float(np.sum((y - yhat) ** 2))

            if sse < best_sse:
                best_sse = sse
                best = (A2, B2, float(alpha), float(beta))

    return best  # (A, B, alpha, beta)

def _psychometric_curve(x, A, B, alpha, beta):
    x = np.asarray(x, float)
    x = np.clip(x, 1e-6, None)
    logx = np.log10(x)
    f = 1.0 / (1.0 + np.exp(-(logx - alpha) / beta))
    return A + (B - A) * f

def calculate_performance(result_list): 
    """ 
    Calculate performance metrics from a Result_list dictionary. 
    result_list[k] format (example): 
      [ER_flag, EL_flag, ER_RT, EL_RT, times, difference_ER_EL, rateER, rateEL, rateIL, rateIR] 
    """ 
    ER_count = 0 
    EL_count = 0 
    ER_reaction_times = [] 
    for trial in result_list.values(): 
        if trial is not None: 
            ER = trial[0]    # 1 if E_R side fired 
            EL = trial[1]    # 1 if E_L side fired 
            ER_RT = trial[2] # E_R reaction time 
            ER_count += ER 
            EL_count += EL 
            if ER == 1 and ER_RT is not None: 
                ER_reaction_times.append(ER_RT) 
    total_decisions = ER_count + EL_count 
    performance = None 
    avg_ER_reaction_time = None 
    std_ER_reaction_time = None 
    if total_decisions > 0: 
        performance = ER_count / total_decisions 
    if len(ER_reaction_times) > 0: 
        avg_ER_reaction_time = np.mean(ER_reaction_times) 
        std_ER_reaction_time = np.std(ER_reaction_times) 
    return performance, avg_ER_reaction_time, std_ER_reaction_time 

def calculate_fail_rate(result_list): 
    """ 
    Fraction of trials for which 'trial is None' (i.e., no decision). 
    """ 
    total = len(result_list) 
    if total == 0: 
        return None 
    no_decisions = sum(trial is None for trial in result_list.values()) 
    return no_decisions / total 

def plot_performance_and_rt(
    strength_list,
    results_polA,
    results_polB,
    labelA="Pol0",
    labelB="Pol2"
):
    perfA, rtA, rtStdA = [], [], []
    perfB, rtB, rtStdB = [], [], []

    for ss in strength_list:
        strength_key = f"strength{ss}"
        if strength_key in results_polA:
            pA = results_polA[strength_key]["Performance"]
            rA = results_polA[strength_key]["Avg_ER_Reaction_Time"]
            rsA= results_polA[strength_key]["Std_ER_Reaction_Time"]
        else:
            pA, rA, rsA = None, None, None
        perfA.append(pA); rtA.append(rA); rtStdA.append(rsA)

        if strength_key in results_polB:
            pB = results_polB[strength_key]["Performance"]
            rB = results_polB[strength_key]["Avg_ER_Reaction_Time"]
            rsB= results_polB[strength_key]["Std_ER_Reaction_Time"]
        else:
            pB, rB, rsB = None, None, None
        perfB.append(pB); rtB.append(rB); rtStdB.append(rsB)

    perfA = np.array([p if p is not None else np.nan for p in perfA])
    rtA   = np.array([r if r is not None else np.nan for r in rtA])
    rtStdA= np.array([s if s is not None else np.nan for s in rtStdA])
    perfB = np.array([p if p is not None else np.nan for p in perfB])
    rtB   = np.array([r if r is not None else np.nan for r in rtB])
    rtStdB= np.array([s if s is not None else np.nan for s in rtStdB])

    # --- x transform ---
    x_log10, _ = _strength_to_log10(strength_list, zero_replacement=1e-3)

    # Psychometric curve
    plt.figure(figsize=(10, 5))
    plt.plot(x_log10, perfA, 'o-', label=labelA)
    plt.plot(x_log10, perfB, 'o-', label=labelB)
    plt.title("Psychometric Curve (Performance vs. Evidence Strength)")
    plt.xlabel("Evidence Strength (%)")
    plt.ylabel("Performance (fraction of E_R decisions)")
    plt.grid(True)
    _apply_log10_ticks_from_strengths(plt.gca(), strength_list, zero_replacement=1e-3)
    plt.legend()
    plt.show()

    # Reaction time with error bars
    plt.figure(figsize=(10, 5))
    plt.errorbar(x_log10, rtA, yerr=rtStdA, fmt='o-', capsize=5, label=labelA)
    plt.errorbar(x_log10, rtB, yerr=rtStdB, fmt='o-', capsize=5, label=labelB)
    plt.title("Reaction Time vs. Evidence Strength")
    plt.xlabel("Evidence Strength (%)")
    plt.ylabel("Reaction Time (ms)")
    plt.grid(True)
    _apply_log10_ticks_from_strengths(plt.gca(), strength_list, zero_replacement=1e-3)
    plt.legend()
    plt.show()

def compute_non_regret_performance(result_list, bm): 
    """ 
    Compute a Non-Regret performance array over a range of benchmarks (bm). 
    For each trial, check difference_array=trial[5]. If difference_array[i]>bm => E_R "triggered". 
    If E_R side actually fired => it's a non-regret success. 
    """ 
    import numpy as np 
    NR_performance = [] 
    for threshold_bm in bm: 
        ER_NR = 0 
        ER_total = 0 
        for trial in result_list.values(): 
            if trial is not None: 
                difference_array = trial[5] 
                ER_flag = trial[0]  # 1 if E_R fired 
                for val in difference_array: 
                    if val > threshold_bm: 
                        ER_total += 1 
                        if ER_flag == 1: 
                            ER_NR += 1 
                        break 
        if ER_total > 0: 
            NR_performance.append(ER_NR / ER_total) 
        else: 
            NR_performance.append(np.nan) 
    return NR_performance 

def plot_non_regret(bm, NR_results_dict, labels_dict, title="Non-Regret Performance"): 
    """ 
    Plot Non-Regret performance vs. benchmark for multiple conditions. 
    """ 
    plt.figure(figsize=(10, 5)) 
    for key, NR_data in NR_results_dict.items(): 
        label = labels_dict.get(key, key) 
        plt.plot(bm, NR_data, marker='o', linestyle='-', label=label) 
    plt.title(title) 
    plt.xlabel("Benchmark") 
    plt.ylabel("Non-Regret Performance") 
    plt.grid(True) 
    plt.legend() 
    plt.show() 

def full_analysis_pipeline(file_paths_dict, benchmark_array): 
    """ 
    Merge data from multiple .pkl files (grouped by evidence strength), then compute performance,  
    reaction times, fail rates, and non-regret metrics for each evidence strength. 
    """ 
    all_merged_results = {} 
    offset = 0 
    for strength, paths in file_paths_dict.items(): 
        merged_dict, offset = load_and_merge_data_with_metadata(paths, initial_offset=offset, step=100) 
        all_merged_results[f"strength{strength}"] = merged_dict 
    perf_dict = {} 
    non_regret_dict = {} 
    for strength, merged_dict in all_merged_results.items(): 
        perf, rt_mean, rt_std = calculate_performance(merged_dict) 
        fail_rate = calculate_fail_rate(merged_dict) 
        perf_dict[strength] = { 
            "Performance": perf, 
            "Avg_ER_Reaction_Time": rt_mean, 
            "Std_ER_Reaction_Time": rt_std, 
            "Fail_Rate": fail_rate 
        } 
        # Non-Regret 
        nr_data = compute_non_regret_performance(merged_dict, benchmark_array) 
        non_regret_dict[strength] = nr_data 
    return all_merged_results, perf_dict, non_regret_dict 

def plot_energy_landscapes(
        Result_list,
        time_window,
        plot_full_scale=False,
        EXC=True,
        lim=40,
        color=None,
        label=None,
        delta_ms=20.0):
    """
    Quick script API: estimate and plot quasi-potential U(x) from ΔE or ΔI; 
    supports full-scale and time-windowed curves.
    """

    plt.figure(figsize=(10, 6))
    plotted_any = False

    def _get_dt_and_stepN():
        for td in Result_list.values():
            if td is None:
                continue
            t = np.asarray(td[4], dtype=float)  # times in ms
            if t.size >= 2:
                dt = float(np.median(np.diff(t)))   # ms / step
                stepN = max(1, int(round(delta_ms / dt)))
                return dt, stepN
        return 1.0, int(round(delta_ms))  # fallback

    dt, stepN = _get_dt_and_stepN()

    def _exc_arr(td):
        return np.asarray(td[5], dtype=float)

    def _inh_arr(td):
        ir = np.asarray(td[9], dtype=float)
        il = np.asarray(td[8], dtype=float)
        n = min(ir.size, il.size)
        return ir[:n] - il[:n]

    # ---------- 1) time-window U(x) ----------
    for (start_ms, end_ms) in time_window:
        vel_map = {}

        for td in Result_list.values():
            if td is None:
                continue

            # RT（ms）
            RT = td[2] if td[2] is not None else td[3]
            if RT is None:
                continue

            times = np.asarray(td[4], dtype=float)  # ms
            rt_idx = int(np.searchsorted(times, RT, side='left'))

            start_idx = max(0, rt_idx + int(round(start_ms / dt)))
            end_idx   = max(0, rt_idx + int(round(end_ms / dt)))

            arr = _exc_arr(td) if EXC else _inh_arr(td)
            if start_idx >= len(arr) - stepN:
                continue

            for j in range(start_idx, min(end_idx, len(arr) - stepN)):
                x = arr[j]
                if -lim <= x <= lim and not np.isnan(x):
                    dv = arr[j + stepN] - arr[j]
                    if not np.isnan(dv):
                        vel_map.setdefault(float(x), []).append(float(dv))

        if not vel_map:
            continue

        df = pd.DataFrame([(k, v) for k, a in vel_map.items() for v in a],
                          columns=['x', 'velocity'])
        df = df[(df['x'] >= -lim) & (df['x'] <= lim)]
        bins = np.arange(-lim, lim + 0.1, 0.1)
        df['bin'] = pd.cut(df['x'], bins=bins)
        g = df.groupby('bin', observed=False).mean(numeric_only=True)
        g['x'] = [c.left + 0.05 for c in g.index.categories]
        g['U(x)'] = (-g['velocity']).cumsum()
        g['U(x)'] -= g['U(x)'].iloc[(np.abs(g['x'])).argmin()]

        kw = {}
        if color is not None: kw['color'] = color
        kw['label'] = label if label is not None else f"U(x): {start_ms}–{end_ms} ms"
        plt.plot(g['x'].values, g['U(x)'].values.astype(float), **kw)
        plotted_any = True

    # ---------- 2) full-scale U(x) ----------
    if plot_full_scale:
        vel_map = {}
        for td in Result_list.values():
            if td is None:
                continue
            arr = _exc_arr(td) if EXC else _inh_arr(td)
            for j in range(0, len(arr) - stepN):
                x = arr[j]
                if -lim <= x <= lim and not np.isnan(x):
                    dv = arr[j + stepN] - arr[j]
                    if not np.isnan(dv):
                        vel_map.setdefault(float(x), []).append(float(dv))

        if vel_map:
            df = pd.DataFrame([(k, v) for k, a in vel_map.items() for v in a],
                              columns=['x', 'velocity'])
            df = df[(df['x'] >= -lim) & (df['x'] <= lim)]
            bins = np.arange(-lim, lim + 0.1, 0.1)
            df['bin'] = pd.cut(df['x'], bins=bins)
            g = df.groupby('bin', observed=False).mean(numeric_only=True)
            g['x'] = [c.left + 0.05 for c in g.index.categories]
            g['U(x)'] = (-g['velocity']).cumsum()
            g['U(x)'] -= g['U(x)'].iloc[(np.abs(g['x'])).argmin()]

            kw = {'linestyle': '--'}
            if color is not None: kw['color'] = color
            kw['label'] = label if label is not None else 'Full scale U(x)'
            plt.plot(g['x'].values, g['U(x)'].values.astype(float), **kw)
            plotted_any = True

    plt.xlabel('ΔE rate (Hz)' if EXC else 'ΔI rate (Hz)')
    plt.ylabel('Potential U(x) (a.u.)')
    plt.title('Energy Landscapes')
    if plotted_any:
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_energy_landscapes_on_figure(
        Result_list, fig, ax, time_window, plot_full_scale,
        EXC=True, lim=40, color=None, label=None, reset_ax=True,
        boundary_value=None):
    """Figure-API version: draw U(x) on the given Matplotlib axes; 
    supports RT-aligned or absolute windows and optional bounds.
    """

    if reset_ax:
        ax.clear()
    if not Result_list:
        raise ValueError("Result_list cannot be empty")

    # ---------- helpers ----------
    def _get_arr_and_times(td):
        t = np.asarray(td[4], dtype=float)
        if EXC:
            arr = np.asarray(td[5], dtype=float)          # ER-EL
        else:
            ir = np.asarray(td[9], dtype=float)
            il = np.asarray(td[8], dtype=float)
            n = min(ir.size, il.size)
            arr = ir[:n] - il[:n]
        L = min(len(t), len(arr))
        t = t[:L]; arr = arr[:L]
        m = np.isfinite(arr)
        return t[m], arr[m]

    def _estimate_dt_stepN():
        for td in Result_list.values():
            if td is None:
                continue
            t, arr = _get_arr_and_times(td)
            if t.size >= 2 and arr.size >= 2:
                dt = float(np.median(np.diff(t)))
                return (dt if dt > 0 else 1.0), max(1, int(round(20.0 / max(dt, 1e-9))))
        return 1.0, 20

    def _add_curve(vel_map, default_lbl, linestyle='-'):
        if not vel_map:
            return False

        df = pd.DataFrame([(x, v) for x, arr in vel_map.items() for v in arr],
                        columns=['x', 'v'])
        df = df[np.isfinite(df['x']) & np.isfinite(df['v'])]
        if df.empty:
            return False

        if boundary_value is not None and boundary_value > 0:
            x0 = max(-lim, -abs(boundary_value))
            x1 = min( lim,  abs(boundary_value))
        else:
            x0, x1 = -lim, lim

        df = df[(df['x'] >= x0) & (df['x'] <= x1)]
        if df.empty:
            return False

        bin_w = 0.1
        bins = np.arange(x0, x1 + bin_w*0.999 + 1e-12, bin_w)
        if len(bins) < 2:
            return False
        df['bin'] = pd.cut(df['x'], bins=bins, include_lowest=True)

        g = df.groupby('bin', observed=False)['v'].mean().to_frame()
        if g.empty:
            return False

        g['x'] = [ (b.left + b.right)/2 for b in g.index.categories ]
        g = g.sort_values('x')

        g['U'] = (-g['v']).cumsum()
        if boundary_value is not None and boundary_value > 0:
            g['U'] -= g['U'].iloc[0]  # U(-B) = 0
        else:
            g['U'] -= g['U'].iloc[np.argmin(np.abs(g['x']))] 

        kw = {'linestyle': linestyle}
        if color is not None:
            kw['color'] = color
        kw['label'] = label if label is not None else default_lbl
        ax.plot(g['x'].values, g['U'].values.astype(float), **kw)
        return True

    def _best_rt_index(RT, t, dt, arr_len, stepN):
        candidates_ms = []
        if RT is not None:
            candidates_ms = [float(RT), float(RT)*10.0]
        scores = []
        for RT_ms in candidates_ms:
            idx = int(round((RT_ms - t[0]) / dt))
            idx = max(0, min(idx, arr_len - 1 - stepN))
            start = idx + int(round(0 / dt))
            end   = idx + int(round(200.0 / dt))
            valid = max(0, min(end, arr_len - stepN) - max(0, start))
            scores.append((valid, idx))
        if scores:
            scores.sort(reverse=True)
            return scores[0][1]
        return None

    dt, stepN = _estimate_dt_stepN()
    plotted = False

    # ---------- Full scale ----------
    if plot_full_scale:
        vel = {}
        for td in Result_list.values():
            if td is None:
                continue
            t, arr = _get_arr_and_times(td)
            if arr.size <= stepN:
                continue
            for j in range(0, arr.size - stepN):
                x = arr[j]; dv = arr[j + stepN] - arr[j]
                if -lim <= x <= lim and np.isfinite(dv):
                    vel.setdefault(float(x), []).append(float(dv))
        plotted |= _add_curve(vel, 'Full scale U(x)', '--')

    # ---------- Time windows ----------
    for (st_ms, ed_ms) in time_window:
        vel = {}
        for td in Result_list.values():
            if td is None:
                continue
            t, arr = _get_arr_and_times(td)
            if arr.size <= stepN:
                continue

            RT = td[2] if td[2] is not None else td[3]
            rt_idx = _best_rt_index(RT, t, dt, arr_len=arr.size, stepN=stepN)

            if rt_idx is None:
                # 沒 RT → 用絕對時間窗
                start = int(round(st_ms / dt))
                end   = int(round(ed_ms / dt))
            else:
                start = rt_idx + int(round(st_ms / dt))
                end   = rt_idx + int(round(ed_ms / dt))

            if start >= arr.size - stepN:
                start = int(round(st_ms / dt))
                end   = int(round(ed_ms / dt))
                if start >= arr.size - stepN:
                    continue

            for j in range(start, min(end, arr.size - stepN)):
                x = arr[j]; dv = arr[j + stepN] - arr[j]
                if -lim <= x <= lim and np.isfinite(dv):
                    vel.setdefault(float(x), []).append(float(dv))

        plotted |= _add_curve(vel, f"U(x): {st_ms}–{ed_ms} ms")
    ax.set_xlabel('ΔE rate (Hz)' if EXC else 'ΔI rate (Hz)')
    ax.set_ylabel('U(x) (a.u.)')
    ax.set_title('Energy Landscapes')
    ax.grid(True)
    if plotted or ax.has_data():
        ax.legend(frameon=False)

def plot_perf_rt_on_figure(strength_list, perf, rt_mean, rt_std, fig=None, ax=None):
    """Draw combined Performance (left y) and RT (right y) in one figure, with log10 x-axis."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    x_arr  = np.array([float(s) for s in strength_list], dtype=float)
    x_safe = np.where(x_arr == 0.0, 1e-3, x_arr)

    # log10 axis, draw from 1
    ax.set_xscale('log', base=10)
    ax.set_xlim(1.0, max(100.0, float(np.max(x_arr[x_arr > 0])) * 1.15))

    # hide marker at the fake 0.001 point (index 0)
    mark_idx = list(range(1, len(x_safe)))

    ax.plot(x_safe, perf, '-o', markevery=mark_idx, color='C0', label='Performance')
    ax.set_xlabel('Evidence Strength (%)')
    ax.set_ylabel('Performance (fraction E_R)', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=.3)

    ax_rt = ax.twinx()

    # RT line includes the fake point so the curve shape matches,
    # but errorbars/markers exclude it.
    ax_rt.plot(x_safe, rt_mean, '--', color='C3', label='Reaction Time')
    ax_rt.errorbar(
        x_safe[1:], np.array(rt_mean)[1:], yerr=np.array(rt_std)[1:],
        fmt='s', capsize=4, color='C3', ecolor='C3', elinewidth=1
    )

    ax_rt.set_ylabel('Reaction Time (ms)', color='C3')
    ax_rt.tick_params(axis='y', labelcolor='C3')
    ymax_rt = max(m+s for m, s in zip(rt_mean, rt_std)) if len(rt_mean) else 1.0
    ax_rt.set_ylim(0, ymax_rt * 1.1)

    # ticks like paper (optional but looks good)
    tick_vals = [1.0, 3.2, 6.4, 12.8, 25.6, 51.2, 100.0]
    xmax = ax.get_xlim()[1]
    tick_vals = [t for t in tick_vals if 1.0 <= t <= xmax]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"{t:g}" for t in tick_vals])

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_rt.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2,
              loc='upper center', bbox_to_anchor=(0.5, 1.18),
              ncol=2, frameon=False)

    fig.tight_layout()
    return fig, ax, ax_rt
