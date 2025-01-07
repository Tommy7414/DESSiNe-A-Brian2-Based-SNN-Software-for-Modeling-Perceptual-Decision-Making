# Functions.py

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from brian2 import *

from Network_set import build_network


def DataGeneration(
    params,
    num_trial=100,
    coherence=0.0,
    threshold=35.0,     # Decision threshold (Hz) consistent with GUI
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
    coherence : float
        Fraction or ratio that biases input to E_R vs. E_L (0.0 for equal input).
        e.g., if coherence=0.2, then E_R gets input_amp*(1+0.2), E_L gets input_amp*(1-0.2).
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

    # 4) Define a "low threshold" (for partial checks)
    low_threshold = threshold * 0.5

    # 5) Loop over trials
    for k in range(num_trial):
        # 5.1) Reset the network to the initial state
        net.restore('default')

        # 5.2) Run the background phase
        neuron_groups['E_L'].I   = 0.0
        neuron_groups['E_R'].I   = 0.0
        neuron_groups['NSE'].I   = 0.0
        neuron_groups['Inh_L'].I = 0.0
        neuron_groups['Inh_R'].I = 0.0
        net.run(BG_len * ms)

        # 5.3) Set the stimulus input
        # If coherence=0.2 => E_R_input = input_amp*(1.2), E_L_input = input_amp*(0.8)
        E_R_input = input_amp * (1.0 + coherence)
        E_L_input = input_amp * (1.0 - coherence)

        neuron_groups['E_R'].I = E_R_input
        neuron_groups['E_L'].I = E_L_input

        # We'll simulate in small chunks (10 ms) to detect threshold in real-time
        chunk = 10 * ms
        time_elapsed = 0 * ms
        decision_made = False

        while time_elapsed < trial_len * ms and not decision_made:
            remaining = (trial_len * ms) - time_elapsed
            this_step = chunk if remaining > chunk else remaining
            net.run(this_step)
            time_elapsed += this_step

            # 5.4) Obtain population firing rates
            rateEL = monitors['rate_EL'].smooth_rate(width=20 * ms) / Hz
            rateER = monitors['rate_ER'].smooth_rate(width=20 * ms) / Hz
            times  = monitors['rate_EL'].t / ms

            # Ensure array lengths match
            min_len = min(len(rateEL), len(rateER), len(times))
            rateEL = rateEL[:min_len]
            rateER = rateER[:min_len]
            times  = times[:min_len]

            # Find indices where E_L or E_R exceed the threshold
            idx_high_EL = np.where(rateEL >= threshold)[0]
            idx_high_ER = np.where(rateER >= threshold)[0]
            idx_low_EL  = np.where(rateEL >= low_threshold)[0]
            idx_low_ER  = np.where(rateER >= low_threshold)[0]

            # Check whether either side reached threshold unambiguously
            if (len(idx_high_EL) > 0) or (len(idx_high_ER) > 0):
                # Decide which side "won"
                # "Unambiguous": other side doesn't come close to half-threshold at same times
                if (len(idx_high_ER) > 0 and
                    (len(idx_low_EL) == 0 or 
                     float(len(set(idx_high_ER) & set(idx_low_EL))) / len(idx_high_ER) < 0.6)):
                    # E_R side wins
                    first_spike_idx = idx_high_ER[0]
                    ER_RT = first_spike_idx / 10.0  # index => (index * 10 ms)
                    ER_firing[k] = 1
                    total_ER_RT.append(ER_RT)

                    difference_ER_EL = rateER - rateEL
                    Result_list[k] = [
                        1, 0, ER_RT, None,
                        list(times),
                        list(difference_ER_EL),
                        list(rateER),
                        list(rateEL),
                        list(monitors['rate_IL'].smooth_rate(width=20*ms)[:min_len]/Hz),
                        list(monitors['rate_IR'].smooth_rate(width=20*ms)[:min_len]/Hz)
                    ]
                    decision_made = True

                elif (len(idx_high_EL) > 0 and
                      (len(idx_low_ER) == 0 or 
                       float(len(set(idx_high_EL) & set(idx_low_ER))) / len(idx_high_EL) < 0.6)):
                    # E_L side wins
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
                        list(monitors['rate_IL'].smooth_rate(width=20*ms)[:min_len]/Hz),
                        list(monitors['rate_IR'].smooth_rate(width=20*ms)[:min_len]/Hz)
                    ]
                    decision_made = True
                else:
                    # Ambiguous or both fired => no definitive decision
                    NO_decision[k] = 1
                    Result_list[k] = None
                    decision_made = True

        # 5.5) If time ran out with no decision:
        if not decision_made:
            NO_decision[k] = 1
            Result_list[k] = None

        print(f"Trial {k+1}/{num_trial}: EL_firing={EL_firing[k]}, ER_firing={ER_firing[k]}, NO_decision={NO_decision[k]}")

    # Return all results
    return Result_list, total_EL_RT, total_ER_RT, NO_decision, EL_firing, ER_firing


def run_experiment(
    params,
    coherence_list=[0, 3.2, 6.4, 12.8, 25.6, 51.2],
    coherence_index=1,
    num_trial=50,
    threshold=40,
    trial_len=2000,
    BG_len=300,
    input_amp=0.00156,
    trial_n='0'
):
    """
    Run a single experiment using DataGeneration, then save results to .pkl.

    Parameters
    ----------
    params : dict
        Network parameters (with gamma_ee, gamma_ei, gamma_ie, gamma_ii, etc.).
    coherence_list : list of floats
        Possible coherence values to pick from.
    coherence_index : int
        Index in coherence_list for the coherence to use.
    num_trial : int
        Number of trials.
    threshold : float
        Firing-rate threshold for decisions (Hz).
    trial_len : float
        Stimulus duration (ms).
    BG_len : float
        Background duration (ms).
    input_amp : float
        Sensory input amplitude (default = 0.00156).
    trial_n : str
        Label/index appended to the output filename (e.g. '0', '1', etc.).

    Returns
    -------
    (Result_list, total_EL_RT, total_ER_RT, NO_decision, EL_firing, ER_firing) : tuple
        The same tuple from DataGeneration. Also saves the Result_list to a pickle file 
        including gamma params and coherence in its name.
    """
    # 1) Pick coherence
    coh = coherence_list[coherence_index]

    # 2) Call DataGeneration
    (Result_list,
     total_EL_RT,
     total_ER_RT,
     NO_decision,
     EL_firing,
     ER_firing) = DataGeneration(
        params=params,
        num_trial=num_trial,
        coherence=coh,
        threshold=threshold,
        trial_len=trial_len,
        BG_len=BG_len,
        input_amp=input_amp
    )

    # Extract gamma values
    gamma_ee = params.get('gamma_ee', 0.0)
    gamma_ei = params.get('gamma_ei', 0.0)
    gamma_ie = params.get('gamma_ie', 0.0)
    gamma_ii = params.get('gamma_ii', 0.0)

    # 3) Construct filename
    filename = (
        f"EE{gamma_ee}_EI{gamma_ei}_"
        f"IE{gamma_ie}_II{gamma_ii}_"
        f"coh{coh}_trialCount{num_trial}_trialN{trial_n}.pkl"
    )

    # 4) Save to pickle
    with open(filename, 'wb') as f:
        pickle.dump(Result_list, f)

    print(f"Saved Result_list to {filename}")
    return (Result_list, total_EL_RT, total_ER_RT, NO_decision, EL_firing, ER_firing)


def load_and_merge_data(file_paths, initial_offset=0, step=100):
    """
    Load multiple .pkl files (each containing a Result_list dict) and merge them into a single dict.
    The keys in each file will be shifted by 'step' for each file to avoid collisions.

    Parameters
    ----------
    file_paths : list of str
        Paths to the .pkl files for a single coherence condition.
    initial_offset : int
        Starting offset for the first file (default 0).
    step : int
        How many trial indices to shift for each subsequent file (default 100).

    Returns
    -------
    merged_dict : dict
        Merged dictionary of all loaded data.
    final_offset : int
        The offset after merging all files in file_paths.
    """
    merged_dict = {}
    key_offset = initial_offset

    for path in file_paths:
        with open(path, 'rb') as f:
            result_list = pickle.load(f)
            # Shift keys to avoid collisions
            shifted_dict = {k + key_offset: v for k, v in result_list.items()}
            merged_dict.update(shifted_dict)
        key_offset += step

    return merged_dict, key_offset


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
    coherence_list,
    results_polA,
    results_polB,
    labelA="Pol0",
    labelB="Pol2"
):
    """
    Plot performance vs. coherence and Reaction Time vs. coherence (with error bars)
    for two sets of results. Each set is a dict keyed by 'coh{value}', with 
    "Performance", "Avg_ER_Reaction_Time", and "Std_ER_Reaction_Time".
    """
    import numpy as np
    import matplotlib.pyplot as plt

    perfA, rtA, rtStdA = [], [], []
    perfB, rtB, rtStdB = [], [], []

    for c in coherence_list:
        coh_key = f"coh{c}"

        # polA
        if coh_key in results_polA:
            pA = results_polA[coh_key]["Performance"]
            rA = results_polA[coh_key]["Avg_ER_Reaction_Time"]
            rsA= results_polA[coh_key]["Std_ER_Reaction_Time"]
        else:
            pA, rA, rsA = None, None, None
        perfA.append(pA)
        rtA.append(rA)
        rtStdA.append(rsA)

        # polB
        if coh_key in results_polB:
            pB = results_polB[coh_key]["Performance"]
            rB = results_polB[coh_key]["Avg_ER_Reaction_Time"]
            rsB= results_polB[coh_key]["Std_ER_Reaction_Time"]
        else:
            pB, rB, rsB = None, None, None
        perfB.append(pB)
        rtB.append(rB)
        rtStdB.append(rsB)

    # Convert None -> np.nan for plotting
    perfA = np.array([p if p is not None else np.nan for p in perfA])
    rtA   = np.array([r if r is not None else np.nan for r in rtA])
    rtStdA= np.array([s if s is not None else np.nan for s in rtStdA])
    perfB = np.array([p if p is not None else np.nan for p in perfB])
    rtB   = np.array([r if r is not None else np.nan for r in rtB])
    rtStdB= np.array([s if s is not None else np.nan for s in rtStdB])

    # Psychometric curve
    plt.figure(figsize=(10, 5))
    plt.plot(coherence_list, perfA, 'o-', label=labelA)
    plt.plot(coherence_list, perfB, 'o-', label=labelB)
    plt.title("Psychometric Curve (Performance vs. Coherence)")
    plt.xlabel("Task Difficulty (%)")
    plt.ylabel("Performance (fraction of E_R decisions)")
    plt.grid(True)
    plt.xscale("log")
    plt.legend()
    plt.show()

    # Reaction time with error bars
    plt.figure(figsize=(10, 5))
    plt.errorbar(coherence_list, rtA, yerr=rtStdA, fmt='o-', capsize=5, label=labelA)
    plt.errorbar(coherence_list, rtB, yerr=rtStdB, fmt='o-', capsize=5, label=labelB)
    plt.title("Reaction Time vs. Coherence")
    plt.xlabel("Task Difficulty (%)")
    plt.ylabel("Reaction Time (ms)")
    plt.grid(True)
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
    Merge data from multiple .pkl files (grouped by coherence), then compute performance, 
    reaction times, fail rates, and non-regret metrics for each coherence.
    """
    all_merged_results = {}
    offset = 0
    for coh, paths in file_paths_dict.items():
        merged_dict, offset = load_and_merge_data(paths, initial_offset=offset, step=100)
        all_merged_results[f"coh{coh}"] = merged_dict

    perf_dict = {}
    non_regret_dict = {}

    for coh, merged_dict in all_merged_results.items():
        perf, rt_mean, rt_std = calculate_performance(merged_dict)
        fail_rate = calculate_fail_rate(merged_dict)

        perf_dict[coh] = {
            "Performance": perf,
            "Avg_ER_Reaction_Time": rt_mean,
            "Std_ER_Reaction_Time": rt_std,
            "Fail_Rate": fail_rate
        }

        # Non-Regret
        nr_data = compute_non_regret_performance(merged_dict, benchmark_array)
        non_regret_dict[coh] = nr_data

    return all_merged_results, perf_dict, non_regret_dict


def plot_energy_landscapes(
    Result_list,
    time_window,
    plot_full_scale=False,
    EXC=True,
    lim=40
):
    """
    Plot approximate energy landscapes U(x) from velocity data. 
    See docstring for details on usage.
    """

    plt.figure(figsize=(10, 6))

    # 1) Partial-window U(x)
    for window in time_window:
        start_time, end_time = window
        mean_time = (start_time + end_time) / 2 / 1000.0
        title_str = f"U(x): {start_time}ms to {end_time}ms (mean={mean_time:.2f}s)"

        Velocity_dict_exc = {}
        Velocity_dict_inh = {}

        for i, trial_data in Result_list.items():
            if trial_data is None:
                continue

            # Reaction time
            if trial_data[0] is not None:
                reaction_start_time = int(trial_data[0] * 10)
            elif trial_data[1] is not None:
                reaction_start_time = int(trial_data[1] * 10)
            else:
                continue

            start_idx = max(0, reaction_start_time + start_time * 10)
            end_idx   = max(0, reaction_start_time + end_time * 10)

            # Accumulate velocity for excitatory or inhibitory
            for j in range(start_idx, end_idx):
                # E_ difference array => trial_data[5]
                if j + 200 < len(trial_data[5]):
                    if trial_data[5][j] != -1 and trial_data[5][j + 200] != -1:
                        x_val = trial_data[5][j]
                        if -lim <= x_val <= lim:
                            velocity = trial_data[5][j + 200] - trial_data[5][j]
                            Velocity_dict_exc.setdefault(x_val, []).append(velocity)

                # I_ difference array => trial_data[10]
                if j + 200 < len(trial_data[10]):
                    if trial_data[10][j] != -1 and trial_data[10][j + 200] != -1:
                        x_inh = trial_data[10][j]
                        if -lim <= x_inh <= lim:
                            vel_inh = trial_data[10][j + 200] - trial_data[10][j]
                            Velocity_dict_inh.setdefault(x_inh, []).append(vel_inh)

        # Convert to DataFrame
        if EXC:
            df_data = [(k, v) for k, arr in Velocity_dict_exc.items() for v in arr]
        else:
            df_data = [(k, v) for k, arr in Velocity_dict_inh.items() for v in arr]

        df = pd.DataFrame(df_data, columns=['x', 'velocity'])
        df = df[(df['x'] >= -lim) & (df['x'] <= lim)]

        # Bin by 0.1
        interval = np.arange(-lim, lim + 0.1, 0.1)
        df['interval'] = pd.cut(df['x'], bins=interval)
        df_grouped = df.groupby('interval').mean(numeric_only=True)
        df_grouped['x'] = [cat.left + 0.05 for cat in df_grouped.index.categories]

        df_grouped['neg_vel'] = -df_grouped['velocity']
        df_grouped['U(x)'] = df_grouped['neg_vel'].cumsum()

        # Shift so U(0)=0
        zero_idx = (np.abs(df_grouped['x'] - 0)).argmin()
        offset_val = df_grouped['U(x)'].iloc[zero_idx]
        df_grouped['U(x)'] -= offset_val

        x_arr = df_grouped['x'].values
        y_arr = df_grouped['U(x)'].values.astype(float)
        plt.plot(x_arr, y_arr, label=title_str)

    # 2) Full-scale U(x)
    if plot_full_scale:
        Velocity_exc_full = {}
        Velocity_inh_full = {}

        for i, trial_data in Result_list.items():
            if trial_data is None:
                continue

            for j in range(len(trial_data[5]) - 200):
                if trial_data[5][j] != -1 and trial_data[5][j + 200] != -1:
                    xx = trial_data[5][j]
                    if -lim <= xx <= lim:
                        velocity = trial_data[5][j + 200] - trial_data[5][j]
                        Velocity_exc_full.setdefault(xx, []).append(velocity)

            for j in range(len(trial_data[10]) - 200):
                if trial_data[10][j] != -1 and trial_data[10][j + 200] != -1:
                    xx_inh = trial_data[10][j]
                    if -lim <= xx_inh <= lim:
                        v_inh = trial_data[10][j + 200] - trial_data[10][j]
                        Velocity_inh_full.setdefault(xx_inh, []).append(v_inh)

        if EXC:
            df_data_full = [(k, v) for k, arr in Velocity_exc_full.items() for v in arr]
        else:
            df_data_full = [(k, v) for k, arr in Velocity_inh_full.items() for v in arr]

        
        df_full = pd.DataFrame(df_data_full, columns=['x', 'velocity'])
        df_full = df_full[(df_full['x'] >= -lim) & (df_full['x'] <= lim)]

        interval_full = np.arange(-lim, lim + 0.1, 0.1)
        df_full['interval'] = pd.cut(df_full['x'], bins=interval_full)
        df_full_grouped = df_full.groupby('interval').mean(numeric_only=True)
        df_full_grouped['x'] = [cat.left + 0.05 for cat in df_full_grouped.index.categories]

        df_full_grouped['neg_vel'] = -df_full_grouped['velocity']
        df_full_grouped['U(x)'] = df_full_grouped['neg_vel'].cumsum()

        zero_idx_full = (np.abs(df_full_grouped['x'] - 0)).argmin()
        offset_full = df_full_grouped['U(x)'].iloc[zero_idx_full]
        df_full_grouped['U(x)'] -= offset_full

        x_f = df_full_grouped['x'].values
        y_f = df_full_grouped['U(x)'].values.astype(float)
        plt.plot(x_f, y_f, label='Full scale U(x)', linestyle='--')

    plt.xlabel('x')
    plt.ylabel('U(x)')
    plt.title('Energy Landscapes')
    plt.legend()
    plt.show()
