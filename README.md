# A Brian2-based Spiking Neural Network Model for Decision-Making

This repository implements a spiking neural network model of **two-choice decision-making**, exploring how **balanced inhibition** and **selective inhibition** can shape decision dynamics. The model follows ideas from:

> **[1]** Wang, C.-T., Lee, C.-T., Wang, X.-J., & Lo, C.-C. (2013). *Top-Down Modulation on Perceptual Decision with Balanced Inhibition through Feedforward and Feedback Inhibitory Neurons*. PLOS ONE, 8(4): e62379. [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0062379)

> **[2]** Liu, B., Lo, C.-C., & Wu, K.-A. (2021). *Choose carefully, act quickly: Efficient decision making with selective inhibition in attractor neural networks*. bioRxiv. [Link](https://www.biorxiv.org/content/10.1101/2021.10.05.463257v2)

---

## Overview

A **biologically plausible** spiking neural network model for decision-making tasks, implemented using [Brian2](https://brian2.readthedocs.io/). This model features:

* **Adjustable selective inhibitory connections**
* **Configurable top-down control signals**
* **Two separate PyQt5-based GUI tools** for:
  * Running simulations and visualizing results (original `PDMSNN_DM.py`)
  * Advanced analysis of performance, reaction times, and “energy landscapes” (`PDMSNN_EL.py`)
* **Real-time** or **post-hoc** visualization of network dynamics and decision outcomes
* **Precompiled macOS Applications** (`PDMSNN_DMApp`, `PDMSNN_ELApp`) that allow users to run the full system without needing to install Python or any packages.

---

## Repository Structure

This repository contains the following main files and folders:

1. **`DESSiNe_EL.py`**
   A standalone PyQt5 application for **running batch simulations** and **plotting**. It offers three workflows: (i) load existing `.pkl` files and plot **energy landscapes** (full-scale or time-windowed) or **Performance/RT**; (ii) run **energy mode** (N trials at a single coherence) and optionally save **one** `.pkl`; (iii) run **Performance/RT mode** (the six coherences: **0, 3.2, 6.4, 12.8, 25.6, 51.2** %) and optionally save **six** `.pkl` files (one per coherence). Plots can overlay up to **five** parameter groups (time-windowed energy supports one group only).
2. **`DESSiNe_DM.py`**
   A PyQt5 interface to **configure and run** the spiking network **live** with real-time visualization (E\_L/E\_R, inhibitory, NSE). You can set population sizes, γ-parameters, top-down (S,R), coherence (%), background/stimulus durations, and decision threshold, then observe the dynamics in a single session. **Note:** `DESSiNe_DM.py` is **live inspection only** and does **not** write result files.
3. **`Plot_func.py`**
   Small plotting utilities used by **`DESSiNe_DM.py`** for in-session visualization (rasters, rates, etc.).
   **Note:** The advanced plotting (energy landscapes, Performance/RT) lives in **`Functions.py`** and is invoked by **`DESSiNe_EL.py`**.
4. **`Functions.py`**
   Core functions for **data generation** and **analysis** used by `DESSiNe_EL.py`, including:

   * `DataGeneration`: multi-trial simulation; records choice flags and RTs.
   * `plot_energy_landscapes_on_figure`: full-scale or time-windowed quasi-potential plots.
   * `plot_perf_rt_on_figure`: psychometric (performance) and chronometric (RT) curves.
   * `load_and_merge_data_with_metadata`: load `.pkl` files and **merge** trials that share the same **behavior-relevant metadata**.
     Saved files have the format `{"metadata": meta, "result_list": ...}` with **minimal metadata** (γ’s, coherence (%), `num_trial`, `threshold`, `trial_len`, `BG_len`, `input_amp`, `timestamp`, and always `top_down`; **add `S` and `R` only when `top_down=True`**).
5. **`Network_set.py`**
   Network construction for Brian2: defines neuron groups, synapses, inputs, and monitors; supports optional top-down control (S,R).
6. **`Data_generator.py`**
   A headless (no-GUI) runner for batch simulations and saving results. Provides a `data_generator(...)` function and a top “user settings” block. Two workflows: (i) **energy mode** — run N trials at a single coherence and save **one** `.pkl`; (ii) **Performance/RT mode** — sweep **\[0, 3.2, 6.4, 12.8, 25.6, 51.2]%** and save **six** `.pkl` files (one per coherence). Filenames strictly follow the GUI convention:
   `EE{γEE}_EI{γEI}_IE{γIE}_II{γII}[_S{S}_R{R}]_coh{coh}_trialCount{N}_{timestamp}.pkl`.
   Each file stores `{"metadata": …, "result_list": …}` and is fully compatible with `DESSiNe_EL.py` for loading/merging/plotting.
7. **`data/`**
   Default folder for saved `.pkl` outputs (optional—users can save/load anywhere via the
8. **`dist/`**

   * This folder contains precompiled `.app` versions for macOS (tested on Apple M2 chips):
     * `PDMSNN_DMApp`: A packaged application for configuring and running spiking network simulations without requiring a Python environment.
     * `PDMSNN_ELApp`: A packaged application for advanced plotting and analysis.
   * **Usage**:
     * Simply double-click the `.app` file to launch.
     * No need to install Python or any libraries manually.
     * The workflow inside the app mirrors the Python scripts:
       * For `PDMSNN_DMApp`: Set parameters → run simulation → view results in real-time.
       * For `PDMSNN_ELApp`: Choose between plotting existing data or running new trials → analyze energy landscapes or performance curves.

---

## Installation and Dependencies

* **Python 3.7+** recommended
* **[Brian2](https://brian2.readthedocs.io/)** for the spiking neural network simulation
* **PyQt5** for the GUI
* **Matplotlib**, **NumPy**, **pickle**, etc.

You can install the necessary packages with:

```bash
pip install brian2 pyqt5 matplotlib numpy pandas
```

---

## How to Use the GUIs

### 1) Live inspection (`DESSiNe_DM.py`)

This is the interactive GUI to **configure the network** and watch activity in real time. It **does not save** result files.

**Run**

```bash
python DESSiNe_DM.py
```

(or launch the packaged app/exe if you built it with PyInstaller)

**Steps**

1. In the main window, set population sizes (`n_E`, `n_I`, `n_NSE`), γ-parameters (`gamma_ee`, `gamma_ei`, `gamma_ie`, `gamma_ii`), and optionally enable **Top-down** (`S`, `R`).
2. Open **Running Trial** and set `BG_len`, `trial_len`, `input_amp`, **`coherence (%)`**, and `threshold`.
3. Click **Start Running** to stream E\_L/E\_R (plus inhibitory/NSE) with background → stimulus → short post-decision.
4. Use **Reset Network** to return to the stored default and try new settings.

> Need saved data for later plotting? Use **`DESSiNe_EL.py`**.

---

### 2) Advanced plotting & batch runs (`DESSiNe_EL.py`)

This GUI runs new trials **and** plots existing data (energy landscapes, performance/RT). It handles saving/merging `.pkl` files with metadata.

**Run**

```bash
python DESSiNe_EL.py
```

(or launch the packaged app/exe)

**Mode selection**

1. **Use existing data to plot**
   Pick one or more `.pkl` files. The app groups by **internal metadata** (filenames don’t matter), merges compatible files, and lets you:

   * Plot **energy landscapes (full-scale)**; overlay up to **5** parameter groups.
   * Plot **energy landscapes (time-windowed)**; requires a **single** parameter set (no overlay).
   * Plot **Performance/RT**; requires a complete set of **six coherences**: `0, 3.2, 6.4, 12.8, 25.6, 51.2 (%)`.
2. **Run new trial (energy)**
   Set top-down/network/simulation parameters, choose a **single coherence**, run **N** trials, view the full-scale landscape (optionally time-windowed), and **optionally save one `.pkl`**.
3. **Run new trial (performance/RT)**
   Same setup, but the app iterates the **six coherences**, shows performance and RT curves, and **optionally saves six `.pkl` files** (one per coherence).

**Typical workflow**

* Start `DESSiNe_EL.py` → choose a mode.
* For plotting existing data: select files → pick plot type → view/export figures.
* For new runs: enter **Top-down** (Window 1) → enter **network & simulation** (Window 2) → execute (Window 3) with progress and plots → optionally save `.pkl` files for later analysis.

**Reminder (to avoid mistakes)**
Performance/RT plots only appear when a parameter group has **all six** coherence files present; time-windowed energy accepts **one** parameter set; and DM never writes `.pkl`—all saved data comes from EL.

---

## Model Architecture

1. **Neuron Model**
   * Spiking neuron implementation with AMPA, NMDA, and GABA synaptic currents.
   * Each neuron’s membrane potential follows a set of differential equations defined in `Network_set.py`.
2. **Network Structure**
   * Two excitatory populations (E\_L, E\_R) competing to make a **decision**.
   * Inhibitory populations (I\_L, I\_R) providing balanced or selective inhibition.
   * Non-specific excitatory neurons (NSE) contributing background drive.
   * Optional top-down input to modulate excitatory/inhibitory populations differently.
3. **Key Features**
   * **Selective Inhibition**: Weighted suppression of the “losing” population to sharpen decision signals.
   * **Top-Down**: Additional input for biasing or speeding up decisions.
   * **Coherence**: Stimulus bias that drives one excitatory population more than the other.
   * **Energy Landscapes**: Interpreting population states as stable attractors that can be visualized over time or aggregated across trials.

---

## Saving & Loading Results

**Pickle files (`.pkl`).** Results are saved **by `DESSiNe_EL` or by scripts in `Functions.py`** (e.g., `run_experiment`, `save_result_with_metadata`). `DESSiNe_DM` is live-only and does **not** save files. The EL app can load **multiple** `.pkl` files, group them by **internal metadata**, merge compatible trials automatically, and make combined plots.

**Metadata (minimal, behavior-relevant).**
Saved files always include:

```
gamma_ee, gamma_ei, gamma_ie, gamma_ii,
coherence (%), num_trial, threshold, trial_len, BG_len, input_amp,
timestamp, top_down  [and S, R if top_down=True]
```

Neuron counts are **not** stored/used for grouping.

**Result data (`result_list`).**
Each trial stores the decision flags and RT plus arrays used for energy-landscape analysis, in the format:
`[ER_flag, EL_flag, ER_RT, EL_RT, times, (ER-EL), rateER, rateEL, rateIL, rateIR]` (or `None` if no decision).

---

## Example: Generating and Plotting Performance Curves (revised)

1. **Run New Trial (performance/RT)** in `DESSiNe_EL.py`
   Choose **Run new trial (performance/RT)** → set (optional) top-down `S, R` and network/simulation parameters → click **Run**. The app sweeps the six coherences **\[0, 3.2, 6.4, 12.8, 25.6, 51.2] (%)**, then plots **Performance vs. Coherence** and **Reaction Time**. You can optionally save **six** `.pkl` files (one per coherence), each with the metadata above.
2. **Post-hoc plotting**
   Later, choose **Use existing data to plot** in EL, select your `.pkl` files, and regenerate Performance/RT or Energy-landscape figures. For Performance/RT, a parameter group is plotted **only if** all six coherence files are present.

---

## Notes on Packaged Apps (revised)

If you distribute compiled apps, use consistent names such as:

* **`DESSiNe_DM.app`** – packaged live visualization (`DESSiNe_DM.py`)
* **`DESSiNe_EL.app`** – packaged analysis/batch runner (`DESSiNe_EL.py`)

These bundles embed Python and dependencies (no separate install). Double-click to launch. For platforms beyond your build target, re-package with PyInstaller on that OS.

---

### Optional code sanity check

In `DESSiNe_DM.py`, the “Coherence(%)” input should be applied as `amp * (1 ± coherence/100)` (to match EL and your docs). If you haven’t already, make sure DM divides by **100** internally.

---

## References

For theoretical background on balanced and selective inhibition, attractor neural networks, and top-down modulation, please see:

1. **Wang et al. (2013)**: [*Top-Down Modulation on Perceptual Decision...*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0062379)
2. **Liu et al. (2021)**: [*Choose carefully, act quickly...*](https://www.biorxiv.org/content/10.1101/2021.10.05.463257v2)
