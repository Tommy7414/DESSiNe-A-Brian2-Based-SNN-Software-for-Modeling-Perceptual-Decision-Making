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
  * Running simulations and visualizing results (`DESSiNe_DM.py`)
  * Advanced analysis of performance, reaction times, and “energy landscapes” (`DESSiNe_EL.py`)
* **Real-time** or **post-hoc** visualization of network dynamics and decision outcomes
* **Precompiled macOS Applications** (`DESSiNe_DMApp`, `DESSiNe_ELApp`) that allow users to run the full system without needing to install Python or any packages.

---

## Repository Structure

The codebase is organized under a `src/` package layout:

### `src/dessine/apps/` — GUI entry points
- **`DESSiNe_DM.py`**  
  A PyQt5 GUI for **single-session, real-time inspection** of the spiking decision network.  
  It is intended for interactive exploration (live traces and rasters) and **does not save** `.pkl` result files.

- **`DESSiNe_EL.py`**  
  A PyQt5 GUI for **batch runs and plotting**. It supports:
  (i) loading existing `.pkl` files to plot energy landscapes (full-scale or RT-aligned time-windowed) and Performance/RT;  
  (ii) running new batches to generate `.pkl` files; and  
  (iii) comparing parameter groups via metadata-based grouping/merging.

### `src/dessine/core/` — simulation, analysis, and plotting core
- **`network_set.py`**  
  Brian2 network construction: neuron groups, synapses, inputs, and monitors.  
  Includes optional top-down control (BSI parameters `S`, `R`, and `n_Ctr` if enabled).

- **`functions.py`**  
  Core simulation + analysis utilities used by the GUIs, including:
  - multi-trial simulation (e.g., `DataGeneration`)
  - data loading/merging by embedded metadata
  - energy landscape plotting (full-scale and time-windowed)
  - Performance/RT plotting

- **`plot_func.py`**  
  Lightweight plotting utilities primarily used for the DM live display (rasters/rates/traces).

- **`data_generator.py`**  
  A headless (no-GUI) runner for scripted batch generation of GUI-compatible `.pkl` files via `data_generator(...)`.  
  Output files follow the same metadata schema as the EL GUI and can be loaded/merged/visualized in `DESSiNe_EL.py`.

### `src/dessine/assets/` — GUI resources
- icons and images used by the GUIs and PyInstaller builds.

### `data/` — example result files (Git LFS)
 - This folder contains example `.pkl` files tracked via **Git LFS**.  
 - If you download the repository as a ZIP from GitHub, these `.pkl` files may download as **LFS pointer files** and cannot be loaded directly.  
 Please follow the “Download example data (Git LFS)” instructions below.
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

### Option A (recommended): Download prebuilt apps from GitHub Releases
1. Go to the GitHub **Releases** page of this repository.
2. Download the packaged app for your OS:
   - macOS: `DESSiNe_DMApp-<version>-macos.zip`, `DESSiNe_ELApp-<version>-macos.zip`
   - Windows: `DESSiNe_DMApp-<version>-windows-x64.zip`, `DESSiNe_ELApp-<version>-windows-x64.zip`
   - Linux: `DESSiNe_DMApp-<version>-linux-x64.zip`, `DESSiNe_ELApp-<version>-linux-x64.zip`
3. Unzip and launch:
   - macOS: right-click the `.app` → **Open** (Gatekeeper may block first launch)
   - Windows/Linux: run the extracted executable

> The DM app is for real-time inspection (single trial).  
> The EL app is for batch runs and plotting (multi-trial summaries, with optional `.pkl` saving/loading).

---

### Option B: Run from source (developers)
Create a Python environment and install dependencies, then run the entry scripts under `src/dessine/apps/`.

(Example)
```bash
pip install -r requirements.txt
pip install -e .
python src/dessine/apps/DESSiNe_DM.py
python src/dessine/apps/DESSiNe_EL.py
````

---

### Download example data (Git LFS)

This repository includes example `.pkl` files under `data/`, tracked with **Git LFS**.
If you use GitHub “Download ZIP”, the `.pkl` files may become **LFS pointer files** and will fail to load.

To download the real `.pkl` files:

```bash
brew install git-lfs          # macOS (Homebrew)
git lfs install
git clone https://github.com/Tommy7414/DESSiNe-A-Brian2-Based-SNN-Software-for-Modeling-Perceptual-Decision-Making.git
cd DESSiNe-A-Brian2-Based-SNN-Software-for-Modeling-Perceptual-Decision-Making
git lfs pull
```

You can then load the `.pkl` files in `DESSiNe_EL` via “Load data and plot”.

---

## 1) Real-time single-trial inspection (DM)

**DM** is designed for **interactive, single-session visualization**.
It streams live signals (rates/rasters/traces) during one trial and is best for quickly understanding how parameter changes affect the decision process.
**DM does not save `.pkl` files.**

Run:

```bash
python src/dessine/apps/DESSiNe_DM.py
```

Typical steps:

1. Set population sizes (`n_E`, `n_I`, `n_NSE`), selectivity parameters (`gamma_ee`, `gamma_ei`, `gamma_ie`, `gamma_ii`), and optional top-down control (`S`, `R`).
2. Configure trial settings (e.g., background/stimulus duration, `input_amp`, `evidence_strength (%)`, and decision `threshold`).
3. Start running and inspect the real-time dynamics.

---

## 2) Batch runs + plotting (EL)

**EL** is designed for **multi-trial analysis and visualization**.
It can (a) **run batches** to generate results and optionally save `.pkl` files, and (b) **load existing `.pkl` files** (including files you generated locally) to plot energy landscapes and Performance/RT summaries.

Run:

```bash
python src/dessine/apps/DESSiNe_EL.py
```

EL provides three common workflows:

### (a) Load data and plot

* Load one or more `.pkl` files.
* Files are grouped/merged by **embedded metadata** (filenames do not matter).
* Supported plots:

  * **Energy landscape (full-scale)**: can overlay up to **5** parameter groups.
  * **Energy landscape (time-windowed / RT-aligned)**: requires a **single** consistent parameter group.
  * **Performance/RT**: requires a complete set of **six evidence strengths**: `0, 3.2, 6.4, 12.8, 25.6, 51.2 (%)`.

### (b) Run new trials (energy landscape)

* Choose a single evidence strength and run **N trials**.
* The energy landscape estimate becomes more stable with larger N (more decision trails).
* Optionally save the aggregated results as **one** `.pkl`.

### (c) Run new trials (Performance/RT)

* Run batches across the six evidence strengths (`0, 3.2, 6.4, 12.8, 25.6, 51.2 (%)`).
* Optionally save **six** `.pkl` files (one per evidence strength), which can later be loaded and compared.

> Note: All `.pkl` saving/loading is handled in EL (or via `src/dessine/core/data_generator.py` for headless batch generation).

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
   * **Evidence Strength**: Stimulus bias that drives one excitatory population more than the other.
   * **Energy Landscapes**: Interpreting population states as stable attractors that can be visualized over time or aggregated across trials.

---

## Saving & Loading Results

**Pickle files (`.pkl`).** Results are saved **by `DESSiNe_EL` or by scripts in `Functions.py`** (e.g., `run_experiment`, `save_result_with_metadata`). `DESSiNe_DM` is live-only and does **not** save files. The EL app can load **multiple** `.pkl` files, group them by **internal metadata**, merge compatible trials automatically, and make combined plots.

**Metadata (minimal, behavior-relevant).**
Saved files always include:

```
gamma_ee, gamma_ei, gamma_ie, gamma_ii,
evidence strength (%), num_trial, threshold, trial_len, BG_len, input_amp,
timestamp, top_down  [and S, R if top_down=True]
```

Neuron counts are **not** stored/used for grouping.

**Result data (`result_list`).**
Each trial stores the decision flags and RT plus arrays used for energy-landscape analysis, in the format:
`[ER_flag, EL_flag, ER_RT, EL_RT, times, (ER-EL), rateER, rateEL, rateIL, rateIR]` (or `None` if no decision).

---

## Example: Generating and Plotting Performance Curves (revised)

1. **Run New Trial (performance/RT)** in `DESSiNe_EL.py`
   Choose **Run new trial (performance/RT)** → set (optional) top-down `S, R` and network/simulation parameters → click **Run**. The app sweeps the six evidence strengths **\[0, 3.2, 6.4, 12.8, 25.6, 51.2] (%)**, then plots **Performance vs. Evidence Strength** and **Reaction Time**. You can optionally save **six** `.pkl` files (one per evidence strength), each with the metadata above.
2. **Post-hoc plotting**
   Later, choose **Use existing data to plot** in EL, select your `.pkl` files, and regenerate Performance/RT or Energy-landscape figures. For Performance/RT, a parameter group is plotted **only if** all six evidence strengths files are present.

---

## Notes on Packaged Apps (revised)

If you distribute compiled apps, use consistent names such as:

* **`DESSiNe_DM.app`** – packaged live visualization (`DESSiNe_DM.py`)
* **`DESSiNe_EL.app`** – packaged analysis/batch runner (`DESSiNe_EL.py`)

These bundles embed Python and dependencies (no separate install). Double-click to launch. For platforms beyond your build target, re-package with PyInstaller on that OS.

---

### Optional code sanity check

In `DESSiNe_DM.py`, the “Evidence Strengths(%)” input should be applied as `amp * (1 ± evidence_strength/100)` (to match EL and your docs). If you haven’t already, make sure DM divides by **100** internally.

---

## References

For theoretical background on balanced and selective inhibition, attractor neural networks, and top-down modulation, please see:

1. **Wang et al. (2013)**: [*Top-Down Modulation on Perceptual Decision...*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0062379)
2. **Liu et al. (2021)**: [*Choose carefully, act quickly...*](https://www.biorxiv.org/content/10.1101/2021.10.05.463257v2)
