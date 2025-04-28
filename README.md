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
  * Running simulations and visualizing results (original `GUI.py`)
  * Advanced analysis of performance, reaction times, and “energy landscapes” (`AdvancedPlot_GUI.py`)
* **Real-time** or **post-hoc** visualization of network dynamics and decision outcomes

---

## Repository Structure

This repository contains the following main files and folders:

1. **`AdvancedPlot_GUI.py`**
   * A standalone PyQt5 application (also compiled as `PLOTApp.app`) for advanced plotting and data analysis.
   * **Features**:
     * Mode selection for either loading and plotting existing data (**energy landscapes** / **psychometric curves** / **RT**) or running new simulations to generate fresh data.
     * Time-window-based “energy landscape” plotting.
     * Performance & reaction-time analysis over multiple coherence levels.
     * Option to save simulation results and merged data in `.pkl` files.
2. **`GUI.py`**
   * The original PyQt5 interface (compiled as `GUIApp.app`) to **configure and run** the spiking network in real time.
   * **Features**:
     * Adjustable network parameters (e.g., `n_E`, `n_I`, `gamma_ee`, etc.).
     * Toggle top-down signals, set threshold, background/stimulus duration, coherence, and more.
     * Live visualization of spike rasters/firing rates (requires Brian2 plots).
3. **`Plot_func.py`**
   * Utility functions to facilitate plotting within the original `GUI.py` (e.g., raster plots, PSTHs).
   * **Note**: The advanced plotting routines (energy landscape, psychometric curves) are instead in `Functions.py` and are invoked by `AdvancedPlot_GUI.py`.
4. **`Functions.py`**
   * Core Python functions for **data generation** and **analysis**, including:
     * `DataGeneration`: Runs multiple trials of the Brian2 network, logs decision outcomes (e.g., which population won, reaction times).
     * `plot_energy_landscapes_on_figure`: Visualizing energy landscapes over specific time windows.
     * `plot_perf_rt_on_figure`: Plotting performance (accuracy) and reaction times across coherence levels.
     * `load_and_merge_data_with_metadata`: Loading `.pkl` files and merging them (useful for aggregated analyses).
5. **`Network_set.py`**
   * Builds and configures the Brian2 network model:
     * Defines excitatory/inhibitory populations, external input, synaptic weight matrices, and monitors.
     * Core equations for AMPA, NMDA, GABA synapses, and how top-down signals are injected.
6. **`data/`**
   * A folder to store or load simulation results (`.pkl`) that can be plotted in `AdvancedPlot_GUI.py`.
   * Place any pre-generated data files here for easy loading (though users can pick any directory from the file dialog).

---

## Installation and Dependencies

* **Python 3.7+** recommended
* **[Brian2](https://brian2.readthedocs.io/)** for the spiking neural network simulation
* **PyQt5** for the GUI
* **Matplotlib**, **NumPy**, **pickle**, etc.

You can install the necessary packages with:

```bash
pip install brian2 pyqt5 matplotlib numpy
```

---

## How to Use the GUIs

### 1. Running the Original GUI (`GUI.py`)

This is the classic interface to **configure network parameters** and run spiking simulations with on-the-fly plots.

* **Command**:

  ```bash
  python GUI.py
  ```

  (Or double-click `GUIApp.app` on supported systems.)
* **Key Steps**:

  1. Set network sizes (`n_E`, `n_I`, `n_NSE`), connection strengths (`gamma_ee`, `gamma_ei`, `gamma_ie`, `gamma_ii`).
  2. Choose top-down usage (`top_down` checkbox), and if checked, set `S`, `R`.
  3. Configure simulation parameters: `num_trial`, `threshold`, `trial_len`, `BG_len`, `input_amp`, `coherence`.
  4. Start the simulation to see live raster plots/firing rates (depending on your system’s Brian2 plotting backend).

### 2. Running the Advanced Plotting GUI (`AdvancedPlot_GUI.py`)

This interface provides **more comprehensive** post-hoc analysis and plotting capabilities, as well as the option to run new trials in a more streamlined manner.

* **Command**:

  ```bash
  python AdvancedPlot_GUI.py
  ```

  (Or run `PLOTApp.app` if you have the compiled version.)
* **Mode Selection**: Upon start, a dialog will appear:

  1. **Use existed data to plot**
     * Select one or multiple `.pkl` files containing saved simulation results.
     * **Options**:
       * **Plot full-scale “energy landscapes”** across all trials.
       * **Plot time-based “energy landscapes”** (you’ll be prompted for time-window ranges).
       * **Plot performance & RT** (requires `.pkl` files for the six standard coherence levels: `[0, 3.2, 6.4, 12.8, 25.6, 51.2]`).
  2. **Run new trial (energy)**
     * Choose top-down parameters (`S`, `R`), network sizes, connection strengths, etc.
     * Run a certain number of trials at **coherence=0** to generate energy-landscape data.
     * Automatically plot the energy landscape (full-scale, plus optional time windows) and optionally save `.pkl`.
  3. **Run new trial (performance/RT)**
     * Similar parameter setup, but runs trials across **six coherence values**.
     * Plots aggregated performance (accuracy) and reaction-time curves, and optionally saves each coherence’s data to `.pkl`.

#### Typical Workflow in `AdvancedPlot_GUI.py`

1. **Open**`AdvancedPlot_GUI.py` → a **Mode Selection** window appears.
2. If “Use existed data to plot”:
   * A file dialog prompts you to select `.pkl` files.
   * Choose how to plot (energy landscape full-scale/time-based or performance & RT).
   * Plot results in a new window (with interactive matplotlib).
3. If “Run new trial (energy)” or “Run new trial (performance/RT)”:
   * Enter **top-down** parameters in the first window (whether enabled, `S`, `R`).
   * Enter **network** and **simulation** parameters in the second window (e.g., `n_E`, `n_I`, `gamma_ee`, `threshold`, etc.).
   * Proceed to **execution** window, watch the progress bar, then see the results in a pop-up plot.
   * Optionally **save** your data in `.pkl` format for later analysis.

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

* **Pickle Files (`.pkl`)**:
  * The GUIs or scripts can save simulation outcomes (metadata + result\_list) into `.pkl` files.
  * The advanced GUI can load multiple `.pkl` files, merge them, and produce combined plots.
* **Metadata** typically includes:
  * Connection strengths (`gamma_ee`, `gamma_ei`, etc.), network sizes, top-down usage.
  * Coherence value, number of trials, thresholds, etc.
  * Timestamps for record-keeping.
* **Result data** (`result_list`) can store:
  * Decision outcome (which population won or if none reached threshold).
  * Reaction time for each trial.
  * Additional data for analyzing “energy landscapes”.

---

## Example: Generating and Plotting Performance Curves

1. **Run New Trial (performance/RT)** in `AdvancedPlot_GUI.py`:
   * Select “Run new trial (performance/RT)” → set top-down parameters (optional).
   * Set network parameters and the number of trials.
   * Hit “Run” → the app sweeps through `[0, 3.2, 6.4, 12.8, 25.6, 51.2]` coherence values.
   * Automatically plots **performance** (accuracy) vs. **coherence** and **reaction times**.
   * Optionally save each coherence’s data in separate `.pkl`.
2. **Post-hoc**:
   * Another user (or you, in a new session) can select “Use existed data to plot” → load the saved `.pkl` files → instantly generate the same performance or energy-landscape plots again.

---

## Notes on the `.app` Files

* Two compiled app bundles are provided (for certain platforms):
  * **`GUIApp.app`** → a packaged version of `GUI.py`.
  * **`PLOTApp.app`** → a packaged version of `AdvancedPlot_GUI.py`.
* If your system supports it, you can simply double-click the `.app` to launch the respective GUI. Otherwise, the `.py` files can be run directly from the terminal.

---

## References

For theoretical background on balanced and selective inhibition, attractor neural networks, and top-down modulation, please see:

1. **Wang et al. (2013)**: [*Top-Down Modulation on Perceptual Decision...*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0062379)
2. **Liu et al. (2021)**: [*Choose carefully, act quickly...*](https://www.biorxiv.org/content/10.1101/2021.10.05.463257v2)
