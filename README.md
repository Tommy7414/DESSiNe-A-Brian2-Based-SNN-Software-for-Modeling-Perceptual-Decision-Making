# E-I-connection-spiking-neural-neuron-model
A Brian2-based spiking neural network model for perceptual decision-making, featuring selective inhibition control and top-down modulation. Includes GUI tools for parameter tuning and real-time visualization of network dynamics and decision outcomes.

This repository implements a spiking neural network model of **two-choice perceptual decision-making**, exploring how **balanced inhibition** and **selective inhibition** can shape decision dynamics. The model follows ideas from:

> **[1]** Wang, C.-T., Lee, C.-T., Wang, X.-J., & Lo, C.-C. (2013). *Top-Down Modulation on Perceptual Decision with Balanced Inhibition through Feedforward and Feedback Inhibitory Neurons*. PLOS ONE, 8(4): e62379. [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0062379)

> **[2]** Liu, B., Lo, C.-C., & Wu, K.-A. (2021). *Choose carefully, act quickly: Efficient decision making with selective inhibition in attractor neural networks*. bioRxiv. [Link](https://www.biorxiv.org/content/10.1101/2021.10.05.463257v2)

## Overview

A biologically plausible spiking neural network model for decision-making tasks, implemented using Brian2. This model features:

- Adjustable selective inhibitory connections
- Configurable top-down control signals
- User-friendly GUI for parameter tuning
- Real-time visualization of network dynamics
- Analysis tools for psychometric curves and energy landscapes

## Repository Overview

1. **`Network_set.py`**
   * Defines neuron equations, synapses, and monitors using Brian2.
   * **Adjustable parameters** include:
     * `n_E`, `n_I`, `n_NSE`: Population sizes for excitatory, inhibitory, and non-specific excitatory neurons.
     * `gamma_ee`, `gamma_ei`, `gamma_ie`, `gamma_ii`: Connection scaling factors.
     * `top_down`, `S`, `R`: Whether top-down is enabled, and how strongly it influences E/I neurons.
2. **`plot_func.py`**
   * Helper script for plotting network results (raster plots, firing rates, etc.) within the Brian2 GUI.
3. **`GUI.ipynb`**
   * A PyQt5-based interface to **configure** and **run** the model.
   * **Key adjustable parameters** here include:
     * `threshold`: Firing-rate threshold for decision.
     * `coherence`: Fractional difference in input to left vs. right excitatory populations.
     * `BG_len`, `trial_len`, `input_amp`: Durations of background/stimulus phases, and amplitude of sensory input.
     * All parameters stored in a `params` dictionary (e.g., `gamma_ei`, `n_Ctr`, etc.).
4. **`Functions.py`**
   * Contains core Python functions for:
     * **`DataGeneration`**: Runs multiple trials of the Brian2 network, logs results, and returns them in a dictionary.
     * **`run_experiment`**: Wrapper that calls `DataGeneration` and saves output to `.pkl`.
     * **Analysis functions** (e.g., `calculate_performance`, `plot_performance_and_rt`, etc.) for psychometric curves, reaction times, and “non-regret” metrics.
   * **Parameter highlights**:
     * In `DataGeneration`: `num_trial`, `coherence`, `threshold`, `trial_len`, `BG_len`, `input_amp` are all user-configurable.
5. **`Performance_EnergyLandscape.ipynb`**
   * A Jupyter Notebook demonstrating how to **generate** data (via `run_experiment` in `Functions.py`) and **analyze** results:
     * Plot psychometric curves (performance vs. coherence).
     * Plot “energy landscapes” by calling `plot_energy_landscapes`.
     * Perform “non-regret” analyses and compare different parameter settings or top-down conditions.

## Model Architecture

### Neuron Model

- Spiking neuron implementation using Brian2
- AMPA, NMDA, and GABA synaptic dynamics
- Detailed membrane potential equations

### Network Structure

- Excitatory populations (E\_L, E\_R)

- Inhibitory populations (I\_L, I\_R)

- Non-specific excitatory neurons (NSE)

- Top-down control units (optional)

### Key Features

- Adjustable selective inhibition ratios
- Configurable top-down modulation
- Stimulus coherence control
- Background noise integration

## Usage Summary

1. **Adjust Parameters**
   * **In `GUI.ipynb`** (for interactive PyQt5 runs) or **in the `params` dictionary** when calling `run_experiment` / `DataGeneration` (for script-based runs).
   * Typical parameters include:
     * `n_E`, `n_I`, `n_NSE`, `gamma_ee`, `gamma_ei`, `gamma_ie`, `gamma_ii`, `top_down`, `S`, `R` (network structure).
     * `threshold`, `BG_len`, `trial_len`, `input_amp`, `coherence` (simulation logic).
2. **Run Simulations**
   * **`GUI.ipynb`**: Launch PyQt5 GUI → adjust parameters → run/plot in real-time.
   * **`Performance_EnergyLandscape.ipynb`**: Import and call `run_experiment(params, ...)` to create `.pkl` data files, then load and analyze them in separate code blocks.
3. **Analyze and Plot**
   * Use **`plot_performance_and_rt`** to generate psychometric curves and reaction-time charts.
   * Use **`plot_energy_landscapes`** to investigate how network states evolve over time windows.
   * Use **`compute_non_regret_performance`** and **`plot_non_regret`** to further assess decision dynamics under different benchmarks.
4. **References**
   * Please see the original publications ([Wang et al. 2013](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0062379) and [Liu et al. 2021](https://www.biorxiv.org/content/10.1101/2021.10.05.463257v2)) for theoretical background and interpretation of “energy landscapes,” balanced inhibition, and selective inhibition in attractor networks.
