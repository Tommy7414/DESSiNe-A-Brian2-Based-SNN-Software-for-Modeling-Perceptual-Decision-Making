# Plot_func.py

"""
## Introduction:
This file contains the `plot_data` function used for visualizing the simulation results of the neural network. The plots display various metrics such as membrane potentials, spike rastergrams, and firing rates for different neuron groups, including excitatory (E_L, E_R), inhibitory (I_L, I_R), and non-specific excitatory (NSE) neurons.

### Function Overview:
- `plot_data`: Creates a 3x3 grid of subplots:
  1. Membrane potentials of E_L and E_R neurons.
  2. Spike rastergrams for E_L and E_R neurons.
  3. Firing rates of E_L and E_R neurons.
  4. Membrane potentials of inhibitory neurons I_L and I_R.
  5. Spike rastergrams for inhibitory neurons.
  6. Firing rates of inhibitory neurons.
  7. Membrane potential of NSE neurons.
  8. Spike rastergrams for NSE neurons.
  9. Firing rates of NSE neurons.
  
The function also displays decision outcomes such as "E_L Firing", "E_R Firing", "No Decision", or "Both Firing", along with reaction times (RT) when applicable.
"""

import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

def plot_data(canvas,
              monitors,
              n_E,
              n_I,
              n_NSE,
              v_threshold=-0.05,    # Membrane potential threshold for spike generation
              window_size=5,        # Smoothing window size for firing rates
              EL_firing=False,      # Whether E_L neurons fired to make a decision
              ER_firing=False,      # Whether E_R neurons fired to make a decision
              NO_decision=False,    # Whether no decision occurred
              NO_decision_2_firing=False,  # Whether both populations fired simultaneously
              EL_RT=0.0,            # Reaction time for E_L decision (in ms)
              ER_RT=0.0):           # Reaction time for E_R decision (in ms)
    """
    Parameters:
    - canvas: Matplotlib canvas to draw the plots.
    - monitors: Dictionary containing all Brian2 monitors (StateMonitor, SpikeMonitor, PopulationRateMonitor).
    - n_E, n_I, n_NSE: Number of excitatory, inhibitory, and NSE neurons, respectively.
    - v_threshold: Membrane potential threshold for spike generation (default -0.05).
    - window_size: Time window size for smoothing firing rates (default 5 ms).
    - EL_firing, ER_firing: Flags indicating whether the left or right excitatory neurons fired to make a decision.
    - NO_decision: Flag indicating no decision.
    - NO_decision_2_firing: Flag indicating simultaneous firing of both populations.
    - EL_RT, ER_RT: Reaction times (RT) for E_L and E_R decisions.
    """

    # Clear the existing figure for new plots
    fig = canvas.figure
    fig.clear()

    y_min = -0.08  # Minimum y-axis value for membrane potential plots
    y_max = 0      # Maximum y-axis value for membrane potential plots

    # Vertical line range for marking spikes in the membrane potential plots
    spike_bottom = v_threshold
    spike_top = spike_bottom + 50.0

    # Load the monitors for different neuron groups
    MP_L = monitors['MP_L']     # StateMonitor for E_L neurons
    MP_R = monitors['MP_R']     # StateMonitor for E_R neurons
    spike_EL = monitors['spike_EL']  # SpikeMonitor for E_L neurons
    spike_ER = monitors['spike_ER']  # SpikeMonitor for E_R neurons
    rate_EL = monitors['rate_EL']    # PopulationRateMonitor for E_L neurons
    rate_ER = monitors['rate_ER']    # PopulationRateMonitor for E_R neurons

    MP_I_L = monitors['MP_I_L']  # StateMonitor for I_L neurons
    MP_I_R = monitors['MP_I_R']  # StateMonitor for I_R neurons
    spike_IL = monitors['spike_IL']  # SpikeMonitor for I_L neurons
    spike_IR = monitors['spike_IR']  # SpikeMonitor for I_R neurons
    rate_IL = monitors['rate_IL']    # PopulationRateMonitor for I_L neurons
    rate_IR = monitors['rate_IR']    # PopulationRateMonitor for I_R neurons

    MP_NSE = monitors['MP_NSE']  # StateMonitor for NSE neurons
    spike_NSE = monitors['spike_NSE']  # SpikeMonitor for NSE neurons
    rate_NSE = monitors['rate_NSE']    # PopulationRateMonitor for NSE neurons

    ## Plot 1: Membrane potentials of E_L and E_R neurons
    ax = fig.add_subplot(331)
    ax.plot(MP_L.t/ms, MP_L.v[0], 'darkblue', label='E_L Neuron 0')
    ax.plot(MP_R.t/ms, MP_R.v[0], 'indigo', label='E_R Neuron 0')

    # Add vertical lines for spike times of the 0th neuron in E_L and E_R
    t_spikes = spike_EL.t[spike_EL.i == 0]
    if len(t_spikes) > 0:
        ax.plot([t_spikes/ms, t_spikes/ms], [spike_top, spike_bottom], color='darkblue', linewidth=1.0)

    t_spikes = spike_ER.t[spike_ER.i == 0]
    if len(t_spikes) > 0:
        ax.plot([t_spikes/ms, t_spikes/ms], [spike_top, spike_bottom], color='indigo', linewidth=1.0)

    ax.set_ylim([y_min, y_max])
    ax.legend()

    ## Plot 2: Spike rastergrams for E_L and E_R neurons
    ax = fig.add_subplot(332)
    ax.scatter(spike_EL.t/ms, spike_EL.i, s=0.125, color='darkblue', label='E_L')
    ax.scatter(spike_ER.t/ms, spike_ER.i + n_E, s=0.125, color='indigo', label='E_R')
    ax.set_ylim([-0.5, n_E * 2 - 0.5])

    ## Plot 3: Firing rates of E_L and E_R neurons
    ax = fig.add_subplot(333)
    ax.plot(rate_EL.t/ms, rate_EL.smooth_rate(width=window_size*ms), color="darkblue")
    ax.plot(rate_ER.t/ms, rate_ER.smooth_rate(width=window_size*ms), color="indigo")
    ax.set_ylim([0, 120])
    ax.grid(True)

    ## Plot 4: Membrane potentials of I_L and I_R neurons
    ax = fig.add_subplot(334)
    ax.plot(MP_I_L.t/ms, MP_I_L.v[0], color='royalblue', label='I_L Neuron 0')
    ax.plot(MP_I_R.t/ms, MP_I_R.v[0], color='mediumorchid', label='I_R Neuron 0')

    # Add vertical lines for spike times of the 0th neuron in I_L and I_R
    t_spikes = spike_IL.t[spike_IL.i == 0]
    if len(t_spikes) > 0:
        ax.plot([t_spikes/ms, t_spikes/ms], [spike_top, spike_bottom], color='royalblue', linewidth=1.0)
    t_spikes = spike_IR.t[spike_IR.i == 0]
    if len(t_spikes) > 0:
        ax.plot([t_spikes/ms, t_spikes/ms], [spike_top, spike_bottom], color='mediumorchid', linewidth=1.0)
    ax.set_ylim([y_min, y_max])
    ax.legend()

    ## Plot 5: Spike rastergrams for inhibitory neurons
    ax = fig.add_subplot(335)
    ax.scatter(spike_IL.t/ms, spike_IL.i, s=0.125, color='royalblue', label='I_L')
    ax.scatter(spike_IR.t/ms, spike_IR.i + n_I, s=0.125, color='mediumorchid', label='I_R')
    ax.set_ylim([-0.5, n_I * 2 - 0.5])

    ## Plot 6: Firing rates of inhibitory neurons
    ax = fig.add_subplot(336)
    ax.plot(rate_IL.t/ms, rate_IL.smooth_rate(width=window_size*ms), color="royalblue", label='I_L')
    ax.plot(rate_IR.t/ms, rate_IR.smooth_rate(width=window_size*ms), color="mediumorchid", label='I_R')
    ax.set_ylim([0, 120])
    ax.grid(True)

    ## Plot 7: Membrane potentials of NSE neurons
    ax = fig.add_subplot(337)
    ax.plot(MP_NSE.t/ms, MP_NSE.v[0], color='green', label='NSE Neuron 0')

    # Add vertical lines for spike times of the 0th neuron in NSE
    t_spikes = spike_NSE.t[spike_NSE.i == 0]
    if len(t_spikes) > 0:
        ax.plot([t_spikes/ms, t_spikes/ms], [spike_top, spike_bottom], color='green', linewidth=1.0)
    ax.set_ylim([y_min, y_max])
    ax.legend()

    ## Plot 8: Spike rastergrams for NSE neurons
    ax = fig.add_subplot(338)
    ax.scatter(spike_NSE.t/ms, spike_NSE.i, s=0.125, color='g')
    ax.set_ylim([-0.5, n_NSE - 0.5])

    ## Plot 9: Firing rates of NSE neurons
    ax = fig.add_subplot(339)
    ax.plot(rate_NSE.t/ms, rate_NSE.smooth_rate(width=window_size*ms), color="green")
    ax.set_ylim([0, 120])
    ax.grid(True)

    # Display decision results and reaction times on the last plot
    if EL_firing:
        ax.text(0.5, 3.4, f'E_L Firing, RT = {EL_RT:.1f} ms',
                transform=ax.transAxes, fontsize=12, color='darkblue',
                bbox=dict(facecolor='white', alpha=0.5))
    elif ER_firing:
        ax.text(0.5, 3.4, f'E_R Firing, RT = {ER_RT:.1f} ms',
                transform=ax.transAxes, fontsize=12, color='indigo',
                bbox=dict(facecolor='white', alpha=0.5))
    elif NO_decision:
        ax.text(0.5, 3.4, 'No Decision',
                transform=ax.transAxes, fontsize=12, color='red',
                bbox=dict(facecolor='white', alpha=0.5))
    else:
        ax.text(0.5, 3.4, 'Both Firing',
                transform=ax.transAxes, fontsize=12, color='purple',
                bbox=dict(facecolor='white', alpha=0.5))

    fig.tight_layout()  # Adjust layout for readability
    canvas.draw()  # Refresh the canvas
