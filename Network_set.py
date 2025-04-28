# Network_set.py
from brian2 import *
import numpy as np
prefs.codegen.target = 'numpy'
import pickle

# ----------------- Introduction ----------------- #
"""
This section outlines the functions used to create the network.

1. init_parameters: Initialize the parameters for the network
- The parameters here are further configured in GUI.ipynb.
- To modify parameters, you can update them within this function 
  and ensure the corresponding changes are reflected in PDMSNN_DM.py/ PDMSNN_EL.py.

2. define_equations: Define the equations for the neurons

3. create_neuron_groups: Create the neuron groups for the network
- Basic neuron groups (5): E_L, E_R, Inh_L, Inh_R, NSE.
- Top-down control neuron groups (optional, 4): Ctr1, Ctr2, Ctr3, Ctr4.

4. create_synapses: Create the synapses for the network
- Total synapses: 40 basic synapses and 4 top-down synapses.
- Synaptic weights are only set within this function. (You can modify them here.)
- In PDMSNN_DM.py/ PDMSNN_EL.py, you can adjust the connection percentage to change the degree of selectivity.
  - gamma_ee: The percentage of weights for exc -> exc
  - gamma_ie: The percentage of weights for exc -> inh
  - gamma_ei: The percentage of weights for inh -> exc
  - gamma_ii: The percentage of weights for inh -> inh

5. init_monitors: Initialize the monitors for the network

6. build_network: Build the network

7. run_stimulus: Run the stimulus for the network
"""

# ----------------- Functions ----------------- #

def init_parameters(n_E=60, n_I=50, n_NSE=280, gamma_ei=0, gamma_ie=0, gamma_ee=1, gamma_ii=0, top_down=0, n_Ctr=125, S=0, R=0):
    """
    Initialize the parameters for the network
    """

    params = {}
    params['n_E'] = n_E
    params['n_I'] = n_I
    params['n_NSE'] = n_NSE
    params['gamma_ei'] = gamma_ei
    params['gamma_ie'] = gamma_ie
    params['gamma_ee'] = gamma_ee
    params['gamma_ii'] = gamma_ii
    params['top_down'] = top_down
    params['n_Ctr'] = n_Ctr
    params['S'] = S
    params['R'] = R
    return params

def define_equations():
    """
    Define the equations for the neurons.
    
    ## Structure of the Equations:
    - eqs_L:    Equations for the left excitatory neurons.
    - eqs_R:    Equations for the right excitatory neurons.
    - eqs_I:    Equations for the inhibitory neurons.
    - eqs_Ctrl: Equations for the top-down control neurons.

    ## Variables (for Each Equation):
    (Note: In the Brian2 unit system, you can use 1 for the unit of the variables.)
    - dv/dt (membrane potential): Sum of all currents minus the membrane potential.
    - v_threshold:                Threshold for firing.
    - v_reset:                    Reset potential after firing.
    - v0:                         Resting potential.
    - ref:                        Refractory period duration.
    - I:                          External input current.
    - tau:                        Membrane time constant.
    """

    
    eqs_L = '''
    dv/dt = (I + Ise + Isi + Isnse + Ibsi -(v-v0))/tau : 1 (unless refractory)
    v_threshold : 1  (constant)
    v_reset : 1    (constant)
    v0 : 1
    ref : second 
    I : 1 
    Ise = IseL_ampa + IseL_nmda + IseR_ampa + IseR_nmda : 1
    IseL_ampa :1
    IseL_nmda :1
    IseR_ampa :1
    IseR_nmda :1
    Isi = Isi_L + Isi_R : 1
    Isi_L :1
    Isi_R :1
    Isnse = Isnse_ampa + Isnse_nmda : 1
    Isnse_ampa :1
    Isnse_nmda :1
    Ibsi = Ibsi_1 + Ibsi_2 : 1
    Ibsi_1 :1
    Ibsi_2 :1
    tau : second
    '''

    eqs_R = '''
    dv/dt = (I + Ise + Isi + Isnse + Ibsi -(v-v0))/tau : 1 (unless refractory)
    v_threshold : 1  (constant)
    v_reset : 1    (constant)
    v0 : 1
    ref : second
    I : 1
    Ise = IseL_ampa + IseL_nmda + IseR_ampa + IseR_nmda : 1
    IseL_ampa :1
    IseL_nmda :1
    IseR_ampa :1
    IseR_nmda :1
    Isi = Isi_L + Isi_R : 1
    Isi_L :1
    Isi_R :1
    Isnse = Isnse_ampa + Isnse_nmda : 1
    Isnse_ampa :1
    Isnse_nmda :1
    Ibsi = Ibsi_3 + Ibsi_4 : 1
    Ibsi_3 :1
    Ibsi_4 :1
    tau : second
    '''

    eqs_I = '''
    dv/dt = (I + Ise + Isi + Isnse  -(v-v0))/tau : 1 (unless refractory)
    v_threshold : 1  (constant)
    v_reset : 1    (constant)
    v0 : 1
    ref : second
    I : 1
    Ise = IseL_ampa + IseL_nmda + IseR_ampa + IseR_nmda : 1
    IseL_ampa :1
    IseL_nmda :1
    IseR_ampa :1
    IseR_nmda :1
    Isi = Isi_L + Isi_R : 1
    Isi_L :1
    Isi_R :1
    Isnse = Isnse_ampa + Isnse_nmda : 1
    Isnse_ampa :1
    Isnse_nmda :1
    tau : second
    '''

    eqs_Ctrl = '''
    dv/dt =(I - (v - v0))/tau : 1( unless refractory)
    v_threshold : 1  (constant)
    v_reset : 1    (constant)
    v0 : 1
    ref : second
    I : 1
    tau : second
    '''
    return eqs_L, eqs_R, eqs_I, eqs_Ctrl

def create_neuron_groups(params, eqs):
    """
    Create the neuron groups for the network.

    ## Neuron Groups:
    - E_L:   Left excitatory neurons.
    - E_R:   Right excitatory neurons.
    - Inh_L: Left inhibitory neurons.
    - Inh_R: Right inhibitory neurons.
    - NSE:   Non-specific excitatory neurons.
    - Ctr1, Ctr2, Ctr3, Ctr4: Top-down control neurons (optional).

    ## Structure for Setting the Neuron Groups:
    - NeuronGroup(n, eqs, threshold='v > v_threshold', reset='v = v_reset', refractory='ref', method='euler')
      - `n`: Number of neurons in the group.
      - `eqs`: Equations defining the dynamics of the neurons.
      - `threshold`: Condition for spike firing (e.g., `v > v_threshold`).
      - `reset`: Action taken when the neuron spikes (e.g., `v = v_reset`).
      - `refractory`: Refractory period after a spike (`ref`).
      - `method='euler'`: Numerical integration method (Euler method).
    """

    ## neuron_groups
    I = 0.000
    v0 = -0.07
    v_reset = -0.055
    v_threshold = -0.05 # Threshold for firing
    t_ref = 2*ms
    E_e = 0  # Reversal potential of excitatory synapses
    E_i = -0.07  # Reversal potential of inhibitory synapses

    ## Membrane time constant
    # for Exc neurons
    C_m_ex = 0.5 * 10**-9
    g_L_ex = 25 * 10**-9
    tau_m_ex = (1000*C_m_ex/g_L_ex)*ms
    # for Inh neurons
    C_m_in = 0.2 * 10**-9
    g_L_in = 20 * 10**-9
    tau_m_in = (1000*C_m_in/g_L_in)*ms

    ### Define the neuron model ###
    eqs_L, eqs_R, eqs_I, eqs_Ctrl = eqs
    # popl E_L
    E_L = NeuronGroup(params['n_E'], eqs_L, threshold='v > v_threshold',  reset='v = v_reset', refractory='ref', method='euler')
    E_L.v0 = -0.07
    E_L.v_threshold = -0.05
    E_L.v_reset = -0.055
    E_L.tau = tau_m_ex
    E_L.v = v0
    E_L.ref = t_ref
    E_L.I = 0.0
    E_L.IseL_ampa = 0.0
    E_L.IseL_nmda = 0.0
    E_L.IseR_ampa = 0.0
    E_L.IseR_nmda = 0.0
    E_L.Isi_L = 0.0
    E_L.Isi_R = 0.0
    E_L.Isnse_ampa = 0.0
    E_L.Isnse_nmda = 0.0
    E_L.Ibsi_1 = 0.0
    E_L.Ibsi_2 = 0.0

    # popl E_R
    E_R = NeuronGroup(params['n_E'], eqs_R, threshold='v > v_threshold',  reset='v = v_reset', refractory='ref', method='euler')
    E_R.v0 = -0.07
    E_R.v_threshold = -0.05
    E_R.v_reset = -0.055
    E_R.tau = tau_m_ex
    E_R.v = v0
    E_R.ref = t_ref
    E_R.I = 0.0
    E_R.IseL_ampa = 0.0
    E_R.IseL_nmda = 0.0
    E_R.IseR_ampa = 0.0
    E_R.IseR_nmda = 0.0
    E_R.Isi_L = 0.0
    E_R.Isi_R = 0.0
    E_R.Isnse_ampa = 0.0
    E_R.Isnse_nmda = 0.0
    E_R.Ibsi_3 = 0.0
    E_R.Ibsi_4 = 0.0


    # popl Inh_L
    Inh_L = NeuronGroup(params['n_I'], eqs_I, threshold='v > v_threshold',  reset='v = v_reset', refractory='ref', method='euler')
    Inh_L.v0 = -0.07
    Inh_L.v_threshold = -0.05
    Inh_L.v_reset = -0.055
    Inh_L.tau = tau_m_in
    Inh_L.v = v0
    Inh_L.ref = t_ref
    Inh_L.I = 0.0
    Inh_L.IseL_ampa = 0.0
    Inh_L.IseL_nmda = 0.0
    Inh_L.IseR_ampa = 0.0
    Inh_L.IseR_nmda = 0.0
    Inh_L.Isi_L = 0.0
    Inh_L.Isi_R = 0.0
    Inh_L.Isnse_ampa = 0.0
    Inh_L.Isnse_nmda = 0.0

    # popl Inh_R
    Inh_R = NeuronGroup(params['n_I'], eqs_I, threshold='v > v_threshold',  reset='v = v_reset', refractory='ref', method='euler')
    Inh_R.v0 = -0.07
    Inh_R.v_threshold = -0.05
    Inh_R.v_reset = -0.055
    Inh_R.tau = tau_m_in
    Inh_R.v = v0
    Inh_R.ref = t_ref
    Inh_R.I = 0.0
    Inh_R.IseL_ampa = 0.0
    Inh_R.IseL_nmda = 0.0
    Inh_R.IseR_ampa = 0.0
    Inh_R.IseR_nmda = 0.0
    Inh_R.Isi_L = 0.0
    Inh_R.Isi_R = 0.0
    Inh_R.Isnse_ampa = 0.0
    Inh_R.Isnse_nmda = 0.0

    # popl NSE
    NSE = NeuronGroup(params['n_NSE'], eqs_I, threshold='v > v_threshold',  reset='v = v_reset', refractory='ref', method='euler')
    NSE.v0 = -0.07
    NSE.v_threshold = -0.05
    NSE.v_reset = -0.055
    NSE.tau = tau_m_ex
    NSE.v = v0
    NSE.ref = t_ref
    NSE.I = 0.00
    NSE.IseL_ampa = 0.0
    NSE.IseL_nmda = 0.0
    NSE.IseR_ampa = 0.0
    NSE.IseR_nmda = 0.0
    NSE.Isi_L = 0.0
    NSE.Isi_R = 0.0
    NSE.Isnse_ampa = 0.0
    NSE.Isnse_nmda = 0.0

    if params['top_down'] :
        # popl Ctr1
        Ctr1 = NeuronGroup(params['n_Ctr'], eqs_Ctrl, threshold='v > v_threshold',  reset='v = v_reset', refractory='ref', method='euler')
        Ctr1.tau = tau_m_ex
        Ctr1.v = v0
        Ctr1.ref = t_ref
        Ctr1.v0 = -0.07
        Ctr1.v_threshold = -0.05
        Ctr1.v_reset = -0.055
        Ctr1.I = 0.0

        # popl Ctr2
        Ctr2 = NeuronGroup(params['n_Ctr'], eqs_Ctrl, threshold='v > v_threshold',  reset='v = v_reset', refractory='ref', method='euler')
        Ctr2.tau = tau_m_in
        Ctr2.v = v0
        Ctr2.ref = t_ref
        Ctr2.v0 = -0.07
        Ctr2.v_threshold = -0.05
        Ctr2.v_reset = -0.055
        Ctr2.I = 0

        # popl Ctr3
        Ctr3 = NeuronGroup(params['n_Ctr'], eqs_Ctrl, threshold='v > v_threshold',  reset='v = v_reset', refractory='ref', method='euler')
        Ctr3.tau = tau_m_ex
        Ctr3.v = v0
        Ctr3.ref = t_ref
        Ctr3.v0 = -0.07
        Ctr3.v_threshold = -0.05
        Ctr3.v_reset = -0.055
        Ctr3.I = 0

        # popl Ctr4
        Ctr4 = NeuronGroup(params['n_Ctr'], eqs_Ctrl, threshold='v > v_threshold',  reset='v = v_reset', refractory='ref', method='euler')
        Ctr4.tau = tau_m_in
        Ctr4.v = v0
        Ctr4.ref = t_ref
        Ctr4.v0 = -0.07
        Ctr4.v_threshold = -0.05
        Ctr4.v_reset = -0.055
        Ctr4.I = 0

    neuron_groups = {
        'E_L': E_L, 'E_R': E_R, 'Inh_L': Inh_L, 'Inh_R': Inh_R, 'NSE': NSE}
    if params['top_down'] :
        neuron_groups['Ctr1'] = Ctr1
        neuron_groups['Ctr2'] = Ctr2
        neuron_groups['Ctr3'] = Ctr3
        neuron_groups['Ctr4'] = Ctr4
    return neuron_groups

def create_synapses(params, neuron_groups):
    """
    Create the synapses for the network.

    ## Synapse Model:
    - **Recurrent Connections:**
      - E_L -> E_L (AMPA / NMDA)
      - E_R -> E_R (AMPA / NMDA)
    - **Cross-Hemispheric Connections:**
      - E_L -> E_R (AMPA / NMDA)
      - E_R -> E_L (AMPA / NMDA)
    - **Excitatory to Inhibitory Connections:**
      - E_L -> I_L (AMPA / NMDA)
      - E_L -> I_R (AMPA / NMDA)
      - E_R -> I_L (AMPA / NMDA)
      - E_R -> I_R (AMPA / NMDA)
    - **Inhibitory to Excitatory Connections:**
      - I_L -> E_L (GABA)
      - I_L -> E_R (GABA)
      - I_R -> E_L (GABA)
      - I_R -> E_R (GABA)
    - **Inhibitory Interneuronal Connections:**
      - I_L -> I_R (GABA)
      - I_R -> I_L (GABA)
    - **Non-Specific Excitatory (NSE) Connections:**
      - NSE -> E_L (AMPA / NMDA)
      - NSE -> E_R (AMPA / NMDA)
      - NSE -> I_L (AMPA / NMDA)
      - NSE -> I_R (AMPA / NMDA)
      - NSE -> NSE (AMPA / NMDA)
      - I_L -> NSE (GABA)
      - I_R -> NSE (GABA)
    
    ## Top-Down Control Signals:
    - Ctr1 -> E_L (AMPA / NMDA)
    - Ctr2 -> I_L (AMPA / NMDA)
    - Ctr3 -> E_R (AMPA / NMDA)
    - Ctr4 -> I_R (AMPA / NMDA)

    ## Structure for Setting the Synapses:
    - Synapses(pre, post, model='''', on_pre='''', method='euler')
      - `pre`: Pre-synaptic neuron group.
      - `post`: Post-synaptic neuron group.
      - `model`: Synapse model equations.
      - `on_pre`: Action performed when a spike is received.
      - `method='euler'`: Numerical integration method (Euler method).
    - `"Name of the current"_post`: Represents the post-synaptic current.
    - **Summed currents:** All incoming synaptic currents are summed.
    - **Clock-driven:** Synapse updates occur at each time step.

    ## Synapse Currents:
    - I_syn = g_AMPA * s_AMPA(t) * (V(t) - V_E)
            + g_NMDA * s_NMDA(t) * (V(t) - V_E) / (1 + 1/2 * [Mg2+] * exp(-0.062 * V(t) / 3.57))
            + g_GABA * s_GABA(t) * (V(t) - V_I)

    ## Gating Variables for AMPA / NMDA / GABA:
    ### AMPA:
    - `s_AMPA`: Gating variable for AMPA.
    - Equation: ds_AMPA(t)/dt = sum_k δ(t - t^k) - s_AMPA / τ_AMPA
    
    ### NMDA:
    - `s_NMDA`: Gating variable for NMDA.
    - Equation: ds_NMDA(t)/dt = α * (1 - s_NMDA(t)) * sum_k δ(t - t^k) - s_NMDA / τ_NMDA

    ### GABA:
    - `s_GABA`: Gating variable for GABA.
    - Equation: ds_GABA(t)/dt = sum_k δ(t - t^k) - s_GABA / τ_GABA

    ## Poisson Input:
    Contributes noisy baseline input to all neuron groups.
    
    ### Characteristics:
    - Adds independent noise to the membrane potential of every neuron, create small baseline fluctuations (5~10Hz)
    - Achieved by injecting a Poisson-distributed spike train (from a single "noisy" neuron) into each neuron.
    - Noise is independent for each neuron.

    ### Parameters:
    - `poisson_rate`: The rate of the Poisson input.
      - High rates are used to prevent saturation effects.
    - `poisson_conductance`: The strength of the Poisson input for regular neurons.
    - `poisson_conductance_Ctrl`: The strength of the Poisson input for top-down control neurons.

    ### Top-Down Control Explanation:
    - The firing rate of top-down control neurons is controlled by fixing the conductance.
    - You can adjust the **ratio (R)** and **strength (S)** of the conductance in `GUI.ipynb` to control the top-down signal.
      - Higher `R`: Stronger top-down inhibition, leading to more conservative decisions.
      - Higher `S`: Both top-down excitation and inhibition are stronger.
    - `gctr_e_AMPA = params['S'] / (0.3 * re)`: AMPA synaptic strength for excitatory top-down control.
    - `gctr_i_GABA = params['R'] * re * gctr_e_AMPA / ri`: GABA synaptic strength for inhibitory top-down control.

    """

    """
    ## Parameter Descriptions:

    ### Membrane Potential and Reset Parameters:
    - `I = 0.000`: External input current (default set to 0).
    - `v0 = -0.07`: Resting membrane potential (in volts).
    - `v_reset = -0.055`: Reset potential after a spike (in volts).
    - `v_threshold = -0.05`: Threshold for firing a spike (in volts).
    - `t_ref = 2*ms`: Refractory period duration (in milliseconds).

    ### Reversal Potentials:
    - `E_e = 0`: Reversal potential for excitatory synapses (in volts).
    - `E_i = -0.07`: Reversal potential for inhibitory synapses (in volts).

    ### Membrane Capacitance and Leak Conductance (Excitatory Neurons):
    - `C_m_ex = 0.5 * 10**-9`: Membrane capacitance for excitatory neurons (in farads).
    - `g_L_ex = 25 * 10**-9`: Leak conductance for excitatory neurons (in siemens).
    - `tau_m_ex = (1000*C_m_ex/g_L_ex)*ms`: Membrane time constant for excitatory neurons, calculated as 20 ms.

    ### Membrane Capacitance and Leak Conductance (Inhibitory Neurons):
    - `C_m_in = 0.2 * 10**-9`: Membrane capacitance for inhibitory neurons (in farads).
    - `g_L_in = 20 * 10**-9`: Leak conductance for inhibitory neurons (in siemens).

    ### Synaptic Time Constants:
    - `tau_AMPA = 2*ms`: Time constant for AMPA receptors (in milliseconds).
    - `tau_NMDA = 100*ms`: Time constant for NMDA receptors (in milliseconds).
    - `tau_GABA = 5*ms`: Time constant for GABA receptors (in milliseconds).

    ### NMDA Receptor Dynamics:
    - `alpha = 0.63`: NMDA gating variable scaling factor.
    - `y = 1`: NMDA receptor state variable constant.
   
    ### Synaptic Conductance Parameters:
    - `ge_AMPA_rec = 0.00013 * (1 + params['gamma_ee'])`: AMPA recurrent excitatory conductance.
    - `ge_NMDA_rec = 0.0006 * (1 + params['gamma_ee'])`: NMDA recurrent excitatory conductance.
    - `ge_AMPA_cross = 0.00013 / (1 + params['gamma_ee'])`: AMPA cross-hemispheric excitatory conductance.
    - `ge_NMDA_cross = 0.0006 / (1 + params['gamma_ee'])`: NMDA cross-hemispheric excitatory conductance.
    - `gie_GABA_same = 0.0147 / (1 + params['gamma_ie'])`: GABA same-hemisphere inhibitory conductance.
    - `gie_GABA_opposite = 0.0147 * (1 + params['gamma_ie'])`: GABA cross-hemisphere inhibitory conductance.
    - `gi_GABA_rec = 0.000108 * 31 / (1 + params['gamma_ii'])`: GABA recurrent inhibitory conductance.
    - `gi_GABA_cross = 0.000108 * 31 * (1 + params['gamma_ii'])`: GABA cross-hemispheric inhibitory conductance.
    - `gei_AMPA_same = 0.00013 * (1 + params['gamma_ei'])`: AMPA same-hemisphere excitatory-to-inhibitory conductance.
    - `gei_NMDA_same = 0.00036 * (1 + params['gamma_ei'])`: NMDA same-hemisphere excitatory-to-inhibitory conductance.
    - `gei_AMPA_opposite = 0.00013 / (1 + params['gamma_ei'])`: AMPA cross-hemisphere excitatory-to-inhibitory conductance.
    - `gei_NMDA_opposite = 0.00036 / (1 + params['gamma_ei'])`: NMDA cross-hemisphere excitatory-to-inhibitory conductance.
    - `gnse_AMPA = 0.00005`: AMPA conductance for non-specific excitatory input.
    - `gnse_NMDA = 0.00015`: NMDA conductance for non-specific excitatory input.
    - `gnsi_AMPA = 0.00006`: AMPA conductance for non-specific inhibitory input.
    - `gnsi_NMDA = 0.00024`: NMDA conductance for non-specific inhibitory input.
    - `gins_GABA = 0.000132 * 80`: GABA conductance for non-specific inhibitory neurons.
    """

    I = 0.000
    v0 = -0.07
    v_reset = -0.055
    v_threshold = -0.05 # Threshold for firing
    t_ref = 2*ms
    E_e = 0  # Reversal potential of excitatory synapses
    E_i = -0.07  # Reversal potential of inhibitory synapses
    C_m_ex = 0.5 * 10**-9
    g_L_ex = 25 * 10**-9
    tau_m_ex = (1000*C_m_ex/g_L_ex)*ms # = 20ms
    # for Inh neurons 
    C_m_in = 0.2 * 10**-9
    g_L_in = 20 * 10**-9
    ### constants
    E_e = 0  # Reversal potential of excitatory synapses
    E_i = -0.07  # Reversal potential of inhibitory synapses
    ## Synaptic time constant
    # for AMPA
    tau_AMPA = 2*ms
    # for NMDA
    tau_NMDA = 100*ms
    y = 1
    alpha = 0.63
    # for GABA
    tau_GABA = 5*ms

    ## Noisy baseline input constant
    poisson_rate_E = 1200*Hz
    poisson_rate_I = 2400*Hz
    # parameters
    poisson_conductance_E =  0.000723
    poisson_conductance_I =  0.000727

    ## if Top-Down Signal
    poisson_conductance_Ctrl_E = 0.000424
    poisson_conductance_Ctrl_I = 0.000746
    re = 27.09627587529249
    ri = 26.84231894755238
    # parameters

    gctr_e_AMPA = params['S']/(0.3*re)
    gctr_i_GABA = params['R']*re*gctr_e_AMPA/ri

    ## Synaptic conductance
    ge_AMPA_rec       = 0.00013 * (1 + params['gamma_ee'])
    ge_NMDA_rec       = 0.00062 * (1 + params['gamma_ee'])
    ge_AMPA_cross     = 0.00013 / (1 + params['gamma_ee'])
    ge_NMDA_cross     = 0.00062 / (1 + params['gamma_ee'])
    gie_GABA_same     = 0.0151 / (1 + params['gamma_ie'])
    gie_GABA_opposite = 0.0151  * (1 + params['gamma_ie'])
    gi_GABA_rec       = 0.000108 * 31 / (1 + params['gamma_ii'])
    gi_GABA_cross     = 0.000108 * 31 * (1 + params['gamma_ii'])
    gei_AMPA_same     = 0.00013 * (1 + params['gamma_ei'])
    gei_NMDA_same     = 0.00038 * (1 + params['gamma_ei'])
    gei_AMPA_opposite = 0.00013 / (1 + params['gamma_ei'])
    gei_NMDA_opposite = 0.00038 / (1 + params['gamma_ei'])
    gnse_AMPA         = 0.00005
    gnse_NMDA         = 0.00015
    gnsi_AMPA         = 0.00006
    gnsi_NMDA         = 0.00024
    gins_GABA         = 0.000132* 80

    # ----------------- Synapse Model ----------------- #
    ## Define synapse connection ##
    # E_L recurrent (excitatory) (AMPA)
    S_elel_AMPA = Synapses(neuron_groups['E_L'], neuron_groups['E_L'], model='''
    IseL_ampa_post = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_elel_AMPA.connect(condition='i!=j', p=1.0)
    S_elel_AMPA.g_AMPA = ge_AMPA_rec
    S_elel_AMPA.taus_AMPA = tau_AMPA
    S_elel_AMPA.Erev = E_e

    # E_L recurrent (excitatory) (NMDA)
    S_elel_NMDA = Synapses(neuron_groups['E_L'], neuron_groups['E_L'], model='''
    IseL_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_elel_NMDA.connect(condition='i!=j', p=1.0)
    S_elel_NMDA.alpha = alpha
    S_elel_NMDA.g_NMDA = ge_NMDA_rec
    S_elel_NMDA.taus_NMDA = tau_NMDA
    S_elel_NMDA.Erev = E_e

    # E_R recurrent (excitatory) (AMPA)
    S_erer_AMPA = Synapses(neuron_groups['E_R'], neuron_groups['E_R'], model= '''
    IseR_ampa_post = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_erer_AMPA.connect(condition='i!=j', p=1.0) 
    S_erer_AMPA.g_AMPA = ge_AMPA_rec 
    S_erer_AMPA.taus_AMPA = tau_AMPA 
    S_erer_AMPA.Erev = E_e 

    # E_R recurrent (excitatory) (NMDA)
    S_erer_NMDA = Synapses(neuron_groups['E_R'], neuron_groups['E_R'], model='''
    IseR_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_erer_NMDA.connect(condition='i!=j', p=1.0 )
    S_erer_NMDA.alpha = alpha
    S_erer_NMDA.g_NMDA = ge_NMDA_rec
    S_erer_NMDA.taus_NMDA = tau_NMDA
    S_erer_NMDA.Erev = E_e

    # E_L -> E_R (excitatory) (AMPA)
    S_eler_AMPA = Synapses(neuron_groups['E_L'], neuron_groups['E_R'], model='''
    IseL_ampa_post  = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_eler_AMPA.connect(p = 1.0)
    S_eler_AMPA.g_AMPA = ge_AMPA_cross
    S_eler_AMPA.taus_AMPA = tau_AMPA
    S_eler_AMPA.Erev = E_e

    # E_L -> E_R (excitatory) (NMDA)
    S_eler_NMDA = Synapses(neuron_groups['E_L'], neuron_groups['E_R'], model='''
    IseL_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_eler_NMDA.connect(p = 1.0)
    S_eler_NMDA.alpha = alpha
    S_eler_NMDA.g_NMDA = ge_NMDA_cross
    S_eler_NMDA.taus_NMDA = tau_NMDA
    S_eler_NMDA.Erev = E_e

    # E_R -> E_L (excitatory) (AMPA)
    S_erel_AMPA = Synapses(neuron_groups['E_R'], neuron_groups['E_L'], model='''
    IseR_ampa_post  = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_erel_AMPA.connect(p = 1.0)
    S_erel_AMPA.g_AMPA = ge_AMPA_cross
    S_erel_AMPA.taus_AMPA = tau_AMPA
    S_erel_AMPA.Erev = E_e

    # E_R -> E_L (excitatory) (NMDA)
    S_erel_NMDA = Synapses(neuron_groups['E_R'], neuron_groups['E_L'], model='''
    IseR_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_erel_NMDA.connect(p = 1.0)
    S_erel_NMDA.alpha = alpha
    S_erel_NMDA.g_NMDA = ge_NMDA_cross
    S_erel_NMDA.taus_NMDA = tau_NMDA
    S_erel_NMDA.Erev = E_e

    # E_L -> I_L (excitatory) (AMPA)
    S_elil_AMPA = Synapses(neuron_groups['E_L'], neuron_groups['Inh_L'], model='''
    IseL_ampa_post = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_elil_AMPA.connect(p = 1.0)
    S_elil_AMPA.g_AMPA = gei_AMPA_same
    S_elil_AMPA.taus_AMPA = tau_AMPA
    S_elil_AMPA.Erev = E_e

    # E_L -> I_L (excitatory) (NMDA)
    S_elil_NMDA = Synapses(neuron_groups['E_L'], neuron_groups['Inh_L'], model='''
    IseL_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_elil_NMDA.connect(p = 1.0)
    S_elil_NMDA.alpha = alpha
    S_elil_NMDA.g_NMDA = gei_NMDA_same
    S_elil_NMDA.taus_NMDA = tau_NMDA
    S_elil_NMDA.Erev = E_e

    # E_L -> I_R (excitatory) (AMPA)
    S_elir_AMPA = Synapses(neuron_groups['E_L'], neuron_groups['Inh_R'], model='''
    IseL_ampa_post = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_elir_AMPA.connect(p = 1.0)
    S_elir_AMPA.g_AMPA = gei_AMPA_opposite
    S_elir_AMPA.taus_AMPA = tau_AMPA
    S_elir_AMPA.Erev = E_e

    # E_L -> I_R (excitatory) (NMDA)
    S_elir_NMDA = Synapses(neuron_groups['E_L'], neuron_groups['Inh_R'], model='''
    IseL_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_elir_NMDA.connect(p = 1.0)
    S_elir_NMDA.alpha = alpha
    S_elir_NMDA.g_NMDA = gei_NMDA_opposite
    S_elir_NMDA.taus_NMDA = tau_NMDA
    S_elir_NMDA.Erev = E_e


    # E_R -> I_L (excitatory) (AMPA)
    S_eril_AMPA = Synapses(neuron_groups['E_R'], neuron_groups['Inh_L'], model='''
    IseR_ampa_post  = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_eril_AMPA.connect(p = 1.0)
    S_eril_AMPA.g_AMPA = gei_AMPA_opposite
    S_eril_AMPA.taus_AMPA = tau_AMPA
    S_eril_AMPA.Erev = E_e

    # E_R -> I_L (excitatory) (NMDA)
    S_eril_NMDA = Synapses(neuron_groups['E_R'], neuron_groups['Inh_L'], model='''
    IseR_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_eril_NMDA.connect(p = 1.0)
    S_eril_NMDA.alpha = alpha
    S_eril_NMDA.g_NMDA = gei_NMDA_opposite
    S_eril_NMDA.taus_NMDA = tau_NMDA
    S_eril_NMDA.Erev = E_e

    # E_R -> I_R (excitatory) (AMPA)
    S_erir_AMPA = Synapses(neuron_groups['E_R'], neuron_groups['Inh_R'], model='''
    IseR_ampa_post  = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_erir_AMPA.connect(p = 1.0)
    S_erir_AMPA.g_AMPA = gei_AMPA_same
    S_erir_AMPA.taus_AMPA = tau_AMPA
    S_erir_AMPA.Erev = E_e

    # E_R -> I_R (excitatory) (NMDA)
    S_erir_NMDA = Synapses(neuron_groups['E_R'], neuron_groups['Inh_R'], model='''
    IseR_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_erir_NMDA.connect(p = 1.0)
    S_erir_NMDA.alpha = alpha
    S_erir_NMDA.g_NMDA = gei_NMDA_same
    S_erir_NMDA.taus_NMDA = tau_NMDA
    S_erir_NMDA.Erev = E_e

    # I_L -> E_L (inhibitory) (GABA)
    S_ilel_GABA = Synapses(neuron_groups['Inh_L'], neuron_groups['E_L'], model='''
    Isi_L_post =  -s_GABA*(v_post-Erev) : 1 (summed)
    ds_GABA/dt = -s_GABA/taus_GABA : 1 (clock-driven)
    Erev : 1
    g_GABA : 1
    taus_GABA : second
    ''', on_pre='s_GABA += g_GABA', method='euler')

    S_ilel_GABA.connect(p = 1.0)
    S_ilel_GABA.g_GABA = gie_GABA_same
    S_ilel_GABA.taus_GABA = tau_GABA
    S_ilel_GABA.Erev = E_i

    # I_L -> E_R (inhibitory) (GABA)
    S_iler_GABA = Synapses(neuron_groups['Inh_L'], neuron_groups['E_R'], model='''
    Isi_L_post = -s_GABA*(v_post-Erev) : 1 (summed)
    ds_GABA/dt = -s_GABA/taus_GABA : 1 (clock-driven)
    Erev : 1
    g_GABA : 1
    taus_GABA : second
    ''', on_pre='s_GABA += g_GABA', method='euler')

    S_iler_GABA.connect(p = 1.0)
    S_iler_GABA.g_GABA = gie_GABA_opposite
    S_iler_GABA.taus_GABA = tau_GABA
    S_iler_GABA.Erev = E_i

    # I_R -> E_R (inhibitory) (GABA)
    S_irer_GABA = Synapses(neuron_groups['Inh_R'], neuron_groups['E_R'], model='''
    Isi_R_post = -s_GABA*(v_post-Erev) : 1 (summed)
    ds_GABA/dt = -s_GABA/taus_GABA : 1 (clock-driven)
    Erev : 1
    g_GABA : 1
    taus_GABA : second
    ''', on_pre='s_GABA += g_GABA', method='euler')

    S_irer_GABA.connect(p = 1.0)
    S_irer_GABA.g_GABA = gie_GABA_same
    S_irer_GABA.taus_GABA = tau_GABA
    S_irer_GABA.Erev = E_i

    # I_R -> E_L (inhibitory) (GABA)
    S_irel_GABA = Synapses(neuron_groups['Inh_R'], neuron_groups['E_L'], model='''
    Isi_R_post =  -s_GABA*(v_post-Erev) : 1 (summed)
    ds_GABA/dt = -s_GABA/taus_GABA : 1 (clock-driven)
    Erev : 1
    g_GABA : 1
    taus_GABA : second
    ''', on_pre='s_GABA += g_GABA', method='euler')

    S_irel_GABA.connect(p = 1.0)
    S_irel_GABA.g_GABA = gie_GABA_opposite
    S_irel_GABA.taus_GABA = tau_GABA
    S_irel_GABA.Erev = E_i


    # I_L -> I_L (inhibitory) (GABA)
    S_ilil_GABA = Synapses(neuron_groups['Inh_L'], neuron_groups['Inh_L'], model='''
    Isi_L_post = -s_GABA*(v_post-Erev) : 1 (summed)
    ds_GABA/dt = -s_GABA/taus_GABA : 1 (clock-driven)
    Erev : 1
    g_GABA : 1
    taus_GABA : second
    ''', on_pre='s_GABA += g_GABA', method='euler')

    S_ilil_GABA.connect(condition='i!=j', p=1.0)
    S_ilil_GABA.g_GABA = gi_GABA_rec 
    S_ilil_GABA.taus_GABA = tau_GABA 
    S_ilil_GABA.Erev = E_i 

    # I_L -> I_R (inhibitory) (GABA)
    S_ilir_GABA = Synapses(neuron_groups['Inh_L'], neuron_groups['Inh_R'], model='''
    Isi_L_post = -s_GABA*(v_post-Erev) : 1 (summed)
    ds_GABA/dt = -s_GABA/taus_GABA : 1 (clock-driven)
    Erev : 1
    g_GABA : 1
    taus_GABA : second
    ''', on_pre='s_GABA += g_GABA', method='euler')

    S_ilir_GABA.connect(p = 1.0)
    S_ilir_GABA.g_GABA = gi_GABA_cross
    S_ilir_GABA.taus_GABA = tau_GABA
    S_ilir_GABA.Erev = E_i

    # I_R -> I_L (inhibitory) (GABA)
    S_iril_GABA = Synapses(neuron_groups['Inh_R'], neuron_groups['Inh_L'], model='''
    Isi_R_post = -s_GABA*(v_post-Erev) : 1 (summed)
    ds_GABA/dt = -s_GABA/taus_GABA : 1 (clock-driven)
    Erev : 1
    g_GABA : 1
    taus_GABA : second
    ''', on_pre='s_GABA += g_GABA', method='euler')

    S_iril_GABA.connect(p = 1.0)
    S_iril_GABA.g_GABA = gi_GABA_cross
    S_iril_GABA.taus_GABA = tau_GABA
    S_iril_GABA.Erev = E_i

    # I_R -> I_R (inhibitory) (GABA)
    S_irir_GABA = Synapses(neuron_groups['Inh_R'], neuron_groups['Inh_R'], model='''
    Isi_R_post = -s_GABA*(v_post-Erev) : 1 (summed)
    ds_GABA/dt = -s_GABA/taus_GABA : 1 (clock-driven)
    Erev : 1
    g_GABA : 1
    taus_GABA : second
    ''', on_pre='s_GABA += g_GABA', method='euler')

    S_irir_GABA.connect(condition='i!=j', p=1.0)  
    S_irir_GABA.g_GABA = gi_GABA_rec 
    S_irir_GABA.taus_GABA = tau_GABA
    S_irir_GABA.Erev = E_i 

    # NSE -> E_L (excitatory) (AMPA)
    S_nseel_AMPA = Synapses(neuron_groups['NSE'], neuron_groups['E_L'], model='''
    Isnse_ampa_post = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_nseel_AMPA.connect(p = 1.0)
    S_nseel_AMPA.g_AMPA = gnse_AMPA
    S_nseel_AMPA.taus_AMPA = tau_AMPA
    S_nseel_AMPA.Erev = E_e

    # NSE -> E_L (excitatory) (NMDA)
    S_nseel_NMDA = Synapses(neuron_groups['NSE'], neuron_groups['E_L'], model='''
    Isnse_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_nseel_NMDA.connect(p = 1.0)
    S_nseel_NMDA.alpha = alpha
    S_nseel_NMDA.g_NMDA = gnse_NMDA
    S_nseel_NMDA.taus_NMDA = tau_NMDA
    S_nseel_NMDA.Erev = E_e

    # NSE -> E_R (excitatory) (AMPA)
    S_nseer_AMPA = Synapses(neuron_groups['NSE'], neuron_groups['E_R'], model='''
    Isnse_ampa_post = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_nseer_AMPA.connect(p = 1.0)
    S_nseer_AMPA.g_AMPA = gnse_AMPA
    S_nseer_AMPA.taus_AMPA = tau_AMPA
    S_nseer_AMPA.Erev = E_e

    # NSE -> E_R (excitatory) (NMDA)
    S_nseer_NMDA = Synapses(neuron_groups['NSE'], neuron_groups['E_R'], model='''
    Isnse_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_nseer_NMDA.connect(p = 1.0)
    S_nseer_NMDA.alpha = alpha
    S_nseer_NMDA.g_NMDA = gnse_NMDA
    S_nseer_NMDA.taus_NMDA = tau_NMDA
    S_nseer_NMDA.Erev = E_e

    # NSE -> I_R (excitatory) (AMPA)
    S_nseir_AMPA = Synapses(neuron_groups['NSE'], neuron_groups['Inh_R'], model='''
    Isnse_ampa_post = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_nseir_AMPA.connect(p = 1.0)
    S_nseir_AMPA.g_AMPA = gnsi_AMPA
    S_nseir_AMPA.taus_AMPA = tau_AMPA
    S_nseir_AMPA.Erev = E_e

    # NSE -> I_R (excitatory) (NMDA)
    S_nseir_NMDA = Synapses(neuron_groups['NSE'], neuron_groups['Inh_R'], model='''
    Isnse_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_nseir_NMDA.connect(p = 1.0)
    S_nseir_NMDA.alpha = alpha
    S_nseir_NMDA.g_NMDA = gnsi_NMDA
    S_nseir_NMDA.taus_NMDA = tau_NMDA
    S_nseir_NMDA.Erev = E_e

    # NSE -> I_L (excitatory) (AMPA)
    S_nseil_AMPA = Synapses(neuron_groups['NSE'], neuron_groups['Inh_L'], model='''
    Isnse_ampa_post = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_nseil_AMPA.connect(p = 1.0)
    S_nseil_AMPA.g_AMPA = gnsi_AMPA
    S_nseil_AMPA.taus_AMPA = tau_AMPA
    S_nseil_AMPA.Erev = E_e

    # NSE -> I_L (excitatory) (NMDA)
    S_nseil_NMDA = Synapses(neuron_groups['NSE'], neuron_groups['Inh_L'], model='''
    Isnse_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_nseil_NMDA.connect(p = 1.0)
    S_nseil_NMDA.alpha = alpha
    S_nseil_NMDA.g_NMDA = gnsi_NMDA
    S_nseil_NMDA.taus_NMDA = tau_NMDA
    S_nseil_NMDA.Erev = E_e

    # E_L -> NSE (excitatory) (AMPA)
    S_elen_AMPA = Synapses(neuron_groups['E_L'], neuron_groups['NSE'], model='''
    IseL_ampa_post  = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_elen_AMPA.connect(p = 1.0)
    S_elen_AMPA.g_AMPA = gnse_AMPA
    S_elen_AMPA.taus_AMPA = tau_AMPA
    S_elen_AMPA.Erev = E_e

    # E_L -> NSE (excitatory) (NMDA)
    S_elen_NMDA = Synapses(neuron_groups['E_L'], neuron_groups['NSE'], model='''
    IseL_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_elen_NMDA.connect(p = 1.0)
    S_elen_NMDA.alpha = alpha
    S_elen_NMDA.g_NMDA = gnse_NMDA
    S_elen_NMDA.taus_NMDA = tau_NMDA
    S_elen_NMDA.Erev = E_e

    # E_R -> NSE (excitatory) (AMPA)
    S_eren_AMPA = Synapses(neuron_groups['E_R'], neuron_groups['NSE'], model='''
    IseR_ampa_post  = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_eren_AMPA.connect(p = 1.0)
    S_eren_AMPA.g_AMPA = gnse_AMPA
    S_eren_AMPA.taus_AMPA = tau_AMPA
    S_eren_AMPA.Erev = E_e

    # E_R -> NSE (excitatory) (NMDA)
    S_eren_NMDA = Synapses(neuron_groups['E_R'], neuron_groups['NSE'], model='''
    IseR_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_eren_NMDA.connect(p = 1.0)
    S_eren_NMDA.alpha = alpha
    S_eren_NMDA.g_NMDA = gnse_NMDA
    S_eren_NMDA.taus_NMDA = tau_NMDA
    S_eren_NMDA.Erev = E_e

    # NSE -> NSE (excitatory) (AMPA)
    S_nsen_AMPA = Synapses(neuron_groups['NSE'], neuron_groups['NSE'], model='''
    Isnse_ampa_post = -s_AMPA*(v_post-Erev) : 1 (summed)
    ds_AMPA/dt = -s_AMPA/taus_AMPA : 1 (clock-driven)
    Erev : 1
    g_AMPA : 1
    taus_AMPA : second
    ''', on_pre='s_AMPA += g_AMPA', method='euler')

    S_nsen_AMPA.connect(condition='i!=j', p=1.0)
    S_nsen_AMPA.g_AMPA = gnse_AMPA
    S_nsen_AMPA.taus_AMPA = tau_AMPA
    S_nsen_AMPA.Erev = E_e

    # NSE -> NSE (excitatory) (NMDA)
    S_nsen_NMDA = Synapses(neuron_groups['NSE'], neuron_groups['NSE'], model='''
    Isnse_nmda_post = -(s_NMDA*(v_post-Erev))/(1+0.001*exp(-0.062*v_post)*(1/3.57))  : 1 (summed)
    ds_NMDA/dt = -s_NMDA/taus_NMDA : 1 (clock-driven)
    alpha : 1 (constant)
    Erev : 1
    g_NMDA : 1
    taus_NMDA : second
    ''', on_pre='s_NMDA += g_NMDA*alpha*(1-s_NMDA)', method='euler')

    S_nsen_NMDA.connect(condition='i!=j', p=1.0)
    S_nsen_NMDA.alpha = alpha
    S_nsen_NMDA.g_NMDA = gnse_NMDA
    S_nsen_NMDA.taus_NMDA = tau_NMDA
    S_nsen_NMDA.Erev = E_e

    # I_R -> NSE (inhibitory) (GABA)
    S_irens_GABA = Synapses(neuron_groups['Inh_R'], neuron_groups['NSE'], model='''
    Isi_R_post = -s_GABA*(v_post-Erev) : 1 (summed)
    ds_GABA/dt = -s_GABA/taus_GABA : 1 (clock-driven)
    Erev : 1
    g_GABA : 1
    taus_GABA : second
    ''', on_pre='s_GABA += g_GABA', method='euler')

    S_irens_GABA.connect(p = 1.0)
    S_irens_GABA.g_GABA = gins_GABA
    S_irens_GABA.taus_GABA = tau_GABA
    S_irens_GABA.Erev = E_i

    # I_L -> NSE (inhibitory) (GABA)
    S_ilens_GABA = Synapses(neuron_groups['Inh_L'], neuron_groups['NSE'], model='''
    Isi_L_post = -s_GABA*(v_post-Erev) : 1 (summed)
    ds_GABA/dt = -s_GABA/taus_GABA : 1 (clock-driven)
    Erev : 1
    g_GABA : 1
    taus_GABA : second
    ''', on_pre='s_GABA += g_GABA', method='euler')

    S_ilens_GABA.connect(p = 1.0)
    S_ilens_GABA.g_GABA = gins_GABA
    S_ilens_GABA.taus_GABA = tau_GABA
    S_ilens_GABA.Erev = E_i

    if params['top_down']:
        ### Top-down connections ###
        # Ctr1 -> E_L (excitatory)
        S_Ctr1_E_L = Synapses(neuron_groups['Ctr1'], neuron_groups['E_L'], model = '''ds_AMPA/dt = -s_AMPA / tau_AMPA : 1 (clock-driven)
        Ibsi_1_post = - g_AMPA * s_AMPA * (v_post - E_rev) : 1 (summed)
        E_rev : 1
        g_AMPA : 1
        tau_AMPA : second
        ''', on_pre = 's_AMPA += g_AMPA', method = 'euler')

        S_Ctr1_E_L.connect(p = 1)
        S_Ctr1_E_L.g_AMPA = gctr_e_AMPA
        S_Ctr1_E_L.tau_AMPA = tau_AMPA
        S_Ctr1_E_L.E_rev = E_e

        # Ctr2 -> E_L (inhibitory)
        S_Ctr2_E_L = Synapses(neuron_groups['Ctr2'], neuron_groups['E_L'], model = '''ds_GABA/dt = -s_GABA / tau_GABA : 1 (clock-driven)
        Ibsi_2_post = -g_GABA * s_GABA * (v_post - E_rev) : 1 (summed)
        E_rev : 1
        g_GABA : 1
        tau_GABA : second
        ''', on_pre = 's_GABA += g_GABA', method = 'euler')

        S_Ctr2_E_L.connect(p = 1)
        S_Ctr2_E_L.g_GABA = gctr_i_GABA
        S_Ctr2_E_L.tau_GABA = tau_GABA
        S_Ctr2_E_L.E_rev = E_i

        # Ctr3 -> E_R (excitatory)
        S_Ctr3_E_R = Synapses(neuron_groups['Ctr3'], neuron_groups['E_R'], model = '''ds_AMPA/dt = -s_AMPA / tau_AMPA : 1 (clock-driven)
        Ibsi_3_post = -g_AMPA * s_AMPA * (v_post - E_rev) : 1 (summed)
        E_rev : 1
        g_AMPA : 1
        tau_AMPA : second
        ''', on_pre = 's_AMPA += g_AMPA', method = 'euler')

        S_Ctr3_E_R.connect(p = 1)
        S_Ctr3_E_R.g_AMPA = gctr_e_AMPA
        S_Ctr3_E_R.tau_AMPA = tau_AMPA
        S_Ctr3_E_R.E_rev = E_e

        # Ctr4 -> E_R (inhibitory)
        S_Ctr4_E_R = Synapses(neuron_groups['Ctr4'], neuron_groups['E_R'], model = '''ds_GABA/dt = -s_GABA / tau_GABA : 1 (clock-driven)
        Ibsi_4_post = -g_GABA * s_GABA * (v_post - E_rev) : 1 (summed)
        E_rev : 1
        g_GABA : 1
        tau_GABA : second
        ''', on_pre = 's_GABA += g_GABA', method = 'euler')

        S_Ctr4_E_R.connect(p = 1)
        S_Ctr4_E_R.g_GABA = gctr_i_GABA
        S_Ctr4_E_R.tau_GABA = tau_GABA
        S_Ctr4_E_R.E_rev = E_i

    poisson_input_nse = PoissonInput(target=neuron_groups['NSE'], target_var='v', N=1, rate=poisson_rate_E, weight= poisson_conductance_E)
    poisson_input_e_l = PoissonInput(target=neuron_groups['E_L'], target_var='v', N=1, rate=poisson_rate_E, weight=poisson_conductance_E)
    poisson_input_e_r = PoissonInput(target=neuron_groups['E_R'], target_var='v', N=1, rate=poisson_rate_E, weight=poisson_conductance_E)
    poisson_input_inh_l = PoissonInput(target=neuron_groups['Inh_L'], target_var='v', N=1, rate=poisson_rate_I, weight=poisson_conductance_I)
    poisson_input_inh_r = PoissonInput(target=neuron_groups['Inh_R'], target_var='v', N=1, rate=poisson_rate_I, weight=poisson_conductance_I)

    if params['top_down']:
        poisson_input_ctr1 = PoissonInput(target=neuron_groups['Ctr1'], target_var='v', N=1, rate=poisson_rate_I, weight=poisson_conductance_Ctrl_E)
        poisson_input_ctr2 = PoissonInput(target=neuron_groups['Ctr2'], target_var='v', N=1, rate=poisson_rate_I, weight=poisson_conductance_Ctrl_I)
        poisson_input_ctr3 = PoissonInput(target=neuron_groups['Ctr3'], target_var='v', N=1, rate=poisson_rate_I, weight=poisson_conductance_Ctrl_E)
        poisson_input_ctr4 = PoissonInput(target=neuron_groups['Ctr4'], target_var='v', N=1, rate=poisson_rate_I, weight=poisson_conductance_Ctrl_I)
    synapses = {
    'S_elel_AMPA': S_elel_AMPA,'S_elel_NMDA': S_elel_NMDA,'S_erer_AMPA': S_erer_AMPA,'S_erer_NMDA': S_erer_NMDA,
    'S_eler_AMPA': S_eler_AMPA,'S_eler_NMDA': S_eler_NMDA,'S_erel_AMPA': S_erel_AMPA,'S_erel_NMDA': S_erel_NMDA,
    'S_elil_AMPA': S_elil_AMPA,'S_elil_NMDA': S_elil_NMDA,'S_elir_AMPA': S_elir_AMPA,'S_elir_NMDA': S_elir_NMDA,'S_eril_AMPA': S_eril_AMPA,
    'S_eril_NMDA': S_eril_NMDA,'S_erir_AMPA': S_erir_AMPA,'S_erir_NMDA': S_erir_NMDA,'S_ilel_GABA': S_ilel_GABA,
    'S_iler_GABA': S_iler_GABA,'S_irer_GABA': S_irer_GABA,'S_irel_GABA': S_irel_GABA,'S_ilil_GABA': S_ilil_GABA,
    'S_ilir_GABA': S_ilir_GABA,'S_iril_GABA': S_iril_GABA,'S_irir_GABA': S_irir_GABA,'S_nseel_AMPA': S_nseel_AMPA,
    'S_nseel_NMDA': S_nseel_NMDA,'S_nseer_AMPA': S_nseer_AMPA,'S_nseer_NMDA': S_nseer_NMDA,'S_nseir_AMPA': S_nseir_AMPA,
    'S_nseir_NMDA': S_nseir_NMDA,'S_nseil_AMPA': S_nseil_AMPA,'S_nseil_NMDA': S_nseil_NMDA,'S_elen_AMPA': S_elen_AMPA,
    'S_elen_NMDA': S_elen_NMDA,'S_eren_AMPA': S_eren_AMPA,'S_eren_NMDA': S_eren_NMDA,'S_nsen_AMPA': S_nsen_AMPA,
    'S_nsen_NMDA': S_nsen_NMDA,'S_irens_GABA': S_irens_GABA,'S_ilens_GABA': S_ilens_GABA,
    'poisson_input_nse': poisson_input_nse,'poisson_input_e_l': poisson_input_e_l, 'poisson_input_e_r': poisson_input_e_r,
    'poisson_input_inh_l': poisson_input_inh_l,'poisson_input_inh_r': poisson_input_inh_r
    }       
    if params['top_down']:
        synapses['S_Ctr1_E_L'] = S_Ctr1_E_L
        synapses['S_Ctr2_E_L'] = S_Ctr2_E_L
        synapses['S_Ctr3_E_R'] = S_Ctr3_E_R
        synapses['S_Ctr4_E_R'] = S_Ctr4_E_R
        synapses['poisson_input_ctr1'] = poisson_input_ctr1
        synapses['poisson_input_ctr2'] = poisson_input_ctr2
        synapses['poisson_input_ctr3'] = poisson_input_ctr3
        synapses['poisson_input_ctr4'] = poisson_input_ctr4
    return synapses

def init_monitors(neuron_groups, params):
    """
    Create and return a dictionary of Brian2 Monitors for recording the simulation.
    - Here we record the membrane potential (v) and spikes of all neuron groups.
    """
    monitors = {}
    monitors['MP_L'] = StateMonitor(neuron_groups['E_L'], 'v', record=True)
    monitors['MP_R'] = StateMonitor(neuron_groups['E_R'], 'v', record=True)
    monitors['MP_I_L'] = StateMonitor(neuron_groups['Inh_L'], 'v', record=True)
    monitors['MP_I_R'] = StateMonitor(neuron_groups['Inh_R'], 'v', record=True)
    monitors['MP_NSE'] = StateMonitor(neuron_groups['NSE'], 'v', record=True)
    monitors['spike_EL'] = SpikeMonitor(neuron_groups['E_L'])
    monitors['spike_ER'] = SpikeMonitor(neuron_groups['E_R'])
    monitors['spike_IL'] = SpikeMonitor(neuron_groups['Inh_L'])
    monitors['spike_IR'] = SpikeMonitor(neuron_groups['Inh_R'])
    monitors['spike_NSE'] = SpikeMonitor(neuron_groups['NSE'])
    monitors['rate_EL'] = PopulationRateMonitor(neuron_groups['E_L'])
    monitors['rate_ER'] = PopulationRateMonitor(neuron_groups['E_R'])
    monitors['rate_IL'] = PopulationRateMonitor(neuron_groups['Inh_L'])
    monitors['rate_IR'] = PopulationRateMonitor(neuron_groups['Inh_R'])
    monitors['rate_NSE'] = PopulationRateMonitor(neuron_groups['NSE'])
    if params['top_down']:
        monitors['MP_Ctr1'] = StateMonitor(neuron_groups['Ctr1'], 'v', record=True)
        monitors['MP_Ctr2'] = StateMonitor(neuron_groups['Ctr2'], 'v', record=True)
        monitors['MP_Ctr3'] = StateMonitor(neuron_groups['Ctr3'], 'v', record=True)
        monitors['MP_Ctr4'] = StateMonitor(neuron_groups['Ctr4'], 'v', record=True)
        monitors['spike_Ctr1'] = SpikeMonitor(neuron_groups['Ctr1'])
        monitors['spike_Ctr2'] = SpikeMonitor(neuron_groups['Ctr2'])
        monitors['spike_Ctr3'] = SpikeMonitor(neuron_groups['Ctr3'])
        monitors['spike_Ctr4'] = SpikeMonitor(neuron_groups['Ctr4'])
        monitors['rate_Ctr1'] = PopulationRateMonitor(neuron_groups['Ctr1'])
        monitors['rate_Ctr2'] = PopulationRateMonitor(neuron_groups['Ctr2'])
        monitors['rate_Ctr3'] = PopulationRateMonitor(neuron_groups['Ctr3'])
        monitors['rate_Ctr4'] = PopulationRateMonitor(neuron_groups['Ctr4'])
    return monitors

def build_network(params):
    """
    Create and return a Brian2 Network object that represents the entire network.
    """
    # To avoid any potential conflicts, we start by resetting the Brian2 environment.
    start_scope()

    # 1) construct the network  
    eqs = define_equations()

    # 2) create neuron groups
    neuron_groups = create_neuron_groups(params, eqs)

    # 3) create synapses
    synapses = create_synapses(params, neuron_groups)

    # 4) create monitors
    monitors = init_monitors(neuron_groups, params)

    # 5) create network
    net = Network()
    # 1) add neuron groups
    for group in neuron_groups.values():
        net.add(group)

    # 2) add synapses
    for syn in synapses.values():
        net.add(syn)

    # 3) add monitors
    for mon in monitors.values():
        net.add(mon)

    return net, eqs, neuron_groups, synapses, monitors

def run_simulation(net, eqs, neuron_groups, monitors,
                   BG_len=200*ms, duration=1000*ms, sensory_input=0.00156):
    """
    Run the simulation and return the monitors.
    - The simulation consists of two stages: background activity (BG_len) and main simulation (duration).
    """
    restore()
    # Initialize the network
    neuron_groups['E_L'].I = 0
    neuron_groups['E_R'].I = 0
    neuron_groups['Inh_L'].I = 0
    neuron_groups['Inh_R'].I = 0
    neuron_groups['NSE'].I = 0
    # If top-down connections are enabled, set the input currents to the control neurons to 0
    if 'Ctr1' in neuron_groups:
        neuron_groups['Ctr1'].I = 0
        neuron_groups['Ctr2'].I = 0
        neuron_groups['Ctr3'].I = 0
        neuron_groups['Ctr4'].I = 0

    # Run the simulation
    net.run(BG_len)

    # Set the sensory input current
    neuron_groups['E_L'].I = sensory_input
    neuron_groups['E_R'].I = sensory_input
    
    # Run the simulation
    net.run(duration)

    # Return the monitors
    return monitors