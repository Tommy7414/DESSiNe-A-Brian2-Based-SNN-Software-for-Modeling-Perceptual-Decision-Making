# PDMSNN_DM.py

# !pip install PyQt5
# !pip install matplotlib
# !pip install brian2
# !pip install numpy
# !pip install pandas

import sys
import numpy as np
from brian2 import *
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QDialog, QFormLayout, QSpinBox, QDoubleSpinBox,
    QPushButton, QMainWindow, QApplication, QLabel, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# The function we design
from Network_set import build_network
from Plot_func import plot_data

# ==== MatplotlibWidget ====
class MatplotlibWidget(QWidget):
    """
    A custom widget for embedding Matplotlib plots within the GUI.
    
    Functionality:
    - Initializes a `FigureCanvas` for displaying plots.
    - Configures the layout to fit within the GUI.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.setMinimumSize(800, 600)

# ==== MatplotlibWidget ====
class MatplotlibWidget(QWidget):
    """
    A custom widget for embedding Matplotlib plots within the GUI.
    
    Functionality:
    - Initializes a `FigureCanvas` for displaying plots.
    - Configures the layout to fit within the GUI.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.setMinimumSize(800, 600)

# ==== RunningTrialDialog ====
class RunningTrialDialog(QDialog):
    """
    A dialog window for running simulation trials and viewing the results.

    Parameters:
    - net: The Brian2 network to simulate.
    - neuron_groups: Dictionary of neuron groups (E_L, E_R, Inh_L, Inh_R, NSE).
    - monitors: Dictionary of monitors for recording network activity.
    - params: Dictionary of simulation parameters.

    GUI Elements:
    - SpinBox inputs for background phase length, stimulus duration, sensory input strength, coherence, and threshold.
    - A "Start Running" button to run the simulation.
    - A "Reset Network" button to restore the network to its default state.
    """
    def __init__(self, parent=None, net=None, neuron_groups=None, monitors=None, params=None):
        super().__init__(parent)
        
        # Brian2 objects
        self.net = net
        self.neuron_groups = neuron_groups
        self.monitors = monitors
        self.params = params

        # GUI configuration
        self.setWindowTitle("Running Trial")
        self.setMinimumSize(1000, 800)

        # Default parameter values
        self.bg_len_val = 400  # Background phase duration (ms)
        self.duration_val = 1000  # Stimulus phase duration (ms)
        self.sensory_input_val = 0.00156  # Sensory input strength
        self.coherence_val = 0.0  # Input coherence level
        self.threshold_val = 60  # Decision threshold (Hz)

        # Create form layout for input fields
        form_layout = QFormLayout()
        self.bg_len_input = QSpinBox()
        self.bg_len_input.setMaximum(9999)
        self.bg_len_input.setValue(self.bg_len_val)
        self.bg_len_input.setSuffix(" ms")

        self.duration_input = QSpinBox()
        self.duration_input.setMaximum(9999)
        self.duration_input.setValue(self.duration_val)
        self.duration_input.setSuffix(" ms")

        self.sensory_input_input = QDoubleSpinBox()
        self.sensory_input_input.setDecimals(6)
        self.sensory_input_input.setValue(self.sensory_input_val)

        self.coherence_input = QDoubleSpinBox()
        self.coherence_input.setDecimals(6)
        self.coherence_input.setValue(self.coherence_val)

        self.threshold_input = QSpinBox()
        self.threshold_input.setMaximum(1000)
        self.threshold_input.setValue(self.threshold_val)
        self.threshold_input.setSuffix(" Hz")

        form_layout.addRow("Background Length:", self.bg_len_input)
        form_layout.addRow("Duration:", self.duration_input)
        form_layout.addRow("Sensory Input:", self.sensory_input_input)
        form_layout.addRow("Coherence:", self.coherence_input)
        form_layout.addRow("Threshold:", self.threshold_input)
        
        # Add buttons
        self.run_button = QPushButton("Start Running")
        self.run_button.clicked.connect(self.run_trial)
        self.reset_button = QPushButton("Reset Network")
        self.reset_button.clicked.connect(self.reset_network)
        
        # Add Matplotlib widget for plot display
        self.plot_widget = MatplotlibWidget(self)

        # Layout configuration
        layout_v = QVBoxLayout()
        layout_v.addLayout(form_layout)
        layout_v.addWidget(self.run_button)
        layout_v.addWidget(self.reset_button)
        layout_v.addWidget(self.plot_widget)
        self.setLayout(layout_v)

    def reset_network(self):
        """
        Restore the network to its default stored state and clear the plot.
        """
        self.net.restore('default')
        self.plot_widget.canvas.figure.clear()
        self.plot_widget.canvas.draw()
        print("Network is restored to default.")

    def run_trial(self):
        """
        Run the network simulation for the specified background and stimulus phases.
        """
        # Get input values from the GUI
        BG_len_val = self.bg_len_input.value() * ms
        max_duration_val = self.duration_input.value() * ms
        sensory_input_val = self.sensory_input_input.value()
        coherence_val = self.coherence_input.value()
        threshold = self.threshold_input.value()
        lowthreshold = threshold / 2
        chunk = 10 * ms

        # Reset network states
        self.neuron_groups['E_L'].I = 0
        self.neuron_groups['E_R'].I = 0
        self.neuron_groups['Inh_L'].I = 0
        self.neuron_groups['Inh_R'].I = 0
        self.neuron_groups['NSE'].I = 0

        time_elapsed = 0 * ms
        fig = self.plot_widget.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing Rate (Hz)')

        # Background phase
        while time_elapsed < BG_len_val:
            remaining = BG_len_val - time_elapsed
            this_step = chunk if remaining > chunk else remaining
            self.net.run(this_step)
            time_elapsed += this_step

            rate_EL = self.monitors['rate_EL'].smooth_rate(width=10*ms)/Hz
            rate_ER = self.monitors['rate_ER'].smooth_rate(width=10*ms)/Hz
            times = np.array(self.monitors['rate_EL'].t/ms)
            
            # to make sure the length of rate and time are the same
            if len(times) > len(rate_EL):
                times = times[:len(rate_EL)]
            elif len(rate_EL) > len(times):
                rate_EL = rate_EL[:len(times)]
                rate_ER = rate_ER[:len(times)]

            ax.cla()
            ax.plot(times, rate_EL, 'b-', label='E_L')
            ax.plot(times, rate_ER, 'r-', label='E_R')
            ax.set_ylim(0, 100)
            ax.legend()
            ax.set_title(f"Background running... {int(time_elapsed/ms)} / {int(BG_len_val/ms)} ms")
            ax.grid(True)
            self.plot_widget.canvas.draw()
            QApplication.processEvents()

        # Stimulus phase
        time_elapsed = 0 * ms
        decision_made = False
        
        self.neuron_groups['E_L'].I = sensory_input_val * (1 - coherence_val)
        self.neuron_groups['E_R'].I = sensory_input_val * (1 + coherence_val)

        while time_elapsed < max_duration_val and not decision_made:
            remaining = max_duration_val - time_elapsed
            this_step = chunk if remaining > chunk else remaining
            self.net.run(this_step)
            time_elapsed += this_step

            rate_EL = self.monitors['rate_EL'].smooth_rate(width=10*ms)/Hz
            rate_ER = self.monitors['rate_ER'].smooth_rate(width=10*ms)/Hz
            times = np.array(self.monitors['rate_EL'].t/ms)
            
            # to make sure the length of rate and time are the same
            if len(times) > len(rate_EL):
                times = times[:len(rate_EL)]
            elif len(rate_EL) > len(times):
                rate_EL = rate_EL[:len(times)]
                rate_ER = rate_ER[:len(times)]

            ax.cla()
            ax.plot(times, rate_EL, 'b-', label='E_L')
            ax.plot(times, rate_ER, 'r-', label='E_R')
            ax.axhline(y=threshold, color='g', linestyle='--', label='Threshold')
            ax.set_ylim(0, 100)
            ax.legend()
            ax.set_title(f"Stimulus running... {int(time_elapsed/ms)} / {int(max_duration_val/ms)} ms")
            ax.grid(True)
            self.plot_widget.canvas.draw()
            QApplication.processEvents()

            # Check thresholds
            indices_high_EL = (rate_EL >= threshold).nonzero()[0]
            indices_high_ER = (rate_ER >= threshold).nonzero()[0]

            if len(indices_high_EL) >= 200 or len(indices_high_ER) >= 200: # after decision, run for 20 ms to make sure the desicion is stable
                decision_made = True
                print(f"Threshold reached at t = {time_elapsed/ms} ms")
                self.neuron_groups['E_L'].I = 0
                self.neuron_groups['E_R'].I = 0
                
                # Post-decision phase with real-time plotting
                post_time = 0 * ms
                while post_time < 500 * ms:
                    remaining = 500*ms - post_time
                    this_step = chunk if remaining > chunk else remaining
                    self.net.run(this_step)
                    post_time += this_step

                    rate_EL = self.monitors['rate_EL'].smooth_rate(width=10*ms)/Hz
                    rate_ER = self.monitors['rate_ER'].smooth_rate(width=10*ms)/Hz
                    times = np.array(self.monitors['rate_EL'].t/ms)
                    
                    if len(times) > len(rate_EL):
                        times = times[:len(rate_EL)]
                    elif len(rate_EL) > len(times):
                        rate_EL = rate_EL[:len(times)]
                        rate_ER = rate_ER[:len(times)]

                    ax.cla()
                    ax.plot(times, rate_EL, 'b-', label='E_L')
                    ax.plot(times, rate_ER, 'r-', label='E_R')
                    ax.axhline(y=threshold, color='g', linestyle='--', label='Threshold')
                    ax.set_ylim(0, 100)
                    ax.legend()
                    ax.set_title(f"Post-decision phase... {int(post_time/ms)} / 500 ms")
                    ax.grid(True)
                    self.plot_widget.canvas.draw()
                    QApplication.processEvents()

        if not decision_made:
            print(f"No decision by the end of max_duration = {max_duration_val}")

        rate_arrayEL = self.monitors['rate_EL'].smooth_rate(width=30 * ms) / Hz
        rate_arrayER = self.monitors['rate_ER'].smooth_rate(width=30 * ms) / Hz

        indices_high_EL = (rate_arrayEL >= threshold).nonzero()[0]
        indices_high_ER = (rate_arrayER >= threshold).nonzero()[0]
        indices_low_EL  = (rate_arrayEL >= lowthreshold).nonzero()[0]
        indices_low_ER  = (rate_arrayER >= lowthreshold).nonzero()[0]

        EL_firing = False
        ER_firing = False
        NO_decision = False
        NO_decision_2_firing = False
        EL_RT = None
        ER_RT = None

        print('high EL:', len(indices_high_EL))
        print('low EL:', len(indices_low_EL))
        print('high ER:', len(indices_high_ER))
        print('low ER:', len(indices_low_ER))

        # Decision making
        if len(indices_high_EL) == 0 and len(indices_high_ER) == 0:
            NO_decision = True
            print("No decision.")
        elif len(indices_high_EL) != 0:
            # if E_L reaches threshold
            if len(indices_low_ER) == 0:
                # E_R does not reach lowthreshold
                EL_firing = True
                EL_RT = indices_high_EL[0] / 10.0
                print('EL firing', EL_RT)
            else:
                # E_R reaches lowthreshold
                LhighRlow = list(set(indices_high_EL).intersection(indices_low_ER))
                if len(LhighRlow) / len(indices_high_EL) < 0.3:
                    EL_firing = True
                    EL_RT = indices_high_EL[0] / 10.0
                    print('EL firing', EL_RT)
                else:
                    NO_decision = True
                    print("No Semi decision.")
        elif len(indices_high_ER) != 0:
            # if E_R reaches threshold
            if len(indices_low_EL) == 0:
                ER_firing = True
                ER_RT = indices_high_ER[0] / 10.0
                print('ER firing', ER_RT)
            else:
                RhighLlow = list(set(indices_high_ER).intersection(indices_low_EL))
                if len(RhighLlow) / len(indices_high_ER) < 0.3:
                    ER_firing = True
                    ER_RT = indices_high_ER[0] / 10.0
                    print('ER firing', ER_RT)
                else:
                    NO_decision = True
                    print("No Semi decision.")
        else:
            NO_decision_2_firing = True
            print("Both sides firing.")

        # Final plot update
        plot_data(
            canvas=self.plot_widget.canvas,
            monitors=self.monitors,
            n_E=self.params['n_E'],
            n_I=self.params['n_I'],
            n_NSE=self.params['n_NSE'],
            v_threshold=-0.05,
            window_size=10,
            EL_firing=EL_firing,
            ER_firing=ER_firing,
            NO_decision=NO_decision,
            NO_decision_2_firing=NO_decision_2_firing,
            EL_RT=EL_RT,
            ER_RT=ER_RT
        )

# ==== MainWindow ====
class MainWindow(QMainWindow):
    """
    The main application window for entering network parameters and launching the simulation.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parameter Input")

        # default parameters
        self.n_E_val = 60
        self.n_I_val = 50
        self.n_NSE_val = 280
        self.gamma_ei_val = 0.0
        self.gamma_ie_val = 0.0
        self.gamma_ee_val = 1.2
        self.gamma_ii_val = 0.0
        self.top_down_val = False  # if top-down
        self.S_val = 0.3
        self.R_val = 1.0

        # GUI configuration
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        form_layout = QFormLayout()
        
        self.n_E_input = QSpinBox()
        self.n_E_input.setMaximum(9999)
        self.n_E_input.setValue(self.n_E_val)

        self.n_I_input = QSpinBox()
        self.n_I_input.setMaximum(9999)
        self.n_I_input.setValue(self.n_I_val)

        self.n_NSE_input = QSpinBox()
        self.n_NSE_input.setMaximum(9999)
        self.n_NSE_input.setValue(self.n_NSE_val)

        self.gamma_ei_input = QDoubleSpinBox()
        self.gamma_ei_input.setValue(self.gamma_ei_val)

        self.gamma_ie_input = QDoubleSpinBox()
        self.gamma_ie_input.setValue(self.gamma_ie_val)

        self.gamma_ee_input = QDoubleSpinBox()
        self.gamma_ee_input.setValue(self.gamma_ee_val)

        self.gamma_ii_input = QDoubleSpinBox()
        self.gamma_ii_input.setValue(self.gamma_ii_val)

        ### if top_down_checkbox is checked, show S, R SpinBox ###
        self.top_down_checkbox = QCheckBox("Use top-down")
        self.top_down_checkbox.setChecked(self.top_down_val)
        self.top_down_checkbox.stateChanged.connect(self.on_top_down_changed)

        ### (A) S, R SpinBox ###
        self.S_input = QDoubleSpinBox()
        self.S_input.setValue(self.S_val)
        self.S_input.setDecimals(3)
        self.S_input.setVisible(self.top_down_val)

        self.R_input = QDoubleSpinBox()
        self.R_input.setValue(self.R_val)
        self.R_input.setDecimals(3)
        self.R_input.setVisible(self.top_down_val)

        form_layout.addRow("n_E:", self.n_E_input)
        form_layout.addRow("n_I:", self.n_I_input)
        form_layout.addRow("n_NSE:", self.n_NSE_input)
        form_layout.addRow("gamma_ei:", self.gamma_ei_input)
        form_layout.addRow("gamma_ie:", self.gamma_ie_input)
        form_layout.addRow("gamma_ee:", self.gamma_ee_input)
        form_layout.addRow("gamma_ii:", self.gamma_ii_input)
        form_layout.addRow("top_down:", self.top_down_checkbox)
        form_layout.addRow("S:", self.S_input)
        form_layout.addRow("R:", self.R_input)

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.update_parameters_and_run)
        
        layout_v = QVBoxLayout()
        layout_v.addLayout(form_layout)
        layout_v.addWidget(self.confirm_button)
        
        central_widget.setLayout(layout_v)

        # Brian2 objects
        self.net = None
        self.neuron_groups = None
        self.monitors = None
        self.params = None

    ### if top_down_checkbox is checked, show S, R SpinBox ###
    def on_top_down_changed(self, state):
        self.top_down_val = (state == Qt.Checked)
        self.S_input.setVisible(self.top_down_val)
        self.R_input.setVisible(self.top_down_val)

    def update_parameters_and_run(self):
        # Get input values from the GUI
        n_E = self.n_E_input.value()
        n_I = self.n_I_input.value()
        n_NSE = self.n_NSE_input.value()
        gamma_ei = self.gamma_ei_input.value()
        gamma_ie = self.gamma_ie_input.value()
        gamma_ee = self.gamma_ee_input.value()
        gamma_ii = self.gamma_ii_input.value()
        n_Ctr = 500 # default
        
        self.S_val = self.S_input.value()
        self.R_val = self.R_input.value()

        # Store the parameters in a dictionary
        self.params = {
            'n_E': n_E,
            'n_I': n_I,
            'n_NSE': n_NSE,
            'gamma_ei': gamma_ei,
            'gamma_ie': gamma_ie,
            'gamma_ee': gamma_ee,
            'gamma_ii': gamma_ii,
            'top_down': self.top_down_val,
            'n_Ctr': n_Ctr,
            'S': self.S_val,
            'R': self.R_val
        }

        # Build the network
        self.net, eqs, self.neuron_groups, self.synapses, self.monitors = build_network(self.params)
        self.net.store('default')


        # Open the dialog window for running the simulation
        dialog = RunningTrialDialog(
            self,
            net=self.net,
            neuron_groups=self.neuron_groups,
            monitors=self.monitors,
            params=self.params
        )
        dialog.exec_()

# ==== Start the application ====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        restore('default')
    except KeyError:
        print("No stored state found, building new network.")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


