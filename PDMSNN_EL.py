# PDMSNN_EL.py

import sys
import pickle
import time
import numpy as np
import re
import datetime

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QFileDialog,
    QLabel, QCheckBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout,
    QProgressBar, QMessageBox, QWidget, QRadioButton,
    QLineEdit
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from Functions import (
    DataGeneration,
    plot_energy_landscapes_on_figure,
    plot_perf_rt_on_figure,
    load_and_merge_data_with_metadata
)


####################
# MatplotlibWidget
####################
class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def get_figure(self):
        return self.canvas.figure

    def draw(self):
        self.canvas.draw()


####################
# PlotDialog
####################
class PlotDialog(QDialog):
    """
    Dialog for plotting energy landscape or performance/RT.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Window")
        self.resize(800, 600)
        layout = QVBoxLayout()
        self.plot_widget = MatplotlibWidget(self)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def draw_energy_landscape(self, 
                              result_dict, 
                              time_window, 
                              plot_full_scale=False, 
                              EXC=True, 
                              lim=40,
                              title_text=None):
        fig = self.plot_widget.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)

        plot_energy_landscapes_on_figure(
            Result_list=result_dict,
            fig=fig, 
            ax=ax, 
            time_window=time_window, 
            plot_full_scale=plot_full_scale, 
            EXC=EXC, 
            lim=lim
        )

        if title_text:
            ax.set_title(title_text, fontsize=12)
        self.plot_widget.draw()

    def draw_perf_rt(self, 
                     coherence_list, 
                     perf_array, 
                     rt_mean_array, 
                     rt_std_array,
                     title_text=None):
        fig = self.plot_widget.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)

        plot_perf_rt_on_figure(
            coherence_list,
            perf_array,
            rt_mean_array,
            rt_std_array,
            fig, 
            ax
        )

        if title_text:
            ax.set_title(title_text, fontsize=12)
        self.plot_widget.draw()


###########################
# TimeWindowInputDialog
###########################
class TimeWindowInputDialog(QDialog):
    """
    for user to input multiple start times (in ms), separated by comma or space.
    code will parse it into list[int], then convert to [(t, t+200), ...].
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input time windows (start times in ms)")
        self.resize(300, 150)

        layout = QVBoxLayout()

        self.label = QLabel("Input multiple start times (separated by comma or space):")
        layout.addWidget(self.label)

        self.edit_line = QLineEdit()
        layout.addWidget(self.edit_line)

        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self.on_ok)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.time_window = None

    def on_ok(self):
        text = self.edit_line.text().strip()
        if not text:
            self.time_window = []
            self.accept()
            return

        import re
        tokens = re.split(r"[,\s]+", text)
        start_list = []
        for tk in tokens:
            if tk.isdigit():
                start_list.append(int(tk))
            else:
                QMessageBox.warning(self, "Warning!!!", f"Unrecognize Input: {tk}")
                return

        # Produce Time_Window: every start => (start, start+200)
        # e.g. start_list=[100,300,400] => [(100,300), (300,500), (400,600)]
        tw = []
        for s in start_list:
            tw.append((s, s+200))
        self.time_window = tw

        self.accept()


###################################################
# 0) ModeSelectionDialog: The first dialog to choose mode
###################################################
class ModeSelectionDialog(QDialog):
    mode_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mode Selection")
        self.resize(300, 200)

        layout = QVBoxLayout()

        self.radio_a = QRadioButton("Use existed data to plot")
        self.radio_b = QRadioButton("Run new trial (energy)")
        self.radio_c = QRadioButton("Run new trial (performance/RT)")

        self.radio_a.setChecked(True)

        layout.addWidget(self.radio_a)
        layout.addWidget(self.radio_b)
        layout.addWidget(self.radio_c)

        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.on_ok_clicked)
        layout.addWidget(btn_ok)

        self.setLayout(layout)

    def on_ok_clicked(self):
        if self.radio_a.isChecked():
            self.mode_selected.emit('a')
        elif self.radio_b.isChecked():
            self.mode_selected.emit('b')
        elif self.radio_c.isChecked():
            self.mode_selected.emit('c')
        self.accept()


###################################################
# (a) PlotExistingDialog: Upload existed file(s) to plot
###################################################
class PlotExistingDialog(QDialog):
    """
    - User can choose multiple files at once by QFileDialog.getOpenFileNames
    - Parse filenames to ensure all files have the same EE/EI/IE/II and coh
    - Load and merge all files, then plot energy landscape using PlotDialog
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Upload existed file to plot")
        self.resize(400, 240)
        self.file_paths = []

        layout = QVBoxLayout()
        self.label_info = QLabel("haven't selected any file yet")
        layout.addWidget(self.label_info)

        btn_open = QPushButton("choose files")
        btn_open.clicked.connect(self.on_select_files)
        layout.addWidget(btn_open)

        # 2) energy landscape (full_scale)
        btn_plot_energy = QPushButton("plot energy_landscape (full scale)")
        btn_plot_energy.clicked.connect(lambda: self.on_plot_energy_clicked(full_scale=True))
        layout.addWidget(btn_plot_energy)
    
        # 2) energy landscape time-base
        btn_plot_energy_time = QPushButton("plot energy_landscape changes by time")
        btn_plot_energy_time.clicked.connect(lambda: self.on_plot_energy_clicked(full_scale=False))
        layout.addWidget(btn_plot_energy_time)

        # 3) plot Performance
        btn_plot_perf = QPushButton("plot Performance (6 cohs required)")
        btn_plot_perf.clicked.connect(self.on_plot_perf_clicked)
        layout.addWidget(btn_plot_perf)

        self.setLayout(layout)

    def on_select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "choose multiple file",
            "",
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if files:
            self.file_paths = files
            self.label_info.setText(f"already choose {len(files)} files")
        else:
            self.label_info.setText("user canceled file dialog")

    def load_and_merge(self):
        """
        Check if all selected files have the same EE/EI/IE/II and coh,
        if not, raise error; otherwise merge all files using load_and_merge_data.
        Return merged_dict and parsed base_info tuple.
        """
        if not self.file_paths:
            QMessageBox.warning(self, "warning", "select more than one file!")
            return None, None

        from Functions import load_and_merge_data_with_metadata
        try:
            merged_dict, base_md = load_and_merge_data_with_metadata(
                file_paths=self.file_paths,
                initial_offset=0,
                step=100
            )
            return merged_dict, base_md
        except ValueError as e:
            QMessageBox.critical(self, "File Error", str(e))
            return None, None
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return None, None

    def on_plot_energy_clicked(self, full_scale):
        merged_data, base_md = self.load_and_merge()
        if merged_data is None or base_md is None:
            return
        
        gamma_ee_val = base_md["gamma_ee"]
        gamma_ei_val = base_md["gamma_ei"]
        gamma_ie_val = base_md["gamma_ie"]
        gamma_ii_val = base_md["gamma_ii"]
        coh_val      = base_md["coherence"]

        title_text = (fr'$\gamma_{{EE}}={gamma_ee_val}, \gamma_{{EI}}={gamma_ei_val}, ' 
                      fr'\gamma_{{IE}}={gamma_ie_val}, \gamma_{{II}}={gamma_ii_val}, '
                      fr'coh={coh_val}$')

        plotdlg = PlotDialog(self)
        plotdlg.setWindowTitle(title_text)

        if full_scale:
            plotdlg.draw_energy_landscape(
                result_dict=merged_data,
                time_window=[],
                plot_full_scale=True,
                EXC=True,
                lim=40,
                title_text=title_text
            )
            plotdlg.exec_()
        else:
            tw_dlg = TimeWindowInputDialog(self)
            if tw_dlg.exec_() == QDialog.Accepted:
                if tw_dlg.time_window is not None:
                    plotdlg2 = PlotDialog(self)
                    plotdlg2.setWindowTitle(title_text)
                    plotdlg2.draw_energy_landscape(
                        result_dict=merged_data,
                        time_window=tw_dlg.time_window,
                        plot_full_scale=False,
                        EXC=True,
                        lim=40,
                        title_text=title_text
                    )
                    plotdlg2.exec_()

    def on_plot_perf_clicked(self):
        if not self.file_paths:
            QMessageBox.warning(self, "warning", "you haven't selected any file!")
            return

        from Functions import load_and_merge_data_with_metadata

        group_dict = {}  # key = (gEE,gEI,gIE,gII), value = { coh_value : [file1, file2, ...] }
        for f in self.file_paths:
            try:
                with open(f, 'rb') as fp:
                    data_in_file = pickle.load(fp)
                md = data_in_file["metadata"]
                res = data_in_file["result_list"]
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error reading file {f}:\n{e}")
                return

            gkey = (md["gamma_ee"], md["gamma_ei"], md["gamma_ie"], md["gamma_ii"])
            coh_val = md["coherence"]

            if gkey not in group_dict:
                group_dict[gkey] = {}
            if coh_val not in group_dict[gkey]:
                group_dict[gkey][coh_val] = []
            group_dict[gkey][coh_val].append(f)

        if len(group_dict) > 1:
            QMessageBox.critical(self, "Error", "You selected files with different gamma(EE,EI,IE,II).")
            return

        only_gamma = list(group_dict.keys())[0]
        sub_coh_dict = group_dict[only_gamma]  # {coh => [file1, file2,...]}

        needed_coh_list = [0.0, 3.2, 6.4, 12.8, 25.6, 51.2]
        sub_coh_found = sorted(sub_coh_dict.keys())
        if len(sub_coh_found) != 6 or not all(c in sub_coh_found for c in needed_coh_list):
            QMessageBox.critical(
                self, "Error",
                f"coh found = {sub_coh_found}\nNeed exactly these 6: {needed_coh_list}"
            )
            return

        performance_array = []
        rt_mean_array = []
        rt_std_array  = []
        
        for coh_val in needed_coh_list:
            files_for_that_coh = sub_coh_dict[coh_val]
            try:
                merged_data, base_md = load_and_merge_data_with_metadata(files_for_that_coh)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                return
            
            ER_count = 0
            EL_count = 0
            ER_RT_list = []

            for k in merged_data:
                trial_data = merged_data[k]
                if trial_data is not None and len(trial_data) >= 3:
                    ER_flag = trial_data[0]
                    EL_flag = trial_data[1]
                    ER_RT   = trial_data[2]
                    ER_count += ER_flag
                    EL_count += EL_flag
                    if ER_flag == 1 and ER_RT is not None:
                        ER_RT_list.append(ER_RT)

            total_decisions = ER_count + EL_count
            if total_decisions == 0:
                perf = 0.0
            else:
                perf = ER_count / total_decisions

            if len(ER_RT_list) > 0:
                mean_rt = float(np.mean(ER_RT_list))
                std_rt  = float(np.std(ER_RT_list))
            else:
                mean_rt = 0.0
                std_rt  = 0.0

            performance_array.append(perf)
            rt_mean_array.append(mean_rt)
            rt_std_array.append(std_rt)

        (gEE, gEI, gIE, gII) = only_gamma
        title_text = (fr'$\gamma_{{EE}}={gEE}, \gamma_{{EI}}={gEI}, '
                      fr'\gamma_{{IE}}={gIE}, \gamma_{{II}}={gII}$')
        plotdlg = PlotDialog(self)
        plotdlg.draw_perf_rt(
            needed_coh_list,
            performance_array,
            rt_mean_array,
            rt_std_array,
            title_text=title_text
        )
        plotdlg.setWindowTitle("Performance Plot")
        plotdlg.exec_()


###################################################
# 1) TopDownDialog
###################################################
class TopDownDialog(QDialog):
    submitted = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Window1 - Top-Down Parameters")
        self.resize(400, 200)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.check_top_down = QCheckBox("Use top-down")
        self.spin_S = QDoubleSpinBox()
        self.spin_S.setValue(0.3)
        self.spin_S.setDecimals(3)
        self.spin_S.setSingleStep(0.1)

        self.spin_R = QDoubleSpinBox()
        self.spin_R.setValue(1.0)
        self.spin_R.setDecimals(3)
        self.spin_R.setSingleStep(0.1)

        form_layout.addRow("Top-down:", self.check_top_down)
        form_layout.addRow("S value:", self.spin_S)
        form_layout.addRow("R value:", self.spin_R)
        layout.addLayout(form_layout)

        btn_next = QPushButton("Next")
        btn_next.clicked.connect(self.on_next_clicked)
        layout.addWidget(btn_next)

        self.setLayout(layout)

    def on_next_clicked(self):
        params = {
            'top_down': self.check_top_down.isChecked(),
            'S': self.spin_S.value(),
            'R': self.spin_R.value()
        }
        self.submitted.emit(params)
        self.accept()


###################################################
# 2) ParamDialog
###################################################
class ParamDialog(QDialog):
    submitted = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Window2 - Other Parameters")
        self.resize(600, 300)
        self.init_ui()

    def init_ui(self):
        main_hlayout = QHBoxLayout()

        # Default values
        self.n_E_val = 60
        self.n_I_val = 50
        self.n_NSE_val = 280
        self.num_trial = 5
        self.gamma_ei_val = 0.0
        self.gamma_ie_val = 0.0
        self.gamma_ee_val = 1.25
        self.gamma_ii_val = 0.0
        self.bg_len_val = 200
        self.duration_val = 2000
        self.sensory_input_val = 0.00156
        self.threshold_val = 60

        # Column 1
        col1_layout = QFormLayout()
        self.spin_nE = QSpinBox()
        self.spin_nE.setMaximum(9999)
        self.spin_nE.setValue(self.n_E_val)
        col1_layout.addRow("n_E:", self.spin_nE)

        self.spin_nI = QSpinBox()
        self.spin_nI.setMaximum(999)
        self.spin_nI.setValue(self.n_I_val)
        col1_layout.addRow("n_I:", self.spin_nI)

        self.spin_nNSE = QSpinBox()
        self.spin_nNSE.setMaximum(999)
        self.spin_nNSE.setValue(self.n_NSE_val)
        col1_layout.addRow("n_NSE:", self.spin_nNSE)

        self.spin_gEE = QDoubleSpinBox()
        self.spin_gEE.setDecimals(3)
        self.spin_gEE.setValue(self.gamma_ee_val)
        col1_layout.addRow("gamma_ee:", self.spin_gEE)

        self.spin_gEI = QDoubleSpinBox()
        self.spin_gEI.setDecimals(3)
        self.spin_gEI.setValue(self.gamma_ei_val)
        col1_layout.addRow("gamma_ei:", self.spin_gEI)

        self.spin_gIE = QDoubleSpinBox()
        self.spin_gIE.setDecimals(3)
        self.spin_gIE.setValue(self.gamma_ie_val)
        col1_layout.addRow("gamma_ie:", self.spin_gIE)

        self.spin_gII = QDoubleSpinBox()
        self.spin_gII.setDecimals(3)
        self.spin_gII.setValue(self.gamma_ii_val)
        col1_layout.addRow("gamma_ii:", self.spin_gII)

        # Column 2
        col2_layout = QFormLayout()
        self.spin_num_trial = QSpinBox()
        self.spin_num_trial.setMaximum(999)
        self.spin_num_trial.setValue(self.num_trial)
        col2_layout.addRow("num_trial:", self.spin_num_trial)

        self.spin_threshold = QSpinBox()
        self.spin_threshold.setMaximum(150)
        self.spin_threshold.setValue(self.threshold_val)
        col2_layout.addRow("threshold:", self.spin_threshold)

        self.spin_trial_len = QSpinBox()
        self.spin_trial_len.setMaximum(9999)
        self.spin_trial_len.setValue(self.duration_val)
        col2_layout.addRow("trial_len (ms):", self.spin_trial_len)

        self.spin_bg_len = QSpinBox()
        self.spin_bg_len.setMaximum(999)
        self.spin_bg_len.setValue(self.bg_len_val)
        col2_layout.addRow("BG_len (ms):", self.spin_bg_len)

        self.spin_input_amp = QDoubleSpinBox()
        self.spin_input_amp.setDecimals(6)
        self.spin_input_amp.setValue(self.sensory_input_val)
        col2_layout.addRow("input_amp:", self.spin_input_amp)

        main_hlayout.addLayout(col1_layout)
        main_hlayout.addLayout(col2_layout)

        main_vlayout = QVBoxLayout()
        main_vlayout.addLayout(main_hlayout)

        btn_go_exec = QPushButton("Go to Execution")
        btn_go_exec.clicked.connect(self.on_submit_clicked)
        main_vlayout.addWidget(btn_go_exec)

        self.setLayout(main_vlayout)

    def on_submit_clicked(self):
        params = {
            'n_E': self.spin_nE.value(),
            'n_I': self.spin_nI.value(),
            'n_NSE': self.spin_nNSE.value(),
            'gamma_ee': self.spin_gEE.value(),
            'gamma_ei': self.spin_gEI.value(),
            'gamma_ie': self.spin_gIE.value(),
            'gamma_ii': self.spin_gII.value(),
        }
        other_dict = {
            'num_trial': self.spin_num_trial.value(),
            'threshold': self.spin_threshold.value(),
            'trial_len': self.spin_trial_len.value(),
            'BG_len': self.spin_bg_len.value(),
            'input_amp': self.spin_input_amp.value()
        }
        merged = {'params': params, 'others': other_dict}
        self.submitted.emit(merged)
        self.accept()


###################################################
# 3) ExecDialog
###################################################
class ExecDialog(QDialog):
    """
    Based on the mode ('b' -> energy or 'c' -> performance), run new trials.
    After running, show the plots in a new PlotDialog, and display a progress bar.
    Final, can choose save file => save as { "metadata": {...}, "result_list": {...} }.
    """
    def __init__(self, parent=None, mode='c'):
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle("Window3 - Execution")
        self.resize(800, 600)

        self.params = None
        self.top_down = False
        self.S = 0.3
        self.R = 1.0
        self.num_trial = 10
        self.threshold = 60
        self.trial_len = 2000
        self.BG_len = 300
        self.input_amp = 0.00156

        self.coherence_list = [0, 3.2, 6.4, 12.8, 25.6, 51.2]

        self.result_dict = {}  # 用來記錄 (coh -> mergedData)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.label_title = QLabel("No Title Yet")
        self.label_title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        main_layout.addWidget(self.label_title)

        h_layout = QHBoxLayout()
        self.label_info = QLabel("Ready.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        h_layout.addWidget(self.label_info)
        h_layout.addWidget(self.progress_bar)
        main_layout.addLayout(h_layout)

        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.on_run_clicked)
        main_layout.addWidget(self.btn_run)

        self.plot_widget = MatplotlibWidget(self)
        main_layout.addWidget(self.plot_widget)

        self.setLayout(main_layout)

    def set_parameters(self, merged_dict, topdown_dict):
        self.top_down = topdown_dict['top_down']
        self.S = topdown_dict['S']
        self.R = topdown_dict['R']

        self.params = merged_dict['params']
        self.params['top_down'] = self.top_down
        self.params['S'] = self.S
        self.params['R'] = self.R

        self.num_trial = merged_dict['others']['num_trial']
        self.threshold = merged_dict['others']['threshold']
        self.trial_len = merged_dict['others']['trial_len']
        self.BG_len    = merged_dict['others']['BG_len']
        self.input_amp = merged_dict['others']['input_amp']

        gamma_ee = self.params.get('gamma_ee', 0.0)
        gamma_ei = self.params.get('gamma_ei', 0.0)
        gamma_ie = self.params.get('gamma_ie', 0.0)
        gamma_ii = self.params.get('gamma_ii', 0.0)

        title_str = f"gamma_ee={gamma_ee}, gamma_ei={gamma_ei}, gamma_ie={gamma_ie}, gamma_ii={gamma_ii}"
        if self.top_down:
            title_str += f", S={self.S}, R={self.R}"

        self.label_title.setText(title_str)
        self.label_info.setText("Parameters loaded. Click 'Run' to start.")

    def on_run_clicked(self):
        """
        based on mode b or c run energy/performance
        """
        self.progress_bar.setValue(0)

        if self.mode == 'b':
            self.run_energy_mode()
        else:
            self.run_perf_mode()

    def run_energy_mode(self):
        """
        run energy mode
        """
        self.label_info.setText("Running new trials for energy landscape (trial by trial)...")
        QApplication.processEvents()

        self.progress_bar.setRange(0, self.num_trial)
        self.progress_bar.setValue(0)

        merged_result_list = {}
        offset = 0

        gamma_ee = self.params.get('gamma_ee', 0.0)
        gamma_ei = self.params.get('gamma_ei', 0.0)
        gamma_ie = self.params.get('gamma_ie', 0.0)
        gamma_ii = self.params.get('gamma_ii', 0.0)
        title_text = (fr'$\gamma_{{EE}}={gamma_ee}, \gamma_{{EI}}={gamma_ei}, '
                      fr'\gamma_{{IE}}={gamma_ie}, \gamma_{{II}}={gamma_ii}, '
                      fr'coh=0.0$')

        from Functions import DataGeneration  # 保證已 import

        for i in range(self.num_trial):
            self.label_info.setText(f"Energy: Running trial {i+1}/{self.num_trial} ...")
            self.label_info.repaint()
            QApplication.processEvents()

            (result_list,
             total_EL_RT,
             total_ER_RT,
             no_decision,
             EL_firing,
             ER_firing) = DataGeneration(
                 params=self.params,
                 num_trial=1,
                 coherence=0.0,
                 threshold=self.threshold,
                 trial_len=self.trial_len,
                 BG_len=self.BG_len,
                 input_amp=self.input_amp
             )
            if 0 in result_list:
                merged_result_list[offset] = result_list[0]
            offset += 1

            self.progress_bar.setValue(i+1)
            time.sleep(0.05)
            QApplication.processEvents()

        self.result_dict['energy'] = merged_result_list
        self.label_info.setText("Done. Plotting in new window (Full Scale)...")
        QApplication.processEvents()

        plotdlg = PlotDialog(self)
        plotdlg.draw_energy_landscape(
            result_dict=merged_result_list,
            time_window=[],
            plot_full_scale=True,
            EXC=True,
            lim=40,
            title_text=title_text
        )
        plotdlg.exec_()
        
        ans = QMessageBox.question(
            self,
            "Time-based Energy?",
            "Draw energy landscape changes by time window?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if ans == QMessageBox.Yes:
            tw_dlg = TimeWindowInputDialog(self)
            if tw_dlg.exec_() == QDialog.Accepted:
                if tw_dlg.time_window:
                    plotdlg2 = PlotDialog(self)
                    plotdlg2.draw_energy_landscape(
                        result_dict=merged_result_list,
                        time_window=tw_dlg.time_window,
                        plot_full_scale=False,
                        EXC=True,
                        lim=40,
                        title_text=title_text
                    )
                    plotdlg2.exec_()
        
        self.ask_to_save_file_energy()

    def ask_to_save_file_energy(self):
        ans = QMessageBox.question(
            self,
            "Save file?",
            "Do you want to save the energy result to a .pkl file (with metadata)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if ans == QMessageBox.Yes:
            gamma_ee = self.params.get('gamma_ee', 0.0)
            gamma_ei = self.params.get('gamma_ei', 0.0)
            gamma_ie = self.params.get('gamma_ie', 0.0)
            gamma_ii = self.params.get('gamma_ii', 0.0)
            coh = 0.0
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # result_list
            energy_dict = self.result_dict.get('energy', {})

            metadata = {
                "gamma_ee": gamma_ee,
                "gamma_ei": gamma_ei,
                "gamma_ie": gamma_ie,
                "gamma_ii": gamma_ii,
                "coherence": coh,
                "num_trial": self.num_trial,
                "threshold": self.threshold,
                "trial_len": self.trial_len,
                "BG_len": self.BG_len,
                "input_amp": self.input_amp,
                "timestamp": now_str
            }

            my_data = {
                "metadata": metadata,
                "result_list": energy_dict
            }

            default_filename = (f"EE{gamma_ee}_EI{gamma_ei}_IE{gamma_ie}_II{gamma_ii}_"
                                f"coh{coh}_trialCount{self.num_trial}_{now_str}.pkl")

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Energy File",
                default_filename,
                "Pickle Files (*.pkl);;All Files (*)"
            )
            if file_path:
                with open(file_path, 'wb') as f:
                    pickle.dump(my_data, f)
                self.label_info.setText(f"Energy data saved to: {file_path}")

    def run_perf_mode(self):
        """
        Performance mode: run multiple coherence levels, each with multiple trials.
        Run each trial, calculate performance and RT, then plot the results.
        """
        self.label_info.setText("Running new trials for performance/RT ...")
        QApplication.processEvents()

        performance_array = []
        rt_mean_array = []
        rt_std_array  = []
        self.result_dict = {}

        total_trials = len(self.coherence_list)*self.num_trial
        self.progress_bar.setRange(0, total_trials)
        current_trial = 0

        gamma_ee = self.params.get('gamma_ee', 0.0)
        gamma_ei = self.params.get('gamma_ei', 0.0)
        gamma_ie = self.params.get('gamma_ie', 0.0)
        gamma_ii = self.params.get('gamma_ii', 0.0)
        title_text = (fr'$\gamma_{{EE}}={gamma_ee}, \gamma_{{EI}}={gamma_ei}, '
                      fr'\gamma_{{IE}}={gamma_ie}, \gamma_{{II}}={gamma_ii}$')

        for i, coh in enumerate(self.coherence_list):
            coherence_result_dict = {}
            ER_count = 0
            EL_count = 0
            ER_RT_list = []

            from Functions import DataGeneration
            for t in range(self.num_trial):
                current_trial += 1
                self.progress_bar.setValue(current_trial)
                self.label_info.setText(f"coh={coh} | trial {current_trial}/{total_trials}")
                self.label_info.repaint()
                QApplication.processEvents()

                (result_list,
                 total_EL_RT,
                 total_ER_RT,
                 no_decision,
                 EL_firing,
                 ER_firing) = DataGeneration(
                    params=self.params,
                    num_trial=1,
                    coherence=coh,
                    threshold=self.threshold,
                    trial_len=self.trial_len,
                    BG_len=self.BG_len,
                    input_amp=self.input_amp
                )
                trial_data = result_list.get(0, None)
                coherence_result_dict[t] = trial_data
                if trial_data is not None:
                    ER_flag = trial_data[0]
                    EL_flag = trial_data[1]
                    ER_RT   = trial_data[2]
                    ER_count += ER_flag
                    EL_count += EL_flag
                    if ER_flag == 1 and ER_RT is not None:
                        ER_RT_list.append(ER_RT)

                time.sleep(0.01)
                QApplication.processEvents()

            self.result_dict[f"coh{coh}"] = coherence_result_dict

            total_decisions = ER_count + EL_count
            if total_decisions == 0:
                perf = 0.0
            else:
                perf = ER_count / total_decisions

            if len(ER_RT_list) > 0:
                mean_rt = float(np.mean(ER_RT_list))
                std_rt  = float(np.std(ER_RT_list))
            else:
                mean_rt = 0.0
                std_rt  = 0.0

            performance_array.append(perf)
            rt_mean_array.append(mean_rt)
            rt_std_array.append(std_rt)

        self.label_info.setText("Finished. Plotting in new window...")
        QApplication.processEvents()

        plotdlg = PlotDialog(self)
        plotdlg.draw_perf_rt(
            self.coherence_list,
            performance_array,
            rt_mean_array,
            rt_std_array,
            title_text=title_text
        )
        plotdlg.exec_()

        self.label_info.setText("Plot done.")
        self.ask_to_save_perf_files()

    def ask_to_save_perf_files(self):
        """
        Ask user if they want to save the performance/RT data to a file.
        """
        ans = QMessageBox.question(
            self,
            "Save multiple files?",
            "Do you want to save each coherence's result to a separate .pkl file (with metadata)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if ans == QMessageBox.Yes:
            gamma_ee = self.params.get('gamma_ee', 0.0)
            gamma_ei = self.params.get('gamma_ei', 0.0)
            gamma_ie = self.params.get('gamma_ie', 0.0)
            gamma_ii = self.params.get('gamma_ii', 0.0)
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            for coh in self.coherence_list:
                sub_dict = self.result_dict.get(f"coh{coh}", None)
                if sub_dict is None:
                    continue

                metadata = {
                    "gamma_ee": gamma_ee,
                    "gamma_ei": gamma_ei,
                    "gamma_ie": gamma_ie,
                    "gamma_ii": gamma_ii,
                    "coherence": coh,
                    "num_trial": self.num_trial,
                    "threshold": self.threshold,
                    "trial_len": self.trial_len,
                    "BG_len": self.BG_len,
                    "input_amp": self.input_amp,
                    "timestamp": now_str
                }
                my_data = {
                    "metadata": metadata,
                    "result_list": sub_dict
                }

                default_filename = (f"EE{gamma_ee}_EI{gamma_ei}_IE{gamma_ie}_II{gamma_ii}_"
                                    f"coh{coh}_trialCount{self.num_trial}_{now_str}.pkl")
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    f"Save File for coh={coh}",
                    default_filename,
                    "Pickle Files (*.pkl);;All Files (*)"
                )
                if file_path:
                    with open(file_path, 'wb') as f:
                        pickle.dump(my_data, f)
                    self.label_info.setText(f"Data for coh={coh} saved to: {file_path}")
                else:
                    self.label_info.setText(f"User canceled save dialog for coh={coh}")
        else:
            self.label_info.setText("User chose not to save any file.")


###################################################
# MainWindow
###################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow")
        self.resize(300, 100)

        btn_start = QPushButton("Start")
        btn_start.clicked.connect(self.start_sequence)

        central_widget = QWidget()
        vlayout = QVBoxLayout()
        vlayout.addWidget(btn_start)
        central_widget.setLayout(vlayout)
        self.setCentralWidget(central_widget)

        self.topdown_params = {}
        self.merged_params = {}

    def start_sequence(self):
        dlg_mode = ModeSelectionDialog(self)
        dlg_mode.mode_selected.connect(self.on_mode_selected)
        dlg_mode.exec_()

    def on_mode_selected(self, mode_str):
        if mode_str == 'a':
            dlg_plot = PlotExistingDialog(self)
            dlg_plot.exec_()
        elif mode_str in ('b', 'c'):
            dlg1 = TopDownDialog(self)
            dlg1.submitted.connect(lambda data_dict: self.on_topdown_submitted(data_dict, mode_str))
            dlg1.exec_()

    def on_topdown_submitted(self, data_dict, mode_str):
        self.topdown_params = data_dict
        dlg2 = ParamDialog(self)
        dlg2.submitted.connect(lambda merged: self.on_param_submitted(merged, mode_str))
        dlg2.exec_()

    def on_param_submitted(self, merged_dict, mode_str):
        self.merged_params = merged_dict
        dlg3 = ExecDialog(self, mode=mode_str)
        dlg3.set_parameters(self.merged_params, self.topdown_params)
        dlg3.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
