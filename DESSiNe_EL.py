# DESSiNe_EL.py

import os
import re
import sys
import time
import datetime
import pickle

import numpy as np
try:
    import numpy.core.multiarray  # noqa: F401
except ModuleNotFoundError:  # PyInstaller edge case
    import importlib
    sys.modules['numpy.core.multiarray'] = importlib.import_module('numpy.core._multiarray_umath')

from PyQt5.QtCore import Qt, pyqtSignal, QStandardPaths
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QFileDialog,
    QLabel, QCheckBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout,
    QProgressBar, QMessageBox, QWidget, QRadioButton, QLineEdit
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from brian2 import ms, Hz

from Functions import (
    DataGeneration,
    plot_energy_landscapes_on_figure,
    plot_perf_rt_on_figure,
    load_and_merge_data_with_metadata
)
from Network_set import build_network


def _safe_filename(name: str) -> str:
    """ Keep only safe characters in a filename."""
    return re.sub(r'[^\w\-.]+', '_', name)

def _default_save_path(default_filename: str) -> str:
    """Put a default save path under Documents."""
    base_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    if not base_dir:
        base_dir = os.path.expanduser("~/Documents")
    return os.path.join(base_dir, _safe_filename(default_filename))

####################
# MatplotlibWidget
####################
class MatplotlibWidget(QWidget):
    """ A QWidget that holds a Matplotlib figure."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def get_figure(self):
        """Return the embedded Figure so callers can draw on it."""
        return self.canvas.figure
    def draw(self):
        """Refresh the canvas on screen."""
        self.canvas.draw()


###############
# PlotDialog
###############
class PlotDialog(QDialog):
    """
    A simple window that shows one plot area. 
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
        """Draw energy landscape (full or time-window) with optional title"""
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
        """Draw performance and RT summary in one panel."""
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
    Ask user for start times; convert each to a 200-ms window.
    (for user to input multiple start times (in ms), separated by comma or space.
    code will parse it into list[int], then convert to [(t, t+200), ...].)
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
        """Parse input into a list of (start, start+200) windows and close."""
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
    """First step—choose “existing files”, “energy”, or “performance/RT”."""
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
        """Emit chosen mode ('a'|'b'|'c') and close."""
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

        # energy landscape (full_scale)
        btn_plot_energy = QPushButton("plot energy_landscape (full scale)")
        btn_plot_energy.clicked.connect(lambda: self.on_plot_energy_clicked(full_scale=True))
        layout.addWidget(btn_plot_energy)
    
        # energy landscape time-base
        btn_plot_energy_time = QPushButton("plot energy_landscape changes by time")
        btn_plot_energy_time.clicked.connect(lambda: self.on_plot_energy_clicked(full_scale=False))
        layout.addWidget(btn_plot_energy_time)

        # plot Performance
        btn_plot_perf = QPushButton("plot Performance (6 cohs required)")
        btn_plot_perf.clicked.connect(self.on_plot_perf_clicked)
        layout.addWidget(btn_plot_perf)

        self.setLayout(layout)

    @staticmethod
    def _make_signature(meta: dict):
        """
        Build a grouping key using γ’s + coherence (+ S,R if top_down):
        (gEE, gEI, gIE, gII, coherence, 'NTD')
        or
        (gEE, gEI, gIE, gII, coherence, 'TD', S, R)
        """
        base = (meta['gamma_ee'], meta['gamma_ei'],
                meta['gamma_ie'], meta['gamma_ii'],
                meta['coherence'])
        if not meta.get('top_down', False):
            return (*base, 'NTD')
        else:
            return (*base, 'TD', meta.get('S', 0), meta.get('R', 0))

    @staticmethod
    def _make_perf_signature(meta: dict):
        """Build a grouping key using γ’s (+ S,R), ignoring coherence.
           Return tuple without coherence for grouping Perf/RT"""
        base = (meta['gamma_ee'], meta['gamma_ei'],
                meta['gamma_ie'], meta['gamma_ii'])
        if not meta.get('top_down', False):
            return (*base, 'NTD')
        else:
            return (*base, 'TD', meta.get('S', 0), meta.get('R', 0)) 
         
    def on_select_files(self):
        """Let the user choose many files and store the paths."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "choose multiple file", "",
            "Pickle Files (*.pkl);;All Files (*)")
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

    def on_plot_energy_clicked(self, full_scale: bool):
        """
        Group by metadata and draw energy landscapes 
        (full-scale can overlay up to 5 groups; time-window requires one group).
        """
        if not getattr(self, 'file_paths', []):
            QMessageBox.warning(self, "Warning", "please select files first!")
            return

        param_groups = {} 
        for fp in self.file_paths:
            try:
                with open(fp, 'rb') as fh:
                    meta = pickle.load(fh)['metadata']
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Fail reading {fp}\n{e}")
                return
            sig = self._make_signature(meta)
            param_groups.setdefault(sig, []).append(fp)

        n_groups = len(param_groups)
        if n_groups > 5:
            QMessageBox.warning(
                self, "Too many",
                f"Detected {n_groups} groups of parameters, "
                "can only compare 5 groups at most.")
            return

        compare_many = False
        if n_groups == 1:
            compare_many = False
        elif 1 < n_groups <= 5:
            ans = QMessageBox.question(
                self, "Compare?",
                f"Detected {n_groups} groups of parameters, "
                "do you want to compare them on the same plot?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if ans == QMessageBox.Yes:
                compare_many = True
            else:
                QMessageBox.information(self, "Info", "cancel compare")
                return

        def _draw(time_window, plot_full_scale):
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            fig, ax = plt.subplots(figsize=(6, 4))

            for idx, (sig, files) in enumerate(param_groups.items()):
                merged, _ = load_and_merge_data_with_metadata(files)

                gEE, gEI, gIE, gII, coh = sig[:5]
                tail = sig[5:]
                if tail[0] == 'NTD':
                    label = (f"γEE={gEE},γEI={gEI},γIE={gIE},γII={gII}"
                            f"/coh={coh}")
                else:
                    _, S, R = tail
                    label = (f"γEE={gEE},γEI={gEI},γIE={gIE},γII={gII}"
                            f"/coh={coh}/S={S},R={R}")

                plot_energy_landscapes_on_figure(
                    Result_list   = merged,
                    fig           = fig,
                    ax            = ax,
                    time_window   = time_window,
                    plot_full_scale = plot_full_scale,
                    EXC           = True,
                    lim           = 40,
                    color = (colors[idx % len(colors)]
                            if plot_full_scale else None),
                    label         = label,
                    reset_ax      = (idx == 0)
                )

            ax.legend(frameon=False)
            dlg = PlotDialog(self)
            dlg.plot_widget.canvas.figure = fig
            dlg.setWindowTitle(
                "Energy landscape comparison"
                + ("" if plot_full_scale else " (time-window)")
            )
            dlg.exec_()
        if not full_scale:
            if len(param_groups) != 1:
                QMessageBox.warning(
                    self, "Only one parameter set",
                    "Plotting energy landscape changes by time only supports one parameter set.\n"
                    "Please select files with identical parameters.")
                return

            (sig, files), = param_groups.items()
            merged, _ = load_and_merge_data_with_metadata(files)

            tw_dlg = TimeWindowInputDialog(self)
            if tw_dlg.exec_() == QDialog.Accepted and tw_dlg.time_window:
                dlg = EnergyBoundaryDialog(
                    self,
                    sources=[(merged, None, None)],
                    time_window=tw_dlg.time_window,
                    plot_full_scale=False,
                    EXC=True, lim=40,
                    title="Energy landscape (time-window)"
                )
                dlg.exec_()
            return
        if full_scale:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            sources = []
            for idx, (sig, files) in enumerate(param_groups.items()):
                merged, _ = load_and_merge_data_with_metadata(files)
                gEE, gEI, gIE, gII, coh = sig[:5]
                tail = sig[5:]
                if tail[0] == 'NTD':
                    label = (f"γEE={gEE},γEI={gEI},γIE={gIE},γII={gII}/coh={coh}")
                else:
                    _, S, R = tail
                    label = (f"γEE={gEE},γEI={gEI},γIE={gIE},γII={gII}/coh={coh}/S={S},R={R}")
                color = colors[idx % len(colors)]
                sources.append((merged, label, color))

            # 單/多組都用同一個對話框；多組可 overlay，並帶 boundary 控制
            dlg = EnergyBoundaryDialog(
                self, sources=sources, time_window=[],
                plot_full_scale=True, EXC=True, lim=40,
                title="Energy landscape comparison (full scale)"
            )
            dlg.exec_()
            return

        # ---------- time-window（僅支援 1 組） ----------
        if len(param_groups) != 1:
            QMessageBox.warning(
                self, "Only one parameter set",
                "Plotting energy landscape changes by time only supports one parameter set.\n"
                "Please select files with identical parameters.")
            return

        (sig, files), = param_groups.items()
        merged, _ = load_and_merge_data_with_metadata(files)

        tw_dlg = TimeWindowInputDialog(self)
        if tw_dlg.exec_() == QDialog.Accepted and tw_dlg.time_window:
            dlg = EnergyBoundaryDialog(
                self,
                sources=[(merged, None, None)],
                time_window=tw_dlg.time_window,
                plot_full_scale=False, EXC=True, lim=40,
                title="Energy landscape (time-window)"
            )
            dlg.exec_()

    def on_plot_perf_clicked(self):
        """
        Require all six coherences per group, compute perf/RT, and plot single or overlay (max 5 groups).
        """
        if not getattr(self, 'file_paths', []):
            QMessageBox.warning(self, "Warning", "please select files first!")
            return
        
        group_dict = {}
        for fp in self.file_paths:
            try:
                with open(fp, 'rb') as fh:
                    data = pickle.load(fh)
                meta = data['metadata']
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Fail reading {fp}\n{e}")
                return

            sig = self._make_perf_signature(meta)
            coh = meta['coherence']
            group_dict.setdefault(sig, {}).setdefault(coh, []).append(fp)

        needed_coh = [0.0, 3.2, 6.4, 12.8, 25.6, 51.2]
        valid_groups = {}
        for sig, coh_map in group_dict.items():
            if all(c in coh_map and coh_map[c] for c in needed_coh):
                valid_groups[sig] = coh_map

        if not valid_groups:
            QMessageBox.warning(self, "Warning",
                "No parameter set owns the whole 6-coherence dataset.")
            return

        if len(valid_groups) > 5:
            QMessageBox.warning(self, "Too many",
                f"Detected {len(valid_groups)} parameter groups, "
                "can only compare 5 groups at most.")
            return

        compare_many = False
        if len(valid_groups) == 1:
            compare_many = False
        else:
            ans = QMessageBox.question(
                self, "Compare?",
                f"Detected {len(valid_groups)} parameter groups, "
                "do you want to compare them in the same figure?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if ans == QMessageBox.Yes:
                compare_many = True
            else:
                return

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        group_results = []      # list of dicts: {sig, perf[], rt_mean[], rt_std[], label, color}

        for idx, (sig, coh_map) in enumerate(valid_groups.items()):
            perf_arr, rt_mean_arr, rt_std_arr = [], [], []
            for coh in needed_coh:
                files = coh_map[coh]
                merged, _ = load_and_merge_data_with_metadata(files)
                # --- 計算 perf / rt ---
                ER_cnt = EL_cnt = 0
                RT_list = []
                for trial in merged.values():
                    if trial is None or len(trial) < 3:
                        continue
                    ER_flag, EL_flag, ER_RT = trial[0], trial[1], trial[2]
                    ER_cnt += ER_flag
                    EL_cnt += EL_flag
                    if ER_flag and ER_RT is not None:
                        RT_list.append(ER_RT)
                total_dec = ER_cnt + EL_cnt
                perf = ER_cnt / total_dec if total_dec else 0.0
                rt_mean = float(np.mean(RT_list)) if RT_list else 0.0
                rt_std  = float(np.std (RT_list)) if RT_list else 0.0
                perf_arr.append(perf); rt_mean_arr.append(rt_mean); rt_std_arr.append(rt_std)

            # ----- label -----
            gEE, gEI, gIE, gII = sig[:4]
            tail = sig[4:]
            if tail[0] == 'NTD':
                label = f"γEE={gEE},γEI={gEI},γIE={gIE},γII={gII}"
            else:
                _, S, R = tail
                label = (f"γEE={gEE},γEI={gEI},γIE={gIE},γII={gII}/S={S},R={R}")

            group_results.append({
                'sig'      : sig,
                'perf'     : perf_arr,
                'rt_mean'  : rt_mean_arr,
                'rt_std'   : rt_std_arr,
                'label'    : label,
                'color'    : colors[idx % len(colors)]
            })

        if not compare_many: 
            g = group_results[0]
            dlg = PlotDialog(self)
            dlg.draw_perf_rt(
                needed_coh,
                g['perf'],
                g['rt_mean'],
                g['rt_std'],
                title_text=g['label']
            )
            dlg.setWindowTitle("Performance / RT")
            dlg.exec_()
            return

        fig, (ax_perf, ax_rt) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

        for g in group_results:
            ax_perf.plot(needed_coh, g['perf'], '-o',
                        color=g['color'], label=g['label'])
        ax_perf.set_ylabel('Performance')
        ax_perf.set_ylim(0, 1.05)
        ax_perf.grid(True, ls='--', alpha=.3)
        ax_perf.legend(frameon=False, fontsize=8, ncol=1)

        for g in group_results:
            ax_rt.errorbar(needed_coh, g['rt_mean'], yerr=g['rt_std'],
                        fmt='s--', capsize=4, color=g['color'],
                        label=g['label'])
        ax_rt.set_ylabel('Reaction Time (ms)')
        ax_rt.set_xlabel('Coherence (%)')
        ax_rt.grid(True, ls='--', alpha=.3)

        fig.tight_layout()
        dlg = PlotDialog(self)
        dlg.plot_widget.canvas.figure = fig
        dlg.setWindowTitle("Performance / RT comparison")
        dlg.exec_()

###################################################
# 1) TopDownDialog
###################################################
class TopDownDialog(QDialog):
    """Collect top-down setting: on/off, S, R (and n_Ctr when on)."""
    submitted = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Window1 - Top-Down Parameters")
        self.resize(400, 200)
        self.init_ui()

    def init_ui(self):
        """Build the form; show/hide n_Ctr with the checkbox."""
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.check_top_down = QCheckBox("Use top-down")
        self.check_top_down.setChecked(False)
        form_layout.addRow("Top-down:", self.check_top_down)

        self.spin_S = QDoubleSpinBox()
        self.spin_S.setDecimals(3); self.spin_S.setSingleStep(0.05)
        self.spin_S.setValue(0.3)
        form_layout.addRow("S value:", self.spin_S)

        self.spin_R = QDoubleSpinBox()
        self.spin_R.setDecimals(3); self.spin_R.setSingleStep(0.05)
        self.spin_R.setValue(1.0)
        form_layout.addRow("R value:", self.spin_R)

        self.lbl_nCtr = QLabel("n_Ctr:")
        self.spin_nCtr = QSpinBox()
        self.spin_nCtr.setRange(1, 500)
        self.spin_nCtr.setValue(250)

        self.row_nCtr = QWidget()
        hl = QHBoxLayout(self.row_nCtr)
        hl.setContentsMargins(0,0,0,0)
        hl.addWidget(self.lbl_nCtr)
        hl.addWidget(self.spin_nCtr)
        form_layout.addRow(self.row_nCtr)
        self.row_nCtr.setVisible(False)

        self.check_top_down.toggled.connect(self.row_nCtr.setVisible)

        layout.addLayout(form_layout)

        btn_next = QPushButton("Next")
        btn_next.clicked.connect(self.on_next_clicked)
        layout.addWidget(btn_next)

        self.setLayout(layout)

    def on_next_clicked(self):
        """Emit a dict with top_down, S, R (and n_Ctr if used).  """
        params = {
            'top_down': self.check_top_down.isChecked(),
            'S': self.spin_S.value(),
            'R': self.spin_R.value()
        }
        if params['top_down']:
            params['n_Ctr'] = self.spin_nCtr.value()
        self.submitted.emit(params)
        self.accept()


###################################################
# 2) ParamDialog
###################################################
class ParamDialog(QDialog):
    """Collect network (γ’s, sizes) and trial settings."""
    submitted = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Window2 - Other Parameters")
        self.resize(600, 300)
        self.init_ui()

    def init_ui(self):
        """Build the form with defaults for γ’s, counts, and trial params."""
        main_hlayout = QHBoxLayout()

        self.n_E_val = 60
        self.n_I_val = 50
        self.n_NSE_val = 280
        self.n_Ctr_val = 250
        self.num_trial = 5
        self.gamma_ei_val = 0.0
        self.gamma_ie_val = 0.0
        self.gamma_ee_val = 1.25
        self.gamma_ii_val = 0.0
        self.bg_len_val = 200
        self.duration_val = 2000
        self.sensory_input_val = 0.00156
        self.threshold_val = 60
        self.top_down_val = False
        self.S_val = 0.3
        self.R_val = 1.0

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
        """Emit {'params': {...}, 'others': {...}} and close."""
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
    Run new trials (energy or performance/RT), show plots, and save .pkl
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

        self.result_dict = {}

        self.init_ui()

    def init_ui(self):
        """Build the run screen (title, progress, optional coherence field, plot)."""
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

        if self.mode == 'b':
            coh_layout = QHBoxLayout()
            lbl_coh = QLabel("Energy-mode coherence(%):")
            self.spin_coh = QDoubleSpinBox()
            self.spin_coh.setDecimals(2)
            self.spin_coh.setSingleStep(0.1)
            self.spin_coh.setRange(0.0, 100.0)
            self.spin_coh.setValue(0.0)
            coh_layout.addWidget(lbl_coh)
            coh_layout.addWidget(self.spin_coh)
            main_layout.addLayout(coh_layout)
        else:
            self.spin_coh = None

        self.plot_widget = MatplotlibWidget(self)
        main_layout.addWidget(self.plot_widget)

        self.setLayout(main_layout)

    def set_parameters(self, merged_dict, topdown_dict):
        """Combine params + top-down, build network, set window title."""
        self.top_down = topdown_dict['top_down']
        self.S        = topdown_dict['S']
        self.R        = topdown_dict['R']

        self.params = merged_dict['params']
        self.params.update({
            'top_down': self.top_down,
            'S'       : self.S,
            'R'       : self.R
        })

        if self.top_down:
            self.params['n_Ctr'] = topdown_dict.get('n_Ctr', 250)
        else:
            self.params.pop('n_Ctr', None)

        self.num_trial = merged_dict['others']['num_trial']
        self.threshold = merged_dict['others']['threshold']
        self.trial_len = merged_dict['others']['trial_len']
        self.BG_len    = merged_dict['others']['BG_len']
        self.input_amp = merged_dict['others']['input_amp']

        gEE = self.params.get('gamma_ee', 0.0)
        gEI = self.params.get('gamma_ei', 0.0)
        gIE = self.params.get('gamma_ie', 0.0)
        gII = self.params.get('gamma_ii', 0.0)

        title_str = f"gamma_ee={gEE}, gamma_ei={gEI}, gamma_ie={gIE}, gamma_ii={gII}"
        if self.top_down:
            title_str += f", S={self.S}, R={self.R}, n_Ctr={self.params['n_Ctr']}"
        self.label_title.setText(title_str)
        self.label_info.setText("Parameters loaded. Click 'Run' to start.")
        from Network_set import build_network
        self.net, eqs, self.neuron_groups, self.synapses, self.monitors = \
            build_network(self.params)
        self.net.store('default') 
    def on_run_clicked(self):
        """
        Branch to energy or performance run.
        """
        self.progress_bar.setValue(0)

        if self.mode == 'b':
            self.run_energy_mode()
        else:
            self.run_perf_mode()

    def run_energy_mode(self):
        """
        Run N trials at one coherence, 
        plot full-scale EL, optional time-window plot, then ask to save a single file.
        """
        coh_val = float(self.spin_coh.value())

        self.label_info.setText(f"Running trials (coh={coh_val}) …")
        QApplication.processEvents()

        self.progress_bar.setRange(0, self.num_trial)
        self.progress_bar.setValue(0)

        merged_result_list = {}
        offset = 0

        gEE = self.params.get('gamma_ee', 0.0)
        gEI = self.params.get('gamma_ei', 0.0)
        gIE = self.params.get('gamma_ie', 0.0)
        gII = self.params.get('gamma_ii', 0.0)
        title_text = (fr'$\gamma_{{EE}}={gEE}, \gamma_{{EI}}={gEI}, '
                      fr'\gamma_{{IE}}={gIE}, \gamma_{{II}}={gII}, '
                      fr'coh={coh_val}$')

        for i in range(self.num_trial):
            self.label_info.setText(f"Energy: trial {i+1}/{self.num_trial}")
            QApplication.processEvents()

            trial_data = self.run_single_trial_live(coh_val, i)

            merged_result_list[offset] = trial_data
            offset += 1

            self.progress_bar.setValue(i+1)
            QApplication.processEvents()

        # ---------- trial 全跑完，後面繪 energy landscape ----------
        self.result_dict['energy'] = merged_result_list
        self.label_info.setText("Done. Plotting …")
        QApplication.processEvents()

        plotdlg = PlotDialog(self)
        plotdlg.draw_energy_landscape(
            result_dict=merged_result_list,
            time_window=[],
            plot_full_scale=True,
            EXC=True,
            lim=40,
            title_text=title_text)
        plotdlg.exec_()

        if QMessageBox.question(
                self, "Time-based Energy?",
                "Draw energy landscape changes by time window?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            tw_dlg = TimeWindowInputDialog(self)
            if tw_dlg.exec_() == QDialog.Accepted and tw_dlg.time_window:
                dlg_tw = EnergyBoundaryDialog(
                    self,
                    sources=[(merged_result_list, None, None)],
                    time_window=tw_dlg.time_window,
                    plot_full_scale=False, EXC=True, lim=40,
                    title=f"Energy landscape (time-window)"
                )
                dlg_tw.exec_()

        self.ask_to_save_file_energy(coh_val)

    def ask_to_save_file_energy(self, coh_val):
        """
        Build metadata + filename and write one .pkl for that coherence.
        """
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
            coh = coh_val
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            energy_dict = self.result_dict.get('energy', {})

            metadata = {
                "gamma_ee": gamma_ee, "gamma_ei": gamma_ei, "gamma_ie": gamma_ie, "gamma_ii": gamma_ii,
                "coherence": coh, "num_trial": self.num_trial, "threshold": self.threshold,
                "trial_len": self.trial_len, "BG_len": self.BG_len, "input_amp": self.input_amp,
                "timestamp": now_str,
                "top_down": self.top_down
            }

            filename_base = f"EE{gamma_ee}_EI{gamma_ei}_IE{gamma_ie}_II{gamma_ii}"

            if self.top_down:
                metadata['S'] = self.S
                metadata['R'] = self.R
                filename_base += f"_S{self.S}_R{self.R}"
            
            my_data = {
                "metadata": metadata,
                "result_list": energy_dict
            }
            default_filename = (f"{filename_base}_coh{coh}_"
                                f"trialCount{self.num_trial}_{now_str}.pkl")

            default_path = _default_save_path(default_filename)
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Energy File",
                default_path,
                "Pickle Files (*.pkl);;All Files (*)"
            )
            if file_path and not file_path.lower().endswith(".pkl"):
                file_path += ".pkl"
            if file_path:
                with open(file_path, 'wb') as f:
                    pickle.dump(my_data, f)
                self.label_info.setText(f"Energy data saved to: {file_path}")

    def run_perf_mode(self):
        """Sweep six coherences, compute perf/RT, plot, then ask to save six files."""
        self.label_info.setText("Running new trials for performance/RT ...")
        QApplication.processEvents()

        performance_array, rt_mean_array, rt_std_array = [], [], []
        self.result_dict = {}

        total_trials = len(self.coherence_list) * self.num_trial
        self.progress_bar.setRange(0, total_trials)
        current_trial = 0

        for coh in self.coherence_list:
            ER_count = EL_count = 0
            ER_RT_list = []
            coherence_result_dict = {}

            for t in range(self.num_trial):
                current_trial += 1
                self.progress_bar.setValue(current_trial)
                self.label_info.setText(f"coh={coh} | trial {current_trial}/{total_trials}")
                QApplication.processEvents()

                trial_data = self.run_single_trial_live(coh, t)
                coherence_result_dict[t] = trial_data

                if trial_data:
                    ER_flag, EL_flag, ER_RT = trial_data[0], trial_data[1], trial_data[2]
                    ER_count += ER_flag
                    EL_count += EL_flag
                    if ER_flag and ER_RT is not None:
                        ER_RT_list.append(ER_RT)

                time.sleep(0.01)

            self.result_dict[f"coh{coh}"] = coherence_result_dict
            total_dec = ER_count + EL_count
            performance_array.append(0 if total_dec == 0 else ER_count / total_dec)
            rt_mean_array.append(float(np.mean(ER_RT_list)) if ER_RT_list else 0.0)
            rt_std_array.append(float(np.std (ER_RT_list)) if ER_RT_list else 0.0)

        self.label_info.setText("Finished. Plotting summary …")
        QApplication.processEvents()

        dlg = PlotDialog(self)
        dlg.draw_perf_rt(
            self.coherence_list, performance_array,
            rt_mean_array, rt_std_array,
            title_text=self.label_title.text()
        )
        dlg.setWindowTitle("Performance / RT")
        dlg.exec_()

        self.label_info.setText("Plot done.")
        self.ask_to_save_perf_files()


    def ask_to_save_perf_files(self):
        """Save one .pkl per coherence with shared metadata."""
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

                # --- 【修正後的邏輯】 ---
                # 1. 建立基本的 metadata
                metadata = {
                    "gamma_ee": gamma_ee, "gamma_ei": gamma_ei, "gamma_ie": gamma_ie, "gamma_ii": gamma_ii,
                    "coherence": coh, "num_trial": self.num_trial, "threshold": self.threshold,
                    "trial_len": self.trial_len, "BG_len": self.BG_len, "input_amp": self.input_amp,
                    "timestamp": now_str,
                    "top_down": self.top_down # 總是記錄 top_down 狀態
                }

                # 2. 建立基本的檔名
                filename_base = f"EE{gamma_ee}_EI{gamma_ei}_IE{gamma_ie}_II{gamma_ii}"

                # 3. 如果 top_down 為 True，才加入 S 和 R
                if self.top_down:
                    metadata['S'] = self.S
                    metadata['R'] = self.R
                    filename_base += f"_S{self.S}_R{self.R}"
                
                # 4. 組合最終的檔名和資料
                my_data = {
                    "metadata": metadata,
                    "result_list": sub_dict
                }
                default_filename = (f"{filename_base}_coh{coh}_"
                                    f"trialCount{self.num_trial}_{now_str}.pkl")

                default_path = _default_save_path(default_filename)
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Energy File",
                    default_path,  # ← 給完整路徑
                    "Pickle Files (*.pkl);;All Files (*)"
                )
                if file_path and not file_path.lower().endswith(".pkl"):
                    file_path += ".pkl"
                if file_path:
                    with open(file_path, 'wb') as f:
                        pickle.dump(my_data, f)
                    self.label_info.setText(f"Data for coh={coh} saved to: {file_path}")
                else:
                    self.label_info.setText(f"User canceled save dialog for coh={coh}")
        else:
            self.label_info.setText("User chose not to save any file.")

    def run_single_trial_live(self, coh_val: float, trial_idx: int):
        """
        Run BG → Stimulus → Post phases, 
        compute decision flags/RTs with 30-ms smoothing + overlap rule, 
        return trial_data (or None if no decision).
        """
        self.net.restore('default')
        chunk        = 10*ms
        threshold    = self.threshold
        lowthreshold = threshold/2

        for g in ('E_L', 'E_R', 'Inh_L', 'Inh_R', 'NSE'):
            self.neuron_groups[g].I = 0

        ax = self.plot_widget.get_figure().gca()
        ax.clear()
        ax.set_ylim(0, 100)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing Rate (Hz)')

        #######################
        # 1. BACKGROUND PHASE
        #######################
        BG_len = self.BG_len * ms
        elapsed = 0*ms
        while elapsed < BG_len:
            step = min(chunk, BG_len - elapsed)
            self.net.run(step); elapsed += step

            rateEL = self.monitors['rate_EL'].smooth_rate(width=10*ms)/Hz
            rateER = self.monitors['rate_ER'].smooth_rate(width=10*ms)/Hz
            t      = self.monitors['rate_EL'].t / ms
            if len(t) > len(rateEL):
                t = t[:len(rateEL)]
            elif len(rateEL) > len(t):
                rateEL = rateEL[:len(t)]
                rateER = rateER[:len(t)]

            ax.cla()
            ax.plot(t, rateEL, 'b-', label='E_L')
            ax.plot(t, rateER, 'r-', label='E_R')
            ax.set_ylim(0, 100)
            ax.set_title(f'BG {int(elapsed/ms)}/{self.BG_len} ms')
            ax.legend(); ax.grid(True)
            self.plot_widget.draw()
            QApplication.processEvents()

        #######################
        # 2. STIMULUS PHASE
        #######################
        max_dur = self.trial_len * ms
        elapsed = 0*ms
        decision_made = False

        amp = self.input_amp
        self.neuron_groups['E_L'].I = amp * (1 - coh_val/100)
        self.neuron_groups['E_R'].I = amp * (1 + coh_val/100)

        while elapsed < max_dur and not decision_made:
            step = min(chunk, max_dur - elapsed)
            self.net.run(step); elapsed += step

            rateEL = self.monitors['rate_EL'].smooth_rate(width=10*ms)/Hz
            rateER = self.monitors['rate_ER'].smooth_rate(width=10*ms)/Hz
            t      = self.monitors['rate_EL'].t / ms
            if len(t) > len(rateEL):
                t = t[:len(rateEL)]
            elif len(rateEL) > len(t):
                rateEL = rateEL[:len(t)]
                rateER = rateER[:len(t)]
            ax.cla()
            ax.plot(t, rateEL, 'b-', label='E_L')
            ax.plot(t, rateER, 'r-', label='E_R')
            ax.axhline(y=threshold, color='g', linestyle='--', label='thr')
            ax.set_ylim(0, 100)
            ax.set_title(f'Stimulus {int(elapsed/ms)}/{self.trial_len} ms')
            ax.legend(); ax.grid(True)
            self.plot_widget.draw()
            QApplication.processEvents()

            high_EL = np.where(rateEL >= threshold)[0]
            high_ER = np.where(rateER >= threshold)[0]

            if len(high_EL) >= 200 or len(high_ER) >= 200:
                decision_made = True
                self.neuron_groups['E_L'].I = 0
                self.neuron_groups['E_R'].I = 0

        #######################
        # 3. POST-DECISION STABILISATION
        #######################
        if decision_made:
            post = 500*ms
            passed = 0*ms
            while passed < post:
                step = min(chunk, post - passed)
                self.net.run(step); passed += step

                rateEL = self.monitors['rate_EL'].smooth_rate(width=10*ms)/Hz
                rateER = self.monitors['rate_ER'].smooth_rate(width=10*ms)/Hz
                t      = self.monitors['rate_EL'].t / ms

                ax.cla()
                ax.plot(t, rateEL, 'b-', label='E_L')
                ax.plot(t, rateER, 'r-', label='E_R')
                ax.axhline(y=threshold, color='g', linestyle='--')
                ax.set_ylim(0, 100)
                ax.set_title(f'Post-decision {int(passed/ms)}/500 ms')
                ax.legend(); ax.grid(True)
                self.plot_widget.draw()
                QApplication.processEvents()

        #######################
        # 4. DECISION ANALYSIS
        #######################
        rate30_EL = self.monitors['rate_EL'].smooth_rate(width=30*ms)/Hz
        rate30_ER = self.monitors['rate_ER'].smooth_rate(width=30*ms)/Hz

        idx_high_EL = np.where(rate30_EL >= threshold)[0]
        idx_high_ER = np.where(rate30_ER >= threshold)[0]
        idx_low_EL  = np.where(rate30_EL >= lowthreshold)[0]
        idx_low_ER  = np.where(rate30_ER >= lowthreshold)[0]

        ER_flag = EL_flag = 0
        ER_RT = EL_RT = None
        NO_decision = NO_decision_2 = 0

        if len(idx_high_EL)==0 and len(idx_high_ER)==0:
            NO_decision = 1
        elif len(idx_high_EL):
            overlap = set(idx_high_EL).intersection(idx_low_ER)
            if len(idx_low_ER)==0 or len(overlap)/len(idx_high_EL) < 0.3:
                EL_flag = 1
                EL_RT   = idx_high_EL[0]/10.0
            else:
                NO_decision = 1
        elif len(idx_high_ER):
            overlap = set(idx_high_ER).intersection(idx_low_EL)
            if len(idx_low_EL)==0 or len(overlap)/len(idx_high_ER) < 0.3:
                ER_flag = 1
                ER_RT   = idx_high_ER[0]/10.0
            else:
                NO_decision = 1
        else:
            NO_decision_2 = 1     # 雙邊同時爆

        #######################
        # 5. merge trial_data（DataGeneration has similar structure）
        #######################
        diff_arr = rate30_ER - rate30_EL
        trial_data = [
            ER_flag, EL_flag, ER_RT, EL_RT,
            list(self.monitors['rate_EL'].t/ms),
            list(diff_arr),
            list(rate30_ER), list(rate30_EL),
            list(self.monitors['rate_IL'].smooth_rate(width=20*ms)/Hz),
            list(self.monitors['rate_IR'].smooth_rate(width=20*ms)/Hz)
        ]
        if NO_decision or NO_decision_2:
            trial_data = None

        return trial_data
###################################################
# MainWindow
###################################################
class MainWindow(QMainWindow):
    """Entry point; chains the three dialogs to run or plot."""
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
        """Open mode selection."""
        dlg_mode = ModeSelectionDialog(self)
        dlg_mode.mode_selected.connect(self.on_mode_selected)
        dlg_mode.exec_()

    def on_mode_selected(self, mode_str):
        """Route to plotting or the run sequence."""
        if mode_str == 'a':
            dlg_plot = PlotExistingDialog(self)
            dlg_plot.exec_()
        elif mode_str in ('b', 'c'):
            dlg1 = TopDownDialog(self)
            dlg1.submitted.connect(lambda data_dict: self.on_topdown_submitted(data_dict, mode_str))
            dlg1.exec_()

    def on_topdown_submitted(self, data_dict, mode_str):
        """Store top-down settings and proceed."""
        self.topdown_params = data_dict
        dlg2 = ParamDialog(self)
        dlg2.submitted.connect(lambda merged: self.on_param_submitted(merged, mode_str))
        dlg2.exec_()

    def on_param_submitted(self, merged_dict, mode_str):
        """Pass settings into ExecDialog and start."""
        self.merged_params = merged_dict
        dlg3 = ExecDialog(self, mode=mode_str)
        dlg3.set_parameters(self.merged_params, self.topdown_params)
        dlg3.exec_()

class EnergyBoundaryDialog(QDialog):
    """
    Show energy landscape with a Boundary (B) control.
    sources: list of tuples (merged_dict, label, color)
    """
    def __init__(self, parent, sources, time_window, plot_full_scale,
                 EXC=True, lim=40, title="Energy landscape"):
        super().__init__(parent)
        self.sources = sources
        self.time_window = time_window
        self.plot_full_scale = plot_full_scale
        self.EXC = EXC
        self.lim = lim

        self.setWindowTitle(title)
        self.resize(900, 650)

        v = QVBoxLayout(self)
        # plot
        self.plot_widget = MatplotlibWidget(self)
        v.addWidget(self.plot_widget)

        # controls
        row = QHBoxLayout()
        row.addWidget(QLabel("Boundary value B (integrate from x = -B):"))
        self.spinB = QDoubleSpinBox()
        self.spinB.setDecimals(1)
        self.spinB.setRange(0.0, float(lim))  # B ∈ [0, lim]
        self.spinB.setValue(0.0)
        row.addWidget(self.spinB)
        self.btn = QPushButton("Regenerate")
        self.btn.clicked.connect(self.redraw)
        row.addWidget(self.btn)
        row.addStretch(1)
        v.addLayout(row)

        self.redraw()

    def redraw(self):
        """Redraw the energy landscape with the current boundary value."""
        B = self.spinB.value()
        fig = self.plot_widget.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)

        for idx, (merged, label, color) in enumerate(self.sources):
            plot_energy_landscapes_on_figure(
                Result_list=merged, fig=fig, ax=ax,
                time_window=self.time_window,
                plot_full_scale=self.plot_full_scale,
                EXC=self.EXC, lim=self.lim,
                color=color, label=label, reset_ax=(idx == 0),
                boundary_value=(None if B <= 0 else float(B))
            )
        ax.set_title(self.windowTitle())
        self.plot_widget.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
