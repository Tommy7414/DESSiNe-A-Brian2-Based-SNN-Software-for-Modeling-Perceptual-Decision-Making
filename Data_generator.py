# Data_generator.py
# ------------------------------------------------------------
# parameters for data_generator
# ------------------------------------------------------------
MODE = 'perf'            # 'energy' -> for single coherence; 'perf' -> for multiple coherences
ENERGY_COHERENCE = 0.0     # Only used when MODE='energy' (percentage, ex: 6.4)
COHERENCE_LIST = [0, 3.2, 6.4, 12.8, 25.6, 51.2]  # Used when MODE='perf'

NUM_TRIAL   = 50
THRESHOLD   = 60
TRIAL_LEN   = 2000
BG_LEN      = 300
INPUT_AMP   = 0.00156

# Network and top-down parameters
PARAMS = dict(
    n_E=60, n_I=50, n_NSE=280,
    gamma_ee=1.25, gamma_ei=1.25, gamma_ie=1.25, gamma_ii=0.0,
    top_down=False,   # True enables S/R/n_Ctr
    S=0.3, R=1.0, n_Ctr=250
)

OUTPUT_DIR = "data"        # Output directory for files
USE_TIMESTAMP_NAMING = True  # True: Same as GUI (add timestamp); False: Use trial_n
TRIAL_N = "0"                # Only used when USE_TIMESTAMP_NAMING=False

# ------------------------------------------------------------
# Below is the implementation: data_producer (can be imported or executed directly)
# ------------------------------------------------------------
import os
import pickle
import datetime
from typing import List, Dict, Tuple

from Functions import DataGeneration 

def _filename_base_from_params(p: Dict) -> str:
    """based on parameters, create a base filename."""
    filename_base = f"EE{p.get('gamma_ee', 0.0)}_EI{p.get('gamma_ei', 0.0)}_IE{p.get('gamma_ie', 0.0)}_II{p.get('gamma_ii', 0.0)}"
    if p.get('top_down', False):
        filename_base += f"_S{p.get('S')}_R{p.get('R')}"
    return filename_base

def _build_metadata(params: Dict, coh: float, num_trial: int, threshold: float, trial_len: float, bg_len: float, input_amp: float) -> Dict:
    """Create metadata structure consistent with GUI."""
    meta = {
        "gamma_ee": params.get('gamma_ee', 0.0),
        "gamma_ei": params.get('gamma_ei', 0.0),
        "gamma_ie": params.get('gamma_ie', 0.0),
        "gamma_ii": params.get('gamma_ii', 0.0),
        "coherence": coh,
        "num_trial": num_trial,
        "threshold": threshold,
        "trial_len": trial_len,
        "BG_len": bg_len,
        "input_amp": input_amp,
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "top_down": params.get('top_down', False)
    }
    if meta["top_down"]:
        meta["S"] = params.get('S')
        meta["R"] = params.get('R')
    return meta

def _compose_filepath(params: Dict, meta: Dict, output_dir: str, use_timestamp_naming: bool, trial_n: str) -> str:
    """Compose the output file path based on parameters and metadata."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base = _filename_base_from_params(params)
    coh  = meta["coherence"]

    if use_timestamp_naming:
        fname = f"{base}_coh{coh}_trialCount{meta['num_trial']}_{meta['timestamp']}.pkl"
    else:
        fname = f"{base}_coh{coh}_trialCount{meta['num_trial']}_trialN{trial_n}.pkl"

    return os.path.join(output_dir, fname)

def _save_pkl(filepath: str, metadata: Dict, result_list: Dict) -> None:
    with open(filepath, "wb") as f:
        pickle.dump({"metadata": metadata, "result_list": result_list}, f)

def data_producer(
    params: Dict,
    mode: str = "energy",
    energy_coherence: float = 0.0,
    coherence_list: List[float] = None,
    num_trial: int = 50,
    threshold: float = 60.0,
    trial_len: float = 2000.0,
    bg_len: float = 300.0,
    input_amp: float = 0.00156,
    output_dir: str = "data",
    use_timestamp_naming: bool = True,
    trial_n: str = "0",
    verbose: bool = True
) -> List[str]:
    """
    Generate data based on the provided parameters and save it to files.
    """
    if coherence_list is None:
        coherence_list = [0, 3.2, 6.4, 12.8, 25.6, 51.2]

    # Ensure parameters are complete: fill in necessary fields for top-down
    params = dict(params)  # 淺拷貝不污染外部
    if not params.get("top_down", False):
        params.setdefault("S", 0.0)
        params.setdefault("R", 0.0)
        params.pop("n_Ctr", None)

    saved_paths: List[str] = []

    def _one_coh_run_and_save(coh: float) -> str:
        if verbose:
            print(f"[data_producer] Running: coh={coh}, trials={num_trial} ...")
        (Result_list, *_rest) = DataGeneration(
            params=params,
            num_trial=num_trial,
            coherence=coh,
            threshold=threshold,
            trial_len=trial_len,
            BG_len=bg_len,
            input_amp=input_amp
        )
        meta = _build_metadata(params, coh, num_trial, threshold, trial_len, bg_len, input_amp)
        filepath = _compose_filepath(params, meta, output_dir, use_timestamp_naming, trial_n)
        _save_pkl(filepath, meta, Result_list)
        if verbose:
            print(f"[data_producer] Saved: {filepath}")
        return filepath

    if mode.lower() == "energy":
        saved_paths.append(_one_coh_run_and_save(float(energy_coherence)))
    elif mode.lower() == "perf":
        for coh in coherence_list:
            saved_paths.append(_one_coh_run_and_save(float(coh)))
    else:
        raise ValueError("mode must be 'energy' or 'perf'")

    return saved_paths


# ------------------------------------------------------------
# Run directly if this script is executed
# ------------------------------------------------------------
if __name__ == "__main__":
    paths = data_producer(
        params=PARAMS,
        mode=MODE,
        energy_coherence=ENERGY_COHERENCE,
        coherence_list=COHERENCE_LIST,
        num_trial=NUM_TRIAL,
        threshold=THRESHOLD,
        trial_len=TRIAL_LEN,
        bg_len=BG_LEN,
        input_amp=INPUT_AMP,
        output_dir=OUTPUT_DIR,
        use_timestamp_naming=USE_TIMESTAMP_NAMING,
        trial_n=TRIAL_N,
        verbose=True
    )
    print("\n=== Done ===")
    for p in paths:
        print(p)
