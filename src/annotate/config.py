import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

# -------------------------------------
#   Data Classes
# -------------------------------------
@dataclass
class UserSettings:
    start_time: str = ""

    # Display/navigation
    duration_s: float = 60.0
    navigation_step_s: float = 30.0

    # Loading / processing
    max_display_freq_hz: float = 100.0
    bandpass_enabled: bool = True
    f_lo_hz: float = 5.0
    f_hi_hz: float = 70.0

    fk_filter_enabled: bool = False
    c_min_mps: float = 1400
    c_max_mps: float = 6000

    # FX / spectrogram
    fx_win_s: float = 2.0
    nfft: int = 256
    overlap: float = 75

    # Display levels
    tx_vmin: float = 0.0
    tx_vmax: float = 0.4
    fx_vmin: float = 0.0
    fx_vmax: float = 0.4
    spec_vmin: float = 0.0
    spec_vmax: float = 0.4

# -------------------------------------
#   Default Event Labels
# -------------------------------------
# Mapping from number keys (1–9) to descriptive label strings
DEFAULT_LABEL_MAPPING = {
    1: "Bp_B",
    2: "Bp_A",
    3: "Bp_40Hz",
    4: "",
    5: "Bm_A",
    6: "Bm_B",
    7: "Bm_D",
    8: "",
    9: "noise"
}

DEFAULT_DATASET_PATH = r"F:"
DEFAULT_SAVE_PATH = r"C:\Users\ers334\Documents\databases\DAS_Annotations\A25.db"

# -------------------------------------
#   Plot color map definition(s)
# -------------------------------------

def turbo_lut():
    # turbo color scheme look-up table
    cmap = plt.get_cmap('turbo')
    return (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)

PLOTCOLOR_LUT = turbo_lut()