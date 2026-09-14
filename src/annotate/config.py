# annotate/config.py
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Application defaults
# ============================================================================

DEFAULT_DATASET_PATH = r"U:\projects\2022_CLOCCB_OR_S1113\Svalbard\data"
DEFAULT_SAVE_PATH = (
    r"C:\Users\ers334\Desktop\projects\labeling_2026\annotations.csv"
)

BANDPASS_PRESETS = {
    "Custom": (None, None),
    "Broadband (5–100 Hz)": (5.0, 100.0),
    "Fin Whale 20 Hz": (10, 40),
    "Fin Whale 40 Hz": (40, 80),
    "Blue whale Type A/B": (14, 20),
    "Blue whale Type C": (13, 32),
    "Blue whale Type D": (22, 130.0),
    "Blue whale Arch": (50.0, 200.0),
}

SOUND_SPEED_MPS = 1480.0  # Default sound speed in meters per second
# ============================================================================
# Default event labels
# ============================================================================

DEFAULT_LABEL_MAPPING: dict[int, str] = {
    1: "Fin - 20 Hz",
    2: "Fin - 20 Hz +",
    3: "Fin - 40 Hz",
    4: "Fin - unknown",
    5: "Blue - D call",
    6: "Blue - Downsweep",
    7: "Blue - unknown",
    8: "Whale - other",
    9: "Anthropogenic",
    0: "other"
}

# ============================================================================
# GUI / processing defaults
# ============================================================================

@dataclass
class UserSettings:
    """
    Default application settings.

    This is currently used as a convenient source of GUI defaults.
    The live current settings can remain a dictionary in DataManager until
    you are ready to refactor it into structured settings classes.
    """

    # Navigation
    start_time: str = ""
    duration_s: float = 30.0
    navigation_step_s: float = 20.0

    # Reader / raw DAS source settings
    interrogator: str = "auto"
    
    # Cable selection / spatial decimation
    use_full_cable_length: bool = False
    cable_start_m: float = 20_000.0
    cable_end_m: float = 100_000.0
    target_dx_m: float | None = 16.0

    # Temporal display resampling
    downsample_for_display: bool = True
    target_fs_hz: float | None = None
    max_display_freq_hz: float = 100.0

    # File-boundary correction
    file_boundary_correction_enabled: bool = True
    file_boundary_edge_duration_s: float = 0.25

    # Bandpass filtering
    bandpass_enabled: bool = True
    f_lo_hz: float = 10.0
    f_hi_hz: float = 90.0

    # F-K filtering
    fk_filter_enabled: bool = True
    c_min_mps: float = 1440.0
    c_max_mps: float = 2500.0

    # F-X settings
    fx_win_s: float = 2.0
    fx_nfft: int = 1024

    # Spectrogram settings
    spec_nfft: int = 256
    spec_overlap: float = 75.0

    # Display levels
    tx_vmin: float = 0.0
    tx_vmax: float = 0.4
    fx_vmin: float = 0.0
    fx_vmax: float = 0.4
    spec_vmin: float = 0.0
    spec_vmax: float = 0.4

# ============================================================================
# Plot colour map
# ============================================================================

def turbo_lut() -> np.ndarray:
    """Return a uint8 lookup table for pyqtgraph images."""
    cmap = plt.get_cmap("turbo")
    return (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)


PLOTCOLOR_LUT = turbo_lut()
    