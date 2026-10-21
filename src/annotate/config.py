# annotate/config.py
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Application defaults
# ============================================================================

DEFAULT_DATASET_PATH = r"C:\Users\ers334\Desktop\testingData\OOI\DASData\OptaSense\blue_whales_North_C3"
DEFAULT_SAVE_PATH = (
    r"C:\Users\ers334\Desktop\projects\labeling_2026\test.csv"
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

SOUND_SPEED_MPS = 1460.0  # Default sound speed in meters per second
# ============================================================================
# Default event labels
# ============================================================================

# ============================================================================
# Default event labels
# ============================================================================

@dataclass(frozen=True)
class LabelDefinition:
    """Stable annotation-label definition."""
    number: int
    name: str
    code: str


LABEL_DEFINITIONS: dict[int, LabelDefinition] = {
    0: LabelDefinition(
        number=0,
        name="Other",
        code="OTHER",
    ),
    1: LabelDefinition(
        number=1,
        name="Fin song - 20 Hz",
        code="Bp20Hz",
    ),
    2: LabelDefinition(
        number=2,
        name="Fin feeding - 40 Hz +",
        code="Bp40Hz",
    ),
    3: LabelDefinition(
        number=3,
        name="Blue song - tonal",
        code="BmTonal",
    ),
    4: LabelDefinition(
        number=4,
        name="Blue song - pulsed",
        code="BmPulsed",
    ),
    5: LabelDefinition(
        number=5,
        name="Baleen - unknown",
        code="Baleen",
    ),
    6: LabelDefinition(
        number=6,
        name="Antrhopogenic",
        code="Anthro",
    ),
    7: LabelDefinition(
        number=7,
        name="Unknown",
        code="UNK",
    ),
    9: LabelDefinition(
        number=9,
        name="Anthropogenic",
        code="ANTH",
    ),
}

# Backward-compatible mapping used by existing GUI code.
DEFAULT_LABEL_MAPPING: dict[int, str] = {
    number: definition.name
    for number, definition in LABEL_DEFINITIONS.items()
}

# Stable numeric-label to short-code mapping.
DEFAULT_LABEL_CODE_MAPPING: dict[int, str] = {
    number: definition.code
    for number, definition in LABEL_DEFINITIONS.items()
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
    duration_s: float = 40.0
    navigation_step_s: float = 25.0

    # Reader / raw DAS source settings
    interrogator: str = "auto"
    
    # Cable selection / spatial decimation
    use_full_cable_length: bool = True
    cable_start_m: float = 0
    cable_end_m: float = 70_000.0
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
    f_lo_hz: float = 5
    f_hi_hz: float = 90.0

    # F-K filtering
    fk_filter_enabled: bool = True
    c_min_mps: float = 1440.0
    c_max_mps: float = 2500.0

    # F-X settings
    fx_win_s: float = 2.0
    fx_nfft: int = 1024

    # Spectrogram settings
    spec_nfft: int = 128
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

def gray_lut() -> np.ndarray:
    """Return a uint8 lookup table for pyqtgraph images."""
    cmap = plt.get_cmap("gray")
    return (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)

PLOTCOLOR_LUT = turbo_lut()
# PLOTCOLOR_LUT = gray_lut()
    