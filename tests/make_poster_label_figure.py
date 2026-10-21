from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PyQt6.QtWidgets import QApplication

from annotate.config import UserSettings
from annotate.data_manager import DataManager
from annotate.analysis.poster_label_plot import (
    DASPlotWindow,
    plot_label_quadrant,
)


# ---------------------------------------------------------------------
# Choose source CSV and label
# ---------------------------------------------------------------------

LABEL_CSV = Path(
    r"C:\Users\ers334\Desktop\projects\labeling_2026\annotations_poster.csv"
)

# Select by saved annotation ID.
TX_ID = 2

# Display window around apex.
BEFORE_APEX_S = 2.0
AFTER_APEX_S = 40.0

# Choose which specific 2-second portion of the call becomes the
# F-X/F-K poster window.
#
# This value is relative to the loaded display window, but it is computed
# below from an offset relative to the apex.
FX_START_RELATIVE_TO_APEX_S = 12.0
FX_WINDOW_S = 2.0

# Poster spectral settings.
FX_NFFT = 2048
SPEC_NFFT = 128
SPEC_OVERLAP_PERCENT = 90.0
MAX_FREQUENCY_HZ = 100.0

# Include a small cable margin around the label distance range.
CABLE_MARGIN_M = 2_000.0

# Use the same approximate spatial selection settings you used during
# annotation, if possible.
TARGET_DX_M = 16.0


def get_label(csv_path: Path, tx_id: int) -> dict:
    """Read one label row from CSV and convert numeric fields."""
    df = pd.read_csv(csv_path)

    matches = df[df["tx_id"] == tx_id]

    if matches.empty:
        raise KeyError(f"No label with tx_id={tx_id} in {csv_path}")

    label = matches.iloc[0].to_dict()

    numeric_fields = (
        "tx_id",
        "apex_time_utc",
        "apex_dist",
        "duration",
        "distance_to_cable",
        "sound_speed_mps",
        "dist_min",
        "dist_max",
        "f_min",
        "f_max",
        "label",
    )

    for field in numeric_fields:
        if field in label and pd.notna(label[field]):
            label[field] = float(label[field])

    return label


def main() -> None:
    # DataManager subclasses QObject, so create a Qt application even though
    # this script does not show the main GUI.
    app = QApplication.instance() or QApplication([])

    label = get_label(LABEL_CSV, TX_ID)

    source_file = Path(label["source_file"]).expanduser().resolve()

    if not source_file.is_file():
        raise FileNotFoundError(
            f"Label source file does not exist:\n{source_file}"
        )

    apex_time_utc = float(label["apex_time_utc"])
    apex_distance_m = float(label["apex_dist"])

    # Use label distance range plus a visual/context margin.
    cable_start_m = max(
        0.0,
        float(label["dist_min"]) - CABLE_MARGIN_M,
    )
    cable_end_m = float(label["dist_max"]) + CABLE_MARGIN_M

    settings = asdict(UserSettings())

    settings.update({
        # Use the source interrogator explicitly if known.
        # For label 1 / Svalbard ASN:
        # "interrogator": "asn",

        # For OptaSense label 2, use:
        "interrogator": "optasense",

        "use_full_cable_length": False,
        "cable_start_m": cable_start_m,
        "cable_end_m": cable_end_m,
        "target_dx_m": TARGET_DX_M,

        # Duration is overwritten below after the dataset is opened.
        "duration_s": BEFORE_APEX_S + AFTER_APEX_S,

        # Keep the figure-generation pipeline reasonably inexpensive.
        "fk_filter_enabled": True,
        "c_min_mps": 1400.0,
        "c_max_mps": 6000.0,

        # Keep normal display processing settings.
        "bandpass_enabled": True,
        "f_lo_hz": 5.0,
        "f_hi_hz": MAX_FREQUENCY_HZ,
        
        "file_boundary_correction_enabled": False,
        "file_boundary_edge_duration_s": .5,
    })

    manager = DataManager()

    # open_raw_dataset() indexes/open the directory and initially loads the
    # selected file's start. We subsequently position the desired window.
    manager.open_raw_dataset(
        filepath=source_file,
        settings=settings,
    )

    requested_window_start = datetime.fromtimestamp(
        apex_time_utc - BEFORE_APEX_S,
        tz=timezone.utc,
    )

    manager.current_start_time = requested_window_start
    manager.user_settings["duration_s"] = (
        BEFORE_APEX_S + AFTER_APEX_S
    )

    # This is the actual DAS load for the poster window.
    manager.load_current_window(recompute_fx=False)

    window = DASPlotWindow(
        amp=manager.loaded_data["amp"],
        t_s=manager.loaded_data["t"],
        x_m=manager.loaded_data["x"],
        fs_hz=float(manager.loaded_fs_hz),
        window_start_utc=float(
            manager.loaded_data["time_stamps"][0]
        ),
    )

    # Calculate actual apex location in this loaded plot.
    apex_relative_s = (
        apex_time_utc - window.window_start_utc
    )

    # Choose one 2-second F-X / F-K window around the apex.
    fx_start_s = max(
        float(window.t_s[0]),
        apex_relative_s + FX_START_RELATIVE_TO_APEX_S,
    )

    # Do not run beyond available time data.
    latest_valid_fx_start_s = (
        float(window.t_s[-1]) - FX_WINDOW_S
    )

    fx_start_s = min(
        fx_start_s,
        latest_valid_fx_start_s,
    )

    print("Poster label:")
    print("  tx_id:", int(label["tx_id"]))
    print("  source:", source_file.name)
    print(
        "  apex UTC:",
        datetime.fromtimestamp(
            apex_time_utc,
            tz=timezone.utc,
        ).isoformat(),
    )
    print(f"  loaded window start UTC: {window.window_start_utc:.6f}")
    print(f"  apex relative to window: {apex_relative_s:.3f} s")
    print(
        f"  selected F-X/F-K window: "
        f"{fx_start_s:.3f}–{fx_start_s + FX_WINDOW_S:.3f} s"
    )
    print(
        f"  loaded distance range: "
        f"{window.x_m[0]:.1f}–{window.x_m[-1]:.1f} m"
    )
    print(f"  loaded dx: {window.dx_m:.3f} m")
    print(f"  loaded fs: {window.fs_hz:.3f} Hz")

    fig, axes, derived = plot_label_quadrant(
        window,
        label,
        fx_start_s=fx_start_s,
        fx_window_s=FX_WINDOW_S,
        fx_nfft=FX_NFFT,
        spec_nfft=SPEC_NFFT,
        spec_overlap_percent=SPEC_OVERLAP_PERCENT,
        spectrogram_distance_m=36_000.0,
        max_frequency_hz=MAX_FREQUENCY_HZ,
    )

    output_path = Path(
        f"C:\\Users\\ers334\\Documents\\presentations\\oceans2026\\poster_label_tx_{int(label['tx_id'])}.png"
    ).resolve()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    print("\nSaved:")
    print(output_path)

    plt.show()

    # Keeps linters quiet; QApplication owns event setup only here.
    _ = app


if __name__ == "__main__":
    main()