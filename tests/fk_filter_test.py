from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from annotate.data.dataset_service import DatasetService
from annotate.data.processing import fk_filter


FILEPATH = Path(
    r"C:\Users\ers334\Desktop\testingData\Svalbard\data\120057.hdf5"
)

INTERROGATOR = "asn"

# Keep the test small at first.
# DAS4Whales convention:
# (start_channel, end_channel, channel_stride)
SELECTED_CHANNELS = (0, 10_000, 4)

# This ASN file is approximately 10 s.
WINDOW_DURATION_S = 10.0


def main():
    service = DatasetService()

    record = service.open_dataset(
        selected_file=FILEPATH,
        requested_interrogator=INTERROGATOR,
    )

    segment = service.load_window(
        start_time=record.start_time,
        duration_s=WINDOW_DURATION_S,
        selected_channels=SELECTED_CHANNELS,
        correct_file_boundaries=False,
    )

    amp = segment.data
    fs = segment.fs_hz

    # Important: derive actual dx from returned distance coordinates.
    # This should include the DAS4Whales selected channel stride.
    dx_m = float(np.median(np.diff(segment.distance_m)))

    print("Loaded segment")
    print("  data shape:", amp.shape)
    print("  fs:", fs)
    print("  dx from segment:", segment.dx_m)
    print("  dx from distance vector:", dx_m)
    print("  distance range:", segment.distance_m[0], "to", segment.distance_m[-1])
    print("  data min/max:", np.min(amp), np.max(amp))

    filtered = fk_filter(
        amp=amp,
        fs=fs,
        dx_m=dx_m,
        c_min_m_s=1400.0,
        c_max_m_s=3500.0,
        f_min_hz=10.0,
        f_max_hz=90.0,

        # This may create a DAS4Whales matplotlib window showing the filter.
        # Set False after you confirm it works.
        display_filter=True,
    )

    print("Filtered segment")
    print("  shape:", filtered.shape)
    print("  min/max:", np.min(filtered), np.max(filtered))

    # Convert to nanostrain only for display.
    amp_ns = amp * 1e9
    filtered_ns = filtered * 1e9

    # Use common robust color limits for fair before/after comparison.
    vmax = max(
        np.nanpercentile(np.abs(amp_ns), 99),
        np.nanpercentile(np.abs(filtered_ns), 99),
    )
    # vmax = 0.4
    
    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        sharey=True,
        figsize=(14, 8),
        constrained_layout=True,
    )

    extent = [
        segment.time_s[0],
        segment.time_s[-1],
        segment.distance_m[0] / 1000,
        segment.distance_m[-1] / 1000,
    ]

    raw_image = axes[0].imshow(
        amp_ns,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[0].set_title("Raw DAS window")
    axes[0].set_ylabel("Distance (km)")

    filtered_image = axes[1].imshow(
        filtered_ns,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[1].set_title(
        "F-K filtered: 1400–3500 m/s, 10–90 Hz"
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Distance (km)")

    fig.colorbar(
        filtered_image,
        ax=axes,
        label="Strain (nanostrain)",
        shrink=0.9,
    )

    plt.show()


if __name__ == "__main__":
    main()