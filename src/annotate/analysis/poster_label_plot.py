from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
from matplotlib.patches import Rectangle


@dataclass
class DASPlotWindow:
    """
    Processed DAS window for poster/review plotting.

    amp:
        Shape (channel, time). Ideally already in display units, such as
        nanostrain.

    t_s:
        Seconds relative to the displayed window start.

    x_m:
        Physical cable distance in metres.

    fs_hz:
        Processed temporal sampling frequency.

    window_start_utc:
        Unix timestamp corresponding to t_s == 0.
    """

    amp: np.ndarray
    t_s: np.ndarray
    x_m: np.ndarray
    fs_hz: float
    window_start_utc: float

    @property
    def dx_m(self) -> float:
        if len(self.x_m) < 2:
            raise ValueError("Need at least two channels to calculate dx.")

        return float(np.median(np.abs(np.diff(self.x_m))))


def _label_apex_time_in_window_s(
    label: dict[str, Any],
    window_start_utc: float,
) -> float:
    """Convert saved UTC apex timestamp into displayed-window-relative time."""
    return float(label["apex_time_utc"]) - float(window_start_utc)


def _hyperbola_time_s(
    x_m: np.ndarray,
    *,
    anchor_time_s: float,
    anchor_distance_m: float,
    distance_to_cable_m: float,
    sound_speed_mps: float,
) -> np.ndarray:
    """
    Calculate predicted arrival time along a straight cable.

    The curve has its minimum at:

        (anchor_time_s, anchor_distance_m)
    """
    range_m = np.sqrt(
        (x_m - anchor_distance_m) ** 2
        + distance_to_cable_m ** 2
    )

    return anchor_time_s + (
        range_m - distance_to_cable_m
    ) / sound_speed_mps


def _intersection_distance_range(
    *,
    x_m: np.ndarray,
    apex_curve_s: np.ndarray,
    endpoint_curve_s: np.ndarray,
    fx_start_s: float,
    fx_end_s: float,
    selected_dist_min_m: float,
    selected_dist_max_m: float,
) -> tuple[float, float, np.ndarray]:
    """
    Find cable channels whose labeled T-X interval overlaps an F-X time window.

    A channel is included when the time interval between the apex and endpoint
    hyperbola overlaps [fx_start_s, fx_end_s].
    """
    call_start_s = np.minimum(apex_curve_s, endpoint_curve_s)
    call_end_s = np.maximum(apex_curve_s, endpoint_curve_s)

    within_selected_distance = (
        (x_m >= selected_dist_min_m)
        & (x_m <= selected_dist_max_m)
    )

    overlaps_fx_window = (
        (call_end_s >= fx_start_s)
        & (call_start_s <= fx_end_s)
    )

    mask = within_selected_distance & overlaps_fx_window

    if not np.any(mask):
        # Conservative fallback: retain the saved min/max distance range.
        mask = within_selected_distance

    selected_x = x_m[mask]

    return (
        float(np.min(selected_x)),
        float(np.max(selected_x)),
        mask,
    )

def _mask_distance_range(
    x_m: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float] | None:
    """
    Return min/max cable distance for a non-empty boolean channel mask.

    Returns None when no selected channels are present.
    """
    values = np.asarray(x_m)[np.asarray(mask, dtype=bool)]

    if values.size == 0:
        return None

    return float(np.min(values)), float(np.max(values))

def _effective_apparent_velocity_mps(
    x_m: np.ndarray,
    *,
    anchor_distance_m: float,
    distance_to_cable_m: float,
    sound_speed_mps: float,
) -> np.ndarray:
    """
    Apparent velocity along the cable for the hyperbola:

        t(x) = t0 + (sqrt((x-x0)^2 + r^2) - r) / c

    Therefore:

        dt/dx = (x-x0) / (c * sqrt((x-x0)^2 + r^2))

        v_app = |dx/dt|
              = c * sqrt((x-x0)^2 + r^2) / |x-x0|

    Velocity tends toward infinity at the hyperbola apex. We represent those
    positions as NaN because they do not define a finite F-K wedge boundary.
    """
    offset_m = np.abs(x_m - anchor_distance_m)

    velocity_mps = np.full_like(
        x_m,
        np.nan,
        dtype=float,
    )

    valid = offset_m > 1e-6

    velocity_mps[valid] = (
        sound_speed_mps
        * np.sqrt(
            offset_m[valid] ** 2
            + distance_to_cable_m ** 2
        )
        / offset_m[valid]
    )

    return velocity_mps


def plot_label_quadrant(
    window: DASPlotWindow,
    label: dict[str, Any],
    *,
    fx_start_s: float,
    fx_window_s: float = 2.0,
    fx_nfft: int = 2048,
    spec_nfft: int = 256,
    spec_overlap_percent: float = 75.0,
    spectrogram_distance_m: float | None = None,
    max_frequency_hz: float = 100.0,
    tx_percentile: float = 99.0,
    spectral_percentile: float = 99.0,
):
    """
    Create a four-panel poster figure for one label.

    Parameters
    ----------
    window:
        Loaded/processed DAS data.

    label:
        Saved annotation dictionary containing at least:

            apex_time_utc
            apex_dist
            duration
            distance_to_cable
            sound_speed_mps
            dist_min
            dist_max
            f_min
            f_max

    fx_start_s:
        Start of the specific 2-second F-X/F-K window, relative to the
        currently loaded DAS display window.

    fx_window_s:
        F-X/F-K duration. Default is 2 seconds.

    fx_nfft:
        FFT size for F-X display. May exceed actual samples for zero-padding.

    spec_nfft:
        STFT size for spectrogram.

    spectrogram_distance_m:
        Cable distance for spectrogram. Defaults to apex distance.

    Returns
    -------
    fig, axes, derived

    `derived` contains useful calculated geometry for poster/export work.
    """
    amp = np.asarray(window.amp, dtype=float)
    t_s = np.asarray(window.t_s, dtype=float)
    x_m = np.asarray(window.x_m, dtype=float)
    x_km = x_m / 1000.0
    
    if amp.ndim != 2:
        raise ValueError(
            "window.amp must have shape (channel, time)."
        )

    if amp.shape != (len(x_m), len(t_s)):
        raise ValueError(
            "amp shape must match x_m and t_s coordinate lengths."
        )

    # ------------------------------------------------------------------
    # Label geometry
    # ------------------------------------------------------------------
    apex_time_s = _label_apex_time_in_window_s(
        label,
        window.window_start_utc,
    )

    apex_distance_m = float(label["apex_dist"])
    duration_s = float(label["duration"])
    endpoint_time_s = apex_time_s + duration_s

    distance_to_cable_m = float(label["distance_to_cable"])
    sound_speed_mps = float(
        label.get("sound_speed_mps", 1500.0)
    )

    dist_min_m = float(label["dist_min"])
    dist_max_m = float(label["dist_max"])

    f_min_hz = float(label["f_min"])
    f_max_hz = float(label["f_max"])

    if f_max_hz <= f_min_hz:
        raise ValueError("Label f_max must exceed f_min.")

    if dist_max_m <= dist_min_m:
        raise ValueError("Label dist_max must exceed dist_min.")

    fx_end_s = float(fx_start_s + fx_window_s)

    apex_curve_s = _hyperbola_time_s(
        x_m,
        anchor_time_s=apex_time_s,
        anchor_distance_m=apex_distance_m,
        distance_to_cable_m=distance_to_cable_m,
        sound_speed_mps=sound_speed_mps,
    )

    endpoint_curve_s = _hyperbola_time_s(
        x_m,
        anchor_time_s=endpoint_time_s,
        anchor_distance_m=apex_distance_m,
        distance_to_cable_m=distance_to_cable_m,
        sound_speed_mps=sound_speed_mps,
    )

    call_start_s = np.minimum(apex_curve_s, endpoint_curve_s)
    call_end_s = np.maximum(apex_curve_s, endpoint_curve_s)

    tx_mask = (
        (x_m[:, None] >= dist_min_m)
        & (x_m[:, None] <= dist_max_m)
        & (t_s[None, :] >= call_start_s[:, None])
        & (t_s[None, :] <= call_end_s[:, None])
    )

    # Cable-distance range that overlaps the chosen 2-second F-X window.
    fx_dist_min_m, fx_dist_max_m, fx_channel_mask = (
        _intersection_distance_range(
            x_m=x_m,
            apex_curve_s=apex_curve_s,
            endpoint_curve_s=endpoint_curve_s,
            fx_start_s=fx_start_s,
            fx_end_s=fx_end_s,
            selected_dist_min_m=dist_min_m,
            selected_dist_max_m=dist_max_m,
        )
    )

    # ------------------------------------------------------------------
    # Split the F-X/F-K spatial selection into the two sides of the apex.
    #
    # Lower-distance branch:
    #     cable distances below the apex distance.
    #
    # Upper-distance branch:
    #     cable distances above the apex distance.
    #
    # The masks retain only channels whose labeled T-X interval overlaps
    # the selected F-X/F-K temporal window.
    # ------------------------------------------------------------------
    lower_branch_mask = (
        fx_channel_mask
        & (x_m < apex_distance_m)
    )

    upper_branch_mask = (
        fx_channel_mask
        & (x_m > apex_distance_m)
    )

    lower_branch_range_m = _mask_distance_range(
        x_m,
        lower_branch_mask,
    )

    upper_branch_range_m = _mask_distance_range(
        x_m,
        upper_branch_mask,
    )

    # ------------------------------------------------------------------
    # F-X/F-K temporal window samples
    # ------------------------------------------------------------------
    fx_sample_start = int(
        np.searchsorted(t_s, fx_start_s, side="left")
    )
    fx_sample_end = int(
        np.searchsorted(t_s, fx_end_s, side="left")
    )

    if fx_sample_end <= fx_sample_start:
        raise ValueError("Selected F-X window contains no samples.")

    fx_segment = amp[:, fx_sample_start:fx_sample_end]

    actual_window_samples = fx_segment.shape[1]

    if actual_window_samples < 2:
        raise ValueError("F-X window is too short.")

    fx_nfft = max(int(fx_nfft), actual_window_samples)

    # Channel-wise demean before temporal FFT.
    fx_segment_demeaned = fx_segment - np.mean(
        fx_segment,
        axis=1,
        keepdims=True,
    )

    fx_complex = np.fft.rfft(
        fx_segment_demeaned,
        n=fx_nfft,
        axis=1,
    )

    fx_amplitude = (
        2.0
        * np.abs(fx_complex)
        / actual_window_samples
    )

    # One-sided normalization correction for DC/Nyquist.
    fx_amplitude[:, 0] *= 0.5

    if fx_nfft % 2 == 0:
        fx_amplitude[:, -1] *= 0.5

    fx_frequency_hz = np.fft.rfftfreq(
        fx_nfft,
        d=1.0 / window.fs_hz,
    )

    frequency_mask = fx_frequency_hz <= max_frequency_hz

    fx_frequency_hz_plot = fx_frequency_hz[frequency_mask]
    fx_amplitude_plot = fx_amplitude[:, frequency_mask]

    # ------------------------------------------------------------------
    # Spectrogram at selected/apex channel
    # ------------------------------------------------------------------
    if spectrogram_distance_m is None:
        spectrogram_distance_m = apex_distance_m

    spec_row_idx = int(
        np.argmin(
            np.abs(x_m - float(spectrogram_distance_m))
        )
    )

    spec_distance_m = float(x_m[spec_row_idx])
    spec_signal = amp[spec_row_idx, :]

    spec_nfft = min(int(spec_nfft), len(spec_signal))

    if spec_nfft < 2:
        raise ValueError("Signal is too short for spectrogram.")

    spec_noverlap = int(
        round(spec_nfft * spec_overlap_percent / 100.0)
    )
    spec_noverlap = min(spec_noverlap, spec_nfft - 1)

    spec_window = sp.windows.tukey(
        spec_nfft,
        alpha=0.25,
    )

    spec_frequency_hz, spec_time_s, spec_amplitude = sp.spectrogram(
        spec_signal,
        fs=window.fs_hz,
        window=spec_window,
        nperseg=spec_nfft,
        noverlap=spec_noverlap,
        scaling="spectrum",
        mode="magnitude",
    )

    spec_frequency_mask = (
        spec_frequency_hz <= max_frequency_hz
    )

    spec_frequency_hz = spec_frequency_hz[spec_frequency_mask]
    spec_amplitude = spec_amplitude[spec_frequency_mask, :]

    # At this selected channel, calculate the T-X time interval between
    # apex and endpoint hyperbolas.
    spec_call_start_s = float(
        min(
            apex_curve_s[spec_row_idx],
            endpoint_curve_s[spec_row_idx],
        )
    )
    spec_call_end_s = float(
        max(
            apex_curve_s[spec_row_idx],
            endpoint_curve_s[spec_row_idx],
        )
    )

    # ------------------------------------------------------------------
    # F-K from same 2-second F-X window
    # ------------------------------------------------------------------
    #
    # Use channels inside the saved distance range that overlap this F-X
    # temporal interval.
    fk_channel_mask = fx_channel_mask

    fk_x_m = x_m[fk_channel_mask]
    fk_segment = fx_segment[fk_channel_mask, :]

    if fk_segment.shape[0] < 2:
        raise ValueError(
            "F-K window needs at least two selected spatial channels."
        )

    fk_segment = fk_segment - np.mean(
        fk_segment,
        axis=1,
        keepdims=True,
    )

    # Taper before 2D FFT to reduce spectral leakage.
    taper_x = sp.windows.tukey(
        fk_segment.shape[0],
        alpha=0.1,
    )
    taper_t = sp.windows.tukey(
        fk_segment.shape[1],
        alpha=0.1,
    )

    fk_segment = (
        fk_segment
        * taper_x[:, None]
        * taper_t[None, :]
    )

    fk_complex = np.fft.fftshift(
        np.fft.fft2(fk_segment),
        axes=(0, 1),
    )

    fk_amplitude = np.abs(fk_complex)

    fk_frequency_hz = np.fft.fftshift(
        np.fft.fftfreq(
            fk_segment.shape[1],
            d=1.0 / window.fs_hz,
        )
    )

    fk_wavenumber_cpm = np.fft.fftshift(
        np.fft.fftfreq(
            fk_segment.shape[0],
            d=window.dx_m,
        )
    )

    # Plot only positive frequencies so F-K and F-X share the same
    # left-to-right frequency orientation.
    fk_frequency_mask = (
        (fk_frequency_hz >= 0.0)
        & (fk_frequency_hz <= max_frequency_hz)
    )

    fk_frequency_hz_plot = fk_frequency_hz[fk_frequency_mask]
    fk_amplitude_plot = fk_amplitude[
        :,
        fk_frequency_mask,
    ]

    # ------------------------------------------------------------------
    # Derive F-K wedge from hyperbola apparent velocity
    # ------------------------------------------------------------------
    #
    # Use channels in the selected F-K subset whose apex/endpoint label
    # interval overlaps the selected F-X/F-K time window.
    fk_apex_curve_s = apex_curve_s[fk_channel_mask]
    fk_endpoint_curve_s = endpoint_curve_s[fk_channel_mask]

    fk_call_start_s = np.minimum(
        fk_apex_curve_s,
        fk_endpoint_curve_s,
    )
    fk_call_end_s = np.maximum(
        fk_apex_curve_s,
        fk_endpoint_curve_s,
    )

    fk_time_overlap = (
        (fk_call_end_s >= fx_start_s)
        & (fk_call_start_s <= fx_end_s)
    )

    wedge_x_m = fk_x_m[fk_time_overlap]

    apparent_velocity_mps = _effective_apparent_velocity_mps(
        wedge_x_m,
        anchor_distance_m=apex_distance_m,
        distance_to_cable_m=distance_to_cable_m,
        sound_speed_mps=sound_speed_mps,
    )

    finite_velocity = apparent_velocity_mps[
        np.isfinite(apparent_velocity_mps)
    ]

    if finite_velocity.size == 0:
        # Fallback: no finite slope if the only selected point is apex.
        velocity_min_mps = sound_speed_mps
        velocity_max_mps = sound_speed_mps * 10.0
    else:
        velocity_min_mps = float(np.nanmin(finite_velocity))
        velocity_max_mps = float(np.nanmax(finite_velocity))

    # Avoid division by zero and ensure a valid ordering.
    velocity_min_mps = max(velocity_min_mps, 1.0)
    velocity_max_mps = max(
        velocity_max_mps,
        velocity_min_mps * 1.001,
    )

    # k = f / v in cycles/m for f in Hz and v in m/s.
    #
    # At a given f:
    #   slower velocity -> larger |k|
    #   faster velocity -> smaller |k|
    # Restrict the apparent-velocity F-K wedge to the labeled frequency band.
    #
    # The F-K image may show 0–100 Hz, but the annotation wedge represents
    # only the frequency range saved for this event.
    wedge_frequency_mask = (
        (fk_frequency_hz_plot >= f_min_hz)
        & (fk_frequency_hz_plot <= f_max_hz)
    )

    f_wedge = fk_frequency_hz_plot[wedge_frequency_mask]

    if f_wedge.size == 0:
        raise ValueError(
            "The label frequency range does not overlap the plotted "
            "positive F-K frequency axis. "
            f"Label range={f_min_hz:.3f}–{f_max_hz:.3f} Hz."
        )

    # k = f / v in cycles/m.
    #
    # Larger velocity -> lower absolute wavenumber.
    # Smaller velocity -> higher absolute wavenumber.
    k_inner = f_wedge / velocity_max_mps
    k_outer = f_wedge / velocity_min_mps

    # ------------------------------------------------------------------
    # Build four-panel poster layout
    # ------------------------------------------------------------------
    fig = plt.figure(
        figsize=(17, 10),
        constrained_layout=True,
    )

    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,

        # T-X is twice as wide as F-X.
        width_ratios=(2.0, 1.0),

        # Top row is twice as tall as bottom row.
        height_ratios=(2.0, 1.0),
    )

    ax_tx = fig.add_subplot(grid[0, 0])
    ax_fx = fig.add_subplot(
        grid[0, 1],
        sharey=ax_tx,
    )

    ax_fx.tick_params(
        axis="y",
        labelleft=False,
    )
    ax_spec = fig.add_subplot(grid[1, 0])
    ax_fk = fig.add_subplot(grid[1, 1])

    # ------------------------------------------------------------------
    # Top-left: T-X with label mask and F-X time-window guide
    # ------------------------------------------------------------------
    envelope = np.abs(sp.hilbert(amp, axis=1))

    tx_vmax = float(
        np.nanpercentile(envelope, tx_percentile)
    )
    tx_vmax = max(tx_vmax, 1e-12)

    tx_image = ax_tx.imshow(
        envelope,
        origin="lower",
        aspect="auto",
        extent=[
            float(t_s[0]),
            float(t_s[-1]),
            float(x_km[0]),
            float(x_km[-1]),
        ],
        cmap="turbo",
        vmin=0.0,
        vmax=tx_vmax,
    )

    # # Transparent magenta label mask.
    # mask_rgba = np.zeros(
    #     (tx_mask.shape[0], tx_mask.shape[1], 4),
    #     dtype=float,
    # )
    # mask_rgba[..., 0] = 1.0
    # mask_rgba[..., 2] = 1.0
    # mask_rgba[..., 3] = tx_mask * 0.35

    # ax_tx.imshow(
    #     mask_rgba,
    #     origin="lower",
    #     aspect="auto",
    #     extent=[
    #         float(t_s[0]),
    #         float(t_s[-1]),
    #         float(x_km[0]),
    #         float(x_km[-1]),
    #     ],
    # )

    ax_tx.contour(
        t_s,
        x_km,
        tx_mask.astype(np.uint8),
        levels=[0.5],
        colors=["magenta"],
        linewidths=1.2,
    )
    
    ax_tx.plot(
        apex_curve_s,
        x_km,
        color="white",
        linestyle="--",
        linewidth=1.5,
        label="Apex hyperbola",
    )
    ax_tx.plot(
        endpoint_curve_s,
        x_km,
        color="white",
        linestyle="--",
        linewidth=1.5,
        label="Endpoint hyperbola",
    )

    ax_tx.axhline(
        dist_min_m / 1000.0,
        color="white",
        linewidth=1.5,
        label="Distance range",
    )
    
    # Green dotted line marks the DAS channel used for the spectrogram.
    ax_tx.axhline(
        spec_distance_m / 1000.0,
        color="lime",
        linestyle=":",
        linewidth=2.0,
        alpha=0.95,
    )
    
    ax_tx.axhline(
        dist_max_m,
        color="white",
        linewidth=1.5,
    )

    # Dotted vertical lines show exact F-X/F-K temporal window.
    ax_tx.axvline(
        fx_start_s,
        color="red",
        linestyle=":",
        linewidth=2.0,
        label=f"{fx_window_s:g} s F-X/F-K window",
    )
    ax_tx.axvline(
        fx_end_s,
        color="red",
        linestyle=":",
        linewidth=2.0,
    )

    ax_tx.scatter(
        [apex_time_s],
        [apex_distance_m / 1000.0],
        s=60,
        color="red",
        edgecolor="black",
        zorder=10,
        label="Apex",
    )

    ax_tx.set_title("T-X label mask")
    ax_tx.set_xlabel("Time relative to displayed window (s)")
    ax_tx.set_ylabel("Cable distance (km)")
    # ax_tx.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------
    # Top-right: F-X with mapped label bounding box
    # ------------------------------------------------------------------
    fx_vmax = float(
        np.nanpercentile(
            fx_amplitude_plot,
            spectral_percentile,
        )
    )
    fx_vmax = max(fx_vmax, 1e-12)

    fx_image = ax_fx.imshow(
        fx_amplitude_plot,
        origin="lower",
        aspect="auto",
        extent=[
            float(fx_frequency_hz_plot[0]),
            float(fx_frequency_hz_plot[-1]),
            float(x_m[0] / 1000.0),
            float(x_m[-1] / 1000.0),
        ],
        cmap="turbo",
        vmin=0.0,
        vmax=fx_vmax,
    )

    # ------------------------------------------------------------------
    # Map the selected T-X label to one or two F-X boxes.
    #
    # Both boxes use the saved label frequency range. Their distance ranges
    # are determined by the channels that actually overlap this selected
    # 2-second F-X/F-K time window.
    #
    # lower_branch_range_m:
    #     Relevant label channels below apex distance.
    #
    # upper_branch_range_m:
    #     Relevant label channels above apex distance.
    # ------------------------------------------------------------------
    def add_fx_box(
        *,
        distance_range_m: tuple[float, float] | None,
        color: str = "red",
    ) -> None:
        """Draw one F-X frequency-distance box if its branch is non-empty."""
        if distance_range_m is None:
            return

        x_min_m, x_max_m = distance_range_m

        if x_max_m <= x_min_m:
            return

        rectangle = Rectangle(
            (f_min_hz, x_min_m / 1000.0),
            f_max_hz - f_min_hz,
            x_max_m / 1000.0 - x_min_m / 1000.0,
            fill=False,
            edgecolor=color,
            linewidth=2.5,
            linestyle="-",
        )

        ax_fx.add_patch(rectangle)

    # Lower-distance propagation branch.
    add_fx_box(
        distance_range_m=lower_branch_range_m,
        color="red",
    )

    # Higher-distance propagation branch.
    add_fx_box(
        distance_range_m=upper_branch_range_m,
        color="red",
    )
    
    ax_fx.axhline(
        dist_min_m / 1000.0,
        color="white",
        linewidth=1.2,
    )
    ax_fx.axhline(
        dist_max_m / 1000.0,
        color="white",
        linewidth=1.2,
    )

    ax_fx.set_title(
        f"F-X: {fx_start_s:.0f}–{fx_end_s:.0f} s"
    )
    ax_fx.set_xlabel("Frequency (Hz)")
    ax_fx.set_ylabel("Cable distance (km)")

    # Critical poster alignment: same exact cable y-limits.
    y_min = float(x_km[0])
    y_max = float(x_km[-1])

    ax_tx.set_ylim(y_min, y_max)
    ax_fx.set_ylim(y_min, y_max)

    # ------------------------------------------------------------------
    # Bottom-left: spectrogram with mapped label box
    # ------------------------------------------------------------------
    spec_vmax = float(
        np.nanpercentile(
            spec_amplitude,
            spectral_percentile,
        )
    )
    spec_vmax = max(spec_vmax, 1e-12)

    spec_image = ax_spec.pcolormesh(
        spec_time_s,
        spec_frequency_hz,
        spec_amplitude,
        shading="auto",
        cmap="turbo",
        vmin=0.0,
        vmax=spec_vmax,
    )

    # Map label to spectrogram:
    # - time: intersection of apex/endpoint hyperbolas at selected channel;
    # - frequency: label f_min/f_max.
    spec_box = Rectangle(
        (
            spec_call_start_s,
            f_min_hz,
        ),
        spec_call_end_s - spec_call_start_s,
        f_max_hz - f_min_hz,
        fill=False,
        edgecolor="red",
        linewidth=2.5,
    )
    ax_spec.add_patch(spec_box)

    ax_spec.set_title(
        f"Spectrogram at {spec_distance_m / 1000.0:.0f} km"
    )
    ax_spec.set_xlabel("Time relative to displayed window (s)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_ylim(
        0.0,
        max_frequency_hz,
    )

    # ------------------------------------------------------------------
    # Bottom-right: F-K and apparent-velocity wedge
    # ------------------------------------------------------------------
    fk_vmax = float(
        np.nanpercentile(
            fk_amplitude_plot,
            spectral_percentile,
        )
    )
    fk_vmax = max(fk_vmax, 1e-12)

    fk_image = ax_fk.imshow(
        fk_amplitude_plot,
        origin="lower",
        aspect="auto",
        extent=[
            float(fk_frequency_hz_plot[0]),
            float(fk_frequency_hz_plot[-1]),
            float(fk_wavenumber_cpm[0]),
            float(fk_wavenumber_cpm[-1]),
        ],
        cmap="turbo",
        vmin=0.0,
        vmax=fk_vmax,
    )

    # Draw symmetric apparent-velocity wedges.
    #
    # Positive propagation direction: +k wedge.
    # ax_fk.fill_between(
    #     f_wedge,
    #     k_inner,
    #     k_outer,
    #     color="red",
    #     alpha=0.25,
    #     label=(
    #         f"Apparent velocity: "
    #         f"{velocity_min_mps:.0f}–{velocity_max_mps:.0f} m/s"
    #     ),
    # )

    # # Negative propagation direction: -k wedge.
    # ax_fk.fill_between(
    #     f_wedge,
    #     -k_outer,
    #     -k_inner,
    #     color="red",
    #     alpha=0.25,
    # )

    ax_fk.plot(
        f_wedge,
        k_inner,
        color="red",
        linewidth=1.5,
    )
    ax_fk.plot(
        f_wedge,
        k_outer,
        color="red",
        linewidth=1.5,
    )
    ax_fk.plot(
        f_wedge,
        -k_inner,
        color="red",
        linewidth=1.5,
    )
    ax_fk.plot(
        f_wedge,
        -k_outer,
        color="red",
        linewidth=1.5,
    )
    
    # Mark label frequency limits on the F-K plot.
    for frequency_hz in (f_min_hz, f_max_hz):
        ax_fk.axvline(
            frequency_hz,
            color="red",
            linestyle=":",
            linewidth=1.0,
            alpha=0.9,
        )

    ax_fk.axhline(
        0.0,
        color="white",
        linestyle="--",
        linewidth=1.0,
    )

    ax_fk.set_title(
        f"F-K: {fx_start_s:.0f}–{fx_end_s:.0f} s"
    )
    ax_fk.set_xlabel("Frequency (Hz)")
    ax_fk.set_ylabel("Wavenumber (cycles/m)")
    # ax_fk.legend(loc="upper right", fontsize=8)

    # Critical poster alignment: F-X/F-K frequency axes use same limits.
    shared_fmax = min(
        max_frequency_hz,
        float(fx_frequency_hz_plot[-1]),
        float(fk_frequency_hz_plot[-1]),
    )

    ax_fx.set_xlim(0.0, shared_fmax)
    ax_fk.set_xlim(0.0, shared_fmax)

    # ------------------------------------------------------------------
    # Shared colorbars / finishing details
    # ------------------------------------------------------------------
    fig.colorbar(
        tx_image,
        ax=ax_tx,
        label="T-X envelope amplitude",
        shrink=0.85,
    )

    fig.colorbar(
        fx_image,
        ax=ax_fx,
        label="F-X amplitude",
        shrink=0.85,
    )

    fig.colorbar(
        spec_image,
        ax=ax_spec,
        label="Spectrogram magnitude",
        shrink=0.85,
    )

    fig.colorbar(
        fk_image,
        ax=ax_fk,
        label="F-K amplitude",
        shrink=0.85,
    )

    fig.suptitle(
        "Cross-domain DAS annotation mapping",
        fontsize=16,
    )

    derived = {
        "apex_time_s": apex_time_s,
        "endpoint_time_s": endpoint_time_s,
        "fx_start_s": fx_start_s,
        "fx_end_s": fx_end_s,
        "fx_dist_min_m": fx_dist_min_m,
        "fx_dist_max_m": fx_dist_max_m,
        "lower_branch_range_m": lower_branch_range_m,
        "upper_branch_range_m": upper_branch_range_m,
        "spectrogram_distance_m": spec_distance_m,
        "spectrogram_time_min_s": spec_call_start_s,
        "spectrogram_time_max_s": spec_call_end_s,
        "velocity_min_mps": velocity_min_mps,
        "velocity_max_mps": velocity_max_mps,
        "fk_k_inner_cpm": k_inner,
        "fk_k_outer_cpm": k_outer,
    }
    
    axes = {
        "tx": ax_tx,
        "fx": ax_fx,
        "spectrogram": ax_spec,
        "fk": ax_fk,
    }

    return fig, axes, derived