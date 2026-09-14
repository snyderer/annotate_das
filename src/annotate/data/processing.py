"""
Shared data-loading interfaces, data models, and processing helpers.

Source-specific loaders (for example PreprocessedDataLoader and
DAS4WhalesDataLoader) should inherit from BaseDataLoader and return a
LoadedWindow from load_window().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import scipy.signal as sp
import das4whales as dw
from annotate.models import RawDASSegment


# ============================================================================
# Common data structures
# ============================================================================

@dataclass
class DatasetMetadata:
    """
    Metadata collected after opening/indexing a dataset.

    Parameters
    ----------
    dataset_path
        Directory containing the dataset/source files.
    format_name
        Human-readable source format, e.g. "preprocessed" or
        "das4whales:optasense".
    fs_native
        Native temporal sample rate in Hz.
    x_native
        Native physical channel-coordinate vector in metres.
    file_paths
        Ordered source-file paths.
    file_start_times
        Absolute Unix start timestamp for every source file.
    file_end_times
        Absolute Unix end timestamp for every source file.
    extra
        Loader-specific metadata not needed by the generic GUI.
    """
    dataset_path: Path
    format_name: str

    fs_native: float
    x_native: np.ndarray

    file_paths: list[Path] = field(default_factory=list)
    file_start_times: np.ndarray | None = None
    file_end_times: np.ndarray | None = None

    extra: dict = field(default_factory=dict)


@dataclass
class LoadedWindow:
    """
    Format-independent data product returned by all loaders.

    `amp` always has shape:

        (n_channels, n_time_samples)

    `t` is relative to `start_timestamp`, in seconds.

    `x` contains physical cable/channel distances in metres.
    """
    amp: np.ndarray
    t: np.ndarray
    x: np.ndarray

    start_timestamp: float
    end_timestamp: float
    fs: float

    source_files: list[Path] = field(default_factory=list)

    # Actual source channel indices used to create `x` and `amp`.
    resolved_channel_indices: np.ndarray | None = None

    # Useful information for displaying resolved loading settings in the GUI.
    resolved_loading_settings: dict = field(default_factory=dict)

    @property
    def time_stamps(self) -> np.ndarray:
        """
        Absolute Unix timestamp for every sample in the loaded window.
        """
        return self.start_timestamp + self.t


# ============================================================================
# Abstract loader interface
# ============================================================================

class BaseDataLoader(ABC):
    """
    Abstract interface implemented by every source-format loader.

    A loader is responsible for:
      1. recognizing whether it can open a selected source file;
      2. reading/indexing source metadata;
      3. loading a requested time/spatial window;
      4. returning a format-independent LoadedWindow.

    DataManager should interact with loaders only through this interface.
    """

    format_name = "unknown"

    @classmethod
    @abstractmethod
    def can_open(cls, filepath: str | Path) -> bool:
        """
        Return True if this loader recognizes the selected file.

        This should be a quick check. It should not load the entire dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def open_dataset(self, filepath: str | Path) -> DatasetMetadata:
        """
        Open/index the dataset containing `filepath`.

        This should gather enough metadata to:
          - determine native fs and channel positions;
          - find related source files;
          - determine their start/end times.

        It should not normally load a large display window.
        """
        raise NotImplementedError

    @abstractmethod
    def load_window(
        self,
        *,
        start_timestamp: float,
        duration_s: float,
        channel_start_m: float | None,
        channel_end_m: float | None,
        target_dx_m: float | None,
        processing_settings: dict,
    ) -> LoadedWindow:
        """
        Load one requested display window.

        Parameters
        ----------
        start_timestamp
            Requested absolute Unix start time.
        duration_s
            Requested display duration in seconds.
        channel_start_m, channel_end_m
            Requested physical cable-distance bounds in metres.
            None means use the available minimum/maximum respectively.
        target_dx_m
            Requested post-selection channel spacing in metres.
            The loader resolves this to the nearest feasible channel stride.
        processing_settings
            Dictionary supplied by DataManager/ControlPanel. Expected keys
            include:

                bandpass_enabled
                f_lo_hz
                f_hi_hz
                fk_filter_enabled
                c_min_m_s
                c_max_m_s
                downsample_for_display
                max_display_freq_hz

        Returns
        -------
        LoadedWindow
            Canonical format-independent loaded data.
        """
        raise NotImplementedError


# ============================================================================
# Spatial selection helpers
# ============================================================================

def estimate_dx(x: np.ndarray) -> float | None:
    """
    Estimate representative channel spacing in metres.

    Uses the median difference so it is somewhat robust to small coordinate
    irregularities. Returns None if fewer than two channels are present.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a 1-D coordinate vector.")

    if x.size < 2:
        return None

    return float(np.median(np.diff(x)))


def is_regularly_spaced(
    x: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> bool:
    """
    Return True if channel spacing is approximately uniform.

    This will be useful before applying f-k filtering, which generally
    assumes a regularly sampled spatial axis.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        return False

    if x.size < 3:
        return True

    dx = np.diff(x)

    return np.allclose(
        dx,
        np.median(dx),
        rtol=rtol,
        atol=atol,
    )


def resolve_spatial_selection(
    x_native: np.ndarray,
    requested_start_m: float | None,
    requested_end_m: float | None,
    requested_dx_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert requested cable coordinates to actual source-channel indices.

    No spatial interpolation is performed. Requested spacing is converted
    into an integer channel stride based on native dx.

    Parameters
    ----------
    x_native
        Native channel positions in metres. Must be a monotonically increasing
        1-D vector.
    requested_start_m, requested_end_m
        Requested cable-distance bounds. The nearest available channel is used.
    requested_dx_m
        Desired channel spacing in metres. If it is smaller than native dx,
        native spacing is retained. If it is not an integer multiple of native
        dx, the nearest integer stride is selected.

    Returns
    -------
    channel_indices
        Integer indices into the native channel axis.
    x_selected
        Actual selected physical channel positions in metres.
    """
    x_native = np.asarray(x_native, dtype=float)

    if x_native.ndim != 1 or x_native.size == 0:
        raise ValueError("x_native must be a non-empty 1-D coordinate vector.")

    if x_native.size > 1 and np.any(np.diff(x_native) <= 0):
        raise ValueError(
            "x_native must be strictly increasing. "
            "Reverse/reorder channels in the source-specific loader first."
        )

    if requested_start_m is None:
        requested_start_m = float(x_native[0])

    if requested_end_m is None:
        requested_end_m = float(x_native[-1])

    if requested_start_m > requested_end_m:
        requested_start_m, requested_end_m = (
            requested_end_m,
            requested_start_m,
        )

    # Resolve requested physical boundaries to nearest available channels.
    start_idx = int(np.argmin(np.abs(x_native - requested_start_m)))
    end_idx = int(np.argmin(np.abs(x_native - requested_end_m)))

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    native_dx_m = estimate_dx(x_native)
    stride = 1

    if (
        requested_dx_m is not None
        and requested_dx_m > 0
        and native_dx_m is not None
        and native_dx_m > 0
    ):
        # Example:
        # native dx = 2.5 m; requested dx = 10 m -> stride = 4.
        #
        # A requested dx smaller than native dx produces stride=1, meaning
        # use the native spacing rather than invent/interpolating channels.
        stride = max(1, int(round(requested_dx_m / native_dx_m)))

    channel_indices = np.arange(
        start_idx,
        end_idx + 1,
        stride,
        dtype=int,
    )

    if channel_indices.size == 0:
        raise ValueError(
            "No channels were selected from the requested spatial range."
        )

    x_selected = x_native[channel_indices]
    return channel_indices, x_selected


def spatial_selection_metadata(
    x_native: np.ndarray,
    channel_indices: np.ndarray,
    requested_start_m: float | None,
    requested_end_m: float | None,
    requested_dx_m: float | None,
) -> dict:
    """
    Build a consistent metadata dictionary for the resolved spatial selection.

    This is suitable for LoadedWindow.resolved_loading_settings and for
    displaying the resolved values in the GUI.
    """
    x_native = np.asarray(x_native, dtype=float)
    channel_indices = np.asarray(channel_indices, dtype=int)
    x_selected = x_native[channel_indices]

    return {
        "requested_channel_start_m": requested_start_m,
        "requested_channel_end_m": requested_end_m,
        "requested_channel_dx_m": requested_dx_m,

        "resolved_channel_start_m": float(x_selected[0]),
        "resolved_channel_end_m": float(x_selected[-1]),
        "resolved_channel_dx_m": estimate_dx(x_selected),

        "native_channel_dx_m": estimate_dx(x_native),
        "channel_stride": (
            int(channel_indices[1] - channel_indices[0])
            if channel_indices.size > 1
            else 1
        ),
        "n_channels": int(channel_indices.size),
        "first_channel_index": int(channel_indices[0]),
        "last_channel_index": int(channel_indices[-1]),
    }


# ============================================================================
# Common post-load processing
# ============================================================================

def apply_processing(
    amp: np.ndarray,
    fs: float,
    dx_m: float | None,
    settings: dict,
) -> tuple[np.ndarray, float]:
    """
    Apply shared processing after a loader has read/cropped data.

    Expected processing order:

        source read
        -> concatenate files
        -> crop requested time window
        -> select spatial channels
        -> bandpass filter
        -> f-k filter
        -> temporal display downsampling

    Parameters
    ----------
    amp
        Array shaped (n_channels, n_time_samples).
    fs
        Current temporal sample rate in Hz.
    dx_m
        Current actual channel spacing in metres.
    settings
        User settings dictionary.

    Returns
    -------
    amp_processed, fs_processed
    """
    amp = np.asarray(amp, dtype=float)

    if amp.ndim != 2:
        raise ValueError(
            "amp must have shape (n_channels, n_time_samples)."
        )

    if fs <= 0:
        raise ValueError(f"fs must be positive; got {fs}.")

    if settings.get("bandpass_enabled", False):
        amp = bandpass_filter(
            amp=amp,
            fs=fs,
            f_lo_hz=settings.get("f_lo_hz"),
            f_hi_hz=settings.get("f_hi_hz"),
        )

    if settings.get("fk_filter_enabled", False):
        if dx_m is None or dx_m <= 0:
            raise ValueError(
                "Cannot apply f-k filtering without a valid channel spacing."
            )

        amp = fk_filter(
            amp=amp,
            fs=fs,
            dx_m=dx_m,
            c_min_m_s=settings.get("c_min_mps"),
            c_max_m_s=settings.get("c_max_mps"),

            # Normally make the F-K frequency range match the selected bandpass.
            f_min_hz=settings.get("f_lo_hz"),
            f_max_hz=settings.get("f_hi_hz"),

            # Never show DAS4Whales matplotlib filter windows from the GUI.
            display_filter=False,
        )

    if settings.get("downsample_for_display", False):
        max_display_freq_hz = settings.get("max_display_freq_hz")

        if (
            max_display_freq_hz is not None
            and float(max_display_freq_hz) > 0
        ):
            amp, fs = temporal_downsample(
                amp=amp,
                fs=fs,
                max_frequency_hz=float(max_display_freq_hz),
            )

    return amp, fs


def bandpass_filter(
    amp: np.ndarray,
    fs: float,
    f_lo_hz: float | None,
    f_hi_hz: float | None,
    order: int = 6,
) -> np.ndarray:
    """
    Apply zero-phase time-domain filtering along the time axis.

    Depending on bounds, this becomes:
      - bandpass: f_lo_hz and f_hi_hz supplied;
      - highpass: f_lo_hz only;
      - lowpass: f_hi_hz only;
      - no operation: neither bound usable.

    Frequencies above Nyquist are treated as unset.
    """
    nyquist_hz = fs / 2.0

    if f_lo_hz is not None:
        f_lo_hz = float(f_lo_hz)
        if f_lo_hz <= 0:
            f_lo_hz = None

    if f_hi_hz is not None:
        f_hi_hz = float(f_hi_hz)
        if f_hi_hz <= 0 or f_hi_hz >= nyquist_hz:
            f_hi_hz = None

    if f_lo_hz is None and f_hi_hz is None:
        return amp

    if f_lo_hz is not None and f_hi_hz is not None:
        if f_lo_hz >= f_hi_hz:
            raise ValueError(
                f"Bandpass requires f_lo_hz < f_hi_hz; "
                f"received {f_lo_hz} and {f_hi_hz}."
            )

        sos = sp.butter(
            order,
            [f_lo_hz, f_hi_hz],
            btype="bandpass",
            fs=fs,
            output="sos",
        )

    elif f_hi_hz is not None:
        sos = sp.butter(
            order,
            f_hi_hz,
            btype="lowpass",
            fs=fs,
            output="sos",
        )

    else:
        sos = sp.butter(
            order,
            f_lo_hz,
            btype="highpass",
            fs=fs,
            output="sos",
        )

    # sosfiltfilt performs zero-phase filtering.
    # It can fail for very short time vectors, so retain a meaningful error.
    try:
        return sp.sosfiltfilt(sos, amp, axis=1)
    except ValueError as exc:
        raise ValueError(
            "Unable to apply zero-phase bandpass filter. "
            "The loaded time window may be too short for this filter."
        ) from exc


def temporal_downsample(
    amp: np.ndarray,
    fs: float,
    max_frequency_hz: float,
    nyquist_margin: float = 2.5,
) -> tuple[np.ndarray, float]:
    """
    Downsample temporally for display using scipy.signal.resample_poly.

    `max_frequency_hz` is interpreted as the highest frequency that should
    remain visible. The target sampling rate is:

        target_fs = nyquist_margin * max_frequency_hz

    A margin greater than 2 provides room above the strict Nyquist limit.

    Returns
    -------
    amp_downsampled, fs_downsampled
    """
    if max_frequency_hz <= 0:
        raise ValueError(
            "max_frequency_hz must be positive when downsampling."
        )

    if nyquist_margin <= 2.0:
        raise ValueError(
            "nyquist_margin should be greater than 2.0."
        )

    target_fs = nyquist_margin * max_frequency_hz

    # No downsampling is necessary if native/current fs is already low enough.
    if target_fs >= fs:
        return amp, fs

    # Integer downsampling factor. resample_poly applies anti-alias filtering.
    down_factor = max(1, int(np.floor(fs / target_fs)))

    if down_factor <= 1:
        return amp, fs

    amp_downsampled = sp.resample_poly(
        amp,
        up=1,
        down=down_factor,
        axis=1,
    )

    return amp_downsampled, fs / down_factor


def fk_filter(
    amp: np.ndarray,
    fs: float,
    dx_m: float,
    c_min_m_s: float | None,
    c_max_m_s: float | None,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    display_filter: bool = False,
) -> np.ndarray:
    """
    Apply DAS4Whales hybrid F-K velocity filter.

    Parameters
    ----------
    amp:
        Data with shape (n_channels, n_time_samples).

    fs:
        Current temporal sampling rate in Hz.

    dx_m:
        Actual spatial channel spacing in metres after any channel decimation.

    c_min_m_s, c_max_m_s:
        Minimum and maximum apparent velocity bounds in m/s.

    f_min_hz, f_max_hz:
        Frequency range where the F-K velocity filter should be active.
        These should normally match or lie inside the bandpass range.

    display_filter:
        If True, DAS4Whales displays the designed filter. Keep False in GUI
        operation because it creates additional plotting windows.

    Returns
    -------
    np.ndarray
        Filtered data with the same shape as amp.
    """
    amp = np.asarray(amp, dtype=np.float32)

    if amp.ndim != 2:
        raise ValueError(
            "F-K filtering requires amp shaped "
            "(n_channels, n_time_samples)."
        )

    n_channels, n_samples = amp.shape

    if n_channels < 2:
        raise ValueError(
            "F-K filtering requires at least two spatial channels."
        )

    if n_samples < 2:
        raise ValueError(
            "F-K filtering requires at least two time samples."
        )

    if fs <= 0:
        raise ValueError(f"F-K filtering requires fs > 0; got {fs}.")

    if dx_m <= 0:
        raise ValueError(
            f"F-K filtering requires dx_m > 0; got {dx_m}."
        )

    if c_min_m_s is None or c_max_m_s is None:
        raise ValueError(
            "F-K filtering requires both c_min_m_s and c_max_m_s."
        )

    c_min_m_s = float(c_min_m_s)
    c_max_m_s = float(c_max_m_s)

    if c_min_m_s <= 0 or c_max_m_s <= 0:
        raise ValueError(
            "F-K velocity bounds must both be positive."
        )

    if c_min_m_s >= c_max_m_s:
        raise ValueError(
            "F-K minimum velocity must be less than maximum velocity."
        )

    nyquist_hz = fs / 2.0

    # Default F-K frequency range.
    if f_min_hz is None:
        f_min_hz = 0.0

    if f_max_hz is None:
        f_max_hz = nyquist_hz * 0.95

    f_min_hz = max(0.0, float(f_min_hz))
    f_max_hz = min(float(f_max_hz), nyquist_hz * 0.95)

    if f_max_hz <= f_min_hz:
        raise ValueError(
            "Invalid F-K frequency bounds: "
            f"f_min={f_min_hz}, f_max={f_max_hz}, "
            f"Nyquist={nyquist_hz}."
        )

    # The data have already been spatially selected/decimated before
    # reaching this function. Therefore describe the array as a contiguous
    # channel sequence with stride 1 and use dx_m as the actual spacing.
    #
    # DAS4Whales uses selected_channels to infer spatial sampling. Since dx_m
    # is already the resolved spacing of the current data matrix, this is:
    selected_channels = (0, n_channels, 1)

    fk_params = {
        "c_min": c_min_m_s,
        "c_max": c_max_m_s,
        "fmin": f_min_hz,
        "fmax": f_max_hz,
    }

    try:
        fk_mask = dw.dsp.hybrid_ninf_gs_filter_design(
            amp.shape,
            selected_channels,
            dx_m,
            fs,
            fk_params,
            display_filter=display_filter,
        )

        filtered = dw.dsp.fk_filter_sparsefilt(
            amp,
            fk_mask,
            tapering=False,
        )

    except Exception as exc:
        raise RuntimeError(
            "DAS4Whales F-K filtering failed. "
            f"shape={amp.shape}, fs={fs}, dx_m={dx_m}, "
            f"c_min={c_min_m_s}, c_max={c_max_m_s}, "
            f"fmin={f_min_hz}, fmax={f_max_hz}."
        ) from exc

    return np.asarray(filtered, dtype=np.float32)

def stitch_file_offsets(
    pieces: list[RawDASSegment],
    edge_duration_s: float = 0.25,
) -> list[RawDASSegment]:
    """
    Remove per-channel constant offsets between contiguous DAS file pieces.

    Each source file may have an arbitrary baseline offset. This function
    shifts every channel in each subsequent file so that its beginning
    matches the end of the preceding file.

    Parameters
    ----------
    pieces:
        Segments sorted in time order. Each must have data shaped
        (n_channels, n_time_samples).

    edge_duration_s:
        Duration at each side of a file boundary used to estimate the
        baseline. A median over this window is robust to spikes/calls.

    Returns
    -------
    list[RawDASSegment]
        New segments with corrected per-channel offsets. The first segment
        is unchanged; each later segment is shifted relative to prior data.
    """
    if len(pieces) <= 1:
        return pieces

    corrected: list[RawDASSegment] = [pieces[0]]

    for current in pieces[1:]:
        previous = corrected[-1]

        if not np.isclose(previous.fs_hz, current.fs_hz):
            raise ValueError(
                "Cannot stitch file offsets when sampling rates differ: "
                f"{previous.fs_hz} vs {current.fs_hz}"
            )

        if previous.data.shape[0] != current.data.shape[0]:
            raise ValueError(
                "Cannot stitch file offsets when channel counts differ: "
                f"{previous.data.shape[0]} vs {current.data.shape[0]}"
            )

        n_edge = max(1, int(round(edge_duration_s * current.fs_hz)))

        # Avoid requesting more samples than are available in either piece.
        n_edge = min(
            n_edge,
            previous.data.shape[1],
            current.data.shape[1],
        )

        # Median is safer than using exactly the final/first sample,
        # which may be noisy or contain an arrival.
        previous_level = np.median(
            previous.data[:, -n_edge:],
            axis=1,
        )

        current_level = np.median(
            current.data[:, :n_edge],
            axis=1,
        )

        # Shape: (n_channels, 1), so broadcasting shifts every time sample
        # of each channel by its own correction.
        offset = (previous_level - current_level)[:, np.newaxis]

        corrected_data = current.data + offset

        corrected.append(
            replace(
                current,
                data=corrected_data,
                metadata={
                    **current.metadata,
                    "boundary_offset_corrected": True,
                    "boundary_edge_duration_s": edge_duration_s,
                    "boundary_offset_per_channel": offset[:, 0],
                },
            )
        )

    return corrected