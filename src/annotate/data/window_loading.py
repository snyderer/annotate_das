from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from annotate.models import RawDASSegment

@dataclass(frozen=True)
class WindowLoadRequest:
    """
    Immutable description of one requested display window.

    Safe to pass from the GUI thread to a worker thread.
    """
    request_id: int

    selected_file: Path
    interrogator: str

    start_time: datetime
    duration_s: float

    selected_channels: tuple[int, int, int]

    correct_file_boundaries: bool
    boundary_edge_duration_s: float

    processing_settings: dict[str, Any]

    # "visible" means the user explicitly requested this window.
    # "prefetch" means it is speculative forward navigation work.
    kind: str = "visible"

@dataclass
class PreparedWindow:
    """
    Fully prepared, non-GUI result from a background loader.
    """
    request_id: int
    kind: str
    requested_start_time: datetime

    raw_segment: RawDASSegment

    amp_display: np.ndarray
    time_s: np.ndarray
    distance_m: np.ndarray
    processed_fs_hz: float

    fx_amp: np.ndarray | None
    fx_freq: np.ndarray | None
    fx_distance_m: np.ndarray | None
    fx_time_s: np.ndarray | None