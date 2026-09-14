# annotate/models.py
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ReaderSettings:
    """
    Settings needed to interpret source DAS files.

    interrogator:
        DAS4Whales interrogator identifier, such as "optasense".
        Use "auto" when automatic detection is desired.
    """
    interrogator: str = "auto"


@dataclass
class RawDASSegment:
    """
    Raw DAS data in the application's standard convention.

    data.shape == (n_channels, n_time_samples)
    """
    data: np.ndarray
    time_s: np.ndarray
    distance_m: np.ndarray

    start_time: datetime
    fs_hz: float
    dx_m: float
    source_file: Path

    metadata: dict[str, Any]