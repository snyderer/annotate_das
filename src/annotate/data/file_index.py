# TODO: fix this file. I think I can simplify it and write my own file time readers as methods instead of frozen classes

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Iterable

import re


@dataclass(frozen=True)
class FileRecord:
    """Time and metadata needed to locate a DAS source file."""

    path: Path
    start_time: datetime
    end_time: datetime | None

    time_source: str           # "header", "filename", or "inferred"
    fs_hz: float | None = None
    n_samples: int | None = None

    metadata: dict | None = None

    def contains(self, requested_time: datetime) -> bool:
        """Return True if the requested time falls in this file."""
        if self.end_time is None:
            return requested_time >= self.start_time

        return self.start_time <= requested_time < self.end_time
    
    
class FileTimeResolver:
    """
    Determines DAS file start time.

    Header Unix timestamps are preferred. Filename parsing is a fallback.
    """

    def __init__(self, header_reader: Callable[[Path], dict]):
        """
        Parameters
        ----------
        header_reader:
            Function that reads lightweight metadata from a DAS file.

            It should return a dict such as:
            {
                "unix_start_time": 1735732800.123,
                "fs_hz": 500.0,
                "n_samples": 15000,
                "interrogator_type": "optasense",
            }
        """
        self.header_reader = header_reader

    def get_record(self, path: str | Path) -> FileRecord:
        path = Path(path)

        header = self._read_header_safely(path)

        unix_start_time = header.get("unix_start_time")
        if unix_start_time is not None:
            start_time = datetime.fromtimestamp(
                float(unix_start_time),
                tz=timezone.utc,
            )
            time_source = "header"
        else:
            start_time = self._parse_filename_time(path)
            time_source = "filename"

        fs_hz = self._as_float_or_none(header.get("fs_hz"))
        n_samples = self._as_int_or_none(header.get("n_samples"))

        end_time = None
        if fs_hz and n_samples:
            end_time = start_time + timedelta(seconds=n_samples / fs_hz)

        return FileRecord(
            path=path,
            start_time=start_time,
            end_time=end_time,
            time_source=time_source,
            fs_hz=fs_hz,
            n_samples=n_samples,
            metadata=header,
        )

    def _read_header_safely(self, path: Path) -> dict:
        try:
            result = self.header_reader(path)
            return result or {}
        except Exception as exc:
            # You may prefer logging this rather than silently swallowing it.
            print(f"Warning: could not read header from {path.name}: {exc}")
            return {}

    @staticmethod
    def _as_float_or_none(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int_or_none(value):
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _parse_filename_time(self, path: Path) -> datetime:
        """
        Fallback parser.

        This must be adapted to your real filename convention.
        """
        match = re.search(r"(\d{8})_(\d{6})", path.stem)
        if match is None:
            raise ValueError(
                "No valid header Unix timestamp and no supported full "
                f"datetime in filename: {path.name}"
            )

        return datetime.strptime(
            match.group(1) + match.group(2),
            "%Y%m%d%H%M%S",
        ).replace(tzinfo=timezone.utc)