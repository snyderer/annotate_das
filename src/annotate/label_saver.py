from __future__ import annotations

import getpass
import os
from pathlib import Path

import numpy as np
import pandas as pd


class LabelSaver:
    """
    CSV-backed repository for DAS annotations.

    One row represents one T-X annotation. F-X boxes update the row's
    global f_min / f_max range.
    """

    COLUMNS = [
        "tx_id",
        "uid",
        "apex_time_utc",
        "apex_time_str",
        "apex_time_s",
        "sound_speed_mps",
        "apex_dist",
        "duration",
        "distance_to_cable",
        "dist_max",
        "dist_min",
        "f_max",
        "f_min",
        "label",
        "label_name",
        "dataset",
        "source_file",
        "saved_timestamp",
        "username",
    ]

    def __init__(self, csv_path: str | Path):
        self.csv_path = str(Path(csv_path).expanduser().resolve())

        Path(self.csv_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)

            # Add any fields missing from older CSV versions.
            for column in self.COLUMNS:
                if column not in self.df.columns:
                    self.df[column] = np.nan

            self.df = self.df[self.COLUMNS]

        else:
            self.df = pd.DataFrame(columns=self.COLUMNS)
            self._save()

    def _save(self) -> None:
        self.df.to_csv(
            self.csv_path,
            index=False,
        )

    def _next_tx_id(self) -> int:
        if self.df.empty:
            return 1

        tx_ids = pd.to_numeric(
            self.df["tx_id"],
            errors="coerce",
        ).dropna()

        if tx_ids.empty:
            return 1

        return int(tx_ids.max()) + 1

    def save_tx_label(
        self,
        *,
        uid: str,
        apex_time_utc: float,
        apex_time_str: str,
        apex_time_s: float,
        apex_dist: float,
        x_m: list[float],
        t_s: list[float],
        dataset: str,
        source_file: str,
        label: int,
        label_name: str,
        distance_to_cable: float | None = None,
        sound_speed_mps: float | None = None,
        saved_timestamp: str | None = None,
        username: str | None = None,
    ) -> int:
        """
        Save one T-X annotation row and return its integer tx_id.

        `x_m` and `t_s` are retained as inputs for compatibility with the
        annotation workflow. Their extrema define saved distance range and
        duration.
        """
        if saved_timestamp is None:
            saved_timestamp = (
                pd.Timestamp.utcnow()
                .strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            )

        if username is None:
            username = getpass.getuser()

        x_m = list(x_m or [])
        t_s = list(t_s or [])

        duration = (
            float(np.max(t_s) - np.min(t_s))
            if t_s
            else np.nan
        )

        dist_min = float(np.min(x_m)) if x_m else np.nan
        dist_max = float(np.max(x_m)) if x_m else np.nan

        tx_id = self._next_tx_id()

        row = {
            "tx_id": tx_id,
            "uid": str(uid),
            "apex_time_utc": float(apex_time_utc),
            "apex_time_str": str(apex_time_str),
            "apex_time_s": float(apex_time_utc),
            "apex_dist": float(apex_dist),
            "sound_speed_mps": float(sound_speed_mps),
            "duration": duration,
            "distance_to_cable": (
                float(distance_to_cable)
                if distance_to_cable is not None
                else np.nan
            ),
            "dist_max": dist_max,
            "dist_min": dist_min,
            "f_max": np.nan,
            "f_min": np.nan,
            "label": int(label),
            "label_name": str(label_name),
            "dataset": Path(dataset).name,
            "source_file": str(
                Path(source_file).expanduser().resolve()
            ),
            "saved_timestamp": saved_timestamp,
            "username": username,
        }

        self.df = pd.concat(
            [self.df, pd.DataFrame([row])],
            ignore_index=True,
        )

        self._save()
        return tx_id

    def save_fx_label(
        self,
        *,
        tx_id: int,
        f_min_hz: float,
        f_max_hz: float,
        **_ignored,
    ) -> None:
        """
        Expand the saved annotation's global F-X frequency range.

        Individual ROI coordinates are currently not persisted separately;
        only the lowest/highest frequency across all F-X boxes is saved.
        """
        matching_rows = self.df.index[
            pd.to_numeric(
                self.df["tx_id"],
                errors="coerce",
            ) == int(tx_id)
        ]

        if len(matching_rows) == 0:
            raise KeyError(
                f"No saved annotation found for tx_id={tx_id}."
            )

        row_index = matching_rows[0]

        old_min = self.df.at[row_index, "f_min"]
        old_max = self.df.at[row_index, "f_max"]

        old_min = (
            float(old_min)
            if pd.notna(old_min)
            else None
        )
        old_max = (
            float(old_max)
            if pd.notna(old_max)
            else None
        )

        f_min_hz = float(f_min_hz)
        f_max_hz = float(f_max_hz)

        self.df.at[row_index, "f_min"] = (
            f_min_hz
            if old_min is None
            else min(old_min, f_min_hz)
        )

        self.df.at[row_index, "f_max"] = (
            f_max_hz
            if old_max is None
            else max(old_max, f_max_hz)
        )

        self._save()

    def remove_label_by_id(self, tx_id: int) -> None:
        """Delete one annotation row permanently."""
        self.df = self.df[
            pd.to_numeric(
                self.df["tx_id"],
                errors="coerce",
            ) != int(tx_id)
        ].reset_index(drop=True)

        self._save()

    def close(self) -> None:
        """Compatibility no-op for future database-backed storage."""
        pass