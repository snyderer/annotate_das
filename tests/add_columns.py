from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from annotate.config import LABEL_DEFINITIONS


INPUT_CSV = Path(
    r"C:\Users\ers334\Desktop\projects\labeling_2026\annotations.csv"
)

# Map old saved `dataset` values, dataset directory names, or known source-file
# patterns to stable profile keys.
DATASET_KEY_RULES = {
    "svalbard": "svalbard",
    "north_c2": "ooi_north_c2",
    "north_c3": "ooi_north_c3",
    "blue_whales_north_c3": "ooi_north_c3",
    "south_c1": "south_c1",
    "silixa": "ooi_silixa",
    "stlawrence": "st_lawrence",
    "st_lawrence": "st_lawrence",
}


def infer_dataset_key(row: pd.Series) -> str:
    """
    Infer stable dataset key from old dataset/source-file fields.

    Review output after running the script. Any blank dataset_key should be
    manually assigned.
    """
    candidates = [
        str(row.get("dataset", "")).lower(),
        str(row.get("source_file", "")).lower(),
    ]

    text = " ".join(candidates)

    for pattern, dataset_key in DATASET_KEY_RULES.items():
        if pattern in text:
            return dataset_key

    return ""


def infer_label_code(row: pd.Series) -> str:
    """
    Infer label code from saved numeric label first, then label name.
    """
    label_number = row.get("label")

    try:
        label_number = int(float(label_number))

        definition = LABEL_DEFINITIONS.get(label_number)

        if definition is not None:
            return definition.code

    except (TypeError, ValueError):
        pass

    old_name = str(row.get("label_name", "")).strip().lower()

    for definition in LABEL_DEFINITIONS.values():
        if definition.name.lower() == old_name:
            return definition.code

    return ""


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)

    backup_path = INPUT_CSV.with_suffix(".before_dataset_key_migration.csv")

    shutil.copy2(INPUT_CSV, backup_path)

    print("Backup created:")
    print(" ", backup_path)

    df = pd.read_csv(INPUT_CSV)

    # ------------------------------------------------------------------
    # Add stable dataset key
    # ------------------------------------------------------------------
    if "dataset_key" not in df.columns:
        df["dataset_key"] = ""

    blank_dataset_key = (
        df["dataset_key"].isna()
        | (df["dataset_key"].astype(str).str.strip() == "")
    )

    df.loc[blank_dataset_key, "dataset_key"] = df[
        blank_dataset_key
    ].apply(
        infer_dataset_key,
        axis=1,
    )

    # ------------------------------------------------------------------
    # Add stable label code
    # ------------------------------------------------------------------
    if "label_code" not in df.columns:
        df["label_code"] = ""

    blank_label_code = (
        df["label_code"].isna()
        | (df["label_code"].astype(str).str.strip() == "")
    )

    df.loc[blank_label_code, "label_code"] = df[
        blank_label_code
    ].apply(
        infer_label_code,
        axis=1,
    )

    # ------------------------------------------------------------------
    # Identify likely invalid apex_time_s values.
    #
    # Old rows may incorrectly have apex_time_s == apex_time_utc. We cannot
    # reconstruct true file-relative time without source-file start metadata,
    # so mark obviously incorrect values as NaN for manual repair.
    # ------------------------------------------------------------------
    if "apex_time_s" in df.columns and "apex_time_utc" in df.columns:
        apex_time_s = pd.to_numeric(
            df["apex_time_s"],
            errors="coerce",
        )

        apex_time_utc = pd.to_numeric(
            df["apex_time_utc"],
            errors="coerce",
        )

        invalid_offset = (
            np.isfinite(apex_time_s)
            & np.isfinite(apex_time_utc)
            & np.isclose(
                apex_time_s,
                apex_time_utc,
                rtol=0.0,
                atol=1.0,
            )
        )

        if invalid_offset.any():
            print(
                "Warning: marking",
                int(invalid_offset.sum()),
                "rows with apex_time_s approximately equal to "
                "apex_time_utc as missing."
            )

            df.loc[invalid_offset, "apex_time_s"] = np.nan

    # ------------------------------------------------------------------
    # Save updated CSV
    # ------------------------------------------------------------------
    df.to_csv(INPUT_CSV, index=False)

    print("\nMigration complete.")
    print("Rows:", len(df))

    unknown_dataset = df[
        df["dataset_key"].astype(str).str.strip() == ""
    ]

    unknown_label = df[
        df["label_code"].astype(str).str.strip() == ""
    ]

    print("Rows with unknown dataset_key:", len(unknown_dataset))
    print("Rows with unknown label_code:", len(unknown_label))

    if not unknown_dataset.empty:
        print("\nUnknown dataset rows:")
        print(
            unknown_dataset[
                [
                    "tx_id",
                    "dataset",
                    "source_file",
                    "dataset_key",
                ]
            ].to_string(index=False)
        )

    if not unknown_label.empty:
        print("\nUnknown label rows:")
        print(
            unknown_label[
                [
                    "tx_id",
                    "label",
                    "label_name",
                    "label_code",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()