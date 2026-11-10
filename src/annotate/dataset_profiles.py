from __future__ import annotations

import csv
import getpass
from dataclasses import dataclass, field
from pathlib import Path

from annotate.config import MODE


# ============================================================================
# USER / DRIVE CONFIGURATION
# ============================================================================

ANNOTATOR_DRIVES: dict[str, Path] = {
    "xd264": Path("E:/"),
    "ipf6": Path("H:/"),
    "mmv52": Path("D:/"),
    "ers334": Path("I:/"),
}

CURRENT_USER = getpass.getuser().strip().lower()

BASE_DRIVE = ANNOTATOR_DRIVES.get(
    CURRENT_USER,
    Path("D:/"),
)

DAS_EXTENSIONS = {
    ".h5",
    ".hdf5",
    ".tdms",
}


# ============================================================================
# PROFILE MODEL
# ============================================================================

@dataclass(frozen=True)
class DatasetProfile:
    """
    Settings for one selectable dataset chunk.

    In annotation mode, each profile corresponds to one chunk directory:

        G:/Svalbard/chunk_000/
        H:/OOI_North_C2/chunk_001/
    """
    key: str
    display_name: str
    source_dir: Path

    annotation_csv: Path | None = None

    interrogator: str | None = None

    use_full_cable_length: bool | None = None
    cable_start_m: float | None = None
    cable_end_m: float | None = None
    target_dx_m: float | None = None

    duration_s: float | None = None
    navigation_step_s: float | None = None

    file_boundary_correction_enabled: bool | None = None
    file_boundary_edge_duration_s: float | None = None

    bandpass_enabled: bool | None = None
    f_lo_hz: float | None = None
    f_hi_hz: float | None = None

    fk_filter_enabled: bool | None = None
    c_min_mps: float | None = None
    c_max_mps: float | None = None

    max_display_freq_hz: float | None = None
    sound_speed_mps: float | None = None

    fx_win_s: float | None = None
    fx_nfft: int | None = None
    spec_nfft: int | None = None
    spec_overlap: float | None = None

    tx_vmin: float | None = None
    tx_vmax: float | None = None
    fx_vmin: float | None = None
    fx_vmax: float | None = None
    spec_vmin: float | None = None
    spec_vmax: float | None = None

    # Chunk metadata from chunk_assignments.csv.
    extra_settings: dict = field(default_factory=dict)

    def to_settings_dict(self) -> dict:
        """Return explicit profile settings plus chunk metadata."""
        values = {
            "interrogator": self.interrogator,

            "use_full_cable_length": self.use_full_cable_length,
            "cable_start_m": self.cable_start_m,
            "cable_end_m": self.cable_end_m,
            "target_dx_m": self.target_dx_m,

            "duration_s": self.duration_s,
            "navigation_step_s": self.navigation_step_s,

            "file_boundary_correction_enabled": (
                self.file_boundary_correction_enabled
            ),
            "file_boundary_edge_duration_s": (
                self.file_boundary_edge_duration_s
            ),

            "bandpass_enabled": self.bandpass_enabled,
            "f_lo_hz": self.f_lo_hz,
            "f_hi_hz": self.f_hi_hz,

            "fk_filter_enabled": self.fk_filter_enabled,
            "c_min_mps": self.c_min_mps,
            "c_max_mps": self.c_max_mps,

            "max_display_freq_hz": self.max_display_freq_hz,
            "sound_speed_mps": self.sound_speed_mps,

            "fx_win_s": self.fx_win_s,
            "fx_nfft": self.fx_nfft,
            "spec_nfft": self.spec_nfft,
            "spec_overlap": self.spec_overlap,

            "tx_vmin": self.tx_vmin,
            "tx_vmax": self.tx_vmax,
            "fx_vmin": self.fx_vmin,
            "fx_vmax": self.fx_vmax,
            "spec_vmin": self.spec_vmin,
            "spec_vmax": self.spec_vmax,
        }

        settings = {
            key: value
            for key, value in values.items()
            if value is not None
        }

        if self.annotation_csv is not None:
            settings["labels_file_path"] = str(
                self.annotation_csv
            )

        settings.update(self.extra_settings)

        return settings


# ============================================================================
# FIXED SETTINGS PER DATASET TYPE
# ============================================================================
#
# Keys must match `dataset_key` written by divvy_data.py.

DATASET_DEFAULTS: dict[str, dict] = {
    "svalbard": {
        "display_name": "Svalbard",
        "folder_name": "Svalbard",

        "interrogator": "asn",

        "use_full_cable_length": False,
        "cable_start_m": 20_000.0,
        "cable_end_m": 100_000.0,
        "target_dx_m": 16.0,

        "duration_s": 35.0,
        "navigation_step_s": 20.0,

        "bandpass_enabled": True,
        "f_lo_hz": 5.0,
        "f_hi_hz": 90.0,

        "fk_filter_enabled": True,

        "sound_speed_mps": 1470.0,
    },

    "ooi_north_c2": {
        "display_name": "OOI North C2",
        "folder_name": "OOI_North_C2",

        "interrogator": "optasense",

        "use_full_cable_length": True,
        "cable_start_m": 0.0,
        "cable_end_m": 70_000.0,
        "target_dx_m": 16.0,

        "duration_s": 40.0,
        "navigation_step_s": 25.0,

        "bandpass_enabled": True,
        "f_lo_hz": 5.0,
        "f_hi_hz": 90.0,

        "fk_filter_enabled": True,

        "sound_speed_mps": 1500.0,
    },
}


# ============================================================================
# CHUNK / SOURCE HELPERS
# ============================================================================

def find_source_file_in_directory(
    source_dir: Path,
) -> Path | None:
    """
    Return one supported DAS data file directly inside a chunk directory.

    DatasetService uses this one source file as an entry point, then indexes
    all compatible files inside the same chunk directory.
    """
    if not source_dir.is_dir():
        return None

    source_files = sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in DAS_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )

    return source_files[0] if source_files else None


def find_profile_source_file(
    profile: DatasetProfile,
) -> Path | None:
    """Return one valid DAS source file for a selected chunk profile."""
    return find_source_file_in_directory(
        profile.source_dir
    )


def assigned_to_current_user(
    row: dict[str, str],
) -> bool:
    """
    Return True if this chunk is assigned to the logged-in Windows user.

    A user may be included either as:
    - primary annotator; or
    - one of multiple semicolon-separated verification annotators.
    """
    primary = row.get(
        "primary_annotator",
        "",
    ).strip().lower()

    verification_annotators = {
        value.strip().lower()
        for value in row.get(
            "verification_annotators",
            "",
        ).split(";")
        if value.strip()
    }

    return (
        CURRENT_USER == primary
        or CURRENT_USER in verification_annotators
    )


def build_chunk_profiles(
    assignment_csv: Path,
) -> dict[str, DatasetProfile]:
    """
    Build one DatasetProfile per local chunk assigned to the current user.

    Required CSV columns:

        dataset_key
        chunk_id
        primary_annotator
        verification_annotators

    Additional useful columns from divvy_data.py:

        core_start_utc
        core_end_utc
        first_file
        last_file
        is_full_panel_qa_chunk
    """
    if not assignment_csv.is_file():
        print(
            "Chunk assignment CSV was not found:\n"
            f"  {assignment_csv}"
        )
        return {}

    profiles: dict[str, DatasetProfile] = {}

    with assignment_csv.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "dataset_key",
            "chunk_id",
            "primary_annotator",
            "verification_annotators",
        }

        if reader.fieldnames is None:
            raise ValueError(
                "Chunk assignment CSV has no header row:\n"
                f"  {assignment_csv}"
            )

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Chunk assignment CSV is missing required columns:\n"
                f"  {sorted(missing_columns)}"
            )

        for row in reader:
            if not assigned_to_current_user(row):
                continue

            dataset_key = row[
                "dataset_key"
            ].strip().lower()

            chunk_id = row[
                "chunk_id"
            ].strip()

            if not chunk_id:
                print(
                    "Skipping assignment row with an empty chunk_id."
                )
                continue

            defaults = DATASET_DEFAULTS.get(
                dataset_key
            )

            if defaults is None:
                print(
                    "Skipping unknown dataset key from assignment CSV: "
                    f"{dataset_key}"
                )
                continue

            # Example:
            #
            # G:/Svalbard/chunk_000/
            #
            chunk_dir = (
                BASE_DRIVE
                / defaults["folder_name"]
                / chunk_id
            )

            if not chunk_dir.is_dir():
                print(
                    "Assigned chunk directory is missing:\n"
                    f"  {chunk_dir}"
                )
                continue

            if find_source_file_in_directory(chunk_dir) is None:
                print(
                    "Assigned chunk has no recognized DAS source files:\n"
                    f"  {chunk_dir}"
                )
                continue

            primary_annotator = row.get(
                "primary_annotator",
                "",
            ).strip().lower()

            verification_annotators = {
                value.strip().lower()
                for value in row.get(
                    "verification_annotators",
                    "",
                ).split(";")
                if value.strip()
            }

            assignment_role = (
                "Primary"
                if CURRENT_USER == primary_annotator
                else "Verification"
            )

            profile_key = (
                f"{dataset_key}__{chunk_id}"
            )

            profiles[profile_key] = DatasetProfile(
                key=profile_key,

                display_name=(
                    f"{defaults['display_name']} — "
                    f"{chunk_id.replace('_', ' ').title()} "
                ),

                source_dir=chunk_dir,

                interrogator=defaults.get(
                    "interrogator"
                ),

                use_full_cable_length=defaults.get(
                    "use_full_cable_length"
                ),
                cable_start_m=defaults.get(
                    "cable_start_m"
                ),
                cable_end_m=defaults.get(
                    "cable_end_m"
                ),
                target_dx_m=defaults.get(
                    "target_dx_m"
                ),

                duration_s=defaults.get(
                    "duration_s"
                ),
                navigation_step_s=defaults.get(
                    "navigation_step_s"
                ),

                bandpass_enabled=defaults.get(
                    "bandpass_enabled"
                ),
                f_lo_hz=defaults.get(
                    "f_lo_hz"
                ),
                f_hi_hz=defaults.get(
                    "f_hi_hz"
                ),

                fk_filter_enabled=defaults.get(
                    "fk_filter_enabled"
                ),

                sound_speed_mps=defaults.get(
                    "sound_speed_mps"
                ),

                extra_settings={
                    # Keep base dataset identity separate from profile key.
                    "base_dataset_key": dataset_key,

                    "chunk_id": chunk_id,

                    "chunk_core_start_utc": row.get(
                        "core_start_utc",
                        "",
                    ),

                    "chunk_core_end_utc": row.get(
                        "core_end_utc",
                        "",
                    ),

                    "chunk_first_file": row.get(
                        "first_file",
                        "",
                    ),

                    "chunk_last_file": row.get(
                        "last_file",
                        "",
                    ),

                    "chunk_primary_annotator": primary_annotator,

                    "chunk_verification_annotators": sorted(
                        verification_annotators
                    ),

                    "is_full_panel_qa_chunk": (
                        row.get(
                            "is_full_panel_qa_chunk",
                            "0",
                        ).strip() == "1"
                    ),

                    "annotation_assignment_role": (
                        assignment_role.lower()
                    ),
                },
            )

    return dict(
        sorted(
            profiles.items(),
            key=lambda item: item[1].display_name.lower(),
        )
    )


# ============================================================================
# MODE-SPECIFIC PROFILES
# ============================================================================

if MODE == "debugging":
    DATASET_PROFILES: dict[str, DatasetProfile] = {
        "svalbard_debug": DatasetProfile(
            key="svalbard_debug",
            display_name="Svalbard Debug",
            source_dir=Path(
                r"C:\Users\ers334\Desktop\testingData\Svalbard\data"
            ),
            interrogator="asn",
            use_full_cable_length=False,
            cable_start_m=20_000.0,
            cable_end_m=100_000.0,
            target_dx_m=16.0,
            duration_s=30.0,
            navigation_step_s=15.0,
            bandpass_enabled=True,
            f_lo_hz=5.0,
            f_hi_hz=90.0,
            fk_filter_enabled=True,
            sound_speed_mps=1470.0,
        ),

        "ooi_north_c2_debug": DatasetProfile(
            key="ooi_north_c2_debug",
            display_name="OOI North C2 Debug",
            source_dir=Path(
                r"C:\Users\ers334\Desktop\testingData\OOI\DASData"
                r"\OptaSense\North_C2"
            ),
            interrogator="optasense",
            use_full_cable_length=True,
            cable_start_m=0.0,
            cable_end_m=70_000.0,
            target_dx_m=16.0,
            duration_s=40.0,
            navigation_step_s=25.0,
            bandpass_enabled=True,
            f_lo_hz=5.0,
            f_hi_hz=90.0,
            fk_filter_enabled=True,
            sound_speed_mps=1500.0,
        ),
    }

elif MODE == "annotating":
    # divvy_data.py should place a copy of this CSV at the root of every
    # recipient drive:
    #
    # G:/chunk_assignments.csv
    # H:/chunk_assignments.csv
    # I:/chunk_assignments.csv
    #
    ASSIGNMENT_CSV = BASE_DRIVE / "chunk_assignments.csv"

    DATASET_PROFILES = build_chunk_profiles(
        ASSIGNMENT_CSV
    )

elif MODE == "training":
    DATASET_PROFILES = {}

elif MODE == "skill_assessment":
    DATASET_PROFILES = {}

else:
    raise ValueError(
        f"Unknown MODE value: {MODE!r}"
    )