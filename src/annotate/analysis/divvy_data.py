"""
Split DAS datasets into simple contiguous annotation chunks.

Each dataset is divided into approximately 30-minute core chunks.

Every chunk:
- receives CONTEXT_PADDING_SECONDS before and after its core interval;
- has one primary annotator;
- optionally has one secondary QA annotator;
- is copied into one folder named chunk_###.

Primary assignments are round-robin:

    chunk_000 -> xd264
    chunk_001 -> ipf6
    chunk_002 -> mmv52
    chunk_003 -> xd264
    ...

A configurable subset of chunks receives a secondary annotator for
inter-annotator agreement / quality assurance.

Output example
--------------

G:/
    Svalbard/
        chunk_000/
            *.hdf5
            chunk_manifest.json
        chunk_003/
            *.hdf5
            chunk_manifest.json

H:/
    Svalbard/
        chunk_001/
        chunk_004/

I:/
    Svalbard/
        chunk_002/
        chunk_005/

Each destination drive also receives:

    chunk_assignments.csv

The same CSV is written locally beside this script:

    divvy_manifests/
        chunk_assignments.csv
        summary.csv
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

# ============================================================================
# USER SETTINGS
# ============================================================================

ANNOTATORS = [
    "xd264",
    "ipf6",
    "mmv52",
]

# Use forward slashes or escaped normal strings for drive roots.
#
# Do not use Path(r"G:\") because raw strings cannot end in one backslash.
ANNOTATOR_OUTPUT_ROOTS: dict[str, Path] = {
    "xd264": Path("G:/"),
    "ipf6": Path("H:/"),
    "mmv52": Path("I:/mmv52"),
}

# ---------------------------------------------------------------------------
# Copy controls
# ---------------------------------------------------------------------------

# Start False for a dry run. The manifests will be generated but no source
# data will be copied.
COPY_FILES = True

# Existing files are skipped if False. This makes interrupted copy jobs safe
# to resume by simply rerunning the script.
OVERWRITE_EXISTING = False

# U: -> G:/H:/I: is cross-volume, so ordinary file copying is appropriate.
USE_HARDLINKS = False

# ---------------------------------------------------------------------------
# Chunk / QA controls
# ---------------------------------------------------------------------------

# Core responsibility interval for each chunk.
BLOCK_SECONDS = 30 * 60

# Extra source data copied before and after every core chunk.
#
# This is context only. It allows calls near core boundaries to be fully
# inspected and annotated.
CONTEXT_PADDING_SECONDS = 60

# ---------------------------------------------------------------------------
# Full-panel QA / inter-annotator agreement settings
# ---------------------------------------------------------------------------

# Randomly choose this many chunks from EACH dataset for comparison by all
# annotators.
#
# With value 1:
# - one Svalbard chunk is copied to xd264, ipf6, and mmv52;
# - one OOI North C2 chunk is copied to xd264, ipf6, and mmv52.
FULL_PANEL_QA_CHUNKS_PER_DATASET = 1

# Keeps random selection reproducible. Re-running the script with the same
# source files and this seed selects the same QA chunk(s).
QA_RANDOM_SEED = 20261106

# Every person in this list receives each selected full-panel QA chunk.
#
# Do not include yourself here unless you also add an "ers334" destination
# root below and intend to independently annotate those QA chunks.
FULL_PANEL_QA_ANNOTATORS = [
    "xd264",
    "ipf6",
    "mmv52",
]

# Local manifest output, beside this script.
MANIFEST_ROOT = Path(__file__).resolve().parent / "divvy_manifests"
print(f"Manifest root: {MANIFEST_ROOT}")

# ============================================================================
# DATASET SETTINGS
# ============================================================================
#
# `dataset_key` must match DATASET_DEFAULTS in dataset_profiles.py.
#
# `timestamp_kind` controls how chunk core UTC times are written to the CSV.
#
# - "hms":
#     Filename is expected to begin with HHMMSS, such as:
#         120057.hdf5
#
#     `recording_date_utc` is required.
#
# - "opta_timestamp":
#     Filename is expected to contain:
#         2021-11-02T215901Z
#
#     for example:
#         North-C2-..._2021-11-02T215901Z.h5

DATASETS = [
    {
        "dataset_key": "svalbard",
        "dataset_name": "Svalbard",
        "source_dir": Path(
            r"U:\projects\2022_CLOCCB_OR_S1113\Svalbard\data"
        ),
        "output_name": "Svalbard",
        "file_duration_seconds": 10,
        "expected_file_count": 1222,
        "extensions": {".h5", ".hdf5"},

        # Confirm this recording date is correct for your Svalbard data.
        "timestamp_kind": "hms",
        "recording_date_utc": datetime(
            2022,
            8,
            22,
            tzinfo=timezone.utc,
        ),
    },
    {
        "dataset_key": "ooi_north_c2",
        "dataset_name": "OOI North C2",
        "source_dir": Path(
            r"U:\projects\2022_CLOCCB_OR_S1113\OOI\DASData"
            r"\OptaSense\North_C2"
        ),
        "output_name": "OOI_North_C2",
        "file_duration_seconds": 30,
        "expected_file_count": 255,
        "extensions": {".h5", ".hdf5"},

        "timestamp_kind": "opta_timestamp",
    },
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class Chunk:
    """
    One conceptual annotation chunk.

    Each chunk has one primary annotator and zero or more independent
    verification annotators.

    A full-panel QA chunk may be physically copied to all annotator drives,
    but it remains one chunk with one chunk_id and one core time range.
    """
    dataset_key: str
    dataset_name: str
    dataset_output_name: str

    chunk_id: str
    block_number: int

    primary_annotator: str
    verification_annotators: tuple[str, ...]

    source_dir: Path
    files: tuple[Path, ...]

    core_indices: tuple[int, ...]
    copied_indices: tuple[int, ...]

    file_duration_seconds: float
    context_padding_seconds: float

    core_start_utc: datetime
    core_end_utc: datetime

    @property
    def recipients(self) -> tuple[str, ...]:
        """
        Return unique annotators who should receive a physical copy.

        Primary is always first. Verification annotators follow.
        """
        return tuple(dict.fromkeys((
            self.primary_annotator,
            *self.verification_annotators,
        )))

    @property
    def core_start_index(self) -> int:
        return min(self.core_indices)

    @property
    def core_end_index(self) -> int:
        return max(self.core_indices)

    @property
    def copied_start_index(self) -> int:
        return min(self.copied_indices)

    @property
    def copied_end_index(self) -> int:
        return max(self.copied_indices)


# ============================================================================
# FILE DISCOVERY / SORTING
# ============================================================================

def natural_sort_key(path: Path) -> list[object]:
    """Natural sort key suitable for numeric/timestamped DAS filenames."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"([0-9]+)", path.name)
    ]


def find_files(
    source_dir: Path,
    extensions: set[str],
) -> list[Path]:
    """Find supported source files and return them in chronological order."""
    normalized_extensions = {
        extension.lower()
        for extension in extensions
    }

    files = [
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in normalized_extensions
    ]

    return sorted(
        files,
        key=natural_sort_key,
    )


# ============================================================================
# TIMESTAMP HELPERS
# ============================================================================

def parse_hms_filename_timestamp(
    path: Path,
    *,
    recording_date_utc: datetime,
) -> datetime:
    """
    Parse a filename beginning with HHMMSS.

    Example:
        120057.hdf5 -> 12:00:57 UTC

    The supplied recording_date_utc provides the date.
    """
    match = re.match(
        r"^(\d{2})(\d{2})(\d{2})",
        path.stem,
    )

    if match is None:
        raise ValueError(
            "Could not parse HHMMSS timestamp from filename: "
            f"{path.name}"
        )

    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3))

    return datetime.combine(
        recording_date_utc.date(),
        time(
            hour=hour,
            minute=minute,
            second=second,
            tzinfo=timezone.utc,
        ),
    )


def parse_opta_timestamp(
    path: Path,
) -> datetime:
    """
    Parse OptaSense timestamp embedded in a filename.

    Example:
        North-C2-..._2021-11-02T215901Z.h5
    """
    match = re.search(
        r"(\d{4}-\d{2}-\d{2}T\d{6}Z)",
        path.name,
    )

    if match is None:
        raise ValueError(
            "Could not parse OptaSense timestamp from filename: "
            f"{path.name}"
        )

    return datetime.strptime(
        match.group(1),
        "%Y-%m-%dT%H%M%SZ",
    ).replace(
        tzinfo=timezone.utc
    )


def determine_file_start_times(
    files: list[Path],
    dataset: dict,
) -> list[datetime]:
    """
    Return UTC start times matching the sorted source-file list.

    For HHMMSS filenames, this also handles a midnight rollover if the
    recording crosses into the next UTC day.
    """
    timestamp_kind = dataset["timestamp_kind"]

    starts: list[datetime] = []

    for path in files:
        if timestamp_kind == "hms":
            start = parse_hms_filename_timestamp(
                path,
                recording_date_utc=dataset[
                    "recording_date_utc"
                ],
            )

            # If timestamp appears to go backward, assume midnight rollover.
            if starts and start < starts[-1]:
                while start < starts[-1]:
                    start += timedelta(days=1)

        elif timestamp_kind == "opta_timestamp":
            start = parse_opta_timestamp(path)

        else:
            raise ValueError(
                "Unknown timestamp_kind: "
                f"{timestamp_kind!r}"
            )

        starts.append(start)

    return starts

def find_continuous_file_runs(
    *,
    files: list[Path],
    file_start_times: list[datetime],
    file_duration_seconds: float,
) -> list[tuple[int, int]]:
    """
    Return contiguous source-file runs as (start_index, stop_index) tuples.

    stop_index is exclusive, matching normal Python slicing.

    A source-file timestamp gap starts a new run. This prevents a copied chunk
    or its context padding from spanning missing DAS data.

    Example result:

        [(0, 720), (720, 1222)]

    means:
        files[0:720]    are one continuous run;
        files[720:1222] are another continuous run.
    """
    if not files:
        return []

    expected_step = timedelta(
        seconds=file_duration_seconds
    )

    tolerance = timedelta(
        seconds=max(
            1.0,
            file_duration_seconds * 0.1,
        )
    )

    runs: list[tuple[int, int]] = []
    run_start_index = 0

    for index in range(1, len(file_start_times)):
        previous_time = file_start_times[index - 1]
        current_time = file_start_times[index]

        actual_step = current_time - previous_time

        if abs(actual_step - expected_step) <= tolerance:
            continue

        # End the previous continuous run immediately before this file.
        runs.append((
            run_start_index,
            index,
        ))

        print(
            "\nWARNING: source-file time gap detected.\n"
            f"  Previous file: {files[index - 1].name}\n"
            f"  Previous time: {previous_time.isoformat()}\n"
            f"  Current file:  {files[index].name}\n"
            f"  Current time:  {current_time.isoformat()}\n"
            f"  Expected step: {expected_step}\n"
            f"  Actual step:   {actual_step}\n"
            "  Starting a new chunk run after this gap.\n"
        )

        run_start_index = index

    # Include the final continuous run.
    runs.append((
        run_start_index,
        len(files),
    ))

    return runs
            
# ============================================================================
# CHUNK ASSIGNMENT
# ============================================================================

def make_blocks(
    *,
    total_files: int,
    file_duration_seconds: float,
) -> list[list[int]]:
    """Divide source-file indices into approximately 30-minute core chunks."""
    files_per_block = max(
        1,
        int(BLOCK_SECONDS // file_duration_seconds),
    )

    return [
        list(
            range(
                start,
                min(
                    start + files_per_block,
                    total_files,
                ),
            )
        )
        for start in range(
            0,
            total_files,
            files_per_block,
        )
    ]


def padded_indices(
    *,
    core_indices: list[int],
    total_files: int,
    file_duration_seconds: float,
    run_start_index: int,
    run_end_index: int,
) -> list[int]:
    """
    Return core indices plus context padding, constrained to one continuous run.

    Parameters
    ----------
    run_start_index:
        Inclusive first source-file index in this continuous run.

    run_end_index:
        Exclusive final source-file index in this continuous run.

    This prevents context padding from crossing a real source-data gap.
    """
    if not core_indices:
        return []

    padding_files = math.ceil(
        CONTEXT_PADDING_SECONDS / file_duration_seconds
    )

    copied_start = max(
        run_start_index,
        min(core_indices) - padding_files,
    )

    copied_end = min(
        run_end_index - 1,
        max(core_indices) + padding_files,
    )

    return list(
        range(
            copied_start,
            copied_end + 1,
        )
    )

def select_full_panel_qa_blocks(
    *,
    dataset_key: str,
    n_blocks: int,
) -> set[int]:
    """
    Choose a reproducible random subset of chunk block numbers for which every
    member of FULL_PANEL_QA_ANNOTATORS receives the same chunk.

    A different dataset key produces a different deterministic selection.
    """
    if FULL_PANEL_QA_CHUNKS_PER_DATASET <= 0:
        return set()

    if n_blocks <= 0:
        return set()

    count = min(
        FULL_PANEL_QA_CHUNKS_PER_DATASET,
        n_blocks,
    )

    rng = random.Random(
        f"{QA_RANDOM_SEED}:{dataset_key}"
    )

    return set(
        rng.sample(
            range(n_blocks),
            count,
        )
    )

def verification_annotators_for_chunk(
    *,
    primary_annotator: str,
    is_full_panel_qa: bool,
) -> tuple[str, ...]:
    """
    Return non-primary independent annotators for one chunk.

    Normal chunk:
        ()

    Full-panel QA chunk:
        every FULL_PANEL_QA_ANNOTATOR except the primary.
    """
    if not is_full_panel_qa:
        return ()

    return tuple(
        annotator
        for annotator in FULL_PANEL_QA_ANNOTATORS
        if annotator != primary_annotator
    )

def build_chunks_for_dataset(
    dataset: dict,
) -> list[Chunk]:
    """
    Build approximately 30-minute chunks for one dataset.

    Real temporal gaps in the source directory are respected:

    - chunks never contain a gap;
    - context padding never crosses a gap;
    - a short final chunk may occur immediately before a gap;
    - the following continuous run begins a new chunk after the gap.
    """
    source_dir = Path(dataset["source_dir"])

    files = find_files(
        source_dir,
        dataset["extensions"],
    )

    if not files:
        raise FileNotFoundError(
            "No matching source files found in:\n"
            f"  {source_dir}"
        )

    expected_file_count = dataset.get(
        "expected_file_count"
    )

    if (
        expected_file_count is not None
        and len(files) != expected_file_count
    ):
        print(
            f"WARNING: {dataset['dataset_name']} expected "
            f"{expected_file_count} files but found {len(files)}."
        )

    file_duration_seconds = float(
        dataset["file_duration_seconds"]
    )

    file_start_times = determine_file_start_times(
        files,
        dataset,
    )

    # Split the input files into uninterrupted chronological runs.
    continuous_runs = find_continuous_file_runs(
        files=files,
        file_start_times=file_start_times,
        file_duration_seconds=file_duration_seconds,
    )

    print(
        f"Detected {len(continuous_runs)} continuous source-data "
        f"run(s) for {dataset['dataset_name']}."
    )

    # Build a single ordered sequence of chunks. Each item stores:
    #
    # core_indices,
    # continuous run start index,
    # continuous run end index (exclusive)
    #
    # Chunk numbering remains globally sequential even if the original dataset
    # has multiple continuous sections.
    chunk_specs: list[
        tuple[list[int], int, int]
    ] = []

    for run_number, (
        run_start_index,
        run_end_index,
    ) in enumerate(continuous_runs):

        run_file_count = (
            run_end_index - run_start_index
        )

        run_blocks = make_blocks(
            total_files=run_file_count,
            file_duration_seconds=file_duration_seconds,
        )

        print(
            f"  Run {run_number}: "
            f"{files[run_start_index].name} -> "
            f"{files[run_end_index - 1].name} | "
            f"{run_file_count} files | "
            f"{len(run_blocks)} chunk(s)"
        )

        for run_relative_indices in run_blocks:
            core_indices = [
                run_start_index + index
                for index in run_relative_indices
            ]

            chunk_specs.append((
                core_indices,
                run_start_index,
                run_end_index,
            ))

    # Select the full-panel QA chunks after all normal chunks have been
    # constructed. This ensures one random QA chunk is selected per dataset,
    # even when the dataset has multiple physical source-data runs.
    qa_block_numbers = select_full_panel_qa_blocks(
        dataset_key=dataset["dataset_key"],
        n_blocks=len(chunk_specs),
    )

    chunks: list[Chunk] = []

    for block_number, (
        core_indices,
        run_start_index,
        run_end_index,
    ) in enumerate(chunk_specs):

        primary = ANNOTATORS[
            block_number % len(ANNOTATORS)
        ]

        verification_annotators = (
            verification_annotators_for_chunk(
                primary_annotator=primary,
                is_full_panel_qa=(
                    block_number in qa_block_numbers
                ),
            )
        )

        copied = padded_indices(
            core_indices=core_indices,
            total_files=len(files),
            file_duration_seconds=file_duration_seconds,
            run_start_index=run_start_index,
            run_end_index=run_end_index,
        )

        core_start_index = min(core_indices)
        core_end_index = max(core_indices)

        core_start_utc = file_start_times[
            core_start_index
        ]

        core_end_utc = (
            file_start_times[core_end_index]
            + timedelta(
                seconds=file_duration_seconds
            )
        )

        chunks.append(
            Chunk(
                dataset_key=dataset["dataset_key"],
                dataset_name=dataset["dataset_name"],
                dataset_output_name=dataset["output_name"],

                chunk_id=f"chunk_{block_number:03d}",
                block_number=block_number,

                primary_annotator=primary,
                verification_annotators=verification_annotators,

                source_dir=source_dir,
                files=tuple(files),

                core_indices=tuple(core_indices),
                copied_indices=tuple(copied),

                file_duration_seconds=file_duration_seconds,
                context_padding_seconds=(
                    CONTEXT_PADDING_SECONDS
                ),

                core_start_utc=core_start_utc,
                core_end_utc=core_end_utc,
            )
        )

    return chunks


# ============================================================================
# MANIFEST DATA
# ============================================================================

def chunk_destination(
    *,
    chunk: Chunk,
    recipient: str,
) -> Path:
    """Return physical destination directory for one recipient copy."""
    return (
        ANNOTATOR_OUTPUT_ROOTS[recipient]
        / chunk.dataset_output_name
        / chunk.chunk_id
    )


def chunk_manifest_data(
    chunk: Chunk,
    *,
    recipient: str,
) -> dict:
    """Return JSON-serializable metadata for one copied chunk."""
    files = chunk.files

    return {
        "dataset_key": chunk.dataset_key,
        "dataset_name": chunk.dataset_name,
        "dataset_output_name": chunk.dataset_output_name,

        "chunk_id": chunk.chunk_id,
        "block_number": chunk.block_number,

        "primary_annotator": chunk.primary_annotator,
        "verification_annotators": ";".join(
            chunk.verification_annotators
        ),

        "local_recipient": recipient,

        "annotation_assignment_role": (
            "primary"
            if recipient == chunk.primary_annotator
            else "verification"
        ),

        "is_full_panel_qa_chunk": bool(
            chunk.verification_annotators
        ),

        "file_duration_seconds": chunk.file_duration_seconds,
        "context_padding_seconds": (
            chunk.context_padding_seconds
        ),

        "core_start_utc": chunk.core_start_utc.isoformat(),
        "core_end_utc": chunk.core_end_utc.isoformat(),

        "core_start_index": chunk.core_start_index,
        "core_end_index": chunk.core_end_index,
        "core_file_count": len(chunk.core_indices),

        "copied_start_index": chunk.copied_start_index,
        "copied_end_index": chunk.copied_end_index,
        "copied_file_count": len(chunk.copied_indices),

        "first_file": files[
            chunk.core_start_index
        ].name,

        "last_file": files[
            chunk.core_end_index
        ].name,

        "copied_first_file": files[
            chunk.copied_start_index
        ].name,

        "copied_last_file": files[
            chunk.copied_end_index
        ].name,

        "core_files": [
            files[index].name
            for index in chunk.core_indices
        ],

        "copied_files": [
            files[index].name
            for index in chunk.copied_indices
        ],
    }


ASSIGNMENT_CSV_FIELDS = [
    "dataset_key",
    "dataset_name",
    "dataset_output_name",

    "chunk_id",
    "block_number",

    "core_start_utc",
    "core_end_utc",

    "first_file",
    "last_file",

    "copied_first_file",
    "copied_last_file",

    "core_file_count",
    "copied_file_count",

    "file_duration_seconds",
    "context_padding_seconds",

    "primary_annotator",
    "verification_annotators",
    "is_full_panel_qa_chunk",
]


def assignment_csv_row(
    chunk: Chunk,
) -> dict:
    """Create one row for the shared conceptual chunk assignment CSV."""
    files = chunk.files

    return {
        "dataset_key": chunk.dataset_key,
        "dataset_name": chunk.dataset_name,
        "dataset_output_name": chunk.dataset_output_name,

        "chunk_id": chunk.chunk_id,
        "block_number": chunk.block_number,

        "core_start_utc": chunk.core_start_utc.isoformat(),
        "core_end_utc": chunk.core_end_utc.isoformat(),

        "first_file": files[
            chunk.core_start_index
        ].name,

        "last_file": files[
            chunk.core_end_index
        ].name,

        "copied_first_file": files[
            chunk.copied_start_index
        ].name,

        "copied_last_file": files[
            chunk.copied_end_index
        ].name,

        "core_file_count": len(chunk.core_indices),
        "copied_file_count": len(chunk.copied_indices),

        "file_duration_seconds": chunk.file_duration_seconds,
        "context_padding_seconds": (
            chunk.context_padding_seconds
        ),

        "primary_annotator": chunk.primary_annotator,

        "verification_annotators": ";".join(
            chunk.verification_annotators
        ),

        "is_full_panel_qa_chunk": int(
            bool(chunk.verification_annotators)
        ),
    }


def write_assignment_csv(
    *,
    chunks: list[Chunk],
    output_path: Path,
) -> None:
    """Write chunk assignments in the format used by dataset_profiles.py."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=ASSIGNMENT_CSV_FIELDS,
        )

        writer.writeheader()

        for chunk in sorted(
            chunks,
            key=lambda item: (
                item.dataset_key,
                item.block_number,
            ),
        ):
            writer.writerow(
                assignment_csv_row(chunk)
            )


def write_summary(
    chunks: list[Chunk],
) -> None:
    """Write workload summary by dataset and annotator."""
    MANIFEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = MANIFEST_ROOT / "summary.csv"

    summary: dict[
        tuple[str, str],
        dict[str, float],
    ] = defaultdict(
        lambda: defaultdict(float)
    )

    for chunk in chunks:
        core_hours = (
            len(chunk.core_indices)
            * chunk.file_duration_seconds
            / 3600.0
        )

        copied_hours = (
            len(chunk.copied_indices)
            * chunk.file_duration_seconds
            / 3600.0
        )

        primary_key = (
            chunk.dataset_key,
            chunk.primary_annotator,
        )

        summary[primary_key]["primary_chunks"] += 1
        summary[primary_key]["primary_core_hours"] += core_hours
        summary[primary_key]["primary_copied_hours"] += copied_hours

        for verification_annotator in chunk.verification_annotators:
            verification_key = (
                chunk.dataset_key,
                verification_annotator,
            )

            summary[verification_key][
                "verification_chunks"
            ] += 1

            summary[verification_key][
                "verification_core_hours"
            ] += core_hours

            summary[verification_key][
                "verification_copied_hours"
            ] += copied_hours

    fields = [
        "dataset_key",
        "annotator",

        "primary_chunks",
        "primary_core_hours",
        "primary_copied_hours",

        "verification_chunks",
        "verification_core_hours",
        "verification_copied_hours",

        "total_core_hours",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for dataset_key in sorted(
            {
                dataset_key
                for dataset_key, _ in summary
            }
        ):
            for annotator in ANNOTATORS:
                values = summary[
                    (dataset_key, annotator)
                ]

                primary_core_hours = values[
                    "primary_core_hours"
                ]

                verification_core_hours = values[
                    "verification_core_hours"
                ]

                writer.writerow({
                    "dataset_key": dataset_key,
                    "annotator": annotator,

                    "primary_chunks": int(
                        values["primary_chunks"]
                    ),

                    "primary_core_hours": round(
                        primary_core_hours,
                        4,
                    ),

                    "primary_copied_hours": round(
                        values["primary_copied_hours"],
                        4,
                    ),

                    "verification_chunks": int(
                        values["verification_chunks"]
                    ),

                    "verification_core_hours": round(
                        verification_core_hours,
                        4,
                    ),

                    "verification_copied_hours": round(
                        values["verification_copied_hours"],
                        4,
                    ),

                    "total_core_hours": round(
                        primary_core_hours
                        + verification_core_hours,
                        4,
                    ),
                })

    print(f"Wrote summary: {summary_path}")


# ============================================================================
# COPY / PREFLIGHT
# ============================================================================

def estimate_storage_by_recipient(
    chunks: list[Chunk],
) -> dict[str, int]:
    """
    Estimate copied bytes per annotator drive.

    QA chunks count twice because they are physically copied to two drives.
    """
    totals: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        chunk_bytes = sum(
            chunk.files[index].stat().st_size
            for index in chunk.copied_indices
        )

        for recipient in chunk.recipients:
            totals[recipient] += chunk_bytes

    return dict(totals)


def print_preflight(
    chunks: list[Chunk],
) -> None:
    """Show estimated copy data and free space for each destination drive."""
    estimates = estimate_storage_by_recipient(
        chunks
    )

    print("\n" + "=" * 72)
    print("COPY PREFLIGHT")
    print("=" * 72)

    for annotator in ANNOTATORS:
        output_root = ANNOTATOR_OUTPUT_ROOTS[
            annotator
        ]

        required_bytes = estimates.get(
            annotator,
            0,
        )

        print(
            f"\nAnnotator: {annotator}\n"
            f"Destination: {output_root}\n"
            f"Estimated data to copy: "
            f"{required_bytes / (1024 ** 3):.2f} GiB"
        )

        if not output_root.exists():
            print(
                "WARNING: destination drive is unavailable."
            )
            continue

        free_bytes = shutil.disk_usage(
            output_root
        ).free

        print(
            f"Available free space: "
            f"{free_bytes / (1024 ** 3):.2f} GiB"
        )

        if free_bytes < required_bytes:
            print(
                "WARNING: estimated free space is insufficient."
            )

    print("\n" + "=" * 72)


def copy_one_file(
    *,
    source: Path,
    destination: Path,
) -> None:
    """Copy one source file, skipping existing files unless overwrite enabled."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        if not OVERWRITE_EXISTING:
            return

        destination.unlink()

    if USE_HARDLINKS:
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            # Normal for cross-volume copies.
            pass

    shutil.copy2(
        source,
        destination,
    )


def copy_chunk_to_recipient(
    *,
    chunk: Chunk,
    recipient: str,
) -> None:
    """Copy one conceptual chunk into one recipient's drive."""
    destination_dir = chunk_destination(
        chunk=chunk,
        recipient=recipient,
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seen_names: set[str] = set()

    for index in chunk.copied_indices:
        source_file = chunk.files[index]

        if source_file.name in seen_names:
            raise RuntimeError(
                "Duplicate source basename within chunk: "
                f"{source_file.name}"
            )

        seen_names.add(source_file.name)

        copy_one_file(
            source=source_file,
            destination=destination_dir / source_file.name,
        )

    manifest_data = chunk_manifest_data(
        chunk,
        recipient=recipient,
    )

    manifest_data["destination_dir"] = str(
        destination_dir
    )

    manifest_path = (
        destination_dir
        / "chunk_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest_data,
            file,
            indent=2,
        )


def copy_all_chunks(
    chunks: list[Chunk],
) -> None:
    """Copy primary chunks and any secondary QA copies."""
    copy_jobs = [
        (chunk, recipient)
        for chunk in chunks
        for recipient in chunk.recipients
    ]

    for number, (chunk, recipient) in enumerate(
        copy_jobs,
        start=1,
    ):
        role = (
            "PRIMARY"
            if recipient == chunk.primary_annotator
            else "VERIFICATION"
        )

        destination = chunk_destination(
            chunk=chunk,
            recipient=recipient,
        )

        print(
            f"[{number}/{len(copy_jobs)}] "
            f"{recipient} | {chunk.dataset_output_name} | "
            f"{chunk.chunk_id} | {role}\n"
            f"  Core files: {len(chunk.core_indices)} | "
            f"Copied with padding: {len(chunk.copied_indices)}\n"
            f"  Destination: {destination}"
        )

        copy_chunk_to_recipient(
            chunk=chunk,
            recipient=recipient,
        )

    print("\nFinished copying all chunks.")


def copy_assignment_csv_to_drives(
    *,
    chunks: list[Chunk],
) -> None:
    """
    Copy the same assignment CSV to each annotator drive root.

    dataset_profiles.py expects:

        G:/chunk_assignments.csv
        H:/chunk_assignments.csv
        I:/chunk_assignments.csv
    """
    recipients = {
        recipient
        for chunk in chunks
        for recipient in chunk.recipients
    }

    for annotator in sorted(recipients):
        output_path = (
            ANNOTATOR_OUTPUT_ROOTS[annotator]
            / "chunk_assignments.csv"
        )

        write_assignment_csv(
            chunks=chunks,
            output_path=output_path,
        )

        print(
            "Wrote drive assignment CSV:\n"
            f"  {output_path}"
        )


# ============================================================================
# MAIN
# ============================================================================

def validate_configuration() -> None:
    """Check basic annotator/output-drive configuration."""
    missing_roots = [
        annotator
        for annotator in ANNOTATORS
        if annotator not in ANNOTATOR_OUTPUT_ROOTS
    ]

    if missing_roots:
        raise ValueError(
            "Missing output roots for annotators: "
            f"{missing_roots}"
        )

    if len(ANNOTATORS) < 2:
        raise ValueError(
            "At least two annotators are required."
        )


def main() -> None:
    validate_configuration()

    all_chunks: list[Chunk] = []

    for dataset in DATASETS:
        print(
            "\n"
            + "=" * 72
            + "\n"
            + f"Building chunks: {dataset['dataset_name']}\n"
            + "=" * 72
        )

        dataset_chunks = build_chunks_for_dataset(
            dataset
        )

        all_chunks.extend(dataset_chunks)

        qa_chunks = [
            chunk
            for chunk in dataset_chunks
            if chunk.verification_annotators
        ]

        print(
            f"Created {len(dataset_chunks)} chunks.\n"
            f"Full-panel QA chunks: {len(qa_chunks)}."
        )

        for chunk in qa_chunks:
            print(
                f"  {chunk.chunk_id}: "
                f"primary={chunk.primary_annotator}, "
                f"verification={';'.join(chunk.verification_annotators)}"
            )

    MANIFEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_assignment_csv = (
        MANIFEST_ROOT
        / "chunk_assignments.csv"
    )

    write_assignment_csv(
        chunks=all_chunks,
        output_path=local_assignment_csv,
    )

    print(
        "Wrote local assignment CSV:\n"
        f"  {local_assignment_csv}"
    )

    write_summary(all_chunks)
    print_preflight(all_chunks)

    if not COPY_FILES:
        print(
            "\nCOPY_FILES = False.\n"
            "No source files were copied.\n\n"
            "Review:\n"
            f"  {local_assignment_csv}\n"
            f"  {MANIFEST_ROOT / 'summary.csv'}\n\n"
            "Then set COPY_FILES = True and rerun."
        )
        return

    copy_all_chunks(all_chunks)

    # Must happen after copy because the drives must be connected/available.
    copy_assignment_csv_to_drives(
        chunks=all_chunks,
    )


if __name__ == "__main__":
    main()