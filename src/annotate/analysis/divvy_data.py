"""
Create chunk-based DAS annotation assignments with context padding and
pairwise inter-annotator overlap.

Each annotator receives:

- Primary chunks:
    Approximately 30-minute core regions assigned to one annotator.

- Pairwise overlap chunks:
    Shared core regions assigned to exactly two annotators, allowing
    inter-annotator agreement analysis.

Every chunk receives copied context before and after its core region.

This avoids artificial data gaps within one GUI-opened chunk and allows
calls near chunk boundaries to be viewed/annotated completely.

Output layout
-------------

E:\
    Svalbard\
        chunk_primary_b000_f00\
            *.hdf5
            chunk_manifest.json
        chunk_overlap_xd264__ipf6_b003\
            *.hdf5
            chunk_manifest.json

I:\
    ...

M:\
    ...

A local manifest directory is also written beside this script:

    divvy_manifests/
        chunk_manifest.csv
        summary.csv
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path


# ============================================================================
# USER SETTINGS
# ============================================================================

ANNOTATORS = [
    "xd264",
    "ipf6",
    "mmv52",
]

# Physical destination roots for each annotator's drive.
ANNOTATOR_OUTPUT_ROOTS: dict[str, Path] = {
    "xd264": Path(r"E:"),
    "ipf6": Path(r"I:"),
    "mmv52": Path(r"M:"),
}

# Start with False for a dry run. The script will still build manifests and
# report estimated storage use, but it will not copy files.
#
# Change to True when you have reviewed the output.
COPY_FILES = False

# If False, existing destination files are retained. This makes interrupted
# weekend copy jobs reasonably safe to resume by re-running the script.
OVERWRITE_EXISTING = False

# If True, source files are copied with shutil.copy2().
#
# If False, hard links are attempted instead. Hard links only work if source
# and destination are on the same NTFS volume, so keep this True for copying
# from U: to E:/I:/M:.
USE_HARDLINKS = False

# Core target duration for normal chronological assignment blocks.
BLOCK_SECONDS = 30 * 60  # 30 minutes

# Pairwise shared overlap duration.
OVERLAP_SECONDS = 10 * 60  # 10 minutes

# Extra data copied before and after every chunk's core assignment.
#
# Your normal plot duration is about 35-40 seconds. 60 seconds gives enough
# room to inspect/label calls near artificial chunk boundaries.
#
# Increase to 90 or 120 seconds if calls may be long or if you want more
# context for filter/F-K edge effects.
CONTEXT_PADDING_SECONDS = 60

# Manifest directory written on the machine running this script.
# This is not placed on an intern drive by default.
MANIFEST_ROOT = Path(__file__).resolve().parent / "divvy_manifests"


# ============================================================================
# DATASET SETTINGS
# ============================================================================

DATASETS = [
    {
        "name": "Svalbard",
        "source_dir": Path(
            r"U:\projects\2022_CLOCCB_OR_S1113\Svalbard\data"
        ),
        "output_name": "Svalbard",
        "file_duration_seconds": 10,
        "expected_file_count": 1222,
        "extensions": {".h5", ".hdf5"},
    },
    {
        "name": "OOI North C2",
        "source_dir": Path(
            r"U:\projects\2022_CLOCCB_OR_S1113\OOI\DASData"
            r"\OptaSense\North_C2"
        ),
        "output_name": "OOI_North_C2",
        "file_duration_seconds": 30,
        "expected_file_count": 255,
        "extensions": {".h5", ".hdf5"},
    },
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class ChunkAssignment:
    """
    One contiguous core assignment plus copied context on one annotator drive.

    core_indices:
        File indices the annotator is responsible for labeling.

    copied_indices:
        core_indices plus before/after contextual padding.
    """

    dataset: str
    dataset_output_name: str
    annotator: str

    chunk_id: str

    role: str
    overlap_pair: str

    chronological_block: int
    fragment_index: int

    source_dir: Path
    files: tuple[Path, ...]

    core_indices: tuple[int, ...]
    copied_indices: tuple[int, ...]

    file_duration_seconds: float
    context_padding_seconds: float

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
# FILE ORDERING
# ============================================================================

def natural_sort_key(path: Path) -> list[object]:
    """
    Natural filename sort key.

    Suitable for names such as:
        090007.hdf5
        120107.hdf5
        recording_0001.h5
        2021-11-02T215901Z.h5

    Verify that sorted output matches chronological acquisition order before
    using this on unfamiliar DAS datasets.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"([0-9]+)", path.name)
    ]


def find_files(
    source_dir: Path,
    extensions: set[str],
) -> list[Path]:
    """Find supported source files recursively and sort naturally."""
    extensions = {
        extension.lower()
        for extension in extensions
    }

    files = [
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    ]

    return sorted(
        files,
        key=natural_sort_key,
    )


# ============================================================================
# ASSIGNMENT LOGIC
# ============================================================================

def make_blocks(
    files: list[Path],
    file_duration_seconds: float,
) -> list[list[int]]:
    """
    Divide source-file indices into approximately BLOCK_SECONDS chunks.
    """
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
                    len(files),
                ),
            )
        )
        for start in range(
            0,
            len(files),
            files_per_block,
        )
    ]


def select_pairwise_overlap_indices(
    *,
    files: list[Path],
    blocks: list[list[int]],
    file_duration_seconds: float,
    annotators: list[str],
) -> dict[tuple[str, str], list[int]]:
    """
    Select one contiguous overlap region for every annotator pair.

    With three annotators:

        xd264/ipf6
        xd264/mmv52
        ipf6/mmv52

    each receives one common overlap segment of OVERLAP_SECONDS.
    """
    overlap_files = math.ceil(
        OVERLAP_SECONDS / file_duration_seconds
    )

    pairs = list(
        combinations(
            annotators,
            2,
        )
    )

    required_files = overlap_files * len(pairs)

    if required_files >= len(files):
        raise ValueError(
            f"Dataset has {len(files)} files, but pairwise overlap "
            f"requires at least {required_files} files. Reduce "
            "OVERLAP_SECONDS or use fewer annotators."
        )

    result: dict[tuple[str, str], list[int]] = {}
    used_indices: set[int] = set()

    # Spread overlap regions across the dataset approximately early/middle/late.
    candidate_block_positions = [
        round(
            i * (len(blocks) - 1)
            / max(1, len(pairs) - 1)
        )
        for i in range(len(pairs))
    ]

    for pair_number, pair in enumerate(pairs):
        preferred_block_index = candidate_block_positions[pair_number]

        search_order = sorted(
            range(len(blocks)),
            key=lambda index: abs(
                index - preferred_block_index
            ),
        )

        chosen_indices: list[int] | None = None

        for block_index in search_order:
            block = blocks[block_index]

            available_runs: list[list[int]] = []
            current_run: list[int] = []

            for file_index in block:
                if file_index not in used_indices:
                    current_run.append(file_index)
                else:
                    if current_run:
                        available_runs.append(current_run)
                        current_run = []

            if current_run:
                available_runs.append(current_run)

            for run in available_runs:
                if len(run) < overlap_files:
                    continue

                # Choose a centered contiguous portion of this run.
                run_start = (
                    len(run) - overlap_files
                ) // 2

                chosen_indices = run[
                    run_start:run_start + overlap_files
                ]

                break

            if chosen_indices is not None:
                break

        if chosen_indices is None:
            raise RuntimeError(
                f"Could not find a contiguous {OVERLAP_SECONDS}-second "
                f"overlap region for pair {pair}."
            )

        result[pair] = chosen_indices
        used_indices.update(chosen_indices)

    return result


def assign_primary_fragments(
    *,
    blocks: list[list[int]],
    excluded_indices: set[int],
    annotators: list[str],
) -> list[tuple[str, int, int, list[int]]]:
    """
    Assign each non-overlap contiguous fragment to one annotator.

    Returns tuples:

        annotator,
        chronological_block,
        fragment_index,
        core_file_indices

    Keeping fragments separate produces physically contiguous chunk folders.
    """
    primary_counts: dict[str, int] = {
        annotator: 0
        for annotator in annotators
    }

    assignments: list[
        tuple[str, int, int, list[int]]
    ] = []

    tie_breaker_position = 0

    for block_number, block in enumerate(blocks):
        fragments: list[list[int]] = []
        current_fragment: list[int] = []

        for file_index in block:
            if file_index in excluded_indices:
                if current_fragment:
                    fragments.append(current_fragment)
                    current_fragment = []
            else:
                current_fragment.append(file_index)

        if current_fragment:
            fragments.append(current_fragment)

        for fragment_index, fragment in enumerate(fragments):
            if not fragment:
                continue

            min_count = min(
                primary_counts.values()
            )

            candidates = [
                annotator
                for annotator in annotators
                if primary_counts[annotator] == min_count
            ]

            annotator = candidates[
                tie_breaker_position % len(candidates)
            ]

            tie_breaker_position += 1

            assignments.append((
                annotator,
                block_number,
                fragment_index,
                fragment,
            ))

            primary_counts[annotator] += len(fragment)

    return assignments


def padded_indices(
    *,
    core_indices: list[int],
    total_files: int,
    file_duration_seconds: float,
    padding_seconds: float,
) -> list[int]:
    """
    Return contiguous copied-file indices for a core assignment plus context.
    """
    if not core_indices:
        return []

    padding_files = math.ceil(
        padding_seconds / file_duration_seconds
    )

    copied_start = max(
        0,
        min(core_indices) - padding_files,
    )

    copied_end = min(
        total_files - 1,
        max(core_indices) + padding_files,
    )

    return list(
        range(
            copied_start,
            copied_end + 1,
        )
    )


def find_block_number_for_index(
    *,
    blocks: list[list[int]],
    file_index: int,
) -> int:
    """Return chronological block containing one source-file index."""
    for block_number, block in enumerate(blocks):
        if file_index in block:
            return block_number

    raise KeyError(
        f"Could not find block for file index {file_index}."
    )


def build_chunk_assignments_for_dataset(
    dataset: dict,
) -> list[ChunkAssignment]:
    """
    Create primary and pairwise-overlap chunk assignments for one dataset.
    """
    source_dir = Path(dataset["source_dir"])
    files = find_files(
        source_dir,
        dataset["extensions"],
    )

    if not files:
        raise FileNotFoundError(
            f"No matching files found in:\n{source_dir}\n"
            f"Extensions searched: {dataset['extensions']}"
        )

    expected_count = dataset.get("expected_file_count")

    if (
        expected_count is not None
        and len(files) != expected_count
    ):
        print(
            f"WARNING: {dataset['name']} expected "
            f"{expected_count} files but found {len(files)}."
        )

    file_duration_seconds = float(
        dataset["file_duration_seconds"]
    )

    blocks = make_blocks(
        files,
        file_duration_seconds,
    )

    overlap_by_pair = select_pairwise_overlap_indices(
        files=files,
        blocks=blocks,
        file_duration_seconds=file_duration_seconds,
        annotators=ANNOTATORS,
    )

    excluded_indices = {
        index
        for indices in overlap_by_pair.values()
        for index in indices
    }

    primary_fragments = assign_primary_fragments(
        blocks=blocks,
        excluded_indices=excluded_indices,
        annotators=ANNOTATORS,
    )

    assignments: list[ChunkAssignment] = []

    # ------------------------------------------------------------------
    # Primary chunks.
    # ------------------------------------------------------------------
    for (
        annotator,
        block_number,
        fragment_index,
        core_indices,
    ) in primary_fragments:

        copied = padded_indices(
            core_indices=core_indices,
            total_files=len(files),
            file_duration_seconds=file_duration_seconds,
            padding_seconds=CONTEXT_PADDING_SECONDS,
        )

        chunk_id = (
            f"chunk_primary_b{block_number:03d}"
            f"_f{fragment_index:02d}"
        )

        assignments.append(
            ChunkAssignment(
                dataset=dataset["name"],
                dataset_output_name=dataset["output_name"],
                annotator=annotator,
                chunk_id=chunk_id,
                role="primary",
                overlap_pair="",
                chronological_block=block_number,
                fragment_index=fragment_index,
                source_dir=source_dir,
                files=tuple(files),
                core_indices=tuple(core_indices),
                copied_indices=tuple(copied),
                file_duration_seconds=file_duration_seconds,
                context_padding_seconds=CONTEXT_PADDING_SECONDS,
            )
        )

    # ------------------------------------------------------------------
    # Pairwise overlap chunks.
    # ------------------------------------------------------------------
    for pair, core_indices in overlap_by_pair.items():
        pair_label = "__".join(pair)

        block_number = find_block_number_for_index(
            blocks=blocks,
            file_index=core_indices[0],
        )

        copied = padded_indices(
            core_indices=core_indices,
            total_files=len(files),
            file_duration_seconds=file_duration_seconds,
            padding_seconds=CONTEXT_PADDING_SECONDS,
        )

        chunk_id = (
            f"chunk_overlap_{pair_label}"
            f"_b{block_number:03d}"
        )

        for annotator in pair:
            assignments.append(
                ChunkAssignment(
                    dataset=dataset["name"],
                    dataset_output_name=dataset["output_name"],
                    annotator=annotator,
                    chunk_id=chunk_id,
                    role="overlap",
                    overlap_pair=pair_label,
                    chronological_block=block_number,
                    fragment_index=0,
                    source_dir=source_dir,
                    files=tuple(files),
                    core_indices=tuple(core_indices),
                    copied_indices=tuple(copied),
                    file_duration_seconds=file_duration_seconds,
                    context_padding_seconds=CONTEXT_PADDING_SECONDS,
                )
            )

    return assignments


# ============================================================================
# OUTPUT / MANIFESTS
# ============================================================================

def chunk_destination(
    assignment: ChunkAssignment,
) -> Path:
    """Return output folder for one chunk assignment."""
    output_root = ANNOTATOR_OUTPUT_ROOTS[
        assignment.annotator
    ]

    return (
        output_root
        / assignment.dataset_output_name
        / assignment.chunk_id
    )


def chunk_manifest_data(
    assignment: ChunkAssignment,
) -> dict:
    """
    Return serializable chunk metadata.

    This records responsibility boundaries separately from copied context.
    """
    files = assignment.files

    return {
        "dataset": assignment.dataset,
        "dataset_output_name": assignment.dataset_output_name,
        "annotator": assignment.annotator,

        "chunk_id": assignment.chunk_id,
        "role": assignment.role,
        "overlap_pair": assignment.overlap_pair,

        "chronological_block": assignment.chronological_block,
        "fragment_index": assignment.fragment_index,

        "file_duration_seconds": assignment.file_duration_seconds,
        "context_padding_seconds": assignment.context_padding_seconds,

        "core_start_index": assignment.core_start_index,
        "core_end_index": assignment.core_end_index,
        "core_file_count": len(assignment.core_indices),

        "copied_start_index": assignment.copied_start_index,
        "copied_end_index": assignment.copied_end_index,
        "copied_file_count": len(assignment.copied_indices),

        "core_first_file": files[
            assignment.core_start_index
        ].name,
        "core_last_file": files[
            assignment.core_end_index
        ].name,

        "copied_first_file": files[
            assignment.copied_start_index
        ].name,
        "copied_last_file": files[
            assignment.copied_end_index
        ].name,

        # File names only, because copied chunk folders are intentionally flat.
        "core_files": [
            files[index].name
            for index in assignment.core_indices
        ],

        "copied_files": [
            files[index].name
            for index in assignment.copied_indices
        ],
    }


def write_global_chunk_manifest(
    assignments: list[ChunkAssignment],
) -> None:
    """Write one row per chunk assignment."""
    MANIFEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        MANIFEST_ROOT
        / "chunk_manifest.csv"
    )

    fieldnames = [
        "dataset",
        "dataset_output_name",
        "annotator",
        "chunk_id",
        "role",
        "overlap_pair",
        "chronological_block",
        "fragment_index",
        "file_duration_seconds",
        "context_padding_seconds",
        "core_start_index",
        "core_end_index",
        "core_file_count",
        "copied_start_index",
        "copied_end_index",
        "copied_file_count",
        "core_first_file",
        "core_last_file",
        "copied_first_file",
        "copied_last_file",
        "destination",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for assignment in sorted(
            assignments,
            key=lambda item: (
                item.dataset,
                item.annotator,
                item.chronological_block,
                item.chunk_id,
            ),
        ):
            metadata = chunk_manifest_data(
                assignment
            )

            writer.writerow({
                key: metadata.get(key, "")
                for key in fieldnames
                if key != "destination"
            } | {
                "destination": str(
                    chunk_destination(assignment)
                )
            })

    print(
        f"Wrote global chunk manifest:\n"
        f"  {manifest_path}"
    )


def write_summary(
    assignments: list[ChunkAssignment],
) -> None:
    """Write core/context workload summary."""
    MANIFEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = MANIFEST_ROOT / "summary.csv"

    counts = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(float)
        )
    )

    for assignment in assignments:
        duration = assignment.file_duration_seconds

        counts[
            assignment.dataset
        ][
            assignment.annotator
        ][
            f"{assignment.role}_core_files"
        ] += len(assignment.core_indices)

        counts[
            assignment.dataset
        ][
            assignment.annotator
        ][
            f"{assignment.role}_copied_files"
        ] += len(assignment.copied_indices)

        counts[
            assignment.dataset
        ][
            assignment.annotator
        ][
            f"{assignment.role}_core_seconds"
        ] += len(assignment.core_indices) * duration

    fieldnames = [
        "dataset",
        "annotator",

        "primary_core_files",
        "primary_copied_files",
        "primary_core_hours",

        "overlap_core_files",
        "overlap_copied_files",
        "overlap_core_hours",

        "total_core_hours",
        "chunk_count",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for dataset_name in sorted(counts):
            for annotator in ANNOTATORS:
                values = counts[dataset_name][annotator]

                primary_core_files = int(
                    values["primary_core_files"]
                )
                primary_copied_files = int(
                    values["primary_copied_files"]
                )

                overlap_core_files = int(
                    values["overlap_core_files"]
                )
                overlap_copied_files = int(
                    values["overlap_copied_files"]
                )

                primary_core_hours = (
                    values["primary_core_seconds"]
                    / 3600.0
                )

                overlap_core_hours = (
                    values["overlap_core_seconds"]
                    / 3600.0
                )

                chunk_count = sum(
                    1
                    for assignment in assignments
                    if (
                        assignment.dataset == dataset_name
                        and assignment.annotator == annotator
                    )
                )

                writer.writerow({
                    "dataset": dataset_name,
                    "annotator": annotator,

                    "primary_core_files": primary_core_files,
                    "primary_copied_files": primary_copied_files,
                    "primary_core_hours": round(
                        primary_core_hours,
                        4,
                    ),

                    "overlap_core_files": overlap_core_files,
                    "overlap_copied_files": overlap_copied_files,
                    "overlap_core_hours": round(
                        overlap_core_hours,
                        4,
                    ),

                    "total_core_hours": round(
                        primary_core_hours
                        + overlap_core_hours,
                        4,
                    ),

                    "chunk_count": chunk_count,
                })

    print(
        f"Wrote summary:\n"
        f"  {summary_path}"
    )


def write_chunk_manifest(
    assignment: ChunkAssignment,
    destination_dir: Path,
) -> None:
    """Write one JSON manifest inside a destination chunk folder."""
    manifest_path = (
        destination_dir
        / "chunk_manifest.json"
    )

    metadata = chunk_manifest_data(
        assignment
    )

    metadata["destination_dir"] = str(
        destination_dir
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


# ============================================================================
# COPY / PREFLIGHT
# ============================================================================

def estimate_storage_by_annotator(
    assignments: list[ChunkAssignment],
) -> dict[str, int]:
    """
    Estimate copied bytes per annotator.

    This intentionally counts duplicated context files once per chunk because
    each chunk has its own physical folder.
    """
    totals: dict[str, int] = defaultdict(int)

    for assignment in assignments:
        for index in assignment.copied_indices:
            totals[
                assignment.annotator
            ] += assignment.files[index].stat().st_size

    return dict(totals)


def print_preflight(
    assignments: list[ChunkAssignment],
) -> None:
    """Print estimated copy volume and available drive space."""
    estimated_bytes = estimate_storage_by_annotator(
        assignments
    )

    print("\n" + "=" * 72)
    print("COPY PREFLIGHT")
    print("=" * 72)

    for annotator in ANNOTATORS:
        output_root = ANNOTATOR_OUTPUT_ROOTS[annotator]

        required_bytes = estimated_bytes.get(
            annotator,
            0,
        )

        print(
            f"\nAnnotator: {annotator}\n"
            f"Destination: {output_root}\n"
            f"Estimated copied data: "
            f"{required_bytes / (1024 ** 3):.2f} GiB"
        )

        if not output_root.exists():
            print(
                "WARNING: destination drive/path does not exist."
            )
            continue

        try:
            free_bytes = shutil.disk_usage(
                output_root
            ).free

            print(
                f"Available free space: "
                f"{free_bytes / (1024 ** 3):.2f} GiB"
            )

            if free_bytes < required_bytes:
                print(
                    "WARNING: insufficient estimated free space."
                )

        except OSError as exc:
            print(
                f"WARNING: could not inspect free space: {exc}"
            )

    print("\n" + "=" * 72)


def copy_one_file(
    source: Path,
    destination: Path,
) -> None:
    """Copy one source file, optionally using a hard link."""
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
            # Expected when source/destination are on different drives.
            pass

    shutil.copy2(
        source,
        destination,
    )


def copy_chunk(
    assignment: ChunkAssignment,
) -> None:
    """
    Copy all core/context files for one chunk and write its JSON manifest.

    Files are intentionally copied directly into the chunk directory rather
    than preserving source nested folder structure. DatasetService currently
    searches only directly inside the selected dataset directory.
    """
    destination_dir = chunk_destination(
        assignment
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_names_seen: set[str] = set()

    for index in assignment.copied_indices:
        source_file = assignment.files[index]

        # A flat destination is intentional. Detect duplicate basenames rather
        # than silently overwriting unrelated source files.
        if source_file.name in source_names_seen:
            raise RuntimeError(
                "Duplicate source filename within chunk: "
                f"{source_file.name}"
            )

        source_names_seen.add(
            source_file.name
        )

        destination_file = (
            destination_dir
            / source_file.name
        )

        copy_one_file(
            source_file,
            destination_file,
        )

    write_chunk_manifest(
        assignment,
        destination_dir,
    )


def copy_all_chunks(
    assignments: list[ChunkAssignment],
) -> None:
    """Copy every assigned chunk to its annotator-specific drive."""
    total_chunks = len(assignments)

    for number, assignment in enumerate(
        sorted(
            assignments,
            key=lambda item: (
                item.annotator,
                item.dataset,
                item.chronological_block,
                item.chunk_id,
            ),
        ),
        start=1,
    ):
        destination_dir = chunk_destination(
            assignment
        )

        print(
            f"[{number}/{total_chunks}] "
            f"{assignment.annotator} | "
            f"{assignment.dataset_output_name} | "
            f"{assignment.chunk_id}\n"
            f"  Core: "
            f"{len(assignment.core_indices)} files | "
            f"Copied with context: "
            f"{len(assignment.copied_indices)} files\n"
            f"  Destination: {destination_dir}"
        )

        copy_chunk(assignment)

    print("\nFinished copying all chunk assignments.")


# ============================================================================
# MAIN
# ============================================================================

def validate_configuration() -> None:
    """Validate annotator names and destination-drive mappings."""
    missing = [
        annotator
        for annotator in ANNOTATORS
        if annotator not in ANNOTATOR_OUTPUT_ROOTS
    ]

    if missing:
        raise ValueError(
            "Missing destination roots for annotators: "
            f"{missing}"
        )

    if len(ANNOTATORS) < 2:
        raise ValueError(
            "At least two annotators are required."
        )


def main() -> None:
    validate_configuration()

    if len(ANNOTATORS) != 3:
        print(
            "NOTE: The code supports other annotator counts, but current "
            "workload assumptions are designed around three annotators."
        )

    all_assignments: list[ChunkAssignment] = []

    for dataset in DATASETS:
        print(
            "\n"
            + "=" * 72
            + "\n"
            + f"Building assignments: {dataset['name']}\n"
            + "=" * 72
        )

        dataset_assignments = (
            build_chunk_assignments_for_dataset(
                dataset
            )
        )

        all_assignments.extend(
            dataset_assignments
        )

        print(
            f"Created {len(dataset_assignments)} chunk assignments "
            f"for {dataset['name']}."
        )

    write_global_chunk_manifest(
        all_assignments
    )

    write_summary(
        all_assignments
    )

    print_preflight(
        all_assignments
    )

    if not COPY_FILES:
        print(
            "\nCOPY_FILES = False.\n"
            "No DAS files were copied.\n"
            "Review divvy_manifests/chunk_manifest.csv and summary.csv, "
            "then change COPY_FILES to True."
        )
        return

    copy_all_chunks(
        all_assignments
    )


if __name__ == "__main__":
    main()