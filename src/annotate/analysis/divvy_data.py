"""
Split annotation datasets among interns with pairwise overlap.

For 3 annotators:
    - A/B share one overlap segment
    - A/C share one overlap segment
    - B/C share one overlap segment

Each pair gets OVERLAP_SECONDS of common data. This allows pairwise
inter-annotator agreement / variance calculations.

Files are assigned in approximately 30-minute chronological blocks so each
annotator receives data from throughout the dataset rather than one contiguous
third.

Output layout:

OUTPUT_ROOT/
    manifests/
        assignment_manifest.csv
        summary.csv
    Alice/
        Svalbard/
            <copied files>
        OOI_North_C2/
            <copied files>
    Bob/
        ...
"""

from __future__ import annotations

import csv
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

ANNOTATORS = [
    "xd264",
    "ipf6",
    "mmv52",
]

# Set this to False first if you want to inspect only the CSV manifests.
COPY_FILES = False

# If True, existing copied files will be replaced.
OVERWRITE_EXISTING = False

# Target duration of a chronological assignment block.
BLOCK_SECONDS = 30 * 60  # 30 minutes

# Pairwise overlap duration.
# Each annotator participates in TWO pairwise overlap groups when there are
# three annotators, but their main assignment is reduced so total workload
# remains close to one-third + this amount.
OVERLAP_SECONDS = 10 * 60  # 10 minutes


# IMPORTANT: Change these paths to your actual source and output locations.
DATASETS = [
    {
        "name": "Svalbard",
        "source_dir": Path("/path/to/Svalbard"),
        "output_name": "Svalbard",
        "file_duration_seconds": 10,
        "expected_file_count": 1222,
        "extensions": {".wav", ".flac", ".mp3", ".WAV", ".FLAC", ".MP3"},
    },
    {
        "name": "OOI North C2",
        "source_dir": Path("U:\projects\2022_CLOCCB_OR_S1113\OOI\DASData\OptaSense\North_C2"),
        "output_name": "OOI_North_C2",
        "file_duration_seconds": 30,
        "expected_file_count": 255,
        "extensions": {".wav", ".flac", ".mp3", ".WAV", ".FLAC", ".MP3"},
    },
]

OUTPUT_ROOT = Path("/path/to/annotation_assignments")


# ---------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Assignment:
    dataset: str
    annotator: str
    source_file: Path
    relative_file: Path
    role: str                 # "primary" or "overlap"
    overlap_pair: str         # e.g. "Alice__Bob", blank for primary
    chronological_block: int


# ---------------------------------------------------------------------
# FILE ORDERING
# ---------------------------------------------------------------------

def natural_sort_key(path: Path):
    """
    Natural filename sorting.

    This is suitable if filenames contain sortable timestamps or sequence
    numbers, such as:
        recording_0001.wav
        recording_0002.wav
        2024-01-01T12-00-00.wav

    If your filenames are NOT chronological when sorted alphabetically,
    replace this function with timestamp extraction from your filename format.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"([0-9]+)", str(path))
    ]


def find_files(source_dir: Path, extensions: set[str]) -> list[Path]:
    """Recursively find audio/data files and sort them chronologically."""
    files = [
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix in extensions
    ]
    return sorted(files, key=natural_sort_key)


# ---------------------------------------------------------------------
# ASSIGNMENT LOGIC
# ---------------------------------------------------------------------

def make_blocks(files: list[Path], file_duration_seconds: int) -> list[list[int]]:
    """
    Divide file indices into approximately BLOCK_SECONDS-long chronological
    blocks.
    """
    files_per_block = max(1, BLOCK_SECONDS // file_duration_seconds)

    return [
        list(range(start, min(start + files_per_block, len(files))))
        for start in range(0, len(files), files_per_block)
    ]


def select_pairwise_overlap_indices(
    files: list[Path],
    blocks: list[list[int]],
    file_duration_seconds: int,
    annotators: list[str],
) -> dict[tuple[str, str], list[int]]:
    """
    Select one overlap region for every annotator pair.

    For three annotators and a 10-minute overlap:
        Alice/Bob get 10 min in common
        Alice/Charlie get 10 min in common
        Bob/Charlie get 10 min in common

    Regions are selected from different points in the dataset where possible,
    so overlap samples are temporally distributed.
    """
    overlap_files = math.ceil(OVERLAP_SECONDS / file_duration_seconds)
    pairs = list(combinations(annotators, 2))

    required = overlap_files * len(pairs)
    if required >= len(files):
        raise ValueError(
            f"Dataset has only {len(files)} files, but pairwise overlap requires "
            f"{required} files. Reduce OVERLAP_SECONDS or use fewer annotators."
        )

    result: dict[tuple[str, str], list[int]] = {}
    used_indices: set[int] = set()

    # Spread overlap regions across the recording duration.
    # For 3 pairs, roughly select early / middle / late blocks.
    candidate_block_positions = [
        round(i * (len(blocks) - 1) / max(1, len(pairs) - 1))
        for i in range(len(pairs))
    ]

    for pair_number, pair in enumerate(pairs):
        preferred_block_index = candidate_block_positions[pair_number]

        # Search preferred block first, then nearby blocks.
        search_order = sorted(
            range(len(blocks)),
            key=lambda i: abs(i - preferred_block_index),
        )

        chosen_indices = None

        for block_index in search_order:
            block = blocks[block_index]

            # Find contiguous available files in the block.
            available_runs = []
            current_run = []

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
                if len(run) >= overlap_files:
                    # Center the overlap subsection in this available run.
                    start = (len(run) - overlap_files) // 2
                    chosen_indices = run[start:start + overlap_files]
                    break

            if chosen_indices is not None:
                break

        if chosen_indices is None:
            raise RuntimeError(
                f"Could not find {overlap_files} contiguous files for overlap "
                f"pair {pair}."
            )

        result[pair] = chosen_indices
        used_indices.update(chosen_indices)

    return result


def assign_primary_files(
    files: list[Path],
    blocks: list[list[int]],
    excluded_indices: set[int],
    annotators: list[str],
) -> dict[str, list[tuple[int, int]]]:
    """
    Assign all non-overlap files to annotators.

    Assignment is performed in chronological block fragments. The annotator
    with the currently smallest primary-file count receives the next fragment.

    This gives reasonably balanced workload while distributing recording time
    among annotators.
    """
    assignments: dict[str, list[tuple[int, int]]] = defaultdict(list)
    primary_counts: dict[str, int] = {name: 0 for name in annotators}

    tie_breaker_position = 0

    for block_number, block in enumerate(blocks):
        # Split a block into contiguous pieces around excluded overlap files.
        fragments = []
        current_fragment = []

        for file_index in block:
            if file_index in excluded_indices:
                if current_fragment:
                    fragments.append(current_fragment)
                    current_fragment = []
            else:
                current_fragment.append(file_index)

        if current_fragment:
            fragments.append(current_fragment)

        for fragment in fragments:
            if not fragment:
                continue

            # Choose least-loaded annotator.
            min_count = min(primary_counts.values())
            candidates = [
                a for a in annotators
                if primary_counts[a] == min_count
            ]

            # Rotate ties so the same person does not always win.
            annotator = candidates[tie_breaker_position % len(candidates)]
            tie_breaker_position += 1

            for file_index in fragment:
                assignments[annotator].append((file_index, block_number))

            primary_counts[annotator] += len(fragment)

    return assignments


def build_assignments_for_dataset(dataset: dict) -> list[Assignment]:
    """Create all primary and overlap assignments for one dataset."""
    source_dir = dataset["source_dir"]
    files = find_files(source_dir, dataset["extensions"])

    if not files:
        raise FileNotFoundError(
            f"No matching files found in: {source_dir}\n"
            f"Extensions searched: {dataset['extensions']}"
        )

    expected_count = dataset.get("expected_file_count")
    if expected_count is not None and len(files) != expected_count:
        print(
            f"WARNING: {dataset['name']} expected {expected_count} files "
            f"but found {len(files)} files."
        )

    file_duration = dataset["file_duration_seconds"]
    blocks = make_blocks(files, file_duration)

    overlap_by_pair = select_pairwise_overlap_indices(
        files=files,
        blocks=blocks,
        file_duration_seconds=file_duration,
        annotators=ANNOTATORS,
    )

    excluded_indices = {
        index
        for indices in overlap_by_pair.values()
        for index in indices
    }

    primary_by_annotator = assign_primary_files(
        files=files,
        blocks=blocks,
        excluded_indices=excluded_indices,
        annotators=ANNOTATORS,
    )

    assignments: list[Assignment] = []

    # Lookup chronological block for each file.
    file_to_block = {}
    for block_number, block in enumerate(blocks):
        for file_index in block:
            file_to_block[file_index] = block_number

    # Add normal, non-overlap assignments.
    for annotator, file_entries in primary_by_annotator.items():
        for file_index, block_number in file_entries:
            source_file = files[file_index]
            assignments.append(
                Assignment(
                    dataset=dataset["name"],
                    annotator=annotator,
                    source_file=source_file,
                    relative_file=source_file.relative_to(source_dir),
                    role="primary",
                    overlap_pair="",
                    chronological_block=block_number,
                )
            )

    # Add pairwise overlap assignments.
    for pair, indices in overlap_by_pair.items():
        pair_label = "__".join(pair)

        for annotator in pair:
            for file_index in indices:
                source_file = files[file_index]
                assignments.append(
                    Assignment(
                        dataset=dataset["name"],
                        annotator=annotator,
                        source_file=source_file,
                        relative_file=source_file.relative_to(source_dir),
                        role="overlap",
                        overlap_pair=pair_label,
                        chronological_block=file_to_block[file_index],
                    )
                )

    return assignments


# ---------------------------------------------------------------------
# OUTPUT / COPYING
# ---------------------------------------------------------------------

def write_manifest(assignments: list[Assignment]) -> None:
    """Write a detailed file-level assignment manifest."""
    manifest_dir = OUTPUT_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "assignment_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "annotator",
                "role",
                "overlap_pair",
                "chronological_block",
                "source_file",
                "relative_file",
            ],
        )
        writer.writeheader()

        for assignment in sorted(
            assignments,
            key=lambda a: (
                a.dataset,
                a.annotator,
                a.chronological_block,
                str(a.source_file),
            ),
        ):
            writer.writerow(
                {
                    "dataset": assignment.dataset,
                    "annotator": assignment.annotator,
                    "role": assignment.role,
                    "overlap_pair": assignment.overlap_pair,
                    "chronological_block": assignment.chronological_block,
                    "source_file": str(assignment.source_file),
                    "relative_file": str(assignment.relative_file),
                }
            )

    print(f"Wrote manifest: {manifest_path}")


def write_summary(assignments: list[Assignment], datasets: list[dict]) -> None:
    """Write workload summary by dataset and annotator."""
    dataset_duration_lookup = {
        d["name"]: d["file_duration_seconds"]
        for d in datasets
    }

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for assignment in assignments:
        counts[assignment.dataset][assignment.annotator][assignment.role] += 1

    summary_path = OUTPUT_ROOT / "manifests" / "summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "annotator",
                "primary_files",
                "overlap_files",
                "total_files",
                "primary_hours",
                "overlap_hours",
                "total_hours",
            ],
        )
        writer.writeheader()

        for dataset_name in sorted(counts):
            duration = dataset_duration_lookup[dataset_name]

            for annotator in ANNOTATORS:
                primary = counts[dataset_name][annotator]["primary"]
                overlap = counts[dataset_name][annotator]["overlap"]
                total = primary + overlap

                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "annotator": annotator,
                        "primary_files": primary,
                        "overlap_files": overlap,
                        "total_files": total,
                        "primary_hours": round(primary * duration / 3600, 4),
                        "overlap_hours": round(overlap * duration / 3600, 4),
                        "total_hours": round(total * duration / 3600, 4),
                    }
                )

    print(f"Wrote summary:  {summary_path}")


def copy_assignment_files(assignments: list[Assignment], datasets: list[dict]) -> None:
    """Copy files into per-annotator/per-dataset directories."""
    dataset_output_names = {
        d["name"]: d["output_name"]
        for d in datasets
    }

    for assignment in assignments:
        destination = (
            OUTPUT_ROOT
            / assignment.annotator
            / dataset_output_names[assignment.dataset]
            / assignment.relative_file
        )

        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not OVERWRITE_EXISTING:
            continue

        shutil.copy2(assignment.source_file, destination)

    print(f"Copied files under: {OUTPUT_ROOT}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    if len(ANNOTATORS) < 2:
        raise ValueError("You need at least two annotators.")

    if len(ANNOTATORS) != 3:
        print(
            "NOTE: This script works with any number of annotators, but the "
            "workload assumptions in your table appear to be based on 3."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_assignments: list[Assignment] = []

    for dataset in DATASETS:
        print(f"\nProcessing: {dataset['name']}")
        dataset_assignments = build_assignments_for_dataset(dataset)
        all_assignments.extend(dataset_assignments)

    write_manifest(all_assignments)
    write_summary(all_assignments, DATASETS)

    if COPY_FILES:
        copy_assignment_files(all_assignments, DATASETS)
    else:
        print("\nCOPY_FILES = False; no files were copied.")


if __name__ == "__main__":
    main()