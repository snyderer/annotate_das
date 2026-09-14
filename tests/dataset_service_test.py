from pathlib import Path

from annotate.data.dataset_service import DatasetService


FILEPATH = Path(r"C:\Users\ers334\Desktop\testingData\OOI\DASData\OptaSense\North_C2\North-C2-HF-P1kHz-GL30m-Sp2m-FS500Hz_2021-11-02T215901Z.h5")

# Use a known interrogator first. Test Auto separately once this works.
INTERROGATOR = "optasense"
# INTERROGATOR = "auto"


def main():
    service = DatasetService()

    selected_record = service.open_dataset(
        selected_file=FILEPATH,
        requested_interrogator=INTERROGATOR,
    )

    print()
    print("=== Dataset service test ===")
    print(f"Dataset directory:     {service.dataset_directory}")
    print(f"Resolved interrogator: {service.interrogator}")
    print(f"Interrogator source:   {service.interrogator_source}")
    print(f"Indexed files:         {len(service.file_records)}")
    print(f"Index failures:        {len(service.index_failures)}")
    print()

    print("Selected file record:")
    print(f"  File:       {selected_record.path.name}")
    print(f"  Start:      {selected_record.start_time.isoformat()}")
    print(f"  End:        {selected_record.end_time.isoformat()}")
    print(f"  Duration:   {(selected_record.end_time - selected_record.start_time).total_seconds():.3f} s")
    print(f"  fs:         {selected_record.fs_hz} Hz")
    print(f"  dx:         {selected_record.dx_m} m")
    print(f"  Samples:    {selected_record.n_samples}")
    print(f"  Channels:   {selected_record.n_channels}")
    print()

    print("First 10 indexed files:")
    for record in service.file_records[:10]:
        duration_s = (record.end_time - record.start_time).total_seconds()

        print(
            f"  {record.path.name}\n"
            f"    {record.start_time.isoformat()} -> "
            f"{record.end_time.isoformat()} "
            f"({duration_s:.3f} s, "
            f"fs={record.fs_hz}, dx={record.dx_m})"
        )

    print()

    if service.index_failures:
        print("Index failures:")
        for failure in service.index_failures:
            print(f"  - {failure}")
        print()

    # Verify that the selected path is represented in the index.
    assert selected_record.path == FILEPATH.resolve()

    # Check time lookup at the selected file's start time.
    found = service.find_file(selected_record.start_time)

    assert found is not None, "find_file() could not find selected file start."
    assert found.path == selected_record.path

    # Ensure index ordering is correct.
    start_times = [record.start_time for record in service.file_records]
    assert start_times == sorted(start_times)

    # Check each record has sensible coverage.
    for record in service.file_records:
        assert record.end_time > record.start_time
        assert record.fs_hz > 0
        assert record.dx_m > 0
        assert record.n_samples > 0
        assert record.n_channels > 0

    print("DatasetService index test passed.")


if __name__ == "__main__":
    main()