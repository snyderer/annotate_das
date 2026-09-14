from datetime import timedelta
from pathlib import Path

from annotate.data.dataset_service import DatasetService


# FILEPATH = Path(r"C:\Users\ers334\Desktop\testingData\OOI\DASData\OptaSense\North_C2\North-C2-HF-P1kHz-GL30m-Sp2m-FS500Hz_2021-11-02T215901Z.h5")

FILEPATH = Path(r"C:\Users\ers334\Desktop\testingData\Svalbard\data\120057.hdf5")
INTERROGATOR = 'auto'  # e.g. "asn", "optasense", "fosina_dxs"
SELECTED_CHANNELS = (0, 10000, 4)

def main():
    service = DatasetService()

    selected_record = service.open_dataset(
        selected_file=FILEPATH,
        requested_interrogator=INTERROGATOR,
    )

    # Test a window that starts 10 seconds into the selected file.
    requested_start = selected_record.start_time + timedelta(seconds=10)
    requested_duration_s = 37.2

    segment = service.load_window(
        start_time=requested_start,
        duration_s=requested_duration_s,
        selected_channels=SELECTED_CHANNELS,
    )

    print("Loaded window")
    print("  requested start:", requested_start.isoformat())
    print("  actual start:   ", segment.start_time.isoformat())
    print("  data shape:     ", segment.data.shape)
    print("  fs:             ", segment.fs_hz)
    print("  dx:             ", segment.dx_m)
    print("  source files:   ", segment.metadata["source_files"])

    actual_duration = (
        segment.time_s[-1] + 1.0 / segment.fs_hz
    )

    print("  actual duration:", actual_duration)

    assert segment.data.shape[0] == len(segment.distance_m)
    assert segment.data.shape[1] == len(segment.time_s)
    assert actual_duration > 0

    print("Window loading test passed.")


if __name__ == "__main__":
    main()