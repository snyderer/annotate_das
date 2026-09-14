from pathlib import Path

from annotate.data.dataset_reader import DAS4WhalesDatasetReader


# Replace these with a real source file and interrogator.
FILEPATH = Path(r"C:\Users\ers334\Desktop\testingData\OOI\DASData\OptaSense\North_C2\North-C2-HF-P1kHz-GL30m-Sp2m-FS500Hz_2021-11-02T215901Z.h5")
INTERROGATOR = None  # e.g. "asn", "optasense", "fosina_dxs"
SELECTED_CHANNELS = (20000, 80000, 8)


def main():
    reader = DAS4WhalesDatasetReader()

    # First test an explicitly known interrogator.
    # This avoids debugging auto-detection and loading simultaneously.
    reader.set_interrogator(INTERROGATOR)

    print(f"File: {FILEPATH}")
    print(f"Interrogator: {INTERROGATOR}")
    print()

    # Test lightweight metadata read.
    info = reader.inspect_file(FILEPATH)

    print("Normalized metadata:")
    for key, value in info.items():
        if key != "das4whales_metadata":
            print(f"  {key}: {value}")

    print()

    # Test header-based file time extraction.
    header_start_time = reader.get_file_start_time(FILEPATH)

    print("Header start time:")
    print(f"  {header_start_time.isoformat()}")

    print()

    # Test complete file loading through DAS4Whales.
    segment = reader.load_file(
        filepath=FILEPATH,
        selected_channels=SELECTED_CHANNELS,
    )

    print("Loaded segment:")
    print(f"  data shape: {segment.data.shape}")
    print(f"  time shape: {segment.time_s.shape}")
    print(f"  distance shape: {segment.distance_m.shape}")
    print(f"  fs: {segment.fs_hz} Hz")
    print(f"  dx: {segment.dx_m} m")
    print(f"  start time: {segment.start_time.isoformat()}")
    print(f"  source: {segment.source_file}")
    print(
        f"  duration: "
        f"{segment.time_s[-1] + 1.0 / segment.fs_hz:.3f} s"
    )
    print(
        f"  distance range: "
        f"{segment.distance_m[0]:.2f} to {segment.distance_m[-1]:.2f} m"
    )

    # Basic consistency checks.
    assert segment.data.ndim == 2

    assert segment.data.shape[0] == len(segment.distance_m), (
        f"Channel mismatch: data has {segment.data.shape[0]} rows, "
        f"but distance has {len(segment.distance_m)} values."
    )

    assert segment.data.shape[1] == len(segment.time_s), (
        f"Time mismatch: data has {segment.data.shape[1]} columns, "
        f"but time has {len(segment.time_s)} values."
    )

    assert segment.fs_hz > 0
    assert segment.dx_m > 0
    assert segment.start_time.tzinfo is not None

    print()
    print("Reader test passed.")


if __name__ == "__main__":
    main()