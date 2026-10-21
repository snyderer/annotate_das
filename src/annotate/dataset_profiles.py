from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetProfile:
    key: str
    display_name: str
    source_file: Path

    # None means: retain config.py / UserSettings default.
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

    # Optional display overrides.
    tx_vmin: float | None = None
    tx_vmax: float | None = None
    fx_vmin: float | None = None
    fx_vmax: float | None = None
    spec_vmin: float | None = None
    spec_vmax: float | None = None

    extra_settings: dict = field(default_factory=dict)
   
    def to_settings_dict(self) -> dict:
        """
        Return only explicitly configured profile overrides.

        Omitted settings are intentionally not returned, allowing GUI widgets
        and config.py UserSettings defaults to remain unchanged.
        """
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

        # Only preserve explicit profile values.
        settings = {
            key: value
            for key, value in values.items()
            if value is not None
        }

        # Shared default save location comes from config.py unless the profile
        # explicitly overrides it.
        if self.annotation_csv is not None:
            settings["labels_file_path"] = str(self.annotation_csv)

        settings.update(self.extra_settings)

        return settings   
    
DATASET_PROFILES: dict[str, DatasetProfile] = {
    "svalbard": DatasetProfile(
        key="svalbard",
        display_name="Svalbard",
        source_file=Path(
            r"C:\Users\ers334\Desktop\testingData\Svalbard\data\120147.hdf5" # FIXME: Update to real Svalbard file path.
        ),
        annotation_csv=Path(
            r"C:\Users\ers334\Desktop\projects\labeling_2026"
            r"\annotations_svalbard.csv"
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

    "ooi_north_c2": DatasetProfile(
        key="ooi_north_c2",
        display_name="OOI North C2",
        source_file=Path(
            r"C:\Users\ers334\Desktop\testingData\OOI\DASData"
            r"\OptaSense\North_C2"
            r"\North-C2-HF-P1kHz-GL30m-Sp2m-FS500Hz_2021-11-02T215901Z.h5"
        ),
        annotation_csv=Path(
            r"C:\Users\ers334\Desktop\projects\labeling_2026"
            r"\annotations_ooi_north_c2.csv"
        ),
        interrogator="optasense",
        use_full_cable_length=True,
        cable_start_m=0,
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

    "ooi_north_c3": DatasetProfile(
        key="ooi_north_c3",
        display_name="OOI North C3",
        source_file=Path(
            r"C:\Users\ers334\Desktop\testingData\OOI\DASData"
            r"\OptaSense\blue_whales_North_C3"
            r"\North-C3-HF-P1kHz-GL30m-Sp2m_2021-11-02T140115Z.h5"
        ),
        annotation_csv=Path(
            r"C:\Users\ers334\Desktop\projects\labeling_2026"
            r"\annotations_ooi_north_c3.csv"
        ),
        interrogator="optasense",
        use_full_cable_length=True,
        cable_start_m=0,
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

    "south_c1": DatasetProfile(
        key="south_c1",
        display_name="OOI South C1",
        source_file=Path(
            r"C:\Users\ers334\Desktop\testingData\OOI\DASData"
            r"\OptaSense\South_C1"
            r"\South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-02T103714Z.h5"
        ),
        annotation_csv=Path(
            r"C:\Users\ers334\Desktop\projects\labeling_2026"
            r"\annotations_ooi_south_c1.csv"
        ),
        interrogator="optasense",
        use_full_cable_length=True,
        cable_start_m=20_000.0,
        cable_end_m=95_000.0,
        target_dx_m=16.0,
        duration_s=40.0,
        navigation_step_s=25.0,
        sound_speed_mps=1500.0,
    ),

    "ooi_silixa": DatasetProfile(
        key="ooi_silixa",
        display_name="OOI Silixa",
        source_file=Path(
            r"C:\path\to\OOI\Silixa\example_file.tdms"
        ),
        annotation_csv=Path(
            r"C:\Users\ers334\Desktop\projects\labeling_2026"
            r"\annotations_ooi_silixa.csv"
        ),
        interrogator="silixa",
        use_full_cable_length=True,
        cable_start_m=0.0,
        cable_end_m=95_000.0,
        target_dx_m=16.0,
        duration_s=40.0,
        navigation_step_s=25.0,
        sound_speed_mps=1500.0,
    ),

    # "st_lawrence": DatasetProfile(
    #     key="st_lawrence",
    #     display_name="St. Lawrence",
    #     source_file=Path(
    #         r"C:\path\to\StLawrence\example_file.h5"
    #     ),
    #     annotation_csv=Path(
    #         r"C:\Users\ers334\Desktop\projects\labeling_2026"
    #         r"\annotations_st_lawrence.csv"
    #     ),
    #     interrogator="auto",
    #     use_full_cable_length=False,
    #     cable_start_m=0.0,
    #     cable_end_m=20_000.0,
    #     target_dx_m=16.0,
    #     sound_speed_mps=1500.0,
    # ),
}