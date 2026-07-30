# annotate/data/dataset_service.py
from pathlib import Path

from annotate.data.dataset_reader import DAS4WhalesDatasetReader


class DatasetService:
    def __init__(self):
        self.reader = DAS4WhalesDatasetReader()

        self.dataset_directory: Path | None = None
        self.interrogator: str | None = None
        self.interrogator_source: str | None = None  # "auto" or "user"
        self.file_index = None

    def open_dataset(
        self,
        selected_file: str | Path,
        requested_interrogator: str = "auto",
    ):
        """
        Open a dataset directory if necessary.

        Detection and indexing happen only when:
        - the directory changes, or
        - the user changes interrogator choice.
        """
        selected_file = Path(selected_file)
        directory = selected_file.parent

        directory_changed = directory != self.dataset_directory

        interrogator_changed = (
            requested_interrogator != "auto"
            and requested_interrogator != self.interrogator
        )

        if directory_changed or interrogator_changed:
            self._open_new_directory(
                selected_file=selected_file,
                requested_interrogator=requested_interrogator,
            )

        return self.file_index.get_record(selected_file)

    def _open_new_directory(
        self,
        selected_file: Path,
        requested_interrogator: str,
    ):
        """Resolve interrogator once and construct the directory file index."""
        if requested_interrogator == "auto":
            resolved = self.reader.detect_interrogator(selected_file)

            if resolved is None:
                raise ValueError(
                    "Could not reliably detect the interrogator type for "
                    f"{selected_file.name}. Please choose it manually."
                )

            self.interrogator = resolved
            self.interrogator_source = "auto"

        else:
            self.interrogator = requested_interrogator
            self.interrogator_source = "user"

        self.dataset_directory = selected_file.parent

        self.file_index = self._build_file_index(
            directory=self.dataset_directory,
            interrogator=self.interrogator,
        )

    def _build_file_index(self, directory: Path, interrogator: str):
        """
        Scan each source file once, extract lightweight metadata, and create
        a sorted time-to-file index.

        This should not load full DAS arrays.
        """
        # Later:
        # return FileIndex.from_directory(
        #     directory=directory,
        #     metadata_reader=lambda path: self.reader.get_metadata(
        #         path,
        #         interrogator=interrogator,
        #     ),
        # )
        raise NotImplementedError