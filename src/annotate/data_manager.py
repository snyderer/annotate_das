import pandas as pd
import numpy as np
import os, sqlite3, json, uuid, getpass, datetime
from PyQt6.QtCore import QObject, pyqtSignal
import scipy.signal as sp
from . import data_io as io

class PreprocessedDataManager(QObject):
    dataset_loaded = pyqtSignal()      # tell panels to redraw with whatever data is loaded
    settings_changed = pyqtSignal()    # for appearance-only changes in TX/FX plots
    file_loaded = pyqtSignal(str, str)  # filename, timestamp_string

    def __init__(self):
        super().__init__()
        self.h5settings = {
            'fs': None, 'dx': None, 'ns': None, 'nx': None,
            'nonzeros_mask': None, 'file_map': {}
        }
        self.filepath = ''
        self.directory = ''
        self.label_saver = None 

        self.loaded_files_indices = []
        self.cursor_mode = ''  # '', 's' (spectrogram), 'a' (annotation)

        # Loaded continuous data
        self.loaded_data = {'amp': None, 't': None, 'x': None, 'time_stamps': None}
        self.display_idx = None

        # Store last applied user settings (sliders etc.)
        self._user_settings = {}

        # FX / Spectrogram managers
        self.fx_manager = FXHandle(self)
        self.spectrogram_manager = SpectrogramHandle(self)

    def apply_user_settings(self, user_settings: dict):
        """Store UI settings (vmin/vmax, nfft, overlap, label mapping, etc.)"""
        self._user_settings = user_settings
        self.settings_changed.emit()

    def get_user_settings(self, name=None):
        if name is None:
            return self._user_settings
        return self._user_settings.get(name)

    def new_file_selected(self, filepath):
        """Load the selected file + following file into a 60s window."""
        filepath = os.path.normpath(filepath)
        self.filepath = filepath
        selected_directory = os.path.dirname(filepath)
        selected_filename = os.path.basename(filepath)

        if selected_directory != self.directory:
            # directory has changed, reload settings
            self.directory = selected_directory
            # Load dataset settings (h5)
            settings_filepath = io.find_settings_h5(filepath)
            if settings_filepath is None:
                raise ValueError("No settings.h5 file found.")
            self.set_h5settings(settings_filepath)

        # Ensure filenames is a Python list
        filenames = list(self.h5settings['file_map']['filename'])
        try:
            idx = filenames.index(selected_filename)
        except ValueError:
            raise RuntimeError(f"File {filepath} not in file_map")

        # Choose previous/current/next file indices
        indices = [i for i in [idx, idx + 1]
                if 0 <= i < len(filenames)]
        self.loaded_files_indices = indices

        # Initial load: recompute FX as well
        self.load_current_window(recompute_fx=True)

    def navigate(self, direction):
        """Move the 60s window forward/backward by 30s – TX only update."""
        filenames = self.h5settings['file_map']['filename']
        # TODO set self.filepath as the new first file in window
        
        if direction == 'forward':
            last_idx = self.loaded_files_indices[-1]
            next_idx = last_idx + 1
            if next_idx >= len(filenames):
                print("Already at end of dataset.")
                return
            self.loaded_files_indices.pop(0)
            self.loaded_files_indices.append(next_idx)

        elif direction == 'backward':
            first_idx = self.loaded_files_indices[0]
            prev_idx = first_idx - 1
            if prev_idx < 0:
                print("Already at beginning of dataset.")
                return
            self.loaded_files_indices.pop()
            self.loaded_files_indices.insert(0, prev_idx)

        # update plots (including fx)
        self.load_current_window(recompute_fx=True)

    def load_current_window(self, recompute_fx=True):
        """Load and concatenate the files in `loaded_files_indices`."""
        amp_list, ts_list = [], []
        x = None
        for idx in self.loaded_files_indices:
            filepath = os.path.join(self.directory, self.h5settings['file_map']['filename'][idx])
            amp, t, x, ts = self.load_and_rehydrate_h5(filepath)
            amp_list.append(amp)
            ts_list.append(np.atleast_1d(ts))  # ensure array shape

        amp = np.concatenate(amp_list, axis=1)
        time_stamps = np.concatenate(ts_list)

        fs = self.h5settings['fs']
        total_samples = amp.shape[1]
        tvec = np.arange(total_samples) / fs

        self.loaded_data['amp'] = amp
        self.loaded_data['x'] = x
        self.loaded_data['time_stamps'] = time_stamps
        self.loaded_data['t'] = tvec

        # Always display the entire window (retained for future use)
        self.display_idx = np.ones(total_samples, dtype=bool)

        if recompute_fx:
            self.fx_manager.update_data()
            self.spectrogram_manager.update_data()
        
        self._emit_file_info() # update filename/timestamp display
        self.dataset_loaded.emit()

    def _emit_file_info(self):
        """Emit file_loaded signal with current file info."""
        if self.loaded_data is not None and 'time_stamps' in self.loaded_data:
            # Get the current loaded filenames
            filenames_str = self.get_loaded_filenames_string()
            timestamp_str = self.get_start_timestamp_string()
            self.file_loaded.emit(filenames_str, timestamp_str)

    def get_loaded_filenames_string(self):
        """Get a formatted string of currently loaded filenames."""
        filenames = self.h5settings['file_map']['filename'][self.loaded_files_indices]
        return ", ".join(filenames)

    def get_start_timestamp_string(self):
        """Get formatted timestamp string for the current data window."""
        try:
            if (self.loaded_data and 
                'time_stamps' in self.loaded_data and 
                self.loaded_data['time_stamps'] is not None and
                len(self.loaded_data['time_stamps']) > 0):
                
                start_timestamp = self.loaded_data['time_stamps'][0]
                dt = datetime.datetime.fromtimestamp(start_timestamp, tz=datetime.timezone.utc)
                return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
        except Exception as e:
            return f"Error reading timestamp: {str(e)}"
        
        return "No timestamp available"
        
    def set_h5settings(self, settings_filepath):
        settings = io.load_settings_preprocessed_h5(settings_filepath)
        self.h5settings['fs'] = settings['processing_settings']['fs']
        self.h5settings['dx'] = settings['processing_settings']['dx']
        self.h5settings['nx'], self.h5settings['ns'] = settings['rehydration_info']['target_shape']
        self.h5settings['nonzeros_mask'] = settings['rehydration_info']['nonzeros_mask']
        self.h5settings['file_map'] = settings['file_map']

    def set_cursor_mode(self, mode):
        self.cursor_mode = mode

    def load_and_rehydrate_h5(self, filepath, filter_lowpass=True):
        fk_dehyd, timestamp = io.load_preprocessed_h5(filepath)
        amp = 1e9 * io.rehydrate(
            fk_dehyd,
            self.h5settings['nonzeros_mask'],
            (self.h5settings['nx'], self.h5settings['ns'])
        )
        if filter_lowpass:
            amp = self.lowpass_filt(amp, cutoff_hz=70)
        t = np.arange(0, self.h5settings['ns'], 1) / self.h5settings['fs']
        x = np.arange(0, self.h5settings['nx'], 1) * self.h5settings['dx']
        return amp, t, x, timestamp
    
    def get_labels_in_current_window(self):
        """Return TX labels in the current display window as a list of dicts."""
        if not self.label_saver:
            return []

        try:
            display_time_start = float(self.loaded_data['time_stamps'][0])
            display_time_end = float(self.loaded_data['time_stamps'][-1])+self.h5settings['ns']/self.h5settings['fs']
            dataset = os.path.basename(self.directory)

            df = self.label_saver.df
            mask = ((df['apex_time_global'] >= display_time_start) &
                    (df['apex_time_global'] <= display_time_end) &
                    (df['dataset'] == dataset))
            rows = df[mask]

            results = []


            for _, row in rows.iterrows():
                results.append({
                    "tx_id": int(row["tx_id"]),
                    "uid": row["uid"],
                    "apex_time_global": row["apex_time_global"],
                    "apex_time_local": row["apex_time_local"],
                    "apex_dist": row["apex_dist"],
                    "duration": row["duration"],
                    "distance_to_cable": row["distance_to_cable"],
                    "dist_max": row["dist_max"],
                    "dist_min": row["dist_min"],
                    "f_max": row["f_max"],
                    "f_min": row["f_min"],
                })
            return results

        except Exception as e:
            print(f"Error querying labels: {e}")
            return []

    def lowpass_filt(self, data, cutoff_hz=70):
        """Lowpass filter the data along time axis."""
        fs = self.h5settings['fs']
        nyq = 0.5 * fs
        b, a = sp.butter(10, cutoff_hz / nyq, btype='low', analog=False)
        filtered_data = sp.filtfilt(b, a, data, axis=1)
        return filtered_data
    
class FXHandle:
    def __init__(self, data_manager: PreprocessedDataManager):
        self.data_manager = data_manager
        self.fx_series_data = None
        self.freq = None
        self.x = None
        self.plot_start_time = None

    def update_data(self):
        fs = self.data_manager.h5settings['fs']
        win_s = self.data_manager.get_user_settings('win_s') or 2.0
        amp = self.data_manager.loaded_data['amp']
        x = self.data_manager.loaded_data['x']
        t = self.data_manager.loaded_data['t']
        win_samples = int(win_s * fs)
        step_samples = win_samples
        slices = []
        t_win = []
        freqs = np.fft.rfftfreq(win_samples, d=1/fs)
        for start in range(0, amp.shape[1] - win_samples + 1, step_samples):
            segment = amp[:, start:start+win_samples]
            t_win.append(t[start])
            F = np.fft.rfft(segment, axis=1)
            slices.append(np.abs(F))
        self.fx_series_data = np.stack(slices, axis=0)
        self.freq = freqs
        self.plot_start_time = t_win
        self.x = x

    def get_dataset(self):
        return {"amp": self.fx_series_data,
                "freq": self.freq,
                "x": self.x,
                "t": self.plot_start_time}


class SpectrogramHandle:
    def __init__(self, data_manager: PreprocessedDataManager):
        self.data_manager = data_manager

    def update_data(self):
        pass  # no precomputation needed

    def calc_spectrogram(self, row_idx):
        nfft = self.data_manager.get_user_settings('nfft') or 256
        percent_overlap = self.data_manager.get_user_settings('overlap') or 50
        fs = self.data_manager.h5settings['fs']
        amp = self.data_manager.loaded_data['amp']
        sig = amp[row_idx, :]
        Noverlap = int(nfft * percent_overlap / 100)
        window = sp.windows.tukey(nfft, .25)
        window_rms = np.sqrt(np.sum(window**2))
        freqs, times, Sxx = sp.spectrogram(sig,
                                           fs=fs,
                                           window=window,
                                           nperseg=nfft,
                                           noverlap=Noverlap,
                                           scaling='spectrum',
                                           mode='magnitude')
        Sxx_corrected = Sxx * nfft / window_rms
        return freqs, times, Sxx_corrected
    

class LabelSaver:
    COLUMNS = [
        "tx_id", "uid",
        "apex_time_global", "apex_time_str", "apex_time_local",
        "apex_dist", "duration", "distance_to_cable",
        "dist_max", "dist_min", "f_max", "f_min",
        "label", "label_name", "dataset", "source_file",
        "saved_timestamp", "username"
    ]

    def __init__(self, csv_path):
        self.csv_path = csv_path
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = pd.DataFrame(columns=self.COLUMNS)
            self._save()
        # self.conn = sqlite3.connect(csv_path)
        # self.conn.execute("PRAGMA foreign_keys = ON;")  # enforce FK checks
        # self.conn.execute("PRAGMA journal_mode = WAL;")
        # self._create_tables()

    def _save(self):
        self.df.to_csv(self.csv_path, index=False)

    def _next_tx_id(self):
        if self.df.empty:
            return 1
        else:
            return self.df['tx_id'].max() + 1
        

    def save_tx_label(self, uid, apex_time_global, apex_time_str, apex_time_local,
                       apex_dist, x_m, t_s, dataset, source_file, label, label_name,
                       distance_to_cable=None, saved_timestamp=None, username=None):
        """
        Insert a new whale-call row.
        x_m / t_s (TX contour points) are used ONLY to derive duration/dist_max/dist_min
        -- they are not stored raw.
        Returns tx_id (int) for linking FX min/max updates.
        """
        
        if saved_timestamp is None:
            saved_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if username is None:
            username = getpass.getuser()
        if distance_to_cable is None:
            distance_to_cable = np.nan

        duration = float(np.max(t_s) - np.min(t_s)) if len(t_s) > 0 else np.nan
        dist_max = float(np.max(x_m)) if len(x_m) > 0 else np.nan
        dist_min = float(np.min(x_m)) if len(x_m) > 0 else np.nan

        tx_id = self._next_tx_id()
        new_row = {
            "tx_id": tx_id,
            "uid": uid,
            "apex_time_global": apex_time_global,
            "apex_time_str": apex_time_str,
            "apex_time_local": apex_time_local,
            "apex_dist": apex_dist,
            "duration": duration,
            "distance_to_cable": distance_to_cable,
            "dist_max": dist_max,
            "dist_min": dist_min,
            "f_max": np.nan,   # filled in via save_fx_label
            "f_min": np.nan,
            "dataset": os.path.basename(dataset),
            "source_file": os.path.abspath(source_file),
            "label": label,
            "label_name": label_name,
            "saved_timestamp": saved_timestamp,
            "username": username
        }

        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self._save()
        return tx_id

    def save_fx_label(self, tx_id, f_min_hz, f_max_hz, **kwargs):
        """
        Expand the row's f_min/f_max to cover this FX box.
        Extra kwargs (x_min_m, x_max_m, t, win_length_s, uid, dataset, label, label_name, ...)
        are accepted but ignored -- kept for call-site compatibility.
        """
        idx = self.df.index[self.df["tx_id"] == tx_id]
        if len(idx) == 0:
            print(f"Warning: no TX row found for tx_id={tx_id}")
            return
        idx = idx[0]

        current_min = self.df.at[idx, "f_min"]
        current_max = self.df.at[idx, "f_max"]

        new_min = f_min_hz if pd.isna(current_min) else min(current_min, f_min_hz)
        new_max = f_max_hz if pd.isna(current_max) else max(current_max, f_max_hz)

        self.df.at[idx, "f_min"] = new_min
        self.df.at[idx, "f_max"] = new_max
        self._save()

    def remove_label_by_id(self, tx_id):
        """Delete the row by tx_id."""
        print("Deleting TX label ID:", tx_id)
        self.df = self.df[self.df["tx_id"] != tx_id].reset_index(drop=True)
        self._save()

    def close(self):
        """No-op, kept for interface compatibility with old SQLite version."""
        pass