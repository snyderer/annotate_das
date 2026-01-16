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

            sql = """
                SELECT id, uid, apex_time, apex_distance, x_m, t_s 
                FROM tx_labels
                WHERE apex_time >= ? AND apex_time <= ?
                AND dataset = ?
            """
            cur = self.label_saver.conn.execute(sql, (display_time_start, display_time_end, dataset))
            rows = cur.fetchall()
            results = []
            for (tx_id, uid, apex_time, apex_distance, x_m_json, t_s_json) in rows:
                results.append({
                    "tx_id": tx_id,
                    "uid": uid,
                    "apex_time": apex_time,
                    "apex_distance": apex_distance,
                    "x_m": json.loads(x_m_json),
                    "t_s": json.loads(t_s_json)
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
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")  # enforce FK checks
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self._create_tables()

    def _create_tables(self):
        # TX table: PK = id, plus human-readable uid
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS tx_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid TEXT NOT NULL,             -- human-readable annotation ID
            apex_time REAL NOT NULL,
            apex_time_str TEXT NOT NULL,
            apex_distance REAL NOT NULL,
            x_m TEXT NOT NULL,
            t_s TEXT NOT NULL,
            dataset TEXT NOT NULL,
            source_file TEXT NOT NULL,
            label INTEGER NOT NULL,
            label_name TEXT NOT NULL,
            saved_timestamp TEXT NOT NULL,
            username TEXT NOT NULL
        );
        """)

        # FX table: PK = id, FK = tx_id references tx_labels.id
        # Also store uid for external/human matching
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS fx_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_id INTEGER NOT NULL,        -- FK to tx_labels PK
            uid TEXT NOT NULL,              -- same human-readable uid as TX
            f_min_hz REAL NOT NULL,
            f_max_hz REAL NOT NULL,
            x_min_m REAL NOT NULL,
            x_max_m REAL NOT NULL,
            t REAL NOT NULL,
            win_length_s REAL NOT NULL,
            dataset TEXT NOT NULL,
            label INTEGER NOT NULL,
            label_name TEXT NOT NULL,
            saved_timestamp TEXT NOT NULL,
            username TEXT NOT NULL,
            FOREIGN KEY (tx_id) REFERENCES tx_labels(id) ON DELETE CASCADE
        );
        """)
        self.conn.commit()

    def remove_label_by_id(self, tx_id):
        """Delete a TX label and its associated FX labels by TX primary key ID."""
        print('Deleting TX label ID:', tx_id)
        self.conn.execute("DELETE FROM tx_labels WHERE id = ?", (tx_id,))
        self.conn.commit()

    def save_tx_label(self, uid, apex_time, apex_time_str, apex_distance,
                      x_m, t_s, dataset, source_file, label, label_name,
                      saved_timestamp=None, username=None):
        """Insert a TX label and return its DB primary key ID (tx_id)."""
        
        if saved_timestamp is None:
            saved_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if username is None:
            username = getpass.getuser()

        cursor = self.conn.execute("""
            INSERT INTO tx_labels (
                uid, apex_time, apex_time_str, apex_distance,
                x_m, t_s, dataset, source_file, label, label_name,
                saved_timestamp, username
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            uid, apex_time, apex_time_str, apex_distance,
            json.dumps(x_m), json.dumps(t_s),
            os.path.basename(dataset),
            os.path.abspath(source_file),
            label, label_name,
            saved_timestamp, username
        ))
        self.conn.commit()

        return cursor.lastrowid  # Return DB PK to use in FX labels

    def save_fx_label(self, tx_id, uid, f_min_hz, f_max_hz, x_min_m, x_max_m,
                      t, win_length_s, dataset, label, label_name,
                      saved_timestamp=None, username=None):
        """Insert an FX label linked to its parent TX label via tx_id (DB PK) and also store uid."""
        print('fmin:', f_min_hz, 'fmax:', f_max_hz, 'xmin:', x_min_m, 'xmax:', x_max_m)
        if saved_timestamp is None:
            saved_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if username is None:
            username = getpass.getuser()

        self.conn.execute("""
            INSERT INTO fx_labels (
                tx_id, uid, f_min_hz, f_max_hz, x_min_m, x_max_m,
                t, win_length_s, dataset, label, label_name,
                saved_timestamp, username
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tx_id, uid, f_min_hz, f_max_hz, x_min_m, x_max_m,
            t, win_length_s,
            os.path.basename(dataset),
            label, label_name,
            saved_timestamp, username
        ))
        self.conn.commit()