from annotate import data_io as io
import sqlite3
import os
import numpy as np

#=== CONFIG PATHS ===#
db_path = r"C:\Users\ers334\Documents\databases\DAS_Annotations\A25.db"
root_path = "F:\\"

# Connect to DB
conn = sqlite3.connect(db_path)
cur = conn.cursor()

#=== 1. Make sure fx_labels has `source_file` column ===#
cur.execute("PRAGMA table_info(fx_labels)")
columns_fx = [row[1] for row in cur.fetchall()]  # row[1] = column name

if "source_file" not in columns_fx:
    print("Adding 'source_file' column to fx_labels...")
    cur.execute("ALTER TABLE fx_labels ADD COLUMN source_file TEXT")
    conn.commit()  # Commit schema change before data changes
else:
    print("'source_file' column already exists in fx_labels.")

#=== 2. Get all datasets from tx_labels ===#
datasets = [row[0] for row in cur.execute("SELECT DISTINCT dataset FROM tx_labels").fetchall()]

for dataset in datasets:
    # Load file_map for this dataset
    dataset_filepath = os.path.join(root_path, dataset)
    settings_filepath = os.path.join(dataset_filepath, 'settings.h5')

    settings = io.load_settings_preprocessed_h5(settings_filepath)
    file_map = settings.get('file_map', {})

    # Ensure file_map sorted by timestamp
    sort_idx = np.argsort(file_map['timestamp'])
    file_map['timestamp'] = np.array(file_map['timestamp'])[sort_idx]
    file_map['filename'] = np.array(file_map['filename'])[sort_idx]

    #=== 3. Update tx_labels using apex_time ===#
    tx_rows = cur.execute("""
        SELECT id, apex_time FROM tx_labels
        WHERE dataset = ?
    """, (dataset,)).fetchall()

    for label_id, apex_time in tx_rows:
        # Find last timestamp <= apex_time
        indices = np.where(file_map['timestamp'] <= apex_time)[0]
        if len(indices) == 0:
            print(f"[WARN] No file match for apex_time={apex_time} in dataset={dataset}")
            continue

        file_idx = indices[-1]
        mapped_file = file_map['filename'][file_idx]
        next_file = file_map['filename'][file_idx + 1] if file_idx + 1 < len(file_map['filename']) else None

        # Update tx_labels.source_file
        cur.execute("""
            UPDATE tx_labels
            SET source_file = ?
            WHERE id = ?
        """, (mapped_file, label_id))

        #=== 4. Update fx_labels.source_file ===#
        fx_rows = cur.execute("""
            SELECT t FROM fx_labels
            WHERE dataset = ? AND tx_id = ?
        """, (dataset, label_id)).fetchall()

        for (t,) in fx_rows:
            if t >= 30:
                mapped_file_fx = next_file if next_file is not None else mapped_file
            else:
                mapped_file_fx = mapped_file

            cur.execute("""
                UPDATE fx_labels
                SET source_file = ?
                WHERE dataset = ? AND tx_id = ? AND t = ?
            """, (mapped_file_fx, dataset, label_id, t))

#=== 5. Commit everything ===#
conn.commit()
conn.close()

print("\n✅ Database updated successfully. All source_file mappings corrected.")