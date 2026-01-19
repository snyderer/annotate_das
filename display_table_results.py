import pandas as pd
import sqlite3

db_path = r"C:\Users\ers334\Documents\databases\DAS_Annotations\A25.db"
conn = sqlite3.connect(db_path)

query = """
SELECT * FROM fx_labels
WHERE label_name LIKE '%Bp%' AND t=14.0;
"""

df = pd.read_sql_query(query, conn)

df2 = pd.read_sql_query("""
    SELECT distinct(apex_time) FROM tx_labels
    """, conn)

# Create backup:
if False:
    cur = conn.cursor()
    query = """
        CREATE TABLE tx_backup_260119 AS
        Select * FROM tx_labels
    """
    cur.execute(query)

    query = """
        CREATE TABLE fx_backup_260119 AS
        Select * FROM fx_labels
    """
    cur.execute(query)

if False:
    cur = conn.cursor()
    query = """
        DELETE FROM tx_labels
        WHERE dataset = 'ooi_optasense_south_c1_full'
        """
    cur.execute(query)

    query = """
        DELETE FROM fx_labels
        WHERE dataset = 'ooi_optasense_south_c1_full'
        """
    cur.execute(query)
conn.close()
