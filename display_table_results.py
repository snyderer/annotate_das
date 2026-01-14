import pandas as pd
import sqlite3
db_path = r"C:\Users\ers334\Documents\gitRepos\annotate\A25.db"
# db_path = r"C:\Users\ers334\Documents\databases\DAS_Annotations\A25.db"
conn = sqlite3.connect(db_path)

query = """
SELECT count(*)
FROM tx_labels;
"""

df = pd.read_sql_query(query, conn)

# conn.execute("""
#              drop table if exists tx_labels;
#              """)

# conn.execute("""
#              drop table if exists fx_labels;
#              """)

# Create backup:
if False:
    cur = conn.cursor()
    query = """
        CREATE TABLE tx_backup AS
        Select * FROM tx_labels
    """
    cur.execute(query)

    query = """
        CREATE TABLE fx_backup AS
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
