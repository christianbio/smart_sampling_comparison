#!/usr/bin/env python3
import os
from cspy.db import CspDataStore

DB1 = "clustered_cases.db"  # smart-sampling DB
DB2 = "clustered_cases_no_smart.db"  # baseline/no-smart DB
OUT = "clustered_cases_combined.db"


def ensure_origin_table(db: CspDataStore):
    db.query("""
    CREATE TABLE IF NOT EXISTS origin (
        id     TEXT PRIMARY KEY,
        source TEXT CHECK (source IN ('smart','nosmart','both'))
    )
    """)
    db.commit()


def main():
    # 0) Start fresh OUT
    if os.path.exists(OUT):
        os.remove(OUT)
    combined = CspDataStore(OUT)
    ensure_origin_table(combined)

    # 1) COPY UNIQUE STRUCTURES FROM DB1 (smart)
    db1 = CspDataStore(DB1)
    db1.copy_unique_structures_to(OUT)
    db1.close()

    # tag everything currently in combined as 'smart' (initial default)
    combined.query("""
    INSERT OR IGNORE INTO origin(id, source)
    SELECT id, 'smart' FROM crystal
    """)
    combined.commit()

    # snapshot current ids = ids from DB1 copy
    combined.query("CREATE TEMP TABLE db1_ids(id TEXT PRIMARY KEY)")
    combined.query("INSERT INTO db1_ids SELECT id FROM crystal")
    combined.commit()

    # 2) COPY UNIQUE STRUCTURES FROM DB2 (nosmart)
    db2 = CspDataStore(DB2)
    # grab db2 id list BEFORE copying (to identify overlaps correctly)
    db2_ids = [row[0] for row in db2.query("SELECT id FROM crystal").fetchall()]
    db2.close()

    # temp table of db2 ids
    combined.query("CREATE TEMP TABLE db2_ids(id TEXT PRIMARY KEY)")
    # insert db2 ids in batches
    BATCH = 500
    for i in range(0, len(db2_ids), BATCH):
        batch = db2_ids[i : i + BATCH]
        values = ",".join(f"('{cid}')" for cid in batch)
        combined.query(f"INSERT INTO db2_ids(id) VALUES {values}")
    combined.commit()

    # now actually copy db2 unique structures into OUT
    db2 = CspDataStore(DB2)
    db2.copy_unique_structures_to(OUT)
    db2.close()

    # 3) UPDATE ORIGIN LABELS ACCURATELY
    # 3a) any id that exists in both db1_ids and db2_ids -> 'both'
    combined.query("""
    UPDATE origin
    SET source = 'both'
    WHERE id IN (SELECT id FROM db1_ids)
      AND id IN (SELECT id FROM db2_ids)
    """)
    # 3b) any id that exists only in db2_ids (i.e., wasn't in db1 copy) -> 'nosmart'
    combined.query("""
    INSERT OR IGNORE INTO origin(id, source)
    SELECT id, 'nosmart'
    FROM db2_ids
    WHERE id NOT IN (SELECT id FROM db1_ids)
    """)
    combined.commit()

    # clean up temps
    combined.query("DROP TABLE IF EXISTS db1_ids")
    combined.query("DROP TABLE IF EXISTS db2_ids")
    combined.commit()

    # quick stats
    n_crys = combined.query("SELECT COUNT(*) FROM crystal").fetchone()[0]
    n_desc = combined.query("SELECT COUNT(*) FROM descriptor").fetchone()[0]
    n_trial = combined.query("SELECT COUNT(*) FROM trial_structure").fetchone()[0]
    by_src = combined.query(
        "SELECT source, COUNT(*) FROM origin GROUP BY source"
    ).fetchall()
    combined.close()

    print(f"Combined DB written to: {OUT}")
    print(f"  crystal         : {n_crys:,}")
    print(f"  descriptor      : {n_desc:,}")
    print(f"  trial_structure : {n_trial:,}")
    for s, c in by_src:
        print(f"  origin[{s:<7}]    : {c:,}")
    print("Now run your clustering on the combined DB to populate 'equivalent_to'.")


if __name__ == "__main__":
    main()
