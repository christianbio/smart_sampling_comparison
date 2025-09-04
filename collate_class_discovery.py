#!/usr/bin/env python3
"""
Collate per-case `class_discovery` tables into one clustered database.

- Input: sampling_metadata.json (has db_path_dict)
- Output: clustered DB with table `all_class_discovery`:
    case_id TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    first_member_id TEXT NOT NULL,
    discovery_order INTEGER NOT NULL,
    epoch_first_discovery INTEGER NOT NULL,
    space_group INTEGER,             # parsed from path if available (sg-XX), optional
    PRIMARY KEY (case_id, unique_id)

Idempotent: uses INSERT OR REPLACE. Safe to stop/restart.
"""

import os
import re
import json
import argparse
import sqlite3
from cspy.db import CspDataStore

SG_RE = re.compile(r"[\\/](?:sg-|SG-)(\d+)[\\/]")  # extract sg from path if present


def parse_sg_from_path(path: str):
    m = SG_RE.search(path)
    return int(m.group(1)) if m else None


def ensure_schema(conn: sqlite3.Connection, table_name: str):
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            case_id TEXT NOT NULL,
            unique_id TEXT NOT NULL,
            first_member_id TEXT NOT NULL,
            discovery_order INTEGER NOT NULL,
            epoch_first_discovery INTEGER NOT NULL,
            space_group INTEGER,
            PRIMARY KEY (case_id, unique_id)
        )
    """)
    # helpful indexes for common slicing
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_epoch ON {table_name}(epoch_first_discovery)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_sg ON {table_name}(space_group)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_uid ON {table_name}(unique_id)"
    )
    conn.commit()


def truncate_table(conn: sqlite3.Connection, table_name: str):
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table_name}")
    conn.commit()


def collect_one_case(case_id: str, db_path: str, epoch_offset: int):
    """Read rows from a single case DB's class_discovery table."""
    if not os.path.exists(db_path):
        return None, []

    db = CspDataStore(db_path)
    # Check table exists
    tables = {
        t[0]
        for t in db.query(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "class_discovery" not in tables:
        # nothing to copy
        return None, []

    sg = parse_sg_from_path(db_path)

    rows = db.query("""
        SELECT unique_id, first_member_id, discovery_order, epoch_first_discovery
        FROM class_discovery
    """).fetchall()

    # Map into output-row shape
    out_rows = [
        (case_id, uid, fid, order, (epoch + epoch_offset), sg)
        for (uid, fid, order, epoch) in rows
    ]
    return sg, out_rows


def main(
    metadata_path: str, out_db: str, table_name: str, epoch_offset: int, truncate: bool
):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    db_path_dict = metadata["db_path_dict"]  # {case_id: path/to/case.db}

    # Open output DB
    os.makedirs(os.path.dirname(out_db) or ".", exist_ok=True)
    out_conn = sqlite3.connect(out_db)
    ensure_schema(out_conn, table_name)
    if truncate:
        truncate_table(out_conn, table_name)

    total_cases = len(db_path_dict)
    total_rows = 0

    cur_out = out_conn.cursor()

    for i, (case_id, db_path) in enumerate(db_path_dict.items(), start=1):
        if not os.path.exists(db_path):
            print(f"[{i}/{total_cases}] ⚠️ Missing DB: {db_path}")
            continue

        sg, rows = collect_one_case(case_id, db_path, epoch_offset)
        if not rows:
            print(f"[{i}/{total_cases}] (no class_discovery) {case_id}")
            continue

        cur_out.executemany(
            f"""INSERT OR REPLACE INTO {table_name}
                (case_id, unique_id, first_member_id, discovery_order, epoch_first_discovery, space_group)
                VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        out_conn.commit()
        total_rows += len(rows)
        sg_str = f" sg={sg}" if sg is not None else ""
        print(
            f"[{i}/{total_cases}] {case_id}{sg_str}  → {len(rows)} rows (running total: {total_rows})"
        )

    out_conn.close()
    print(
        f"\n✅ Collation complete. Wrote {total_rows} rows to {out_db} in table {table_name}."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Path to sampling_metadata.json")
    ap.add_argument(
        "--out",
        required=True,
        help="Path to clustered output DB (will be created if missing)",
    )
    ap.add_argument("--table", default="all_class_discovery", help="Output table name")
    ap.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help="Add this to epoch_first_discovery on write (use 1 to convert 0-based → 1-based)",
    )
    ap.add_argument(
        "--truncate",
        action="store_true",
        help="If set, clears the output table before inserting",
    )
    args = ap.parse_args()
    main(args.metadata, args.out, args.table, args.epoch_offset, args.truncate)
