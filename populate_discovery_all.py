#!/usr/bin/env python3
"""
Populate class_discovery tables for all cases listed in sampling_metadata.json.
Uses the *actual* epoch value from the .sat file (5th column), not the row index.
"""

import json
import os
import argparse
from bisect import bisect_right

from chemina.sample.sample import Smart_sampling
from cspy.db import CspDataStore


# ---- SAT parsing ----


def load_sat(sat_file):
    """
    Returns (cum_nvalid_list, epoch_list), aligned by row.
      - cum[i]   = cumulative valid count at row i
      - epochs[i]= actual global epoch value at row i (5th column in the SAT)
    """
    cum = []
    epochs = []
    with open(sat_file) as f:
        _ = f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            # columns: index, cum_nunique, cum_nlowe, cum_nvalid, epoch
            cum.append(int(parts[3]))  # 4th col = cum_nvalid
            epochs.append(int(parts[4]))  # 5th col = epoch
    return cum, epochs


def order_to_epoch(order, cum, epochs):
    """
    Map a 0-based discovery order (counting minimization_step=3 structures)
    to the corresponding *actual epoch*.

    We find the first SAT row whose cum_nvalid > order (bisect_right),
    then return that row's epoch value.
    """
    idx = bisect_right(cum, order)  # idx in 0..len(cum)
    if idx <= 0:
        return epochs[0]
    if idx > len(epochs):
        return epochs[-1]
    return epochs[idx - 1]


# ---- Per-case processing ----


def process_case(db_file, sat_file, sampling):
    db = CspDataStore(db_file)
    discovery_map = sampling.build_discovery_map(db)  # {unique_id: (first_id, order)}
    cum, epochs = load_sat(sat_file)

    db.query("""
        CREATE TABLE IF NOT EXISTS class_discovery (
            unique_id TEXT PRIMARY KEY,
            first_member_id TEXT NOT NULL,
            discovery_order INTEGER NOT NULL,
            epoch_first_discovery INTEGER NOT NULL
        )
    """)

    rows = []
    for uid, (fid, order) in discovery_map.items():
        epoch_val = order_to_epoch(order, cum, epochs)
        rows.append((uid, fid, order, epoch_val))

    cur = db.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO class_discovery
        (unique_id, first_member_id, discovery_order, epoch_first_discovery)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    cur.connection.commit()
    return len(rows)


# ---- Driver ----


def main(metadata_file):
    with open(metadata_file) as f:
        metadata = json.load(f)

    sampling = Smart_sampling([], {}, sample_dir=metadata["sample_dir"])
    db_path_dict = metadata["db_path_dict"]

    total_cases = len(db_path_dict)
    total_rows = 0

    for i, (case_id, db_file) in enumerate(db_path_dict.items(), start=1):
        sat_file = os.path.join(
            "Sample", "sats", os.path.basename(db_file).replace(".db", ".sat")
        )

        if not os.path.exists(db_file):
            print(f"[{i}/{total_cases}] ⚠️ Missing DB: {db_file}")
            continue
        if not os.path.exists(sat_file):
            print(f"[{i}/{total_cases}] ⚠️ Missing SAT: {sat_file}")
            continue

        print(f"[{i}/{total_cases}] Processing {case_id} …")
        rows = process_case(db_file, sat_file, sampling)
        total_rows += rows
        print(f"   ↳ Inserted {rows} rows into {db_file} (running total: {total_rows})")

    print(f"\n✅ Done. Inserted {total_rows} total rows across {total_cases} cases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to sampling_metadata.json produced by log_cases.py",
    )
    args = parser.parse_args()
    main(args.metadata)
