#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# ------------------------- Helpers -------------------------

TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})]")

FIELDS = [
    "JobID",
    "JobName",
    "Partition",
    "Submit",
    "Start",
    "End",
    "Elapsed",
    "CPUTimeRAW",
    "NCPUS",
    "State",
]


def run_start_from_log(logfile: Path) -> datetime:
    """
    Return the datetime of the FIRST timestamped line in chemina.log
    (e.g., '[2025-08-12-09:29:58] ...').
    """
    with logfile.open("r", errors="replace") as f:
        for line in f:
            m = TIMESTAMP_RE.match(line)
            if m:
                return datetime.strptime(m.group(1), "%Y-%m-%d-%H:%M:%S")
    raise RuntimeError(f"No timestamp found in {logfile}")


def sacct_query(
    job_ids: List[str],
    fields: List[str],
    batch_size: int = 200,
    extra_args: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Query sacct for given job_ids in batches.
    Returns rows (pipe-delimited split) for TOP-LEVEL jobs only (no step IDs like 12345.0).
    """
    if not job_ids:
        return []
    rows: List[List[str]] = []
    fmt = ",".join(fields)
    extra_args = extra_args or []
    for i in range(0, len(job_ids), batch_size):
        batch = ",".join(job_ids[i : i + batch_size])
        cmd = ["sacct", "-j", batch, "-n", "-P", "--format", fmt] + extra_args
        out = subprocess.check_output(cmd, text=True, errors="replace").splitlines()
        for line in out:
            if not line.strip():
                continue
            cols = line.split("|")
            # skip steps (keep only top-level job lines)
            if "." in cols[0]:
                continue
            rows.append(cols)
    return rows


def parse_dt(s: str | None):
    """Return datetime or None for empty/unknown sacct timestamps."""
    if not s:
        return None
    s = s.strip()
    if s in {"None", "Unknown", "N/A", ""}:
        return None
    # sacct may emit 'YYYY-MM-DDTHH:MM:SS', 'YYYY-MM-DD HH:MM:SS', or with TZ suffixes
    for candidate in (s, s.replace(" ", "T")):
        try:
            # strip any timezone/offset like +01:00
            return datetime.fromisoformat(candidate.split("+")[0])
        except ValueError:
            continue
    return None


def collect_pairs(rows):
    """
    rows: iterable of (JobID, JobName, Partition, Submit, Start, End, ElapsedRaw, CPUTimeRAW, NCPUS, State)
    returns: dict[jid] = (start_dt, end_dt)
    """
    pairs = {}
    skipped = 0
    for jid, name, part, sub, st, en, elapsed_raw, cpu_raw, ncpus, state in rows:
        sdt = parse_dt(st)
        edt = parse_dt(en)
        if not sdt or not edt:
            skipped += 1
            continue
        # guard against clock oddities
        if edt < sdt:
            sdt, edt = edt, sdt
        pairs[jid] = (sdt, edt)
    if skipped:
        print(f"# Skipped {skipped} rows without usable Start/End")
    return pairs


def find_sampling_ids(root: Path) -> List[str]:
    """
    Harvest sampling JobIDs from log_dir/*/*_submission.sh (or log_dir/*_submission.sh).
    """
    ids = []
    ld = root / "log_dir"
    if ld.is_dir():
        for sub in ld.rglob("*_submission.sh"):
            try:
                ids.append(sub.stem.replace("_submission", ""))
            except Exception:
                pass
    # Dedup + numeric sort if possible
    try:
        ids = sorted(set(ids), key=lambda x: int(x))
    except ValueError:
        ids = sorted(set(ids))
    return ids


def find_cluster_ids(root: Path) -> List[str]:
    """
    Harvest clustering JobIDs from log_cluster/*_submission.sh.
    """
    ids = []
    lc = root / "log_cluster"
    if lc.is_dir():
        for sub in lc.glob("*_submission.sh"):
            try:
                ids.append(sub.stem.replace("_submission", ""))
            except Exception:
                pass
    try:
        ids = sorted(set(ids), key=lambda x: int(x))
    except ValueError:
        ids = sorted(set(ids))
    return ids


def write_xmgrace(
    segments: List[Tuple[int, float, float]], out_path: Path, precision: int = 5
) -> int:
    """
    Write an xmgrace-friendly file:
        start_x  row
        end_x    row

        (blank line between segments)
    segments: list of (row_index, start_h, end_h).
    Returns number of segments written.
    """
    n = 0
    with out_path.open("w") as f:
        for row_idx, xs, xe in segments:
            f.write(f"{xs:.{precision}f} {row_idx}\n")
            f.write(f"{xe:.{precision}f} {row_idx}\n\n")
            n += 1
    return n


# ------------------------- Main -------------------------
def collect_pairs(rows: List[List[str]]) -> Dict[str, Tuple[datetime, datetime, int]]:
    """
    Map JobID -> (Start, End, CPUTimeRAW).
    """
    pairs: Dict[str, Tuple[datetime, datetime, int]] = {}
    for r in rows:
        jobid = r[0]
        st, en = r[4], r[5]
        cputime = int(r[7]) if r[7].isdigit() else 0
        sdt, edt = parse_dt(st), parse_dt(en)
        if sdt:
            pairs[jobid] = (sdt, edt or sdt, cputime)
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Make xmgrace timelines with SHARED row indices for sampling & clustering."
    )
    ap.add_argument(
        "logfile", type=Path, help="Path to main chemina.log (used to set t0)."
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Run directory containing log_dir/ and log_cluster/. "
        "Defaults to parent of the logfile's directory.",
    )
    ap.add_argument(
        "--run-start",
        default=None,
        help='Override t0: "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS".',
    )
    ap.add_argument(
        "--partition", default=None, help="Optional sacct --partition filter."
    )
    ap.add_argument("--start-date", default=None, help="Optional sacct -S YYYY-MM-DD.")
    ap.add_argument(
        "--end-date", default="now", help='Optional sacct -E (default "now").'
    )
    ap.add_argument("--sampling-out", default="sampling_timeline.dat")
    ap.add_argument("--clustering-out", default="clustering_timeline.dat")
    ap.add_argument("--map-out", default="timeline_row_map.tsv")
    args = ap.parse_args()

    logfile = args.logfile.resolve()
    if not logfile.exists():
        raise SystemExit(f"Log file not found: {logfile}")

    # Derive run_dir if not provided (assume .../<run>/.logs/chemina.log)
    root = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else logfile.parent.parent.resolve()
    )

    # ---- t0 (zero hour)
    if args.run_start:
        fmt = "%Y-%m-%d %H:%M:%S" if " " in args.run_start else "%Y-%m-%dT%H:%M:%S"
        t0 = datetime.strptime(args.run_start, fmt)
    else:
        t0 = run_start_from_log(logfile)

    # ---- harvest IDs
    samp_ids = find_sampling_ids(root)
    clus_ids = find_cluster_ids(root)
    if not samp_ids and not clus_ids:
        raise SystemExit(f"No job IDs found in {root}/log_dir or {root}/log_cluster.")

    # ---- sacct filters
    extra: List[str] = []
    if args.partition:
        extra += ["--partition", args.partition]
    if args.start_date:
        extra += ["-S", args.start_date]
    if args.end_date:
        extra += ["-E", args.end_date]

    # ---- query sacct
    samp_rows = sacct_query(samp_ids, FIELDS, extra_args=extra) if samp_ids else []
    clus_rows = sacct_query(clus_ids, FIELDS, extra_args=extra) if clus_ids else []

    # collect_pairs returns {JobID: (Start, End, CPUTimeRAW)}
    samp_pairs = collect_pairs(samp_rows)
    clus_pairs = collect_pairs(clus_rows)

    if not samp_pairs and not clus_pairs:
        raise SystemExit(
            "No jobs with valid Start/End found via sacct for the collected IDs."
        )

    # ---- filter out pre-run items
    GRACE = timedelta(minutes=2)
    samp_pairs = {
        jid: (s, e, c)
        for jid, (s, e, c) in samp_pairs.items()
        if s >= t0 - GRACE and (e is None or e >= s)
    }
    clus_pairs = {
        jid: (s, e, c)
        for jid, (s, e, c) in clus_pairs.items()
        if s >= t0 - GRACE and (e is None or e >= s)
    }
    if not samp_pairs and not clus_pairs:
        raise SystemExit(
            "All jobs filtered as pre-run; adjust --run-start or pass -S/-E for sacct."
        )

    # ---- shared row indices across BOTH types
    all_jobs = [("sampling", jid, s, e) for jid, (s, e, _c) in samp_pairs.items()] + [
        ("clustering", jid, s, e) for jid, (s, e, _c) in clus_pairs.items()
    ]
    all_jobs.sort(key=lambda x: x[2])  # by Start

    row_index: Dict[Tuple[str, str], int] = {
        (typ, jid): i + 1 for i, (typ, jid, _s, _e) in enumerate(all_jobs)
    }

    to_hours = lambda dt: (dt - t0).total_seconds() / 3600.0

    # ---- build segments & sum CPU
    sampling_segments: List[Tuple[int, float, float]] = []
    cpu_samp_total = 0
    for jid, (s, e, cpu) in samp_pairs.items():
        sampling_segments.append(
            (row_index[("sampling", jid)], to_hours(s), to_hours(e))
        )
        cpu_samp_total += int(cpu)

    clustering_segments: List[Tuple[int, float, float]] = []
    cpu_clus_total = 0
    for jid, (s, e, cpu) in clus_pairs.items():
        clustering_segments.append(
            (row_index[("clustering", jid)], to_hours(s), to_hours(e))
        )
        cpu_clus_total += int(cpu)

    sampling_segments.sort(key=lambda t: t[0])
    clustering_segments.sort(key=lambda t: t[0])

    # ---- write outputs
    n_s = write_xmgrace(sampling_segments, root / args.sampling_out, precision=5)
    n_c = write_xmgrace(clustering_segments, root / args.clustering_out, precision=5)

    with (root / args.map_out).open("w", newline="") as mp:
        w = csv.writer(mp, delimiter="\t")
        w.writerow(["row_index", "type", "JobID", "Start", "End"])
        for typ, jid, s, e in all_jobs:
            w.writerow(
                [
                    row_index[(typ, jid)],
                    typ,
                    jid,
                    s.isoformat(sep=" "),
                    e.isoformat(sep=" "),
                ]
            )

    # ---- summary
    print(f"t0 (zero hour): {t0.isoformat(sep=' ')}")
    print(
        f"Sampling jobs:  {n_s}  -> {args.sampling_out}   (CPU {cpu_samp_total/3600:.2f} h)"
    )
    print(
        f"Clustering jobs:{n_c}  -> {args.clustering_out}   (CPU {cpu_clus_total/3600:.2f} h)"
    )
    print(f"Row map:               -> {args.map_out}")


if __name__ == "__main__":
    main()
