#!/usr/bin/env python3
import argparse
import glob
import os
from datetime import datetime
from typing import List, Tuple, Optional

# Defaults that match your runs unless overridden
DEFAULT_MIN_PER_EPOCH = 20.0
DEFAULT_CORES_PER_JOB = 20
DEFAULT_CORES_AVAILABLE = 321


def parse_sat_rows(path: str) -> List[Tuple[int, int, int, int, int]]:
    """
    Parse one .sat file.

    Returns rows of: (index, cum_nunique, cum_nlowe, cum_nvalid, epoch)
    Skips blank lines and the header line.
    """
    rows: List[Tuple[int, int, int, int, int]] = []
    with open(path) as f:
        header_seen = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not header_seen:
                header_seen = True
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            idx, nunique, nlowe, nvalid, epoch = map(int, parts[:5])
            rows.append((idx, nunique, nlowe, nvalid, epoch))
    return rows


def minutes_to_str(mins: float) -> str:
    return f"{mins:,.2f} min ({mins/60:,.2f} h)"


def per_case_rate(rows: List[Tuple[int, int, int, int, int]]) -> Optional[float]:
    """
    Valids per *sampled epoch* (row): r_i = last_cum_valid / n_rows.
    We use number of sampled rows (not absolute epoch numbers).
    """
    if not rows:
        return None
    n_rows = len(rows)
    last_valid = rows[-1][3]
    if n_rows <= 0 or last_valid <= 0:
        return None
    return last_valid / n_rows


def epoch1_cum_valid(rows: List[Tuple[int, int, int, int, int]]) -> Optional[int]:
    for _, _, _, nvalid, ep in rows:
        if ep == 1:
            return nvalid
    return None


def sum_all_job_minutes(sats_dir: str, minutes_per_epoch: float) -> float:
    """
    Sum 'job minutes' across all SAT rows (each row = one sampled epoch for that case).
    """
    total_rows = 0
    for path in glob.glob(os.path.join(sats_dir, "*.sat")):
        with open(path) as f:
            # count nonblank lines, subtract header (1) if any nonblank exists
            nonblank = [ln for ln in f if ln.strip()]
            if nonblank:
                total_rows += max(0, len(nonblank) - 1)
    return total_rows * minutes_per_epoch


def wall_hours_from_logfile(logfile: str) -> float:
    """
    Read the first and last timestamped lines from chemina.log and return wall hours.
    Timestamps expected like: [YYYY-MM-DD-HH:MM:SS]...
    """
    fmt = "[%Y-%m-%d-%H:%M:%S]"
    start, end = None, None
    with open(logfile, "r", errors="replace") as f:
        for line in f:
            if not line.startswith("["):
                continue
            right = line.find("]")
            if right <= 0:
                continue
            ts = line[: right + 1]
            try:
                dt = datetime.strptime(ts, fmt)
            except ValueError:
                continue
            if start is None:
                start = dt
            end = dt
    if start is None or end is None:
        raise RuntimeError(f"No timestamps found in {logfile}")
    return (end - start).total_seconds() / 3600.0


def main():
    ap = argparse.ArgumentParser(
        description="Smart-sampling accounting vs equi-N baseline (from .sat files)."
    )
    ap.add_argument("sats_dir", nargs="?", default="./Sample/sats",
                    help="Directory containing *.sat files")

    # Common target N (if omitted, use max cum_nvalid across INCLUDED cases)
    ap.add_argument("--target-n", type=int, default=None,
                    help="Force equi-N target (cum_nvalid). Default: max over included cases.")

    # Legacy filter: require that each included case ever reached at least this many valids
    ap.add_argument("--min-required-valids", type=int, default=None,
                    help="Exclude cases that never reach this many cum_nvalid (ever).")

    # Epoch-1 gate for the baseline
    ap.add_argument("--epoch1-min-valids", type=int, default=None,
                    help="For equi-N baseline, exclude cases if cum_nvalid at epoch 1 is below this.")
    ap.add_argument("--align-epoch1", action="store_true",
                    help="If set, add epoch-1 cost (one epoch per excluded case) back into the equi-N baseline total.")

    # Resource model
    ap.add_argument("--minutes-per-epoch", type=float, default=DEFAULT_MIN_PER_EPOCH,
                    help="Job minutes per sampled epoch (default 20).")
    ap.add_argument("--cores-per-job", type=int, default=DEFAULT_CORES_PER_JOB,
                    help="Cores used by each job (default 20).")
    ap.add_argument("--cores-available", type=int, default=DEFAULT_CORES_AVAILABLE,
                    help="Concurrent cores available in the pool (default 321).")

    # Observed run wall time (for campaign efficiency)
    ap.add_argument("--total-wall-hours", type=float, default=None,
                    help="Observed total run wall time (hours) to compute cluster efficiency.")
    ap.add_argument("--logfile", type=str, default=None,
                    help="Path to chemina.log to auto-compute total wall hours "
                         "(first & last timestamp). If provided, overrides --total-wall-hours.")

    args = ap.parse_args()

    # Resolve observed wall hours (logfile takes precedence)
    observed_wall_hours: Optional[float] = None
    if args.logfile:
        observed_wall_hours = wall_hours_from_logfile(args.logfile)
        print(f"Observed total wall (from logfile): {observed_wall_hours:.2f} h")
    elif args.total_wall_hours is not None:
        observed_wall_hours = float(args.total_wall_hours)
        print(f"Observed total wall (manual):      {observed_wall_hours:.2f} h")

    files = sorted(glob.glob(os.path.join(args.sats_dir, "*.sat")))
    if not files:
        print(f"No .sat files found under: {args.sats_dir}")
        return

    # Load all files
    per_file = []
    total_scanned = 0
    excluded_by_cutoff = 0
    excluded_after_epoch1 = []  # for accounting back into baseline if align-epoch1
    for path in files:
        total_scanned += 1
        rows = parse_sat_rows(path)
        if not rows:
            continue

        # Legacy global capability cutoff (ever)
        if args.min_required_valids is not None:
            max_ever = max(r[3] for r in rows)
            if max_ever < args.min_required_valids:
                excluded_by_cutoff += 1
                continue

        per_file.append((path, rows))

    if not per_file:
        print("No eligible files after filters.")
        return

    # Determine target N (equi-N) from INCLUDED (for-now) set
    if args.target_n is not None:
        N_target = args.target_n
    else:
        N_target = max(max(r[3] for r in rows) for _, rows in per_file)

    # Smart-sampling totals (actual)
    actual_wall_min_all = sum_all_job_minutes(args.sats_dir, args.minutes_per_epoch)
    actual_cpu_min_all = actual_wall_min_all * args.cores_per_job
    actual_wall_pool_min_all = actual_cpu_min_all / args.cores_available

    # Baseline (equi-N) pool
    included_for_baseline = []
    zero_or_bad_rate = 0

    for path, rows in per_file:
        # Epoch-1 gate (baseline only)
        if args.epoch1_min_valids is not None:
            e1 = epoch1_cum_valid(rows)
            if e1 is None or e1 < args.epoch1_min_valids:
                # Exclude from baseline; optionally remember for epoch-1 cost add-back
                if args.align_epoch1:
                    excluded_after_epoch1.append((path, rows))
                continue

        r_i = per_case_rate(rows)
        if r_i is None or r_i <= 0.0:
            zero_or_bad_rate += 1
            continue
        included_for_baseline.append((path, rows, r_i))

    if not included_for_baseline:
        print("No cases left for baseline after filters.")
        return

    # Harmonic-mean rate (reporting only)
    recip_sum = sum(1.0 / r for _, _, r in included_for_baseline)
    r_harm = len(included_for_baseline) / recip_sum

    # Baseline cost: epochs_i = N / r_i
    epochs_total_est = sum((N_target / r) for _, _, r in included_for_baseline)
    baseline_wall_min = epochs_total_est * args.minutes_per_epoch
    baseline_cpu_min = baseline_wall_min * args.cores_per_job
    baseline_wall_pool_min = baseline_cpu_min / args.cores_available

    # Optionally add epoch-1 cost back in for excluded-after-epoch1 cases
    epoch1_extra_wall_min = 0.0
    if args.align_epoch1 and args.epoch1_min_valids is not None:
        epoch1_extra_wall_min = len(excluded_after_epoch1) * args.minutes_per_epoch
        baseline_wall_min += epoch1_extra_wall_min
        baseline_cpu_min += epoch1_extra_wall_min * args.cores_per_job
        baseline_wall_pool_min = baseline_cpu_min / args.cores_available

    # Efficiency metrics
    speedup_vs_baseline = (
        baseline_cpu_min / actual_cpu_min_all if actual_cpu_min_all > 0 else float("inf")
    )

    sampling_efficiency = None
    total_available_cpu_min = None
    if observed_wall_hours is not None and observed_wall_hours > 0:
        total_available_cpu_min = observed_wall_hours * 60.0 * args.cores_available
        sampling_efficiency = actual_cpu_min_all / total_available_cpu_min

    # Report
    print(f"Global target cum_nvalid N = {N_target}\n")
    print("# Summary (inputs & filters)")
    print("----------------------------")
    print(f"Files scanned                         : {total_scanned}")
    if args.min_required_valids is not None:
        print(f"min required max(cum_nvalid)          : {args.min_required_valids}")
    if args.epoch1_min_valids is not None:
        print(f"epoch-1 cutoff (cum_nvalid)           : {args.epoch1_min_valids}  (align-epoch1={'yes' if args.align_epoch1 else 'no'})")
    print(f"Eligible after filters                : {len(per_file)}")
    print(f"Excluded after epoch-1 gate           : {len(excluded_after_epoch1) if args.epoch1_min_valids is not None else 0}")
    print(f"Cases excluded (zero/invalid rate)    : {zero_or_bad_rate}")
    print(f"Included cases in baseline pool       : {len(included_for_baseline)}")
    print(f"Harmonic-mean rate (valids/epoch)     : {r_harm:,.4f}\n")

    print("Smart Sampling (actual usage from .sat rows)")
    print("--------------------------------------------")
    print(f"Actual WALL (job minutes, all files)  : {minutes_to_str(actual_wall_min_all)}")
    print(f"Actual CPU  (core minutes, all files) : {minutes_to_str(actual_cpu_min_all)}")
    print(f"Actual WALL with {args.cores_available} cores : {minutes_to_str(actual_wall_pool_min_all)}\n")

    print("Equi-N Baseline (estimated to reach common N)")
    print("---------------------------------------------")
    print(f"Estimated epochs (sum over cases)     : {epochs_total_est:,.2f} epochs")
    print(f"Estimated WALL (job minutes)          : {minutes_to_str(baseline_wall_min)}")
    print(f"Estimated CPU  (core minutes)         : {minutes_to_str(baseline_cpu_min)}")
    print(f"Estimated WALL with {args.cores_available} cores : {minutes_to_str(baseline_wall_pool_min)}")
    if epoch1_extra_wall_min > 0:
        print(f"(Includes epoch-1 cost added back for excluded cases: {minutes_to_str(epoch1_extra_wall_min)})")
    print()
    print("Comparison (Smart vs Equi-N)")
    print("----------------------------")
    diff_cpu_min = baseline_cpu_min - actual_cpu_min_all
    print(f"CPU difference (baseline - smart)     : {minutes_to_str(diff_cpu_min)}")
    print(f"Speedup (baseline CPU / smart CPU)    : {speedup_vs_baseline:,.2f}x")

    if sampling_efficiency is not None and total_available_cpu_min is not None:
        print("\nEfficiency vs cluster capacity (observed run)")
        print("---------------------------------------------")
        print(f"Total available CPU (given wall)      : {minutes_to_str(total_available_cpu_min)}")
        print(f"Sampling efficiency (smart/available) : {sampling_efficiency*100:.2f}%")

    print("\n# Parameters assumed")
    print("--------------------")
    print(f"minutes-per-epoch                     : {args.minutes_per_epoch}")
    print(f"cores-per-job                         : {args.cores_per_job}")
    print(f"cores-available (pool)                : {args.cores_available}")
    if observed_wall_hours is not None:
        print(f"total-wall-hours (observed)           : {observed_wall_hours:.2f}")


if __name__ == "__main__":
    main()

