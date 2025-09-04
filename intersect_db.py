#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from cspy.db import CspDataStore

# ---------------------------
# Helpers to build filters
# ---------------------------


def build_energy_filter_cte(energy_window):
    """
    Return (cte_sql, join_sql) that limits clusters to those whose representative
    (equivalent_to.unique_id) has crystal.energy ≤ Emin + energy_window,
    where Emin is the global min energy across all crystals.
    If energy_window is None, returns ("", "") (no filter).
    """
    if energy_window is None:
        return "", ""

    ew = float(energy_window)
    cte = f"""
min_e AS (
  SELECT MIN(energy) AS Emin
  FROM crystal
  WHERE energy IS NOT NULL
),
eligible_reps AS (
  SELECT DISTINCT et.unique_id AS rep
  FROM equivalent_to et
  JOIN crystal cr ON cr.id = et.unique_id
  JOIN min_e     ON 1=1
  WHERE cr.energy IS NOT NULL
    AND cr.energy <= (min_e.Emin + {ew})
)
"""
    # When present, join coverage/reps/etc. on eligible_reps(rep)
    return cte, "JOIN eligible_reps er ON er.rep = coverage.rep"


def build_filter_cte(mode, epoch, epoch_min, epoch_max, acd_table):
    """
    Return (cte_sql, filter_join_sql) to filter clusters by first-discovery epoch
    across ANY case (based on the aggregated discovery table).
    If mode is None, returns ("", "") and no filtering is applied.
    """
    if mode is None:
        return "", ""

    if mode == "discovered_by":
        if epoch is None:
            raise ValueError("--epoch required for mode=discovered_by")
        cte = f"""
filtered_reps AS (
  SELECT unique_id AS rep
  FROM {acd_table}
  GROUP BY unique_id
  HAVING MIN(epoch_first_discovery) <= {int(epoch)}
)
"""
    elif mode == "first_at":
        if epoch is None:
            raise ValueError("--epoch required for mode=first_at")
        cte = f"""
filtered_reps AS (
  SELECT unique_id AS rep
  FROM {acd_table}
  GROUP BY unique_id
  HAVING MIN(epoch_first_discovery) = {int(epoch)}
)
"""
    elif mode == "range":
        if epoch_min is None or epoch_max is None:
            raise ValueError("--epoch-min and --epoch-max required for mode=range")
        cte = f"""
filtered_reps AS (
  SELECT unique_id AS rep
  FROM {acd_table}
  GROUP BY unique_id
  HAVING MIN(epoch_first_discovery) BETWEEN {int(epoch_min)} AND {int(epoch_max)}
)
"""
    elif mode == "coverage_curve":
        # No filtering CTE; coverage curve is handled in main()
        return "", ""
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Later subqueries can join filtered_reps on rep
    return cte, "JOIN filtered_reps fr ON fr.rep = coverage.rep"


# ---------------------------
# SQL builders (epoch + energy aware)
# ---------------------------


def _with_block(*ctes):
    blocks = [c.strip() for c in ctes if c and c.strip()]
    if not blocks:
        return ""  # <- critical: no WITH when no blocks
    return "WITH\n" + ",\n".join(blocks) + "\n"


def _coverage_joins(epoch_join, energy_join):
    joins = []
    if epoch_join:
        joins.append(epoch_join)
    if energy_join:
        joins.append(energy_join)
    return ("\n  " + "\n  ".join(joins)) if joins else ""


def make_sql_counts(epoch_cte, epoch_join, energy_cte, energy_join):
    with_prefix = _with_block(epoch_cte, energy_cte)
    joins_clause = _coverage_joins(epoch_join, energy_join)

    return f"""
{with_prefix}
members AS (
  -- all mapped members
  SELECT e.unique_id AS rep, c.id, COALESCE(o.source,'unknown') AS source
  FROM equivalent_to e
  JOIN crystal      c ON c.id = e.equivalent_id
  LEFT JOIN origin  o ON o.id = c.id
  WHERE e.equivalent_id IS NOT NULL
  UNION ALL
  -- add the representative itself to catch singletons
  SELECT u.unique_id AS rep, u.unique_id AS id, COALESCE(o.source,'unknown') AS source
  FROM (SELECT DISTINCT unique_id FROM equivalent_to) u
  LEFT JOIN origin o ON o.id = u.unique_id
),
coverage AS (
  SELECT rep,
         MAX(CASE WHEN source IN ('smart','both')   THEN 1 ELSE 0 END) AS has_smart,
         MAX(CASE WHEN source IN ('nosmart','both') THEN 1 ELSE 0 END) AS has_nosmart
  FROM members
  GROUP BY rep
)
SELECT
  (SELECT COUNT(*) FROM crystal)                                      AS total_structs,
  (SELECT COUNT(*) FROM origin WHERE source='smart')                   AS n_smart,
  (SELECT COUNT(*) FROM origin WHERE source='nosmart')                 AS n_nosmart,
  (SELECT COUNT(DISTINCT coverage.rep)
     FROM coverage{joins_clause})                                      AS total_clusters,
  (SELECT COUNT(*)
     FROM coverage{joins_clause}
     WHERE coverage.has_smart=1 AND coverage.has_nosmart=1)           AS clusters_with_both,
  (SELECT COUNT(*)
     FROM coverage{joins_clause}
     WHERE coverage.has_smart=1 AND coverage.has_nosmart=0)           AS clusters_smart_only,
  (SELECT COUNT(*)
     FROM coverage{joins_clause}
     WHERE coverage.has_smart=0 AND coverage.has_nosmart=1)           AS clusters_nosmart_only;
"""


def make_sql_struct_match_counts(epoch_cte, epoch_join, energy_cte, energy_join):
    with_prefix = _with_block(epoch_cte, energy_cte)
    joins_clause = _coverage_joins(epoch_join, energy_join)

    return f"""
{with_prefix}
members AS (
  SELECT e.unique_id AS rep, c.id, COALESCE(o.source,'unknown') AS source
  FROM equivalent_to e
  JOIN crystal      c ON c.id = e.equivalent_id
  LEFT JOIN origin  o ON o.id = c.id
  WHERE e.equivalent_id IS NOT NULL
  UNION ALL
  SELECT u.unique_id AS rep, u.unique_id AS id, COALESCE(o.source,'unknown') AS source
  FROM (SELECT DISTINCT unique_id FROM equivalent_to) u
  LEFT JOIN origin o ON o.id = u.unique_id
),
coverage AS (
  SELECT rep,
         MAX(CASE WHEN source IN ('smart','both')   THEN 1 ELSE 0 END) AS has_smart,
         MAX(CASE WHEN source IN ('nosmart','both') THEN 1 ELSE 0 END) AS has_nosmart
  FROM members
  GROUP BY rep
),
members_filtered AS (
  SELECT m.*
  FROM members m
  JOIN coverage cv ON cv.rep = m.rep
  {joins_clause}
)
SELECT
  SUM(CASE WHEN m.source IN ('smart','both')   THEN 1 ELSE 0 END)                       AS smart_total,
  SUM(CASE WHEN m.source IN ('smart','both')   AND cv.has_nosmart=1 THEN 1 ELSE 0 END)  AS smart_matched_in_nosmart,
  SUM(CASE WHEN m.source IN ('nosmart','both') THEN 1 ELSE 0 END)                       AS nosmart_total,
  SUM(CASE WHEN m.source IN ('nosmart','both') AND cv.has_smart=1 THEN 1 ELSE 0 END)    AS nosmart_matched_in_smart
FROM members_filtered m
JOIN coverage cv ON cv.rep = m.rep;
"""


def make_sql_points(epoch_cte, epoch_join, energy_cte, energy_join):
    with_prefix = _with_block(epoch_cte, energy_cte)
    joins_clause = _coverage_joins(epoch_join, energy_join)

    return f"""
{with_prefix}
members AS (
  SELECT e.unique_id AS rep, c.id, COALESCE(o.source,'unknown') AS source
  FROM equivalent_to e
  JOIN crystal      c ON c.id = e.equivalent_id
  LEFT JOIN origin  o ON o.id = c.id
  WHERE e.equivalent_id IS NOT NULL
  UNION ALL
  SELECT u.unique_id AS rep, u.unique_id AS id, COALESCE(o.source,'unknown') AS source
  FROM (SELECT DISTINCT unique_id FROM equivalent_to) u
  LEFT JOIN origin o ON o.id = u.unique_id
),
coverage AS (
  SELECT rep,
         MAX(CASE WHEN source IN ('smart','both')   THEN 1 ELSE 0 END) AS has_smart,
         MAX(CASE WHEN source IN ('nosmart','both') THEN 1 ELSE 0 END) AS has_nosmart
  FROM members
  GROUP BY rep
),
reps AS (
  SELECT rep,
         CASE
           WHEN has_smart=1 AND has_nosmart=1 THEN 'both'
           WHEN has_smart=1 AND has_nosmart=0 THEN 'smart_only'
           WHEN has_smart=0 AND has_nosmart=1 THEN 'nosmart_only'
           ELSE 'unknown'
         END AS cat
  FROM coverage
  {joins_clause}
)
SELECT r.cat, cr.energy, cr.density
FROM reps r
JOIN crystal cr ON cr.id = r.rep
WHERE cr.energy IS NOT NULL AND cr.density IS NOT NULL;
"""


# ---------------------------
# Misc helpers
# ---------------------------


def pct(x, n):
    return f"{(100.0*x/n):.2f}%" if n else "n/a"


def coverage_vs_epoch(db, acd_table, max_epoch=200, step=5, energy_window=None):
    """
    Print a header explaining A(E) and B, show the energy window (if any),
    print |B|, then stream per-epoch rows. Returns the results list.
    """
    # Optional energy-eligible global reps (by representative energy)
    eligible = None
    if energy_window is not None:
        sql_elig = f"""
        WITH
        min_e AS (
          SELECT MIN(energy) AS Emin FROM crystal WHERE energy IS NOT NULL
        ),
        eligible_reps AS (
          SELECT DISTINCT et.unique_id AS rep
          FROM equivalent_to et
          JOIN crystal cr ON cr.id = et.unique_id
          JOIN min_e     ON 1=1
          WHERE cr.energy IS NOT NULL
            AND cr.energy <= (min_e.Emin + {float(energy_window)})
        )
        SELECT rep FROM eligible_reps
        """
        eligible = {r[0] for r in db.query(sql_elig).fetchall()}

    # All NOSMART reps (fixed set)
    sql_B = """
        WITH members AS (
          SELECT e.unique_id AS rep, COALESCE(o.source,'unknown') AS source
          FROM equivalent_to e
          JOIN crystal      c ON c.id = e.equivalent_id
          LEFT JOIN origin  o ON o.id = c.id
          UNION ALL
          SELECT u.unique_id AS rep, COALESCE(o.source,'unknown') AS source
          FROM (SELECT DISTINCT unique_id FROM equivalent_to) u
          LEFT JOIN origin o ON o.id = u.unique_id
        )
        SELECT DISTINCT rep
        FROM members
        WHERE source IN ('nosmart','both')
    """
    nosmart_ids = {r[0] for r in db.query(sql_B).fetchall()}
    if eligible is not None:
        nosmart_ids &= eligible

    # ---- Header summary ----
    print("# SMART vs NOSMART coverage curve")
    print("----------------------------------")
    if energy_window is not None:
        print(f"Energy window: Emin + {energy_window:g}")
    print("Definitions:")
    if energy_window is not None:
        print(
            "  A(E): global reps first discovered by SMART at epoch ≤ E, within the energy window"
        )
        print("  B   : all NOSMART global reps within the energy window")
    else:
        print("  A(E): global reps first discovered by SMART at epoch ≤ E")
        print("  B   : all NOSMART global reps (no energy filter)")
    print(f"|B| (NOSMART set size): {len(nosmart_ids):,}")
    print()

    # ---- Epoch sweep ----
    results = []
    for epoch in range(step, max_epoch + 1, step):
        sql_map = f"""
            SELECT DISTINCT COALESCE(et.unique_id, et2.unique_id) AS glob_rep
            FROM {acd_table} acd
            LEFT JOIN equivalent_to et  ON et.unique_id = acd.unique_id
            LEFT JOIN equivalent_to et2 ON et2.equivalent_id = acd.unique_id
            WHERE acd.epoch_first_discovery <= {epoch}
              AND (et.unique_id IS NOT NULL OR et2.unique_id IS NOT NULL)
        """
        smart_ids = {r[0] for r in db.query(sql_map).fetchall()}
        if eligible is not None:
            smart_ids &= eligible

        inter = smart_ids & nosmart_ids
        a_only = smart_ids - nosmart_ids
        b_only = nosmart_ids - smart_ids

        print(
            f"E={epoch:<4d} |A(E)|={len(smart_ids):<6d} "
            f"|A∩B|={len(inter):<6d} |A\\B|={len(a_only):<6d} |B\\A|={len(b_only):<6d}"
        )
        results.append((epoch, len(smart_ids), len(inter), len(a_only), len(b_only)))

    return results


def plot_coverage_curve(results):
    import matplotlib.pyplot as plt

    epochs = [r[0] for r in results]
    A = [r[1] for r in results]
    inter = [r[2] for r in results]
    A_only = [r[3] for r in results]
    B_only = [r[4] for r in results]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, A, label="SMART uniques (|A(E)|)")
    plt.plot(epochs, inter, label="Intersection (|A∩B|)")
    plt.plot(epochs, A_only, label="SMART-only (|A\\B|)")
    plt.plot(epochs, B_only, label="NOSMART-only (|B\\A|)")
    plt.xlabel("Epoch cutoff E")
    plt.ylabel("Cluster counts")
    plt.title("SMART vs NOSMART coverage vs epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("coverage_curve.png", dpi=200)
    print("Wrote coverage_curve.png")


def print_ab_definitions_and_counts(db, acd_table, E, energy_cte, energy_window=None):
    """
    Prints:
      - Clear definitions of A and B (with optional energy filter info)
      - |A| and |B| as counts of *global representatives* (clusters)
      - Member-structure breakdowns inside A and B by source
    """
    # Build WITH section reusing your energy CTE if present
    with_section = "WITH\n" + energy_cte.strip() + ",\n" if energy_cte else "WITH\n"
    sql = f"""
{with_section}
params(epoch_cutoff) AS (VALUES ({E})),

per_case AS (
  SELECT unique_id AS case_uid,
         MIN(epoch_first_discovery) AS min_epoch_case
  FROM {acd_table}
  GROUP BY unique_id
),
mapped AS (
  SELECT DISTINCT et.unique_id AS glob_rep, pc.min_epoch_case
  FROM per_case pc
  JOIN equivalent_to et
    ON et.unique_id     = pc.case_uid
    OR et.equivalent_id = pc.case_uid
),
smart_global AS (
  SELECT glob_rep,
         MIN(min_epoch_case) AS min_epoch_smart
  FROM mapped
  GROUP BY glob_rep
),
A AS (
  SELECT glob_rep AS rep
  FROM smart_global
  WHERE min_epoch_smart <= (SELECT * FROM params)
),
members AS (
  SELECT e.unique_id AS rep, COALESCE(o.source,'unknown') AS source
  FROM equivalent_to e
  JOIN crystal      c ON c.id = e.equivalent_id
  LEFT JOIN origin  o ON o.id = c.id
  UNION ALL
  SELECT u.unique_id AS rep, COALESCE(o.source,'unknown') AS source
  FROM (SELECT DISTINCT unique_id FROM equivalent_to) u
  LEFT JOIN origin o ON o.id = u.unique_id
),
B AS (
  SELECT DISTINCT rep
  FROM members
  WHERE source IN ('nosmart','both')
),

-- Apply energy filter to A/B if energy_cte was provided
A2 AS (SELECT A.rep FROM A {("JOIN eligible_reps erA ON erA.rep = A.rep") if energy_cte else ""}),
B2 AS (SELECT B.rep FROM B {("JOIN eligible_reps erB ON erB.rep = B.rep") if energy_cte else ""}),

-- Member breakdowns inside A2 and B2
A_members AS (
  SELECT m.source
  FROM members m
  JOIN A2 ON A2.rep = m.rep
),
B_members AS (
  SELECT m.source
  FROM members m
  JOIN B2 ON B2.rep = m.rep
)
SELECT
  (SELECT COUNT(*) FROM A2) AS A_reps,
  (SELECT COUNT(*) FROM B2) AS B_reps,
  (SELECT COUNT(*) FROM A_members)                                           AS A_members_total,
  (SELECT COUNT(*) FROM A_members WHERE source IN ('smart','both'))          AS A_members_from_smart,
  (SELECT COUNT(*) FROM A_members WHERE source IN ('nosmart','both'))        AS A_members_from_nosmart,
  (SELECT COUNT(*) FROM A_members WHERE source NOT IN ('smart','nosmart','both')) AS A_members_unknown,
  (SELECT COUNT(*) FROM B_members)                                           AS B_members_total,
  (SELECT COUNT(*) FROM B_members WHERE source IN ('smart','both'))          AS B_members_from_smart,
  (SELECT COUNT(*) FROM B_members WHERE source IN ('nosmart','both'))        AS B_members_from_nosmart,
  (SELECT COUNT(*) FROM B_members WHERE source NOT IN ('smart','nosmart','both')) AS B_members_unknown;
"""
    (
        A_reps,
        B_reps,
        A_m_total,
        A_m_smart,
        A_m_nos,
        A_m_unk,
        B_m_total,
        B_m_smart,
        B_m_nos,
        B_m_unk,
    ) = db.query(sql).fetchone()

    print("A definition:")
    print(
        "  A(E) = set of global reps (clusters) whose earliest SMART discovery epoch ≤ E"
    )
    print("B definition:")
    print(
        "  B    = set of global reps (clusters) that have ANY NOSMART membership (source ∈ {nosmart, both})"
    )
    if energy_window is not None:
        print(
            f"Energy filter applied: reps with crystal.energy ≤ Emin + {energy_window:g}"
        )
    else:
        print("Energy filter applied: none")
    print()

    print(f"|A| (clusters) = {A_reps:,}")
    print(f"  members in A (structures)         : {A_m_total:,}")
    print(f"    from SMART (incl. both)         : {A_m_smart:,}")
    print(f"    from NOSMART (incl. both)       : {A_m_nos:,}")
    print(f"    unknown                         : {A_m_unk:,}")
    print(f"|B| (clusters) = {B_reps:,}")
    print(f"  members in B (structures)         : {B_m_total:,}")
    print(f"    from SMART (incl. both)         : {B_m_smart:,}")
    print(f"    from NOSMART (incl. both)       : {B_m_nos:,}")
    print(f"    unknown                         : {B_m_unk:,}")
    print()


# ---------------------------
# Main
# ---------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Report overlap between smart vs nosmart sets in a combined clustered DB, with optional epoch & energy filters."
    )
    ap.add_argument(
        "db", help="Path to combined SQLite DB (with origin + equivalent_to populated)"
    )
    ap.add_argument(
        "--acd-table",
        default="all_class_discovery",
        help="Aggregated discovery table name (default: all_class_discovery)",
    )

    # Epoch filter knobs for standard reports / scatter
    ap.add_argument(
        "--mode",
        choices=["discovered_by", "first_at", "range", "coverage_curve"],
        help="Filter clusters by earliest discovery epoch across ANY case",
    )
    ap.add_argument(
        "--epoch", type=int, help="Epoch value for modes discovered_by / first_at"
    )
    ap.add_argument(
        "--epoch-min", type=int, help="Min epoch (inclusive) for mode=range"
    )
    ap.add_argument(
        "--epoch-max", type=int, help="Max epoch (inclusive) for mode=range"
    )
    ap.add_argument(
        "--max-epoch",
        type=int,
        default=200,
        help="Max epoch for coverage_curve (default=200)",
    )
    ap.add_argument(
        "--step", type=int, default=5, help="Step size for coverage_curve (default=5)"
    )

    # Energy filter (applies to overview, per-structure, scatter, and SMART-by-epoch stats)
    ap.add_argument(
        "--energy-window",
        type=float,
        help="Only include clusters whose representative energy ≤ (global min + WINDOW). Units must match crystal.energy.",
    )

    # SMART-by-epoch set stats
    ap.add_argument(
        "--smart-by-epoch",
        type=int,
        help="Compute set stats between A(E)=SMART uniques first-discovered by epoch ≤E "
        "(from all_class_discovery) mapped to global reps, and B=ALL NOSMART uniques.",
    )

    ap.add_argument(
        "--scatter",
        action="store_true",
        help="Also make an energy–density scatter plot",
    )
    args = ap.parse_args()

    # Build filters
    try:
        epoch_cte, epoch_join = build_filter_cte(
            args.mode, args.epoch, args.epoch_min, args.epoch_max, args.acd_table
        )
    except ValueError as e:
        raise SystemExit(str(e))
    energy_cte, energy_join = build_energy_filter_cte(args.energy_window)

    db = CspDataStore(args.db)

    # ---- SMART-by-epoch (A(E) vs B) using global reps (with optional energy filter) ----
    if args.smart_by_epoch is not None:
        E = int(args.smart_by_epoch)

        # We’ll apply the energy filter by intersecting A and B with eligible_reps if present.
        energy_with = energy_cte.strip() if energy_cte else ""
        energy_a_join = (
            "JOIN eligible_reps erA ON erA.rep = A.rep" if energy_cte else ""
        )
        energy_b_join = (
            "JOIN eligible_reps erB ON erB.rep = B.rep" if energy_cte else ""
        )

        # Build the WITH section safely (no backslashes inside f-string expressions)
        if energy_with:
            # energy_with already contains the CTEs for min_e, eligible_reps
            with_section = (
                f"WITH\n{energy_with},\nparams(epoch_cutoff) AS (VALUES ({E}))"
            )
        else:
            with_section = f"WITH params(epoch_cutoff) AS (VALUES ({E}))"

        sql_ab = f"""
        {with_section},

        -- Per-case SMART IDs with earliest epoch in that case table
        per_case AS (
          SELECT unique_id AS case_uid,
                MIN(epoch_first_discovery) AS min_epoch_case
          FROM {args.acd_table}
          GROUP BY unique_id
        ),

        -- Map per-case IDs to global representatives in the combined DB
        mapped AS (
          SELECT DISTINCT et.unique_id AS glob_rep, pc.min_epoch_case
          FROM per_case pc
          JOIN equivalent_to et
            ON et.unique_id     = pc.case_uid
            OR et.equivalent_id = pc.case_uid
        ),

        -- Earliest SMART epoch per global rep across all contributing cases
        smart_global AS (
          SELECT glob_rep,
                MIN(min_epoch_case) AS min_epoch_smart
          FROM mapped
          GROUP BY glob_rep
        ),

        -- A(E): global reps whose earliest SMART discovery ≤ E
        A AS (
          SELECT glob_rep AS rep
          FROM smart_global
          WHERE min_epoch_smart <= (SELECT epoch_cutoff FROM params)
        ),

        -- Build membership with sources from the combined DB (rep + members)
        members AS (
          SELECT e.unique_id AS rep, COALESCE(o.source,'unknown') AS source
          FROM equivalent_to e
          JOIN crystal      c ON c.id = e.equivalent_id
          LEFT JOIN origin  o ON o.id = c.id
          UNION ALL
          SELECT u.unique_id AS rep, COALESCE(o.source,'unknown') AS source
          FROM (SELECT DISTINCT unique_id FROM equivalent_to) u
          LEFT JOIN origin o ON o.id = u.unique_id
        ),

        -- B: ALL NOSMART global reps (has any member from nosmart or both)
        B AS (
          SELECT DISTINCT rep
          FROM members
          WHERE source IN ('nosmart','both')
        ),

        A2 AS (SELECT A.rep FROM A {energy_a_join}),
        B2 AS (SELECT B.rep FROM B {energy_b_join}),

        AiB     AS (SELECT rep FROM A2 INTERSECT SELECT rep FROM B2),
        AminusB AS (SELECT rep FROM A2 EXCEPT    SELECT rep FROM B2),
        BminusA AS (SELECT rep FROM B2 EXCEPT    SELECT rep FROM A2)

        SELECT
          (SELECT COUNT(*) FROM A2),
          (SELECT COUNT(*) FROM B2),
          (SELECT COUNT(*) FROM AiB),
          (SELECT COUNT(*) FROM AminusB),
          (SELECT COUNT(*) FROM BminusA);
        """

        a, b, ab, a_only, b_only = db.query(sql_ab).fetchone()
        print("# SMART-by-epoch vs NOSMART (set intersection, global reps)")
        print("-----------------------------------------------------------")
        print(f"E (epoch cutoff)                  : {E}")
        if args.energy_window is not None:
            print(f"Energy window                     : Emin + {args.energy_window:g}")
        print(f"|A(E)|  SMART uniques by E        : {a:,}")
        print(f"|B|     NOSMART uniques (all)     : {b:,}")
        print(f"|A∩B|   intersection               : {ab:,}")
        print("|A\\B|   SMART-only by E            :", f"{a_only:,}")
        print("|B\\A|   NOSMART-only               :", f"{b_only:,}")
        print()
        # Extra: clear definitions + richer counts for A and B
        print_ab_definitions_and_counts(
            db=db,
            acd_table=args.acd_table,
            E=E,
            energy_cte=energy_cte,  # pass the string; helper will handle whether it’s empty
        )

    # ---- Special coverage curve mode (not energy-filtered) ----
    if args.mode == "coverage_curve":
        results = coverage_vs_epoch(
            db,
            args.acd_table,
            max_epoch=args.max_epoch,
            step=args.step,
            energy_window=args.energy_window,
        )
        plot_coverage_curve(results)
        db.close()
        return

    # ---- Standard overview (optionally epoch- & energy-filtered) ----
    sql_counts = make_sql_counts(epoch_cte, epoch_join, energy_cte, energy_join)
    (
        total_structs,
        n_smart,
        n_nosmart,
        total_clusters,
        clusters_with_both,
        clusters_smart_only,
        clusters_nosmart_only,
    ) = db.query(sql_counts).fetchone()

    print("# Overview")
    print("----------")
    print(f"Total structures                    : {total_structs:,}")
    print(f"  from smart                        : {n_smart:,}")
    print(f"  from nosmart                      : {n_nosmart:,}")
    if args.mode in ("discovered_by", "first_at"):
        print(
            f"Epoch filter                        : mode={args.mode} (epoch={args.epoch})"
        )
    elif args.mode == "range":
        print(
            f"Epoch filter                        : mode={args.mode} (range={args.epoch_min}..{args.epoch_max})"
        )
    else:
        print("Epoch filter                        : (none)")
    if args.energy_window is not None:
        print(f"Energy window                       : Emin + {args.energy_window:g}")
    else:
        print("Energy window                       : (none)")
    print()
    print(f"Total clusters (unique_id)          : {total_clusters:,}")
    print(f"Clusters with both sources          : {clusters_with_both:,}")
    print(f"Clusters smart-only                 : {clusters_smart_only:,}")
    print(f"Clusters nosmart-only               : {clusters_nosmart_only:,}")
    print()

    # ---- Per-structure coverage ----
    sql_struct_match = make_sql_struct_match_counts(
        epoch_cte, epoch_join, energy_cte, energy_join
    )
    smart_total, smart_matched, nos_total, nos_matched = db.query(
        sql_struct_match
    ).fetchone()

    print("# Per-structure coverage")
    print("------------------------")
    print(f"Smart structures                    : {smart_total:,}")
    print(
        f"  matched in nosmart (same cluster) : {smart_matched:,}  ({pct(smart_matched, smart_total)})"
    )
    print(f"Nosmart structures                  : {nos_total:,}")
    print(
        f"  matched in smart (same cluster)   : {nos_matched:,}  ({pct(nos_matched, nos_total)})"
    )
    print()

    # ---- Optional scatter ----
    if args.scatter:
        sql_points = make_sql_points(epoch_cte, epoch_join, energy_cte, energy_join)
        rows = db.query(sql_points).fetchall()
        if rows:
            cats = [r[0] for r in rows]
            energies = [r[1] for r in rows]
            densities = [r[2] for r in rows]

            color_map = {
                "both": "lightgreen",
                "smart_only": "blue",
                "nosmart_only": "red",
            }
            colors = [color_map.get(c, "black") for c in cats]

            plt.figure(figsize=(6, 5))
            plt.scatter(densities, energies, c=colors, s=6, alpha=0.7)
            plt.xlabel("Density (g/cm³)")
            plt.ylabel("Energy")
            plt.title("Energy–Density overlap")
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_map[k],
                    markersize=8,
                    label=k,
                )
                for k in ["both", "smart_only", "nosmart_only"]
            ]
            plt.legend(handles=legend_elements, loc="best")
            plt.tight_layout()
            plt.savefig("energy_density_overlap.png", dpi=200)
            print("Wrote scatter: energy_density_overlap.png")
        else:
            print("No points matched the filters; skipping scatter.")

    db.close()


if __name__ == "__main__":
    main()
