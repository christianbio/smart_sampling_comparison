# SMART vs NOSMART Sampling Comparison

This repository contains scripts to analyze **SMART** versus **NOSMART** sampling strategies in crystal structure prediction (CSP). The workflow constructs discovery timelines, collates results across cases, and evaluates coverage within configurable energy and epoch windows.  

---

## Contents

- **`combine_db.py`**  
  Merges the results of SMART and NOSMART runs into a single combined database.  
  - Ensures schema consistency.  
  - Populates the `origin` table to record provenance (`smart` vs `nosmart`).  
  - Produces a `clustered_cases_combined.db` ready for downstream analysis.  

- **`populate_discovery_all.py`**  
  Populates `class_discovery` tables in each case database. Each entry records when (discovery order and epoch) a unique structure class was first found.  

- **`collate_class_discovery.py`**  
  Collates `class_discovery` tables from all cases into a single combined database (`all_class_discovery`). This unified view enables cross-case comparisons.  

- **`intersect_db.py`**  
  Analyzes the overlap between SMART and NOSMART results in the combined database.  
  - Supports multiple modes (coverage curves, epoch filters, scatter plots).  
  - Energy windows (e.g. *Emin + 10 kJ/mol*) can be applied to restrict analysis to low-energy structures.  
  - Generates coverage curve plots (`coverage_curve.png`) and optional scatter plots.  

- **`compare_smart_vs_equin.py`**  
  Estimates the computational efficiency of SMART sampling versus an **equi-N baseline**.  
  - Uses `.sat` files (per-case saturation curves) to model equi-N costs.  
  - Reports actual vs baseline CPU/wall time, speedup factors, and cluster efficiency.  
  - Accepts filters (e.g. epoch-1 cutoffs) and observed run logs for calibration.  

---

## Typical Workflow

0. Write metadata JSON  

Before discovery tables can be populated, you need a `sampling_metadata.json` file
that maps **case IDs** to their database paths. This metadata file is used by
`populate_discovery_all.py` and `collate_class_discovery.py`.

A helper script can be used to generate this file. For example:

```python
from chemina.sample.sample import Smart_sampling
from chemina.conformers.conformers import ConformerCollection
from chemina.repositories.conformer_database import Conformer_db
import json

conf_db = Conformer_db("mol22.db")
mol_id = conf_db.query("select mol_id from molecule").fetchone()[0]
confs = conf_db.read_to_conformercollection(mol_id)

job_dict = {}
space_groups = {sg: int(1e12) for sg in [1, 2, 4, 14]}  # example

for conf in confs:
    job_dict[conf.label] = {
        "space_groups": space_groups,
        "ids": [conf.label],
    }

smart_sampling = Smart_sampling(
    conformer_collection_list=[confs],
    job_dict=job_dict,
    sample_dir="Sample",
    cores_per_job=20,
)

metadata = {"db_path_dict": smart_sampling.db_path_dict}
with open("sampling_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Wrote sampling_metadata.json")

This will create a JSON file like:

```
{
  "db_path_dict": {
    "mol22-44786623-2-1": "Sample/mol22-44786623-2/sg-1.db",
    "mol22-44786623-2-2": "Sample/mol22-44786623-2/sg-2.db",
    ...
  }
}
```
This file is required for downstream scripts.


1. Combine SMART and NOSMART databases  

   Use `combine_db.py` to merge the outputs of SMART and NOSMART sampling runs into a single database with a consistent schema.  
   For example:  

    ```
    python combine_db.py clustered_cases.db clustered_cases_no_smart.db \
        --out clustered_cases_combined.db
    ```

    where clustered_cases.db is the clustered output db from smart_sampling, and clustered_cases_combined.db is the 
    clustered db from a reference 'no-smart' sampling run, where

    • clustered_cases.db → SMART run  
    • clustered_cases_no_smart.db → NOSMART run  
    • clustered_cases_combined.db → merged output  

2. Populate per-case discovery data  

    ```
    python populate_discovery_all.py --metadata sampling_metadata.json
    ```

    This script will visit all the case databases and add the table *class_discovery* defined by

    ```
    CREATE TABLE IF NOT EXISTS class_discovery (
        unique_id TEXT PRIMARY KEY,
        first_member_id TEXT NOT NULL,
        discovery_order INTEGER NOT NULL,
        epoch_first_discovery INTEGER NOT NULL
    )
    ```

    where `unique_id` is the representative id of the equivalence class, which appears in the equivalent_to table,  
    `first_member_id` is the id of that equivalence class which was discovered first in the run,  
    `discovery_order` is the position that id was discovered, e.g. it was the 73rd structure to be found for this case,  
    and `epoch_first_discovery` is the *global* epoch index on which that structure was discovered.  

    **Note:** `populate_discovery_all.py` requires `.sat` files for each case.  
    These must be located in `Sample/sats/` and named consistently with the case database (e.g., `sg-14.db` → `Sample/sats/sg-14.sat`).  

3. Collate into a combined database  

    ```
    python collate_class_discovery.py --metadata sampling_metadata.json --out clustered_cases_combined.db
    ```

4. Cluster the clustered_cases_combined.db  

    ```
    cspy-db cluster clustered_cases_combined.db
    ```

5. Run intersection analysis  

    ```
    python intersect_db.py clustered_cases_combined.db \
        --mode coverage_curve \
        --max-epoch 260 \
        --step 20 \
        --energy-window 10
    ```

6. Examine output  

    ```
    SMART vs NOSMART coverage curve
    ----------------------------------
    Energy window: Emin + 10
    Definitions:
      A(E): global reps first discovered by SMART at epoch ≤ E, within the energy window
      B   : all NOSMART global reps within the energy window
    |B| (NOSMART set size): 203

    E=20   |A(E)|=208    |A∩B|=197    |A\B|=11     |B\A|=6
    E=40   |A(E)|=209    |A∩B|=198    |A\B|=11     |B\A|=5
    ```

---

## Notes

- The `origin` table must exist and mark each structure as smart or nosmart.  
- The `equivalent_to` table must be populated (clustering step) for correct mapping of global representatives.  
- Plots are written to the working directory (`coverage_curve.png`, `energy_density_overlap.png`).  

---

## Other scripts

### compare_smart_vs_equin.py

Evaluates SMART sampling efficiency versus an equi-N baseline.  

- Reads per-case `.sat` files containing cumulative unique, low-energy, and valid structure counts.  
- Estimates the CPU and wall-clock time that an equi-N baseline (running each case until N valid structures are discovered) would require.  
- Compares this against the actual SMART run.  
- Supports filters (minimum valids, epoch-1 gating) and efficiency metrics relative to available cluster capacity.  
- Optionally reads `chemina.log` to compute observed wall-time for direct efficiency reporting.  

Typical usage:  

```
python compare_smart_vs_equin.py Sample/sats 
–epoch1-min-valids 10 
–align-epoch1 
–logfile .logs/chemina.log
```

This will:  
- Load all `.sat` files from `Sample/sats/`.  
- Exclude cases with fewer than 10 valid structures at epoch 1 (but count their cost back in with `--align-epoch1`).  
- Read start/end timestamps from `.logs/chemina.log` to measure observed wall-time.  
- Report:  
  - Actual SMART sampling wall/CPU usage.  
  - Estimated equi-N baseline wall/CPU usage.  
  - Speedup factor of SMART vs baseline.  
  - Efficiency vs available cluster capacity.  

## `make_timelines.py`

Generate **xmgrace-friendly timelines** (Gantt-style segments) for both **sampling** and **clustering** jobs, using Slurm accounting data. Useful for visualizing campaign concurrency and run cadence on a shared time axis.

### What it reads

- `chemina.log` (to infer *t₀*, the campaign start time from the first timestamped line).
- Job IDs scraped from:
  - `log_dir/**/_submission.sh` (sampling jobs)
  - `log_cluster/*_submission.sh` (clustering jobs)
- Slurm accounting via `sacct` (requires Slurm environment and permission).

### What it writes (to the run directory)

- `sampling_timeline.dat` — line segments for sampling jobs  
- `clustering_timeline.dat` — line segments for clustering jobs  
- `timeline_row_map.tsv` — mapping of row indices to `(type, JobID, Start, End)`

Each `*.dat` file contains blocks like:

```
<start_hours_since_t0>  <row_index>
<end_hours_since_t0>    <row_index>
```
(blank line separates segments). This format plots nicely as line segments in xmgrace (or can be post-processed elsewhere).

### Basic usage

```bash
python make_timelines.py .logs/chemina.log
```

By default, the script assumes the run directory is the parent of the log’s folder (i.e., .../<run>/.logs/chemina.log → run dir is .../<run>). You can override:

```
python make_timelines.py .logs/chemina.log \
  --run-dir . \
  --partition dev_sampling \
  --start-date 2025-08-01 \
  --end-date now
  ```

Or pin the zero-hour explicitly:

```
python make_timelines.py .logs/chemina.log \
  --run-start "2025-08-12 09:30:00"
  ```

