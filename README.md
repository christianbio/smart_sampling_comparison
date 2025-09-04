# SMART vs NOSMART Sampling Comparison

This repository contains scripts to analyze **SMART** versus **NOSMART** sampling strategies in crystal structure prediction (CSP). The workflow constructs discovery timelines, collates results across cases, and evaluates coverage within configurable energy and epoch windows.  

---

## Contents

- **`populate_discovery_all.py`**  
  Populates `class_discovery` tables in each case database. Each entry records when (discovery order and epoch) a unique structure class was first found.  

- **`collate_class_discovery.py`**  
  Collates `class_discovery` tables from all cases into a single combined database (`all_class_discovery`). This unified view enables cross-case comparisons.  

- **`intersect_db.py`**  
  Analyzes the overlap between SMART and NOSMART results in the combined database.  
  - Supports multiple modes (coverage curves, epoch filters, scatter plots).  
  - Energy windows (e.g. *Emin + 10 kJ/mol*) can be applied to restrict analysis to low-energy structures.  
  - Generates coverage curve plots (`coverage_curve.png`) and optional scatter plots.  

---

## Typical Workflow

1. Combine SMART and NOSMART databases  

   Use `combine_db.py` to merge the outputs of SMART and NOSMART sampling runs into a single database with a consistent schema.  
   For example:  

    ```
    python combine_db.py clustered_cases.db clustered_cases_no_smart.db 
–out clustered_cases_combined.db
    ```

    where clustered_cases.db is the clustered output db from smart_sampling, and clustered_cases_combined.db is the 
    clustered db from a reference 'no-smart' sampling run. 

    This will create `clustered_cases_combined.db` containing structures from both runs, with their provenance recorded in the `origin` table (`source='smart'` or `'nosmart'`).  


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

    where `unique_id` is the representative id of 
    the equivalence class, which appears in the equivalent_to table, `first_member_id` is the id of that equivalence class which was discovered first in the run, discovery_order is the position that id was discovered, e.g. it was the 73rd structure to be found for this case. Finally, `epoch_first_discovery` is the *global* epoch index on which that structure was discovered. 



    Note: `populate_discovery_all.py` requires `.sat` files for each case.

    These must be located in `Sample/sats/` and named consistently with the case
    database (e.g., `sg-14.db` → `Sample/sats/sg-14.sat`).



3. Collate into a combined database

```python collate_class_discovery.py --metadata sampling_metadata.json --out clustered_cases_combined.db```

4. Cluster the clustered_cases_combined.db

```
cspy-db cluster clustered_cases_combined.db
```

5. Run intersection analysis

```python intersect_db.py clustered_cases_combined.db \
    --mode coverage_curve \
    --max-epoch 260 \
    --step 20 \
    --energy-window 10
```

6. Examine output

SMART vs NOSMART coverage curve
----------------------------------
Energy window: Emin + 10
Definitions:
  A(E): global reps first discovered by SMART at epoch ≤ E, within the energy window
  B   : all NOSMART global reps within the energy window
|B| (NOSMART set size): 203

E=20   |A(E)|=208    |A∩B|=197    |A\B|=11     |B\A|=6
E=40   |A(E)|=209    |A∩B|=198    |A\B|=11     |B\A|=5


...

#NOTES

	
    The origin table must exist and mark each structure as smart or nosmart.
	•	The equivalent_to table must be populated (clustering step) for correct mapping of global representatives.
	•	Plots are written to the working directory (coverage_curve.png, energy_density_overlap.png).