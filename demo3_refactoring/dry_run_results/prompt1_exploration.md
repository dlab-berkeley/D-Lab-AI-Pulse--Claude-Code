# Demo 3 - Prompt 1: Exploration Results

## Prompt Used:
```
Explore this research codebase in the messy_codebase folder.
I'm a PhD student who built this over 2 years of thesis work
and need to clean it up before my defense.

Give me an overview of:
- The overall structure and what each file does
- The data pipeline (download -> process -> merge -> analyze)
- Main issues you see with the organization
```

---

## Claude's Response:

### Overall Structure

Your codebase represents a complete empirical research pipeline with these main components:

**Root-level scripts (19 files):**
- 4 data downloaders (BLS, Census, FRED, Qualtrics survey)
- 1 data processor (survey cleaning)
- 1 data merger (combines all sources)
- 3 analysis scripts (Chapter 3, Chapter 4, robustness checks)
- 2 figure generators (Chapters 3 & 4)
- 1 table generator
- 1 master pipeline runner
- 1 helper utilities file

**Subdirectories:**
- `data_raw/` - downloaded raw data
- `data_processed/` - cleaned intermediate data
- `figures/` - PNG and PDF output figures
- `tables/` - LaTeX table outputs
- `outputs/` - regression results and summary statistics
- `for_advisor/` - ad-hoc analysis for meetings
- `old_scripts/` - deprecated code from earlier versions

---

### Data Pipeline (Download → Process → Merge → Analyze)

**Phase 1: Download (Multiple Sources)**
- download_bls.py: State-level unemployment rates
- download_census.py: ACS 5-year estimates
- download_fred.py: Macroeconomic indicators
- download_survey.py: Primary survey data from Qualtrics

**Phase 2: Process**
- process_survey.py: Cleans demographics, creates outcome variables

**Phase 3: Merge**
- merge_all_data.py: Creates final analysis dataset

**Phase 4: Analysis**
- analysis_chapter3.py: Individual-level determinants (3 models)
- analysis_chapter4.py: Contextual/state-level effects (3 models)
- analysis_robustness.py: Sensitivity checks for journal R&R

**Phase 5: Output Generation**
- make_figures_chapter3.py, make_figures_chapter4.py
- make_tables.py
- run_all.py: Master orchestrator

---

### Major Issues Identified

1. **Hardcoded Paths Throughout (CRITICAL)** - Every script uses absolute Windows paths
2. **Massive Code Duplication** - load_data() repeated 5 times identically
3. **Mixed Execution Methods** - run_all.py uses exec() instead of imports
4. **Incomplete Ad-hoc Scripts** - for_advisor/ has one-off scripts
5. **Old Scripts Cluttering** - old_scripts/ should be deleted
6. **No Data Versioning** - No record of when/how data was downloaded
7. **Inconsistent Methodology** - Different SE methods without documentation
8. **Hard-coded Results** - Table coefficients typed manually, not extracted
9. **Missing Error Handling** - No validation between pipeline stages
10. **Temporal Data Issues** - Macro data not properly matched to survey dates
11. **Missing Documentation** - No README, no codebook
