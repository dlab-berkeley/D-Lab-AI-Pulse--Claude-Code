# Demo 3 - Prompt 2: Refactoring Plan

## Prompt Used:
```
Create a refactoring plan for this codebase. Goals:
- Make it work on any computer (no hardcoded paths)
- Single source of truth for data loading
- Clear project structure with config file
- Remove dead/duplicate code
- Proper entry point that doesn't use exec()
- Keep all functionality intact
```

---

## Claude's Response:

### Proposed New Folder Structure

```
thesis_analysis/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Central configuration (paths, API keys)
│   └── constants.py         # Fixed mappings (state abbrevs, etc.)
├── src/
│   ├── data/
│   │   ├── loader.py        # Single source of truth for data loading
│   │   ├── downloaders/     # census.py, bls.py, fred.py, survey.py
│   │   └── processors/      # survey_processor.py, data_merger.py
│   ├── analysis/
│   │   ├── chapter3.py
│   │   ├── chapter4.py
│   │   └── robustness.py
│   ├── visualization/
│   │   ├── figures.py       # Combined figure generation
│   │   └── tables.py
│   └── utils/
│       ├── stats.py
│       └── formatting.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── results/
│   ├── figures/
│   └── tables/
├── main.py                  # Clean entry point
├── requirements.txt
├── .env.example
└── README.md
```

### Files to Delete
- `old_scripts/` (entire directory)
- `for_advisor/` (entire directory - extract region logic first)
- `run_all.py` (replaced by main.py)
- `helpers.py` (content moved to src/utils/)

### Consolidation
- 5 duplicate `load_data()` functions → 1 in `src/data/loader.py`
- 2 figure scripts → 1 combined `src/visualization/figures.py`
- Scattered path definitions → `config/settings.py`
- Scattered mappings → `config/constants.py`

### 7-Phase Implementation Order
1. Foundation (config + utils modules)
2. Migrate downloaders
3. Migrate processors
4. Migrate analysis scripts
5. Migrate visualization
6. Create new entry point
7. Cleanup old files
