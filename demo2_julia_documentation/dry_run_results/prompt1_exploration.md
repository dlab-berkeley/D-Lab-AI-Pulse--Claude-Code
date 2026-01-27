# Demo 2 - Prompt 1: DSGE.jl Exploration

## Prompt Used:
```
Explore the DSGE.jl Julia codebase. Give me an overview of:
- Its general purpose (what economic questions does it answer?)
- The main file structure
- Julia version and key dependencies from Project.toml
- Where to find example scripts
```

---

## Claude's Response Summary:

### General Purpose

**DSGE.jl** is a Julia package from the **Federal Reserve Bank of New York** for Dynamic Stochastic General Equilibrium models. It answers:

- Economic Forecasting (GDP, inflation, unemployment)
- Monetary Policy Analysis
- Shock Decomposition (what drives economic fluctuations)
- Impulse Response Analysis
- Scenario Analysis

### File Structure (12 major modules)

```
src/
├── models/           # Model implementations (An-Schorfheide, Model1002, etc.)
├── solve/            # Solution algorithms (gensys, klein)
├── estimate/         # Estimation (Kalman filter, MCMC, SMC)
├── forecast/         # Forecasting routines
├── data/             # Data loading (FRED API integration)
├── analysis/         # Results analysis
├── altpolicy/        # Alternative monetary policies
└── plot/             # Visualization
```

### Julia Version & Dependencies

- **Julia**: v1.x (v1.1.6+)
- **47 dependencies** including:
  - Core: LinearAlgebra, Distributions, Optim, ForwardDiff
  - State-space: StateSpaceRoutines, SMC
  - Data: FredData, CSV, DataFrames, HDF5
  - Plotting: Plots, StatsPlots

### Example Scripts (in `/examples/`)

1. **`run_default.jl`** - Beginner example (data → estimate → forecast)
2. **`make_packet.jl`** - Production workflow with parallel computing
3. **`test_smc.jl`** - Sequential Monte Carlo estimation
4. Model complexity: AnSchorfheide (simple) → Model1002 (production)
