# DSGE.jl Quick-Start Guide for PhD Researchers

A practical guide to getting started with the New York Fed DSGE Model in Julia.

---

## 1. Installation

### Prerequisites

- **Julia 1.x** (tested on Julia 1.6+)
- A working internet connection for downloading data from FRED

### Install the Package

Open Julia and enter package mode by pressing `]`, then run:

```julia
pkg> add DSGE
```

This will install DSGE.jl and all its dependencies, including:
- `ModelConstructors.jl` - Base model type definitions
- `StateSpaceRoutines.jl` - Kalman filtering and smoothing
- `SMC.jl` - Sequential Monte Carlo sampling
- `FredData.jl` - Automatic data retrieval from FRED

### Verify Installation

```julia
using DSGE
m = AnSchorfheide()  # Load a simple test model
println("DSGE.jl installed successfully!")
```

---

## 2. FRED API Key Setup

DSGE.jl automatically downloads macroeconomic data from the Federal Reserve Economic Data (FRED) database. You must obtain a free API key to use this feature.

### Step 1: Register at FRED

1. Go to [https://fred.stlouisfed.org/](https://fred.stlouisfed.org/)
2. Click "My Account" and create a free account
3. Navigate to "API Keys" in your account settings
4. Request a new API key

### Step 2: Configure Julia to Use Your Key

Create a file at `~/.fredapikey` (Linux/Mac) or `%USERPROFILE%\.fredapikey` (Windows) containing only your API key:

```
your_api_key_here
```

Alternatively, set the environment variable:

```bash
export FRED_API_KEY="your_api_key_here"
```

### Step 3: Verify FRED Access

```julia
using FredData
f = Fred()
# If this runs without error, your API key is configured correctly
```

---

## 3. Running Your First Example

### The Simplest Workflow

Create a file called `my_first_dsge.jl`:

```julia
using DSGE, ModelConstructors

# 1. Instantiate the model
#    AnSchorfheide is a simple 3-equation New Keynesian model
#    Good for learning the workflow before moving to larger models
m = AnSchorfheide()

# 2. Check model parameters
println("Number of parameters: ", length(m.parameters))
println("Number of observables: ", length(m.observables))

# 3. Solve the model at the prior mode
solve(m)
println("Model solved successfully!")
```

### Running the Full NY Fed DSGE Workflow

For a complete estimation and forecast example, use the `run_default.jl` script:

```julia
using DSGE, ModelConstructors, Distributed
using Nullables, DataFrames, OrderedCollections, Dates

# Instantiate the NY Fed DSGE model (Model 1002)
m = Model1002("ss10")

# Configure data vintage and forecast start
m <= Setting(:data_vintage, "181115")
m <= Setting(:date_forecast_start, quartertodate("2018-Q4"))

# Quick test settings (remove for production runs)
m <= Setting(:n_mh_simulations, 100)  # Reduce MH steps for testing
m <= Setting(:use_population_forecast, false)
m <= Setting(:forecast_block_size, 5)

# Load data (requires FRED API key)
df = load_data(m, try_disk = false, check_empty_columns = false, summary_statistics = :none)
data = df_to_matrix(m, df)

# Estimate the model
estimate(m, data)

# Generate forecasts
output_vars = [:histobs, :forecastobs]
forecast_one(m, :mode, :none, output_vars; check_empty_columns = false)
compute_meansbands(m, :mode, :none, output_vars; check_empty_columns = false)
```

**Note:** Full estimation with default settings takes 2-3 hours. The quick settings above reduce this to ~10-15 minutes for testing.

### Understanding the Output

After running, check these locations (relative to your Julia packages directory):

- **Estimation results:** `~/.julia/packages/DSGE/.../save/output_data/m1002/ss10/estimate/`
- **Forecast results:** `~/.julia/packages/DSGE/.../save/output_data/m1002/ss10/forecast/`

---

## 4. Where to Look Next

### Documentation Resources

| Resource | Description |
|----------|-------------|
| [Official Docs](https://frbny-dsge.github.io/DSGE.jl/stable) | Full API reference and tutorials |
| [Model Documentation PDF](https://github.com/FRBNY-DSGE/DSGE.jl/blob/main/docs/DSGE_Model_Documentation_1002.pdf) | Technical details of Model 1002 |
| [Example Scripts](https://github.com/FRBNY-DSGE/DSGE.jl/tree/main/examples) | Ready-to-run examples |

### Example Scripts to Study

| Script | Purpose |
|--------|---------|
| `run_default.jl` | Basic estimation and forecasting workflow |
| `make_packet.jl` | Generate publication-ready plots and tables |
| `test_smc.jl` | Use Sequential Monte Carlo instead of MCMC |
| `regime_switching.jl` | Time-varying parameters and policy regimes |
| `decompose_forecast.jl` | Understand forecast changes |

### Available Models

Start simple and work up:

1. **`AnSchorfheide`** - 3-equation textbook model (start here!)
2. **`SmetsWouters`** - Medium-scale workhorse model
3. **`Model990`** - Earlier NY Fed DSGE version
4. **`Model1002`** - Current NY Fed DSGE model (most features)

### Key Documentation Sections

For your research, prioritize reading:

1. **[Input Data](https://frbny-dsge.github.io/DSGE.jl/stable/input_data/)** - How to load and customize data
2. **[Estimation](https://frbny-dsge.github.io/DSGE.jl/stable/estimation/)** - MCMC and SMC sampling options
3. **[Forecasting](https://frbny-dsge.github.io/DSGE.jl/stable/forecast/)** - Running and interpreting forecasts
4. **[Model Design](https://frbny-dsge.github.io/DSGE.jl/stable/model_design/)** - Building your own models
5. **[Alternative Policies](https://frbny-dsge.github.io/DSGE.jl/stable/altpolicy/)** - Policy counterfactuals

### Building Your Own Model

If you want to implement a custom DSGE model:

1. Start by copying the `AnSchorfheide` source code from:
   `src/models/representative/an_schorfheide/`

2. Study these files in order:
   - `an_schorfheide.jl` - Model definition and parameters
   - `eqcond.jl` - Equilibrium conditions
   - `measurement.jl` - Measurement equation
   - `observables.jl` - Observable definitions

3. Read [Solving the Model](https://frbny-dsge.github.io/DSGE.jl/stable/solving/) for the required equation format

---

## 5. Common Issues and Solutions

### Windows Users

- Enable [long paths](https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file#enable-long-paths-in-windows-10-version-1607-and-later) to avoid filename length errors
- If you see `AssertionError: length(dirs) == 1`, see [this issue](https://github.com/JuliaLang/Pkg.jl/issues/1943)

### Missing Data Columns

If you get warnings about empty data columns, add:
```julia
load_data(m, check_empty_columns = false)
```

Some data series require non-FRED sources that must be manually provided.

### Memory Issues with Full-Distribution Forecasts

Reduce the forecast block size:
```julia
m <= Setting(:forecast_block_size, 1000)  # Default is 5000
```

### Parallel Computing

For faster estimation and forecasting:
```julia
using Distributed
addprocs(4)  # Add 4 worker processes
@everywhere using DSGE

m <= Setting(:use_parallel_workers, true)
# ... run your estimation/forecast ...

rmprocs(workers())  # Clean up when done
```

---

## 6. Getting Help

- **GitHub Issues:** [https://github.com/FRBNY-DSGE/DSGE.jl/issues](https://github.com/FRBNY-DSGE/DSGE.jl/issues)
- **Contact developers:** See `Project.toml` for maintainer emails
- **Julia Discourse:** [https://discourse.julialang.org/](https://discourse.julialang.org/) (tag with `dsge`)

---

*This guide was generated based on DSGE.jl v1.3.0. Check the official documentation for the latest updates.*
