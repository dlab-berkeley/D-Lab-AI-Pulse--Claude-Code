# Demo 2 - Prompt 2: Modeling Choices Analysis

## Prompt Used:
```
Explain the modeling choices in this DSGE implementation:
- What type of DSGE model is this? (New Keynesian, RBC, etc.)
- What are the key economic assumptions?
- What parameters can users adjust?
- How does estimation work vs simulation/forecasting?
```

---

## Claude's Response Summary:

### Model Types

**Model1002 (NY Fed Production Model)**
- Large-scale New Keynesian with financial frictions
- 70+ endogenous states
- Capital accumulation, sticky prices/wages
- Financial accelerator with credit spreads

**AnSchorfheide (Pedagogical Model)**
- Minimal 3-equation New Keynesian
- Only 8 states, 3 observables
- No capital, no financial frictions

### Key Economic Assumptions

| Feature | AnSchorfheide | Model1002 |
|---------|---------------|-----------|
| Households | Habit formation | Habit + labor supply |
| Pricing | Calvo sticky prices | Calvo prices + wages |
| Capital | None | Full accumulation |
| Financial | None | Financial accelerator |
| Shocks | 3 | 18+ |

### Adjustable Parameters

- **AnSchorfheide**: 13 parameters (τ, κ, ψ_1, ψ_2, etc.)
- **Model1002**: 50+ parameters organized by:
  - Preferences (σ_c, h, ν_l)
  - Rigidities (ζ_p, ζ_w, ι_p, ι_w)
  - Financial (ζ_spb, Φ, S'')
  - Policy (ψ_1, ψ_2, ρ_R)
  - Shock processes (persistences, std devs)

### Estimation vs Forecasting

**Estimation:**
1. Load data from FRED
2. Kalman filter for likelihood
3. MCMC (Metropolis-Hastings) or SMC
4. Output: posterior parameter draws

**Forecasting:**
1. Start from estimated state
2. Draw shocks from posterior
3. Propagate state-space forward
4. Options: unconditional, conditional, alternative policy
