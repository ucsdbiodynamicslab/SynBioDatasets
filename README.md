# SynBioDatasets
Curation of high-quality datasets (experimental and simulated) for genetic circuits exhibiting interesting nonlinear behaviors (multistability, limit cycles, etc...). Datasets and benchmarks will be used to develop and analyze system identification methods for nonlinear biosystems.

# Chemical Reaction Network Simulation Guide

## Overview

The `ChemicalReactionNetwork` class now includes three complete simulation methods for modeling chemical reaction networks:

1. **ODE (Ordinary Differential Equations)** - Deterministic continuous simulation
2. **SSA (Stochastic Simulation Algorithm / Gillespie)** - Exact stochastic simulation  
3. **CLE (Chemical Langevin Equation)** - Stochastic continuous simulation with noise

## Basic Usage

### Simple Simulation

```python
from species import ChemicalSpecies
from reaction import ChemicalReaction
from network import ChemicalReactionNetwork
import numpy as np

# Create species
a = ChemicalSpecies("A", initial_conc=1.0)
b = ChemicalSpecies("B", initial_conc=0.0)

# Create reaction A -> B
rxn = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})

# Create network
crn = ChemicalReactionNetwork(rxn)

# Simulate with ODE
t, y = crn.simulate({"A": 1.0, "B": 0.0}, (0, 10), method="ode",
                    t_eval=np.linspace(0, 10, 101))

# Results
print(f"Times: {t}")
print(f"Species A: {y[0]}")
print(f"Species B: {y[1]}")
```

## Simulation Methods

### 1. ODE Simulation

**Best for:** Large populations, continuous approximation, fast deterministic dynamics

```python
t, y = crn.simulate_ode({"A": 1.0, "B": 0.0}, (0, 10), 
                        t_eval=np.linspace(0, 10, 101),
                        method='RK45')  # scipy.integrate.solve_ivp options
```

Features:
- Uses `scipy.integrate.solve_ivp`
- Deterministic time evolution
- Fast, smooth solutions
- Proper handling of degradation (exponential decay)
- Supports all scipy solver options (method='RK45', 'DOP853', etc.)

### 2. SSA (Gillespie Algorithm)

**Best for:** Small populations, stochastic effects important, discrete state space

```python
t, y = crn.simulate_ssa({"A": 100, "B": 0}, t_end=10.0, seed=42)
```

Features:
- Gillespie algorithm (exact stochastic simulation)
- Discrete molecule counts
- Handles both reactions and degradation
- Uses `reaction.fire()` method to update species
- Returns variable-length time series (event-driven)
- Reproducible with seed parameter

### 3. CLE (Chemical Langevin Equation)

**Best for:** Moderate populations, want stochastic effects but continuous state

```python
t, y = crn.simulate_cle({"A": 10.0, "B": 0.0}, (0, 10),
                        t_eval=np.linspace(0, 10, 101), 
                        seed=42, dt_cle=0.01)
```

Features:
- Euler-Maruyama stochastic numerical integration
- Continuous state space with stochastic noise
- Faster than SSA for moderate-large systems
- Includes reaction and degradation noise
- Tunable step size via `dt_cle` parameter

## Using the Wrapper `simulate()` Method

All three methods accessible via the general `simulate()` wrapper:

```python
# ODE
t_ode, y_ode = crn.simulate(initial, (0, 10), method="ode")

# SSA
t_ssa, y_ssa = crn.simulate(initial, (0, 10), method="ssa", seed=42)

# CLE
t_cle, y_cle = crn.simulate(initial, (0, 10), method="cle", seed=42)
```

## Degradation Support

All simulation methods properly handle species degradation defined in `ChemicalSpecies`:

```python
# Create species with exponential degradation
protein = ChemicalSpecies("Protein", initial_conc=1.0, 
                         degrade=True, degradation_rate=0.1)

# Degradation is automatically included in flux calculations
```

**ODE:** Degradation flux term `-k_deg * [X]` added to RHS  
**SSA:** Degradation applied exponentially after each reaction: `c(t+tau) = c(t) * exp(-k_deg * tau)`  
**CLE:** Degradation included in drift term of stochastic differential equation

## Output Format

All methods return:
- `t`: Time array (numpy array)
- `y`: Concentration array with shape `(n_species, len(t))`
  - `y[i, :]` = concentrations of species i over time
  - Access by name: `y[crn.get_species_index("A"), :]`

Species objects are updated in-place with simulation results, so you can access via:
```python
Spec_A = crn.species["A"]
times_A = Spec_A._times
conc_A = Spec_A._concentrations
```

## Advanced Options

### ODE Solver Options
```python
t, y = crn.simulate_ode(initial, (0, 10),
                       method='DOP853',  # Different solver
                       dense_output=True,  # Continuous output
                       max_step=0.1)  # Control step size
```

### SSA Options
```python
t, y = crn.simulate_ssa({"A": 100}, t_end=10.0,
                       seed=12345)  # Reproducible randomness
```

### CLE Options
```python
t, y = crn.simulate_cle(initial, (0, 10),
                       t_eval=np.linspace(0, 10, 1001),
                       seed=12345,
                       dt_cle=0.001)  # Smaller step size for accuracy
```

## Examples

### Example 1: Simple Sequential Reaction (A → B → C)

```python
a = ChemicalSpecies("A", initial_conc=1.0)
b = ChemicalSpecies("B", initial_conc=0.0)
c = ChemicalSpecies("C", initial_conc=0.0)

rxn1 = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})
rxn2 = ChemicalReaction({b: 1}, {c: 1}, "massAction", {"k": 0.3})

crn = ChemicalReactionNetwork(rxn1, rxn2)
t, y = crn.simulate({"A": 1.0, "B": 0.0, "C": 0.0}, (0, 10), method="ode")
```

### Example 2: Bimolecular Reaction with Degradation

```python
a = ChemicalSpecies("A", initial_conc=1.0, degrade=True, degradation_rate=0.1)
b = ChemicalSpecies("B", initial_conc=1.0)
c = ChemicalSpecies("C", initial_conc=0.0)

rxn = ChemicalReaction({a: 1, b: 1}, {c: 1}, "massAction", {"k": 1.0})
crn = ChemicalReactionNetwork(rxn)

# Compare ODE and SSA
t_ode, y_ode = crn.simulate({"A": 100, "B": 100, "C": 0}, (0, 5), method="ode")
t_ssa, y_ssa = crn.simulate({"A": 100, "B": 100, "C": 0}, (0, 5), method="ssa")
```

### Example 3: Stochastic Comparison

```python
# Run 10 SSA simulations and compare
final_values = []
for seed in range(10):
    t, y = crn.simulate(initial, (0, 10), method="ssa", seed=seed)
    final_values.append(y[:, -1])

mean_final = np.mean(final_values, axis=0)
std_final = np.std(final_values, axis=0)
print(f"Mean final concentration: {mean_final}")
print(f"Standard deviation: {std_final}")
```

## Performance Considerations

| Method | Speed | Accuracy | Best Use |
|--------|-------|----------|----------|
| ODE | Fast | High (deterministic) | Large systems, smooth dynamics |
| SSA | Medium-Slow | Exact | Small systems, discrete molecules |
| CLE | Medium | Good | Moderate systems, some stochasticity |

## Troubleshooting

### ODE Solution has negative concentrations
- Increase solver accuracy: `method='DOP853'`
- Reduce time step: `max_step=0.01`
- Check reaction rate constants

### SSA is too slow
- Use CLE instead for faster approximate solution
- Reduce number of molecules (scale down initial concentrations)
- Check if some reactions have very slow rates

### CLE solution doesn't conserve mass
- CLE includes noise which affects conservation slightly
- Reduce `dt_cle` for better accuracy
- Check if degradation is unintentionally high

## References

- Gillespie, D. T. (1977). "Exact stochastic simulation of coupled chemical reactions"
- Higham, D. J. (2008). "Modeling and simulating chemical reactions"

# Simulated Datasets