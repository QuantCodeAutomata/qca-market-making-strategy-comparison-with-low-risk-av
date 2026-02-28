# Market Making Strategy Comparison with Multiple Risk Aversion Levels

This project implements and compares inventory-aware versus symmetric market making strategies across three different risk aversion levels (γ=0.01, 0.1, 0.5).

## Overview

The project simulates market making strategies based on the Avellaneda-Stoikov framework, comparing:
- **Inventory Strategy**: Quotes centered at reservation price to manage inventory risk
- **Symmetric Strategy**: Quotes centered at mid-price with no inventory adjustment

## Experiments

### Experiment 1: Low Risk Aversion (γ=0.01)
Validates that strategies converge when risk aversion approaches zero.

### Experiment 2: Medium Risk Aversion (γ=0.1)
Demonstrates clear variance reduction benefits of inventory management.

### Experiment 3: High Risk Aversion (γ=0.5)
Shows maximum inventory control at the cost of reduced profits.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:
```bash
python run_experiments.py
```

Run specific experiment:
```python
from experiments import run_experiment_1, run_experiment_2, run_experiment_3

# Run experiment 1 (γ=0.01)
results = run_experiment_1()
print(results)
```

## Running Tests

```bash
pytest
```

## Project Structure

```
.
├── src/
│   ├── simulator.py       # Core simulation engine
│   ├── strategies.py      # Market making strategies
│   └── experiments.py     # Experiment configurations and runners
├── tests/
│   ├── test_simulator.py
│   ├── test_strategies.py
│   └── test_experiments.py
├── run_experiments.py     # Main entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Key Parameters

- **Mid-price dynamics**: Discrete Brownian motion with σ=2, dt=0.005, T=1
- **Order arrivals**: Exponential decay intensity λ(δ)=A*exp(-k*δ) where A=140, k=1.5
- **Initial conditions**: S₀=100, q₀=0, X₀=0
- **Simulations**: 1000 Monte Carlo paths per strategy

## Expected Results

Results are validated against the original paper's Tables 1, 2, and 3.

| Risk Aversion | Strategy    | Mean Profit | Std Profit | Mean Final q | Std Final q |
|---------------|-------------|-------------|------------|--------------|-------------|
| γ=0.01        | Inventory   | ~66.78      | ~8.76      | ~-0.02       | ~4.70       |
| γ=0.01        | Symmetric   | ~67.36      | ~13.40     | ~-0.31       | ~8.65       |
| γ=0.1         | Inventory   | ~62.94      | ~5.89      | ~0.10        | ~2.80       |
| γ=0.1         | Symmetric   | ~67.21      | ~13.43     | ~-0.018      | ~8.66       |
| γ=0.5         | Inventory   | ~33.92      | ~4.72      | ~-0.02       | ~1.88       |
| γ=0.5         | Symmetric   | ~66.20      | ~14.53     | ~0.25        | ~9.06       |

## License

MIT
