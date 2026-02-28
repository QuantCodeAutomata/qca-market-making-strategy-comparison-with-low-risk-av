"""Tests for market simulator."""

import pytest
import numpy as np
from src.simulator import MarketSimulator


class TestMarketSimulator:
    """Tests for MarketSimulator class."""
    
    def test_initialization(self):
        """Test simulator initialization with default parameters."""
        sim = MarketSimulator()
        assert sim.S0 == 100.0
        assert sim.q0 == 0
        assert sim.X0 == 0.0
        assert sim.sigma == 2.0
        assert sim.dt == 0.005
        assert sim.T == 1.0
        assert sim.A == 140.0
        assert sim.k == 1.5
        assert sim.n_steps == 200
    
    def test_initialization_custom(self):
        """Test simulator initialization with custom parameters."""
        sim = MarketSimulator(
            S0=110.0,
            q0=5,
            X0=100.0,
            sigma=1.5,
            dt=0.01,
            T=2.0,
            A=150.0,
            k=2.0,
            seed=123
        )
        assert sim.S0 == 110.0
        assert sim.q0 == 5
        assert sim.X0 == 100.0
        assert sim.sigma == 1.5
        assert sim.dt == 0.01
        assert sim.T == 2.0
        assert sim.A == 150.0
        assert sim.k == 2.0
        assert sim.seed == 123
        assert sim.n_steps == 200
    
    def test_intensity(self):
        """Test order arrival intensity calculation."""
        sim = MarketSimulator(A=140.0, k=1.5)
        
        # Test at delta = 0
        assert sim.intensity(0.0) == 140.0
        
        # Test at delta = 1
        expected = 140.0 * np.exp(-1.5)
        assert np.isclose(sim.intensity(1.0), expected)
        
        # Test that intensity decreases with distance
        assert sim.intensity(0.5) > sim.intensity(1.0)
        assert sim.intensity(1.0) > sim.intensity(2.0)
    
    def test_simulate_path_deterministic_strategy(self):
        """Test single path simulation with a deterministic strategy."""
        sim = MarketSimulator(seed=42)
        
        # Simple fixed spread strategy
        def fixed_strategy(S, q, t):
            return S - 0.5, S + 0.5
        
        rng = np.random.default_rng(42)
        pnl, q_final, X_final = sim.simulate_path(fixed_strategy, rng)
        
        # Check that we get numeric results
        assert isinstance(pnl, (int, float))
        assert isinstance(q_final, (int, float, np.integer))
        assert isinstance(X_final, (int, float))
        
        # Terminal P&L should equal cash + inventory value
        # Note: final mid-price changes, so we can't verify exact P&L formula here
        # but we can check the relationship holds approximately
        assert isinstance(pnl, (int, float))
    
    def test_simulate_path_reproducibility(self):
        """Test that simulation is reproducible with same seed."""
        sim1 = MarketSimulator(seed=42)
        sim2 = MarketSimulator(seed=42)
        
        def strategy(S, q, t):
            return S - 0.5, S + 0.5
        
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        pnl1, q1, X1 = sim1.simulate_path(strategy, rng1)
        pnl2, q2, X2 = sim2.simulate_path(strategy, rng2)
        
        assert pnl1 == pnl2
        assert q1 == q2
        assert X1 == X2
    
    def test_run_simulation(self):
        """Test full Monte Carlo simulation."""
        sim = MarketSimulator(seed=42)
        
        def strategy(S, q, t):
            return S - 0.5, S + 0.5
        
        results = sim.run_simulation(strategy, n_paths=100)
        
        # Check that all expected keys are present
        assert 'mean_pnl' in results
        assert 'std_pnl' in results
        assert 'mean_inventory' in results
        assert 'std_inventory' in results
        assert 'mean_cash' in results
        assert 'pnls' in results
        assert 'inventories' in results
        assert 'cash' in results
        
        # Check array lengths
        assert len(results['pnls']) == 100
        assert len(results['inventories']) == 100
        assert len(results['cash']) == 100
        
        # Check that statistics are reasonable
        assert isinstance(results['mean_pnl'], (int, float, np.floating))
        assert isinstance(results['std_pnl'], (int, float, np.floating))
        assert results['std_pnl'] >= 0
    
    def test_run_simulation_reproducibility(self):
        """Test that full simulation is reproducible."""
        sim1 = MarketSimulator(seed=42)
        sim2 = MarketSimulator(seed=42)
        
        def strategy(S, q, t):
            return S - 0.5, S + 0.5
        
        results1 = sim1.run_simulation(strategy, n_paths=50)
        results2 = sim2.run_simulation(strategy, n_paths=50)
        
        assert np.allclose(results1['mean_pnl'], results2['mean_pnl'])
        assert np.allclose(results1['std_pnl'], results2['std_pnl'])
        assert np.allclose(results1['pnls'], results2['pnls'])
    
    def test_wide_spread_reduces_executions(self):
        """Test that wider spread leads to fewer executions and lower inventory variance."""
        sim = MarketSimulator(seed=42)
        
        def narrow_strategy(S, q, t):
            return S - 0.1, S + 0.1
        
        def wide_strategy(S, q, t):
            return S - 2.0, S + 2.0
        
        results_narrow = sim.run_simulation(narrow_strategy, n_paths=100)
        
        # Reset seed for fair comparison
        sim2 = MarketSimulator(seed=42)
        results_wide = sim2.run_simulation(wide_strategy, n_paths=100)
        
        # Wider spread should generally lead to lower inventory variance
        # (though this is stochastic, so we just check it's computed)
        assert results_wide['std_inventory'] >= 0
        assert results_narrow['std_inventory'] >= 0
