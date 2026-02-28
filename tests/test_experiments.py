"""Tests for experiments."""

import pytest
import numpy as np
from src.experiments import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_all_experiments,
    print_results
)


class TestExperiments:
    """Tests for experiment execution."""
    
    def test_run_experiment_1(self):
        """Test experiment 1 execution."""
        results = run_experiment_1(n_paths=100, seed=42)
        
        # Check structure
        assert 'gamma' in results
        assert 'spread' in results
        assert 'inventory_strategy' in results
        assert 'symmetric_strategy' in results
        
        # Check gamma value
        assert results['gamma'] == 0.01
        
        # Check that both strategies have results
        for strategy_key in ['inventory_strategy', 'symmetric_strategy']:
            strategy_results = results[strategy_key]
            assert 'mean_pnl' in strategy_results
            assert 'std_pnl' in strategy_results
            assert 'mean_inventory' in strategy_results
            assert 'std_inventory' in strategy_results
    
    def test_run_experiment_2(self):
        """Test experiment 2 execution."""
        results = run_experiment_2(n_paths=100, seed=42)
        
        # Check structure
        assert 'gamma' in results
        assert 'spread' in results
        assert 'inventory_strategy' in results
        assert 'symmetric_strategy' in results
        
        # Check gamma value
        assert results['gamma'] == 0.1
        
        # Check that both strategies have results
        for strategy_key in ['inventory_strategy', 'symmetric_strategy']:
            strategy_results = results[strategy_key]
            assert 'mean_pnl' in strategy_results
            assert 'std_pnl' in strategy_results
            assert 'mean_inventory' in strategy_results
            assert 'std_inventory' in strategy_results
    
    def test_run_experiment_3(self):
        """Test experiment 3 execution."""
        results = run_experiment_3(n_paths=100, seed=42)
        
        # Check structure
        assert 'gamma' in results
        assert 'spread' in results
        assert 'inventory_strategy' in results
        assert 'symmetric_strategy' in results
        
        # Check gamma value
        assert results['gamma'] == 0.5
        
        # Check that both strategies have results
        for strategy_key in ['inventory_strategy', 'symmetric_strategy']:
            strategy_results = results[strategy_key]
            assert 'mean_pnl' in strategy_results
            assert 'std_pnl' in strategy_results
            assert 'mean_inventory' in strategy_results
            assert 'std_inventory' in strategy_results
    
    def test_run_all_experiments(self):
        """Test running all experiments together."""
        results = run_all_experiments(n_paths=50, seed=42)
        
        # Check that all experiments are present
        assert 'exp_1' in results
        assert 'exp_2' in results
        assert 'exp_3' in results
        
        # Check gamma values for each experiment
        assert results['exp_1']['gamma'] == 0.01
        assert results['exp_2']['gamma'] == 0.1
        assert results['exp_3']['gamma'] == 0.5
    
    def test_inventory_variance_reduction(self):
        """Test that inventory strategy reduces variance compared to symmetric."""
        # Run with small sample for speed
        results = run_experiment_2(n_paths=200, seed=42)
        
        inv_std_pnl = results['inventory_strategy']['std_pnl']
        sym_std_pnl = results['symmetric_strategy']['std_pnl']
        
        inv_std_inventory = results['inventory_strategy']['std_inventory']
        sym_std_inventory = results['symmetric_strategy']['std_inventory']
        
        # Inventory strategy should have lower inventory variance
        # (This is the key benefit of the strategy)
        assert inv_std_inventory < sym_std_inventory
        
        # For medium risk aversion (gamma=0.1), should also reduce P&L variance
        assert inv_std_pnl < sym_std_pnl
    
    def test_reproducibility(self):
        """Test that experiments are reproducible with same seed."""
        results1 = run_experiment_1(n_paths=50, seed=123)
        results2 = run_experiment_1(n_paths=50, seed=123)
        
        # Results should be identical
        assert np.isclose(
            results1['inventory_strategy']['mean_pnl'],
            results2['inventory_strategy']['mean_pnl']
        )
        assert np.isclose(
            results1['symmetric_strategy']['mean_pnl'],
            results2['symmetric_strategy']['mean_pnl']
        )
    
    def test_spread_decreases_with_gamma(self):
        """Test that optimal spread decreases as risk aversion increases."""
        exp1 = run_experiment_1(n_paths=10, seed=42)
        exp2 = run_experiment_2(n_paths=10, seed=42)
        exp3 = run_experiment_3(n_paths=10, seed=42)
        
        # Spread should decrease as gamma increases
        assert exp1['spread'] > exp2['spread']
        assert exp2['spread'] > exp3['spread']
    
    def test_high_gamma_reduces_profit(self):
        """Test that high risk aversion reduces expected profit."""
        exp2 = run_experiment_2(n_paths=200, seed=42)
        exp3 = run_experiment_3(n_paths=200, seed=42)
        
        # High risk aversion (exp3) should have lower profit for inventory strategy
        # due to more conservative quoting
        inv_profit_medium = exp2['inventory_strategy']['mean_pnl']
        inv_profit_high = exp3['inventory_strategy']['mean_pnl']
        
        assert inv_profit_high < inv_profit_medium
    
    def test_print_results_no_error(self):
        """Test that print_results doesn't raise errors."""
        results = run_experiment_1(n_paths=10, seed=42)
        
        # Should not raise any exceptions
        try:
            print_results(results, "Test Experiment")
        except Exception as e:
            pytest.fail(f"print_results raised {type(e).__name__}: {e}")
    
    def test_expected_outcomes_experiment_1(self):
        """Test that experiment 1 results are in reasonable range."""
        results = run_experiment_1(n_paths=1000, seed=42)
        
        inv = results['inventory_strategy']
        sym = results['symmetric_strategy']
        
        # Check that results are in plausible range (not exact match due to randomness)
        # Expected: Inventory profit ~66.78, Symmetric profit ~67.36
        assert 50 < inv['mean_pnl'] < 85
        assert 50 < sym['mean_pnl'] < 85
        
        # Check variance ordering
        assert inv['std_inventory'] < sym['std_inventory']
    
    def test_expected_outcomes_experiment_2(self):
        """Test that experiment 2 results are in reasonable range."""
        results = run_experiment_2(n_paths=1000, seed=42)
        
        inv = results['inventory_strategy']
        sym = results['symmetric_strategy']
        
        # Check that results are in plausible range
        # Expected: Inventory profit ~62.94, Symmetric profit ~67.21
        assert 45 < inv['mean_pnl'] < 80
        assert 50 < sym['mean_pnl'] < 85
        
        # Check variance reduction
        assert inv['std_pnl'] < sym['std_pnl']
        assert inv['std_inventory'] < sym['std_inventory']
    
    def test_expected_outcomes_experiment_3(self):
        """Test that experiment 3 results are in reasonable range."""
        results = run_experiment_3(n_paths=1000, seed=42)
        
        inv = results['inventory_strategy']
        sym = results['symmetric_strategy']
        
        # Check that results are in plausible range
        # Expected: Inventory profit ~33.92, Symmetric profit ~66.20
        assert 20 < inv['mean_pnl'] < 50
        assert 50 < sym['mean_pnl'] < 85
        
        # High gamma should show very tight inventory control
        assert inv['std_inventory'] < 3.0
        assert inv['std_inventory'] < sym['std_inventory']
