"""Experiment configurations and runners for market making strategy comparison."""

import numpy as np
from typing import Dict, Any
from src.simulator import MarketSimulator
from src.strategies import create_inventory_strategy, create_symmetric_strategy


def run_experiment_1(n_paths: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """
    Run Experiment 1: Low Risk Aversion (γ=0.01).
    
    Validates that strategies converge when risk aversion approaches zero.
    
    Args:
        n_paths: Number of Monte Carlo simulation paths
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing results for both strategies
    """
    # Parameters
    gamma = 0.01
    sigma = 2.0
    k = 1.5
    A = 140.0
    dt = 0.005
    T = 1.0
    S0 = 100.0
    
    # Create simulator
    simulator = MarketSimulator(
        S0=S0,
        q0=0,
        X0=0.0,
        sigma=sigma,
        dt=dt,
        T=T,
        A=A,
        k=k,
        seed=seed
    )
    
    # Create strategies
    inventory_strategy = create_inventory_strategy(gamma, sigma, k, T)
    symmetric_strategy = create_symmetric_strategy(gamma, k)
    
    # Run simulations
    print(f"\nExperiment 1: γ={gamma}")
    print(f"Expected spread: {(2.0 / gamma) * np.log(1.0 + gamma / k):.4f}")
    
    print("\nRunning Inventory Strategy...")
    inventory_results = simulator.run_simulation(inventory_strategy, n_paths)
    
    print("Running Symmetric Strategy...")
    symmetric_results = simulator.run_simulation(symmetric_strategy, n_paths)
    
    results = {
        'gamma': gamma,
        'spread': (2.0 / gamma) * np.log(1.0 + gamma / k),
        'inventory_strategy': inventory_results,
        'symmetric_strategy': symmetric_results
    }
    
    print_results(results, "Experiment 1 (γ=0.01)")
    
    return results


def run_experiment_2(n_paths: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """
    Run Experiment 2: Medium Risk Aversion (γ=0.1).
    
    Demonstrates clear variance reduction benefits of inventory management.
    
    Args:
        n_paths: Number of Monte Carlo simulation paths
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing results for both strategies
    """
    # Parameters
    gamma = 0.1
    sigma = 2.0
    k = 1.5
    A = 140.0
    dt = 0.005
    T = 1.0
    S0 = 100.0
    
    # Create simulator
    simulator = MarketSimulator(
        S0=S0,
        q0=0,
        X0=0.0,
        sigma=sigma,
        dt=dt,
        T=T,
        A=A,
        k=k,
        seed=seed
    )
    
    # Create strategies
    inventory_strategy = create_inventory_strategy(gamma, sigma, k, T)
    symmetric_strategy = create_symmetric_strategy(gamma, k)
    
    # Run simulations
    print(f"\nExperiment 2: γ={gamma}")
    print(f"Expected spread: {(2.0 / gamma) * np.log(1.0 + gamma / k):.4f}")
    
    print("\nRunning Inventory Strategy...")
    inventory_results = simulator.run_simulation(inventory_strategy, n_paths)
    
    print("Running Symmetric Strategy...")
    symmetric_results = simulator.run_simulation(symmetric_strategy, n_paths)
    
    results = {
        'gamma': gamma,
        'spread': (2.0 / gamma) * np.log(1.0 + gamma / k),
        'inventory_strategy': inventory_results,
        'symmetric_strategy': symmetric_results
    }
    
    print_results(results, "Experiment 2 (γ=0.1)")
    
    return results


def run_experiment_3(n_paths: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """
    Run Experiment 3: High Risk Aversion (γ=0.5).
    
    Demonstrates maximum inventory control at the cost of reduced profits.
    
    Args:
        n_paths: Number of Monte Carlo simulation paths
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing results for both strategies
    """
    # Parameters
    gamma = 0.5
    sigma = 2.0
    k = 1.5
    A = 140.0
    dt = 0.005
    T = 1.0
    S0 = 100.0
    
    # Create simulator
    simulator = MarketSimulator(
        S0=S0,
        q0=0,
        X0=0.0,
        sigma=sigma,
        dt=dt,
        T=T,
        A=A,
        k=k,
        seed=seed
    )
    
    # Create strategies
    inventory_strategy = create_inventory_strategy(gamma, sigma, k, T)
    symmetric_strategy = create_symmetric_strategy(gamma, k)
    
    # Run simulations
    print(f"\nExperiment 3: γ={gamma}")
    print(f"Expected spread: {(2.0 / gamma) * np.log(1.0 + gamma / k):.4f}")
    
    print("\nRunning Inventory Strategy...")
    inventory_results = simulator.run_simulation(inventory_strategy, n_paths)
    
    print("Running Symmetric Strategy...")
    symmetric_results = simulator.run_simulation(symmetric_strategy, n_paths)
    
    results = {
        'gamma': gamma,
        'spread': (2.0 / gamma) * np.log(1.0 + gamma / k),
        'inventory_strategy': inventory_results,
        'symmetric_strategy': symmetric_results
    }
    
    print_results(results, "Experiment 3 (γ=0.5)")
    
    return results


def print_results(results: Dict[str, Any], title: str) -> None:
    """
    Print formatted results.
    
    Args:
        results: Dictionary containing simulation results
        title: Title for the results display
    """
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"Risk Aversion (γ): {results['gamma']}")
    print(f"Spread: {results['spread']:.4f}")
    print(f"\n{'-' * 80}")
    print(f"{'Strategy':<20} {'Mean Profit':<15} {'Std Profit':<15} {'Mean Final q':<15} {'Std Final q':<15}")
    print(f"{'-' * 80}")
    
    inv = results['inventory_strategy']
    print(f"{'Inventory':<20} {inv['mean_pnl']:<15.2f} {inv['std_pnl']:<15.2f} "
          f"{inv['mean_inventory']:<15.2f} {inv['std_inventory']:<15.2f}")
    
    sym = results['symmetric_strategy']
    print(f"{'Symmetric':<20} {sym['mean_pnl']:<15.2f} {sym['std_pnl']:<15.2f} "
          f"{sym['mean_inventory']:<15.2f} {sym['std_inventory']:<15.2f}")
    print(f"{'-' * 80}")
    
    # Calculate variance reduction
    pnl_var_reduction = (1 - (inv['std_pnl'] / sym['std_pnl'])) * 100
    inv_var_reduction = (1 - (inv['std_inventory'] / sym['std_inventory'])) * 100
    
    print(f"\nVariance Reduction by Inventory Strategy:")
    print(f"  P&L Variance Reduction: {pnl_var_reduction:.1f}%")
    print(f"  Inventory Variance Reduction: {inv_var_reduction:.1f}%")
    print(f"{'=' * 80}\n")


def run_all_experiments(n_paths: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """
    Run all three experiments.
    
    Args:
        n_paths: Number of Monte Carlo simulation paths
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing results for all experiments
    """
    results = {
        'exp_1': run_experiment_1(n_paths, seed),
        'exp_2': run_experiment_2(n_paths, seed),
        'exp_3': run_experiment_3(n_paths, seed)
    }
    
    # Print summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: All Experiments")
    print("=" * 80)
    print(f"{'Gamma':<10} {'Strategy':<15} {'Mean Profit':<15} {'Std Profit':<15} "
          f"{'Mean q':<12} {'Std q':<12}")
    print("-" * 80)
    
    for exp_name in ['exp_1', 'exp_2', 'exp_3']:
        exp = results[exp_name]
        gamma = exp['gamma']
        
        inv = exp['inventory_strategy']
        print(f"{gamma:<10} {'Inventory':<15} {inv['mean_pnl']:<15.2f} {inv['std_pnl']:<15.2f} "
              f"{inv['mean_inventory']:<12.2f} {inv['std_inventory']:<12.2f}")
        
        sym = exp['symmetric_strategy']
        print(f"{gamma:<10} {'Symmetric':<15} {sym['mean_pnl']:<15.2f} {sym['std_pnl']:<15.2f} "
              f"{sym['mean_inventory']:<12.2f} {sym['std_inventory']:<12.2f}")
        print()
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_all_experiments()
