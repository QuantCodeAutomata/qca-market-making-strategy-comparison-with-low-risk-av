"""Core market making simulation engine."""

import numpy as np
from typing import Tuple, Callable


class MarketSimulator:
    """Simulates market making with discrete mid-price dynamics and Poisson order arrivals."""
    
    def __init__(
        self,
        S0: float = 100.0,
        q0: int = 0,
        X0: float = 0.0,
        sigma: float = 2.0,
        dt: float = 0.005,
        T: float = 1.0,
        A: float = 140.0,
        k: float = 1.5,
        seed: int = 42
    ):
        """
        Initialize the market simulator.
        
        Args:
            S0: Initial mid-price
            q0: Initial inventory
            X0: Initial cash position
            sigma: Volatility parameter for mid-price
            dt: Time step size
            T: Terminal time
            A: Order arrival intensity parameter
            k: Exponential decay parameter for arrival intensity
            seed: Random seed for reproducibility
        """
        self.S0 = S0
        self.q0 = q0
        self.X0 = X0
        self.sigma = sigma
        self.dt = dt
        self.T = T
        self.A = A
        self.k = k
        self.seed = seed
        self.n_steps = int(T / dt)
        
    def intensity(self, delta: float) -> float:
        """
        Calculate order arrival intensity based on distance from mid-price.
        
        Args:
            delta: Distance from mid-price
            
        Returns:
            Arrival intensity λ(δ) = A * exp(-k * δ)
        """
        return self.A * np.exp(-self.k * delta)
    
    def simulate_path(
        self,
        quote_strategy: Callable[[float, int, float], Tuple[float, float]],
        rng: np.random.Generator
    ) -> Tuple[float, int, float]:
        """
        Simulate one path of market making.
        
        Args:
            quote_strategy: Function that takes (S_t, q_t, t) and returns (bid_price, ask_price)
            rng: Random number generator
            
        Returns:
            Tuple of (terminal_pnl, final_inventory, final_cash)
        """
        S = self.S0
        q = self.q0
        X = self.X0
        
        for i in range(self.n_steps):
            t = i * self.dt
            
            # Get quotes from strategy
            bid_price, ask_price = quote_strategy(S, q, t)
            
            # Calculate distances from mid-price
            delta_bid = S - bid_price
            delta_ask = ask_price - S
            
            # Calculate intensities
            lambda_bid = self.intensity(delta_bid)
            lambda_ask = self.intensity(delta_ask)
            
            # Simulate executions as Bernoulli trials
            prob_bid = lambda_bid * self.dt
            prob_ask = lambda_ask * self.dt
            
            # Bid execution (buy from market maker, increases inventory)
            if rng.random() < prob_bid:
                q += 1
                X -= bid_price
                
            # Ask execution (sell from market maker, decreases inventory)
            if rng.random() < prob_ask:
                q -= 1
                X += ask_price
            
            # Update mid-price with discrete Brownian motion
            # S_{t+dt} = S_t + ε_t * σ * sqrt(dt) where ε_t ∈ {+1, -1} with P=0.5
            epsilon = 1.0 if rng.random() < 0.5 else -1.0
            S += epsilon * self.sigma * np.sqrt(self.dt)
        
        # Calculate terminal P&L
        terminal_pnl = X + q * S
        
        return terminal_pnl, q, X
    
    def run_simulation(
        self,
        quote_strategy: Callable[[float, int, float], Tuple[float, float]],
        n_paths: int = 1000
    ) -> dict:
        """
        Run Monte Carlo simulation with multiple paths.
        
        Args:
            quote_strategy: Function that takes (S_t, q_t, t) and returns (bid_price, ask_price)
            n_paths: Number of simulation paths
            
        Returns:
            Dictionary with simulation results
        """
        rng = np.random.default_rng(self.seed)
        
        pnls = np.zeros(n_paths)
        final_inventories = np.zeros(n_paths)
        final_cash = np.zeros(n_paths)
        
        for i in range(n_paths):
            pnl, q_final, X_final = self.simulate_path(quote_strategy, rng)
            pnls[i] = pnl
            final_inventories[i] = q_final
            final_cash[i] = X_final
        
        return {
            'mean_pnl': np.mean(pnls),
            'std_pnl': np.std(pnls, ddof=1),
            'mean_inventory': np.mean(final_inventories),
            'std_inventory': np.std(final_inventories, ddof=1),
            'mean_cash': np.mean(final_cash),
            'pnls': pnls,
            'inventories': final_inventories,
            'cash': final_cash
        }
