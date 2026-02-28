"""Market making strategies implementation."""

import numpy as np
from typing import Tuple


class InventoryStrategy:
    """Inventory-aware market making strategy using reservation price."""
    
    def __init__(
        self,
        gamma: float,
        sigma: float,
        k: float,
        T: float
    ):
        """
        Initialize inventory-aware strategy.
        
        Args:
            gamma: Risk aversion parameter
            sigma: Mid-price volatility
            k: Exponential decay parameter for arrival intensity
            T: Terminal time
        """
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.T = T
        # Calculate optimal spread: (2/γ) * ln(1 + γ/k)
        self.spread = (2.0 / gamma) * np.log(1.0 + gamma / k)
    
    def get_quotes(self, S: float, q: int, t: float) -> Tuple[float, float]:
        """
        Generate bid and ask quotes centered at reservation price.
        
        Args:
            S: Current mid-price
            q: Current inventory
            t: Current time
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        # Calculate reservation price: r_t = S_t - q_t * γ * σ² * (T - t)
        time_remaining = self.T - t
        reservation_price = S - q * self.gamma * (self.sigma ** 2) * time_remaining
        
        # Center quotes at reservation price with optimal spread
        half_spread = self.spread / 2.0
        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread
        
        return bid_price, ask_price


class SymmetricStrategy:
    """Symmetric market making strategy with no inventory adjustment."""
    
    def __init__(
        self,
        gamma: float,
        k: float
    ):
        """
        Initialize symmetric strategy.
        
        Args:
            gamma: Risk aversion parameter (used for spread calculation)
            k: Exponential decay parameter for arrival intensity
        """
        self.gamma = gamma
        self.k = k
        # Use same spread formula as inventory strategy
        self.spread = (2.0 / gamma) * np.log(1.0 + gamma / k)
    
    def get_quotes(self, S: float, q: int, t: float) -> Tuple[float, float]:
        """
        Generate bid and ask quotes centered at mid-price.
        
        Args:
            S: Current mid-price
            q: Current inventory (not used in symmetric strategy)
            t: Current time (not used in symmetric strategy)
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        # Center quotes at mid-price with optimal spread
        half_spread = self.spread / 2.0
        bid_price = S - half_spread
        ask_price = S + half_spread
        
        return bid_price, ask_price


def create_inventory_strategy(gamma: float, sigma: float, k: float, T: float) -> callable:
    """
    Create a callable inventory strategy.
    
    Args:
        gamma: Risk aversion parameter
        sigma: Mid-price volatility
        k: Exponential decay parameter
        T: Terminal time
        
    Returns:
        Callable that takes (S, q, t) and returns (bid_price, ask_price)
    """
    strategy = InventoryStrategy(gamma, sigma, k, T)
    return strategy.get_quotes


def create_symmetric_strategy(gamma: float, k: float) -> callable:
    """
    Create a callable symmetric strategy.
    
    Args:
        gamma: Risk aversion parameter
        k: Exponential decay parameter
        
    Returns:
        Callable that takes (S, q, t) and returns (bid_price, ask_price)
    """
    strategy = SymmetricStrategy(gamma, k)
    return strategy.get_quotes
