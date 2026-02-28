"""Tests for market making strategies."""

import pytest
import numpy as np
from src.strategies import (
    InventoryStrategy,
    SymmetricStrategy,
    create_inventory_strategy,
    create_symmetric_strategy
)


class TestInventoryStrategy:
    """Tests for InventoryStrategy class."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = InventoryStrategy(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
        assert strategy.gamma == 0.1
        assert strategy.sigma == 2.0
        assert strategy.k == 1.5
        assert strategy.T == 1.0
        
        # Check spread calculation
        expected_spread = (2.0 / 0.1) * np.log(1.0 + 0.1 / 1.5)
        assert np.isclose(strategy.spread, expected_spread)
    
    def test_spread_calculation(self):
        """Test spread calculation for different gamma values."""
        gamma_values = [0.01, 0.1, 0.5]
        k = 1.5
        
        for gamma in gamma_values:
            strategy = InventoryStrategy(gamma=gamma, sigma=2.0, k=k, T=1.0)
            expected = (2.0 / gamma) * np.log(1.0 + gamma / k)
            assert np.isclose(strategy.spread, expected)
    
    def test_get_quotes_zero_inventory(self):
        """Test quotes with zero inventory."""
        strategy = InventoryStrategy(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
        S = 100.0
        q = 0
        t = 0.5
        
        bid, ask = strategy.get_quotes(S, q, t)
        
        # With zero inventory, quotes should be centered at mid-price
        assert np.isclose(bid + ask, 2 * S)
        assert np.isclose(ask - bid, strategy.spread)
    
    def test_get_quotes_positive_inventory(self):
        """Test quotes with positive inventory."""
        strategy = InventoryStrategy(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
        S = 100.0
        q = 5
        t = 0.5
        
        bid, ask = strategy.get_quotes(S, q, t)
        
        # With positive inventory, reservation price should be below mid-price
        # So quotes should be shifted down
        reservation_price = S - q * strategy.gamma * (strategy.sigma ** 2) * (strategy.T - t)
        assert reservation_price < S
        
        # Quotes should be centered at reservation price
        assert np.isclose((bid + ask) / 2, reservation_price)
        assert np.isclose(ask - bid, strategy.spread)
    
    def test_get_quotes_negative_inventory(self):
        """Test quotes with negative inventory."""
        strategy = InventoryStrategy(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
        S = 100.0
        q = -5
        t = 0.5
        
        bid, ask = strategy.get_quotes(S, q, t)
        
        # With negative inventory, reservation price should be above mid-price
        # So quotes should be shifted up
        reservation_price = S - q * strategy.gamma * (strategy.sigma ** 2) * (strategy.T - t)
        assert reservation_price > S
        
        # Quotes should be centered at reservation price
        assert np.isclose((bid + ask) / 2, reservation_price)
        assert np.isclose(ask - bid, strategy.spread)
    
    def test_get_quotes_time_dependency(self):
        """Test that quotes adjust as time approaches terminal."""
        strategy = InventoryStrategy(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
        S = 100.0
        q = 5
        
        # Early in the period
        bid_early, ask_early = strategy.get_quotes(S, q, t=0.1)
        
        # Near terminal time
        bid_late, ask_late = strategy.get_quotes(S, q, t=0.9)
        
        # Inventory adjustment should be stronger early (more time remaining)
        reservation_early = (bid_early + ask_early) / 2
        reservation_late = (bid_late + ask_late) / 2
        
        # With positive inventory, early reservation should be further below mid
        assert reservation_early < reservation_late


class TestSymmetricStrategy:
    """Tests for SymmetricStrategy class."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = SymmetricStrategy(gamma=0.1, k=1.5)
        assert strategy.gamma == 0.1
        assert strategy.k == 1.5
        
        # Check spread calculation
        expected_spread = (2.0 / 0.1) * np.log(1.0 + 0.1 / 1.5)
        assert np.isclose(strategy.spread, expected_spread)
    
    def test_get_quotes_centered_at_mid(self):
        """Test that quotes are always centered at mid-price."""
        strategy = SymmetricStrategy(gamma=0.1, k=1.5)
        
        # Test with different inventory levels and times
        test_cases = [
            (100.0, 0, 0.0),
            (100.0, 5, 0.5),
            (100.0, -5, 0.5),
            (105.0, 10, 0.9),
            (95.0, -10, 0.1),
        ]
        
        for S, q, t in test_cases:
            bid, ask = strategy.get_quotes(S, q, t)
            
            # Should always be centered at mid-price
            assert np.isclose((bid + ask) / 2, S)
            assert np.isclose(ask - bid, strategy.spread)
    
    def test_inventory_independence(self):
        """Test that quotes don't depend on inventory."""
        strategy = SymmetricStrategy(gamma=0.1, k=1.5)
        S = 100.0
        t = 0.5
        
        bid_q0, ask_q0 = strategy.get_quotes(S, 0, t)
        bid_q5, ask_q5 = strategy.get_quotes(S, 5, t)
        bid_qneg, ask_qneg = strategy.get_quotes(S, -5, t)
        
        # All should be identical
        assert bid_q0 == bid_q5 == bid_qneg
        assert ask_q0 == ask_q5 == ask_qneg
    
    def test_time_independence(self):
        """Test that quotes don't depend on time."""
        strategy = SymmetricStrategy(gamma=0.1, k=1.5)
        S = 100.0
        q = 0
        
        bid_t0, ask_t0 = strategy.get_quotes(S, q, 0.0)
        bid_t05, ask_t05 = strategy.get_quotes(S, q, 0.5)
        bid_t1, ask_t1 = strategy.get_quotes(S, q, 1.0)
        
        # All should be identical
        assert bid_t0 == bid_t05 == bid_t1
        assert ask_t0 == ask_t05 == ask_t1


class TestStrategyCreators:
    """Tests for strategy creator functions."""
    
    def test_create_inventory_strategy(self):
        """Test create_inventory_strategy function."""
        strategy_func = create_inventory_strategy(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
        
        # Should be callable
        assert callable(strategy_func)
        
        # Should return bid and ask prices
        bid, ask = strategy_func(100.0, 0, 0.5)
        assert isinstance(bid, (int, float))
        assert isinstance(ask, (int, float))
        assert bid < ask
    
    def test_create_symmetric_strategy(self):
        """Test create_symmetric_strategy function."""
        strategy_func = create_symmetric_strategy(gamma=0.1, k=1.5)
        
        # Should be callable
        assert callable(strategy_func)
        
        # Should return bid and ask prices
        bid, ask = strategy_func(100.0, 0, 0.5)
        assert isinstance(bid, (int, float))
        assert isinstance(ask, (int, float))
        assert bid < ask
    
    def test_strategies_have_same_spread(self):
        """Test that both strategies use the same spread formula."""
        gamma = 0.1
        k = 1.5
        
        inv_strategy = create_inventory_strategy(gamma=gamma, sigma=2.0, k=k, T=1.0)
        sym_strategy = create_symmetric_strategy(gamma=gamma, k=k)
        
        S = 100.0
        q = 0
        t = 0.5
        
        bid_inv, ask_inv = inv_strategy(S, q, t)
        bid_sym, ask_sym = sym_strategy(S, q, t)
        
        spread_inv = ask_inv - bid_inv
        spread_sym = ask_sym - bid_sym
        
        # Both should have same spread
        assert np.isclose(spread_inv, spread_sym)
