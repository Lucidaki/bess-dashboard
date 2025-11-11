"""
Unit tests for BESS Optimizer
Tests MILP optimization logic and constraint implementation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.bess_optimizer import BESSOptimizer
from src.config_loader import get_config_loader


class TestBESSOptimizer:
    """Test suite for BESS Optimizer"""

    def setup_method(self):
        """Setup test fixtures"""
        config_loader = get_config_loader()
        configs = config_loader.load_all_configs()
        # Convert Pydantic models to dicts for BESSOptimizer
        self.asset_config_model = configs['config'].bess_assets['UK_BESS_001']
        self.market_config_model = configs['config'].market
        self.asset_config = self.asset_config_model.model_dump()
        self.market_config = self.market_config_model.model_dump()

    def test_optimizer_initialization(self):
        """Test optimizer initializes with correct parameters"""
        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        assert optimizer.capacity_mwh == self.asset_config['capacity_mwh']
        assert optimizer.power_mw == self.asset_config['power_mw']
        assert optimizer.rte == self.asset_config['rte_percent'] / 100.0
        assert optimizer.soc_min == self.asset_config['constraints']['soc_min_percent']
        assert optimizer.soc_max == self.asset_config['constraints']['soc_max_percent']

    def test_optimization_with_simple_arbitrage(self):
        """Test optimization with simple price arbitrage scenario"""
        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        # Create simple arbitrage scenario: low price at t=0, high price at t=1
        timestamps = [
            datetime(2025, 1, 1, 0, 0),
            datetime(2025, 1, 1, 0, 30),
            datetime(2025, 1, 1, 1, 0)
        ]

        schedule_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_gbp_mwh': [10.0, 50.0, 100.0],  # Clear arbitrage opportunity
            'actual_power_mw': [0.0, 0.0, 0.0],
            'actual_soc_percent': [50.0, 50.0, 50.0]
        })

        # Run optimization
        result = optimizer.optimize(schedule_df, initial_soc_percent=50.0)

        # Check results
        assert result['status'] == 'Optimal', "Optimization should converge"
        assert result['optimal_revenue_gbp'] > 0, "Should generate positive revenue"
        assert 'optimal_power_mw' in result['schedule'].columns
        assert 'optimal_soc_percent' in result['schedule'].columns

    def test_soc_constraints_respected(self):
        """Test that SoC stays within bounds"""
        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        # Create test scenario
        timestamps = pd.date_range('2025-01-01', periods=10, freq='30min')
        schedule_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_gbp_mwh': np.random.uniform(20, 80, 10),
            'actual_power_mw': np.zeros(10),
            'actual_soc_percent': np.full(10, 50.0)
        })

        result = optimizer.optimize(schedule_df, initial_soc_percent=50.0)
        optimal_soc = result['schedule']['optimal_soc_percent']

        # Check SoC bounds
        soc_min = self.asset_config['constraints']['soc_min_percent']
        soc_max = self.asset_config['constraints']['soc_max_percent']

        assert optimal_soc.min() >= soc_min - 0.01, f"SoC should be >= {soc_min}%"
        assert optimal_soc.max() <= soc_max + 0.01, f"SoC should be <= {soc_max}%"

    def test_power_constraints_respected(self):
        """Test that power stays within charge/discharge limits"""
        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        # Create test scenario with high price volatility
        timestamps = pd.date_range('2025-01-01', periods=10, freq='30min')
        schedule_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_gbp_mwh': [10, 100, 10, 100, 10, 100, 10, 100, 10, 100],  # Alternating
            'actual_power_mw': np.zeros(10),
            'actual_soc_percent': np.full(10, 50.0)
        })

        result = optimizer.optimize(schedule_df, initial_soc_percent=50.0)
        optimal_power = result['schedule']['optimal_power_mw']

        # Check power bounds (positive = discharge, negative = charge)
        max_discharge = self.asset_config['constraints']['max_discharge_power_mw']
        max_charge = self.asset_config['constraints']['max_charge_power_mw']

        assert optimal_power.max() <= max_discharge + 0.01, \
            f"Discharge power should be <= {max_discharge} MW"
        assert optimal_power.min() >= -max_charge - 0.01, \
            f"Charge power should be >= -{max_charge} MW"

    def test_cycle_limit_constraint(self):
        """Test that daily cycle limit is respected"""
        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        # Create 24-hour scenario with many arbitrage opportunities
        timestamps = pd.date_range('2025-01-01', periods=48, freq='30min')  # Full day
        prices = [10, 100] * 24  # Alternating high/low
        schedule_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_gbp_mwh': prices,
            'actual_power_mw': np.zeros(48),
            'actual_soc_percent': np.full(48, 50.0)
        })

        result = optimizer.optimize(schedule_df, initial_soc_percent=50.0)

        # Calculate cycles used
        cycles_used = result['cycles_used']
        max_cycles = self.asset_config['warranty']['max_daily_cycles']

        assert cycles_used <= max_cycles + 0.01, \
            f"Cycles used ({cycles_used:.2f}) should be <= {max_cycles}"

    def test_energy_balance_maintained(self):
        """Test that energy balance equation is maintained"""
        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        timestamps = pd.date_range('2025-01-01', periods=10, freq='30min')
        schedule_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_gbp_mwh': np.random.uniform(20, 80, 10),
            'actual_power_mw': np.zeros(10),
            'actual_soc_percent': np.full(10, 50.0)
        })

        initial_soc = 50.0
        result = optimizer.optimize(schedule_df, initial_soc_percent=initial_soc)

        schedule = result['schedule']
        dt_hours = self.market_config['settlement_duration_min'] / 60.0
        capacity = self.asset_config['capacity_mwh']
        rte = self.asset_config['rte_percent'] / 100.0

        # Check energy balance for each period
        soc_values = schedule['optimal_soc_percent'].values
        power_values = schedule['optimal_power_mw'].values

        for i in range(1, len(schedule)):
            expected_soc = soc_values[i-1] - (power_values[i-1] * dt_hours / capacity) * 100

            # Allow small tolerance for numerical errors
            assert abs(soc_values[i] - expected_soc) < 0.5, \
                f"Energy balance violated at period {i}: expected {expected_soc:.2f}%, got {soc_values[i]:.2f}%"

    def test_zero_price_scenario(self):
        """Test optimization with zero prices (should do nothing)"""
        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        timestamps = pd.date_range('2025-01-01', periods=10, freq='30min')
        schedule_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_gbp_mwh': np.zeros(10),  # All zero prices
            'actual_power_mw': np.zeros(10),
            'actual_soc_percent': np.full(10, 50.0)
        })

        result = optimizer.optimize(schedule_df, initial_soc_percent=50.0)

        assert result['status'] == 'Optimal', "Should still solve"
        assert result['optimal_revenue_gbp'] == 0, "Revenue should be zero"

    def test_uniform_price_scenario(self):
        """Test optimization with uniform prices (no arbitrage)"""
        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        timestamps = pd.date_range('2025-01-01', periods=10, freq='30min')
        schedule_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_gbp_mwh': np.full(10, 50.0),  # All same price
            'actual_power_mw': np.zeros(10),
            'actual_soc_percent': np.full(10, 50.0)
        })

        result = optimizer.optimize(schedule_df, initial_soc_percent=50.0)

        # With uniform prices, optimal action is to do nothing
        optimal_power = result['schedule']['optimal_power_mw']
        assert optimal_power.abs().sum() < 0.1, "Should not trade with uniform prices"

    def test_optimization_performance(self):
        """Test that optimization solves quickly"""
        import time

        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        # 48 periods (1 day)
        timestamps = pd.date_range('2025-01-01', periods=48, freq='30min')
        schedule_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_gbp_mwh': np.random.uniform(10, 100, 48),
            'actual_power_mw': np.zeros(48),
            'actual_soc_percent': np.full(48, 50.0)
        })

        start_time = time.time()
        result = optimizer.optimize(schedule_df, initial_soc_percent=50.0)
        solve_time = time.time() - start_time

        assert solve_time < 5.0, f"Optimization should complete in <5 seconds (took {solve_time:.2f}s)"
        assert result['status'] == 'Optimal', "Should converge to optimal solution"


class TestOptimizationResults:
    """Test optimization result structure and completeness"""

    def setup_method(self):
        """Setup test fixtures"""
        config_loader = get_config_loader()
        configs = config_loader.load_all_configs()
        # Convert Pydantic models to dicts for BESSOptimizer
        self.asset_config = configs['config'].bess_assets['UK_BESS_001'].model_dump()
        self.market_config = configs['config'].market.model_dump()

    def test_result_dictionary_structure(self):
        """Test that optimization returns all required fields"""
        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        timestamps = pd.date_range('2025-01-01', periods=10, freq='30min')
        schedule_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_gbp_mwh': np.random.uniform(20, 80, 10),
            'actual_power_mw': np.zeros(10),
            'actual_soc_percent': np.full(10, 50.0)
        })

        result = optimizer.optimize(schedule_df, initial_soc_percent=50.0)

        # Check required fields
        required_fields = [
            'status', 'optimal_revenue_gbp', 'schedule', 'cycles_used',
            'discharge_energy_mwh', 'charge_energy_mwh', 'solve_time_seconds'
        ]

        for field in required_fields:
            assert field in result, f"Result should contain '{field}'"

    def test_schedule_dataframe_columns(self):
        """Test that schedule has all required columns"""
        optimizer = BESSOptimizer(self.asset_config, self.market_config)

        timestamps = pd.date_range('2025-01-01', periods=10, freq='30min')
        schedule_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_gbp_mwh': np.random.uniform(20, 80, 10),
            'actual_power_mw': np.zeros(10),
            'actual_soc_percent': np.full(10, 50.0)
        })

        result = optimizer.optimize(schedule_df, initial_soc_percent=50.0)
        schedule = result['schedule']

        # Check required columns
        required_cols = ['optimal_power_mw', 'optimal_soc_percent']
        for col in required_cols:
            assert col in schedule.columns, f"Schedule should contain '{col}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
