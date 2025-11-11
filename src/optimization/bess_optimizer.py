"""
BESS Optimization Module
MILP-based arbitrage optimization using PuLP
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import pulp


class BESSOptimizer:
    """
    Mixed Integer Linear Programming optimizer for BESS arbitrage
    """

    def __init__(
        self,
        asset_config: Dict,
        settlement_duration_min: int = 30,
        solver_name: str = "PULP_CBC_CMD",
        solver_timeout_sec: int = 30
    ):
        """
        Initialize BESS optimizer

        Args:
            asset_config: BESS asset configuration (capacity, power, RTE, constraints)
            settlement_duration_min: Settlement period duration in minutes
            solver_name: PuLP solver name (default: PULP_CBC_CMD)
            solver_timeout_sec: Maximum solver time in seconds
        """
        self.asset_config = asset_config
        self.settlement_duration_min = settlement_duration_min
        self.solver_name = solver_name
        self.solver_timeout_sec = solver_timeout_sec

        # Extract key parameters
        self.capacity_mwh = asset_config['capacity_mwh']
        self.power_mw = asset_config['power_mw']
        self.rte_percent = asset_config['rte_percent']
        self.rte = self.rte_percent / 100.0  # Convert to decimal

        # Constraints
        constraints = asset_config['constraints']
        self.soc_min = constraints['soc_min_percent'] / 100.0
        self.soc_max = constraints['soc_max_percent'] / 100.0
        self.power_export_max = constraints['power_export_max_mw']  # Discharge (positive)
        self.power_import_max = constraints['power_import_max_mw']  # Charge (negative)

        # Warranty constraints
        warranty = asset_config['warranty']
        self.max_daily_cycles = warranty['max_daily_cycles']

        # Time step in hours
        self.dt_hours = settlement_duration_min / 60.0

    def optimize(
        self,
        scada_df: pd.DataFrame,
        market_df: pd.DataFrame,
        initial_soc_percent: Optional[float] = None
    ) -> Dict:
        """
        Perform MILP optimization for BESS arbitrage

        Args:
            scada_df: SCADA data with timestamp_utc, power_mw, soc_percent
            market_df: Market price data with timestamp_utc, price_gbp_mwh
            initial_soc_percent: Initial SoC (if None, use first value from SCADA)

        Returns:
            Dictionary with optimization results
        """
        # Merge SCADA and market data on timestamp
        df = pd.merge(scada_df, market_df, on='timestamp_utc', how='inner')

        if len(df) == 0:
            raise ValueError("No common timestamps between SCADA and market data")

        # Extract data
        timestamps = df['timestamp_utc'].tolist()
        prices = df['price_gbp_mwh'].values
        actual_power = df['power_mw'].values
        actual_soc = df['soc_percent'].values

        n_periods = len(timestamps)

        # Initial SoC
        if initial_soc_percent is None:
            initial_soc_percent = actual_soc[0]

        initial_soc = initial_soc_percent / 100.0  # Convert to decimal

        # Create optimization problem
        prob = pulp.LpProblem("BESS_Arbitrage", pulp.LpMaximize)

        # Decision variables
        # Power: positive = discharge, negative = charge
        power = pulp.LpVariable.dicts(
            "power",
            range(n_periods),
            lowBound=-self.power_import_max,
            upBound=self.power_export_max,
            cat='Continuous'
        )

        # SoC at end of each period
        soc = pulp.LpVariable.dicts(
            "soc",
            range(n_periods),
            lowBound=self.soc_min,
            upBound=self.soc_max,
            cat='Continuous'
        )

        # Binary variables for charge/discharge direction
        is_charging = pulp.LpVariable.dicts(
            "is_charging",
            range(n_periods),
            cat='Binary'
        )

        is_discharging = pulp.LpVariable.dicts(
            "is_discharging",
            range(n_periods),
            cat='Binary'
        )

        # Auxiliary variables for charge and discharge power (for RTE calculation)
        charge_power = pulp.LpVariable.dicts(
            "charge_power",
            range(n_periods),
            lowBound=0,
            upBound=self.power_import_max,
            cat='Continuous'
        )

        discharge_power = pulp.LpVariable.dicts(
            "discharge_power",
            range(n_periods),
            lowBound=0,
            upBound=self.power_export_max,
            cat='Continuous'
        )

        # Objective: Maximize revenue
        # Revenue = discharge_energy * price - charge_energy * price
        # Note: charge has cost (buying), discharge has revenue (selling)
        prob += pulp.lpSum([
            power[t] * prices[t] * self.dt_hours
            for t in range(n_periods)
        ]), "Total_Revenue"

        # Constraint 1: Initial SoC
        prob += (
            soc[0] == initial_soc - (power[0] * self.dt_hours / self.capacity_mwh),
            "Initial_SoC"
        )

        # Constraint 2: Energy balance with RTE
        for t in range(1, n_periods):
            # SoC change = -(discharge_power - charge_power * RTE) * dt / capacity
            # Discharge decreases SoC, charge increases SoC (with RTE efficiency)
            prob += (
                soc[t] == soc[t-1] - (discharge_power[t] * self.dt_hours / self.capacity_mwh)
                + (charge_power[t] * self.rte * self.dt_hours / self.capacity_mwh),
                f"Energy_Balance_{t}"
            )

        # Constraint 3: Link power to charge/discharge power
        for t in range(n_periods):
            # power = discharge_power - charge_power
            prob += (
                power[t] == discharge_power[t] - charge_power[t],
                f"Power_Split_{t}"
            )

        # Constraint 4: Charge/discharge mutual exclusivity
        for t in range(n_periods):
            prob += (
                is_charging[t] + is_discharging[t] <= 1,
                f"Mutually_Exclusive_{t}"
            )

        # Constraint 5: Link binary variables to power
        M = max(self.power_export_max, self.power_import_max) * 1.1  # Big M

        for t in range(n_periods):
            # If discharging, discharge_power > 0
            prob += (
                discharge_power[t] <= M * is_discharging[t],
                f"Discharge_Indicator_{t}"
            )

            # If charging, charge_power > 0
            prob += (
                charge_power[t] <= M * is_charging[t],
                f"Charge_Indicator_{t}"
            )

        # Constraint 6: Daily cycle limit
        # One cycle = full charge + full discharge = 2 * capacity
        # Total throughput = sum of discharge energy
        total_discharge_energy = pulp.lpSum([
            discharge_power[t] * self.dt_hours
            for t in range(n_periods)
        ])

        # Calculate days in optimization period
        duration_hours = n_periods * self.dt_hours
        duration_days = duration_hours / 24.0

        max_throughput = self.max_daily_cycles * self.capacity_mwh * duration_days

        prob += (
            total_discharge_energy <= max_throughput,
            "Daily_Cycle_Limit"
        )

        # Solve
        start_time = datetime.now()

        if self.solver_name == "PULP_CBC_CMD":
            solver = pulp.PULP_CBC_CMD(
                msg=0,  # Suppress solver output
                timeLimit=self.solver_timeout_sec
            )
        else:
            solver = pulp.getSolver(self.solver_name, msg=0)

        prob.solve(solver)

        solve_time = (datetime.now() - start_time).total_seconds()

        # Extract results
        solver_status = pulp.LpStatus[prob.status]

        if prob.status != pulp.LpStatusOptimal:
            raise RuntimeError(f"Solver did not find optimal solution. Status: {solver_status}")

        # Extract optimal schedules
        optimal_power = [power[t].varValue for t in range(n_periods)]
        optimal_soc = [soc[t].varValue * 100 for t in range(n_periods)]  # Convert to percentage

        # Calculate optimal revenue
        optimal_revenue = sum([
            optimal_power[t] * prices[t] * self.dt_hours
            for t in range(n_periods)
        ])

        # Calculate actual revenue
        actual_revenue = sum([
            actual_power[t] * prices[t] * self.dt_hours
            for t in range(n_periods)
        ])

        # Calculate metrics
        revenue_variance = optimal_revenue - actual_revenue
        market_capture_ratio = (actual_revenue / optimal_revenue * 100) if optimal_revenue > 0 else 0

        # Calculate MIP gap
        mip_gap = 0.0  # CBC solver doesn't always provide gap

        # Calculate cycle usage
        optimal_discharge_energy = sum([
            max(0, optimal_power[t]) * self.dt_hours
            for t in range(n_periods)
        ])
        cycles_used = optimal_discharge_energy / self.capacity_mwh

        return {
            'solver_status': solver_status,
            'solve_time_sec': solve_time,
            'objective_value': optimal_revenue,
            'mip_gap': mip_gap,
            'optimal_power_mw': optimal_power,
            'optimal_soc_percent': optimal_soc,
            'actual_power_mw': actual_power.tolist(),
            'actual_soc_percent': actual_soc.tolist(),
            'timestamps': timestamps,
            'prices_gbp_mwh': prices.tolist(),
            'optimal_revenue_gbp': optimal_revenue,
            'actual_revenue_gbp': actual_revenue,
            'revenue_variance_gbp': revenue_variance,
            'market_capture_ratio': market_capture_ratio,
            'cycles_used': cycles_used,
            'max_daily_cycles': self.max_daily_cycles,
            'duration_days': duration_days
        }

    def calculate_actual_performance(
        self,
        scada_df: pd.DataFrame,
        market_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate actual BESS performance metrics from SCADA data

        Args:
            scada_df: SCADA data
            market_df: Market price data

        Returns:
            Dictionary with actual performance metrics
        """
        # Merge data
        df = pd.merge(scada_df, market_df, on='timestamp_utc', how='inner')

        if len(df) == 0:
            raise ValueError("No common timestamps between SCADA and market data")

        # Extract data
        actual_power = df['power_mw'].values
        prices = df['price_gbp_mwh'].values
        actual_soc = df['soc_percent'].values

        # Calculate actual revenue
        actual_revenue = sum([
            actual_power[t] * prices[t] * self.dt_hours
            for t in range(len(actual_power))
        ])

        # Calculate energy throughput
        discharge_energy = sum([
            max(0, actual_power[t]) * self.dt_hours
            for t in range(len(actual_power))
        ])

        charge_energy = sum([
            abs(min(0, actual_power[t])) * self.dt_hours
            for t in range(len(actual_power))
        ])

        # Calculate cycles
        cycles_used = discharge_energy / self.capacity_mwh

        # Calculate RTE from actual data
        actual_rte = (discharge_energy / charge_energy * 100) if charge_energy > 0 else 0

        # SoC statistics
        soc_min = actual_soc.min()
        soc_max = actual_soc.max()
        soc_mean = actual_soc.mean()
        soc_range = soc_max - soc_min

        # Power statistics
        power_min = actual_power.min()
        power_max = actual_power.max()
        power_mean = actual_power.mean()

        return {
            'actual_revenue_gbp': actual_revenue,
            'discharge_energy_mwh': discharge_energy,
            'charge_energy_mwh': charge_energy,
            'cycles_used': cycles_used,
            'actual_rte_percent': actual_rte,
            'soc_min_percent': soc_min,
            'soc_max_percent': soc_max,
            'soc_mean_percent': soc_mean,
            'soc_range_percent': soc_range,
            'power_min_mw': power_min,
            'power_max_mw': power_max,
            'power_mean_mw': power_mean,
            'n_periods': len(actual_power)
        }
