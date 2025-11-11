"""
O&M (Operations & Maintenance) KPI Calculator
Calculates operational key performance indicators for BESS operations
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta


class OMKPICalculator:
    """
    Calculate O&M-focused KPIs for BESS operations
    """

    def __init__(self, asset_config: Dict):
        """
        Initialize O&M KPI calculator

        Args:
            asset_config: BESS asset configuration
        """
        self.asset_config = asset_config
        self.capacity_mwh = asset_config['capacity_mwh']
        self.power_mw = asset_config['power_mw']
        self.rated_rte = asset_config['rte_percent']
        self.max_daily_cycles = asset_config['warranty']['max_daily_cycles']

    def calculate_kpis(
        self,
        optimization_summary: Dict,
        schedule_df: Optional[pd.DataFrame] = None,
        settlement_duration_min: int = 30
    ) -> Dict:
        """
        Calculate all O&M KPIs from optimization results

        Args:
            optimization_summary: Output from optimize_bess.py
            schedule_df: Optional schedule dataframe for detailed analysis
            settlement_duration_min: Settlement period duration

        Returns:
            Dictionary with O&M KPIs
        """
        kpis = {}

        # Extract data from optimization summary
        actual_perf = optimization_summary['actual_performance']
        duration_days = optimization_summary['duration_days']
        n_periods = actual_perf['n_periods']

        # 1. Availability (%)
        # Percentage of time the BESS was operational
        # For now, assume 100% if data exists (can be refined with fault detection)
        kpis['availability_percent'] = 100.0  # Default - can be enhanced with fault data

        # 2. Cycle Utilization (%)
        # Actual cycles used vs maximum allowed cycles
        actual_cycles = actual_perf['cycles_used']
        max_allowed_cycles = self.max_daily_cycles * duration_days
        cycle_utilization = (actual_cycles / max_allowed_cycles * 100) if max_allowed_cycles > 0 else 0
        kpis['cycle_utilization_percent'] = cycle_utilization
        kpis['actual_cycles'] = actual_cycles
        kpis['max_allowed_cycles'] = max_allowed_cycles

        # 3. Actual RTE (%)
        # Round-trip efficiency from actual operation
        kpis['actual_rte_percent'] = actual_perf['actual_rte_percent']
        kpis['rated_rte_percent'] = self.rated_rte
        kpis['rte_deviation_percent'] = actual_perf['actual_rte_percent'] - self.rated_rte

        # 4. Throughput (MWh)
        # Total energy discharged
        kpis['discharge_throughput_mwh'] = actual_perf['discharge_energy_mwh']
        kpis['charge_throughput_mwh'] = actual_perf['charge_energy_mwh']

        # 5. Capacity Factor (%)
        # Actual discharge energy vs theoretical maximum
        dt_hours = settlement_duration_min / 60.0
        theoretical_max_energy = self.power_mw * dt_hours * n_periods
        capacity_factor = (actual_perf['discharge_energy_mwh'] / theoretical_max_energy * 100) if theoretical_max_energy > 0 else 0
        kpis['capacity_factor_percent'] = capacity_factor

        # 6. SoC Operating Range
        # How much of the available SoC range was used
        soc_range_used = actual_perf['soc_range_percent']
        constraints = self.asset_config['constraints']
        soc_range_available = constraints['soc_max_percent'] - constraints['soc_min_percent']
        soc_range_utilization = (soc_range_used / soc_range_available * 100) if soc_range_available > 0 else 0
        kpis['soc_range_utilization_percent'] = soc_range_utilization
        kpis['soc_range_used_percent'] = soc_range_used

        # 7. Power Utilization
        # Peak power vs rated power
        power_max_abs = max(abs(actual_perf['power_min_mw']), abs(actual_perf['power_max_mw']))
        power_utilization = (power_max_abs / self.power_mw * 100) if self.power_mw > 0 else 0
        kpis['power_utilization_percent'] = power_utilization

        # 8. Idle Time (%)
        # Percentage of time with near-zero power
        if schedule_df is not None:
            idle_threshold_mw = 0.1  # Below 0.1 MW considered idle
            idle_periods = (schedule_df['actual_power_mw'].abs() < idle_threshold_mw).sum()
            idle_time_percent = (idle_periods / len(schedule_df) * 100) if len(schedule_df) > 0 else 0
            kpis['idle_time_percent'] = idle_time_percent
        else:
            kpis['idle_time_percent'] = None

        # 9. Average Cycle Depth
        # Average depth of discharge per cycle
        if actual_cycles > 0:
            avg_cycle_depth_mwh = actual_perf['discharge_energy_mwh'] / actual_cycles
            avg_cycle_depth_percent = (avg_cycle_depth_mwh / self.capacity_mwh * 100)
            kpis['avg_cycle_depth_percent'] = avg_cycle_depth_percent
        else:
            kpis['avg_cycle_depth_percent'] = 0

        # 10. Degradation Estimate
        # Simple linear degradation model based on cycles
        annual_fade_rate = self.asset_config['degradation']['annual_fade_percent']
        cycles_per_year = actual_cycles / duration_days * 365
        estimated_annual_degradation = (cycles_per_year / 365) * annual_fade_rate
        kpis['estimated_annual_degradation_percent'] = estimated_annual_degradation

        # 11. O&M Grade (A-F rating based on multiple factors)
        kpis['om_grade'] = self._calculate_grade(kpis)

        return kpis

    def _calculate_grade(self, kpis: Dict) -> str:
        """
        Calculate O&M grade based on multiple operational factors

        Args:
            kpis: Dictionary of O&M KPIs

        Returns:
            Grade (A-F)
        """
        # Weighted scoring system
        scores = []

        # Availability (weight: 30%)
        availability = kpis.get('availability_percent', 0)
        if availability >= 99:
            scores.append(100 * 0.3)
        elif availability >= 95:
            scores.append(90 * 0.3)
        elif availability >= 90:
            scores.append(80 * 0.3)
        else:
            scores.append(70 * 0.3)

        # Cycle Utilization (weight: 25%)
        cycle_util = kpis.get('cycle_utilization_percent', 0)
        if cycle_util >= 80:
            scores.append(100 * 0.25)
        elif cycle_util >= 60:
            scores.append(90 * 0.25)
        elif cycle_util >= 40:
            scores.append(80 * 0.25)
        else:
            scores.append(70 * 0.25)

        # RTE Performance (weight: 25%)
        rte_deviation = abs(kpis.get('rte_deviation_percent', 0))
        if rte_deviation <= 2:
            scores.append(100 * 0.25)
        elif rte_deviation <= 5:
            scores.append(90 * 0.25)
        elif rte_deviation <= 10:
            scores.append(80 * 0.25)
        else:
            scores.append(70 * 0.25)

        # Capacity Factor (weight: 20%)
        capacity_factor = kpis.get('capacity_factor_percent', 0)
        if capacity_factor >= 30:
            scores.append(100 * 0.2)
        elif capacity_factor >= 20:
            scores.append(90 * 0.2)
        elif capacity_factor >= 10:
            scores.append(80 * 0.2)
        else:
            scores.append(70 * 0.2)

        total_score = sum(scores)

        # Convert to letter grade
        if total_score >= 95:
            return "A"
        elif total_score >= 85:
            return "B"
        elif total_score >= 75:
            return "C"
        elif total_score >= 65:
            return "D"
        elif total_score >= 55:
            return "E"
        else:
            return "F"

    def generate_om_report(
        self,
        kpis: Dict,
        asset_name: str
    ) -> str:
        """
        Generate human-readable O&M report

        Args:
            kpis: Dictionary of O&M KPIs
            asset_name: Asset name

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("O&M KPI REPORT")
        report.append("=" * 80)
        report.append(f"Asset: {asset_name}")
        report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Grade: {kpis.get('om_grade', 'N/A')}")
        report.append("")

        # Availability Section
        report.append("AVAILABILITY & RELIABILITY")
        report.append("-" * 80)
        report.append(f"  Availability:          {kpis.get('availability_percent', 0):>12.1f}%")
        report.append("")

        # Cycle Utilization Section
        report.append("CYCLE UTILIZATION")
        report.append("-" * 80)
        report.append(f"  Actual Cycles:         {kpis.get('actual_cycles', 0):>12.2f}")
        report.append(f"  Max Allowed Cycles:    {kpis.get('max_allowed_cycles', 0):>12.2f}")
        report.append(f"  Cycle Utilization:     {kpis.get('cycle_utilization_percent', 0):>12.1f}%")
        report.append(f"  Avg Cycle Depth:       {kpis.get('avg_cycle_depth_percent', 0):>12.1f}%")
        report.append("")

        # Efficiency Section
        report.append("EFFICIENCY PERFORMANCE")
        report.append("-" * 80)
        report.append(f"  Actual RTE:            {kpis.get('actual_rte_percent', 0):>12.1f}%")
        report.append(f"  Rated RTE:             {kpis.get('rated_rte_percent', 0):>12.1f}%")
        report.append(f"  RTE Deviation:         {kpis.get('rte_deviation_percent', 0):>12.1f}%")
        report.append("")

        # Throughput Section
        report.append("ENERGY THROUGHPUT")
        report.append("-" * 80)
        report.append(f"  Discharge Energy:      {kpis.get('discharge_throughput_mwh', 0):>12.2f} MWh")
        report.append(f"  Charge Energy:         {kpis.get('charge_throughput_mwh', 0):>12.2f} MWh")
        report.append(f"  Capacity Factor:       {kpis.get('capacity_factor_percent', 0):>12.1f}%")
        report.append("")

        # Utilization Section
        report.append("ASSET UTILIZATION")
        report.append("-" * 80)
        report.append(f"  SoC Range Used:        {kpis.get('soc_range_used_percent', 0):>12.1f}%")
        report.append(f"  SoC Range Utilization: {kpis.get('soc_range_utilization_percent', 0):>12.1f}%")
        report.append(f"  Power Utilization:     {kpis.get('power_utilization_percent', 0):>12.1f}%")
        if kpis.get('idle_time_percent') is not None:
            report.append(f"  Idle Time:             {kpis.get('idle_time_percent', 0):>12.1f}%")
        report.append("")

        # Degradation Section
        report.append("DEGRADATION TRACKING")
        report.append("-" * 80)
        report.append(f"  Est. Annual Degradation: {kpis.get('estimated_annual_degradation_percent', 0):>10.2f}%")
        report.append("")

        report.append("=" * 80)
        return "\n".join(report)
