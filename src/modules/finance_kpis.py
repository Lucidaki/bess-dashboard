"""
Finance KPI Calculator
Calculates finance-focused key performance indicators for BESS operations
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime


class FinanceKPICalculator:
    """
    Calculate finance-focused KPIs for BESS operations
    """

    def __init__(self, asset_config: Dict):
        """
        Initialize Finance KPI calculator

        Args:
            asset_config: BESS asset configuration
        """
        self.asset_config = asset_config
        self.capacity_mwh = asset_config['capacity_mwh']

    def calculate_kpis(
        self,
        optimization_summary: Dict,
        settlement_duration_min: int = 30
    ) -> Dict:
        """
        Calculate all finance KPIs from optimization results

        Args:
            optimization_summary: Output from optimize_bess.py
            settlement_duration_min: Settlement period duration

        Returns:
            Dictionary with finance KPIs
        """
        kpis = {}

        # Extract data from optimization summary
        actual_revenue = optimization_summary['actual_revenue_gbp']
        optimal_revenue = optimization_summary['optimal_revenue_gbp']
        revenue_variance = optimization_summary['revenue_variance_gbp']
        actual_perf = optimization_summary['actual_performance']

        # 1. Market Capture Ratio (%)
        # Measures how much of the optimal revenue was actually captured
        market_capture_ratio = (actual_revenue / optimal_revenue * 100) if optimal_revenue != 0 else 0
        kpis['market_capture_ratio'] = market_capture_ratio

        # 2. Revenue Variance (absolute and percentage)
        kpis['revenue_variance_gbp'] = revenue_variance
        kpis['revenue_variance_percent'] = (revenue_variance / abs(optimal_revenue) * 100) if optimal_revenue != 0 else 0

        # 3. Arbitrage Revenue (actual)
        kpis['arbitrage_revenue_gbp'] = actual_revenue
        kpis['optimal_revenue_gbp'] = optimal_revenue

        # 4. Arbitrage Efficiency (%)
        # Measures how efficiently the BESS captured price spreads
        # Based on actual cycles and revenue per cycle
        actual_cycles = actual_perf['cycles_used']
        if actual_cycles > 0:
            revenue_per_cycle = actual_revenue / actual_cycles
            kpis['revenue_per_cycle_gbp'] = revenue_per_cycle
        else:
            kpis['revenue_per_cycle_gbp'] = 0

        # 5. Lost Opportunity Cost
        # Revenue that could have been earned but was missed
        kpis['lost_opportunity_gbp'] = max(0, revenue_variance)

        # 6. Revenue per MWh Throughput
        discharge_energy = actual_perf['discharge_energy_mwh']
        if discharge_energy > 0:
            revenue_per_mwh = actual_revenue / discharge_energy
            kpis['revenue_per_discharge_mwh'] = revenue_per_mwh
        else:
            kpis['revenue_per_discharge_mwh'] = 0

        # 7. Average Price Capture
        # Average price achieved for discharge vs market average
        # This would require schedule data, so mark as unavailable
        kpis['average_discharge_price_gbp_mwh'] = None  # Requires schedule data

        # 8. IRR Impact Estimate
        # Simple estimate: lost opportunity as % of capacity value
        # Assume CAPEX = £300k/MWh (typical BESS cost)
        capex_per_mwh = 300000  # £300k/MWh
        total_capex = capex_per_mwh * self.capacity_mwh

        # Annualized lost opportunity (assuming 2 days = sample of 365 days)
        duration_days = optimization_summary['duration_days']
        if duration_days > 0:
            annualized_lost_opportunity = revenue_variance * (365 / duration_days)
            irr_impact_basis_points = (annualized_lost_opportunity / total_capex) * 10000  # Convert to basis points
            kpis['irr_impact_estimate_bps'] = irr_impact_basis_points
        else:
            kpis['irr_impact_estimate_bps'] = 0

        # 9. Capacity-Adjusted Revenue
        # Revenue per MWh of installed capacity
        kpis['revenue_per_capacity_mwh'] = actual_revenue / self.capacity_mwh

        # 10. Finance Grade (A-F rating based on market capture)
        kpis['finance_grade'] = self._calculate_grade(market_capture_ratio)

        return kpis

    def _calculate_grade(self, market_capture_ratio: float) -> str:
        """
        Calculate finance grade based on market capture ratio

        Args:
            market_capture_ratio: Market capture ratio (%)

        Returns:
            Grade (A-F)
        """
        if market_capture_ratio >= 95:
            return "A"
        elif market_capture_ratio >= 85:
            return "B"
        elif market_capture_ratio >= 75:
            return "C"
        elif market_capture_ratio >= 60:
            return "D"
        elif market_capture_ratio >= 40:
            return "E"
        else:
            return "F"

    def calculate_schedule_based_kpis(
        self,
        schedule_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate KPIs that require schedule data

        Args:
            schedule_df: Schedule dataframe with power, SoC, and prices

        Returns:
            Dictionary with schedule-based KPIs
        """
        kpis = {}

        # Average discharge price (weighted by discharge energy)
        discharge_periods = schedule_df[schedule_df['actual_power_mw'] > 0].copy()
        if len(discharge_periods) > 0:
            discharge_periods['discharge_energy'] = discharge_periods['actual_power_mw'] * 0.5  # 30 min = 0.5 hr
            weighted_price = (discharge_periods['price_gbp_mwh'] * discharge_periods['discharge_energy']).sum()
            total_discharge = discharge_periods['discharge_energy'].sum()
            avg_discharge_price = weighted_price / total_discharge if total_discharge > 0 else 0
            kpis['average_discharge_price_gbp_mwh'] = avg_discharge_price
        else:
            kpis['average_discharge_price_gbp_mwh'] = 0

        # Average charge price (weighted by charge energy)
        charge_periods = schedule_df[schedule_df['actual_power_mw'] < 0].copy()
        if len(charge_periods) > 0:
            charge_periods['charge_energy'] = abs(charge_periods['actual_power_mw']) * 0.5
            weighted_price = (charge_periods['price_gbp_mwh'] * charge_periods['charge_energy']).sum()
            total_charge = charge_periods['charge_energy'].sum()
            avg_charge_price = weighted_price / total_charge if total_charge > 0 else 0
            kpis['average_charge_price_gbp_mwh'] = avg_charge_price
        else:
            kpis['average_charge_price_gbp_mwh'] = 0

        # Price spread captured
        if kpis['average_discharge_price_gbp_mwh'] > 0 and kpis['average_charge_price_gbp_mwh'] > 0:
            spread_captured = kpis['average_discharge_price_gbp_mwh'] - kpis['average_charge_price_gbp_mwh']
            kpis['price_spread_captured_gbp_mwh'] = spread_captured
        else:
            kpis['price_spread_captured_gbp_mwh'] = 0

        # Market price statistics
        kpis['market_price_mean'] = schedule_df['price_gbp_mwh'].mean()
        kpis['market_price_std'] = schedule_df['price_gbp_mwh'].std()
        kpis['market_price_min'] = schedule_df['price_gbp_mwh'].min()
        kpis['market_price_max'] = schedule_df['price_gbp_mwh'].max()
        kpis['market_price_spread'] = kpis['market_price_max'] - kpis['market_price_min']

        return kpis

    def generate_finance_report(
        self,
        kpis: Dict,
        asset_name: str
    ) -> str:
        """
        Generate human-readable finance report

        Args:
            kpis: Dictionary of finance KPIs
            asset_name: Asset name

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("FINANCE KPI REPORT")
        report.append("=" * 80)
        report.append(f"Asset: {asset_name}")
        report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Grade: {kpis.get('finance_grade', 'N/A')}")
        report.append("")

        # Revenue Section
        report.append("REVENUE PERFORMANCE")
        report.append("-" * 80)
        report.append(f"  Actual Revenue:        £{kpis.get('arbitrage_revenue_gbp', 0):>12,.2f}")
        report.append(f"  Optimal Revenue:       £{kpis.get('optimal_revenue_gbp', 0):>12,.2f}")
        report.append(f"  Revenue Variance:      £{kpis.get('revenue_variance_gbp', 0):>12,.2f} ({kpis.get('revenue_variance_percent', 0):+.1f}%)")
        report.append(f"  Lost Opportunity:      £{kpis.get('lost_opportunity_gbp', 0):>12,.2f}")
        report.append("")

        # Market Capture Section
        report.append("MARKET CAPTURE ANALYSIS")
        report.append("-" * 80)
        report.append(f"  Market Capture Ratio:  {kpis.get('market_capture_ratio', 0):>12.1f}%")
        report.append(f"  Revenue per Cycle:     £{kpis.get('revenue_per_cycle_gbp', 0):>12,.2f}")
        report.append(f"  Revenue per MWh:       £{kpis.get('revenue_per_discharge_mwh', 0):>12,.2f}/MWh")
        report.append(f"  Revenue per Capacity:  £{kpis.get('revenue_per_capacity_mwh', 0):>12,.2f}/MWh")
        report.append("")

        # IRR Impact Section
        report.append("INVESTMENT IMPACT")
        report.append("-" * 80)
        report.append(f"  IRR Impact (Est.):     {kpis.get('irr_impact_estimate_bps', 0):>12,.0f} bps")
        report.append("")

        # Price Analysis (if available)
        if 'average_discharge_price_gbp_mwh' in kpis and kpis['average_discharge_price_gbp_mwh'] is not None:
            report.append("PRICE CAPTURE ANALYSIS")
            report.append("-" * 80)
            report.append(f"  Avg Discharge Price:   £{kpis.get('average_discharge_price_gbp_mwh', 0):>12,.2f}/MWh")
            report.append(f"  Avg Charge Price:      £{kpis.get('average_charge_price_gbp_mwh', 0):>12,.2f}/MWh")
            report.append(f"  Spread Captured:       £{kpis.get('price_spread_captured_gbp_mwh', 0):>12,.2f}/MWh")
            report.append(f"  Market Spread:         £{kpis.get('market_price_spread', 0):>12,.2f}/MWh")
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)
