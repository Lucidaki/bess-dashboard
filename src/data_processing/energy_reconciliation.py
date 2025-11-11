"""
Energy Reconciliation Module
Validates energy balance between power measurements and SoC changes
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnergyReconciliationResult:
    """
    Results from energy reconciliation validation
    """
    passed: bool
    error_percent: float
    error_mwh: float

    # Energy calculations
    energy_from_power_mwh: float
    energy_from_soc_mwh: float

    # Power breakdown
    charge_energy_mwh: float
    discharge_energy_mwh: float
    net_energy_from_power_mwh: float

    # SoC breakdown
    soc_start_percent: float
    soc_end_percent: float
    soc_delta_percent: float

    # Confidence metrics
    confidence_score: float
    within_tolerance: bool
    tolerance_percent: float

    # Metadata
    capacity_mwh: float
    total_periods: int
    duration_hours: float

    def __str__(self) -> str:
        """Human-readable summary"""
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"Energy Reconciliation: {status}\n"
            f"  Error: {self.error_percent:.2f}% ({self.error_mwh:.3f} MWh)\n"
            f"  Power Integration: {self.net_energy_from_power_mwh:.3f} MWh\n"
            f"  SoC Change: {self.energy_from_soc_mwh:.3f} MWh\n"
            f"  Tolerance: ±{self.tolerance_percent}%\n"
            f"  Confidence: {self.confidence_score:.1f}%"
        )


class EnergyReconciliation:
    """
    Performs energy reconciliation between power measurements and SoC changes
    """

    def __init__(self, capacity_mwh: float, tolerance_percent: float = 5.0):
        """
        Initialize energy reconciliation

        Args:
            capacity_mwh: BESS usable capacity in MWh
            tolerance_percent: Acceptable error tolerance (default 5%)
        """
        self.capacity_mwh = capacity_mwh
        self.tolerance_percent = tolerance_percent

    def reconcile(
        self,
        df: pd.DataFrame,
        settlement_duration_min: int = 30
    ) -> EnergyReconciliationResult:
        """
        Perform energy reconciliation on SCADA data

        Args:
            df: SCADA dataframe with power_mw and soc_percent columns
            settlement_duration_min: Settlement period duration in minutes

        Returns:
            EnergyReconciliationResult with detailed analysis
        """
        # Calculate time delta in hours
        dt_hours = settlement_duration_min / 60.0

        # Energy from power integration
        charge_energy_mwh = abs((df[df['power_mw'] < 0]['power_mw'] * dt_hours).sum())
        discharge_energy_mwh = (df[df['power_mw'] > 0]['power_mw'] * dt_hours).sum()
        net_energy_from_power = discharge_energy_mwh - charge_energy_mwh

        # Energy from SoC change
        soc_start = df['soc_percent'].iloc[0]
        soc_end = df['soc_percent'].iloc[-1]
        soc_delta = soc_end - soc_start

        # Negative SoC delta means discharge (energy out)
        # Positive SoC delta means charge (energy in)
        # We flip the sign to match power convention
        energy_from_soc = -(soc_delta / 100) * self.capacity_mwh

        # Calculate error
        error_mwh = abs(net_energy_from_power - energy_from_soc)
        error_percent = (error_mwh / self.capacity_mwh) * 100

        # Check if within tolerance
        within_tolerance = error_percent <= self.tolerance_percent

        # Calculate confidence score (100% - normalized error)
        # Error of 0% = 100% confidence
        # Error of tolerance% = 50% confidence
        # Error > 2*tolerance% = 0% confidence
        if error_percent == 0:
            confidence_score = 100.0
        else:
            confidence_score = max(0, 100 - (error_percent / self.tolerance_percent) * 50)

        # Metadata
        total_periods = len(df)
        duration_hours = total_periods * dt_hours

        return EnergyReconciliationResult(
            passed=within_tolerance,
            error_percent=error_percent,
            error_mwh=error_mwh,
            energy_from_power_mwh=net_energy_from_power,
            energy_from_soc_mwh=energy_from_soc,
            charge_energy_mwh=charge_energy_mwh,
            discharge_energy_mwh=discharge_energy_mwh,
            net_energy_from_power_mwh=net_energy_from_power,
            soc_start_percent=soc_start,
            soc_end_percent=soc_end,
            soc_delta_percent=soc_delta,
            confidence_score=confidence_score,
            within_tolerance=within_tolerance,
            tolerance_percent=self.tolerance_percent,
            capacity_mwh=self.capacity_mwh,
            total_periods=total_periods,
            duration_hours=duration_hours
        )

    def analyze_metering_drift(
        self,
        df: pd.DataFrame,
        window_hours: int = 24,
        settlement_duration_min: int = 30
    ) -> pd.DataFrame:
        """
        Analyze energy reconciliation over rolling time windows to detect metering drift

        Args:
            df: SCADA dataframe with timestamp_utc index
            window_hours: Rolling window size in hours
            settlement_duration_min: Settlement period duration in minutes

        Returns:
            DataFrame with rolling reconciliation errors
        """
        periods_per_window = int((window_hours * 60) / settlement_duration_min)

        results = []

        # Ensure timestamp_utc is the index
        if 'timestamp_utc' in df.columns:
            df = df.set_index('timestamp_utc')

        # Rolling window analysis
        for i in range(len(df) - periods_per_window + 1):
            window_df = df.iloc[i:i+periods_per_window]

            # Perform reconciliation on window
            result = self.reconcile(window_df, settlement_duration_min)

            results.append({
                'window_start': window_df.index[0],
                'window_end': window_df.index[-1],
                'error_percent': result.error_percent,
                'error_mwh': result.error_mwh,
                'confidence_score': result.confidence_score,
                'passed': result.passed
            })

        return pd.DataFrame(results)

    def diagnose_error_sources(
        self,
        df: pd.DataFrame,
        settlement_duration_min: int = 30
    ) -> Dict[str, any]:
        """
        Diagnose potential sources of energy reconciliation errors

        Args:
            df: SCADA dataframe
            settlement_duration_min: Settlement period duration

        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {}

        # Check for power measurement anomalies
        power_stats = df['power_mw'].describe()
        diagnostics['power_stats'] = power_stats.to_dict()

        # Check for zero power periods (idle BESS)
        zero_power_periods = (df['power_mw'].abs() < 0.001).sum()
        zero_power_pct = (zero_power_periods / len(df)) * 100
        diagnostics['zero_power_periods'] = zero_power_periods
        diagnostics['zero_power_percent'] = zero_power_pct

        # Check for SoC drift during zero power
        if zero_power_periods > 0:
            zero_power_df = df[df['power_mw'].abs() < 0.001]
            if len(zero_power_df) > 1:
                soc_drift = abs(zero_power_df['soc_percent'].iloc[-1] - zero_power_df['soc_percent'].iloc[0])
                diagnostics['soc_drift_during_idle'] = soc_drift
                diagnostics['idle_period_error_flag'] = soc_drift > 1.0  # More than 1% drift during idle

        # Check for sudden SoC jumps
        soc_changes = df['soc_percent'].diff().abs()
        max_soc_jump = soc_changes.max()
        diagnostics['max_soc_jump_percent'] = max_soc_jump
        diagnostics['sudden_jump_flag'] = max_soc_jump > 10.0  # >10% jump between periods

        # Check for power/SoC direction mismatch
        # Negative power (charging) should increase SoC
        # Positive power (discharging) should decrease SoC
        charging_periods = df[df['power_mw'] < -0.1]  # Significant charging
        if len(charging_periods) > 1:
            soc_increase_during_charge = charging_periods['soc_percent'].diff().mean()
            diagnostics['soc_change_during_charge'] = soc_increase_during_charge
            diagnostics['charge_direction_error'] = soc_increase_during_charge < 0  # SoC decreasing during charge

        discharging_periods = df[df['power_mw'] > 0.1]  # Significant discharging
        if len(discharging_periods) > 1:
            soc_decrease_during_discharge = discharging_periods['soc_percent'].diff().mean()
            diagnostics['soc_change_during_discharge'] = soc_decrease_during_discharge
            diagnostics['discharge_direction_error'] = soc_decrease_during_discharge > 0  # SoC increasing during discharge

        # Overall reconciliation
        recon_result = self.reconcile(df, settlement_duration_min)
        diagnostics['overall_error_percent'] = recon_result.error_percent
        diagnostics['overall_confidence'] = recon_result.confidence_score

        # Likely error source
        if diagnostics.get('idle_period_error_flag', False):
            diagnostics['likely_error_source'] = "SoC meter drift during idle periods"
        elif diagnostics.get('sudden_jump_flag', False):
            diagnostics['likely_error_source'] = "Sudden SoC jumps (measurement errors or recalibration)"
        elif diagnostics.get('charge_direction_error', False) or diagnostics.get('discharge_direction_error', False):
            diagnostics['likely_error_source'] = "Power/SoC direction mismatch (metering sign error)"
        elif zero_power_pct > 80:
            diagnostics['likely_error_source'] = "Mostly idle BESS (insufficient active periods for validation)"
        else:
            diagnostics['likely_error_source'] = "Cumulative metering accuracy or RTE losses"

        return diagnostics

    def generate_reconciliation_report(
        self,
        df: pd.DataFrame,
        settlement_duration_min: int = 30
    ) -> str:
        """
        Generate a comprehensive energy reconciliation report

        Args:
            df: SCADA dataframe
            settlement_duration_min: Settlement period duration

        Returns:
            Formatted report string
        """
        result = self.reconcile(df, settlement_duration_min)
        diagnostics = self.diagnose_error_sources(df, settlement_duration_min)

        report = []
        report.append("=" * 70)
        report.append("ENERGY RECONCILIATION REPORT")
        report.append("=" * 70)
        report.append("")

        # Overall result
        report.append(str(result))
        report.append("")

        # Energy breakdown
        report.append("Energy Breakdown:")
        report.append(f"  Charge Energy:    {result.charge_energy_mwh:8.3f} MWh")
        report.append(f"  Discharge Energy: {result.discharge_energy_mwh:8.3f} MWh")
        report.append(f"  Net (Power):      {result.net_energy_from_power_mwh:8.3f} MWh")
        report.append(f"  Net (SoC):        {result.energy_from_soc_mwh:8.3f} MWh")
        report.append(f"  Difference:       {result.error_mwh:8.3f} MWh ({result.error_percent:.2f}%)")
        report.append("")

        # SoC summary
        report.append("SoC Summary:")
        report.append(f"  Start: {result.soc_start_percent:.2f}%")
        report.append(f"  End:   {result.soc_end_percent:.2f}%")
        report.append(f"  Delta: {result.soc_delta_percent:+.2f}%")
        report.append("")

        # Diagnostics
        report.append("Diagnostics:")
        report.append(f"  Likely Error Source: {diagnostics['likely_error_source']}")
        report.append(f"  Zero Power Periods: {diagnostics['zero_power_periods']} ({diagnostics['zero_power_percent']:.1f}%)")

        if 'soc_drift_during_idle' in diagnostics:
            report.append(f"  SoC Drift During Idle: {diagnostics['soc_drift_during_idle']:.2f}%")

        if 'max_soc_jump_percent' in diagnostics:
            report.append(f"  Max SoC Jump: {diagnostics['max_soc_jump_percent']:.2f}%")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)
