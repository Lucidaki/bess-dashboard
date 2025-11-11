"""
Data Quality Scorer
Calculates comprehensive data quality scores with 4 components:
1. Completeness - Percentage of non-null values
2. Continuity - Timestamp gap analysis
3. Bounds - Values within expected ranges
4. Energy Reconciliation - Power integrates to SoC changes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import timedelta
from .schemas import DQComponent, DQReport


class DataQualityScorer:
    """
    Calculate comprehensive data quality scores
    """

    def __init__(self, config, dq_rules, asset_config):
        """
        Initialize DQ scorer

        Args:
            config: Main configuration
            dq_rules: Data quality remediation rules
            asset_config: BESS asset configuration
        """
        self.config = config
        self.dq_rules = dq_rules
        self.asset_config = asset_config
        self.dq_config = config['data_quality']
        self.settlement_duration_min = config['market']['settlement_duration_min']

    def score_scada(self, scada_df: pd.DataFrame) -> DQReport:
        """
        Calculate DQ score for SCADA data

        Args:
            scada_df: SCADA DataFrame

        Returns:
            DQReport with all components
        """
        print("\n📊 Calculating SCADA Data Quality Score...")

        # Calculate each component
        completeness = self._score_completeness(scada_df, 'scada')
        continuity = self._score_continuity(scada_df)
        bounds = self._score_bounds(scada_df)
        energy_recon = self._score_energy_reconciliation(scada_df)

        # Calculate weighted overall score
        weights = self.dq_config['weights']
        overall_score = (
            completeness.score * weights['completeness'] +
            continuity.score * weights['continuity'] +
            bounds.score * weights['bounds'] +
            energy_recon.score * weights['energy_reconciliation']
        )
        # Round to avoid floating point precision issues
        overall_score = round(overall_score, 2)

        # Check if passed
        min_dq_score = self.dq_config['min_dq_score']
        passed = overall_score >= min_dq_score

        # Generate remediation guidance
        remediation_required = []
        can_auto_remediate = True

        if not completeness.passed:
            remediation_required.extend(completeness.issues)
            if completeness.score < self.dq_rules['remediation_policies']['scada']['completeness']['threshold_auto_interpolate']:
                can_auto_remediate = False

        if not continuity.passed:
            remediation_required.extend(continuity.issues)

        if not bounds.passed:
            remediation_required.extend(bounds.issues)

        if not energy_recon.passed:
            remediation_required.extend(energy_recon.issues)
            can_auto_remediate = False  # Energy reconciliation failures cannot be auto-fixed

        # Build report
        report = DQReport(
            overall_score=overall_score,
            passed=passed,
            completeness=completeness,
            continuity=continuity,
            bounds=bounds,
            energy_reconciliation=energy_recon,
            total_periods=len(scada_df),
            valid_periods=int((completeness.score / 100) * len(scada_df)),
            missing_periods=scada_df.isna().any(axis=1).sum(),
            out_of_bounds_periods=int(((100 - bounds.score) / 100) * len(scada_df)),
            remediation_required=remediation_required,
            can_auto_remediate=can_auto_remediate,
            start_time=scada_df['timestamp_utc'].min(),
            end_time=scada_df['timestamp_utc'].max()
        )

        # Print summary
        self._print_dq_summary(report, 'SCADA')

        return report

    def score_market(self, market_df: pd.DataFrame) -> DQReport:
        """
        Calculate DQ score for market data

        Args:
            market_df: Market price DataFrame

        Returns:
            DQReport with relevant components
        """
        print("\n💰 Calculating Market Data Quality Score...")

        # Market data has different requirements
        completeness = self._score_completeness(market_df, 'market')
        continuity = self._score_market_continuity(market_df)
        bounds = self._score_market_bounds(market_df)

        # Energy reconciliation not applicable for market data
        energy_recon = DQComponent(
            score=100.0,
            passed=True,
            issues=[]
        )

        # Adjusted weights (redistribute energy_recon weight)
        weights = self.dq_config['weights']
        total_weight = weights['completeness'] + weights['continuity'] + weights['bounds']
        adjusted_weights = {
            'completeness': weights['completeness'] / total_weight,
            'continuity': weights['continuity'] / total_weight,
            'bounds': weights['bounds'] / total_weight
        }

        overall_score = (
            completeness.score * adjusted_weights['completeness'] +
            continuity.score * adjusted_weights['continuity'] +
            bounds.score * adjusted_weights['bounds']
        )
        # Round to avoid floating point precision issues
        overall_score = round(overall_score, 2)

        # Check if passed (higher threshold for market data)
        min_dq_score = self.dq_rules['remediation_policies']['market']['completeness']['threshold_reject']
        passed = overall_score >= min_dq_score

        # Remediation guidance
        remediation_required = []
        can_auto_remediate = True

        if not completeness.passed:
            remediation_required.extend(completeness.issues)

        if not continuity.passed:
            remediation_required.extend(continuity.issues)

        if not bounds.passed:
            remediation_required.extend(bounds.issues)
            can_auto_remediate = False  # Price violations cannot be auto-fixed

        # Build report
        report = DQReport(
            overall_score=overall_score,
            passed=passed,
            completeness=completeness,
            continuity=continuity,
            bounds=bounds,
            energy_reconciliation=energy_recon,
            total_periods=len(market_df),
            valid_periods=int((completeness.score / 100) * len(market_df)),
            missing_periods=market_df.isna().any(axis=1).sum(),
            out_of_bounds_periods=int(((100 - bounds.score) / 100) * len(market_df)),
            remediation_required=remediation_required,
            can_auto_remediate=can_auto_remediate,
            start_time=market_df['timestamp_utc'].min(),
            end_time=market_df['timestamp_utc'].max()
        )

        # Print summary
        self._print_dq_summary(report, 'Market')

        return report

    def _score_completeness(self, df: pd.DataFrame, data_type: str) -> DQComponent:
        """Calculate completeness score (percentage of non-null values)"""
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.notna().sum().sum()
        score = (non_null_cells / total_cells) * 100

        # Get threshold from rules
        if data_type == 'scada':
            threshold = self.dq_rules['remediation_policies']['scada']['completeness']['threshold_reject']
        else:
            threshold = self.dq_rules['remediation_policies']['market']['completeness']['threshold_reject']

        passed = score >= threshold

        issues = []
        if not passed:
            missing = total_cells - non_null_cells
            issues.append(f"Completeness {score:.1f}% below threshold {threshold}% ({missing} missing values)")

        return DQComponent(score=score, passed=passed, issues=issues)

    def _score_continuity(self, df: pd.DataFrame) -> DQComponent:
        """Calculate continuity score (timestamp gaps analysis)"""
        # Calculate time differences
        df = df.sort_values('timestamp_utc')
        time_diffs = df['timestamp_utc'].diff()

        # Expected interval
        expected_interval = timedelta(minutes=self.settlement_duration_min)

        # Count gaps
        gaps = time_diffs[time_diffs > expected_interval * 1.5]  # Allow 50% tolerance
        gap_count = len(gaps)
        total_intervals = len(time_diffs) - 1

        # Score based on proportion of gaps
        score = max(0, 100 - (gap_count / total_intervals * 100)) if total_intervals > 0 else 100

        # Check max gap size
        max_gap_min = self.dq_rules['remediation_policies']['scada']['continuity']['max_single_gap_minutes']
        max_gap_found = time_diffs.max().total_seconds() / 60 if len(time_diffs) > 0 else 0

        passed = (score >= 90) and (max_gap_found <= max_gap_min)

        issues = []
        if gap_count > 0:
            issues.append(f"Found {gap_count} timestamp gaps (largest: {max_gap_found:.1f} minutes)")
        if max_gap_found > max_gap_min:
            issues.append(f"Maximum gap {max_gap_found:.1f} min exceeds limit {max_gap_min} min")

        return DQComponent(score=score, passed=passed, issues=issues)

    def _score_bounds(self, df: pd.DataFrame) -> DQComponent:
        """Calculate bounds score (values within expected ranges)"""
        violations = 0
        total_values = 0
        issues = []

        # Check SoC bounds (0-100%)
        if 'soc_percent' in df.columns:
            soc_violations = ((df['soc_percent'] < 0) | (df['soc_percent'] > 100)).sum()
            violations += soc_violations
            total_values += len(df)
            if soc_violations > 0:
                issues.append(f"{soc_violations} SoC values outside 0-100% range")

        # Check power bounds (within asset limits)
        if 'power_mw' in df.columns:
            max_discharge = self.asset_config['constraints']['power_export_max_mw']
            max_charge = self.asset_config['constraints']['power_import_max_mw']

            power_too_high = (df['power_mw'] > max_discharge).sum()
            power_too_low = (df['power_mw'] < -max_charge).sum()

            violations += power_too_high + power_too_low
            total_values += len(df)

            if power_too_high > 0:
                issues.append(f"{power_too_high} power values exceed discharge limit {max_discharge} MW")
            if power_too_low > 0:
                issues.append(f"{power_too_low} power values exceed charge limit {max_charge} MW")

        # Calculate score
        score = max(0, 100 - (violations / total_values * 100)) if total_values > 0 else 100

        # Check threshold
        max_violations_pct = self.dq_rules['remediation_policies']['scada']['bounds']['max_violations_percent']
        passed = (violations / total_values * 100) <= max_violations_pct if total_values > 0 else True

        return DQComponent(score=score, passed=passed, issues=issues)

    def _score_energy_reconciliation(self, df: pd.DataFrame) -> DQComponent:
        """
        Calculate energy reconciliation score
        Validates that power integrates to SoC changes within tolerance
        """
        if len(df) < 2:
            return DQComponent(score=100.0, passed=True, issues=["Insufficient data for energy reconciliation"])

        # Calculate energy from power (MW × hours)
        dt_hours = self.settlement_duration_min / 60
        df = df.sort_values('timestamp_utc')

        # Energy in (charging, negative power)
        energy_in_mwh = abs((df[df['power_mw'] < 0]['power_mw'] * dt_hours).sum())

        # Energy out (discharging, positive power)
        energy_out_mwh = (df[df['power_mw'] > 0]['power_mw'] * dt_hours).sum()

        # Net energy from power
        net_energy_from_power = energy_out_mwh - energy_in_mwh  # Positive = net discharge

        # Energy from SoC change
        soc_start = df.iloc[0]['soc_percent']
        soc_end = df.iloc[-1]['soc_percent']
        soc_delta_pct = soc_end - soc_start

        capacity_mwh = self.asset_config['capacity_mwh']
        net_energy_from_soc = -(soc_delta_pct / 100) * capacity_mwh  # Negative delta = discharge

        # Calculate error
        error_mwh = abs(net_energy_from_power - net_energy_from_soc)
        error_pct = (error_mwh / capacity_mwh * 100) if capacity_mwh > 0 else 0

        # Tolerance from rules
        tolerance_pct = self.dq_rules['remediation_policies']['scada']['energy_reconciliation']['tolerance_percent']

        passed = error_pct <= tolerance_pct

        # Calculate score (100% if within tolerance, scaled down if over)
        if error_pct <= tolerance_pct:
            score = 100.0
        else:
            # Degrade score linearly from 100 to 0 as error increases from tolerance to 2x tolerance
            score = max(0, 100 - ((error_pct - tolerance_pct) / tolerance_pct * 100))

        issues = []
        if not passed:
            issues.append(
                f"Energy reconciliation error {error_pct:.2f}% exceeds tolerance {tolerance_pct}% "
                f"(Power: {net_energy_from_power:.2f} MWh, SoC: {net_energy_from_soc:.2f} MWh, "
                f"Diff: {error_mwh:.2f} MWh)"
            )

        return DQComponent(score=score, passed=passed, issues=issues)

    def _score_market_continuity(self, df: pd.DataFrame) -> DQComponent:
        """Calculate market data continuity (must have required settlement count)"""
        required_count = self.dq_rules['remediation_policies']['market']['continuity']['required_settlement_count']
        actual_count = len(df)

        score = min(100, (actual_count / required_count) * 100)
        passed = actual_count >= required_count

        issues = []
        if not passed:
            issues.append(f"Market data has {actual_count} periods, need {required_count} for daily optimization")

        return DQComponent(score=score, passed=passed, issues=issues)

    def _score_market_bounds(self, df: pd.DataFrame) -> DQComponent:
        """Calculate market price bounds score"""
        # This would check against market_constraints.yaml price caps
        # For now, just check for positive prices
        violations = (df['price_gbp_mwh'] <= 0).sum()
        total = len(df)

        score = max(0, 100 - (violations / total * 100)) if total > 0 else 100
        passed = violations == 0

        issues = []
        if violations > 0:
            issues.append(f"{violations} non-positive price values found")

        return DQComponent(score=score, passed=passed, issues=issues)

    def _print_dq_summary(self, report: DQReport, data_type: str):
        """Print DQ report summary"""
        print(f"\n{'='*60}")
        print(f"{data_type} DATA QUALITY REPORT")
        print(f"{'='*60}")
        print(f"Overall Score: {report.overall_score:.1f}% {'✅ PASS' if report.passed else '❌ FAIL'}")
        print(f"\nComponent Scores:")
        print(f"  Completeness:          {report.completeness.score:5.1f}% {'✅' if report.completeness.passed else '❌'}")
        print(f"  Continuity:            {report.continuity.score:5.1f}% {'✅' if report.continuity.passed else '❌'}")
        print(f"  Bounds:                {report.bounds.score:5.1f}% {'✅' if report.bounds.passed else '❌'}")
        print(f"  Energy Reconciliation: {report.energy_reconciliation.score:5.1f}% {'✅' if report.energy_reconciliation.passed else '❌'}")
        print(f"\nData Summary:")
        print(f"  Total Periods:    {report.total_periods}")
        print(f"  Valid Periods:    {report.valid_periods}")
        print(f"  Missing Periods:  {report.missing_periods}")
        print(f"  Time Range:       {report.start_time} to {report.end_time}")

        if report.remediation_required:
            print(f"\n⚠️  Remediation Required:")
            for issue in report.remediation_required:
                print(f"  - {issue}")
            if report.can_auto_remediate:
                print(f"\n✅ Can auto-remediate (will interpolate small gaps)")
            else:
                print(f"\n❌ Cannot auto-remediate (manual intervention required)")

        print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Data Quality Scorer module ready")
    print("Run via CLI tool: python ingest_data.py")
