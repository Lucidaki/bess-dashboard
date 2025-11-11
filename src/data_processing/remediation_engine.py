"""
Data Quality Remediation Engine
Implements auto-remediation logic based on dq_remediation_rules.yaml
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import timedelta
from .schemas import DQReport


class RemediationEngine:
    """
    Applies data quality remediation fixes based on configured rules
    """

    def __init__(self, dq_rules: Dict, settlement_duration_min: int = 30):
        """
        Initialize remediation engine

        Args:
            dq_rules: DQ remediation rules from config
            settlement_duration_min: Settlement period duration in minutes
        """
        self.dq_rules = dq_rules
        self.settlement_duration_min = settlement_duration_min
        self.remediation_log: List[str] = []

    def remediate_scada(self, df: pd.DataFrame, dq_report: DQReport) -> Tuple[pd.DataFrame, bool, List[str]]:
        """
        Apply remediation to SCADA data based on DQ report

        Args:
            df: SCADA dataframe with timestamp_utc index
            dq_report: Data quality assessment report

        Returns:
            Tuple of (remediated_df, success, remediation_log)
        """
        self.remediation_log = []
        remediated_df = df.copy()

        # Check if remediation is needed
        if dq_report.passed and not dq_report.remediation_required:
            self.remediation_log.append("✅ No remediation needed - data quality passed")
            return remediated_df, True, self.remediation_log

        # Get remediation policies
        scada_policies = self.dq_rules['remediation_policies']['scada']

        # Step 1: Bounds checking (flag and continue)
        if not dq_report.bounds.passed:
            remediated_df, bounds_success = self._remediate_bounds(remediated_df, scada_policies['bounds'])
            if not bounds_success:
                self.remediation_log.append("❌ Bounds remediation failed - too many violations")
                return remediated_df, False, self.remediation_log

        # Step 2: Completeness - interpolate missing values
        if not dq_report.completeness.passed or dq_report.missing_periods > 0:
            remediated_df, completeness_success = self._remediate_completeness(
                remediated_df,
                scada_policies['completeness'],
                dq_report
            )
            if not completeness_success:
                self.remediation_log.append("❌ Completeness remediation failed")
                return remediated_df, False, self.remediation_log

        # Step 3: Continuity - check gap sizes
        if not dq_report.continuity.passed:
            continuity_success = self._check_continuity(remediated_df, scada_policies['continuity'])
            if not continuity_success:
                self.remediation_log.append("❌ Continuity check failed - gaps too large")
                return remediated_df, False, self.remediation_log

        self.remediation_log.append("✅ Remediation completed successfully")
        return remediated_df, True, self.remediation_log

    def _remediate_bounds(self, df: pd.DataFrame, bounds_policy: Dict) -> Tuple[pd.DataFrame, bool]:
        """
        Remediate out-of-bounds values

        Args:
            df: SCADA dataframe
            bounds_policy: Bounds remediation policy

        Returns:
            Tuple of (remediated_df, success)
        """
        remediated_df = df.copy()
        action = bounds_policy['action']
        max_violations_pct = bounds_policy['max_violations_percent']

        # Check SoC hard limits
        soc_min = bounds_policy['soc_hard_limits']['min']
        soc_max = bounds_policy['soc_hard_limits']['max']

        soc_violations = (
            (remediated_df['soc_percent'] < soc_min) |
            (remediated_df['soc_percent'] > soc_max)
        )

        violation_pct = (soc_violations.sum() / len(remediated_df)) * 100

        if violation_pct > max_violations_pct:
            self.remediation_log.append(
                f"❌ SoC violations ({violation_pct:.1f}%) exceed threshold ({max_violations_pct}%)"
            )
            return remediated_df, False

        if action == "flag_and_continue" and violation_pct > 0:
            # Clip values to hard limits
            remediated_df.loc[remediated_df['soc_percent'] < soc_min, 'soc_percent'] = soc_min
            remediated_df.loc[remediated_df['soc_percent'] > soc_max, 'soc_percent'] = soc_max
            self.remediation_log.append(
                f"⚠️  Clipped {soc_violations.sum()} SoC values to hard limits [{soc_min}, {soc_max}]"
            )

        return remediated_df, True

    def _remediate_completeness(
        self,
        df: pd.DataFrame,
        completeness_policy: Dict,
        dq_report: DQReport
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Remediate missing data through interpolation

        Args:
            df: SCADA dataframe with timestamp_utc index
            completeness_policy: Completeness remediation policy
            dq_report: DQ report with completeness score

        Returns:
            Tuple of (remediated_df, success)
        """
        # Calculate actual completeness percentage
        completeness_pct = dq_report.completeness.score

        threshold_reject = completeness_policy['threshold_reject']
        threshold_interpolate = completeness_policy['threshold_auto_interpolate']

        # Hard reject if below rejection threshold
        if completeness_pct < threshold_reject:
            self.remediation_log.append(
                f"❌ Completeness {completeness_pct:.1f}% below rejection threshold {threshold_reject}%"
            )
            return df, False

        # Auto-interpolate if above interpolation threshold
        if completeness_pct >= threshold_interpolate:
            max_gap_min = completeness_policy['max_gap_minutes']
            method = completeness_policy['method']

            remediated_df = self._interpolate_gaps(df, max_gap_min, method)

            # Count how many values were interpolated
            original_nulls = df.isnull().sum().sum()
            remaining_nulls = remediated_df.isnull().sum().sum()
            interpolated_count = original_nulls - remaining_nulls

            if interpolated_count > 0:
                self.remediation_log.append(
                    f"✅ Interpolated {interpolated_count} missing values (method: {method}, max_gap: {max_gap_min}min)"
                )

            return remediated_df, True

        # Between thresholds - cannot auto-remediate
        self.remediation_log.append(
            f"⚠️  Completeness {completeness_pct:.1f}% requires manual intervention "
            f"(between {threshold_reject}% and {threshold_interpolate}%)"
        )
        return df, False

    def _interpolate_gaps(self, df: pd.DataFrame, max_gap_minutes: int, method: str) -> pd.DataFrame:
        """
        Interpolate missing values in gaps up to max_gap_minutes

        Args:
            df: SCADA dataframe with timestamp_utc index
            max_gap_minutes: Maximum gap size to interpolate
            method: Interpolation method ('linear', 'forward_fill', 'backward_fill')

        Returns:
            DataFrame with interpolated values
        """
        remediated_df = df.copy()

        # Calculate max gap in number of periods
        max_gap_periods = max_gap_minutes // self.settlement_duration_min

        if method == "linear":
            # Linear interpolation with limit on consecutive NaNs
            remediated_df['power_mw'] = remediated_df['power_mw'].interpolate(
                method='linear',
                limit=max_gap_periods,
                limit_area='inside'  # Only interpolate between valid values
            )
            remediated_df['soc_percent'] = remediated_df['soc_percent'].interpolate(
                method='linear',
                limit=max_gap_periods,
                limit_area='inside'
            )
        elif method == "forward_fill":
            remediated_df['power_mw'] = remediated_df['power_mw'].fillna(method='ffill', limit=max_gap_periods)
            remediated_df['soc_percent'] = remediated_df['soc_percent'].fillna(method='ffill', limit=max_gap_periods)
        elif method == "backward_fill":
            remediated_df['power_mw'] = remediated_df['power_mw'].fillna(method='bfill', limit=max_gap_periods)
            remediated_df['soc_percent'] = remediated_df['soc_percent'].fillna(method='bfill', limit=max_gap_periods)

        return remediated_df

    def _check_continuity(self, df: pd.DataFrame, continuity_policy: Dict) -> bool:
        """
        Check if continuity constraints are met after remediation

        Args:
            df: SCADA dataframe
            continuity_policy: Continuity policy

        Returns:
            True if continuity is acceptable, False otherwise
        """
        max_single_gap_min = continuity_policy['max_single_gap_minutes']
        max_total_gap_pct = continuity_policy['max_total_gap_percent']

        # Find gaps in timestamp index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp_utc')

        time_diffs = df.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=self.settlement_duration_min)

        # Identify gaps (time differences larger than expected)
        gaps = time_diffs[time_diffs > expected_diff]

        if len(gaps) > 0:
            # Check largest single gap
            max_gap = gaps.max()
            max_gap_minutes = max_gap.total_seconds() / 60

            if max_gap_minutes > max_single_gap_min:
                self.remediation_log.append(
                    f"❌ Maximum gap ({max_gap_minutes:.0f} min) exceeds limit ({max_single_gap_min} min)"
                )
                return False

            # Check total gap percentage
            total_gap_time = (gaps - expected_diff).sum()
            total_time = df.index[-1] - df.index[0]
            gap_percentage = (total_gap_time / total_time) * 100

            if gap_percentage > max_total_gap_pct:
                self.remediation_log.append(
                    f"❌ Total gaps ({gap_percentage:.1f}%) exceed limit ({max_total_gap_pct}%)"
                )
                return False

            self.remediation_log.append(
                f"✅ Continuity acceptable (max gap: {max_gap_minutes:.0f}min, total gaps: {gap_percentage:.1f}%)"
            )

        return True

    def remediate_market(self, df: pd.DataFrame, dq_report: DQReport) -> Tuple[pd.DataFrame, bool, List[str]]:
        """
        Apply remediation to market price data based on DQ report

        Args:
            df: Market price dataframe with timestamp_utc index
            dq_report: Data quality assessment report

        Returns:
            Tuple of (remediated_df, success, remediation_log)
        """
        self.remediation_log = []
        remediated_df = df.copy()

        # Check if remediation is needed
        if dq_report.passed and not dq_report.remediation_required:
            self.remediation_log.append("✅ No remediation needed - data quality passed")
            return remediated_df, True, self.remediation_log

        # Get remediation policies
        market_policies = self.dq_rules['remediation_policies']['market']

        # Step 1: Bounds checking (strict - reject on violation)
        if not dq_report.bounds.passed:
            self.remediation_log.append("❌ Market prices out of bounds - cannot remediate")
            return remediated_df, False, self.remediation_log

        # Step 2: Completeness - forward-fill missing prices
        if not dq_report.completeness.passed or dq_report.missing_periods > 0:
            remediated_df, completeness_success = self._remediate_market_completeness(
                remediated_df,
                market_policies['completeness'],
                dq_report
            )
            if not completeness_success:
                self.remediation_log.append("❌ Market completeness remediation failed")
                return remediated_df, False, self.remediation_log

        # Step 3: Continuity - check settlement count
        continuity_success = self._check_market_continuity(remediated_df, market_policies['continuity'])
        if not continuity_success:
            self.remediation_log.append("❌ Market continuity check failed")
            return remediated_df, False, self.remediation_log

        self.remediation_log.append("✅ Market remediation completed successfully")
        return remediated_df, True, self.remediation_log

    def _remediate_market_completeness(
        self,
        df: pd.DataFrame,
        completeness_policy: Dict,
        dq_report: DQReport
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Remediate missing market price data through forward-fill

        Args:
            df: Market price dataframe
            completeness_policy: Market completeness policy
            dq_report: DQ report

        Returns:
            Tuple of (remediated_df, success)
        """
        completeness_pct = dq_report.completeness.score

        threshold_reject = completeness_policy['threshold_reject']
        threshold_forward_fill = completeness_policy['threshold_auto_forward_fill']
        max_gap_periods = completeness_policy['max_gap_periods']

        # Hard reject if below rejection threshold
        if completeness_pct < threshold_reject:
            self.remediation_log.append(
                f"❌ Market completeness {completeness_pct:.1f}% below rejection threshold {threshold_reject}%"
            )
            return df, False

        # Auto forward-fill if above threshold
        if completeness_pct >= threshold_forward_fill:
            remediated_df = df.copy()
            original_nulls = remediated_df['price_gbp_mwh'].isnull().sum()

            # Forward-fill with limit
            remediated_df['price_gbp_mwh'] = remediated_df['price_gbp_mwh'].fillna(
                method='ffill',
                limit=max_gap_periods
            )

            remaining_nulls = remediated_df['price_gbp_mwh'].isnull().sum()
            filled_count = original_nulls - remaining_nulls

            if filled_count > 0:
                self.remediation_log.append(
                    f"✅ Forward-filled {filled_count} missing prices (max_gap: {max_gap_periods} periods)"
                )

            return remediated_df, True

        # Between thresholds - cannot auto-remediate
        self.remediation_log.append(
            f"⚠️  Market completeness {completeness_pct:.1f}% requires manual intervention"
        )
        return df, False

    def _check_market_continuity(self, df: pd.DataFrame, continuity_policy: Dict) -> bool:
        """
        Check if market data has required settlement period count

        Args:
            df: Market price dataframe
            continuity_policy: Market continuity policy

        Returns:
            True if continuity requirements met
        """
        required_count = continuity_policy['required_settlement_count']
        allow_partial = continuity_policy['allow_partial_days']

        actual_count = len(df)

        if not allow_partial and actual_count < required_count:
            self.remediation_log.append(
                f"❌ Market data has {actual_count} periods, requires {required_count} for daily optimization"
            )
            return False

        self.remediation_log.append(
            f"✅ Market continuity acceptable ({actual_count} periods)"
        )
        return True
