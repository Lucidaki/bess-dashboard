"""
Data Cleaner
Handles resampling, interpolation, and timestamp alignment
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import timedelta


class DataCleaner:
    """
    Clean and resample SCADA and market data to canonical format
    """

    def __init__(self, config, dq_rules):
        """
        Initialize data cleaner

        Args:
            config: Main configuration object
            dq_rules: Data quality remediation rules
        """
        self.config = config
        self.dq_rules = dq_rules
        self.settlement_duration_min = config['market']['settlement_duration_min']

    def resample_scada(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample SCADA data to settlement period frequency

        Args:
            df: DataFrame with timestamp_utc, power_mw, soc_percent

        Returns:
            Resampled DataFrame at settlement period frequency
        """
        print(f"\n🔄 Resampling SCADA data to {self.settlement_duration_min}-minute intervals...")

        # Set timestamp as index
        df = df.set_index('timestamp_utc')

        # Determine current resolution
        time_diff = df.index.to_series().diff().median()
        current_resolution_min = time_diff.total_seconds() / 60

        print(f"   Current resolution: {current_resolution_min:.1f} minutes")
        print(f"   Target resolution: {self.settlement_duration_min} minutes")

        if current_resolution_min == self.settlement_duration_min:
            print("✅ Data already at target resolution")
            return df.reset_index()

        # Resample rule
        resample_rule = f"{self.settlement_duration_min}min"  # min = minute frequency

        # Resample different columns appropriately
        resampled = pd.DataFrame()

        # Power: Average (mean power over period)
        resampled['power_mw'] = df['power_mw'].resample(resample_rule).mean()

        # SoC: Last value of period (end-of-period SoC)
        resampled['soc_percent'] = df['soc_percent'].resample(resample_rule).last()

        # Reset index to get timestamp_utc back as column
        resampled = resampled.reset_index()

        # Remove any NaN rows
        initial_count = len(resampled)
        resampled = resampled.dropna()
        final_count = len(resampled)

        if initial_count > final_count:
            print(f"⚠️  Dropped {initial_count - final_count} periods with NaN values during resampling")

        print(f"✅ Resampled from {len(df)} to {len(resampled)} periods")

        return resampled

    def interpolate_missing(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Interpolate missing values according to remediation rules

        Args:
            df: DataFrame with potential missing values
            data_type: 'scada' or 'market'

        Returns:
            DataFrame with interpolated values
        """
        rules = self.dq_rules['remediation_policies'][data_type]

        # Set timestamp as index if not already
        if 'timestamp_utc' in df.columns:
            df = df.set_index('timestamp_utc')

        print(f"\n🔍 Checking for missing values in {data_type} data...")

        # Check for missing values
        missing_counts = df.isna().sum()

        if missing_counts.sum() == 0:
            print("✅ No missing values found")
            return df.reset_index()

        print(f"   Found missing values: {missing_counts[missing_counts > 0].to_dict()}")

        # Get remediation method
        if data_type == 'scada':
            method = rules['completeness'].get('method', 'linear')
            max_gap_min = rules['completeness'].get('max_gap_minutes', 60)
        else:  # market
            method = 'forward_fill'  # Forward-fill for prices
            max_gap_min = rules['completeness'].get('max_gap_periods', 2) * self.settlement_duration_min

        # Identify gaps
        for column in df.columns:
            if df[column].isna().any():
                # Find gap sizes
                is_missing = df[column].isna()
                gap_starts = is_missing & ~is_missing.shift(1, fill_value=False)
                gap_ends = is_missing & ~is_missing.shift(-1, fill_value=False)

                gap_indices = []
                for start_idx in df[gap_starts].index:
                    end_idx = df[gap_ends & (df.index >= start_idx)].index[0] if any(gap_ends & (df.index >= start_idx)) else df.index[-1]
                    gap_duration = (end_idx - start_idx).total_seconds() / 60
                    gap_indices.append((start_idx, end_idx, gap_duration))

                # Interpolate gaps within limit
                for start, end, duration in gap_indices:
                    if duration <= max_gap_min:
                        # Interpolate
                        if method == 'linear':
                            df[column] = df[column].interpolate(method='linear', limit_direction='both')
                        elif method == 'forward_fill':
                            df[column] = df[column].fillna(method='ffill')
                        print(f"✅ Interpolated {column} gap of {duration:.1f} minutes using {method}")
                    else:
                        print(f"⚠️  Gap too large to interpolate: {duration:.1f} minutes (max: {max_gap_min} minutes)")

        return df.reset_index()

    def align_timestamps(
        self,
        scada_df: pd.DataFrame,
        market_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align SCADA and market data timestamps

        Args:
            scada_df: SCADA DataFrame
            market_df: Market DataFrame

        Returns:
            Tuple of (aligned_scada, aligned_market)
        """
        print("\n🔗 Aligning SCADA and market timestamps...")

        # Find common time range
        scada_start = scada_df['timestamp_utc'].min()
        scada_end = scada_df['timestamp_utc'].max()
        market_start = market_df['timestamp_utc'].min()
        market_end = market_df['timestamp_utc'].max()

        # Common range
        common_start = max(scada_start, market_start)
        common_end = min(scada_end, market_end)

        if common_start >= common_end:
            raise ValueError(
                f"No timestamp overlap between SCADA and market data!\n"
                f"SCADA range: {scada_start} to {scada_end}\n"
                f"Market range: {market_start} to {market_end}"
            )

        # Filter to common range
        scada_aligned = scada_df[
            (scada_df['timestamp_utc'] >= common_start) &
            (scada_df['timestamp_utc'] <= common_end)
        ].copy()

        market_aligned = market_df[
            (market_df['timestamp_utc'] >= common_start) &
            (market_df['timestamp_utc'] <= common_end)
        ].copy()

        print(f"   Common time range: {common_start} to {common_end}")
        print(f"   SCADA periods: {len(scada_aligned)}")
        print(f"   Market periods: {len(market_aligned)}")

        # Check for perfect alignment
        scada_times = set(scada_aligned['timestamp_utc'])
        market_times = set(market_aligned['timestamp_utc'])

        missing_in_scada = market_times - scada_times
        missing_in_market = scada_times - market_times

        if missing_in_scada:
            print(f"⚠️  {len(missing_in_scada)} timestamps in market but not in SCADA")

        if missing_in_market:
            print(f"⚠️  {len(missing_in_market)} timestamps in SCADA but not in market")

        # Merge to ensure perfect alignment (inner join)
        merged = pd.merge(
            scada_aligned,
            market_aligned,
            on='timestamp_utc',
            how='inner',
            suffixes=('_scada', '_market')
        )

        if len(merged) == 0:
            raise ValueError("No matching timestamps after alignment!")

        # Separate back
        scada_final = merged[['timestamp_utc', 'power_mw', 'soc_percent']].copy()
        market_final = merged[['timestamp_utc', 'price_gbp_mwh', 'market_type']].copy()

        print(f"✅ Aligned to {len(merged)} common periods")

        return scada_final, market_final

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate timestamps, keeping the first occurrence

        Args:
            df: DataFrame with potential duplicates

        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp_utc'], keep='first')
        final_count = len(df)

        if initial_count > final_count:
            print(f"⚠️  Removed {initial_count - final_count} duplicate timestamps")

        return df

    def clean_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = 'clip',
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        std_threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Clean outliers from data

        Args:
            df: DataFrame
            column: Column to clean
            method: 'clip' (clip to bounds) or 'remove' (remove outliers)
            lower_bound: Lower bound (if None, use mean - std_threshold * std)
            upper_bound: Upper bound (if None, use mean + std_threshold * std)
            std_threshold: Number of standard deviations for auto bounds

        Returns:
            Cleaned DataFrame
        """
        if column not in df.columns:
            return df

        # Calculate bounds if not provided
        if lower_bound is None or upper_bound is None:
            mean = df[column].mean()
            std = df[column].std()
            if lower_bound is None:
                lower_bound = mean - std_threshold * std
            if upper_bound is None:
                upper_bound = mean + std_threshold * std

        # Identify outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = outliers.sum()

        if outlier_count == 0:
            return df

        print(f"⚠️  Found {outlier_count} outliers in {column}")

        if method == 'clip':
            # Clip values to bounds
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            print(f"✅ Clipped outliers to [{lower_bound:.2f}, {upper_bound:.2f}]")
        elif method == 'remove':
            # Remove outlier rows
            df = df[~outliers].copy()
            print(f"✅ Removed {outlier_count} outlier rows")

        return df


if __name__ == "__main__":
    print("Data Cleaner module ready")
    print("Run via CLI tool: python ingest_data.py")
