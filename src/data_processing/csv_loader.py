"""
CSV Loader
Handles loading and initial validation of SCADA and market price CSV files
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
import pytz


class CSVLoader:
    """
    Generic CSV loader with robust error handling
    Handles BOM characters, various timestamp formats, and column name normalization
    """

    # Supported timestamp formats (in order of preference)
    TIMESTAMP_FORMATS = [
        "%d/%m/%Y %H:%M:%S",      # 15/10/2025 00:00:00
        "%d-%m-%Y %H:%M",         # 15-10-2025 00:00
        "%Y-%m-%d %H:%M:%S",      # 2025-10-15 00:00:00
        "%Y-%m-%dT%H:%M:%S",      # 2025-10-15T00:00:00
        "%Y-%m-%dT%H:%M:%SZ",     # 2025-10-15T00:00:00Z (ISO8601)
    ]

    def __init__(self, config):
        """
        Initialize CSV loader

        Args:
            config: Configuration object containing timezone and other settings
        """
        self.config = config
        self.timezone = pytz.timezone(config['market']['timezone'])

    def load_scada_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load SCADA CSV file

        Args:
            file_path: Path to SCADA CSV file

        Returns:
            DataFrame with columns: timestamp_utc, power_mw, soc_percent

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"SCADA file not found: {file_path}")

        # Load CSV with BOM handling
        df = pd.read_csv(
            file_path,
            encoding='utf-8-sig',  # Handles BOM character
            skipinitialspace=True
        )

        # Normalize column names (lowercase, strip whitespace)
        df.columns = df.columns.str.strip().str.lower()

        # Validate required columns
        required_columns = {'timestamp', 'power_mw', 'soc_percent'}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns in SCADA CSV: {missing_columns}\n"
                f"Found columns: {list(df.columns)}"
            )

        # Parse timestamp
        df['timestamp_utc'] = self._parse_timestamp(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp_utc').reset_index(drop=True)

        # Return only canonical columns
        return df[['timestamp_utc', 'power_mw', 'soc_percent']].copy()

    def load_market_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load market price CSV file

        Args:
            file_path: Path to market price CSV file

        Returns:
            DataFrame with columns: timestamp_utc, price_gbp_mwh, market_type

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Market price file not found: {file_path}")

        # Load CSV with BOM handling
        df = pd.read_csv(
            file_path,
            encoding='utf-8-sig',  # Handles BOM character
            skipinitialspace=True
        )

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Validate required columns
        required_columns = {'timestamp', 'price_gbp_mwh', 'market_type'}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns in market CSV: {missing_columns}\n"
                f"Found columns: {list(df.columns)}"
            )

        # Parse timestamp
        df['timestamp_utc'] = self._parse_timestamp(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp_utc').reset_index(drop=True)

        # Normalize market_type
        df['market_type'] = df['market_type'].str.strip().str.lower()

        # Return only canonical columns
        return df[['timestamp_utc', 'price_gbp_mwh', 'market_type']].copy()

    def _parse_timestamp(self, timestamp_series: pd.Series) -> pd.Series:
        """
        Parse timestamps with multiple format support

        Args:
            timestamp_series: Series of timestamp strings

        Returns:
            Series of timezone-aware UTC datetimes

        Raises:
            ValueError: If timestamps cannot be parsed
        """
        # Try each format until one works
        for fmt in self.TIMESTAMP_FORMATS:
            try:
                # Parse with specified format
                parsed = pd.to_datetime(timestamp_series, format=fmt)

                # Check if parsing was successful (not all NaT)
                if not parsed.isna().all():
                    # Localize to configured timezone, then convert to UTC
                    if parsed.dt.tz is None:
                        # Assume local timezone from config
                        parsed = parsed.dt.tz_localize(self.timezone, ambiguous='infer')

                    # Convert to UTC
                    parsed = parsed.dt.tz_convert('UTC')

                    print(f"✅ Parsed timestamps using format: {fmt}")
                    return parsed

            except (ValueError, TypeError):
                continue

        # If all formats failed, try pandas auto-detection as fallback
        try:
            parsed = pd.to_datetime(timestamp_series, dayfirst=True)  # UK format default

            if parsed.dt.tz is None:
                parsed = parsed.dt.tz_localize(self.timezone, ambiguous='infer')

            parsed = parsed.dt.tz_convert('UTC')

            print("✅ Parsed timestamps using pandas auto-detection")
            return parsed

        except Exception as e:
            raise ValueError(
                f"Failed to parse timestamps. Tried formats: {self.TIMESTAMP_FORMATS}\n"
                f"Error: {e}\n"
                f"Sample timestamps: {timestamp_series.head(3).tolist()}"
            )

    def validate_data_types(self, df: pd.DataFrame, data_type: str) -> Tuple[bool, list]:
        """
        Validate data types and ranges

        Args:
            df: DataFrame to validate
            data_type: 'scada' or 'market'

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if data_type == 'scada':
            # Check power_mw is numeric
            if not pd.api.types.is_numeric_dtype(df['power_mw']):
                issues.append("power_mw must be numeric")

            # Check soc_percent is numeric and in range
            if not pd.api.types.is_numeric_dtype(df['soc_percent']):
                issues.append("soc_percent must be numeric")
            elif (df['soc_percent'] < 0).any() or (df['soc_percent'] > 100).any():
                invalid_count = ((df['soc_percent'] < 0) | (df['soc_percent'] > 100)).sum()
                issues.append(f"soc_percent has {invalid_count} values outside 0-100% range")

        elif data_type == 'market':
            # Check price_gbp_mwh is numeric and positive
            if not pd.api.types.is_numeric_dtype(df['price_gbp_mwh']):
                issues.append("price_gbp_mwh must be numeric")
            elif (df['price_gbp_mwh'] <= 0).any():
                invalid_count = (df['price_gbp_mwh'] <= 0).sum()
                issues.append(f"price_gbp_mwh has {invalid_count} non-positive values")

            # Check market_type is valid
            valid_types = {'day_ahead', 'imbalance', 'blended'}
            invalid_types = set(df['market_type'].unique()) - valid_types
            if invalid_types:
                issues.append(f"Invalid market_type values: {invalid_types}")

        return len(issues) == 0, issues

    def get_date_range(self, df: pd.DataFrame) -> Tuple[datetime, datetime]:
        """
        Get date range of data

        Args:
            df: DataFrame with timestamp_utc column

        Returns:
            Tuple of (start_time, end_time)
        """
        return df['timestamp_utc'].min(), df['timestamp_utc'].max()

    def check_temporal_resolution(self, df: pd.DataFrame) -> dict:
        """
        Analyze temporal resolution of data

        Args:
            df: DataFrame with timestamp_utc column

        Returns:
            Dictionary with resolution statistics
        """
        # Calculate time differences
        time_diffs = df['timestamp_utc'].diff()

        # Most common interval (mode)
        mode_interval = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else None

        return {
            'min_interval': time_diffs.min(),
            'max_interval': time_diffs.max(),
            'mode_interval': mode_interval,
            'total_periods': len(df),
            'has_gaps': (time_diffs > mode_interval * 1.5).any() if mode_interval else False
        }


if __name__ == "__main__":
    # Test CSV loader
    from src.config_loader import get_config_loader

    print("Testing CSV Loader...")

    # Load config
    loader = get_config_loader()
    configs = loader.load_all_configs()

    # Initialize CSV loader
    csv_loader = CSVLoader(configs['config'].dict())

    try:
        # Test SCADA loading
        print("\n📊 Loading SCADA CSV...")
        scada_df = csv_loader.load_scada_csv("data/raw/Scada csv.csv")
        print(f"✅ Loaded {len(scada_df)} SCADA records")
        print(f"   Date range: {scada_df['timestamp_utc'].min()} to {scada_df['timestamp_utc'].max()}")

        # Check resolution
        scada_res = csv_loader.check_temporal_resolution(scada_df)
        print(f"   Temporal resolution: {scada_res['mode_interval']}")
        print(f"   Has gaps: {scada_res['has_gaps']}")

        # Validate
        valid, issues = csv_loader.validate_data_types(scada_df, 'scada')
        if valid:
            print("✅ SCADA data validation passed")
        else:
            print(f"⚠️  SCADA validation issues: {issues}")

        print(f"\nFirst 3 records:\n{scada_df.head(3)}")

        # Test market loading
        print("\n💰 Loading Market Price CSV...")
        market_df = csv_loader.load_market_csv("data/raw/Market price csv.csv")
        print(f"✅ Loaded {len(market_df)} market price records")
        print(f"   Date range: {market_df['timestamp_utc'].min()} to {market_df['timestamp_utc'].max()}")

        # Check resolution
        market_res = csv_loader.check_temporal_resolution(market_df)
        print(f"   Temporal resolution: {market_res['mode_interval']}")

        # Validate
        valid, issues = csv_loader.validate_data_types(market_df, 'market')
        if valid:
            print("✅ Market data validation passed")
        else:
            print(f"⚠️  Market validation issues: {issues}")

        print(f"\nFirst 3 records:\n{market_df.head(3)}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise
