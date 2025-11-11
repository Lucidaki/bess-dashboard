"""
Price Selector
Handles selection and aggregation of market price stacks
"""

import pandas as pd
from typing import Optional


class PriceSelector:
    """
    Select and aggregate market price data based on configuration rules
    """

    def __init__(self, price_rules):
        """
        Initialize price selector

        Args:
            price_rules: Price selection rules from config
        """
        self.price_rules = price_rules
        self.uk_rules = price_rules['uk_market']

    def select_optimization_prices(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select prices for optimization based on configured mode

        Args:
            market_df: Market price DataFrame with all price types

        Returns:
            DataFrame with selected prices for optimization
        """
        mode = self.uk_rules['optimization_price_source']['mode']

        print(f"\n💰 Selecting optimization prices (mode: {mode})...")

        if mode == "day_ahead_only":
            return self._select_day_ahead(market_df)

        elif mode == "imbalance_only":
            return self._select_imbalance(market_df)

        elif mode == "blended":
            return self._select_blended(market_df)

        else:
            raise ValueError(f"Unknown price selection mode: {mode}")

    def _select_day_ahead(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """Select only day-ahead prices"""
        day_ahead = market_df[market_df['market_type'] == 'day_ahead'].copy()

        if len(day_ahead) == 0:
            # Check fallback
            fallback = self.uk_rules['fallback'].get('if_day_ahead_missing', 'reject')

            if fallback == 'use_imbalance':
                print("⚠️  No day-ahead prices found, using imbalance prices as fallback")
                return self._select_imbalance(market_df)
            else:
                raise ValueError("No day-ahead prices found and fallback is 'reject'")

        print(f"✅ Selected {len(day_ahead)} day-ahead price periods")
        return day_ahead

    def _select_imbalance(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """Select only imbalance prices"""
        imbalance = market_df[market_df['market_type'] == 'imbalance'].copy()

        if len(imbalance) == 0:
            # Check fallback
            fallback = self.uk_rules['fallback'].get('if_imbalance_missing', 'reject')

            if fallback == 'use_day_ahead':
                print("⚠️  No imbalance prices found, using day-ahead prices as fallback")
                return self._select_day_ahead(market_df)
            else:
                raise ValueError("No imbalance prices found and fallback is 'reject'")

        print(f"✅ Selected {len(imbalance)} imbalance price periods")
        return imbalance

    def _select_blended(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """Blend day-ahead and imbalance prices"""
        blended_config = self.uk_rules['blended_mode']

        if not blended_config.get('enabled', False):
            raise ValueError("Blended mode not enabled in configuration")

        # Get weights
        weights = blended_config['weights']
        day_ahead_weight = weights.get('day_ahead', 0.7)
        imbalance_weight = weights.get('imbalance', 0.3)

        # Get aggregation method
        aggregation = blended_config.get('aggregation', 'weighted_average')

        # Extract price types
        day_ahead = market_df[market_df['market_type'] == 'day_ahead'].copy()
        imbalance = market_df[market_df['market_type'] == 'imbalance'].copy()

        # Merge on timestamp
        merged = pd.merge(
            day_ahead[['timestamp_utc', 'price_gbp_mwh']],
            imbalance[['timestamp_utc', 'price_gbp_mwh']],
            on='timestamp_utc',
            how='outer',
            suffixes=('_da', '_ib')
        )

        # Apply aggregation
        if aggregation == 'weighted_average':
            # Forward-fill missing values
            merged['price_gbp_mwh_da'] = merged['price_gbp_mwh_da'].fillna(method='ffill')
            merged['price_gbp_mwh_ib'] = merged['price_gbp_mwh_ib'].fillna(method='ffill')

            # Weighted average
            merged['price_gbp_mwh'] = (
                merged['price_gbp_mwh_da'] * day_ahead_weight +
                merged['price_gbp_mwh_ib'] * imbalance_weight
            )

        elif aggregation == 'max':
            merged['price_gbp_mwh'] = merged[['price_gbp_mwh_da', 'price_gbp_mwh_ib']].max(axis=1)

        elif aggregation == 'min':
            merged['price_gbp_mwh'] = merged[['price_gbp_mwh_da', 'price_gbp_mwh_ib']].min(axis=1)

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Create result DataFrame
        result = merged[['timestamp_utc', 'price_gbp_mwh']].copy()
        result['market_type'] = 'blended'

        print(f"✅ Blended {len(result)} price periods using {aggregation}")
        print(f"   Weights: day_ahead={day_ahead_weight}, imbalance={imbalance_weight}")

        return result

    def validate_prices(self, market_df: pd.DataFrame, market_constraints: dict) -> bool:
        """
        Validate prices against market constraints

        Args:
            market_df: Market price DataFrame
            market_constraints: Market constraints from config

        Returns:
            True if all prices are valid
        """
        print("\n🔍 Validating prices against market constraints...")

        uk_bm = market_constraints.get('uk_balancing_mechanism', {})
        price_caps = uk_bm.get('price_caps', {})

        max_price = price_caps.get('max_gbp_per_mwh', float('inf'))
        min_price = price_caps.get('min_gbp_per_mwh', float('-inf'))

        # Check for out-of-range prices
        too_high = market_df['price_gbp_mwh'] > max_price
        too_low = market_df['price_gbp_mwh'] < min_price

        if too_high.any():
            count = too_high.sum()
            max_found = market_df.loc[too_high, 'price_gbp_mwh'].max()
            print(f"⚠️  {count} prices exceed maximum (£{max_price}/MWh), highest: £{max_found:.2f}/MWh")
            return False

        if too_low.any():
            count = too_low.sum()
            min_found = market_df.loc[too_low, 'price_gbp_mwh'].min()
            print(f"⚠️  {count} prices below minimum (£{min_price}/MWh), lowest: £{min_found:.2f}/MWh")
            return False

        print(f"✅ All prices within valid range: £{min_price} to £{max_price}/MWh")
        print(f"   Price range in data: £{market_df['price_gbp_mwh'].min():.2f} to £{market_df['price_gbp_mwh'].max():.2f}/MWh")

        return True

    def get_price_statistics(self, market_df: pd.DataFrame) -> dict:
        """
        Calculate price statistics

        Args:
            market_df: Market price DataFrame

        Returns:
            Dictionary of price statistics
        """
        return {
            'mean_price': market_df['price_gbp_mwh'].mean(),
            'median_price': market_df['price_gbp_mwh'].median(),
            'min_price': market_df['price_gbp_mwh'].min(),
            'max_price': market_df['price_gbp_mwh'].max(),
            'std_price': market_df['price_gbp_mwh'].std(),
            'price_spread': market_df['price_gbp_mwh'].max() - market_df['price_gbp_mwh'].min()
        }


if __name__ == "__main__":
    print("Price Selector module ready")
    print("Run via CLI tool: python ingest_data.py")
