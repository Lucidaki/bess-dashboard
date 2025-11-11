"""
Data Ingestion CLI Tool
Main entry point for SCADA and market data processing

Usage:
    python ingest_data.py --scada data/raw/Scada_csv.csv --market data/raw/Market_price_csv.csv --asset UK_BESS_001
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import get_config_loader
from src.data_processing.csv_loader import CSVLoader
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.price_selector import PriceSelector
from src.data_processing.data_quality_scorer import DataQualityScorer
from src.data_processing.remediation_engine import RemediationEngine
from src.data_processing.energy_reconciliation import EnergyReconciliation


def main():
    """Main data ingestion workflow"""
    parser = argparse.ArgumentParser(
        description="Ingest and process SCADA and market price data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ingestion
  python ingest_data.py --scada data/raw/Scada_csv.csv --market data/raw/Market_price_csv.csv --asset UK_BESS_001

  # With remediation
  python ingest_data.py --scada data/raw/Scada_csv.csv --market data/raw/Market_price_csv.csv --asset UK_BESS_001 --remediate

  # Custom output directory
  python ingest_data.py --scada data/raw/Scada_csv.csv --market data/raw/Market_price_csv.csv --asset UK_BESS_001 --output data/canonical/
        """
    )

    parser.add_argument('--scada', required=True, help='Path to SCADA CSV file')
    parser.add_argument('--market', required=True, help='Path to market price CSV file')
    parser.add_argument('--asset', required=True, help='Asset name (e.g., UK_BESS_001)')
    parser.add_argument('--output', default='data/canonical/', help='Output directory for cleaned data')
    parser.add_argument('--remediate', action='store_true', help='Apply automatic data remediation')
    parser.add_argument('--max-iterations', type=int, default=3, help='Max remediation iterations')

    args = parser.parse_args()

    print("="*80)
    print("BESS DASHBOARD - DATA INGESTION")
    print("="*80)
    print(f"Asset: {args.asset}")
    print(f"SCADA File: {args.scada}")
    print(f"Market File: {args.market}")
    print(f"Output Directory: {args.output}")
    print(f"Remediation: {'Enabled' if args.remediate else 'Disabled'}")
    print("="*80)

    try:
        # Load configuration
        print("\n📋 Loading configuration...")
        config_loader = get_config_loader()
        configs = config_loader.load_all_configs()

        config_dict = configs['config'].model_dump()
        dq_rules = configs['dq_rules']
        price_rules = configs['price_rules']
        market_constraints = configs['market_constraints']

        # Get asset config
        asset_config = config_dict['bess_assets'].get(args.asset)
        if not asset_config:
            print(f"❌ Asset '{args.asset}' not found in configuration")
            print(f"   Available assets: {list(config_dict['bess_assets'].keys())}")
            return 1

        print(f"✅ Configuration loaded")
        print(f"   Asset: {args.asset}")
        print(f"   Capacity: {asset_config['capacity_mwh']} MWh")
        print(f"   Power: {asset_config['power_mw']} MW")
        print(f"   RTE: {asset_config['rte_percent']}%")

        # Initialize processors
        csv_loader = CSVLoader(config_dict)
        data_cleaner = DataCleaner(config_dict, dq_rules)
        price_selector = PriceSelector(price_rules)
        dq_scorer = DataQualityScorer(config_dict, dq_rules, asset_config)
        settlement_duration_min = config_dict['market']['settlement_duration_min']
        remediation_engine = RemediationEngine(dq_rules, settlement_duration_min)
        energy_recon = EnergyReconciliation(
            capacity_mwh=asset_config['capacity_mwh'],
            tolerance_percent=dq_rules['remediation_policies']['scada']['energy_reconciliation']['tolerance_percent']
        )

        # STEP 1: Load CSV files
        print("\n" + "="*80)
        print("STEP 1: LOAD CSV FILES")
        print("="*80)

        scada_df = csv_loader.load_scada_csv(args.scada)
        print(f"✅ Loaded {len(scada_df)} SCADA records")

        market_df = csv_loader.load_market_csv(args.market)
        print(f"✅ Loaded {len(market_df)} market price records")

        # Validate data types
        scada_valid, scada_issues = csv_loader.validate_data_types(scada_df, 'scada')
        if not scada_valid:
            print(f"❌ SCADA validation issues: {scada_issues}")
            return 1

        market_valid, market_issues = csv_loader.validate_data_types(market_df, 'market')
        if not market_valid:
            print(f"❌ Market validation issues: {market_issues}")
            return 1

        # STEP 2: Clean and resample data
        print("\n" + "="*80)
        print("STEP 2: CLEAN AND RESAMPLE DATA")
        print("="*80)

        # Resample SCADA to settlement period
        scada_resampled = data_cleaner.resample_scada(scada_df)

        # Remove duplicates
        scada_clean = data_cleaner.remove_duplicates(scada_resampled)
        market_clean = data_cleaner.remove_duplicates(market_df)

        # Align timestamps
        scada_aligned, market_aligned = data_cleaner.align_timestamps(scada_clean, market_clean)

        # STEP 3: Select optimization prices
        print("\n" + "="*80)
        print("STEP 3: SELECT OPTIMIZATION PRICES")
        print("="*80)

        market_selected = price_selector.select_optimization_prices(market_aligned)

        # Validate prices
        prices_valid = price_selector.validate_prices(market_selected, market_constraints)
        if not prices_valid:
            print("⚠️  Some prices outside market constraints")

        # Get price statistics
        price_stats = price_selector.get_price_statistics(market_selected)
        print(f"\n💰 Price Statistics:")
        print(f"   Mean:   £{price_stats['mean_price']:.2f}/MWh")
        print(f"   Median: £{price_stats['median_price']:.2f}/MWh")
        print(f"   Range:  £{price_stats['min_price']:.2f} - £{price_stats['max_price']:.2f}/MWh")
        print(f"   Spread: £{price_stats['price_spread']:.2f}/MWh")

        # STEP 4: Calculate data quality scores
        print("\n" + "="*80)
        print("STEP 4: DATA QUALITY ASSESSMENT")
        print("="*80)

        scada_dq_report = dq_scorer.score_scada(scada_aligned)
        market_dq_report = dq_scorer.score_market(market_selected)

        # Check if both passed
        both_passed = scada_dq_report.passed and market_dq_report.passed

        # Re-ingestion workflow with max iterations
        if not both_passed and args.remediate:
            print("\n🔧 DATA QUALITY CHECK FAILED - STARTING REMEDIATION WORKFLOW")
            print(f"   SCADA DQ: {scada_dq_report.overall_score:.1f}% {'✅' if scada_dq_report.passed else '❌'}")
            print(f"   Market DQ: {market_dq_report.overall_score:.1f}% {'✅' if market_dq_report.passed else '❌'}")
            print(f"   Max iterations: {args.max_iterations}")
            print()

            max_iterations = args.max_iterations
            iteration = 0

            while iteration < max_iterations and not both_passed:
                iteration += 1
                print(f"{'='*80}")
                print(f"REMEDIATION ITERATION {iteration}/{max_iterations}")
                print(f"{'='*80}")

                # Apply SCADA remediation if needed
                if not scada_dq_report.passed and scada_dq_report.can_auto_remediate:
                    print("\n🔧 Applying SCADA remediation...")
                    scada_aligned, scada_success, scada_log = remediation_engine.remediate_scada(
                        scada_aligned.set_index('timestamp_utc') if 'timestamp_utc' in scada_aligned.columns else scada_aligned,
                        scada_dq_report
                    )
                    scada_aligned = scada_aligned.reset_index()

                    # Print remediation log
                    for log_entry in scada_log:
                        print(f"   {log_entry}")

                    if scada_success:
                        # Re-score after remediation
                        scada_dq_report = dq_scorer.score_scada(scada_aligned)
                        print(f"\n✅ SCADA DQ after remediation: {scada_dq_report.overall_score:.1f}% {'✅' if scada_dq_report.passed else '❌'}")
                    else:
                        print(f"\n❌ SCADA remediation failed")
                elif not scada_dq_report.passed:
                    print(f"\n⚠️  SCADA DQ failed but cannot auto-remediate")

                # Apply market remediation if needed
                if not market_dq_report.passed and market_dq_report.can_auto_remediate:
                    print("\n🔧 Applying market remediation...")
                    market_selected, market_success, market_log = remediation_engine.remediate_market(
                        market_selected.set_index('timestamp_utc') if 'timestamp_utc' in market_selected.columns else market_selected,
                        market_dq_report
                    )
                    market_selected = market_selected.reset_index()

                    # Print remediation log
                    for log_entry in market_log:
                        print(f"   {log_entry}")

                    if market_success:
                        # Re-score after remediation
                        market_dq_report = dq_scorer.score_market(market_selected)
                        print(f"\n✅ Market DQ after remediation: {market_dq_report.overall_score:.1f}% {'✅' if market_dq_report.passed else '❌'}")
                    else:
                        print(f"\n❌ Market remediation failed")
                elif not market_dq_report.passed:
                    print(f"\n⚠️  Market DQ failed but cannot auto-remediate")

                # Check if both passed after this iteration
                both_passed = scada_dq_report.passed and market_dq_report.passed

                if both_passed:
                    print(f"\n✅ Remediation successful after {iteration} iteration(s)!")
                    break
                elif iteration == max_iterations:
                    print(f"\n❌ Maximum remediation iterations ({max_iterations}) exceeded")
                    print(f"   Manual intervention required")
                    return 1

        elif not both_passed:
            print("\n❌ DATA QUALITY CHECK FAILED")
            print(f"   SCADA DQ: {scada_dq_report.overall_score:.1f}% {'✅' if scada_dq_report.passed else '❌'}")
            print(f"   Market DQ: {market_dq_report.overall_score:.1f}% {'✅' if market_dq_report.passed else '❌'}")
            print("\n⚠️  Remediation not enabled.")
            print("   Run with --remediate flag to apply automatic fixes.")
            return 1

        # STEP 5: Save canonical data
        print("\n" + "="*80)
        print("STEP 5: SAVE CANONICAL DATA")
        print("="*80)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = scada_aligned['timestamp_utc'].min().strftime("%Y-%m-%d")

        scada_output = output_dir / f"scada_{args.asset}_{date_str}_{timestamp}.csv"
        market_output = output_dir / f"market_{args.asset}_{date_str}_{timestamp}.csv"

        # Save to CSV
        scada_aligned.to_csv(scada_output, index=False)
        market_selected.to_csv(market_output, index=False)

        print(f"✅ Saved canonical SCADA data: {scada_output}")
        print(f"   {len(scada_aligned)} records")

        print(f"✅ Saved canonical market data: {market_output}")
        print(f"   {len(market_selected)} records")

        # STEP 6: Summary
        print("\n" + "="*80)
        print("INGESTION SUMMARY")
        print("="*80)
        print(f"✅ Data ingestion completed successfully!")
        print(f"\nData Quality:")
        print(f"  SCADA:  {scada_dq_report.overall_score:.1f}% ✅")
        print(f"  Market: {market_dq_report.overall_score:.1f}% ✅")
        print(f"\nData Range:")
        print(f"  Start: {scada_aligned['timestamp_utc'].min()}")
        print(f"  End:   {scada_aligned['timestamp_utc'].max()}")
        print(f"  Periods: {len(scada_aligned)}")
        print(f"\nNext Steps:")
        print(f"  1. Review canonical data in {output_dir}")
        print(f"  2. Run optimization: python optimize_bess.py --scada-file {scada_output} --market-file {market_output}")
        print("="*80)

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
