"""
KPI Calculator CLI Tool
Calculate Finance and O&M KPIs from optimization results

Usage:
    python calculate_kpis.py --summary-file data/optimization_results/summary_UK_BESS_001_2025-10-14.json --stakeholder both
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import get_config_loader
from src.modules.finance_kpis import FinanceKPICalculator
from src.modules.om_kpis import OMKPICalculator


def main():
    """Main KPI calculation workflow"""
    parser = argparse.ArgumentParser(
        description="Calculate Finance and O&M KPIs from optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate both finance and O&M KPIs
  python calculate_kpis.py --summary-file data/optimization_results/summary_UK_BESS_001_2025-10-14.json --stakeholder both

  # Calculate only finance KPIs
  python calculate_kpis.py --summary-file data/optimization_results/summary_UK_BESS_001_2025-10-14.json --stakeholder finance

  # Calculate with schedule data for detailed analysis
  python calculate_kpis.py --summary-file data/optimization_results/summary.json --schedule-file data/optimization_results/schedule.csv --stakeholder both
        """
    )

    parser.add_argument('--summary-file', required=True, help='Path to optimization summary JSON file')
    parser.add_argument('--schedule-file', help='Path to optimization schedule CSV file (optional)')
    parser.add_argument('--stakeholder', choices=['finance', 'om', 'both'], default='both', help='Stakeholder focus')
    parser.add_argument('--output', default='data/kpi_reports/', help='Output directory for KPI reports')

    args = parser.parse_args()

    print("="*80)
    print("KPI CALCULATOR - FINANCE & O&M METRICS")
    print("="*80)
    print(f"Summary File: {args.summary_file}")
    print(f"Schedule File: {args.schedule_file if args.schedule_file else 'Not provided'}")
    print(f"Stakeholder: {args.stakeholder.upper()}")
    print(f"Output Directory: {args.output}")
    print("="*80)

    try:
        # Load optimization summary
        print("\n📋 Loading optimization results...")
        with open(args.summary_file, 'r') as f:
            optimization_summary = json.load(f)

        asset_name = optimization_summary['asset_name']
        optimization_date = optimization_summary['optimization_date']
        print(f"✅ Loaded summary for {asset_name}")
        print(f"   Optimization Date: {optimization_date}")

        # Load schedule if provided
        schedule_df = None
        if args.schedule_file:
            schedule_df = pd.read_csv(args.schedule_file)
            schedule_df['timestamp_utc'] = pd.to_datetime(schedule_df['timestamp_utc'])
            print(f"✅ Loaded schedule: {len(schedule_df)} periods")

        # Load configuration
        print("\n📋 Loading configuration...")
        config_loader = get_config_loader()
        configs = config_loader.load_all_configs()
        config_dict = configs['config'].model_dump()

        # Get asset config
        asset_config = config_dict['bess_assets'].get(asset_name)
        if not asset_config:
            print(f"❌ Asset '{asset_name}' not found in configuration")
            return 1

        settlement_duration_min = config_dict['market']['settlement_duration_min']
        print(f"✅ Configuration loaded for {asset_name}")

        # Initialize calculators
        finance_calc = FinanceKPICalculator(asset_config)
        om_calc = OMKPICalculator(asset_config)

        # Calculate KPIs based on stakeholder
        finance_kpis = {}
        om_kpis = {}

        if args.stakeholder in ['finance', 'both']:
            print("\n" + "="*80)
            print("CALCULATING FINANCE KPIs")
            print("="*80)

            finance_kpis = finance_calc.calculate_kpis(
                optimization_summary=optimization_summary,
                settlement_duration_min=settlement_duration_min
            )

            # Add schedule-based KPIs if available
            if schedule_df is not None:
                schedule_kpis = finance_calc.calculate_schedule_based_kpis(schedule_df)
                finance_kpis.update(schedule_kpis)

            print(f"✅ Calculated {len(finance_kpis)} finance KPIs")

            # Display finance report
            finance_report = finance_calc.generate_finance_report(finance_kpis, asset_name)
            print("\n" + finance_report)

        if args.stakeholder in ['om', 'both']:
            print("\n" + "="*80)
            print("CALCULATING O&M KPIs")
            print("="*80)

            om_kpis = om_calc.calculate_kpis(
                optimization_summary=optimization_summary,
                schedule_df=schedule_df,
                settlement_duration_min=settlement_duration_min
            )

            print(f"✅ Calculated {len(om_kpis)} O&M KPIs")

            # Display O&M report
            om_report = om_calc.generate_om_report(om_kpis, asset_name)
            print("\n" + om_report)

        # Save KPI reports
        print("\n" + "="*80)
        print("SAVING KPI REPORTS")
        print("="*80)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save finance KPIs
        if finance_kpis:
            finance_csv = output_dir / f"finance_kpis_{asset_name}_{optimization_date}_{timestamp}.csv"
            finance_json = output_dir / f"finance_kpis_{asset_name}_{optimization_date}_{timestamp}.json"

            # CSV format
            finance_df = pd.DataFrame([finance_kpis])
            finance_df.to_csv(finance_csv, index=False)
            print(f"✅ Saved finance KPIs (CSV): {finance_csv}")

            # JSON format
            with open(finance_json, 'w') as f:
                json.dump(finance_kpis, f, indent=2, default=str)
            print(f"✅ Saved finance KPIs (JSON): {finance_json}")

        # Save O&M KPIs
        if om_kpis:
            om_csv = output_dir / f"om_kpis_{asset_name}_{optimization_date}_{timestamp}.csv"
            om_json = output_dir / f"om_kpis_{asset_name}_{optimization_date}_{timestamp}.json"

            # CSV format
            om_df = pd.DataFrame([om_kpis])
            om_df.to_csv(om_csv, index=False)
            print(f"✅ Saved O&M KPIs (CSV): {om_csv}")

            # JSON format
            with open(om_json, 'w') as f:
                json.dump(om_kpis, f, indent=2, default=str)
            print(f"✅ Saved O&M KPIs (JSON): {om_json}")

        # Save combined report
        if finance_kpis and om_kpis:
            combined_json = output_dir / f"kpis_combined_{asset_name}_{optimization_date}_{timestamp}.json"
            combined = {
                'asset_name': asset_name,
                'optimization_date': optimization_date,
                'report_timestamp': timestamp,
                'finance_kpis': finance_kpis,
                'om_kpis': om_kpis
            }
            with open(combined_json, 'w') as f:
                json.dump(combined, f, indent=2, default=str)
            print(f"✅ Saved combined KPIs (JSON): {combined_json}")

        # Final summary
        print("\n" + "="*80)
        print("KPI CALCULATION COMPLETE")
        print("="*80)

        if finance_kpis:
            print(f"\n📊 Finance Summary:")
            print(f"   Market Capture Ratio: {finance_kpis.get('market_capture_ratio', 0):.1f}%")
            print(f"   Revenue Variance: £{finance_kpis.get('revenue_variance_gbp', 0):,.2f}")
            print(f"   Finance Grade: {finance_kpis.get('finance_grade', 'N/A')}")

        if om_kpis:
            print(f"\n⚙️  O&M Summary:")
            print(f"   Availability: {om_kpis.get('availability_percent', 0):.1f}%")
            print(f"   Cycle Utilization: {om_kpis.get('cycle_utilization_percent', 0):.1f}%")
            print(f"   Actual RTE: {om_kpis.get('actual_rte_percent', 0):.1f}%")
            print(f"   O&M Grade: {om_kpis.get('om_grade', 'N/A')}")

        print(f"\n✅ All KPI reports saved to {output_dir}")
        print("="*80)

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")
        return 1
    except KeyError as e:
        print(f"\n❌ Missing key in optimization summary: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
