"""
BESS Optimization CLI Tool
Run MILP optimization for arbitrage revenue maximization

Usage:
    python optimize_bess.py --scada-file data/canonical/scada_UK_BESS_001_2025-10-14.csv --market-file data/canonical/market_UK_BESS_001_2025-10-14.csv --asset UK_BESS_001
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
from src.optimization.bess_optimizer import BESSOptimizer


def main():
    """Main optimization workflow"""
    parser = argparse.ArgumentParser(
        description="Optimize BESS operation for arbitrage revenue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimization
  python optimize_bess.py --scada-file data/canonical/scada_UK_BESS_001_2025-10-14.csv --market-file data/canonical/market_UK_BESS_001_2025-10-14.csv --asset UK_BESS_001

  # With custom output directory
  python optimize_bess.py --scada-file data/canonical/scada.csv --market-file data/canonical/market.csv --asset UK_BESS_001 --output results/
        """
    )

    parser.add_argument('--scada-file', required=True, help='Path to canonical SCADA CSV file')
    parser.add_argument('--market-file', required=True, help='Path to canonical market price CSV file')
    parser.add_argument('--asset', required=True, help='Asset name (e.g., UK_BESS_001)')
    parser.add_argument('--output', default='data/optimization_results/', help='Output directory for results')
    parser.add_argument('--initial-soc', type=float, help='Initial SoC percentage (if not using SCADA value)')
    parser.add_argument('--solver', default='PULP_CBC_CMD', help='Solver name (default: PULP_CBC_CMD)')
    parser.add_argument('--timeout', type=int, default=30, help='Solver timeout in seconds')

    args = parser.parse_args()

    print("="*80)
    print("BESS OPTIMIZATION - ARBITRAGE MAXIMIZATION")
    print("="*80)
    print(f"Asset: {args.asset}")
    print(f"SCADA File: {args.scada_file}")
    print(f"Market File: {args.market_file}")
    print(f"Output Directory: {args.output}")
    print(f"Solver: {args.solver}")
    print(f"Timeout: {args.timeout}s")
    print("="*80)

    try:
        # Load configuration
        print("\n📋 Loading configuration...")
        config_loader = get_config_loader()
        configs = config_loader.load_all_configs()

        config_dict = configs['config'].model_dump()

        # Get asset config
        asset_config = config_dict['bess_assets'].get(args.asset)
        if not asset_config:
            print(f"❌ Asset '{args.asset}' not found in configuration")
            print(f"   Available assets: {list(config_dict['bess_assets'].keys())}")
            return 1

        settlement_duration_min = config_dict['market']['settlement_duration_min']

        print(f"✅ Configuration loaded")
        print(f"   Asset: {args.asset}")
        print(f"   Capacity: {asset_config['capacity_mwh']} MWh")
        print(f"   Power Export: {asset_config['constraints']['power_export_max_mw']} MW")
        print(f"   Power Import: {asset_config['constraints']['power_import_max_mw']} MW")
        print(f"   RTE: {asset_config['rte_percent']}%")
        print(f"   Max Daily Cycles: {asset_config['warranty']['max_daily_cycles']}")

        # Load data files
        print("\n" + "="*80)
        print("STEP 1: LOAD DATA FILES")
        print("="*80)

        scada_df = pd.read_csv(args.scada_file)
        scada_df['timestamp_utc'] = pd.to_datetime(scada_df['timestamp_utc'])
        print(f"✅ Loaded SCADA data: {len(scada_df)} periods")

        market_df = pd.read_csv(args.market_file)
        market_df['timestamp_utc'] = pd.to_datetime(market_df['timestamp_utc'])
        print(f"✅ Loaded market data: {len(market_df)} periods")

        # Data summary
        print(f"\n📊 Data Summary:")
        print(f"   Time Range: {scada_df['timestamp_utc'].min()} to {scada_df['timestamp_utc'].max()}")
        print(f"   Price Range: £{market_df['price_gbp_mwh'].min():.2f} - £{market_df['price_gbp_mwh'].max():.2f}/MWh")
        print(f"   Actual SoC Range: {scada_df['soc_percent'].min():.1f}% - {scada_df['soc_percent'].max():.1f}%")
        print(f"   Actual Power Range: {scada_df['power_mw'].min():.2f} - {scada_df['power_mw'].max():.2f} MW")

        # Initialize optimizer
        print("\n" + "="*80)
        print("STEP 2: INITIALIZE OPTIMIZER")
        print("="*80)

        optimizer = BESSOptimizer(
            asset_config=asset_config,
            settlement_duration_min=settlement_duration_min,
            solver_name=args.solver,
            solver_timeout_sec=args.timeout
        )
        print(f"✅ Optimizer initialized")
        print(f"   Solver: {args.solver}")
        print(f"   Settlement Period: {settlement_duration_min} minutes")

        # Calculate actual performance
        print("\n" + "="*80)
        print("STEP 3: CALCULATE ACTUAL PERFORMANCE")
        print("="*80)

        actual_perf = optimizer.calculate_actual_performance(scada_df, market_df)

        print(f"\n📊 Actual Performance:")
        print(f"   Revenue: £{actual_perf['actual_revenue_gbp']:.2f}")
        print(f"   Discharge Energy: {actual_perf['discharge_energy_mwh']:.2f} MWh")
        print(f"   Charge Energy: {actual_perf['charge_energy_mwh']:.2f} MWh")
        print(f"   Cycles Used: {actual_perf['cycles_used']:.2f}")
        print(f"   Actual RTE: {actual_perf['actual_rte_percent']:.1f}%")

        # Run optimization
        print("\n" + "="*80)
        print("STEP 4: RUN MILP OPTIMIZATION")
        print("="*80)
        print("⏳ Solving optimization problem...")

        result = optimizer.optimize(
            scada_df=scada_df,
            market_df=market_df,
            initial_soc_percent=args.initial_soc
        )

        print(f"✅ Optimization completed!")
        print(f"   Status: {result['solver_status']}")
        print(f"   Solve Time: {result['solve_time_sec']:.2f} seconds")

        # Results summary
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)

        print(f"\n💰 Revenue Comparison:")
        print(f"   Actual Revenue:    £{result['actual_revenue_gbp']:>10.2f}")
        print(f"   Optimal Revenue:   £{result['optimal_revenue_gbp']:>10.2f}")
        print(f"   Revenue Variance:  £{result['revenue_variance_gbp']:>10.2f} ({result['revenue_variance_gbp']/result['optimal_revenue_gbp']*100:+.1f}%)")
        print(f"   Market Capture:    {result['market_capture_ratio']:>10.1f}%")

        print(f"\n🔋 Operational Comparison:")
        print(f"   Optimal Cycles:    {result['cycles_used']:.2f} / {result['max_daily_cycles']} daily max")
        print(f"   Actual Cycles:     {actual_perf['cycles_used']:.2f}")
        print(f"   Duration:          {result['duration_days']:.1f} days")

        print(f"\n📊 SoC Statistics:")
        optimal_soc_min = min(result['optimal_soc_percent'])
        optimal_soc_max = max(result['optimal_soc_percent'])
        print(f"   Optimal SoC Range: {optimal_soc_min:.1f}% - {optimal_soc_max:.1f}%")
        print(f"   Actual SoC Range:  {actual_perf['soc_min_percent']:.1f}% - {actual_perf['soc_max_percent']:.1f}%")

        print(f"\n⚡ Power Statistics:")
        optimal_power_min = min(result['optimal_power_mw'])
        optimal_power_max = max(result['optimal_power_mw'])
        print(f"   Optimal Power Range: {optimal_power_min:.2f} - {optimal_power_max:.2f} MW")
        print(f"   Actual Power Range:  {actual_perf['power_min_mw']:.2f} - {actual_perf['power_max_mw']:.2f} MW")

        # Save results
        print("\n" + "="*80)
        print("STEP 5: SAVE RESULTS")
        print("="*80)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = scada_df['timestamp_utc'].min().strftime("%Y-%m-%d")

        # Save schedule CSV
        schedule_df = pd.DataFrame({
            'timestamp_utc': result['timestamps'],
            'optimal_power_mw': result['optimal_power_mw'],
            'optimal_soc_percent': result['optimal_soc_percent'],
            'actual_power_mw': result['actual_power_mw'],
            'actual_soc_percent': result['actual_soc_percent'],
            'price_gbp_mwh': result['prices_gbp_mwh']
        })

        schedule_file = output_dir / f"schedule_{args.asset}_{date_str}_{timestamp}.csv"
        schedule_df.to_csv(schedule_file, index=False)
        print(f"✅ Saved optimization schedule: {schedule_file}")

        # Save summary JSON
        summary = {
            'asset_name': args.asset,
            'optimization_date': date_str,
            'timestamp': timestamp,
            'solver_status': result['solver_status'],
            'solve_time_sec': result['solve_time_sec'],
            'optimal_revenue_gbp': result['optimal_revenue_gbp'],
            'actual_revenue_gbp': result['actual_revenue_gbp'],
            'revenue_variance_gbp': result['revenue_variance_gbp'],
            'market_capture_ratio': result['market_capture_ratio'],
            'cycles_used': result['cycles_used'],
            'max_daily_cycles': result['max_daily_cycles'],
            'duration_days': result['duration_days'],
            'actual_performance': actual_perf
        }

        summary_file = output_dir / f"summary_{args.asset}_{date_str}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✅ Saved optimization summary: {summary_file}")

        # Final summary
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"✅ Market Capture Ratio: {result['market_capture_ratio']:.1f}%")
        print(f"✅ Revenue Opportunity: £{result['revenue_variance_gbp']:.2f}")
        print(f"\nNext Steps:")
        print(f"  1. Review optimization schedule in {schedule_file}")
        print(f"  2. Analyze actual vs optimal performance")
        print(f"  3. Calculate KPIs: python calculate_kpis.py --summary-file {summary_file}")
        print("="*80)

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")
        return 1
    except RuntimeError as e:
        print(f"\n❌ Optimization error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
