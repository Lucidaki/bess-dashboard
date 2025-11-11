"""
Chart Generation CLI Tool
Generate Plotly visualizations from optimization results

Usage:
    python generate_charts.py --summary-file data/optimization_results/summary.json --schedule-file data/optimization_results/schedule.csv --output data/charts/
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
from src.visualization.bess_charts import BESSVisualizer


def main():
    """Main chart generation workflow"""
    parser = argparse.ArgumentParser(
        description="Generate Plotly visualizations from BESS optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all charts
  python generate_charts.py --summary-file data/optimization_results/summary.json --schedule-file data/optimization_results/schedule.csv

  # Generate specific chart types
  python generate_charts.py --summary-file data/optimization_results/summary.json --schedule-file data/optimization_results/schedule.csv --charts power soc price

  # Save as PNG instead of HTML
  python generate_charts.py --summary-file data/optimization_results/summary.json --schedule-file data/optimization_results/schedule.csv --format png
        """
    )

    parser.add_argument('--summary-file', required=True, help='Path to optimization summary JSON file')
    parser.add_argument('--schedule-file', required=True, help='Path to optimization schedule CSV file')
    parser.add_argument('--output', default='data/charts/', help='Output directory for charts')
    parser.add_argument('--format', choices=['html', 'png', 'svg', 'json'], default='html', help='Output format')
    parser.add_argument('--charts', nargs='+',
                       choices=['power', 'soc', 'price', 'revenue', 'gauge', 'cycles', 'dashboard', 'all'],
                       default=['all'],
                       help='Chart types to generate')
    parser.add_argument('--theme', default='plotly_white', help='Plotly theme')

    args = parser.parse_args()

    print("="*80)
    print("CHART GENERATION - BESS OPTIMIZATION VISUALIZATIONS")
    print("="*80)
    print(f"Summary File: {args.summary_file}")
    print(f"Schedule File: {args.schedule_file}")
    print(f"Output Directory: {args.output}")
    print(f"Output Format: {args.format}")
    print(f"Charts: {', '.join(args.charts)}")
    print("="*80)

    try:
        # Load optimization summary
        print("\n📋 Loading optimization results...")
        with open(args.summary_file, 'r') as f:
            optimization_summary = json.load(f)

        asset_name = optimization_summary['asset_name']
        optimization_date = optimization_summary['optimization_date']
        print(f"✅ Loaded summary for {asset_name}")

        # Load schedule
        schedule_df = pd.read_csv(args.schedule_file)
        schedule_df['timestamp_utc'] = pd.to_datetime(schedule_df['timestamp_utc'])
        print(f"✅ Loaded schedule: {len(schedule_df)} periods")

        # Load configuration for SoC limits
        print("\n📋 Loading configuration...")
        config_loader = get_config_loader()
        configs = config_loader.load_all_configs()
        config_dict = configs['config'].model_dump()

        asset_config = config_dict['bess_assets'].get(asset_name)
        if not asset_config:
            print(f"❌ Asset '{asset_name}' not found in configuration")
            return 1

        soc_limits = {
            'soc_min_percent': asset_config['constraints']['soc_min_percent'],
            'soc_max_percent': asset_config['constraints']['soc_max_percent']
        }
        print(f"✅ Configuration loaded")

        # Initialize visualizer
        print("\n📊 Initializing visualizer...")
        visualizer = BESSVisualizer(theme=args.theme)
        print(f"✅ Visualizer initialized (theme: {args.theme})")

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine which charts to generate
        chart_types = args.charts
        if 'all' in chart_types:
            chart_types = ['power', 'soc', 'price', 'revenue', 'gauge', 'cycles', 'dashboard']

        charts_generated = []

        print("\n" + "="*80)
        print("GENERATING CHARTS")
        print("="*80)

        # 1. Power Profile Chart
        if 'power' in chart_types:
            print("\n📊 Generating power profile chart...")
            fig = visualizer.create_power_profile_chart(schedule_df)
            filename = output_dir / f"power_profile_{asset_name}_{optimization_date}_{timestamp}.{args.format}"
            visualizer.save_chart(fig, str(filename), args.format)
            print(f"✅ Saved: {filename}")
            charts_generated.append(('Power Profile', str(filename)))

        # 2. SoC Curve Chart
        if 'soc' in chart_types:
            print("\n📊 Generating SoC curve chart...")
            fig = visualizer.create_soc_curve_chart(schedule_df, soc_limits)
            filename = output_dir / f"soc_curve_{asset_name}_{optimization_date}_{timestamp}.{args.format}"
            visualizer.save_chart(fig, str(filename), args.format)
            print(f"✅ Saved: {filename}")
            charts_generated.append(('SoC Curve', str(filename)))

        # 3. Price Spread Chart
        if 'price' in chart_types:
            print("\n📊 Generating price spread chart...")
            fig = visualizer.create_price_spread_chart(schedule_df)
            filename = output_dir / f"price_spread_{asset_name}_{optimization_date}_{timestamp}.{args.format}"
            visualizer.save_chart(fig, str(filename), args.format)
            print(f"✅ Saved: {filename}")
            charts_generated.append(('Price Spread', str(filename)))

        # 4. Revenue Comparison Chart
        if 'revenue' in chart_types:
            print("\n📊 Generating revenue comparison chart...")
            fig = visualizer.create_revenue_comparison_chart(optimization_summary)
            filename = output_dir / f"revenue_comparison_{asset_name}_{optimization_date}_{timestamp}.{args.format}"
            visualizer.save_chart(fig, str(filename), args.format)
            print(f"✅ Saved: {filename}")
            charts_generated.append(('Revenue Comparison', str(filename)))

        # 5. Market Capture Gauge
        if 'gauge' in chart_types:
            print("\n📊 Generating market capture gauge...")
            market_capture = optimization_summary['market_capture_ratio']
            fig = visualizer.create_market_capture_gauge(market_capture)
            filename = output_dir / f"market_capture_gauge_{asset_name}_{optimization_date}_{timestamp}.{args.format}"
            visualizer.save_chart(fig, str(filename), args.format)
            print(f"✅ Saved: {filename}")
            charts_generated.append(('Market Capture Gauge', str(filename)))

        # 6. Cycle Utilization Chart
        if 'cycles' in chart_types:
            print("\n📊 Generating cycle utilization chart...")
            fig = visualizer.create_cycle_utilization_chart(optimization_summary)
            filename = output_dir / f"cycle_utilization_{asset_name}_{optimization_date}_{timestamp}.{args.format}"
            visualizer.save_chart(fig, str(filename), args.format)
            print(f"✅ Saved: {filename}")
            charts_generated.append(('Cycle Utilization', str(filename)))

        # 7. Dashboard Summary
        if 'dashboard' in chart_types:
            print("\n📊 Generating dashboard summary...")
            fig = visualizer.create_dashboard_summary(schedule_df, optimization_summary, soc_limits)
            filename = output_dir / f"dashboard_{asset_name}_{optimization_date}_{timestamp}.{args.format}"
            visualizer.save_chart(fig, str(filename), args.format)
            print(f"✅ Saved: {filename}")
            charts_generated.append(('Dashboard Summary', str(filename)))

        # Summary
        print("\n" + "="*80)
        print("CHART GENERATION COMPLETE")
        print("="*80)
        print(f"\n✅ Generated {len(charts_generated)} chart(s):")
        for chart_name, chart_path in charts_generated:
            print(f"   • {chart_name}: {chart_path}")

        print(f"\n📁 All charts saved to: {output_dir}")
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
