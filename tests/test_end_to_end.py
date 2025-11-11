"""
Integration tests for end-to-end pipeline
Tests complete workflow: Data Ingestion → Optimization → KPIs → Visualization
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_config_loader
from src.data_processing.data_quality_scorer import DataQualityScorer
from src.optimization.bess_optimizer import BESSOptimizer
from src.modules.finance_kpis import FinanceKPICalculator
from src.modules.om_kpis import OMKPICalculator
from src.visualization.bess_charts import BESSVisualizer


class TestEndToEndPipeline:
    """Test complete end-to-end workflow"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config_loader = get_config_loader()
        self.configs = self.config_loader.load_all_configs()
        self.asset_name = 'UK_BESS_001'
        self.asset_config = self.configs['config'].bess_assets[self.asset_name]
        self.market_config = self.configs['config'].market

        # File paths
        self.scada_file = Path('data/canonical/scada_UK_BESS_001_2025-10-14.csv')
        self.market_file = Path('data/canonical/market_UK_BESS_001_2025-10-14.csv')

    def test_pipeline_data_loading(self):
        """Test Step 1: Data loading from canonical files"""
        # Check files exist
        assert self.scada_file.exists(), f"SCADA file not found: {self.scada_file}"
        assert self.market_file.exists(), f"Market file not found: {self.market_file}"

        # Load SCADA data
        scada_df = pd.read_csv(self.scada_file)
        assert not scada_df.empty, "SCADA data should not be empty"
        assert 'timestamp_utc' in scada_df.columns
        assert 'power_mw' in scada_df.columns
        assert 'soc_percent' in scada_df.columns

        # Load market data
        market_df = pd.read_csv(self.market_file)
        assert not market_df.empty, "Market data should not be empty"
        assert 'timestamp_utc' in market_df.columns
        assert 'price_gbp_mwh' in market_df.columns

    def test_pipeline_data_quality_check(self):
        """Test Step 2: Data quality validation"""
        # Load data
        scada_df = pd.read_csv(self.scada_file)
        scada_df['timestamp_utc'] = pd.to_datetime(scada_df['timestamp_utc'])

        # Check data quality
        scorer = DataQualityScorer(self.asset_config.model_dump())
        dq_results = scorer.score_data_quality(scada_df, data_type='scada')

        assert 'overall_score' in dq_results
        assert dq_results['overall_score'] >= 80, \
            f"Data quality should be ≥80%, got {dq_results['overall_score']:.1f}%"

    def test_pipeline_merge_datasets(self):
        """Test Step 3: Merging SCADA and market data"""
        # Load both datasets
        scada_df = pd.read_csv(self.scada_file)
        market_df = pd.read_csv(self.market_file)

        # Convert timestamps
        scada_df['timestamp_utc'] = pd.to_datetime(scada_df['timestamp_utc'])
        market_df['timestamp_utc'] = pd.to_datetime(market_df['timestamp_utc'])

        # Merge on timestamp
        merged_df = pd.merge(
            scada_df,
            market_df,
            on='timestamp_utc',
            how='inner'
        )

        assert not merged_df.empty, "Merged data should not be empty"
        assert 'power_mw' in merged_df.columns
        assert 'soc_percent' in merged_df.columns
        assert 'price_gbp_mwh' in merged_df.columns

    def test_pipeline_optimization(self):
        """Test Step 4: BESS optimization"""
        # Load and merge data
        scada_df = pd.read_csv(self.scada_file)
        market_df = pd.read_csv(self.market_file)

        scada_df['timestamp_utc'] = pd.to_datetime(scada_df['timestamp_utc'])
        market_df['timestamp_utc'] = pd.to_datetime(market_df['timestamp_utc'])

        schedule_df = pd.merge(scada_df, market_df, on='timestamp_utc', how='inner')
        schedule_df = schedule_df.rename(columns={
            'power_mw': 'actual_power_mw',
            'soc_percent': 'actual_soc_percent'
        })

        # Run optimization
        optimizer = BESSOptimizer(self.asset_config, self.market_config)
        initial_soc = schedule_df['actual_soc_percent'].iloc[0]
        result = optimizer.optimize(schedule_df, initial_soc_percent=initial_soc)

        # Verify optimization results
        assert result['status'] == 'Optimal', "Optimization should converge"
        assert 'optimal_revenue_gbp' in result
        assert 'schedule' in result
        assert 'cycles_used' in result

    def test_pipeline_kpi_calculation(self):
        """Test Step 5: KPI calculation"""
        # Load optimization results from actual run
        summary_file = Path('data/optimization_results/summary_UK_BESS_001_2025-10-14.json')

        if not summary_file.exists():
            pytest.skip(f"Optimization results not found: {summary_file}")

        with open(summary_file, 'r') as f:
            optimization_summary = json.load(f)

        # Calculate Finance KPIs
        asset_config_dict = self.asset_config.model_dump()
        finance_calc = FinanceKPICalculator(asset_config_dict)
        finance_kpis = finance_calc.calculate_kpis(
            optimization_summary=optimization_summary,
            settlement_duration_min=self.market_config.settlement_duration_min
        )

        # Verify Finance KPIs
        assert 'market_capture_ratio' in finance_kpis
        assert 'revenue_variance_gbp' in finance_kpis
        assert 'finance_grade' in finance_kpis

        # Calculate O&M KPIs
        om_calc = OMKPICalculator(asset_config_dict)
        om_kpis = om_calc.calculate_kpis(
            optimization_summary=optimization_summary,
            settlement_duration_min=self.market_config.settlement_duration_min
        )

        # Verify O&M KPIs
        assert 'availability_percent' in om_kpis
        assert 'cycle_utilization_percent' in om_kpis
        assert 'om_grade' in om_kpis

    def test_pipeline_visualization(self):
        """Test Step 6: Visualization generation"""
        # Load optimization results
        summary_file = Path('data/optimization_results/summary_UK_BESS_001_2025-10-14.json')
        schedule_file = Path('data/optimization_results/schedule_UK_BESS_001_2025-10-14.csv')

        if not summary_file.exists() or not schedule_file.exists():
            pytest.skip("Optimization results not found")

        with open(summary_file, 'r') as f:
            optimization_summary = json.load(f)

        schedule_df = pd.read_csv(schedule_file)
        schedule_df['timestamp_utc'] = pd.to_datetime(schedule_df['timestamp_utc'])

        # Create visualizer
        visualizer = BESSVisualizer()

        # Generate charts
        soc_limits = {
            'soc_min_percent': self.asset_config.constraints.soc_min_percent,
            'soc_max_percent': self.asset_config.constraints.soc_max_percent
        }

        # Test power profile chart
        fig = visualizer.create_power_profile_chart(schedule_df)
        assert fig is not None, "Power profile chart should be created"

        # Test SoC curve chart
        fig = visualizer.create_soc_curve_chart(schedule_df, soc_limits)
        assert fig is not None, "SoC curve chart should be created"

        # Test revenue comparison chart
        fig = visualizer.create_revenue_comparison_chart(optimization_summary)
        assert fig is not None, "Revenue comparison chart should be created"

    def test_complete_pipeline_execution(self):
        """Test full pipeline execution from start to finish"""
        # Step 1: Load data
        scada_df = pd.read_csv(self.scada_file)
        market_df = pd.read_csv(self.market_file)

        scada_df['timestamp_utc'] = pd.to_datetime(scada_df['timestamp_utc'])
        market_df['timestamp_utc'] = pd.to_datetime(market_df['timestamp_utc'])

        # Step 2: Merge data
        schedule_df = pd.merge(scada_df, market_df, on='timestamp_utc', how='inner')
        schedule_df = schedule_df.rename(columns={
            'power_mw': 'actual_power_mw',
            'soc_percent': 'actual_soc_percent'
        })

        # Step 3: Optimize
        optimizer = BESSOptimizer(self.asset_config, self.market_config)
        initial_soc = schedule_df['actual_soc_percent'].iloc[0]
        optimization_result = optimizer.optimize(schedule_df, initial_soc_percent=initial_soc)

        assert optimization_result['status'] == 'Optimal'

        # Step 4: Calculate KPIs
        asset_config_dict = self.asset_config.model_dump()

        # Build optimization summary
        optimization_summary = {
            'asset_name': self.asset_name,
            'optimization_date': schedule_df['timestamp_utc'].min().strftime('%Y-%m-%d'),
            'optimal_revenue_gbp': optimization_result['optimal_revenue_gbp'],
            'actual_revenue_gbp': 0.0,  # Would be calculated from actual schedule
            'revenue_variance_gbp': optimization_result['optimal_revenue_gbp'],
            'cycles_used': optimization_result['cycles_used'],
            'max_daily_cycles': self.asset_config.warranty.max_daily_cycles,
            'duration_days': 1.0,
            'actual_performance': {
                'cycles_used': 0.5,
                'discharge_energy_mwh': optimization_result['discharge_energy_mwh'] / 2,
                'charge_energy_mwh': optimization_result['charge_energy_mwh'] / 2,
                'actual_rte_percent': 90.0,
                'power_min_mw': schedule_df['actual_power_mw'].min(),
                'power_max_mw': schedule_df['actual_power_mw'].max(),
                'soc_range_percent': 40.0,
                'n_periods': len(schedule_df)
            }
        }

        finance_calc = FinanceKPICalculator(asset_config_dict)
        finance_kpis = finance_calc.calculate_kpis(
            optimization_summary=optimization_summary,
            settlement_duration_min=self.market_config.settlement_duration_min
        )

        om_calc = OMKPICalculator(asset_config_dict)
        om_kpis = om_calc.calculate_kpis(
            optimization_summary=optimization_summary,
            settlement_duration_min=self.market_config.settlement_duration_min
        )

        # Step 5: Verify complete pipeline
        assert optimization_result['status'] == 'Optimal'
        assert 'market_capture_ratio' in finance_kpis
        assert 'cycle_utilization_percent' in om_kpis

        print("\n" + "="*80)
        print("END-TO-END PIPELINE TEST RESULTS")
        print("="*80)
        print(f"Optimization Status: {optimization_result['status']}")
        print(f"Optimal Revenue: £{optimization_result['optimal_revenue_gbp']:,.2f}")
        print(f"Finance Grade: {finance_kpis['finance_grade']}")
        print(f"O&M Grade: {om_kpis['om_grade']}")
        print(f"Market Capture Ratio: {finance_kpis['market_capture_ratio']:.1f}%")
        print(f"Cycle Utilization: {om_kpis['cycle_utilization_percent']:.1f}%")
        print("="*80)


class TestAcceptanceCriteria:
    """Test acceptance criteria from config"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config_loader = get_config_loader()
        self.acceptance_criteria = self.config_loader.load_acceptance_criteria()

    def test_finance_acceptance_criteria(self):
        """Test that finance acceptance criteria are defined"""
        assert 'finance' in self.acceptance_criteria
        finance = self.acceptance_criteria['finance']

        # Check required criteria
        assert 'market_capture_ratio_min' in finance
        assert 'solve_time_max_seconds' in finance

    def test_om_acceptance_criteria(self):
        """Test that O&M acceptance criteria are defined"""
        assert 'om' in self.acceptance_criteria
        om = self.acceptance_criteria['om']

        # Check required criteria
        assert 'availability_min_percent' in om

    def test_data_quality_acceptance_criteria(self):
        """Test that data quality acceptance criteria are defined"""
        assert 'data_quality' in self.acceptance_criteria
        dq = self.acceptance_criteria['data_quality']

        # Check required criteria
        assert 'min_dq_score' in dq
        assert dq['min_dq_score'] >= 80, "Min DQ score should be ≥80%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
