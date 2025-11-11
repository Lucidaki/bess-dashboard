"""
Unit tests for KPI Calculators
Tests Finance and O&M KPI calculation logic
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.finance_kpis import FinanceKPICalculator
from src.modules.om_kpis import OMKPICalculator
from src.config_loader import get_config_loader


class TestFinanceKPICalculator:
    """Test suite for Finance KPI Calculator"""

    def setup_method(self):
        """Setup test fixtures"""
        config_loader = get_config_loader()
        configs = config_loader.load_all_configs()
        self.asset_config = configs['config'].model_dump()['bess_assets']['UK_BESS_001']
        self.calculator = FinanceKPICalculator(self.asset_config)

        # Create mock optimization summary
        self.optimization_summary = {
            'asset_name': 'UK_BESS_001',
            'optimization_date': '2025-01-01',
            'optimal_revenue_gbp': 2000.0,
            'actual_revenue_gbp': -100.0,
            'revenue_variance_gbp': 2100.0,
            'cycles_used': 1.5,
            'discharge_energy_mwh': 10.0,
            'charge_energy_mwh': 11.0,
            'max_daily_cycles': 2.0,
            'duration_days': 1.0,
            'actual_performance': {
                'cycles_used': 0.8,
                'discharge_energy_mwh': 5.0,
                'n_periods': 48
            }
        }

    def test_market_capture_ratio_calculation(self):
        """Test market capture ratio KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'market_capture_ratio' in kpis
        mcr = kpis['market_capture_ratio']

        # MCR = (actual / optimal) * 100
        expected_mcr = (self.optimization_summary['actual_revenue_gbp'] /
                       self.optimization_summary['optimal_revenue_gbp'] * 100)

        assert abs(mcr - expected_mcr) < 0.1, \
            f"Market capture ratio should be {expected_mcr:.1f}%, got {mcr:.1f}%"

    def test_revenue_variance_calculation(self):
        """Test revenue variance KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'revenue_variance_gbp' in kpis
        assert 'revenue_variance_percent' in kpis

        variance_gbp = kpis['revenue_variance_gbp']
        variance_pct = kpis['revenue_variance_percent']

        expected_variance = self.optimization_summary['revenue_variance_gbp']
        assert abs(variance_gbp - expected_variance) < 0.01

    def test_lost_opportunity_calculation(self):
        """Test lost opportunity KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'lost_opportunity_gbp' in kpis
        lost_opp = kpis['lost_opportunity_gbp']

        # Lost opportunity = optimal - actual
        expected_lost = (self.optimization_summary['optimal_revenue_gbp'] -
                        self.optimization_summary['actual_revenue_gbp'])

        assert abs(lost_opp - expected_lost) < 0.01

    def test_revenue_per_cycle_calculation(self):
        """Test revenue per cycle KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'revenue_per_cycle_gbp' in kpis
        revenue_per_cycle = kpis['revenue_per_cycle_gbp']

        # Revenue per cycle = optimal revenue / cycles used
        cycles = self.optimization_summary['actual_performance']['cycles_used']
        if cycles > 0:
            expected_rpc = self.optimization_summary['actual_revenue_gbp'] / cycles
            assert abs(revenue_per_cycle - expected_rpc) < 0.01

    def test_finance_grade_calculation(self):
        """Test finance grade (A-F) assignment"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'finance_grade' in kpis
        grade = kpis['finance_grade']

        # Grade should be A-F
        assert grade in ['A', 'B', 'C', 'D', 'E', 'F'], \
            f"Finance grade should be A-F, got {grade}"

        # With negative actual revenue, should be F grade
        assert grade == 'F', "Negative revenue should result in F grade"

    def test_irr_impact_estimate(self):
        """Test IRR impact estimate KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'irr_impact_estimate_bps' in kpis
        irr_impact = kpis['irr_impact_estimate_bps']

        # Should be numeric
        assert isinstance(irr_impact, (int, float))


class TestOMKPICalculator:
    """Test suite for O&M KPI Calculator"""

    def setup_method(self):
        """Setup test fixtures"""
        config_loader = get_config_loader()
        configs = config_loader.load_all_configs()
        self.asset_config = configs['config'].model_dump()['bess_assets']['UK_BESS_001']
        self.calculator = OMKPICalculator(self.asset_config)

        # Create mock optimization summary
        self.optimization_summary = {
            'asset_name': 'UK_BESS_001',
            'optimization_date': '2025-01-01',
            'optimal_revenue_gbp': 2000.0,
            'actual_revenue_gbp': -100.0,
            'cycles_used': 1.5,
            'discharge_energy_mwh': 10.0,
            'max_daily_cycles': 2.0,
            'duration_days': 1.0,
            'actual_performance': {
                'cycles_used': 0.8,
                'discharge_energy_mwh': 5.0,
                'charge_energy_mwh': 5.5,
                'actual_rte_percent': 90.9,
                'power_min_mw': -7.0,
                'power_max_mw': 4.0,
                'soc_range_percent': 40.0,
                'n_periods': 48
            }
        }

    def test_availability_calculation(self):
        """Test availability KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'availability_percent' in kpis
        availability = kpis['availability_percent']

        # Default availability should be 100%
        assert availability == 100.0, "Default availability should be 100%"

    def test_cycle_utilization_calculation(self):
        """Test cycle utilization KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'cycle_utilization_percent' in kpis
        cycle_util = kpis['cycle_utilization_percent']

        # Cycle utilization = (actual cycles / max allowed) * 100
        actual_cycles = self.optimization_summary['actual_performance']['cycles_used']
        max_cycles = self.optimization_summary['max_daily_cycles'] * \
                    self.optimization_summary['duration_days']

        expected_util = (actual_cycles / max_cycles * 100)
        assert abs(cycle_util - expected_util) < 0.1

    def test_rte_performance(self):
        """Test RTE performance KPIs"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'actual_rte_percent' in kpis
        assert 'rated_rte_percent' in kpis
        assert 'rte_deviation_percent' in kpis

        actual_rte = kpis['actual_rte_percent']
        rated_rte = kpis['rated_rte_percent']
        deviation = kpis['rte_deviation_percent']

        # Check deviation calculation
        expected_deviation = actual_rte - rated_rte
        assert abs(deviation - expected_deviation) < 0.01

    def test_capacity_factor_calculation(self):
        """Test capacity factor KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'capacity_factor_percent' in kpis
        capacity_factor = kpis['capacity_factor_percent']

        # Capacity factor should be between 0-100%
        assert 0 <= capacity_factor <= 100, \
            f"Capacity factor should be 0-100%, got {capacity_factor}%"

    def test_power_utilization(self):
        """Test power utilization KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'power_utilization_percent' in kpis
        power_util = kpis['power_utilization_percent']

        # Power utilization = (peak power / rated power) * 100
        power_max_abs = max(
            abs(self.optimization_summary['actual_performance']['power_min_mw']),
            abs(self.optimization_summary['actual_performance']['power_max_mw'])
        )
        rated_power = self.asset_config['power_mw']
        expected_util = (power_max_abs / rated_power * 100)

        assert abs(power_util - expected_util) < 0.1

    def test_soc_range_utilization(self):
        """Test SoC range utilization KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'soc_range_utilization_percent' in kpis
        soc_util = kpis['soc_range_utilization_percent']

        # Should be between 0-100%
        assert 0 <= soc_util <= 100, \
            f"SoC range utilization should be 0-100%, got {soc_util}%"

    def test_om_grade_calculation(self):
        """Test O&M grade (A-F) assignment"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'om_grade' in kpis
        grade = kpis['om_grade']

        # Grade should be A-F
        assert grade in ['A', 'B', 'C', 'D', 'E', 'F'], \
            f"O&M grade should be A-F, got {grade}"

    def test_degradation_estimate(self):
        """Test degradation estimate KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'estimated_annual_degradation_percent' in kpis
        degradation = kpis['estimated_annual_degradation_percent']

        # Should be non-negative
        assert degradation >= 0, "Degradation should be non-negative"

    def test_avg_cycle_depth(self):
        """Test average cycle depth KPI"""
        kpis = self.calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        assert 'avg_cycle_depth_percent' in kpis
        cycle_depth = kpis['avg_cycle_depth_percent']

        # Should be between 0-100%
        assert 0 <= cycle_depth <= 100, \
            f"Cycle depth should be 0-100%, got {cycle_depth}%"


class TestKPIReportGeneration:
    """Test KPI report generation"""

    def setup_method(self):
        """Setup test fixtures"""
        config_loader = get_config_loader()
        configs = config_loader.load_all_configs()
        self.asset_config = configs['config'].model_dump()['bess_assets']['UK_BESS_001']

        self.optimization_summary = {
            'asset_name': 'UK_BESS_001',
            'optimization_date': '2025-01-01',
            'optimal_revenue_gbp': 2000.0,
            'actual_revenue_gbp': -100.0,
            'revenue_variance_gbp': 2100.0,
            'cycles_used': 1.5,
            'max_daily_cycles': 2.0,
            'duration_days': 1.0,
            'actual_performance': {
                'cycles_used': 0.8,
                'discharge_energy_mwh': 5.0,
                'charge_energy_mwh': 5.5,
                'actual_rte_percent': 90.9,
                'power_min_mw': -7.0,
                'power_max_mw': 4.0,
                'soc_range_percent': 40.0,
                'n_periods': 48
            }
        }

    def test_finance_report_generation(self):
        """Test finance report string generation"""
        calculator = FinanceKPICalculator(self.asset_config)
        kpis = calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        report = calculator.generate_finance_report(kpis, 'UK_BESS_001')

        # Check report contains key sections
        assert 'FINANCE KPI REPORT' in report
        assert 'UK_BESS_001' in report
        assert 'Market Capture' in report
        assert 'Revenue Performance' in report

    def test_om_report_generation(self):
        """Test O&M report string generation"""
        calculator = OMKPICalculator(self.asset_config)
        kpis = calculator.calculate_kpis(
            optimization_summary=self.optimization_summary,
            settlement_duration_min=30
        )

        report = calculator.generate_om_report(kpis, 'UK_BESS_001')

        # Check report contains key sections
        assert 'O&M KPI REPORT' in report
        assert 'UK_BESS_001' in report
        assert 'Availability' in report or 'AVAILABILITY' in report
        assert 'Cycle' in report or 'CYCLE' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
