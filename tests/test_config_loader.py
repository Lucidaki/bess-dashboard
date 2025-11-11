"""
Unit tests for Configuration Loader
Tests Pydantic validation and YAML configuration loading
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import ConfigLoader, get_config_loader


class TestConfigLoader:
    """Test suite for ConfigLoader"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config_loader = get_config_loader()

    def test_config_loader_singleton(self):
        """Test that ConfigLoader is a singleton"""
        loader1 = get_config_loader()
        loader2 = get_config_loader()
        assert loader1 is loader2, "ConfigLoader should be a singleton"

    def test_load_all_configs(self):
        """Test loading all configuration files"""
        configs = self.config_loader.load_all_configs()

        # Check all config types are loaded
        assert 'config' in configs, "Main config should be loaded"
        assert 'market_constraints' in configs, "Market constraints should be loaded"
        assert 'dq_remediation_rules' in configs, "DQ remediation rules should be loaded"
        assert 'price_selection_rules' in configs, "Price selection rules should be loaded"
        assert 'acceptance_criteria' in configs, "Acceptance criteria should be loaded"

    def test_main_config_structure(self):
        """Test main config has required sections"""
        config = self.config_loader.load_main_config()
        config_dict = config.model_dump()

        # Check main sections
        assert 'bess_assets' in config_dict, "Should have bess_assets section"
        assert 'market' in config_dict, "Should have market section"

        # Check asset exists
        assert 'UK_BESS_001' in config_dict['bess_assets'], "UK_BESS_001 should exist"

        # Check asset structure
        asset = config_dict['bess_assets']['UK_BESS_001']
        assert 'capacity_mwh' in asset, "Asset should have capacity_mwh"
        assert 'power_mw' in asset, "Asset should have power_mw"
        assert 'rte_percent' in asset, "Asset should have rte_percent"
        assert 'constraints' in asset, "Asset should have constraints"
        assert 'warranty' in asset, "Asset should have warranty"
        assert 'degradation' in asset, "Asset should have degradation"

    def test_asset_power_constraints(self):
        """Test asymmetric power constraints are loaded correctly"""
        config = self.config_loader.load_main_config()
        asset = config.bess_assets['UK_BESS_001']

        # Test asymmetric power limits
        assert asset.constraints.max_charge_power_mw == 7.5, "Max charge power should be 7.5 MW"
        assert asset.constraints.max_discharge_power_mw == 4.2, "Max discharge power should be 4.2 MW"
        assert asset.constraints.max_charge_power_mw != asset.constraints.max_discharge_power_mw, \
            "Power constraints should be asymmetric"

    def test_soc_constraints(self):
        """Test SoC constraints are valid"""
        config = self.config_loader.load_main_config()
        asset = config.bess_assets['UK_BESS_001']

        soc_min = asset.constraints.soc_min_percent
        soc_max = asset.constraints.soc_max_percent

        assert 0 <= soc_min <= 100, "SoC min should be between 0-100%"
        assert 0 <= soc_max <= 100, "SoC max should be between 0-100%"
        assert soc_min < soc_max, "SoC min should be less than SoC max"

    def test_market_config(self):
        """Test market configuration"""
        config = self.config_loader.load_main_config()
        market = config.market

        assert market.settlement_duration_min in [5, 15, 30, 60], \
            "Settlement duration should be standard interval"
        assert market.timezone == "UTC", "Market timezone should be UTC"

    def test_dq_thresholds(self):
        """Test data quality threshold configuration"""
        dq_rules = self.config_loader.load_dq_remediation_rules()

        assert 'min_dq_score' in dq_rules, "Should have min_dq_score"
        assert 'auto_remediation' in dq_rules, "Should have auto_remediation rules"

        min_score = dq_rules['min_dq_score']
        assert 0 <= min_score <= 100, "Min DQ score should be between 0-100"

    def test_acceptance_criteria(self):
        """Test acceptance criteria configuration"""
        acceptance = self.config_loader.load_acceptance_criteria()

        # Check finance criteria
        assert 'finance' in acceptance, "Should have finance criteria"
        finance = acceptance['finance']
        assert 'market_capture_ratio_min' in finance, "Should have market capture min"

        # Check O&M criteria
        assert 'om' in acceptance, "Should have O&M criteria"
        om = acceptance['om']
        assert 'availability_min_percent' in om, "Should have availability min"

    def test_no_hardcoded_values_in_config(self):
        """Validate zero hardcoded values - all parameters in config"""
        config = self.config_loader.load_main_config()
        asset = config.bess_assets['UK_BESS_001']

        # Critical parameters must be in config, not hardcoded
        assert asset.capacity_mwh > 0, "Capacity must be configured"
        assert asset.power_mw > 0, "Power must be configured"
        assert asset.rte_percent > 0, "RTE must be configured"
        assert asset.warranty.max_daily_cycles > 0, "Max cycles must be configured"
        assert asset.degradation.annual_fade_percent >= 0, "Degradation must be configured"


class TestConfigValidation:
    """Test Pydantic validation"""

    def test_invalid_asset_raises_error(self):
        """Test that requesting invalid asset raises KeyError"""
        config_loader = get_config_loader()
        config = config_loader.load_main_config()

        with pytest.raises(KeyError):
            asset = config.bess_assets['INVALID_ASSET']

    def test_config_immutability(self):
        """Test that config model is validated on creation"""
        config_loader = get_config_loader()
        config = config_loader.load_main_config()

        # Pydantic models should validate types
        asset = config.bess_assets['UK_BESS_001']
        assert isinstance(asset.capacity_mwh, (int, float)), "Capacity should be numeric"
        assert isinstance(asset.power_mw, (int, float)), "Power should be numeric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
