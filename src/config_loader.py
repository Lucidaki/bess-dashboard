"""
Configuration Loader with Pydantic Validation
Loads and validates all YAML configuration files
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from pydantic import BaseModel, Field, validator


class BESSAssetConfig(BaseModel):
    """BESS Asset Configuration"""
    capacity_mwh: float = Field(gt=0, description="Total energy capacity in MWh")
    power_mw: float = Field(gt=0, description="Maximum charge/discharge power in MW")
    rte_percent: float = Field(gt=0, le=100, description="Round-trip efficiency percentage")

    class ConstraintsConfig(BaseModel):
        soc_min_percent: float = Field(ge=0, le=100)
        soc_max_percent: float = Field(ge=0, le=100)
        power_export_max_mw: float = Field(gt=0, description="Maximum export/discharge power (MW)")
        power_import_max_mw: float = Field(gt=0, description="Maximum import/charge power (MW)")

        @validator('soc_max_percent')
        def max_greater_than_min(cls, v, values):
            if 'soc_min_percent' in values and v <= values['soc_min_percent']:
                raise ValueError('soc_max_percent must be greater than soc_min_percent')
            return v

    class DegradationConfig(BaseModel):
        annual_fade_percent: float = Field(ge=0, le=100)
        model: str = Field(pattern="^(linear|piecewise)$")

    class WarrantyConfig(BaseModel):
        max_lifetime_cycles: float = Field(gt=0)
        max_daily_cycles: float = Field(gt=0)

    class GridCodeConfig(BaseModel):
        frequency_response_hold_min: Optional[float] = None
        ramp_rate_mw_per_min: Optional[float] = None

    constraints: ConstraintsConfig
    degradation: DegradationConfig
    warranty: WarrantyConfig
    grid_code: GridCodeConfig


class MarketConfig(BaseModel):
    """Market Configuration"""
    name: str
    settlement_duration_min: int = Field(gt=0)
    timezone: str
    currency: str
    data_source: str = Field(pattern="^(csv|api)$")


class DataQualityConfig(BaseModel):
    """Data Quality Configuration"""
    min_dq_score: float = Field(ge=0, le=100)
    remediation: str = Field(pattern="^(interpolate|reject)$")

    class WeightsConfig(BaseModel):
        completeness: float = Field(ge=0, le=1)
        continuity: float = Field(ge=0, le=1)
        bounds: float = Field(ge=0, le=1)
        energy_reconciliation: float = Field(ge=0, le=1)

        @validator('energy_reconciliation')
        def weights_sum_to_one(cls, v, values):
            total = v + sum(values.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating point error
                raise ValueError(f'Weights must sum to 1.0, got {total}')
            return v

    weights: WeightsConfig


class OptimizationConfig(BaseModel):
    """Optimization Configuration"""
    solver: str
    time_limit_sec: int = Field(gt=0)
    mip_gap_tolerance: float = Field(gt=0, le=1)
    objective: str


class VisualizationConfig(BaseModel):
    """Visualization Configuration"""
    theme: str
    color_palette: Dict[str, str]
    chart_config: Dict[str, Any]


class ProjectConfig(BaseModel):
    """Main Project Configuration"""
    name: str
    version: str
    description: str


class ConfigSchema(BaseModel):
    """Complete Configuration Schema"""
    project: ProjectConfig
    market: MarketConfig
    bess_assets: Dict[str, BESSAssetConfig]
    data_quality: DataQualityConfig
    optimization: OptimizationConfig
    visualization: VisualizationConfig


class ConfigLoader:
    """
    Load and validate all configuration files
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config = None
        self.market_constraints = None
        self.dq_rules = None
        self.price_rules = None
        self.acceptance_criteria = None

    @staticmethod
    def load_yaml(file_path: Path) -> Dict[str, Any]:
        """Load YAML file and return as dictionary"""
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load and validate all configuration files

        Returns:
            Dictionary containing all validated configurations
        """
        # Load main config with Pydantic validation
        config_path = self.config_dir / "config_schema.yaml"
        config_dict = self.load_yaml(config_path)
        self.config = ConfigSchema(**config_dict)

        # Load other configs (validation can be added later)
        self.market_constraints = self.load_yaml(
            self.config_dir / "market_constraints.yaml"
        )
        self.dq_rules = self.load_yaml(
            self.config_dir / "dq_remediation_rules.yaml"
        )
        self.price_rules = self.load_yaml(
            self.config_dir / "price_selection_rules.yaml"
        )
        self.acceptance_criteria = self.load_yaml(
            self.config_dir / "acceptance_criteria.yaml"
        )

        return {
            'config': self.config,
            'market_constraints': self.market_constraints,
            'dq_rules': self.dq_rules,
            'price_rules': self.price_rules,
            'acceptance_criteria': self.acceptance_criteria
        }

    def get_asset_config(self, asset_name: str) -> BESSAssetConfig:
        """Get configuration for a specific BESS asset"""
        if self.config is None:
            self.load_all_configs()

        if asset_name not in self.config.bess_assets:
            raise ValueError(f"Asset '{asset_name}' not found in configuration")

        return self.config.bess_assets[asset_name]

    def get_market_config(self) -> MarketConfig:
        """Get market configuration"""
        if self.config is None:
            self.load_all_configs()

        return self.config.market

    def validate_config(self) -> bool:
        """
        Validate all configurations

        Returns:
            True if all validations pass

        Raises:
            ValidationError if any validation fails
        """
        try:
            self.load_all_configs()
            print("✅ Configuration validation passed")
            return True
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            raise


# Singleton instance
_config_loader = None


def get_config_loader(config_dir: str = "config") -> ConfigLoader:
    """Get or create singleton ConfigLoader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration loader...")
    loader = ConfigLoader()

    try:
        configs = loader.load_all_configs()
        print(f"✅ Loaded {len(configs)} configuration files")
        print(f"   - Project: {loader.config.project.name} v{loader.config.project.version}")
        print(f"   - Market: {loader.config.market.name}")
        print(f"   - Assets: {list(loader.config.bess_assets.keys())}")
        print(f"   - DQ threshold: {loader.config.data_quality.min_dq_score}%")
        print(f"   - Solver: {loader.config.optimization.solver}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
