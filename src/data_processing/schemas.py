"""
Canonical Data Schemas
Defines Pydantic models for validated data structures
"""

from datetime import datetime
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, validator


class SCADACanonical(BaseModel):
    """
    Canonical BESS SCADA data schema
    All SCADA data must conform to this structure after cleaning
    """
    timestamp_utc: datetime
    power_mw: float = Field(description="Power in MW (positive=discharge, negative=charge)")
    soc_percent: float = Field(ge=0, le=100, description="State of Charge percentage")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp_utc": "2024-01-01T00:00:00Z",
                "power_mw": -1.5,
                "soc_percent": 45.2
            }
        }


class MarketPriceCanonical(BaseModel):
    """
    Canonical market price data schema
    All market price data must conform to this structure after cleaning
    """
    timestamp_utc: datetime
    price_gbp_mwh: float = Field(gt=0, description="Price in GBP/MWh")
    market_type: Literal["day_ahead", "imbalance", "blended"] = Field(
        description="Type of market price"
    )
    source_priority: int = Field(default=1, description="Priority for price selection (1=highest)")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp_utc": "2024-01-01T00:00:00Z",
                "price_gbp_mwh": 45.50,
                "market_type": "day_ahead",
                "source_priority": 1
            }
        }


class DQComponent(BaseModel):
    """Individual data quality component score"""
    score: float = Field(ge=0, le=100, description="Score 0-100")
    passed: bool
    issues: List[str] = Field(default_factory=list)


class DQReport(BaseModel):
    """
    Data Quality Report
    Comprehensive data quality assessment
    """
    overall_score: float = Field(ge=0, le=100, description="Overall DQ score (weighted)")
    passed: bool = Field(description="True if DQ >= threshold")

    # Component scores
    completeness: DQComponent
    continuity: DQComponent
    bounds: DQComponent
    energy_reconciliation: DQComponent

    # Metadata
    total_periods: int
    valid_periods: int
    missing_periods: int
    out_of_bounds_periods: int

    # Remediation guidance
    remediation_required: List[str] = Field(default_factory=list)
    can_auto_remediate: bool = Field(default=False)

    # Timestamps
    start_time: datetime
    end_time: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 92.5,
                "passed": True,
                "completeness": {
                    "score": 98.0,
                    "passed": True,
                    "issues": []
                },
                "total_periods": 48,
                "valid_periods": 47,
                "remediation_required": []
            }
        }


class OptimizationResult(BaseModel):
    """
    Optimization result schema
    Output from BESS optimizer
    """
    asset_name: str
    optimization_date: datetime

    # Solver status
    solver_status: str
    solve_time_sec: float
    objective_value: float
    mip_gap: float

    # Schedules
    power_schedule_mw: List[float]
    soc_schedule_percent: List[float]
    timestamps: List[datetime]

    # Revenue
    actual_revenue_gbp: float
    optimal_revenue_gbp: float
    revenue_variance_gbp: float
    market_capture_ratio: float

    # Metadata
    data_quality_score: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "asset_name": "UK_BESS_001",
                "solver_status": "Optimal",
                "market_capture_ratio": 85.5
            }
        }


class KPIReport(BaseModel):
    """
    KPI Report schema
    Finance and O&M key performance indicators
    """
    asset_name: str
    report_date: datetime
    stakeholder: Literal["finance", "om", "both"]

    # Finance KPIs
    arbitrage_revenue_gbp: Optional[float] = None
    market_capture_ratio: Optional[float] = None
    revenue_variance_gbp: Optional[float] = None
    revenue_variance_percent: Optional[float] = None

    # O&M KPIs
    availability_percent: Optional[float] = None
    cycle_utilization_percent: Optional[float] = None
    actual_rte_percent: Optional[float] = None
    throughput_mwh: Optional[float] = None
    capacity_factor_percent: Optional[float] = None

    # Degradation tracking
    capacity_fade_actual_percent: Optional[float] = None
    capacity_fade_expected_percent: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "asset_name": "UK_BESS_001",
                "stakeholder": "finance",
                "market_capture_ratio": 85.5,
                "availability_percent": 98.2
            }
        }


# Type aliases for convenience
SCADADataFrame = List[SCADACanonical]
MarketDataFrame = List[MarketPriceCanonical]
