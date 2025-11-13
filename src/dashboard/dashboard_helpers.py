"""
Dashboard Helper Functions for Streamlit Integration

This module provides wrapper functions that integrate the CLI workflow
(data ingestion, optimization) into the Streamlit dashboard interface.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.csv_loader import CSVLoader
from data_processing.data_cleaner import DataCleaner
from data_processing.price_selector import PriceSelector
from data_processing.data_quality_scorer import DataQualityScorer
from data_processing.remediation_engine import RemediationEngine
from optimization.bess_optimizer import BESSOptimizer
from config_loader import ConfigLoader


@dataclass
class IngestionResult:
    """Result of data ingestion pipeline"""
    success: bool
    scada_canonical_path: Optional[Path] = None
    market_canonical_path: Optional[Path] = None
    scada_dq_score: Optional[float] = None
    market_dq_score: Optional[float] = None
    scada_dq_report: Optional[Dict] = None
    market_dq_report: Optional[Dict] = None
    remediation_applied: bool = False
    remediation_messages: list = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.remediation_messages is None:
            self.remediation_messages = []


@dataclass
class OptimizationResult:
    """Result of BESS optimization"""
    success: bool
    schedule_df: Optional[pd.DataFrame] = None
    summary: Optional[Dict] = None
    solve_time: Optional[float] = None
    solver_status: Optional[str] = None
    error_message: Optional[str] = None


def save_uploaded_file(uploaded_file, directory: str = "data/raw") -> Path:
    """
    Save Streamlit uploaded file to disk

    Args:
        uploaded_file: Streamlit UploadedFile object
        directory: Directory to save file (default: data/raw)

    Returns:
        Path object pointing to saved file
    """
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def validate_csv_structure(file_path: Path, csv_type: str) -> Tuple[bool, str]:
    """
    Validate CSV file structure before processing

    Args:
        file_path: Path to CSV file
        csv_type: Either 'scada' or 'market'

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        df = pd.read_csv(file_path, nrows=5, encoding='utf-8-sig')  # Just check header

        # Normalize column names (lowercase, strip whitespace) - matches CSVLoader behavior
        df.columns = df.columns.str.strip().str.lower()

        if csv_type == 'scada':
            # CSVLoader expects 'timestamp' (not 'timestamp_utc')
            required_cols = ['timestamp', 'power_mw', 'soc_percent']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                return False, f"Missing required columns: {', '.join(missing)}. Found: {list(df.columns)}"

        elif csv_type == 'market':
            # CSVLoader expects 'timestamp' (not 'timestamp_utc')
            required_cols = ['timestamp', 'price_gbp_mwh', 'market_type']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                return False, f"Missing required columns: {', '.join(missing)}. Found: {list(df.columns)}"

        return True, ""

    except Exception as e:
        return False, f"Failed to read CSV: {str(e)}"


def run_data_ingestion(
    scada_path: Path,
    market_path: Path,
    asset_name: str,
    remediate: bool = True,
    max_iterations: int = 3
) -> IngestionResult:
    """
    Run complete data ingestion pipeline

    This is a wrapper around the ingest_data.py workflow that returns
    structured results suitable for Streamlit display.

    Args:
        scada_path: Path to raw SCADA CSV
        market_path: Path to raw market price CSV
        asset_name: Asset identifier (e.g., 'UK_BESS_001')
        remediate: Enable auto-remediation if DQ fails
        max_iterations: Maximum remediation attempts

    Returns:
        IngestionResult dataclass with success status and file paths
    """
    try:
        # Load configuration
        config_loader = ConfigLoader()
        configs = config_loader.load_all_configs()

        config_dict = configs['config'].model_dump()
        dq_rules = configs['dq_rules']
        price_rules = configs['price_rules']
        market_constraints = configs['market_constraints']

        # Get asset config
        asset_config = config_dict['bess_assets'].get(asset_name)
        if not asset_config:
            return IngestionResult(
                success=False,
                error_message=f"Asset '{asset_name}' not found in configuration"
            )

        settlement_duration_min = config_dict['market']['settlement_duration_min']

        # Initialize processors
        csv_loader = CSVLoader(config_dict)
        data_cleaner = DataCleaner(config_dict, dq_rules)
        price_selector = PriceSelector(price_rules)
        dq_scorer = DataQualityScorer(config_dict, dq_rules, asset_config, market_constraints)
        remediation_engine = RemediationEngine(dq_rules, settlement_duration_min)

        # Load SCADA data
        scada_df = csv_loader.load_scada_csv(scada_path)
        is_valid, errors = csv_loader.validate_data_types(scada_df, 'scada')
        if not is_valid:
            return IngestionResult(
                success=False,
                error_message=f"SCADA validation failed: {'; '.join(errors)}"
            )

        # Load market data
        market_df = csv_loader.load_market_csv(market_path)
        is_valid, errors = csv_loader.validate_data_types(market_df, 'market')
        if not is_valid:
            return IngestionResult(
                success=False,
                error_message=f"Market data validation failed: {'; '.join(errors)}"
            )

        # Clean and resample SCADA data
        scada_df = data_cleaner.resample_scada(scada_df)
        scada_df = data_cleaner.remove_duplicates(scada_df)

        # Clean market data
        market_df = data_cleaner.remove_duplicates(market_df)

        # Align timestamps
        scada_df, market_df = data_cleaner.align_timestamps(scada_df, market_df)

        # Select optimization prices
        market_df = price_selector.select_optimization_prices(market_df)

        # Remediation loop
        remediation_applied = False
        remediation_messages = []

        for iteration in range(max_iterations):
            # Score data quality
            scada_dq_report = dq_scorer.score_scada(scada_df)
            market_dq_report = dq_scorer.score_market(market_df)

            scada_score = scada_dq_report.overall_score
            market_score = market_dq_report.overall_score
            both_passed = scada_dq_report.passed and market_dq_report.passed

            # Check if DQ passes
            if both_passed:
                # Success! Save canonical files
                date_str = scada_df['timestamp_utc'].iloc[0].strftime('%Y-%m-%d')
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

                output_dir = Path('data/canonical')
                output_dir.mkdir(parents=True, exist_ok=True)

                scada_output = output_dir / f"scada_{asset_name}_{date_str}_{timestamp_str}.csv"
                market_output = output_dir / f"market_{asset_name}_{date_str}_{timestamp_str}.csv"

                scada_df.to_csv(scada_output, index=False)
                market_df.to_csv(market_output, index=False)

                return IngestionResult(
                    success=True,
                    scada_canonical_path=scada_output,
                    market_canonical_path=market_output,
                    scada_dq_score=scada_score,
                    market_dq_score=market_score,
                    scada_dq_report=scada_dq_report.model_dump(),
                    market_dq_report=market_dq_report.model_dump(),
                    remediation_applied=remediation_applied,
                    remediation_messages=remediation_messages
                )

            # DQ failed - try remediation if enabled
            if not remediate or iteration == max_iterations - 1:
                return IngestionResult(
                    success=False,
                    scada_dq_score=scada_score,
                    market_dq_score=market_score,
                    scada_dq_report=scada_dq_report.model_dump(),
                    market_dq_report=market_dq_report.model_dump(),
                    error_message=f"Data quality failed. SCADA: {scada_score:.1f}% {'✅' if scada_dq_report.passed else '❌'}, Market: {market_score:.1f}% {'✅' if market_dq_report.passed else '❌'}"
                )

            # Apply remediation
            if not scada_dq_report.passed and scada_dq_report.can_auto_remediate:
                # Set index for remediation (it expects indexed dataframe)
                scada_indexed = scada_df.set_index('timestamp_utc') if 'timestamp_utc' in scada_df.columns else scada_df
                scada_remediated, fixed, messages = remediation_engine.remediate_scada(scada_indexed, scada_dq_report)
                scada_df = scada_remediated.reset_index()
                if fixed:
                    remediation_applied = True
                    remediation_messages.extend([f"[SCADA Iteration {iteration+1}] {msg}" for msg in messages])

            if not market_dq_report.passed and market_dq_report.can_auto_remediate:
                # Set index for remediation (it expects indexed dataframe)
                market_indexed = market_df.set_index('timestamp_utc') if 'timestamp_utc' in market_df.columns else market_df
                market_remediated, fixed, messages = remediation_engine.remediate_market(market_indexed, market_dq_report)
                market_df = market_remediated.reset_index()
                if fixed:
                    remediation_applied = True
                    remediation_messages.extend([f"[Market Iteration {iteration+1}] {msg}" for msg in messages])

        # Should not reach here
        return IngestionResult(
            success=False,
            error_message="Remediation loop exhausted without success"
        )

    except Exception as e:
        return IngestionResult(
            success=False,
            error_message=f"Ingestion failed: {str(e)}"
        )


def run_optimization(
    scada_file: Path,
    market_file: Path,
    asset_name: str,
    initial_soc_percent: float = None,
    solver_name: str = 'PULP_CBC_CMD',
    timeout: int = 30
) -> OptimizationResult:
    """
    Run BESS optimization

    Wrapper around optimize_bess.py workflow that returns structured results.

    Args:
        scada_file: Path to canonical SCADA CSV
        market_file: Path to canonical market CSV
        asset_name: Asset identifier
        initial_soc_percent: Starting SoC (None = use first SCADA value)
        solver_name: MILP solver to use
        timeout: Solver timeout in seconds

    Returns:
        OptimizationResult dataclass with schedule and summary
    """
    try:
        # Load configuration
        config_loader = ConfigLoader()
        configs = config_loader.load_all_configs()
        config_dict = configs['config'].model_dump()

        # Get asset config
        asset_config = config_dict['bess_assets'].get(asset_name)
        if not asset_config:
            return OptimizationResult(
                success=False,
                error_message=f"Asset '{asset_name}' not found in configuration"
            )

        settlement_duration_min = config_dict['market']['settlement_duration_min']

        # Load canonical data
        scada_df = pd.read_csv(scada_file)
        scada_df['timestamp_utc'] = pd.to_datetime(scada_df['timestamp_utc'])

        market_df = pd.read_csv(market_file)
        market_df['timestamp_utc'] = pd.to_datetime(market_df['timestamp_utc'])

        # Initialize optimizer
        optimizer = BESSOptimizer(
            asset_config=asset_config,
            settlement_duration_min=settlement_duration_min,
            solver_name=solver_name,
            solver_timeout_sec=timeout
        )

        # Calculate actual performance
        actual_perf = optimizer.calculate_actual_performance(scada_df, market_df)

        # Run optimization
        if initial_soc_percent is None:
            initial_soc_percent = scada_df['soc_percent'].iloc[0]

        optimization_result = optimizer.optimize(scada_df, market_df, initial_soc_percent)

        # Calculate optimal discharge and charge energy for summary
        optimal_power = optimization_result['optimal_power_mw']
        dt_hours = settlement_duration_min / 60.0
        optimal_discharge_energy = sum([max(0, p) * dt_hours for p in optimal_power])
        optimal_charge_energy = sum([abs(min(0, p)) * dt_hours for p in optimal_power])

        # Prepare schedule DataFrame
        schedule_df = pd.DataFrame({
            'timestamp_utc': scada_df['timestamp_utc'],
            'optimal_power_mw': optimization_result['optimal_power_mw'],
            'optimal_soc_percent': optimization_result['optimal_soc_percent'],
            'actual_power_mw': scada_df['power_mw'],
            'actual_soc_percent': scada_df['soc_percent'],
            'price_gbp_mwh': market_df['price_gbp_mwh']
        })

        # Prepare summary dictionary to match optimize_bess.py structure
        # This flat structure is expected by the KPI calculators
        try:
            summary = {
                'asset_name': asset_name,
                'optimization_date': datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'solver_status': optimization_result['solver_status'],
                'solve_time_sec': optimization_result['solve_time_sec'],
                'optimal_revenue_gbp': optimization_result['optimal_revenue_gbp'],
                'actual_revenue_gbp': optimization_result['actual_revenue_gbp'],
                'revenue_variance_gbp': optimization_result['revenue_variance_gbp'],
                'market_capture_ratio': optimization_result['market_capture_ratio'],
                'cycles_used': optimization_result['cycles_used'],
                'max_daily_cycles': optimization_result['max_daily_cycles'],
                'duration_days': optimization_result['duration_days'],
                'optimal_discharge_energy_mwh': optimal_discharge_energy,
                'optimal_charge_energy_mwh': optimal_charge_energy,
                'actual_performance': actual_perf,
                'asset_config': asset_config
            }
        except KeyError as e:
            available_keys = list(optimization_result.keys())
            raise KeyError(f"Missing key {e} in optimization_result. Available keys: {available_keys}")

        # Save outputs
        date_str = scada_df['timestamp_utc'].iloc[0].strftime('%Y-%m-%d')
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        output_dir = Path('data/optimization_results')
        output_dir.mkdir(parents=True, exist_ok=True)

        schedule_output = output_dir / f"schedule_{asset_name}_{date_str}_{timestamp_str}.csv"
        summary_output = output_dir / f"summary_{asset_name}_{date_str}_{timestamp_str}.json"

        schedule_df.to_csv(schedule_output, index=False)

        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return OptimizationResult(
            success=True,
            schedule_df=schedule_df,
            summary=summary,
            solve_time=optimization_result['solve_time_sec'],
            solver_status=optimization_result['solver_status']
        )

    except Exception as e:
        return OptimizationResult(
            success=False,
            error_message=f"Optimization failed: {str(e)}"
        )


def format_dq_report(dq_report: Dict) -> Dict[str, Any]:
    """
    Format DQ report for Streamlit display

    Args:
        dq_report: DQ report dictionary from scorer

    Returns:
        Formatted report with display-friendly structure
    """
    if not dq_report:
        return {}

    return {
        'overall_score': dq_report['overall_score'],
        'passed': dq_report['passed'],
        'components': {
            'Completeness': {
                'score': dq_report['completeness']['score'],
                'details': f"{dq_report['completeness']['num_valid']}/{dq_report['completeness']['num_total']} records valid"
            },
            'Continuity': {
                'score': dq_report['continuity']['score'],
                'details': f"{dq_report['continuity']['num_gaps']} gaps detected (max: {dq_report['continuity']['max_gap_minutes']:.0f} min)"
            },
            'Bounds': {
                'score': dq_report['bounds']['score'],
                'details': f"{dq_report['bounds']['num_violations']} violations"
            },
            'Energy Reconciliation': {
                'score': dq_report.get('energy_reconciliation', {}).get('score', 100),
                'details': f"Error: {dq_report.get('energy_reconciliation', {}).get('reconciliation_error_percent', 0):.2f}%"
            } if 'energy_reconciliation' in dq_report else None
        },
        'recommendations': dq_report.get('recommendations', [])
    }
