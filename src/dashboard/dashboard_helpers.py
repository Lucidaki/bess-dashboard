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
        df = pd.read_csv(file_path, nrows=5)  # Just check header

        if csv_type == 'scada':
            required_cols = ['timestamp_utc', 'power_mw', 'soc_percent']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                return False, f"Missing required columns: {', '.join(missing)}"

        elif csv_type == 'market':
            required_cols = ['timestamp_utc', 'price_gbp_mwh', 'market_type']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                return False, f"Missing required columns: {', '.join(missing)}"

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
        config = ConfigLoader()
        asset_config = config.get_asset_config(asset_name)
        market_config = config.get_market_config()
        dq_config = config.get_data_quality_config()

        # Initialize processors
        csv_loader = CSVLoader(encoding='utf-8-sig')
        data_cleaner = DataCleaner(
            settlement_duration_min=market_config['settlement_duration_min'],
            timezone=market_config['timezone']
        )
        price_selector = PriceSelector(config.config)
        dq_scorer = DataQualityScorer(
            asset_config=asset_config,
            market_config=market_config,
            dq_config=dq_config
        )
        remediation_engine = RemediationEngine(config.config)

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
        scada_df = data_cleaner.resample_scada(scada_df, asset_config['capacity_mwh'])
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

            # Check if DQ passes
            min_dq_score = dq_config['min_dq_score']

            if scada_score >= min_dq_score and market_score >= min_dq_score:
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
                    scada_dq_report=scada_dq_report.to_dict(),
                    market_dq_report=market_dq_report.to_dict(),
                    remediation_applied=remediation_applied,
                    remediation_messages=remediation_messages
                )

            # DQ failed - try remediation if enabled
            if not remediate or iteration == max_iterations - 1:
                return IngestionResult(
                    success=False,
                    scada_dq_score=scada_score,
                    market_dq_score=market_score,
                    scada_dq_report=scada_dq_report.to_dict(),
                    market_dq_report=market_dq_report.to_dict(),
                    error_message=f"Data quality failed. SCADA: {scada_score:.1f}%, Market: {market_score:.1f}% (Minimum: {min_dq_score}%)"
                )

            # Apply remediation
            if scada_score < min_dq_score:
                scada_df, fixed, messages = remediation_engine.remediate_scada(scada_df, scada_dq_report)
                if fixed:
                    remediation_applied = True
                    remediation_messages.extend([f"[SCADA Iteration {iteration+1}] {msg}" for msg in messages])

            if market_score < min_dq_score:
                market_df, fixed, messages = remediation_engine.remediate_market(market_df, market_dq_report)
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
        config = ConfigLoader()
        asset_config = config.get_asset_config(asset_name)
        market_config = config.get_market_config()

        # Load canonical data
        scada_df = pd.read_csv(scada_file)
        scada_df['timestamp_utc'] = pd.to_datetime(scada_df['timestamp_utc'])

        market_df = pd.read_csv(market_file)
        market_df['timestamp_utc'] = pd.to_datetime(market_df['timestamp_utc'])

        # Initialize optimizer
        optimizer = BESSOptimizer(
            asset_config=asset_config,
            settlement_duration_min=market_config['settlement_duration_min'],
            solver_name=solver_name,
            timeout=timeout
        )

        # Calculate actual performance
        actual_perf = optimizer.calculate_actual_performance(scada_df, market_df)

        # Run optimization
        if initial_soc_percent is None:
            initial_soc_percent = scada_df['soc_percent'].iloc[0]

        optimization_result = optimizer.optimize(scada_df, market_df, initial_soc_percent)

        # Prepare schedule DataFrame
        schedule_df = pd.DataFrame({
            'timestamp_utc': scada_df['timestamp_utc'],
            'optimal_power_mw': optimization_result['schedule']['power_mw'],
            'optimal_soc_percent': optimization_result['schedule']['soc_percent'],
            'actual_power_mw': scada_df['power_mw'],
            'actual_soc_percent': scada_df['soc_percent'],
            'price_gbp_mwh': market_df['price_gbp_mwh']
        })

        # Prepare summary dictionary
        summary = {
            'asset_name': asset_name,
            'optimization_date': datetime.now().strftime('%Y-%m-%d'),
            'solver_status': optimization_result['solver_status'],
            'solve_time_sec': optimization_result['solve_time_sec'],
            'actual': actual_perf,
            'optimal': {
                'revenue_gbp': optimization_result['revenue_gbp'],
                'discharge_energy_mwh': optimization_result['discharge_energy_mwh'],
                'charge_energy_mwh': optimization_result['charge_energy_mwh'],
                'cycles': optimization_result['cycles'],
                'soc_min_percent': optimization_result['soc_min_percent'],
                'soc_max_percent': optimization_result['soc_max_percent']
            },
            'comparison': {
                'revenue_variance_gbp': optimization_result['revenue_gbp'] - actual_perf['revenue_gbp'],
                'revenue_variance_percent': ((optimization_result['revenue_gbp'] - actual_perf['revenue_gbp']) / optimization_result['revenue_gbp'] * 100) if optimization_result['revenue_gbp'] != 0 else 0
            },
            'market': {
                'price_min_gbp_mwh': float(market_df['price_gbp_mwh'].min()),
                'price_max_gbp_mwh': float(market_df['price_gbp_mwh'].max()),
                'price_mean_gbp_mwh': float(market_df['price_gbp_mwh'].mean()),
                'price_spread_gbp_mwh': float(market_df['price_gbp_mwh'].max() - market_df['price_gbp_mwh'].min())
            },
            'asset_config': asset_config
        }

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
