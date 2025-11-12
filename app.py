"""
Ampyr BESS Asset Intelligence Platform
Streamlit Dashboard for BESS Optimization Analysis

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import get_config_loader
from src.visualization.bess_charts import BESSVisualizer
from src.modules.finance_kpis import FinanceKPICalculator
from src.modules.om_kpis import OMKPICalculator
from src.dashboard.dashboard_helpers import (
    save_uploaded_file,
    run_data_ingestion,
    run_optimization,
    format_dq_report,
    validate_csv_structure
)

# Page configuration
st.set_page_config(
    page_title="Ampyr BESS Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        padding-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .grade-A { color: #2ca02c; font-weight: 700; font-size: 2rem; }
    .grade-B { color: #8bc34a; font-weight: 700; font-size: 2rem; }
    .grade-C { color: #ffeb3b; font-weight: 700; font-size: 2rem; }
    .grade-D { color: #ff9800; font-weight: 700; font-size: 2rem; }
    .grade-E { color: #ff5722; font-weight: 700; font-size: 2rem; }
    .grade-F { color: #d62728; font-weight: 700; font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.schedule_df = None
    st.session_state.optimization_summary = None
    st.session_state.finance_kpis = None
    st.session_state.om_kpis = None
    st.session_state.asset_config = None

    # Workflow state variables
    st.session_state.uploaded_scada_path = None
    st.session_state.uploaded_market_path = None
    st.session_state.canonical_scada_path = None
    st.session_state.canonical_market_path = None
    st.session_state.scada_dq_score = None
    st.session_state.market_dq_score = None
    st.session_state.scada_dq_report = None
    st.session_state.market_dq_report = None
    st.session_state.remediation_applied = False
    st.session_state.remediation_messages = []
    st.session_state.optimization_status = None
    st.session_state.workflow_stage = "upload"  # upload, processing, optimized, complete
    st.session_state.selected_asset = None


def load_data(summary_file, schedule_file):
    """Load optimization data and calculate KPIs"""
    try:
        # Load optimization summary
        with open(summary_file, 'r') as f:
            optimization_summary = json.load(f)

        # Load schedule
        schedule_df = pd.read_csv(schedule_file)
        schedule_df['timestamp_utc'] = pd.to_datetime(schedule_df['timestamp_utc'])

        # Load configuration
        config_loader = get_config_loader()
        configs = config_loader.load_all_configs()
        config_dict = configs['config'].model_dump()

        asset_name = optimization_summary['asset_name']
        asset_config = config_dict['bess_assets'].get(asset_name)
        settlement_duration_min = config_dict['market']['settlement_duration_min']

        # Calculate KPIs
        finance_calc = FinanceKPICalculator(asset_config)
        om_calc = OMKPICalculator(asset_config)

        finance_kpis = finance_calc.calculate_kpis(optimization_summary, settlement_duration_min)
        schedule_kpis = finance_calc.calculate_schedule_based_kpis(schedule_df)
        finance_kpis.update(schedule_kpis)

        om_kpis = om_calc.calculate_kpis(optimization_summary, schedule_df, settlement_duration_min)

        return schedule_df, optimization_summary, finance_kpis, om_kpis, asset_config

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None


def show_data_upload():
    """Data upload and processing page"""
    st.markdown('<div class="main-header">📤 Data Upload & Processing</div>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to the BESS Asset Intelligence Platform! Upload your raw SCADA and market price data
    to begin the optimization workflow.
    """)

    st.markdown("---")

    # Step 1: Asset Selection
    st.subheader("Step 1: Select Asset")

    config_loader = get_config_loader()
    configs = config_loader.load_all_configs()
    config_dict = configs['config'].model_dump()
    asset_names = list(config_dict['bess_assets'].keys())

    selected_asset = st.selectbox(
        "Choose BESS Asset",
        options=asset_names,
        index=0,
        key='asset_selector'
    )

    if selected_asset:
        asset_config = config_dict['bess_assets'][selected_asset]
        st.session_state.selected_asset = selected_asset

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Capacity", f"{asset_config['capacity_mwh']} MWh")
        with col2:
            st.metric("Power", f"{asset_config['power_mw']} MW")
        with col3:
            st.metric("RTE", f"{asset_config['rte_percent']}%")
        with col4:
            st.metric("Max Cycles/Day", f"{asset_config['warranty']['max_daily_cycles']}")

    st.markdown("---")

    # Step 2: File Upload
    st.subheader("Step 2: Upload Data Files")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SCADA Data (CSV)**")
        st.markdown("Required columns: `timestamp_utc`, `power_mw`, `soc_percent`")
        scada_file = st.file_uploader(
            "Upload SCADA CSV",
            type=['csv'],
            key='scada_upload',
            help="Historical SCADA data with power and SoC measurements"
        )

        if scada_file is not None:
            st.success(f"✅ {scada_file.name} uploaded ({scada_file.size / 1024:.1f} KB)")

    with col2:
        st.markdown("**Market Price Data (CSV)**")
        st.markdown("Required columns: `timestamp_utc`, `price_gbp_mwh`, `market_type`")
        market_file = st.file_uploader(
            "Upload Market Price CSV",
            type=['csv'],
            key='market_upload',
            help="Market prices (day-ahead, imbalance, etc.)"
        )

        if market_file is not None:
            st.success(f"✅ {market_file.name} uploaded ({market_file.size / 1024:.1f} KB)")

    st.markdown("---")

    # Step 3: Processing Controls
    st.subheader("Step 3: Process Data")

    remediate = st.checkbox(
        "Enable Auto-Remediation",
        value=True,
        help="Automatically fix data quality issues (gaps ≤60 min, bounds violations)"
    )

    max_iterations = st.slider(
        "Max Remediation Iterations",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of attempts to improve data quality"
    )

    can_process = scada_file is not None and market_file is not None and selected_asset is not None

    if st.button("🚀 Process Data", type="primary", disabled=not can_process):
        if not can_process:
            st.error("Please upload both SCADA and market data files and select an asset")
        else:
            # Save uploaded files
            with st.spinner("Saving uploaded files..."):
                scada_path = save_uploaded_file(scada_file, "data/raw")
                market_path = save_uploaded_file(market_file, "data/raw")

                st.session_state.uploaded_scada_path = scada_path
                st.session_state.uploaded_market_path = market_path

            # Validate CSV structure
            st.info("Validating CSV structure...")
            scada_valid, scada_error = validate_csv_structure(scada_path, 'scada')
            market_valid, market_error = validate_csv_structure(market_path, 'market')

            if not scada_valid:
                st.error(f"❌ SCADA CSV validation failed: {scada_error}")
                return

            if not market_valid:
                st.error(f"❌ Market CSV validation failed: {market_error}")
                return

            # Run data ingestion
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Loading and cleaning data...")
            progress_bar.progress(20)

            result = run_data_ingestion(
                scada_path=scada_path,
                market_path=market_path,
                asset_name=selected_asset,
                remediate=remediate,
                max_iterations=max_iterations
            )

            progress_bar.progress(60)
            status_text.text("Scoring data quality...")

            if result.success:
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                # Store results in session state
                st.session_state.canonical_scada_path = result.scada_canonical_path
                st.session_state.canonical_market_path = result.market_canonical_path
                st.session_state.scada_dq_score = result.scada_dq_score
                st.session_state.market_dq_score = result.market_dq_score
                st.session_state.scada_dq_report = result.scada_dq_report
                st.session_state.market_dq_report = result.market_dq_report
                st.session_state.remediation_applied = result.remediation_applied
                st.session_state.remediation_messages = result.remediation_messages
                st.session_state.workflow_stage = "processing"

                st.success("🎉 Data processing successful!")

                # Show DQ scores
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SCADA DQ Score", f"{result.scada_dq_score:.1f}%",
                             delta=f"{result.scada_dq_score - 80:.1f}%" if result.scada_dq_score >= 80 else None)
                with col2:
                    st.metric("Market DQ Score", f"{result.market_dq_score:.1f}%",
                             delta=f"{result.market_dq_score - 80:.1f}%" if result.market_dq_score >= 80 else None)

                if result.remediation_applied:
                    st.info(f"ℹ️ Auto-remediation was applied ({len(result.remediation_messages)} fixes)")

                st.markdown("**Canonical Files Created:**")
                st.code(str(result.scada_canonical_path))
                st.code(str(result.market_canonical_path))

                st.info("👉 Go to **Optimization** page to run BESS optimization")

            else:
                progress_bar.progress(0)
                status_text.text("")
                st.error(f"❌ Data processing failed: {result.error_message}")

                if result.scada_dq_score is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("SCADA DQ Score", f"{result.scada_dq_score:.1f}%")
                    with col2:
                        st.metric("Market DQ Score", f"{result.market_dq_score:.1f}%")

                st.warning("💡 Suggestions:")
                st.markdown("""
                - Enable auto-remediation if not already enabled
                - Check for large data gaps (>60 minutes)
                - Verify timestamp format (ISO8601 UTC required)
                - Ensure power and SoC values are within valid bounds
                """)

    # Show current workflow state
    if st.session_state.canonical_scada_path:
        st.markdown("---")
        st.subheader("✅ Processing Status")

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**SCADA DQ:** {st.session_state.scada_dq_score:.1f}%")
            st.text(f"File: {Path(st.session_state.canonical_scada_path).name}")

        with col2:
            st.success(f"**Market DQ:** {st.session_state.market_dq_score:.1f}%")
            st.text(f"File: {Path(st.session_state.canonical_market_path).name}")

        if st.session_state.remediation_applied and st.session_state.remediation_messages:
            with st.expander("View Remediation Details"):
                for msg in st.session_state.remediation_messages:
                    st.text(msg)


def show_optimization():
    """Optimization execution page"""
    st.markdown('<div class="main-header">⚙️ BESS Optimization</div>', unsafe_allow_html=True)

    if st.session_state.canonical_scada_path is None:
        st.warning("⚠️ Please upload and process data first (go to **Data Upload** page)")
        return

    st.markdown(f"""
    Ready to optimize **{st.session_state.selected_asset}** using cleaned data.
    The MILP solver will find the optimal arbitrage strategy.
    """)

    st.markdown("---")

    # DQ Score Summary
    st.subheader("Data Quality Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("SCADA DQ Score", f"{st.session_state.scada_dq_score:.1f}%")

    with col2:
        st.metric("Market DQ Score", f"{st.session_state.market_dq_score:.1f}%")

    with col3:
        dq_passed = (st.session_state.scada_dq_score >= 80 and
                     st.session_state.market_dq_score >= 80)
        if dq_passed:
            st.success("✅ DQ Passed")
        else:
            st.error("❌ DQ Failed")

    st.markdown("---")

    # Optimization Settings
    st.subheader("Optimization Settings")

    col1, col2 = st.columns(2)

    with col1:
        # Load initial SoC from canonical SCADA
        if st.session_state.canonical_scada_path:
            try:
                scada_df = pd.read_csv(st.session_state.canonical_scada_path)
                default_soc = float(scada_df['soc_percent'].iloc[0])
            except:
                default_soc = 50.0
        else:
            default_soc = 50.0

        initial_soc = st.slider(
            "Initial SoC (%)",
            min_value=5.0,
            max_value=95.0,
            value=default_soc,
            step=1.0,
            help="Starting state of charge for optimization"
        )

    with col2:
        solver_timeout = st.number_input(
            "Solver Timeout (seconds)",
            min_value=5,
            max_value=300,
            value=30,
            step=5,
            help="Maximum time for solver to find solution"
        )

    st.markdown("---")

    # Run Optimization
    if st.button("▶️ Run Optimization", type="primary", disabled=not dq_passed):
        if not dq_passed:
            st.error("Cannot run optimization: Data quality below 80% threshold")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Initializing optimizer...")
            progress_bar.progress(10)

            status_text.text("Loading cleaned data...")
            progress_bar.progress(20)

            status_text.text("Building MILP model...")
            progress_bar.progress(40)

            status_text.text("Running solver (this may take up to 30 seconds)...")
            progress_bar.progress(50)

            # Run optimization
            result = run_optimization(
                scada_file=st.session_state.canonical_scada_path,
                market_file=st.session_state.canonical_market_path,
                asset_name=st.session_state.selected_asset,
                initial_soc_percent=initial_soc,
                solver_name='PULP_CBC_CMD',
                timeout=solver_timeout
            )

            if result.success:
                progress_bar.progress(80)
                status_text.text("Calculating KPIs...")

                # Load configuration for KPI calculation
                config_loader = get_config_loader()
                configs = config_loader.load_all_configs()
                config_dict = configs['config'].model_dump()
                asset_config = config_dict['bess_assets'][st.session_state.selected_asset]
                settlement_duration_min = config_dict['market']['settlement_duration_min']

                # Calculate KPIs
                finance_calc = FinanceKPICalculator(asset_config)
                om_calc = OMKPICalculator(asset_config)

                finance_kpis = finance_calc.calculate_kpis(result.summary, settlement_duration_min)
                schedule_kpis = finance_calc.calculate_schedule_based_kpis(result.schedule_df)
                finance_kpis.update(schedule_kpis)

                om_kpis = om_calc.calculate_kpis(result.summary, result.schedule_df, settlement_duration_min)

                progress_bar.progress(100)
                status_text.text("✅ Optimization complete!")

                # Store in session state
                st.session_state.schedule_df = result.schedule_df
                st.session_state.optimization_summary = result.summary
                st.session_state.finance_kpis = finance_kpis
                st.session_state.om_kpis = om_kpis
                st.session_state.asset_config = asset_config
                st.session_state.optimization_status = result.solver_status
                st.session_state.workflow_stage = "complete"
                st.session_state.data_loaded = True

                st.success("🎉 Optimization successful!")

                # Show key results
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Solver Status", result.solver_status)

                with col2:
                    st.metric("Solve Time", f"{result.solve_time:.2f}s")

                with col3:
                    actual_revenue = result.summary['actual']['revenue_gbp']
                    optimal_revenue = result.summary['optimal']['revenue_gbp']
                    st.metric("Actual Revenue", f"£{actual_revenue:,.0f}")

                with col4:
                    st.metric("Optimal Revenue", f"£{optimal_revenue:,.0f}",
                             delta=f"£{optimal_revenue - actual_revenue:,.0f}")

                st.markdown("---")

                st.subheader("Optimization Summary")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Actual Performance**")
                    st.markdown(f"- Revenue: £{actual_revenue:,.2f}")
                    st.markdown(f"- Cycles: {result.summary['actual']['cycles']:.2f}")
                    st.markdown(f"- Discharge: {result.summary['actual']['discharge_energy_mwh']:.2f} MWh")

                with col2:
                    st.markdown("**Optimal Performance**")
                    st.markdown(f"- Revenue: £{optimal_revenue:,.2f}")
                    st.markdown(f"- Cycles: {result.summary['optimal']['cycles']:.2f}")
                    st.markdown(f"- Discharge: {result.summary['optimal']['discharge_energy_mwh']:.2f} MWh")

                st.info("👉 View detailed analysis in **Finance Dashboard** and **O&M Dashboard** pages")

            else:
                progress_bar.progress(0)
                status_text.text("")
                st.error(f"❌ Optimization failed: {result.error_message}")

                st.warning("💡 Troubleshooting:")
                st.markdown("""
                - Check if solver timeout is sufficient (increase if needed)
                - Verify data quality scores are above 80%
                - Ensure time series data has no large gaps
                - Try with a shorter time horizon
                """)

    # Show current optimization status
    if st.session_state.schedule_df is not None:
        st.markdown("---")
        st.subheader("✅ Optimization Status")

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"**Status:** {st.session_state.optimization_status}")
            st.text(f"Asset: {st.session_state.selected_asset}")

        with col2:
            st.success(f"**Periods Optimized:** {len(st.session_state.schedule_df)}")
            duration_days = len(st.session_state.schedule_df) * 0.5 / 24  # 30-min periods
            st.text(f"Duration: {duration_days:.1f} days")

        # Schedule preview
        with st.expander("View Schedule Preview (First 10 Periods)"):
            st.dataframe(
                st.session_state.schedule_df.head(10),
                use_container_width=True,
                hide_index=True
            )


def show_overview():
    """Overview page with key metrics"""
    st.markdown('<div class="main-header">⚡ BESS Asset Intelligence Platform</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.info("👈 Please load optimization results from the sidebar to begin")

        # Show example metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Asset", "UK_BESS_001")
        with col2:
            st.metric("Capacity", "8.4 MWh")
        with col3:
            st.metric("Power", "7.5 MW")
        with col4:
            st.metric("RTE", "87%")

        st.markdown("---")
        st.subheader("Platform Capabilities")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📊 Finance Analytics
            - Market Capture Ratio
            - Revenue Variance Analysis
            - IRR Impact Assessment
            - Price Spread Analysis
            - Letter Grade Performance
            """)

        with col2:
            st.markdown("""
            ### ⚙️ O&M Metrics
            - Availability Tracking
            - Cycle Utilization
            - RTE Performance
            - Capacity Factor
            - Degradation Monitoring
            """)

        return

    # Show loaded data overview
    summary = st.session_state.optimization_summary
    finance = st.session_state.finance_kpis
    om = st.session_state.om_kpis

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Market Capture",
            f"{finance['market_capture_ratio']:.1f}%",
            delta=f"{finance['market_capture_ratio'] - 100:.1f}%",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "Revenue Opportunity",
            f"£{finance['lost_opportunity_gbp']:,.0f}",
            delta=f"{finance['revenue_variance_percent']:.0f}%",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            "Availability",
            f"{om['availability_percent']:.1f}%",
            delta=f"{om['availability_percent'] - 95:.1f}%",
            delta_color="normal"
        )

    with col4:
        st.metric(
            "Cycle Utilization",
            f"{om['cycle_utilization_percent']:.1f}%",
            delta=f"{om['cycle_utilization_percent'] - 80:.1f}%",
            delta_color="normal"
        )

    st.markdown("---")

    # Performance grades
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Finance Performance")
        grade = finance['finance_grade']
        st.markdown(f'<div class="grade-{grade}">Grade: {grade}</div>', unsafe_allow_html=True)
        st.markdown(f"**Actual Revenue:** £{summary['actual_revenue_gbp']:,.2f}")
        st.markdown(f"**Optimal Revenue:** £{summary['optimal_revenue_gbp']:,.2f}")
        st.markdown(f"**Variance:** £{summary['revenue_variance_gbp']:,.2f}")

    with col2:
        st.markdown("### ⚙️ O&M Performance")
        grade = om['om_grade']
        st.markdown(f'<div class="grade-{grade}">Grade: {grade}</div>', unsafe_allow_html=True)
        st.markdown(f"**Actual Cycles:** {om['actual_cycles']:.2f}")
        st.markdown(f"**Max Allowed:** {om['max_allowed_cycles']:.2f}")
        st.markdown(f"**Actual RTE:** {om['actual_rte_percent']:.1f}%")


def show_finance_dashboard():
    """Finance-focused dashboard"""
    st.markdown('<div class="main-header">📊 Finance Dashboard</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("Please load optimization results from the sidebar first")
        return

    finance = st.session_state.finance_kpis
    summary = st.session_state.optimization_summary
    schedule_df = st.session_state.schedule_df
    visualizer = BESSVisualizer()

    # Performance summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Finance Grade",
            finance['finance_grade'],
            help="Letter grade based on market capture ratio"
        )

    with col2:
        st.metric(
            "Market Capture Ratio",
            f"{finance['market_capture_ratio']:.1f}%",
            delta=f"{finance['market_capture_ratio'] - 100:.1f}%"
        )

    with col3:
        st.metric(
            "IRR Impact",
            f"{finance['irr_impact_estimate_bps']:.0f} bps",
            help="Estimated impact on project IRR"
        )

    st.markdown("---")

    # Revenue comparison chart
    st.subheader("Revenue Performance")
    fig = visualizer.create_revenue_comparison_chart(summary)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Actual Revenue**")
        st.markdown(f"£{finance['arbitrage_revenue_gbp']:,.2f}")

        st.markdown("**Optimal Revenue**")
        st.markdown(f"£{finance['optimal_revenue_gbp']:,.2f}")

    with col2:
        st.markdown("**Revenue Variance**")
        st.markdown(f"£{finance['revenue_variance_gbp']:,.2f} ({finance['revenue_variance_percent']:.1f}%)")

        st.markdown("**Lost Opportunity**")
        st.markdown(f"£{finance['lost_opportunity_gbp']:,.2f}")

    st.markdown("---")

    # Market capture gauge
    st.subheader("Market Capture Performance")
    fig = visualizer.create_market_capture_gauge(finance['market_capture_ratio'])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Price analysis
    st.subheader("Price Capture Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Avg Discharge Price",
            f"£{finance.get('average_discharge_price_gbp_mwh', 0):.2f}/MWh",
            help="Weighted average price for discharge periods"
        )

    with col2:
        st.metric(
            "Avg Charge Price",
            f"£{finance.get('average_charge_price_gbp_mwh', 0):.2f}/MWh",
            help="Weighted average price for charge periods"
        )

    with col3:
        st.metric(
            "Spread Captured",
            f"£{finance.get('price_spread_captured_gbp_mwh', 0):.2f}/MWh",
            delta=f"£{finance.get('market_price_spread', 0) - finance.get('price_spread_captured_gbp_mwh', 0):.2f}/MWh opportunity"
        )

    # Price spread chart
    fig = visualizer.create_price_spread_chart(schedule_df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Detailed KPIs
    st.subheader("Detailed Finance KPIs")

    kpi_df = pd.DataFrame({
        'Metric': [
            'Revenue per Cycle',
            'Revenue per MWh Discharged',
            'Revenue per Capacity MWh',
            'Market Price Mean',
            'Market Price Std Dev',
            'Market Price Spread'
        ],
        'Value': [
            f"£{finance.get('revenue_per_cycle_gbp', 0):.2f}",
            f"£{finance.get('revenue_per_discharge_mwh', 0):.2f}/MWh",
            f"£{finance.get('revenue_per_capacity_mwh', 0):.2f}/MWh",
            f"£{finance.get('market_price_mean', 0):.2f}/MWh",
            f"£{finance.get('market_price_std', 0):.2f}/MWh",
            f"£{finance.get('market_price_spread', 0):.2f}/MWh"
        ]
    })

    st.dataframe(kpi_df, hide_index=True, use_container_width=True)

    # Download buttons
    st.markdown("---")
    st.subheader("📥 Download Finance Report")

    col1, col2 = st.columns(2)

    with col1:
        # Finance KPIs as JSON
        finance_json = json.dumps(finance, indent=2, default=str)
        st.download_button(
            label="Download Finance KPIs (JSON)",
            data=finance_json,
            file_name=f"finance_kpis_{summary['asset_name']}_{summary['optimization_date']}.json",
            mime="application/json"
        )

    with col2:
        # Finance KPIs as CSV
        finance_df = pd.DataFrame([finance])
        finance_csv = finance_df.to_csv(index=False)
        st.download_button(
            label="Download Finance KPIs (CSV)",
            data=finance_csv,
            file_name=f"finance_kpis_{summary['asset_name']}_{summary['optimization_date']}.csv",
            mime="text/csv"
        )


def show_om_dashboard():
    """O&M-focused dashboard"""
    st.markdown('<div class="main-header">⚙️ O&M Dashboard</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("Please load optimization results from the sidebar first")
        return

    om = st.session_state.om_kpis
    summary = st.session_state.optimization_summary
    schedule_df = st.session_state.schedule_df
    asset_config = st.session_state.asset_config
    visualizer = BESSVisualizer()

    # Performance summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "O&M Grade",
            om['om_grade'],
            help="Letter grade based on multiple operational factors"
        )

    with col2:
        st.metric(
            "Availability",
            f"{om['availability_percent']:.1f}%",
            delta=f"{om['availability_percent'] - 99:.1f}%"
        )

    with col3:
        st.metric(
            "Cycle Utilization",
            f"{om['cycle_utilization_percent']:.1f}%",
            delta=f"{om['cycle_utilization_percent'] - 80:.1f}%"
        )

    with col4:
        st.metric(
            "Capacity Factor",
            f"{om['capacity_factor_percent']:.1f}%",
            delta=f"{om['capacity_factor_percent'] - 20:.1f}%"
        )

    st.markdown("---")

    # Cycle utilization
    st.subheader("Cycle Utilization")
    fig = visualizer.create_cycle_utilization_chart(summary)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Actual Cycles**")
        st.markdown(f"{om['actual_cycles']:.2f}")

    with col2:
        st.markdown("**Max Allowed Cycles**")
        st.markdown(f"{om['max_allowed_cycles']:.2f}")

    with col3:
        st.markdown("**Avg Cycle Depth**")
        st.markdown(f"{om['avg_cycle_depth_percent']:.1f}%")

    st.markdown("---")

    # Power profile
    st.subheader("Power Profile - Actual vs Optimal")
    fig = visualizer.create_power_profile_chart(schedule_df)
    st.plotly_chart(fig, use_container_width=True)

    # SoC curve
    st.subheader("State of Charge - Actual vs Optimal")
    soc_limits = {
        'soc_min_percent': asset_config['constraints']['soc_min_percent'],
        'soc_max_percent': asset_config['constraints']['soc_max_percent']
    }
    fig = visualizer.create_soc_curve_chart(schedule_df, soc_limits)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Efficiency metrics
    st.subheader("Efficiency Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Actual RTE",
            f"{om['actual_rte_percent']:.1f}%",
            delta=f"{om['rte_deviation_percent']:.1f}%"
        )

    with col2:
        st.metric(
            "Discharge Throughput",
            f"{om['discharge_throughput_mwh']:.2f} MWh"
        )

    with col3:
        st.metric(
            "Charge Throughput",
            f"{om['charge_throughput_mwh']:.2f} MWh"
        )

    st.markdown("---")

    # Asset utilization
    st.subheader("Asset Utilization")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "SoC Range Utilization",
            f"{om['soc_range_utilization_percent']:.1f}%",
            help="Percentage of available SoC range used"
        )

    with col2:
        st.metric(
            "Power Utilization",
            f"{om['power_utilization_percent']:.1f}%",
            help="Peak power vs rated power"
        )

    with col3:
        if om.get('idle_time_percent') is not None:
            st.metric(
                "Idle Time",
                f"{om['idle_time_percent']:.1f}%",
                delta=f"{50 - om['idle_time_percent']:.1f}%",
                delta_color="inverse"
            )

    st.markdown("---")

    # Detailed O&M KPIs
    st.subheader("Detailed O&M KPIs")

    kpi_df = pd.DataFrame({
        'Metric': [
            'Actual Cycles',
            'Max Allowed Cycles',
            'Cycle Utilization',
            'Actual RTE',
            'Rated RTE',
            'RTE Deviation',
            'Estimated Annual Degradation'
        ],
        'Value': [
            f"{om['actual_cycles']:.2f}",
            f"{om['max_allowed_cycles']:.2f}",
            f"{om['cycle_utilization_percent']:.1f}%",
            f"{om['actual_rte_percent']:.1f}%",
            f"{om['rated_rte_percent']:.1f}%",
            f"{om['rte_deviation_percent']:+.1f}%",
            f"{om['estimated_annual_degradation_percent']:.2f}%"
        ]
    })

    st.dataframe(kpi_df, hide_index=True, use_container_width=True)

    # Download buttons
    st.markdown("---")
    st.subheader("📥 Download O&M Report")

    col1, col2, col3 = st.columns(3)

    with col1:
        # O&M KPIs as JSON
        om_json = json.dumps(om, indent=2, default=str)
        st.download_button(
            label="Download O&M KPIs (JSON)",
            data=om_json,
            file_name=f"om_kpis_{summary['asset_name']}_{summary['optimization_date']}.json",
            mime="application/json"
        )

    with col2:
        # O&M KPIs as CSV
        om_df = pd.DataFrame([om])
        om_csv = om_df.to_csv(index=False)
        st.download_button(
            label="Download O&M KPIs (CSV)",
            data=om_csv,
            file_name=f"om_kpis_{summary['asset_name']}_{summary['optimization_date']}.csv",
            mime="text/csv"
        )

    with col3:
        # Combined report (Finance + O&M)
        finance = st.session_state.finance_kpis
        combined = {
            'asset_name': summary['asset_name'],
            'optimization_date': summary['optimization_date'],
            'finance_kpis': finance,
            'om_kpis': om
        }
        combined_json = json.dumps(combined, indent=2, default=str)
        st.download_button(
            label="Download Combined Report (JSON)",
            data=combined_json,
            file_name=f"combined_report_{summary['asset_name']}_{summary['optimization_date']}.json",
            mime="application/json"
        )


def show_data_quality():
    """Data quality metrics page"""
    st.markdown('<div class="main-header">🔍 Data Quality</div>', unsafe_allow_html=True)

    # Check if we have DQ reports from upload workflow
    if st.session_state.scada_dq_report is not None:
        st.subheader("Data Quality Scores")

        col1, col2, col3 = st.columns(3)

        with col1:
            score = st.session_state.scada_dq_score
            st.metric(
                "SCADA DQ Score",
                f"{score:.1f}%",
                delta=f"{score - 80:.1f}%" if score >= 80 else None,
                delta_color="normal" if score >= 80 else "inverse"
            )

        with col2:
            score = st.session_state.market_dq_score
            st.metric(
                "Market DQ Score",
                f"{score:.1f}%",
                delta=f"{score - 80:.1f}%" if score >= 80 else None,
                delta_color="normal" if score >= 80 else "inverse"
            )

        with col3:
            both_passed = (st.session_state.scada_dq_score >= 80 and
                          st.session_state.market_dq_score >= 80)
            if both_passed:
                st.success("✅ All DQ Checks Passed")
            else:
                st.error("❌ DQ Below Threshold")

        st.markdown("---")

        # SCADA DQ Breakdown
        st.subheader("SCADA Data Quality Components")

        if st.session_state.scada_dq_report:
            formatted_report = format_dq_report(st.session_state.scada_dq_report)

            cols = st.columns(4)
            for idx, (component, data) in enumerate(formatted_report['components'].items()):
                if data is not None:
                    with cols[idx % 4]:
                        st.metric(component, f"{data['score']:.1f}%")
                        st.caption(data['details'])

        st.markdown("---")

        # Market DQ Breakdown
        st.subheader("Market Data Quality Components")

        if st.session_state.market_dq_report:
            formatted_report = format_dq_report(st.session_state.market_dq_report)

            cols = st.columns(3)
            valid_components = [
                (comp, data) for comp, data in formatted_report['components'].items()
                if data is not None
            ]

            for idx, (component, data) in enumerate(valid_components):
                with cols[idx % 3]:
                    st.metric(component, f"{data['score']:.1f}%")
                    st.caption(data['details'])

        # Remediation Info
        if st.session_state.remediation_applied:
            st.markdown("---")
            st.subheader("Remediation Applied")

            st.info(f"✓ Auto-remediation was successfully applied ({len(st.session_state.remediation_messages)} fixes)")

            with st.expander("View Remediation Details"):
                for msg in st.session_state.remediation_messages:
                    st.text(msg)

        st.markdown("---")

    if not st.session_state.data_loaded:
        st.warning("Please process data through the **Data Upload** page or load optimization results from the sidebar")
        return

    schedule_df = st.session_state.schedule_df

    st.subheader("Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Periods", len(schedule_df))

    with col2:
        st.metric("Start Time", schedule_df['timestamp_utc'].min().strftime('%Y-%m-%d %H:%M'))

    with col3:
        st.metric("End Time", schedule_df['timestamp_utc'].max().strftime('%Y-%m-%d %H:%M'))

    with col4:
        duration_hours = len(schedule_df) * 0.5  # 30-min periods
        st.metric("Duration", f"{duration_hours/24:.1f} days")

    st.markdown("---")

    # Data statistics
    st.subheader("SCADA Data Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Power (MW)**")
        power_stats = schedule_df['actual_power_mw'].describe()
        st.dataframe(power_stats, use_container_width=True)

    with col2:
        st.markdown("**SoC (%)**")
        soc_stats = schedule_df['actual_soc_percent'].describe()
        st.dataframe(soc_stats, use_container_width=True)

    st.markdown("---")

    st.subheader("Market Price Statistics")

    price_stats = schedule_df['price_gbp_mwh'].describe()
    st.dataframe(price_stats, use_container_width=True)

    # Data preview
    st.markdown("---")
    st.subheader("Data Preview")
    st.dataframe(schedule_df.head(20), use_container_width=True)


# Main app
def main():
    """Main application"""

    # Sidebar
    with st.sidebar:
        st.title("⚡ BESS Dashboard")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigate",
            ["Data Upload", "Optimization", "Overview", "Finance Dashboard", "O&M Dashboard", "Data Quality"],
            index=0
        )

        st.markdown("---")

        # File selection
        st.subheader("Load Optimization Results")

        # Look for existing files
        results_dir = Path("data/optimization_results")
        if results_dir.exists():
            summary_files = list(results_dir.glob("summary_*.json"))
            schedule_files = list(results_dir.glob("schedule_*.csv"))

            if summary_files and schedule_files:
                summary_file = st.selectbox(
                    "Summary File",
                    summary_files,
                    format_func=lambda x: x.name
                )

                schedule_file = st.selectbox(
                    "Schedule File",
                    schedule_files,
                    format_func=lambda x: x.name
                )

                if st.button("Load Data", type="primary"):
                    with st.spinner("Loading data and calculating KPIs..."):
                        schedule_df, opt_summary, finance, om, asset_config = load_data(
                            summary_file, schedule_file
                        )

                        if opt_summary is not None:
                            st.session_state.data_loaded = True
                            st.session_state.schedule_df = schedule_df
                            st.session_state.optimization_summary = opt_summary
                            st.session_state.finance_kpis = finance
                            st.session_state.om_kpis = om
                            st.session_state.asset_config = asset_config
                            st.success("✅ Data loaded successfully!")
                            st.rerun()
            else:
                st.info("No optimization results found in data/optimization_results/")

        st.markdown("---")

        # Show loaded data info
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
            summary = st.session_state.optimization_summary
            st.markdown(f"**Asset:** {summary['asset_name']}")
            st.markdown(f"**Date:** {summary['optimization_date']}")
            st.markdown(f"**Periods:** {len(st.session_state.schedule_df)}")

            if st.button("Clear Data"):
                st.session_state.data_loaded = False
                st.rerun()

    # Main content area
    if page == "Data Upload":
        show_data_upload()
    elif page == "Optimization":
        show_optimization()
    elif page == "Overview":
        show_overview()
    elif page == "Finance Dashboard":
        show_finance_dashboard()
    elif page == "O&M Dashboard":
        show_om_dashboard()
    elif page == "Data Quality":
        show_data_quality()


if __name__ == "__main__":
    main()
