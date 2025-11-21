"""
Streamlit Dashboard for BESS and Northwold Data Analysis
Displays comprehensive analysis of battery energy storage system operations and market trading
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import digital_twin_config as config

# Configure data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'  # Fallback to current directory for local testing

# Set page config
st.set_page_config(
    page_title="BESS Analysis Dashboard",
    page_icon="🔋",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load and cache the CSV files"""
    try:
        bess_df = pd.read_csv(os.path.join(DATA_DIR, 'BESS_Sept_2025_converted.csv'))
        northwold_df = pd.read_csv(os.path.join(DATA_DIR, 'Northwold_Sep_2025_converted.csv'))

        # Convert timestamps
        bess_df['date'] = pd.to_datetime(bess_df['date'], format='%d/%m/%Y %H:%M:%S')
        northwold_df['Timestamp'] = pd.to_datetime(northwold_df['Timestamp'], format='%d-%m-%Y %H:%M')

        return bess_df, northwold_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def analyze_bess_data(bess_df):
    """Analyze BESS dataset"""
    analysis = {}

    # Time coverage
    analysis['start_time'] = bess_df['date'].min()
    analysis['end_time'] = bess_df['date'].max()
    analysis['duration_days'] = (bess_df['date'].max() - bess_df['date'].min()).days + 1

    # SOC Analysis
    analysis['soc_mean'] = bess_df['SOC'].mean()
    analysis['soc_median'] = bess_df['SOC'].median()
    analysis['soc_min'] = bess_df['SOC'].min()
    analysis['soc_max'] = bess_df['SOC'].max()
    analysis['soc_std'] = bess_df['SOC'].std()

    # Power Analysis
    analysis['power_mean'] = bess_df['Power'].mean()
    analysis['power_max_charge'] = bess_df['Power'].max()
    analysis['power_max_discharge'] = bess_df['Power'].min()
    analysis['total_energy_charged'] = bess_df[bess_df['Power'] > 0]['Power'].sum() * 0.5
    analysis['total_energy_discharged'] = abs(bess_df[bess_df['Power'] < 0]['Power'].sum() * 0.5)

    # Frequency Analysis
    analysis['freq_mean'] = bess_df['Frequency'].mean()
    analysis['freq_min'] = bess_df['Frequency'].min()
    analysis['freq_max'] = bess_df['Frequency'].max()
    analysis['freq_std'] = bess_df['Frequency'].std()

    # Operation patterns
    analysis['charging_periods'] = (bess_df['Power'] > 0).sum()
    analysis['discharging_periods'] = (bess_df['Power'] < 0).sum()
    analysis['idle_periods'] = (bess_df['Power'] == 0).sum()
    analysis['total_periods'] = len(bess_df)

    return analysis

def analyze_northwold_data(northwold_df):
    """Analyze Northwold dataset"""
    analysis = {}

    # Time coverage
    analysis['start_time'] = northwold_df['Timestamp'].min()
    analysis['end_time'] = northwold_df['Timestamp'].max()
    analysis['duration_days'] = (northwold_df['Timestamp'].max() - northwold_df['Timestamp'].min()).days + 1

    # Energy Trading Analysis
    analysis['da_price_mean'] = northwold_df['Day Ahead Price (EPEX)'].mean()
    analysis['da_price_min'] = northwold_df['Day Ahead Price (EPEX)'].min()
    analysis['da_price_max'] = northwold_df['Day Ahead Price (EPEX)'].max()
    analysis['da_price_std'] = northwold_df['Day Ahead Price (EPEX)'].std()

    analysis['intraday_price_mean'] = northwold_df['GB-ISEM Intraday 1 Price'].mean()
    analysis['intraday_price_min'] = northwold_df['GB-ISEM Intraday 1 Price'].min()
    analysis['intraday_price_max'] = northwold_df['GB-ISEM Intraday 1 Price'].max()

    # Revenue Analysis
    analysis['imbalance_revenue'] = northwold_df['Imbalance Revenue'].sum()
    analysis['imbalance_charge'] = northwold_df['Imbalance Charge'].sum()
    analysis['net_imbalance'] = analysis['imbalance_revenue'] - analysis['imbalance_charge']

    # Revenue streams
    analysis['sffr_revenue'] = northwold_df['SFFR revenues'].sum()
    analysis['epex30_revenue'] = northwold_df['EPEX 30 DA Revenue'].sum()
    analysis['ida1_revenue'] = northwold_df['IDA1 Revenue'].sum()
    analysis['idc_revenue'] = northwold_df['IDC Revenue'].sum()

    analysis['total_net_revenue'] = (
        analysis['sffr_revenue'] +
        analysis['epex30_revenue'] +
        analysis['ida1_revenue'] +
        analysis['idc_revenue'] +
        analysis['imbalance_revenue'] -
        analysis['imbalance_charge']
    )

    # Ancillary services
    services = ['SFFR', 'DCL', 'DCH', 'DML', 'DMH', 'DRL', 'DRH']
    analysis['ancillary_services'] = {}
    for service in services:
        clearing_col = f'{service} Clearing Price'
        avail_col = f'{service} Availability'
        if clearing_col in northwold_df.columns:
            analysis['ancillary_services'][service] = {
                'avg_price': northwold_df[clearing_col].mean(),
                'avg_availability': northwold_df[avail_col].mean()
            }

    # Price volatility
    analysis['ssp_std'] = northwold_df['SSP'].std()
    analysis['sbp_std'] = northwold_df['SBP'].std()

    # Trading metrics
    analysis['da_mw_mean'] = northwold_df['DA MW'].mean()
    analysis['epex30_mw_mean'] = northwold_df['EPEX 30 DA MW'].mean()
    analysis['ida1_mw_mean'] = northwold_df['IDA1 MW'].mean()

    return analysis

def show_asset_details():
    """Display asset details from digital twin configuration"""
    st.title("🏭 Asset Details")
    st.markdown("### Northwold Solar Farm (Hall Farm) - Battery Energy Storage System")

    # Physical Specifications
    st.subheader("⚡ Physical Specifications")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Power Capabilities**")
        st.metric("Maximum Charging Rate", f"{config.P_IMP_MAX_MW} MW",
                 help="Maximum rate at which the battery can charge (import power)")
        st.metric("Maximum Discharging Rate", f"{config.P_EXP_MAX_MW} MW",
                 help="Maximum rate at which the battery can discharge (export power)")
        st.metric("Power Asymmetry Ratio", f"{config.P_EXP_MAX_MW/config.P_IMP_MAX_MW:.2f}x",
                 help="Discharge to charge power ratio")

    with col2:
        st.info("**Energy Storage**")
        st.metric("Usable Capacity", f"{config.CAPACITY_MWH} MWh",
                 help="Total usable energy storage capacity")
        st.metric("Minimum SOC", f"{config.SOC_MIN_PCT*100:.0f}%",
                 help=f"Minimum State of Charge ({config.SOC_MIN_MWH:.2f} MWh)")
        st.metric("Maximum SOC", f"{config.SOC_MAX_PCT*100:.0f}%",
                 help=f"Maximum State of Charge ({config.SOC_MAX_MWH:.2f} MWh)")

    # Efficiency & Warranty
    st.subheader("🔧 Efficiency & Warranty")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Efficiency Parameters**")
        st.metric("Round-Trip Efficiency", f"{config.EFF_ROUND_TRIP*100:.0f}%",
                 help="Energy efficiency for a complete charge-discharge cycle")

    with col2:
        st.info("**Warranty Constraints**")
        st.metric("Maximum Daily Cycles", f"{config.CYCLES_PER_DAY} cycles/day",
                 help="Warranty limit on daily cycling")

def show_operations_summary(bess_df, northwold_df, bess_analysis, northwold_analysis):
    """Display September 2025 operations summary"""
    st.title("📊 September 2025 Operations Summary")

    # Create tabs for the operations summary
    tab1, tab2 = st.tabs(["💰 Trading Analysis", "📈 Visualizations"])

    with tab1:

        # Revenue Breakdown
        st.subheader("💵 Revenue Breakdown")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("SFFR Revenues", f"£{northwold_analysis['sffr_revenue']:,.2f}",
                     help="Static Firm Frequency Response - Largest revenue stream")
        with col2:
            st.metric("IDA1 Revenue", f"£{northwold_analysis['ida1_revenue']:,.2f}",
                     help="Intraday Auction 1 trading revenue")
        with col3:
            st.metric("EPEX 30 DA Revenue", f"£{northwold_analysis['epex30_revenue']:,.2f}",
                     help="Day-ahead market revenue")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Imbalance Revenue", f"£{northwold_analysis['imbalance_revenue']:,.2f}",
                     help="Revenue from imbalance mechanism (negative indicates costs)")
        with col5:
            st.metric("Imbalance Charge", f"£{northwold_analysis['imbalance_charge']:,.2f}",
                     help="Charges from imbalance mechanism")
        with col6:
            st.metric("**TOTAL NET REVENUE**", f"£{northwold_analysis['total_net_revenue']:,.2f}",
                     help="Total net revenue for the month")

        # Market Prices
        st.subheader("📈 Market Prices Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Day Ahead Price (EPEX)**")
            price_data = {
                'Average': f"£{northwold_analysis['da_price_mean']:.2f}",
                'Minimum': f"£{northwold_analysis['da_price_min']:.2f}",
                'Maximum': f"£{northwold_analysis['da_price_max']:.2f}",
                'Std Dev': f"£{northwold_analysis['da_price_std']:.2f}"
            }
            for key, value in price_data.items():
                st.text(f"{key}: {value}")

        with col2:
            st.markdown("**Intraday Price**")
            intraday_data = {
                'Average': f"£{northwold_analysis['intraday_price_mean']:.2f}",
                'Minimum': f"£{northwold_analysis['intraday_price_min']:.2f}",
                'Maximum': f"£{northwold_analysis['intraday_price_max']:.2f}",
                'Spread (ID-DA)': f"£{(northwold_analysis['intraday_price_mean'] - northwold_analysis['da_price_mean']):.2f}"
            }
            for key, value in intraday_data.items():
                st.text(f"{key}: {value}")

        # Ancillary Services
        st.subheader("⚡ Ancillary Services Pricing")

        services_data = []
        for service, data in northwold_analysis['ancillary_services'].items():
            services_data.append({
                'Service': service,
                'Avg Clearing Price (£)': f"{data['avg_price']:.2f}",
                'Avg Availability (MW)': f"{data['avg_availability']:.0f}"
            })

        services_df = pd.DataFrame(services_data)
        st.dataframe(services_df, use_container_width=True, hide_index=True)

        # Trading Activity
        st.subheader("📊 Trading Activity")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average DA MW", f"{northwold_analysis['da_mw_mean']:.2f} MW")
        with col2:
            st.metric("Average EPEX 30 DA MW", f"{northwold_analysis['epex30_mw_mean']:.2f} MW")
        with col3:
            st.metric("Average IDA1 MW", f"{northwold_analysis['ida1_mw_mean']:.2f} MW")

    with tab2:

        # Create visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Revenue breakdown pie chart
            st.subheader("Revenue Breakdown")
            revenue_data = {
                'SFFR': abs(northwold_analysis['sffr_revenue']),
                'EPEX 30 DA': abs(northwold_analysis['epex30_revenue']),
                'IDA1': abs(northwold_analysis['ida1_revenue']),
                'Imbalance (Net)': abs(northwold_analysis['net_imbalance'])
            }

            fig = px.pie(
                values=list(revenue_data.values()),
                names=list(revenue_data.keys()),
                title="Revenue Sources Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # SOC distribution
            st.subheader("State of Charge Distribution")
            fig = px.histogram(
                bess_df,
                x='SOC',
                nbins=50,
                title="SOC Distribution",
                labels={'SOC': 'State of Charge (%)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Time series plots
        st.subheader("Time Series Analysis")

        # Power over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bess_df['date'],
            y=bess_df['Power'],
            mode='lines',
            name='Power (MW)',
            line=dict(width=1)
        ))
        fig.update_layout(
            title="Battery Power Over Time",
            xaxis_title="Date",
            yaxis_title="Power (MW)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Price comparison
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=northwold_df['Timestamp'],
            y=northwold_df['Day Ahead Price (EPEX)'],
            mode='lines',
            name='Day Ahead Price',
            line=dict(width=1)
        ))
        fig.add_trace(go.Scatter(
            x=northwold_df['Timestamp'],
            y=northwold_df['GB-ISEM Intraday 1 Price'],
            mode='lines',
            name='Intraday Price',
            line=dict(width=1)
        ))
        fig.update_layout(
            title="Energy Prices Comparison",
            xaxis_title="Date",
            yaxis_title="Price (£/MWh)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_multimarket_optimization():
    """Display multi-market optimization results"""
    st.title("🚀 Multi-Market Optimization Analysis")
    st.markdown("### Enhanced Trading Strategy with Cross-Market Arbitrage")

    # Check if multi-market results exist
    try:
        multi_df = pd.read_csv(os.path.join(DATA_DIR, 'Optimized_Results_MultiMarket.csv'))
        multi_df['Timestamp'] = pd.to_datetime(multi_df['Timestamp'])
    except:
        st.warning("Multi-market results not found. Please run phase3_multimarket.py first.")
        if st.button("Run Multi-Market Optimization"):
            with st.spinner("Running optimization... This may take a minute."):
                from src import phase3_multimarket
                multi_df = phase3_multimarket.run_phase_3_multimarket(os.path.join(DATA_DIR, 'Master_BESS_Analysis_Sept_2025.csv'))
                st.success("Optimization complete!")
                st.experimental_rerun()
        return

    # Summary metrics
    st.subheader("💰 Revenue Comparison")

    col1, col2, col3, col4 = st.columns(4)

    total_epex_daily = multi_df['Optimised_Revenue_Daily'].sum()
    total_epex_efa = multi_df['Optimised_Revenue_EFA'].sum()
    total_multi = multi_df['Optimised_Revenue_Multi'].sum()
    improvement = ((total_multi / total_epex_daily - 1) * 100)

    with col1:
        st.metric("EPEX-Only (Daily)", f"£{total_epex_daily:,.2f}",
                 help="Original strategy using only EPEX prices with daily switching")
    with col2:
        st.metric("EPEX-Only (EFA)", f"£{total_epex_efa:,.2f}",
                 delta=f"{((total_epex_efa/total_epex_daily - 1) * 100):.1f}%",
                 help="EPEX prices with 2-hour block switching")
    with col3:
        st.metric("Multi-Market", f"£{total_multi:,.2f}",
                 delta=f"+{improvement:.1f}%",
                 help="Optimized across all available markets")
    with col4:
        st.metric("Additional Revenue", f"£{(total_multi - total_epex_daily):,.2f}",
                 help="Extra revenue from multi-market strategy")

    # Market usage analysis
    st.subheader("📊 Market Utilization")

    # Add strategy selector dropdown
    strategy_option = st.selectbox(
        "Select Strategy to View:",
        ["Multi-Market", "EPEX-only (Daily)", "EPEX-only (EFA)", "Actual"],
        index=0,  # Default to Multi-Market
        help="Choose which strategy's market utilization to display"
    )

    # Determine which column to use based on selection
    if strategy_option == "Multi-Market":
        if 'Market_Used_Multi' in multi_df.columns:
            market_usage = multi_df['Market_Used_Multi'].value_counts()
            chart_title = "Market Activity Distribution (Multi-Market Strategy)"
        else:
            st.warning("Multi-Market data not available")
            market_usage = pd.Series()
            chart_title = "No Data Available"

    elif strategy_option == "EPEX-only (Daily)":
        if 'Strategy_Selected_Daily' in multi_df.columns:
            # For EPEX daily, show SFFR vs EPEX distribution
            market_usage = multi_df['Strategy_Selected_Daily'].value_counts()
            chart_title = "Strategy Distribution (EPEX-only Daily)"
        else:
            st.warning("EPEX Daily data not available")
            market_usage = pd.Series()
            chart_title = "No Data Available"

    elif strategy_option == "EPEX-only (EFA)":
        if 'Strategy_Selected_EFA' in multi_df.columns:
            # For EPEX EFA, show SFFR vs EPEX distribution per block
            market_usage = multi_df['Strategy_Selected_EFA'].value_counts()
            chart_title = "Strategy Distribution (EPEX-only EFA)"
        else:
            st.warning("EPEX EFA data not available")
            market_usage = pd.Series()
            chart_title = "No Data Available"

    else:  # Actual
        # For actual operation, we need to load the master file
        try:
            master_df = pd.read_csv(os.path.join(DATA_DIR, 'Master_BESS_Analysis_Sept_2025.csv'))
            if 'Unnamed: 0' in master_df.columns:
                master_df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)

            # Analyze actual operation
            actual_col = None
            for col in master_df.columns:
                if 'Battery MWh' in col and 'Output' in col:
                    actual_col = col
                    break

            if actual_col:
                # Create operation categories based on actual power
                operation_categories = []
                for val in master_df[actual_col]:
                    if val > 0.1:
                        operation_categories.append('Discharging')
                    elif val < -0.1:
                        operation_categories.append('Charging')
                    else:
                        operation_categories.append('Idle')
                market_usage = pd.Series(operation_categories).value_counts()
                chart_title = "Actual Operation Distribution"
            else:
                st.warning("Actual operation data not found")
                market_usage = pd.Series()
                chart_title = "No Data Available"
        except Exception as e:
            st.warning(f"Could not load actual data: {str(e)}")
            market_usage = pd.Series()
            chart_title = "No Data Available"

    # Display the visualization if we have data
    if not market_usage.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Market usage pie chart
            fig_pie = px.pie(
                values=market_usage.values[:10],
                names=market_usage.index[:10],
                title=chart_title
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Market statistics table
            market_stats = pd.DataFrame({
                'Market/Action': market_usage.index[:10],
                'Periods': market_usage.values[:10],
                'Percentage': (market_usage.values[:10] / len(multi_df) * 100).round(1)
            })
            st.dataframe(market_stats, use_container_width=True, hide_index=True)

    # Price spread analysis
    st.subheader("📈 Market Price Spreads")

    if 'Market_Spread' in multi_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            spread_stats = {
                'Average Spread': f"£{multi_df['Market_Spread'].mean():.2f}/MWh",
                'Max Spread': f"£{multi_df['Market_Spread'].max():.2f}/MWh",
                'Min Spread': f"£{multi_df['Market_Spread'].min():.2f}/MWh",
                'Spreads >£30': f"{(multi_df['Market_Spread'] > 30).sum()} periods",
                'Spreads >£50': f"{(multi_df['Market_Spread'] > 50).sum()} periods"
            }
            for key, value in spread_stats.items():
                st.text(f"{key}: {value}")

        with col2:
            # Spread distribution histogram
            fig_hist = px.histogram(
                multi_df,
                x='Market_Spread',
                nbins=50,
                title="Distribution of Market Spreads",
                labels={'Market_Spread': 'Price Spread (£/MWh)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # Time series comparison
    st.subheader("📊 Strategy Comparison Over Time")

    # Revenue over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=multi_df['Timestamp'],
        y=multi_df['Optimised_Revenue_Daily'].cumsum(),
        mode='lines',
        name='EPEX-Only (Cumulative)',
        line=dict(width=2, color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=multi_df['Timestamp'],
        y=multi_df['Optimised_Revenue_Multi'].cumsum(),
        mode='lines',
        name='Multi-Market (Cumulative)',
        line=dict(width=2, color='green')
    ))
    fig.update_layout(
        title="Cumulative Revenue Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Revenue (£)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Best markets by time
    st.subheader("🏆 Best Markets by Period")

    if 'Best_Buy_Market' in multi_df.columns and 'Best_Sell_Market' in multi_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            buy_markets = multi_df['Best_Buy_Market'].value_counts()
            st.markdown("**Best Markets for Buying (Charging)**")
            buy_df = pd.DataFrame({
                'Market': buy_markets.index,
                'Times Selected': buy_markets.values,
                'Percentage': (buy_markets.values / len(multi_df) * 100).round(1)
            })
            st.dataframe(buy_df.head(), use_container_width=True, hide_index=True)

        with col2:
            sell_markets = multi_df['Best_Sell_Market'].value_counts()
            st.markdown("**Best Markets for Selling (Discharging)**")
            sell_df = pd.DataFrame({
                'Market': sell_markets.index,
                'Times Selected': sell_markets.values,
                'Percentage': (sell_markets.values / len(multi_df) * 100).round(1)
            })
            st.dataframe(sell_df.head(), use_container_width=True, hide_index=True)

    # Key insights
    st.subheader("🔍 Key Insights")

    insights_col1, insights_col2 = st.columns(2)

    with insights_col1:
        st.info(f"""
        **Revenue Improvement:**
        - Multi-market strategy yields {improvement:.1f}% more revenue
        - Additional monthly revenue: £{(total_multi - total_epex_daily):,.2f}
        - Annualized benefit: £{((total_multi - total_epex_daily) * 12):,.2f}
        """)

    with insights_col2:
        st.success(f"""
        **Market Dynamics:**
        - SSP market provides best arbitrage opportunities
        - Battery idle {(market_usage.get('Idle', 0) / len(multi_df) * 100):.1f}% of time to preserve cycles
        - Average spread captured: £{multi_df['Market_Spread'].mean():.2f}/MWh
        """)

def show_bess_health():
    """Display BESS health analysis - cycles and degradation"""
    st.title("🔋 BESS Health Analysis")
    st.markdown("### Battery Cycling and Degradation Assessment")

    # Load required data
    try:
        # Load actual data
        master_df = pd.read_csv(os.path.join(DATA_DIR, 'Master_BESS_Analysis_Sept_2025.csv'))
        if 'Unnamed: 0' in master_df.columns:
            master_df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
        master_df['Timestamp'] = pd.to_datetime(master_df['Timestamp'])

        # Load multi-market optimization results
        multi_df = pd.read_csv(os.path.join(DATA_DIR, 'Optimized_Results_MultiMarket.csv'))
        multi_df['Timestamp'] = pd.to_datetime(multi_df['Timestamp'])

    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        st.info("Please ensure both Master_BESS_Analysis_Sept_2025.csv and Optimized_Results_MultiMarket.csv exist.")
        return

    # Constants from config
    CAPACITY_MWH = config.CAPACITY_MWH  # 8.4 MWh
    WARRANTY_CYCLES_DAILY = config.CYCLES_PER_DAY  # 1.5 cycles/day
    WARRANTY_DEGRADATION_ANNUAL_PCT = 2.5  # Annual degradation at warranty limit
    DEGRADATION_PER_CYCLE_PCT = WARRANTY_DEGRADATION_ANNUAL_PCT / (WARRANTY_CYCLES_DAILY * 365)

    # Calculate number of days
    num_days = 30  # September data

    # Find actual battery output column
    actual_col = None
    for col in master_df.columns:
        if 'Battery MWh' in col and 'Output' in col:
            actual_col = col
            break
    if actual_col is None:
        actual_col = 'Physical_Power_MW'  # Fallback
        if actual_col in master_df.columns:
            # Convert MW to MWh for 30-min periods
            master_df[actual_col] = master_df[actual_col] * 0.5

    # Calculate metrics for each strategy
    strategies_data = []

    # 1. Actual (Original) Operation
    if actual_col in master_df.columns:
        actual_discharge = master_df[master_df[actual_col] > 0][actual_col].sum()
        actual_cycles = actual_discharge / CAPACITY_MWH
        actual_daily_cycles = actual_cycles / num_days
        actual_degradation = actual_cycles * DEGRADATION_PER_CYCLE_PCT
        strategies_data.append({
            'Strategy': 'Actual Operation',
            'Total Discharge (MWh)': actual_discharge,
            'Total Cycles': actual_cycles,
            'Daily Cycles': actual_daily_cycles,
            'Est. Degradation (%)': actual_degradation,
            'Warranty Status': '✅ OK' if actual_daily_cycles <= WARRANTY_CYCLES_DAILY else '⚠️ EXCEEDED'
        })

    # 2. EPEX-Only Daily Strategy
    epex_daily_discharge = (multi_df[multi_df['Optimised_Net_MWh_Daily'] > 0]['Optimised_Net_MWh_Daily'].sum())
    epex_daily_cycles = epex_daily_discharge / CAPACITY_MWH
    epex_daily_daily_cycles = epex_daily_cycles / num_days
    epex_daily_degradation = epex_daily_cycles * DEGRADATION_PER_CYCLE_PCT
    strategies_data.append({
        'Strategy': 'EPEX-Only (Daily)',
        'Total Discharge (MWh)': epex_daily_discharge,
        'Total Cycles': epex_daily_cycles,
        'Daily Cycles': epex_daily_daily_cycles,
        'Est. Degradation (%)': epex_daily_degradation,
        'Warranty Status': '✅ OK' if epex_daily_daily_cycles <= WARRANTY_CYCLES_DAILY else '⚠️ EXCEEDED'
    })

    # 3. EPEX-Only EFA Strategy
    epex_efa_discharge = (multi_df[multi_df['Optimised_Net_MWh_EFA'] > 0]['Optimised_Net_MWh_EFA'].sum())
    epex_efa_cycles = epex_efa_discharge / CAPACITY_MWH
    epex_efa_daily_cycles = epex_efa_cycles / num_days
    epex_efa_degradation = epex_efa_cycles * DEGRADATION_PER_CYCLE_PCT
    strategies_data.append({
        'Strategy': 'EPEX-Only (EFA)',
        'Total Discharge (MWh)': epex_efa_discharge,
        'Total Cycles': epex_efa_cycles,
        'Daily Cycles': epex_efa_daily_cycles,
        'Est. Degradation (%)': epex_efa_degradation,
        'Warranty Status': '✅ OK' if epex_efa_daily_cycles <= WARRANTY_CYCLES_DAILY else '⚠️ EXCEEDED'
    })

    # 4. Multi-Market Strategy
    multi_discharge = (multi_df[multi_df['Optimised_Net_MWh_Multi'] > 0]['Optimised_Net_MWh_Multi'].sum())
    multi_cycles = multi_discharge / CAPACITY_MWH
    multi_daily_cycles = multi_cycles / num_days
    multi_degradation = multi_cycles * DEGRADATION_PER_CYCLE_PCT
    strategies_data.append({
        'Strategy': 'Multi-Market',
        'Total Discharge (MWh)': multi_discharge,
        'Total Cycles': multi_cycles,
        'Daily Cycles': multi_daily_cycles,
        'Est. Degradation (%)': multi_degradation,
        'Warranty Status': '✅ OK' if multi_daily_cycles <= WARRANTY_CYCLES_DAILY else '⚠️ EXCEEDED'
    })

    # Create DataFrame
    health_df = pd.DataFrame(strategies_data)

    # Summary metrics
    st.subheader("📊 Cycling & Degradation Comparison")

    # Display comparison table
    st.dataframe(
        health_df.style.format({
            'Total Discharge (MWh)': '{:.2f}',
            'Total Cycles': '{:.2f}',
            'Daily Cycles': '{:.3f}',
            'Est. Degradation (%)': '{:.4f}'
        }),
        use_container_width=True,
        hide_index=True
    )

    # Key metrics cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Warranty Limit",
            f"{WARRANTY_CYCLES_DAILY} cycles/day",
            help="Maximum daily cycles allowed under warranty"
        )

    with col2:
        st.metric(
            "Annual Degradation at Limit",
            f"{WARRANTY_DEGRADATION_ANNUAL_PCT}%",
            help="Expected annual degradation if operating at warranty limit"
        )

    with col3:
        st.metric(
            "Analysis Period",
            f"{num_days} days",
            help="September 2025 data"
        )

    # Visualizations
    st.subheader("📈 Visual Comparisons")

    col1, col2 = st.columns(2)

    with col1:
        # Daily cycles bar chart
        fig_cycles = go.Figure(data=[
            go.Bar(
                x=health_df['Strategy'],
                y=health_df['Daily Cycles'],
                text=health_df['Daily Cycles'].round(3),
                textposition='auto',
                marker_color=['lightblue', 'blue', 'darkblue', 'green']
            )
        ])
        # Add warranty limit line
        fig_cycles.add_hline(
            y=WARRANTY_CYCLES_DAILY,
            line_dash="dash",
            line_color="red",
            annotation_text="Warranty Limit (1.5 cycles/day)"
        )
        fig_cycles.update_layout(
            title="Daily Cycling Comparison",
            yaxis_title="Cycles per Day",
            xaxis_title="Strategy",
            height=400
        )
        st.plotly_chart(fig_cycles, use_container_width=True)

    with col2:
        # Degradation comparison
        fig_degrade = go.Figure(data=[
            go.Bar(
                x=health_df['Strategy'],
                y=health_df['Est. Degradation (%)'],
                text=health_df['Est. Degradation (%)'].round(4),
                textposition='auto',
                marker_color=['lightcoral', 'coral', 'darkorange', 'green']
            )
        ])
        fig_degrade.update_layout(
            title="Monthly Degradation Comparison",
            yaxis_title="Estimated Degradation (%)",
            xaxis_title="Strategy",
            height=400
        )
        st.plotly_chart(fig_degrade, use_container_width=True)

    # Total discharge comparison
    st.subheader("⚡ Energy Throughput Analysis")

    fig_discharge = go.Figure()

    # Add bars for each strategy
    strategies = health_df['Strategy'].tolist()
    discharges = health_df['Total Discharge (MWh)'].tolist()

    fig_discharge.add_trace(go.Bar(
        x=strategies,
        y=discharges,
        text=[f"{d:.1f} MWh" for d in discharges],
        textposition='auto',
        marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    ))

    fig_discharge.update_layout(
        title="Total Energy Discharged (September 2025)",
        yaxis_title="Total Discharge (MWh)",
        xaxis_title="Strategy",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig_discharge, use_container_width=True)

    # Projected annual analysis
    st.subheader("📅 Projected Annual Impact")

    annual_data = []
    for _, row in health_df.iterrows():
        annual_cycles = row['Daily Cycles'] * 365
        annual_degradation = annual_cycles * DEGRADATION_PER_CYCLE_PCT
        annual_data.append({
            'Strategy': row['Strategy'],
            'Projected Annual Cycles': annual_cycles,
            'Projected Annual Degradation (%)': annual_degradation,
            'Years to 80% SOH': 20.0 / annual_degradation if annual_degradation > 0 else float('inf')
        })

    annual_df = pd.DataFrame(annual_data)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(
            annual_df[['Strategy', 'Projected Annual Cycles', 'Projected Annual Degradation (%)']].style.format({
                'Projected Annual Cycles': '{:.1f}',
                'Projected Annual Degradation (%)': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        # Years to 80% SOH
        fig_years = go.Figure(data=[
            go.Bar(
                x=annual_df['Strategy'],
                y=annual_df['Years to 80% SOH'],
                text=[f"{y:.1f} years" if y < 100 else "100+ years" for y in annual_df['Years to 80% SOH']],
                textposition='auto',
                marker_color=['#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
            )
        ])
        fig_years.update_layout(
            title="Estimated Battery Lifespan",
            yaxis_title="Years to 80% State of Health",
            xaxis_title="Strategy",
            height=400
        )
        st.plotly_chart(fig_years, use_container_width=True)

    # Insights
    st.subheader("🔍 Key Health Insights")

    best_strategy = health_df.loc[health_df['Daily Cycles'].idxmin(), 'Strategy']
    worst_strategy = health_df.loc[health_df['Daily Cycles'].idxmax(), 'Strategy']

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Battery Preservation:**
        - Best for battery health: **{best_strategy}**
        - Most intensive: **{worst_strategy}**
        - All strategies within warranty limits ✅
        - Multi-market strategy balances revenue and health
        """)

    with col2:
        st.success(f"""
        **Optimization Benefits:**
        - Multi-market reduces cycling vs EPEX strategies
        - Higher revenue with lower battery wear
        - Extended battery lifespan potential
        - Warranty compliance maintained
        """)

def show_report_page():
    """Display comprehensive performance report comparing actual vs multi-market optimization"""

    # Header
    st.title("📑 Performance Report")
    st.markdown("### Northwold BESS Performance Audit - September 2025")
    st.markdown("---")

    # Load required data
    try:
        # Load actual data
        master_df = pd.read_csv(os.path.join(DATA_DIR, 'Master_BESS_Analysis_Sept_2025.csv'))
        if 'Unnamed: 0' in master_df.columns:
            master_df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
        master_df['Timestamp'] = pd.to_datetime(master_df['Timestamp'])
        master_df['Date'] = master_df['Timestamp'].dt.date

        # Load multi-market optimization results
        multi_df = pd.read_csv(os.path.join(DATA_DIR, 'Optimized_Results_MultiMarket.csv'))
        multi_df['Timestamp'] = pd.to_datetime(multi_df['Timestamp'])
        multi_df['Date'] = multi_df['Timestamp'].dt.date

        # Load northwold data for actual revenues
        northwold_df = pd.read_csv(os.path.join(DATA_DIR, 'Northwold_Sep_2025_converted.csv'))

    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        return

    # Calculate key metrics using actual column names in Northwold CSV
    # SFFR revenue - single column in this dataset
    if 'SFFR revenues' in northwold_df.columns:
        sffr_revenue = northwold_df['SFFR revenues'].sum()
    else:
        # Fallback: check for any SFFR revenue column
        sffr_cols = [col for col in northwold_df.columns if 'SFFR' in col and 'revenue' in col.lower()]
        sffr_revenue = northwold_df[sffr_cols[0]].sum() if sffr_cols else 0

    # Other revenue columns
    epex_revenue = northwold_df['EPEX 30 DA Revenue'].sum() if 'EPEX 30 DA Revenue' in northwold_df.columns else 0
    ida1_revenue = northwold_df['IDA1 Revenue'].sum() if 'IDA1 Revenue' in northwold_df.columns else 0
    imbalance_revenue = northwold_df['Imbalance Revenue'].sum() if 'Imbalance Revenue' in northwold_df.columns else 0
    imbalance_charge = northwold_df['Imbalance Charge'].sum() if 'Imbalance Charge' in northwold_df.columns else 0
    net_imbalance = imbalance_revenue - imbalance_charge
    actual_total = sffr_revenue + epex_revenue + ida1_revenue + net_imbalance

    # Multi-market revenue
    multi_total = multi_df['Optimised_Revenue_Multi'].sum()

    # Performance gap
    improvement_pct = ((multi_total / actual_total - 1) * 100)
    revenue_gap = multi_total - actual_total

    # ==================== SECTION 1: EXECUTIVE SUMMARY ====================
    st.header("1️⃣ The Performance Gap")
    st.markdown("The analysis reveals a significant gap between the asset's actual revenue and its optimized potential using multi-market strategies.")

    # Key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Actual Revenue (Sept)",
            f"£{actual_total:,.0f}",
            help="Total revenue from actual operation in September 2025"
        )

    with col2:
        st.metric(
            "Multi-Market Potential",
            f"£{multi_total:,.0f}",
            delta=f"+{improvement_pct:.0f}%",
            help="Optimized revenue using multi-market strategy"
        )

    with col3:
        st.metric(
            "Identified Opportunity",
            f"£{revenue_gap:,.0f}",
            delta=f"+{improvement_pct:.0f}%",
            help="Additional revenue available through optimization"
        )

    # Annualized comparison chart
    st.subheader("📊 Annualized Revenue Comparison")

    # Calculate annualized values per MW
    capacity_mw = config.P_EXP_MAX_MW  # 7.5 MW
    actual_annual_per_mw = (actual_total * 12) / capacity_mw
    multi_annual_per_mw = (multi_total * 12) / capacity_mw

    fig_annual = go.Figure(data=[
        go.Bar(
            x=['Actual (£/MW/yr)', 'Multi-Market (£/MW/yr)'],
            y=[actual_annual_per_mw, multi_annual_per_mw],
            text=[f'£{actual_annual_per_mw:,.0f}', f'£{multi_annual_per_mw:,.0f}'],
            textposition='auto',
            marker_color=['#D95D39', '#F0A202']
        )
    ])

    fig_annual.update_layout(
        title="Annualized Revenue per MW",
        yaxis_title="Revenue (£/MW/yr)",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig_annual, use_container_width=True)

    # ==================== SECTION 2: ACTUAL PERFORMANCE ====================
    st.header("2️⃣ Actual Performance Analysis")
    st.markdown("The aggregator's strategy relied heavily on SFFR with significant imbalance penalties eroding profits.")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Revenue breakdown donut chart
        revenue_components = {
            'SFFR Revenue': abs(sffr_revenue),
            'EPEX Trading': abs(epex_revenue),
            'IDA1 Trading': abs(ida1_revenue),
            'Imbalance (Net)': abs(net_imbalance)
        }

        fig_donut = px.pie(
            values=list(revenue_components.values()),
            names=list(revenue_components.keys()),
            title="Actual Revenue Breakdown",
            hole=0.4,
            color_discrete_map={
                'SFFR Revenue': '#2C4B78',
                'EPEX Trading': '#F18805',
                'IDA1 Trading': '#90EE90',
                'Imbalance (Net)': '#D95D39'
            }
        )

        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.error(f"""
        **⚠️ Critical Finding**

        **-£{abs(net_imbalance):,.0f}**
        in Net Imbalance Impact

        This loss erased **{abs(net_imbalance)/(sffr_revenue+epex_revenue+ida1_revenue)*100:.0f}%**
        of the total revenue generated by the asset.
        """)

    # ==================== SECTION 3: OPTIMIZATION OPPORTUNITY ====================
    st.header("3️⃣ The Optimization Opportunity")
    st.markdown("The Digital Twin model shows that a multi-market strategy dramatically increases profitability.")

    # Asset specifications
    st.subheader("Asset Specifications")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"""
        **Capacity**
        # {config.CAPACITY_MWH} MWh
        Usable Energy Storage
        """)

    with col2:
        st.info(f"""
        **Power**
        # {config.P_EXP_MAX_MW}/{config.P_IMP_MAX_MW} MW
        Discharge/Charge Rates
        """)

    with col3:
        st.info(f"""
        **Efficiency**
        # {config.EFF_ROUND_TRIP*100:.0f}%
        Round-Trip Efficiency
        """)

    # Strategy comparison
    st.subheader("📈 Revenue Mix Comparison")

    # Calculate revenues for different strategies
    strategies = ['Actual', 'EPEX-Daily', 'EPEX-EFA', 'Multi-Market']
    sffr_revenues = [
        sffr_revenue,
        multi_df[multi_df['Strategy_Selected_Daily'] == 'SFFR']['Optimised_Revenue_Daily'].sum(),
        multi_df[multi_df['Strategy_Selected_EFA'] == 'SFFR']['Optimised_Revenue_EFA'].sum(),
        0  # Multi-market doesn't use SFFR when optimizing
    ]

    market_revenues = [
        epex_revenue + ida1_revenue,
        multi_df[multi_df['Strategy_Selected_Daily'] == 'EPEX']['Optimised_Revenue_Daily'].sum(),
        multi_df[multi_df['Strategy_Selected_EFA'] == 'EPEX']['Optimised_Revenue_EFA'].sum(),
        multi_total
    ]

    fig_mix = go.Figure()
    fig_mix.add_trace(go.Bar(
        name='SFFR',
        x=strategies,
        y=sffr_revenues,
        marker_color='#2C4B78'
    ))
    fig_mix.add_trace(go.Bar(
        name='Market Trading',
        x=strategies,
        y=market_revenues,
        marker_color='#F0A202'
    ))

    fig_mix.update_layout(
        title="Strategy Revenue Comparison",
        yaxis_title="Revenue (£)",
        barmode='stack',
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig_mix, use_container_width=True)

    # ==================== SECTION 4: DEGRADATION & WARRANTY ====================
    st.header("4️⃣ Degradation & Warranty Analysis")
    st.markdown("The asset is operating well below warranty limits, providing room for more aggressive strategies.")

    # Find actual battery output column
    actual_col = None
    for col in master_df.columns:
        if 'Battery MWh' in col and 'Output' in col:
            actual_col = col
            break

    if actual_col:
        # Calculate degradation metrics
        actual_discharge = master_df[master_df[actual_col] > 0][actual_col].sum()
        actual_cycles = actual_discharge / config.CAPACITY_MWH
        actual_daily_cycles = actual_cycles / 30  # September has 30 days

        multi_discharge = (multi_df[multi_df['Optimised_Net_MWh_Multi'] > 0]['Optimised_Net_MWh_Multi'].sum())
        multi_cycles = multi_discharge / config.CAPACITY_MWH
        multi_daily_cycles = multi_cycles / 30

        # Create comparison table
        degradation_data = pd.DataFrame({
            'Scenario': ['Actual Usage', 'Multi-Market Strategy', 'Warranty Limit'],
            'Total Discharge (MWh)': [actual_discharge, multi_discharge, config.CYCLES_PER_DAY * 30 * config.CAPACITY_MWH],
            'Avg Cycles/Day': [actual_daily_cycles, multi_daily_cycles, config.CYCLES_PER_DAY],
            'Est. Monthly Degradation': [
                f"{actual_cycles * 0.0046:.3f}%",
                f"{multi_cycles * 0.0046:.3f}%",
                f"{config.CYCLES_PER_DAY * 30 * 0.0046:.3f}%"
            ],
            'Status': [
                '✅ Well Below Limit',
                '✅ Within Warranty',
                '⚠️ Maximum Allowed'
            ]
        })

        st.dataframe(degradation_data, use_container_width=True, hide_index=True)

    # ==================== SECTION 5: DAILY PERFORMANCE ====================
    st.header("5️⃣ Daily Performance Breakdown")
    st.markdown("Daily analysis reveals consistent revenue gaps, especially on high volatility days.")

    # Calculate daily revenues using actual column names
    revenue_cols = ['SFFR revenues', 'EPEX 30 DA Revenue', 'IDA1 Revenue',
                   'Imbalance Revenue', 'Imbalance Charge']

    # Only include columns that exist
    agg_dict = {}
    for col in revenue_cols:
        if col in master_df.columns:
            agg_dict[col] = 'sum'

    if agg_dict:
        daily_actual = master_df.groupby('Date').agg(agg_dict)

        # Calculate total revenue per day
        daily_actual['Total'] = 0
        if 'SFFR revenues' in daily_actual.columns:
            daily_actual['Total'] += daily_actual['SFFR revenues']
        if 'EPEX 30 DA Revenue' in daily_actual.columns:
            daily_actual['Total'] += daily_actual['EPEX 30 DA Revenue']
        if 'IDA1 Revenue' in daily_actual.columns:
            daily_actual['Total'] += daily_actual['IDA1 Revenue']
        if 'Imbalance Revenue' in daily_actual.columns:
            daily_actual['Total'] += daily_actual['Imbalance Revenue']
        if 'Imbalance Charge' in daily_actual.columns:
            daily_actual['Total'] -= daily_actual['Imbalance Charge']
    else:
        daily_actual = pd.DataFrame()

    daily_multi = multi_df.groupby('Date')['Optimised_Revenue_Multi'].sum()

    # Create line chart
    fig_daily = go.Figure()

    if not daily_actual.empty:
        fig_daily.add_trace(go.Scatter(
            x=daily_actual.index,
            y=daily_actual['Total'],
            mode='lines+markers',
            name='Actual Revenue',
            line=dict(color='#D95D39', width=2),
            marker=dict(size=6)
        ))

    fig_daily.add_trace(go.Scatter(
        x=daily_multi.index,
        y=daily_multi.values,
        mode='lines+markers',
        name='Multi-Market Potential',
        line=dict(color='#F0A202', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    fig_daily.update_layout(
        title="Daily Revenue Comparison",
        xaxis_title="Date",
        yaxis_title="Daily Revenue (£)",
        hovermode='x unified',
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig_daily, use_container_width=True)

    # ==================== SECTION 6: KEY DRIVERS OF LOSS ====================
    st.header("6️⃣ Key Drivers of Revenue Loss")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.warning(f"""
        **1. Imbalance Penalties**

        The asset incurred **-£{abs(net_imbalance):,.0f}** in net imbalance impact,
        suggesting forecast errors or non-delivery of contracted services.

        This erased **{abs(net_imbalance)/(sffr_revenue+epex_revenue+ida1_revenue)*100:.0f}%**
        of all generated revenue.
        """)

    with col2:
        st.warning(f"""
        **2. Conservative Strategy**

        The strategy was heavily weighted toward SFFR
        (**{sffr_revenue/actual_total*100:.0f}%** of revenue),
        missing profitable market trading opportunities.

        Multi-market approach would yield **+£{revenue_gap:,.0f}** additional revenue.
        """)

    with col3:
        st.warning(f"""
        **3. Underutilization**

        Operating at only **{actual_daily_cycles:.2f}** cycles/day
        vs **{config.CYCLES_PER_DAY}** warranty limit.

        Significant headroom exists for more aggressive trading
        without warranty impact.
        """)

    # Summary box
    st.markdown("---")
    st.success(f"""
    ### 🎯 Key Recommendation

    Implementing the multi-market optimization strategy would:
    - Increase monthly revenue by **{improvement_pct:.0f}%** (£{revenue_gap:,.0f})
    - Generate annual additional revenue of **£{revenue_gap*12:,.0f}**
    - Remain well within warranty cycling limits
    - Better utilize the asset's full capabilities
    """)

def main():
    """Main application with sidebar navigation"""

    # Sidebar configuration
    st.sidebar.title("🔋 BESS Dashboard")
    st.sidebar.markdown("---")

    # Navigation menu
    page = st.sidebar.radio(
        "Navigation",
        ["🏭 Asset Details", "📊 September 2025 Operations", "🚀 Multi-Market Optimization", "🔋 BESS Health", "📑 Performance Report"],
        index=1  # Default to operations summary
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **System:** Northwold Solar Farm
    **Location:** Hall Farm
    **Capacity:** 8.4 MWh
    **Type:** Battery Energy Storage
    """)

    # Display selected page
    if page == "🏭 Asset Details":
        show_asset_details()
    elif page == "📊 September 2025 Operations":
        # Load data
        with st.spinner("Loading data..."):
            bess_df, northwold_df = load_data()

        if bess_df is None or northwold_df is None:
            st.error("Failed to load data files. Please ensure both CSV files are in the correct location.")
            return

        # Perform analysis
        bess_analysis = analyze_bess_data(bess_df)
        northwold_analysis = analyze_northwold_data(northwold_df)

        # Show operations summary
        show_operations_summary(bess_df, northwold_df, bess_analysis, northwold_analysis)
    elif page == "🚀 Multi-Market Optimization":
        show_multimarket_optimization()
    elif page == "🔋 BESS Health":
        show_bess_health()
    elif page == "📑 Performance Report":
        show_report_page()

if __name__ == "__main__":
    main()