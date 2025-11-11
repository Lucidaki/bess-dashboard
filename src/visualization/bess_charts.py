"""
BESS Visualization Module
Generate Plotly charts for BESS optimization analysis
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List
from datetime import datetime
import json


class BESSVisualizer:
    """
    Create interactive Plotly visualizations for BESS optimization results
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize BESS visualizer

        Args:
            theme: Plotly theme (default: plotly_white)
        """
        self.theme = theme
        self.colors = {
            'actual': '#1f77b4',      # Blue
            'optimal': '#2ca02c',     # Green
            'price': '#ff7f0e',       # Orange
            'positive': '#2ca02c',    # Green (discharge)
            'negative': '#d62728',    # Red (charge)
            'revenue': '#9467bd'      # Purple
        }

    def create_power_profile_chart(
        self,
        schedule_df: pd.DataFrame,
        title: str = "BESS Power Profile - Actual vs Optimal"
    ) -> go.Figure:
        """
        Create power profile comparison chart

        Args:
            schedule_df: Schedule dataframe with actual and optimal power
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Actual power
        fig.add_trace(go.Scatter(
            x=schedule_df['timestamp_utc'],
            y=schedule_df['actual_power_mw'],
            name='Actual Power',
            line=dict(color=self.colors['actual'], width=2),
            mode='lines'
        ))

        # Optimal power
        fig.add_trace(go.Scatter(
            x=schedule_df['timestamp_utc'],
            y=schedule_df['optimal_power_mw'],
            name='Optimal Power',
            line=dict(color=self.colors['optimal'], width=2, dash='dash'),
            mode='lines'
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Time (UTC)",
            yaxis_title="Power (MW)",
            template=self.theme,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Add annotations for charge/discharge
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text="<b>Positive = Discharge (Export)<br>Negative = Charge (Import)</b>",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )

        return fig

    def create_soc_curve_chart(
        self,
        schedule_df: pd.DataFrame,
        soc_limits: Dict[str, float],
        title: str = "BESS State of Charge - Actual vs Optimal"
    ) -> go.Figure:
        """
        Create SoC curve comparison chart

        Args:
            schedule_df: Schedule dataframe with actual and optimal SoC
            soc_limits: Dictionary with soc_min_percent and soc_max_percent
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Actual SoC
        fig.add_trace(go.Scatter(
            x=schedule_df['timestamp_utc'],
            y=schedule_df['actual_soc_percent'],
            name='Actual SoC',
            line=dict(color=self.colors['actual'], width=2),
            fill='tozeroy',
            fillcolor=f'rgba(31, 119, 180, 0.1)',
            mode='lines'
        ))

        # Optimal SoC
        fig.add_trace(go.Scatter(
            x=schedule_df['timestamp_utc'],
            y=schedule_df['optimal_soc_percent'],
            name='Optimal SoC',
            line=dict(color=self.colors['optimal'], width=2, dash='dash'),
            mode='lines'
        ))

        # SoC limits
        fig.add_hline(
            y=soc_limits['soc_max_percent'],
            line_dash="dot",
            line_color="red",
            annotation_text=f"Max SoC ({soc_limits['soc_max_percent']}%)",
            annotation_position="right"
        )

        fig.add_hline(
            y=soc_limits['soc_min_percent'],
            line_dash="dot",
            line_color="red",
            annotation_text=f"Min SoC ({soc_limits['soc_min_percent']}%)",
            annotation_position="right"
        )

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Time (UTC)",
            yaxis_title="State of Charge (%)",
            template=self.theme,
            hovermode='x unified',
            yaxis=dict(range=[0, 100]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def create_price_spread_chart(
        self,
        schedule_df: pd.DataFrame,
        title: str = "Market Price Profile & Arbitrage Opportunities"
    ) -> go.Figure:
        """
        Create price spread and arbitrage opportunity chart

        Args:
            schedule_df: Schedule dataframe with prices and power
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Price line
        fig.add_trace(
            go.Scatter(
                x=schedule_df['timestamp_utc'],
                y=schedule_df['price_gbp_mwh'],
                name='Market Price',
                line=dict(color=self.colors['price'], width=2),
                mode='lines'
            ),
            secondary_y=False
        )

        # Optimal power as bar chart
        colors = [self.colors['positive'] if p > 0 else self.colors['negative']
                  for p in schedule_df['optimal_power_mw']]

        fig.add_trace(
            go.Bar(
                x=schedule_df['timestamp_utc'],
                y=schedule_df['optimal_power_mw'],
                name='Optimal Action',
                marker_color=colors,
                opacity=0.6
            ),
            secondary_y=True
        )

        # Layout
        fig.update_xaxes(title_text="Time (UTC)")
        fig.update_yaxes(title_text="Price (£/MWh)", secondary_y=False)
        fig.update_yaxes(title_text="Power (MW)", secondary_y=True)

        fig.update_layout(
            title=title,
            template=self.theme,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Add annotation
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text="<b>Green bars = Discharge (high prices)<br>Red bars = Charge (low prices)</b>",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )

        return fig

    def create_revenue_comparison_chart(
        self,
        optimization_summary: Dict,
        title: str = "Revenue Comparison - Actual vs Optimal"
    ) -> go.Figure:
        """
        Create revenue comparison bar chart

        Args:
            optimization_summary: Optimization summary dictionary
            title: Chart title

        Returns:
            Plotly Figure object
        """
        actual_revenue = optimization_summary['actual_revenue_gbp']
        optimal_revenue = optimization_summary['optimal_revenue_gbp']
        variance = optimization_summary['revenue_variance_gbp']

        # Create bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=['Actual Revenue', 'Optimal Revenue', 'Lost Opportunity'],
            y=[actual_revenue, optimal_revenue, variance],
            marker_color=[self.colors['actual'], self.colors['optimal'], self.colors['negative']],
            text=[f'£{actual_revenue:,.2f}', f'£{optimal_revenue:,.2f}', f'£{variance:,.2f}'],
            textposition='outside'
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray")

        # Layout
        fig.update_layout(
            title=title,
            yaxis_title="Revenue (£)",
            template=self.theme,
            showlegend=False
        )

        return fig

    def create_market_capture_gauge(
        self,
        market_capture_ratio: float,
        title: str = "Market Capture Ratio"
    ) -> go.Figure:
        """
        Create market capture ratio gauge chart

        Args:
            market_capture_ratio: Market capture ratio (%)
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Determine color based on performance
        if market_capture_ratio >= 90:
            color = "green"
        elif market_capture_ratio >= 70:
            color = "yellow"
        elif market_capture_ratio >= 50:
            color = "orange"
        else:
            color = "red"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=market_capture_ratio,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 24}},
            delta={'reference': 100, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, 100], 'ticksuffix': '%'},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(255, 0, 0, 0.2)"},
                    {'range': [50, 70], 'color': "rgba(255, 165, 0, 0.2)"},
                    {'range': [70, 90], 'color': "rgba(255, 255, 0, 0.2)"},
                    {'range': [90, 100], 'color': "rgba(0, 255, 0, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig.update_layout(
            template=self.theme,
            height=400
        )

        return fig

    def create_cycle_utilization_chart(
        self,
        optimization_summary: Dict,
        title: str = "Cycle Utilization"
    ) -> go.Figure:
        """
        Create cycle utilization bar chart

        Args:
            optimization_summary: Optimization summary dictionary
            title: Chart title

        Returns:
            Plotly Figure object
        """
        actual_cycles = optimization_summary['actual_performance']['cycles_used']
        optimal_cycles = optimization_summary['cycles_used']
        # Calculate max allowed cycles from daily limit and duration
        max_daily_cycles = optimization_summary['max_daily_cycles']
        duration_days = optimization_summary['duration_days']
        max_cycles = max_daily_cycles * duration_days

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=['Actual Cycles', 'Optimal Cycles', 'Max Allowed Cycles'],
            y=[actual_cycles, optimal_cycles, max_cycles],
            marker_color=[self.colors['actual'], self.colors['optimal'], 'lightgray'],
            text=[f'{actual_cycles:.2f}', f'{optimal_cycles:.2f}', f'{max_cycles:.2f}'],
            textposition='outside'
        ))

        # Layout
        fig.update_layout(
            title=title,
            yaxis_title="Cycles",
            template=self.theme,
            showlegend=False
        )

        return fig

    def create_dashboard_summary(
        self,
        schedule_df: pd.DataFrame,
        optimization_summary: Dict,
        soc_limits: Dict[str, float]
    ) -> go.Figure:
        """
        Create comprehensive dashboard with multiple subplots

        Args:
            schedule_df: Schedule dataframe
            optimization_summary: Optimization summary dictionary
            soc_limits: SoC limits dictionary

        Returns:
            Plotly Figure object with subplots
        """
        # Create subplots (2 rows, 2 columns)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Power Profile',
                'State of Charge',
                'Price & Arbitrage Opportunities',
                'Revenue Comparison'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # Row 1, Col 1: Power Profile
        fig.add_trace(
            go.Scatter(
                x=schedule_df['timestamp_utc'],
                y=schedule_df['actual_power_mw'],
                name='Actual Power',
                line=dict(color=self.colors['actual'], width=1.5),
                legendgroup='power'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=schedule_df['timestamp_utc'],
                y=schedule_df['optimal_power_mw'],
                name='Optimal Power',
                line=dict(color=self.colors['optimal'], width=1.5, dash='dash'),
                legendgroup='power'
            ),
            row=1, col=1
        )

        # Row 1, Col 2: SoC
        fig.add_trace(
            go.Scatter(
                x=schedule_df['timestamp_utc'],
                y=schedule_df['actual_soc_percent'],
                name='Actual SoC',
                line=dict(color=self.colors['actual'], width=1.5),
                legendgroup='soc'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=schedule_df['timestamp_utc'],
                y=schedule_df['optimal_soc_percent'],
                name='Optimal SoC',
                line=dict(color=self.colors['optimal'], width=1.5, dash='dash'),
                legendgroup='soc'
            ),
            row=1, col=2
        )

        # Row 2, Col 1: Price & Power
        fig.add_trace(
            go.Scatter(
                x=schedule_df['timestamp_utc'],
                y=schedule_df['price_gbp_mwh'],
                name='Price',
                line=dict(color=self.colors['price'], width=1.5),
                legendgroup='price'
            ),
            row=2, col=1, secondary_y=False
        )

        colors = [self.colors['positive'] if p > 0 else self.colors['negative']
                  for p in schedule_df['optimal_power_mw']]
        fig.add_trace(
            go.Bar(
                x=schedule_df['timestamp_utc'],
                y=schedule_df['optimal_power_mw'],
                name='Optimal Action',
                marker_color=colors,
                opacity=0.5,
                legendgroup='action'
            ),
            row=2, col=1, secondary_y=True
        )

        # Row 2, Col 2: Revenue Comparison
        actual_revenue = optimization_summary['actual_revenue_gbp']
        optimal_revenue = optimization_summary['optimal_revenue_gbp']
        variance = optimization_summary['revenue_variance_gbp']

        fig.add_trace(
            go.Bar(
                x=['Actual', 'Optimal', 'Variance'],
                y=[actual_revenue, optimal_revenue, variance],
                marker_color=[self.colors['actual'], self.colors['optimal'], self.colors['negative']],
                text=[f'£{actual_revenue:,.0f}', f'£{optimal_revenue:,.0f}', f'£{variance:,.0f}'],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=2
        )

        # Update axes labels
        fig.update_xaxes(title_text="Time (UTC)", row=1, col=1)
        fig.update_xaxes(title_text="Time (UTC)", row=1, col=2)
        fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
        fig.update_xaxes(title_text="", row=2, col=2)

        fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
        fig.update_yaxes(title_text="SoC (%)", row=1, col=2)
        fig.update_yaxes(title_text="Price (£/MWh)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Power (MW)", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Revenue (£)", row=2, col=2)

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template=self.theme,
            title_text="BESS Optimization Dashboard Summary",
            title_x=0.5
        )

        return fig

    def save_chart(self, fig: go.Figure, filename: str, format: str = 'html'):
        """
        Save chart to file

        Args:
            fig: Plotly Figure object
            filename: Output filename
            format: Output format ('html', 'png', 'svg', 'json')
        """
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'svg':
            fig.write_image(filename)
        elif format == 'json':
            fig.write_json(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
