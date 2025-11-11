"""
Real-Time Model Performance Evaluation Dashboard
=================================================

Provides interactive HTML dashboard for monitoring model performance,
retraining events, and system health during live trading.

Author: Production ML System
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class EvaluationDashboard:
    """
    Generates real-time HTML dashboards for model performance monitoring.
    """
    
    def __init__(self, performance_tracker: Any, retraining_manager: Any,
                 model_updater: Any, output_dir: str = "./monitoring/dashboards"):
        self.performance_tracker = performance_tracker
        self.retraining_manager = retraining_manager
        self.model_updater = model_updater
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Evaluation dashboard initialized")
    
    def generate_dashboard(self, save_html: bool = True) -> str:
        """
        Generate comprehensive performance dashboard.
        
        Args:
            save_html: If True, save dashboard as HTML file
            
        Returns:
            HTML string of dashboard
        """
        logger.info("Generating performance dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Prediction Accuracy Over Time',
                'Win Rate & Profitability',
                'Execution Quality Metrics',
                'Model Confidence Calibration',
                'Retraining Events Timeline',
                'Feature Distribution Drift',
                'Trade Outcomes Distribution',
                'System Health Status'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Get data
        perf_history = list(self.performance_tracker.performance_history)
        current_metrics = self.performance_tracker.get_current_metrics()
        retrain_stats = self.retraining_manager.get_retraining_stats()
        
        if not perf_history:
            logger.warning("No performance history available")
            return self._generate_empty_dashboard()
        
        # 1. Accuracy Over Time
        timestamps = [m.timestamp for m in perf_history]
        accuracies = [m.prediction_accuracy for m in perf_history]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=accuracies,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add baseline if available
        if self.performance_tracker.baseline_metrics:
            baseline_acc = self.performance_tracker.baseline_metrics.prediction_accuracy
            fig.add_hline(
                y=baseline_acc,
                line_dash="dash",
                line_color="green",
                annotation_text="Baseline",
                row=1, col=1
            )
        
        # 2. Win Rate & Profitability
        win_rates = [m.win_rate for m in perf_history]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=win_rates,
                mode='lines+markers',
                name='Win Rate',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # 3. Execution Quality (Latency & Slippage)
        latencies = [m.avg_execution_latency_ms for m in perf_history]
        slippages = [m.avg_slippage_pct * 100 for m in perf_history]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=latencies,
                mode='lines+markers',
                name='Latency (ms)',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        # Secondary axis for slippage
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=slippages,
                mode='lines+markers',
                name='Slippage (%)',
                line=dict(color='red', width=2),
                yaxis='y2'
            ),
            row=2, col=1
        )
        
        # 4. Confidence Calibration
        calibrations = [m.model_confidence_calibration for m in perf_history]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=calibrations,
                mode='lines+markers',
                name='Calibration Score',
                line=dict(color='purple', width=2),
                fill='tozeroy'
            ),
            row=2, col=2
        )
        
        # 5. Retraining Events
        if self.retraining_manager.retraining_history:
            retrain_times = [datetime.fromisoformat(e['timestamp']) 
                           for e in self.retraining_manager.retraining_history]
            retrain_samples = [e['training_samples'] 
                             for e in self.retraining_manager.retraining_history]
            
            fig.add_trace(
                go.Scatter(
                    x=retrain_times,
                    y=retrain_samples,
                    mode='markers',
                    name='Retraining Events',
                    marker=dict(size=15, color='red', symbol='star'),
                    text=[f"Retrain #{e['retrain_number']}" 
                         for e in self.retraining_manager.retraining_history],
                    hovertemplate='<b>%{text}</b><br>Samples: %{y}<br>%{x}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # 6. Feature Drift Detection
        drift_results = self.performance_tracker.detect_distribution_drift()
        if drift_results:
            features = list(drift_results.keys())[:10]  # Top 10
            p_values = [drift_results[f]['p_value'] for f in features]
            
            colors = ['red' if drift_results[f]['drift_detected'] else 'green' 
                     for f in features]
            
            fig.add_trace(
                go.Bar(
                    x=features,
                    y=p_values,
                    marker_color=colors,
                    name='Drift P-value',
                    hovertemplate='<b>%{x}</b><br>P-value: %{y:.4f}<extra></extra>'
                ),
                row=3, col=2
            )
            
            fig.add_hline(
                y=0.05,
                line_dash="dash",
                line_color="red",
                annotation_text="Drift Threshold",
                row=3, col=2
            )
        
        # 7. Trade Outcomes Distribution
        completed_trades = list(self.performance_tracker.completed_trades)
        if completed_trades:
            returns = [t.actual_return * 100 for t in completed_trades if t.actual_return is not None]
            
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=30,
                    name='Returns Distribution',
                    marker_color='steelblue',
                    opacity=0.7
                ),
                row=4, col=1
            )
        
        # 8. System Health Indicator
        if current_metrics:
            health_score = self._calculate_health_score(current_metrics)
            health_color = self._get_health_color(health_score)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=health_score * 100,
                    title={'text': "System Health"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': health_color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ),
                row=4, col=2
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Live Trading Performance Dashboard - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=1600,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Win Rate", row=1, col=2)
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)
        
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Calibration Score", row=2, col=2)
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Training Samples", row=3, col=1)
        
        fig.update_xaxes(title_text="Feature", row=3, col=2)
        fig.update_yaxes(title_text="P-value", row=3, col=2)
        
        fig.update_xaxes(title_text="Return (%)", row=4, col=1)
        fig.update_yaxes(title_text="Count", row=4, col=1)
        
        # Generate HTML
        html_content = self._generate_html_wrapper(fig, current_metrics, retrain_stats)
        
        if save_html:
            filepath = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(filepath, 'w') as f:
                f.write(html_content)
            logger.info(f"Dashboard saved to {filepath}")
            
            # Also save as latest
            latest_path = self.output_dir / "dashboard_latest.html"
            with open(latest_path, 'w') as f:
                f.write(html_content)
        
        return html_content
    
    def _generate_html_wrapper(self, fig: go.Figure, current_metrics: Any, 
                               retrain_stats: Dict[str, Any]) -> str:
        """Generate complete HTML with metrics summary"""
        
        # Get plotly HTML
        plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        # Build metrics summary
        metrics_html = self._build_metrics_summary(current_metrics, retrain_stats)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Live Trading Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-card .change {{
            font-size: 14px;
            margin-top: 5px;
        }}
        .positive {{ color: #10b981; }}
        .negative {{ color: #ef4444; }}
        .dashboard-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 20px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
        .alert {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
        .alert.success {{
            background: #d1fae5;
            border-left-color: #10b981;
        }}
        .alert.danger {{
            background: #fee2e2;
            border-left-color: #ef4444;
        }}
    </style>
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function(){{
            location.reload();
        }}, 30000);
    </script>
</head>
<body>
    <div class="header">
        <h1>🚀 Live Trading Performance Dashboard</h1>
        <p>Real-time model monitoring and evaluation | Auto-refreshes every 30 seconds</p>
    </div>
    
    {metrics_html}
    
    <div class="dashboard-container">
        {plot_html}
    </div>
    
    <div class="footer">
        Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
        Production ML Trading System v1.0
    </div>
</body>
</html>
"""
        return html
    
    def _build_metrics_summary(self, current_metrics: Any, 
                               retrain_stats: Dict[str, Any]) -> str:
        """Build HTML for metrics summary cards"""
        
        if not current_metrics:
            return '<div class="alert">No metrics available yet</div>'
        
        # Check for alerts
        alerts_html = ""
        recommendation = self.performance_tracker.get_retraining_recommendation()
        if recommendation['should_retrain']:
            reasons = '<br>'.join(recommendation['reasons'])
            alerts_html = f'''
            <div class="alert danger">
                <strong>⚠️ Retraining Recommended</strong><br>
                {reasons}
            </div>
            '''
        else:
            alerts_html = '''
            <div class="alert success">
                <strong>✓ System Operating Normally</strong><br>
                Model performance within acceptable thresholds
            </div>
            '''
        
        # Build metric cards
        cards_html = f'''
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Prediction Accuracy</h3>
                <div class="value">{current_metrics.prediction_accuracy:.1%}</div>
                <div class="change">Last {current_metrics.lookback_trades} trades</div>
            </div>
            
            <div class="metric-card">
                <h3>Win Rate</h3>
                <div class="value">{current_metrics.win_rate:.1%}</div>
                <div class="change">{current_metrics.profitable_trades} / {current_metrics.total_trades} profitable</div>
            </div>
            
            <div class="metric-card">
                <h3>Precision / Recall</h3>
                <div class="value">{current_metrics.precision:.3f} / {current_metrics.recall:.3f}</div>
                <div class="change">F1: {current_metrics.f1_score:.3f}</div>
            </div>
            
            <div class="metric-card">
                <h3>Execution Latency</h3>
                <div class="value">{current_metrics.avg_execution_latency_ms:.0f}ms</div>
                <div class="change">Average execution time</div>
            </div>
            
            <div class="metric-card">
                <h3>Slippage</h3>
                <div class="value">{current_metrics.avg_slippage_pct:.3%}</div>
                <div class="change">Average slippage</div>
            </div>
            
            <div class="metric-card">
                <h3>Model Confidence</h3>
                <div class="value">{current_metrics.model_confidence_calibration:.3f}</div>
                <div class="change">Calibration score (0-1)</div>
            </div>
            
            <div class="metric-card">
                <h3>Total Retrains</h3>
                <div class="value">{retrain_stats.get('total_retrains', 0)}</div>
                <div class="change">Last: {retrain_stats.get('last_retrain', 'Never')[:19] if retrain_stats.get('last_retrain') else 'Never'}</div>
            </div>
            
            <div class="metric-card">
                <h3>Training Buffer</h3>
                <div class="value">{retrain_stats.get('buffered_samples', 0)}</div>
                <div class="change">Samples ready for retraining</div>
            </div>
        </div>
        '''
        
        return alerts_html + cards_html
    
    def _calculate_health_score(self, metrics: Any) -> float:
        """Calculate overall system health score (0-1)"""
        scores = []
        
        # Accuracy score
        if metrics.prediction_accuracy >= 0.7:
            scores.append(1.0)
        elif metrics.prediction_accuracy >= 0.6:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        # Win rate score
        if metrics.win_rate >= 0.6:
            scores.append(1.0)
        elif metrics.win_rate >= 0.5:
            scores.append(0.8)
        else:
            scores.append(0.5)
        
        # Calibration score
        scores.append(metrics.model_confidence_calibration)
        
        # Execution quality (latency)
        if metrics.avg_execution_latency_ms < 100:
            scores.append(1.0)
        elif metrics.avg_execution_latency_ms < 500:
            scores.append(0.8)
        else:
            scores.append(0.6)
        
        # Slippage score
        if metrics.avg_slippage_pct < 0.005:
            scores.append(1.0)
        elif metrics.avg_slippage_pct < 0.02:
            scores.append(0.7)
        else:
            scores.append(0.5)
        
        return np.mean(scores)
    
    def _get_health_color(self, score: float) -> str:
        """Get color for health score"""
        if score >= 0.8:
            return "green"
        elif score >= 0.6:
            return "orange"
        else:
            return "red"
    
    def _generate_empty_dashboard(self) -> str:
        """Generate empty dashboard when no data available"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Live Trading Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .message {
            background: white;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="message">
        <h1>📊 Dashboard Initializing</h1>
        <p>Waiting for performance data...</p>
        <p style="color: #666; font-size: 14px;">
            Dashboard will appear after the first trades are completed
        </p>
    </div>
</body>
</html>
"""
    
    def export_metrics_csv(self) -> Path:
        """Export all performance metrics to CSV"""
        perf_history = list(self.performance_tracker.performance_history)
        
        if not perf_history:
            logger.warning("No metrics to export")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': m.timestamp,
            'accuracy': m.prediction_accuracy,
            'precision': m.precision,
            'recall': m.recall,
            'f1_score': m.f1_score,
            'win_rate': m.win_rate,
            'avg_execution_latency_ms': m.avg_execution_latency_ms,
            'avg_slippage_pct': m.avg_slippage_pct,
            'confidence_calibration': m.model_confidence_calibration,
            'total_trades': m.total_trades
        } for m in perf_history])
        
        filepath = self.output_dir / f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filepath, index=False)
        
        logger.info(f"Metrics exported to {filepath}")
        return filepath
