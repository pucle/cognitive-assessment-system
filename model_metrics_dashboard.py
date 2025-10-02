#!/usr/bin/env python3
"""
Model Metrics Dashboard for Cognitive Assessment System
Generates comprehensive visualization of key performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class ModelMetricsDashboard:
    """Dashboard for visualizing model performance metrics"""

    def __init__(self):
        self.metrics_data = {}
        self.timestamps = []
        self.generate_mock_data()

    def generate_mock_data(self):
        """Generate realistic mock data for all metrics"""
        np.random.seed(42)

        # Generate timestamps for last 30 days
        base_date = datetime.now() - timedelta(days=30)
        self.timestamps = [base_date + timedelta(hours=i) for i in range(24*30)]

        # WER (Word Error Rate) - Vietnamese ASR
        # Typical range: 5-25% for good ASR systems
        wer_base = 12.0
        wer_noise = np.random.normal(0, 2, len(self.timestamps))
        wer_trend = np.sin(np.arange(len(self.timestamps)) * 0.1) * 3
        self.metrics_data['WER'] = np.clip(wer_base + wer_noise + wer_trend, 5, 25)

        # Sensitivity (Tầng 1) - Classification sensitivity
        # Should be high for medical diagnosis
        sens_base = 92.0
        sens_noise = np.random.normal(0, 3, len(self.timestamps))
        self.metrics_data['Sensitivity_T1'] = np.clip(sens_base + sens_noise, 85, 98)

        # Specificity (Tầng 1) - Classification specificity
        # Should be high for medical diagnosis
        spec_base = 88.0
        spec_noise = np.random.normal(0, 4, len(self.timestamps))
        self.metrics_data['Specificity_T1'] = np.clip(spec_base + spec_noise, 80, 95)

        # AUC (Tầng 2) - Area Under Curve
        # Excellent: >0.9, Good: 0.8-0.9, Fair: 0.7-0.8
        auc_base = 0.915
        auc_noise = np.random.normal(0, 0.02, len(self.timestamps))
        self.metrics_data['AUC_T2'] = np.clip(auc_base + auc_noise, 0.85, 0.98)

        # MAE (Mean Absolute Error) - Regression error
        # Clinical acceptable: <4.0 for MMSE scale
        mae_base = 2.8
        mae_noise = np.random.normal(0, 0.5, len(self.timestamps))
        mae_trend = np.sin(np.arange(len(self.timestamps)) * 0.05) * 0.3
        self.metrics_data['MAE'] = np.clip(mae_base + mae_noise + mae_trend, 2.0, 4.5)

        # Processing Latency (ms)
        # Target: <500ms for real-time, <2000ms acceptable
        latency_base = 450
        latency_noise = np.random.normal(0, 80, len(self.timestamps))
        latency_spikes = np.random.exponential(200, len(self.timestamps)) * 0.1
        self.metrics_data['Latency'] = np.clip(latency_base + latency_noise + latency_spikes, 200, 1500)

        # System Uptime (%)
        # Target: >99.5% for production systems
        uptime_base = 99.7
        uptime_noise = np.random.normal(0, 0.2, len(self.timestamps))
        # Occasional downtime events
        downtime_events = np.random.choice([0, -5, -10], len(self.timestamps), p=[0.98, 0.015, 0.005])
        self.metrics_data['Uptime'] = np.clip(uptime_base + uptime_noise + downtime_events, 95, 100)

    def create_comprehensive_dashboard(self):
        """Create comprehensive dashboard with all metrics"""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Cognitive Assessment System - Performance Metrics Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        # Create subplot grid
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. WER Time Series
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_wer_timeseries(ax1)

        # 2. Classification Metrics (Sensitivity & Specificity)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_classification_metrics(ax2)

        # 3. AUC Gauge Chart
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_auc_gauge(ax3)

        # 4. MAE Time Series with Clinical Threshold
        ax4 = fig.add_subplot(gs[1, 1:3])
        self._plot_mae_timeseries(ax4)

        # 5. Processing Latency Distribution
        ax5 = fig.add_subplot(gs[1, 3])
        self._plot_latency_distribution(ax5)

        # 6. System Uptime
        ax6 = fig.add_subplot(gs[2, :2])
        self._plot_uptime_timeseries(ax6)

        # 7. Performance Summary Table
        ax7 = fig.add_subplot(gs[2, 2:])
        self._plot_performance_summary(ax7)

        # 8. Correlation Matrix
        ax8 = fig.add_subplot(gs[3, :2])
        self._plot_metrics_correlation(ax8)

        # 9. Trend Analysis
        ax9 = fig.add_subplot(gs[3, 2:])
        self._plot_trend_analysis(ax9)

        plt.tight_layout()
        plt.savefig('model_metrics_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_wer_timeseries(self, ax):
        """Plot WER time series with Vietnamese ASR focus"""
        wer_data = self.metrics_data['WER']
        timestamps = self.timestamps

        ax.plot(timestamps, wer_data, 'b-', linewidth=2, alpha=0.8, label='WER (%)')

        # Add rolling average
        rolling_avg = pd.Series(wer_data).rolling(window=24).mean()
        ax.plot(timestamps, rolling_avg, 'r-', linewidth=2, label='24h Average')

        # Target line
        ax.axhline(y=15, color='green', linestyle='--', alpha=0.7, label='Target (<15%)')

        ax.set_title('WER - Vietnamese ASR Performance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Word Error Rate (%)', fontsize=10)
        ax.set_xlabel('Time', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add latest value annotation
        latest_wer = wer_data[-1]
        ax.annotate(f'{latest_wer:.1f}%',
                   xy=(timestamps[-1], latest_wer),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9,
                           edgecolor='orange', linewidth=2),
                   fontsize=11, fontweight='bold', color='darkblue')

    def _plot_classification_metrics(self, ax):
        """Plot sensitivity and specificity comparison"""
        sens_data = self.metrics_data['Sensitivity_T1']
        spec_data = self.metrics_data['Specificity_T1']

        # Create box plots
        data = [sens_data[-168:], spec_data[-168:]]  # Last 7 days
        labels = ['Sensitivity\n(Tầng 1)', 'Specificity\n(Tầng 1)']

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       medianprops=dict(color='black', linewidth=2))

        # Color the boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title('Classification Metrics (Tầng 1)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance (%)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value annotations
        sens_median = np.median(sens_data[-168:])
        spec_median = np.median(spec_data[-168:])
        ax.text(1, sens_median + 2, f'{sens_median:.1f}%',
                ha='center', va='bottom', fontweight='bold',
                fontsize=12, color='darkblue',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
        ax.text(2, spec_median + 2, f'{spec_median:.1f}%',
                ha='center', va='bottom', fontweight='bold',
                fontsize=12, color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))

    def _plot_auc_gauge(self, ax):
        """Create AUC gauge chart"""
        auc_value = np.mean(self.metrics_data['AUC_T2'][-24:])  # Last 24 hours average

        # Create gauge background
        theta = np.linspace(np.pi, 0, 100)
        r = 1

        # Color zones
        ax.fill_between(np.linspace(0.7, 0.8, 50), 0, 0.3,
                       color='red', alpha=0.3, label='Poor (0.7-0.8)')
        ax.fill_between(np.linspace(0.8, 0.9, 50), 0, 0.3,
                       color='orange', alpha=0.3, label='Good (0.8-0.9)')
        ax.fill_between(np.linspace(0.9, 1.0, 50), 0, 0.3,
                       color='green', alpha=0.3, label='Excellent (0.9-1.0)')

        # AUC needle
        auc_angle = np.pi - (auc_value - 0.7) / 0.3 * np.pi
        ax.plot([0, np.cos(auc_angle)], [0, np.sin(auc_angle)],
               'k-', linewidth=4)
        ax.plot([0, np.cos(auc_angle)], [0, np.sin(auc_angle)],
               'r-', linewidth=2)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title('AUC Score (Tầng 2)', fontsize=12, fontweight='bold')

        # Add value text
        ax.text(0, -0.3, f'{auc_value:.3f}', ha='center', va='center',
               fontsize=16, fontweight='bold', color='darkblue',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                        edgecolor='blue', linewidth=2, alpha=0.9))

        # Add legend
        ax.legend(loc='upper right', fontsize=8)

    def _plot_mae_timeseries(self, ax):
        """Plot MAE with clinical thresholds"""
        mae_data = self.metrics_data['MAE']
        timestamps = self.timestamps

        ax.plot(timestamps, mae_data, 'purple', linewidth=2, alpha=0.8, label='MAE')

        # Clinical thresholds
        ax.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, label='Clinical Limit (4.0)')
        ax.axhline(y=3.0, color='orange', linestyle='--', alpha=0.7, label='Good Performance (3.0)')
        ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Excellent (2.0)')

        ax.fill_between(timestamps, 0, mae_data, alpha=0.1, color='purple')

        ax.set_title('MAE - Regression Performance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error', fontsize=10)
        ax.set_xlabel('Time', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add latest value
        latest_mae = mae_data[-1]
        ax.annotate(f'{latest_mae:.2f}',
                   xy=(timestamps[-1], latest_mae),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9,
                           edgecolor='orange', linewidth=2),
                   fontsize=12, fontweight='bold', color='darkred')

    def _plot_latency_distribution(self, ax):
        """Plot latency distribution with performance zones"""
        latency_data = self.metrics_data['Latency'][-168:]  # Last 7 days

        # Create histogram
        n, bins, patches = ax.hist(latency_data, bins=20, alpha=0.7,
                                 color='lightcoral', edgecolor='black')

        # Color zones
        for i, patch in enumerate(patches):
            if bins[i] < 500:  # Excellent
                patch.set_facecolor('green')
            elif bins[i] < 1000:  # Good
                patch.set_facecolor('orange')
            else:  # Poor
                patch.set_facecolor('red')

        ax.axvline(x=500, color='green', linestyle='--', alpha=0.7, label='Target (<500ms)')
        ax.axvline(x=1000, color='red', linestyle='--', alpha=0.7, label='Limit (<1000ms)')

        ax.set_title('Processing Latency Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Latency (ms)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()

        # Add statistics
        p25, p50, p75 = np.percentile(latency_data, [25, 50, 75])
        ax.text(0.02, 0.98, f'P25: {p25:.0f}ms', transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
        ax.text(0.02, 0.90, f'P50: {p50:.0f}ms', transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))
        ax.text(0.02, 0.82, f'P75: {p75:.0f}ms', transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.8))

    def _plot_uptime_timeseries(self, ax):
        """Plot system uptime with SLA requirements"""
        uptime_data = self.metrics_data['Uptime']
        timestamps = self.timestamps

        ax.plot(timestamps, uptime_data, 'darkgreen', linewidth=2, alpha=0.8)

        # SLA requirements
        ax.axhline(y=99.9, color='red', linestyle='--', alpha=0.7, label='99.9% SLA')
        ax.axhline(y=99.5, color='orange', linestyle='--', alpha=0.7, label='99.5% SLA')
        ax.axhline(y=99.0, color='yellow', linestyle='--', alpha=0.7, label='99.0% Target')

        ax.fill_between(timestamps, 99, uptime_data,
                       where=(uptime_data >= 99), color='green', alpha=0.2)
        ax.fill_between(timestamps, 95, uptime_data,
                       where=(uptime_data < 99), color='red', alpha=0.2)

        ax.set_title('System Uptime (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Uptime (%)', fontsize=10)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylim(95, 100.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add current uptime
        current_uptime = uptime_data[-1]
        ax.annotate(f'{current_uptime:.2f}%',
                   xy=(timestamps[-1], current_uptime),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9,
                           edgecolor='darkgreen', linewidth=2),
                   fontsize=12, fontweight='bold', color='darkgreen')

    def _plot_performance_summary(self, ax):
        """Create performance summary table"""
        ax.axis('off')

        # Calculate current values
        current_wer = np.mean(self.metrics_data['WER'][-24:])
        current_sens = np.mean(self.metrics_data['Sensitivity_T1'][-24:])
        current_spec = np.mean(self.metrics_data['Specificity_T1'][-24:])
        current_auc = np.mean(self.metrics_data['AUC_T2'][-24:])
        current_mae = np.mean(self.metrics_data['MAE'][-24:])
        current_latency = np.mean(self.metrics_data['Latency'][-24:])
        current_uptime = np.mean(self.metrics_data['Uptime'][-24:])

        # Calculate summary statistics
        summary_data = {
            'Metric': ['WER (ASR)', 'Sensitivity T1', 'Specificity T1',
                      'AUC T2', 'MAE', 'Latency', 'Uptime'],
            'Current': [
                f'{current_wer:.1f}%',
                f'{current_sens:.1f}%',
                f'{current_spec:.1f}%',
                f'{current_auc:.3f}',
                f'{current_mae:.2f}',
                f'{current_latency:.0f}ms',
                f'{current_uptime:.2f}%'
            ],
            'Target': ['<15%', '>90%', '>85%', '>0.9', '<4.0', '<500ms', '>99.5%'],
            'Status': [
                'Good' if current_wer < 15 else 'Needs Work',
                'Excellent' if current_sens > 90 else 'Good',
                'Good' if current_spec > 85 else 'Needs Work',
                'Excellent' if current_auc > 0.9 else 'Good',
                'Excellent' if current_mae < 4.0 else 'Good',
                'Good' if current_latency < 500 else 'Needs Work',
                'Excellent' if current_uptime > 99.5 else 'Good'
            ]
        }

        table_data = list(zip(summary_data['Metric'], summary_data['Current'],
                             summary_data['Target'], summary_data['Status']))

        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Current', 'Target', 'Status'],
                        cellLoc='center', loc='center',
                        colColours=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

        ax.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    def _plot_metrics_correlation(self, ax):
        """Plot correlation matrix of all metrics"""
        # Create dataframe for correlation
        df = pd.DataFrame(self.metrics_data)

        # Calculate correlation
        corr = df.corr()

        # Plot heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, ax=ax,
                   annot_kws={'size': 8})

        ax.set_title('Metrics Correlation Matrix', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=8)

    def _plot_trend_analysis(self, ax):
        """Plot trend analysis for key metrics"""
        key_metrics = ['WER', 'MAE', 'Latency', 'Uptime']
        colors = ['blue', 'purple', 'red', 'green']

        for i, (metric, color) in enumerate(zip(key_metrics, colors)):
            data = self.metrics_data[metric]
            # Calculate trend (simple moving average difference)
            ma_short = pd.Series(data).rolling(window=24).mean()
            ma_long = pd.Series(data).rolling(window=168).mean()  # 7 days
            trend = ma_short - ma_long

            ax.plot(self.timestamps, trend, color=color, linewidth=2,
                   label=f'{metric} Trend', alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('Trend Analysis (Short vs Long MA)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trend Value', fontsize=10)
        ax.set_xlabel('Time', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

def create_individual_plots():
    """Create individual detailed plots for each metric"""
    dashboard = ModelMetricsDashboard()

    # 1. Detailed WER Analysis
    plt.figure(figsize=(12, 6))
    plt.plot(dashboard.timestamps, dashboard.metrics_data['WER'],
            'b-', linewidth=2, alpha=0.7)
    plt.title('WER - Vietnamese ASR Performance (Detailed)', fontsize=14, fontweight='bold')
    plt.ylabel('Word Error Rate (%)')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=15, color='red', linestyle='--', label='Target')
    plt.legend()
    plt.tight_layout()
    plt.savefig('wer_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Clinical Metrics Focus
    plt.figure(figsize=(12, 6))
    plt.plot(dashboard.timestamps, dashboard.metrics_data['MAE'],
            'purple', linewidth=2, label='MAE')
    plt.axhline(y=4.0, color='red', linestyle='--', label='Clinical Limit')
    plt.axhline(y=2.5, color='green', linestyle='--', label='Excellent Performance')
    plt.title('MAE - Clinical Performance Focus', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clinical_mae_focus.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. System Reliability
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Uptime
    ax1.plot(dashboard.timestamps, dashboard.metrics_data['Uptime'],
            'darkgreen', linewidth=2)
    ax1.axhline(y=99.9, color='red', linestyle='--', label='99.9% SLA')
    ax1.set_title('System Uptime Reliability', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Uptime (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Latency
    ax2.plot(dashboard.timestamps, dashboard.metrics_data['Latency'],
            'red', linewidth=2)
    ax2.axhline(y=500, color='green', linestyle='--', label='Real-time Target')
    ax2.axhline(y=1000, color='orange', linestyle='--', label='Acceptable Limit')
    ax2.set_title('Processing Latency Performance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('system_reliability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    print("🔬 Creating Model Metrics Dashboard...")
    print("="*50)

    # Create dashboard
    dashboard = ModelMetricsDashboard()
    dashboard.create_comprehensive_dashboard()

    # Create individual plots
    create_individual_plots()

    print("✅ Dashboard created successfully!")
    print("📊 Files generated:")
    print("  • model_metrics_dashboard.png - Comprehensive dashboard")
    print("  • wer_detailed_analysis.png - WER detailed analysis")
    print("  • clinical_mae_focus.png - Clinical performance focus")
    print("  • system_reliability_analysis.png - System reliability")
    print("="*50)

if __name__ == "__main__":
    main()
