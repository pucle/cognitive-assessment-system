#!/usr/bin/env python3
"""
Individual Metrics Plots for Cognitive Assessment System
Creates separate plots for each metric with highlighted current values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class IndividualMetricsPlots:
    """Create individual plots for each metric"""

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
        wer_base = 12.0
        wer_noise = np.random.normal(0, 2, len(self.timestamps))
        wer_trend = np.sin(np.arange(len(self.timestamps)) * 0.1) * 3
        self.metrics_data['WER'] = np.clip(wer_base + wer_noise + wer_trend, 5, 25)

        # Sensitivity (Tầng 1)
        sens_base = 92.0
        sens_noise = np.random.normal(0, 3, len(self.timestamps))
        self.metrics_data['Sensitivity_T1'] = np.clip(sens_base + sens_noise, 85, 98)

        # Specificity (Tầng 1)
        spec_base = 88.0
        spec_noise = np.random.normal(0, 4, len(self.timestamps))
        self.metrics_data['Specificity_T1'] = np.clip(spec_base + spec_noise, 80, 95)

        # AUC (Tầng 2)
        auc_base = 0.915
        auc_noise = np.random.normal(0, 0.02, len(self.timestamps))
        self.metrics_data['AUC_T2'] = np.clip(auc_base + auc_noise, 0.85, 0.98)

        # MAE (Mean Absolute Error)
        mae_base = 2.8
        mae_noise = np.random.normal(0, 0.5, len(self.timestamps))
        mae_trend = np.sin(np.arange(len(self.timestamps)) * 0.05) * 0.3
        self.metrics_data['MAE'] = np.clip(mae_base + mae_noise + mae_trend, 2.0, 4.5)

        # Processing Latency (ms)
        latency_base = 450
        latency_noise = np.random.normal(0, 80, len(self.timestamps))
        latency_spikes = np.random.exponential(200, len(self.timestamps)) * 0.1
        self.metrics_data['Latency'] = np.clip(latency_base + latency_noise + latency_spikes, 200, 1500)

        # System Uptime (%)
        uptime_base = 99.7
        uptime_noise = np.random.normal(0, 0.2, len(self.timestamps))
        downtime_events = np.random.choice([0, -5, -10], len(self.timestamps), p=[0.98, 0.015, 0.005])
        self.metrics_data['Uptime'] = np.clip(uptime_base + uptime_noise + downtime_events, 95, 100)

    def plot_wer_analysis(self):
        """Create detailed WER analysis plot"""
        plt.figure(figsize=(14, 8))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('WER - Vietnamese ASR Performance Analysis', fontsize=16, fontweight='bold')

        wer_data = self.metrics_data['WER']
        timestamps = self.timestamps

        # 1. Time series with rolling average
        ax1.plot(timestamps, wer_data, 'b-', linewidth=2, alpha=0.7, label='WER')
        rolling_avg = pd.Series(wer_data).rolling(window=24).mean()
        ax1.plot(timestamps, rolling_avg, 'r-', linewidth=3, label='24h Average')
        ax1.axhline(y=15, color='green', linestyle='--', linewidth=2, label='Target (<15%)')
        ax1.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Poor (>20%)')

        # Highlight current value
        latest_wer = wer_data[-1]
        ax1.scatter(timestamps[-1], latest_wer, s=100, color='red', zorder=5)
        ax1.annotate(f'Current: {latest_wer:.1f}%',
                    xy=(timestamps[-1], latest_wer),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9,
                             edgecolor='orange', linewidth=2),
                    fontsize=12, fontweight='bold', color='darkblue',
                    arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

        ax1.set_title('WER Time Series', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Word Error Rate (%)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. WER Distribution
        ax2.hist(wer_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(x=np.mean(wer_data), color='red', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(wer_data):.1f}%')
        ax2.axvline(x=np.median(wer_data), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(wer_data):.1f}%')
        ax2.set_title('WER Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('WER (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()

        # 3. WER Performance Zones
        zones = ['Excellent (<10%)', 'Good (10-15%)', 'Fair (15-20%)', 'Poor (>20%)']
        colors = ['green', 'lightgreen', 'orange', 'red']
        sizes = [
            len([x for x in wer_data if x < 10]),
            len([x for x in wer_data if 10 <= x < 15]),
            len([x for x in wer_data if 15 <= x < 20]),
            len([x for x in wer_data if x >= 20])
        ]

        ax3.pie(sizes, labels=zones, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('WER Performance Zones', fontsize=14, fontweight='bold')

        # 4. WER Statistics Summary
        ax4.axis('off')
        stats_text = f"""
        WER STATISTICS SUMMARY

        Current WER:     {latest_wer:.1f}%
        24h Average:     {np.mean(wer_data[-24:]):.1f}%
        7-day Average:   {np.mean(wer_data[-168:]):.1f}%

        Performance:
        • Best:          {np.min(wer_data):.1f}%
        • Worst:         {np.max(wer_data):.1f}%
        • Std Dev:       {np.std(wer_data):.1f}%

        Target Achievement:
        • <15% Target:   {'✅' if latest_wer < 15 else '❌'} ({latest_wer < 15})
        • <20% Acceptable:{'✅' if latest_wer < 20 else '❌'} ({latest_wer < 20})
        """

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig('wer_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ WER Analysis Plot Saved!")

    def plot_classification_metrics(self):
        """Create classification metrics analysis"""
        plt.figure(figsize=(14, 8))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('Classification Metrics Analysis (Tầng 1)', fontsize=16, fontweight='bold')

        sens_data = self.metrics_data['Sensitivity_T1']
        spec_data = self.metrics_data['Specificity_T1']
        timestamps = self.timestamps

        # 1. Sensitivity Time Series
        ax1.plot(timestamps, sens_data, 'blue', linewidth=2, alpha=0.8, label='Sensitivity')
        ax1.axhline(y=90, color='green', linestyle='--', linewidth=2, label='Excellent (≥90%)')
        ax1.axhline(y=85, color='orange', linestyle='--', linewidth=2, label='Good (≥85%)')

        latest_sens = sens_data[-1]
        ax1.scatter(timestamps[-1], latest_sens, s=100, color='red', zorder=5)
        ax1.annotate(f'Current: {latest_sens:.1f}%',
                    xy=(timestamps[-1], latest_sens),
                    xytext=(10, 15), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9,
                             edgecolor='blue', linewidth=2),
                    fontsize=12, fontweight='bold', color='darkblue',
                    arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

        ax1.set_title('Sensitivity (Tầng 1)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Sensitivity (%)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Specificity Time Series
        ax2.plot(timestamps, spec_data, 'green', linewidth=2, alpha=0.8, label='Specificity')
        ax2.axhline(y=85, color='green', linestyle='--', linewidth=2, label='Excellent (≥85%)')
        ax2.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='Good (≥80%)')

        latest_spec = spec_data[-1]
        ax2.scatter(timestamps[-1], latest_spec, s=100, color='red', zorder=5)
        ax2.annotate(f'Current: {latest_spec:.1f}%',
                    xy=(timestamps[-1], latest_spec),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9,
                             edgecolor='darkgreen', linewidth=2),
                    fontsize=12, fontweight='bold', color='darkgreen',
                    arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

        ax2.set_title('Specificity (Tầng 1)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Specificity (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Sensitivity vs Specificity Scatter
        ax3.scatter(sens_data, spec_data, alpha=0.6, color='purple', s=30)
        ax3.plot([80, 100], [80, 100], 'r--', alpha=0.7, label='Equal Performance')
        ax3.set_xlabel('Sensitivity (%)', fontsize=12)
        ax3.set_ylabel('Specificity (%)', fontsize=12)
        ax3.set_title('Sensitivity vs Specificity', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Performance Summary
        ax4.axis('off')
        summary_text = f"""
        CLASSIFICATION PERFORMANCE SUMMARY

        Current Values:
        • Sensitivity:     {latest_sens:.1f}%
        • Specificity:     {latest_spec:.1f}%

        24h Averages:
        • Sensitivity:     {np.mean(sens_data[-24:]):.1f}%
        • Specificity:     {np.mean(spec_data[-24:]):.1f}%

        Target Achievement:
        • Sensitivity ≥90%: {'✅' if latest_sens >= 90 else '❌'}
        • Specificity ≥85%: {'✅' if latest_spec >= 85 else '❌'}

        Overall Assessment:
        {'🟢 EXCELLENT' if latest_sens >= 90 and latest_spec >= 85 else
         '🟡 GOOD' if latest_sens >= 85 and latest_spec >= 80 else
         '🔴 NEEDS IMPROVEMENT'}
        """

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))

        plt.tight_layout()
        plt.savefig('classification_metrics_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Classification Metrics Plot Saved!")

    def plot_auc_analysis(self):
        """Create AUC analysis plot"""
        plt.figure(figsize=(12, 8))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('AUC Analysis (Tầng 2)', fontsize=16, fontweight='bold')

        auc_data = self.metrics_data['AUC_T2']
        timestamps = self.timestamps

        # 1. AUC Time Series
        ax1.plot(timestamps, auc_data, 'purple', linewidth=2, alpha=0.8, label='AUC')
        ax1.axhline(y=0.9, color='green', linestyle='--', linewidth=2, label='Excellent (≥0.9)')
        ax1.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='Good (≥0.8)')
        ax1.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Poor (<0.7)')

        latest_auc = auc_data[-1]
        ax1.scatter(timestamps[-1], latest_auc, s=100, color='red', zorder=5)
        ax1.annotate(f'Current: {latest_auc:.3f}',
                    xy=(timestamps[-1], latest_auc),
                    xytext=(10, 15), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.9,
                             edgecolor='purple', linewidth=2),
                    fontsize=12, fontweight='bold', color='purple',
                    arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

        ax1.set_title('AUC Time Series', fontsize=14, fontweight='bold')
        ax1.set_ylabel('AUC Score', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. AUC Distribution
        ax2.hist(auc_data, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(x=np.mean(auc_data), color='red', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(auc_data):.3f}')
        ax2.axvline(x=np.median(auc_data), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(auc_data):.3f}')
        ax2.set_title('AUC Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('AUC Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()

        # 3. AUC Performance Gauge
        ax3.axis('off')

        # Create gauge-like visualization
        center_x, center_y = 0.5, 0.5
        radius = 0.4

        # Performance zones
        zones = [
            (0.5, 0.7, 'red', 'Poor'),
            (0.7, 0.8, 'orange', 'Fair'),
            (0.8, 0.9, 'lightgreen', 'Good'),
            (0.9, 1.0, 'green', 'Excellent')
        ]

        for start, end, color, label in zones:
            ax3.add_patch(plt.Circle((center_x, center_y), radius, fill=True,
                                   color=color, alpha=0.3))
            ax3.text(center_x, center_y + radius + 0.1, label, ha='center', fontsize=10)

        # AUC needle
        angle = np.pi - (latest_auc - 0.5) / 0.5 * np.pi
        needle_x = center_x + radius * 0.8 * np.cos(angle)
        needle_y = center_y + radius * 0.8 * np.sin(angle)

        ax3.plot([center_x, needle_x], [center_y, needle_y], 'k-', linewidth=4)
        ax3.plot([center_x, needle_x], [center_y, needle_y], 'r-', linewidth=2)

        # Current value
        ax3.text(center_x, center_y - 0.1, f'{latest_auc:.3f}', ha='center', va='center',
                fontsize=18, fontweight='bold', color='darkblue',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_title('AUC Performance Gauge', fontsize=14, fontweight='bold')

        # 4. AUC Statistics
        ax4.axis('off')
        auc_stats = f"""
        AUC STATISTICS SUMMARY

        Current AUC:      {latest_auc:.3f}
        24h Average:      {np.mean(auc_data[-24:]):.3f}
        7-day Average:    {np.mean(auc_data[-168:]):.3f}

        Performance Range:
        • Best:           {np.max(auc_data):.3f}
        • Worst:          {np.min(auc_data):.3f}
        • Std Dev:        {np.std(auc_data):.3f}

        Classification Quality:
        • Excellent (≥0.9): {'✅' if latest_auc >= 0.9 else '❌'}
        • Good (≥0.8):      {'✅' if latest_auc >= 0.8 else '❌'}
        • Poor (<0.7):      {'❌' if latest_auc < 0.7 else '✅'}
        """

        ax4.text(0.1, 0.9, auc_stats, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.8))

        plt.tight_layout()
        plt.savefig('auc_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ AUC Analysis Plot Saved!")

def create_individual_plots():
    """Create all individual plots"""
    print("🔬 Creating Individual Metrics Plots...")
    print("="*50)

    plotter = IndividualMetricsPlots()

    print("\n1. Creating WER Analysis...")
    plotter.plot_wer_analysis()

    print("\n2. Creating Classification Metrics Analysis...")
    plotter.plot_classification_metrics()

    print("\n3. Creating AUC Analysis...")
    plotter.plot_auc_analysis()

    print("\n" + "="*50)
    print("✅ All Individual Plots Created!")
    print("📊 Files generated:")
    print("  • wer_comprehensive_analysis.png")
    print("  • classification_metrics_analysis.png")
    print("  • auc_analysis.png")
    print("="*50)

if __name__ == "__main__":
    create_individual_plots()
