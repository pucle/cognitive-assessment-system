#!/usr/bin/env python3
"""
Remaining Metrics Plots for Cognitive Assessment System
MAE, Latency, and Uptime Analysis
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

class RemainingMetricsPlots:
    """Create plots for MAE, Latency, and Uptime"""

    def __init__(self):
        self.metrics_data = {}
        self.timestamps = []
        self.generate_mock_data()

    def generate_mock_data(self):
        """Generate realistic mock data"""
        np.random.seed(42)

        # Generate timestamps for last 30 days
        base_date = datetime.now() - timedelta(days=30)
        self.timestamps = [base_date + timedelta(hours=i) for i in range(24*30)]

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

    def plot_mae_analysis(self):
        """Create comprehensive MAE analysis"""
        plt.figure(figsize=(14, 8))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('MAE - Mean Absolute Error Analysis', fontsize=16, fontweight='bold')

        mae_data = self.metrics_data['MAE']
        timestamps = self.timestamps

        # 1. MAE Time Series with Clinical Thresholds
        ax1.plot(timestamps, mae_data, 'darkred', linewidth=2, alpha=0.8, label='MAE')
        ax1.axhline(y=4.0, color='red', linestyle='--', linewidth=2, label='Clinical Limit (4.0)')
        ax1.axhline(y=3.0, color='orange', linestyle='--', linewidth=2, label='Good Performance (3.0)')
        ax1.axhline(y=2.5, color='green', linestyle='--', linewidth=2, label='Excellent (2.5)')
        ax1.axhline(y=2.0, color='blue', linestyle='--', linewidth=2, label='Outstanding (2.0)')

        # Fill zones
        ax1.fill_between(timestamps, 0, mae_data, alpha=0.1, color='red')

        # Highlight current value
        latest_mae = mae_data[-1]
        ax1.scatter(timestamps[-1], latest_mae, s=120, color='red', zorder=5, edgecolor='black', linewidth=2)
        ax1.annotate(f'Current: {latest_mae:.2f}',
                    xy=(timestamps[-1], latest_mae),
                    xytext=(15, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.95,
                             edgecolor='red', linewidth=2),
                    fontsize=13, fontweight='bold', color='darkred',
                    arrowprops=dict(arrowstyle='->', color='red', linewidth=2.5))

        ax1.set_title('MAE Time Series with Clinical Thresholds', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Mean Absolute Error', fontsize=12)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. MAE Distribution with Performance Zones
        mae_recent = mae_data[-168:]  # Last 7 days
        ax2.hist(mae_recent, bins=25, alpha=0.8, color='lightcoral', edgecolor='black')

        # Performance zone lines
        ax2.axvline(x=2.0, color='blue', linestyle='--', linewidth=2, label='Outstanding (<2.0)')
        ax2.axvline(x=2.5, color='green', linestyle='--', linewidth=2, label='Excellent (<2.5)')
        ax2.axvline(x=3.0, color='orange', linestyle='--', linewidth=2, label='Good (<3.0)')
        ax2.axvline(x=4.0, color='red', linestyle='--', linewidth=2, label='Clinical Limit (<4.0)')

        # Current MAE line
        ax2.axvline(x=latest_mae, color='purple', linestyle='-', linewidth=3,
                   label=f'Current: {latest_mae:.2f}')

        ax2.set_title('MAE Distribution (Last 7 Days)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('MAE Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend(fontsize=9)

        # 3. MAE Performance Assessment
        ax3.axis('off')

        # Calculate performance metrics
        excellent_pct = (mae_recent < 2.5).sum() / len(mae_recent) * 100
        good_pct = ((mae_recent >= 2.5) & (mae_recent < 3.0)).sum() / len(mae_recent) * 100
        acceptable_pct = ((mae_recent >= 3.0) & (mae_recent < 4.0)).sum() / len(mae_recent) * 100
        poor_pct = (mae_recent >= 4.0).sum() / len(mae_recent) * 100

        # Create performance pie chart
        sizes = [excellent_pct, good_pct, acceptable_pct, poor_pct]
        labels = [f'Excellent\n{excellent_pct:.1f}%', f'Good\n{good_pct:.1f}%',
                 f'Acceptable\n{acceptable_pct:.1f}%', f'Poor\n{poor_pct:.1f}%']
        colors = ['green', 'lightgreen', 'orange', 'red']

        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, pctdistance=0.85)

        # Add center circle for donut effect
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax3.add_artist(centre_circle)

        # Add center text
        ax3.text(0, 0, f'MAE\n{np.mean(mae_recent):.2f}', ha='center', va='center',
                fontsize=14, fontweight='bold')

        ax3.set_title('MAE Performance Distribution', fontsize=14, fontweight='bold')

        # 4. MAE Statistics Summary
        ax4.axis('off')
        mae_stats = f"""
        MAE STATISTICS SUMMARY

        Current MAE:      {latest_mae:.2f}
        24h Average:      {np.mean(mae_data[-24:]):.2f}
        7-day Average:    {np.mean(mae_data[-168:]):.2f}

        Performance Range:
        • Best:           {np.min(mae_data):.2f}
        • Worst:          {np.max(mae_data):.2f}
        • Std Dev:        {np.std(mae_data):.2f}

        Clinical Assessment:
        • Excellent (<2.5): {'✅' if latest_mae < 2.5 else '❌'}
        • Good (<3.0):      {'✅' if latest_mae < 3.0 else '❌'}
        • Acceptable (<4.0):{'✅' if latest_mae < 4.0 else '❌'}
        • Poor (≥4.0):      {'❌' if latest_mae >= 4.0 else '✅'}

        Overall Rating:
        {'🟢 EXCELLENT' if latest_mae < 2.5 else
         '🟡 GOOD' if latest_mae < 3.0 else
         '🟠 ACCEPTABLE' if latest_mae < 4.0 else
         '🔴 POOR - NEEDS IMPROVEMENT'}
        """

        ax4.text(0.1, 0.9, mae_stats, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig('mae_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ MAE Analysis Plot Saved!")

    def plot_latency_analysis(self):
        """Create comprehensive latency analysis"""
        plt.figure(figsize=(14, 8))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('Processing Latency Analysis', fontsize=16, fontweight='bold')

        latency_data = self.metrics_data['Latency']
        timestamps = self.timestamps

        # 1. Latency Time Series
        ax1.plot(timestamps, latency_data, 'orange', linewidth=2, alpha=0.8, label='Latency')
        ax1.axhline(y=500, color='green', linestyle='--', linewidth=2, label='Real-time (<500ms)')
        ax1.axhline(y=1000, color='orange', linestyle='--', linewidth=2, label='Acceptable (<1000ms)')
        ax1.axhline(y=2000, color='red', linestyle='--', linewidth=2, label='Poor (≥2000ms)')

        # Highlight current value
        latest_latency = latency_data[-1]
        ax1.scatter(timestamps[-1], latest_latency, s=120, color='red', zorder=5,
                   edgecolor='black', linewidth=2)
        ax1.annotate(f'Current: {latest_latency:.0f}ms',
                    xy=(timestamps[-1], latest_latency),
                    xytext=(15, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.95,
                             edgecolor='orange', linewidth=2),
                    fontsize=13, fontweight='bold', color='darkorange',
                    arrowprops=dict(arrowstyle='->', color='red', linewidth=2.5))

        ax1.set_title('Processing Latency Time Series', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Latency Distribution
        latency_recent = latency_data[-168:]  # Last 7 days
        ax2.hist(latency_recent, bins=30, alpha=0.8, color='lightblue', edgecolor='black')

        # Performance zone lines
        ax2.axvline(x=500, color='green', linestyle='--', linewidth=2, label='Real-time (<500ms)')
        ax2.axvline(x=1000, color='orange', linestyle='--', linewidth=2, label='Acceptable (<1000ms)')
        ax2.axvline(x=2000, color='red', linestyle='--', linewidth=2, label='Poor (≥2000ms)')

        # Current latency line
        ax2.axvline(x=latest_latency, color='purple', linestyle='-', linewidth=3,
                   label=f'Current: {latest_latency:.0f}ms')

        ax2.set_title('Latency Distribution (Last 7 Days)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Latency (ms)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend(fontsize=9)

        # 3. Latency Performance Zones
        ax3.axis('off')

        # Calculate performance zones
        realtime_pct = (latency_recent < 500).sum() / len(latency_recent) * 100
        acceptable_pct = ((latency_recent >= 500) & (latency_recent < 1000)).sum() / len(latency_recent) * 100
        slow_pct = ((latency_recent >= 1000) & (latency_recent < 2000)).sum() / len(latency_recent) * 100
        poor_pct = (latency_recent >= 2000).sum() / len(latency_recent) * 100

        # Performance pie chart
        sizes = [realtime_pct, acceptable_pct, slow_pct, poor_pct]
        labels = [f'Real-time\n{realtime_pct:.1f}%', f'Acceptable\n{acceptable_pct:.1f}%',
                 f'Slow\n{slow_pct:.1f}%', f'Poor\n{poor_pct:.1f}%']
        colors = ['green', 'lightgreen', 'orange', 'red']

        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, pctdistance=0.85)

        # Donut effect
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax3.add_artist(centre_circle)

        # Center text
        ax3.text(0, 0, f'Latency\n{np.mean(latency_recent):.0f}ms', ha='center', va='center',
                fontsize=14, fontweight='bold')

        ax3.set_title('Latency Performance Distribution', fontsize=14, fontweight='bold')

        # 4. Latency Statistics
        ax4.axis('off')
        latency_stats = f"""
        LATENCY STATISTICS SUMMARY

        Current Latency:  {latest_latency:.0f}ms
        24h Average:      {np.mean(latency_data[-24:]):.0f}ms
        7-day Average:    {np.mean(latency_data[-168:]):.0f}ms

        Response Time:
        • P25:            {np.percentile(latency_data, 25):.0f}ms
        • P50 (Median):   {np.percentile(latency_data, 50):.0f}ms
        • P75:            {np.percentile(latency_data, 75):.0f}ms
        • P95:            {np.percentile(latency_data, 95):.0f}ms

        Performance Assessment:
        • Real-time (<500ms):  {'✅' if latest_latency < 500 else '❌'}
        • Acceptable (<1000ms): {'✅' if latest_latency < 1000 else '❌'}
        • Poor (≥2000ms):       {'❌' if latest_latency >= 2000 else '✅'}

        Overall Rating:
        {'🟢 EXCELLENT' if latest_latency < 500 else
         '🟡 GOOD' if latest_latency < 1000 else
         '🟠 ACCEPTABLE' if latest_latency < 2000 else
         '🔴 POOR - NEEDS OPTIMIZATION'}
        """

        ax4.text(0.1, 0.9, latency_stats, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))

        plt.tight_layout()
        plt.savefig('latency_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Latency Analysis Plot Saved!")

    def plot_uptime_analysis(self):
        """Create comprehensive uptime analysis"""
        plt.figure(figsize=(14, 8))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('System Uptime Analysis', fontsize=16, fontweight='bold')

        uptime_data = self.metrics_data['Uptime']
        timestamps = self.timestamps

        # 1. Uptime Time Series
        ax1.plot(timestamps, uptime_data, 'darkgreen', linewidth=2, alpha=0.8, label='Uptime')
        ax1.axhline(y=99.9, color='red', linestyle='--', linewidth=2, label='99.9% SLA')
        ax1.axhline(y=99.5, color='orange', linestyle='--', linewidth=2, label='99.5% SLA')
        ax1.axhline(y=99.0, color='yellow', linestyle='--', linewidth=2, label='99.0% Target')
        ax1.axhline(y=95.0, color='red', linestyle='-', linewidth=1, alpha=0.5, label='Minimum (95%)')

        # Fill uptime zones
        ax1.fill_between(timestamps, 99.9, 100, alpha=0.2, color='green', label='Excellent Zone')
        ax1.fill_between(timestamps, 99.5, 99.9, alpha=0.2, color='lightgreen', label='Good Zone')

        # Highlight current value
        latest_uptime = uptime_data[-1]
        ax1.scatter(timestamps[-1], latest_uptime, s=120, color='red', zorder=5,
                   edgecolor='black', linewidth=2)
        ax1.annotate(f'Current: {latest_uptime:.2f}%',
                    xy=(timestamps[-1], latest_uptime),
                    xytext=(15, -25), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', alpha=0.95,
                             edgecolor='darkgreen', linewidth=2),
                    fontsize=13, fontweight='bold', color='darkgreen',
                    arrowprops=dict(arrowstyle='->', color='red', linewidth=2.5))

        ax1.set_title('System Uptime Time Series', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Uptime (%)', fontsize=12)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylim(94, 100.5)
        ax1.legend(fontsize=9, loc='lower right')
        ax1.grid(True, alpha=0.3)

        # 2. Uptime Distribution
        uptime_recent = uptime_data[-168:]  # Last 7 days
        ax2.hist(uptime_recent, bins=20, alpha=0.8, color='lightgreen', edgecolor='black')

        # SLA lines
        ax2.axvline(x=99.9, color='red', linestyle='--', linewidth=2, label='99.9% SLA')
        ax2.axvline(x=99.5, color='orange', linestyle='--', linewidth=2, label='99.5% SLA')
        ax2.axvline(x=99.0, color='yellow', linestyle='--', linewidth=2, label='99.0% Target')

        # Current uptime line
        ax2.axvline(x=latest_uptime, color='purple', linestyle='-', linewidth=3,
                   label=f'Current: {latest_uptime:.2f}%')

        ax2.set_title('Uptime Distribution (Last 7 Days)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Uptime (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend(fontsize=9)

        # 3. Uptime SLA Compliance
        ax3.axis('off')

        # Calculate SLA compliance
        excellent_pct = (uptime_recent >= 99.9).sum() / len(uptime_recent) * 100
        good_pct = ((uptime_recent >= 99.5) & (uptime_recent < 99.9)).sum() / len(uptime_recent) * 100
        acceptable_pct = ((uptime_recent >= 99.0) & (uptime_recent < 99.5)).sum() / len(uptime_recent) * 100
        poor_pct = (uptime_recent < 99.0).sum() / len(uptime_recent) * 100

        # SLA compliance pie
        sizes = [excellent_pct, good_pct, acceptable_pct, poor_pct]
        labels = [f'99.9%+\n{excellent_pct:.1f}%', f'99.5-99.9%\n{good_pct:.1f}%',
                 f'99.0-99.5%\n{acceptable_pct:.1f}%', f'<99.0%\n{poor_pct:.1f}%']
        colors = ['green', 'lightgreen', 'orange', 'red']

        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, pctdistance=0.85)

        # Donut effect
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax3.add_artist(centre_circle)

        # Center text
        ax3.text(0, 0, f'Uptime\n{np.mean(uptime_recent):.2f}%', ha='center', va='center',
                fontsize=14, fontweight='bold')

        ax3.set_title('SLA Compliance Distribution', fontsize=14, fontweight='bold')

        # 4. Uptime Statistics
        ax4.axis('off')
        uptime_stats = f"""
        UPTIME STATISTICS SUMMARY

        Current Uptime:   {latest_uptime:.2f}%
        24h Average:      {np.mean(uptime_data[-24:]):.2f}%
        7-day Average:    {np.mean(uptime_data[-168:]):.2f}
        30-day Average:   {np.mean(uptime_data):.2f}%

        Monthly Uptime:
        • Expected:       99.5%
        • Actual:         {np.mean(uptime_data):.2f}%
        • Difference:     {np.mean(uptime_data) - 99.5:.2f}%

        SLA Compliance:
        • 99.9% SLA:      {'✅' if latest_uptime >= 99.9 else '❌'}
        • 99.5% SLA:      {'✅' if latest_uptime >= 99.5 else '❌'}
        • 99.0% Target:   {'✅' if latest_uptime >= 99.0 else '❌'}

        Downtime Analysis:
        • Total Hours:    {720 * (100 - np.mean(uptime_data)) / 100:.1f}h/month
        • SLA Penalty:    {'None' if latest_uptime >= 99.9 else 'Minor' if latest_uptime >= 99.5 else 'Major'}

        Overall Rating:
        {'🟢 EXCELLENT' if latest_uptime >= 99.9 else
         '🟡 GOOD' if latest_uptime >= 99.5 else
         '🟠 ACCEPTABLE' if latest_uptime >= 99.0 else
         '🔴 POOR - NEEDS IMPROVEMENT'}
        """

        ax4.text(0.1, 0.9, uptime_stats, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.savefig('uptime_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Uptime Analysis Plot Saved!")

def create_remaining_plots():
    """Create all remaining metrics plots"""
    print("🔬 Creating Remaining Metrics Plots...")
    print("="*50)

    plotter = RemainingMetricsPlots()

    print("\n1. Creating MAE Analysis...")
    plotter.plot_mae_analysis()

    print("\n2. Creating Latency Analysis...")
    plotter.plot_latency_analysis()

    print("\n3. Creating Uptime Analysis...")
    plotter.plot_uptime_analysis()

    print("\n" + "="*50)
    print("✅ All Remaining Plots Created!")
    print("📊 Files generated:")
    print("  • mae_comprehensive_analysis.png")
    print("  • latency_comprehensive_analysis.png")
    print("  • uptime_comprehensive_analysis.png")
    print("="*50)

if __name__ == "__main__":
    create_remaining_plots()
