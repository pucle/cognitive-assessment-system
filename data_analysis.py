#!/usr/bin/env python3
"""
Data Analysis for Regression Model Improvement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_data():
    """Analyze the dx-mmse.csv data to understand regression issues"""

    # Load data
    df = pd.read_csv('backend/dx-mmse.csv')

    print("="*80)
    print("DATA ANALYSIS FOR REGRESSION MODEL IMPROVEMENT")
    print("="*80)

    print(f"\n📊 Basic Info:")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"MMSE range: {df['mmse'].min()} - {df['mmse'].max()}")
    print(f"MMSE mean: {df['mmse'].mean():.2f} ± {df['mmse'].std():.2f}")

    # Check for problematic values
    print(f"\n🔍 Problematic Values:")
    na_count = (df == 'NA').sum().sum()
    inf_count = (df == '-Inf').sum().sum() + (df == 'Inf').sum().sum()
    print(f"NA values: {na_count}")
    print(f"Inf/-Inf values: {inf_count}")

    # Key columns analysis
    key_cols = ['mmse', 'age', 'dur.mean', 'dur.sd', 'srate.mean', 'number.utt']
    print(f"\n📈 Key Columns Analysis:")
    for col in key_cols:
        if col in df.columns:
            try:
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                missing = numeric_vals.isnull().sum()
                print(f"{col:15}: {numeric_vals.mean():.2f} ± {numeric_vals.std():.2f} (missing: {missing})")
            except:
                print(f"{col:15}: Non-numeric")

    # MMSE distribution
    print(f"\n🎯 MMSE Distribution by Diagnosis:")
    print(df.groupby('dx')['mmse'].describe())

    # Feature correlation with MMSE
    numeric_df = df.select_dtypes(include=[np.number])
    if 'mmse' in numeric_df.columns:
        correlations = numeric_df.corr()['mmse'].abs().sort_values(ascending=False)
        print(f"\n🔗 Top 10 Features Correlated with MMSE:")
        print(correlations.head(10))

    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Data Analysis for Regression Model Improvement', fontsize=16, fontweight='bold')

    # MMSE distribution
    ax1.hist(df['mmse'], bins=20, alpha=0.7, edgecolor='black')
    ax1.set_title('MMSE Score Distribution', fontweight='bold')
    ax1.set_xlabel('MMSE Score')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # MMSE by diagnosis
    if 'dx' in df.columns:
        mmse_by_dx = [df[df['dx']==dx]['mmse'].values for dx in df['dx'].unique()]
        ax2.boxplot(mmse_by_dx, labels=df['dx'].unique())
        ax2.set_title('MMSE by Diagnosis', fontweight='bold')
        ax2.set_ylabel('MMSE Score')
        ax2.grid(True, alpha=0.3)

    # Correlation heatmap (top features)
    if len(numeric_df.columns) > 5:
        top_features = correlations.head(8).index
        corr_matrix = numeric_df[top_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax3, fmt='.2f')
        ax3.set_title('Feature Correlation Heatmap', fontweight='bold')

    # Age vs MMSE scatter
    if 'age' in df.columns:
        ax4.scatter(df['age'], df['mmse'], alpha=0.6, s=50)
        ax4.set_title('Age vs MMSE', fontweight='bold')
        ax4.set_xlabel('Age')
        ax4.set_ylabel('MMSE Score')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n📈 Visualizations saved as 'data_analysis.png'")
    print("\n🔧 Key Insights for Regression Improvement:")
    print("1. Dataset size:", df.shape[0], "samples - QUITE SMALL!")
    print("2. Many NA/Inf values that need cleaning")
    print("3. MMSE range:", df['mmse'].min(), "-", df['mmse'].max())
    print("4. Need better feature engineering")
    print("5. Consider data augmentation or more samples")

if __name__ == "__main__":
    analyze_data()
