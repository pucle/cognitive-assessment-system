#!/usr/bin/env python3
"""
Simple Overfitting Check for Cognitive Assessment Models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleOverfittingCheck:
    """Simple overfitting analysis"""

    def __init__(self):
        self.results = {}

    def load_data(self):
        """Load and prepare data"""
        # Load the dx-mmse.csv data
        df = pd.read_csv('backend/dx-mmse.csv')

        # Convert all columns to numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Use 'mmse' as target
        target_col = 'mmse'
        df = df.dropna(subset=[target_col])

        # Remove columns with too many missing values
        missing_pct = df.isnull().sum() / len(df)
        df = df.drop(columns=missing_pct[missing_pct > 0.5].index)

        # Fill remaining missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Remove non-numeric columns except target
        feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
        X = df[feature_cols]
        y = df[target_col]

        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"MMSE range: {y.min():.1f} - {y.max():.1f}")

        return X, y

    def check_overfitting(self, X, y):
        """Check for overfitting in different models"""
        print("\n" + "="*80)
        print("🔍 OVERFITTING ANALYSIS")
        print("="*80)

        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01)
        }

        results = {}

        for name, model in models.items():
            print(f"\n📊 Analyzing {name}...")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Overfitting indicators
            overfitting_gap = train_r2 - test_r2
            cv_gap = train_r2 - cv_mean

            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_r2_mean': cv_mean,
                'cv_r2_std': cv_std,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'overfitting_gap': overfitting_gap,
                'cv_gap': cv_gap
            }

            print(f"  • Train MAE: {result['train_mae']:.3f}")
            print(f"  • Test MAE: {result['test_mae']:.3f}")
            print(f"  • CV R²: {result['cv_r2_mean']:.3f} ± {result['cv_r2_std']:.3f}")
            print(f"  • Train-Test gap: {result['overfitting_gap']:.3f}")
            print(f"  • Train-CV gap: {result['cv_gap']:.3f}")
            print(f"  • CV std: {result['cv_r2_std']:.3f}")

        self.results = results
        return results

    def create_report(self):
        """Create comprehensive overfitting report"""
        print("\n" + "="*80)
        print("📋 OVERFITTING ASSESSMENT REPORT")
        print("="*80)

        print("\n📊 DATASET INFO:")
        print("-" * 40)
        print(f"• Total samples: 236")
        print(f"• Features: 21 (after cleaning)")
        print(f"• Test split: 20% (47 samples)")
        print(f"• CV folds: 5-fold")

        print("\n🎯 OVERFITTING CRITERIA:")
        print("-" * 50)
        print("• Train-Test gap > 0.15: POTENTIAL OVERFITTING")
        print("• Train-CV gap > 0.20: SIGNIFICANT OVERFITTING")
        print("• CV std > 0.25: HIGH VARIANCE")
        print("• R² > 0.9 with small gaps: GOOD GENERALIZATION")

        print("\n🏆 MODEL ANALYSIS:")
        print("-" * 50)

        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            print(f"  • Train R²: {result['train_r2']:.3f}")
            print(f"  • Test R²: {result['test_r2']:.3f}")
            print(f"  • CV R²: {result['cv_r2_mean']:.3f} ± {result['cv_r2_std']:.3f}")
            print(f"  • Train-Test gap: {result['overfitting_gap']:.3f}")
            print(f"  • Train-CV gap: {result['cv_gap']:.3f}")
            print(f"  • Test MAE: {result['test_mae']:.3f}")

            # Risk assessment
            risk = self.assess_risk(result)
            print(f"  • Risk Level: {risk}")

        # Overall conclusion
        print("\n" + "="*80)
        print("🎯 CONCLUSION:")
        print("-" * 50)

        best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])
        model_name, result = best_model

        print(f"BEST MODEL: {model_name}")
        print(f"• Test R²: {result['test_r2']:.3f}")
        print(f"• Test MAE: {result['test_mae']:.3f}")
        print(f"• Overfitting gap: {result['overfitting_gap']:.3f}")

        # Final assessment
        if result['test_r2'] > 0.9 and result['overfitting_gap'] < 0.1:
            print("✅ EXCELLENT: High performance, minimal overfitting")
        elif result['test_r2'] > 0.8 and result['overfitting_gap'] < 0.15:
            print("✅ GOOD: Strong performance, acceptable overfitting")
        elif result['test_r2'] > 0.7 and result['overfitting_gap'] < 0.2:
            print("⚠️ MODERATE: Decent performance, some overfitting concerns")
        else:
            print("❌ POOR: Either low performance or significant overfitting")

        # Clinical relevance
        if result['test_mae'] <= 4.0:
            print("🏥 CLINICALLY ACCEPTABLE: MAE ≤ 4.0 on MMSE scale")
        elif result['test_mae'] <= 5.0:
            print("⚕️ CLINICALLY CAUTION: MAE ≤ 5.0 (use with caution)")
        else:
            print("❌ CLINICALLY CONCERNING: MAE > 5.0 (needs improvement)")

    def assess_risk(self, result):
        """Assess overfitting risk"""
        gap = result['overfitting_gap']
        cv_gap = result['cv_gap']
        cv_std = result['cv_r2_std']

        if gap < 0.1 and cv_gap < 0.15 and cv_std < 0.2:
            return "LOW RISK ✅"
        elif gap < 0.15 and cv_gap < 0.2 and cv_std < 0.25:
            return "MODERATE RISK ⚠️"
        else:
            return "HIGH RISK ❌"

def main():
    """Main function"""
    print("🔬 SIMPLE OVERFITTING CHECK")
    print("="*50)

    checker = SimpleOverfittingCheck()

    # Load data
    X, y = checker.load_data()

    # Check overfitting
    results = checker.check_overfitting(X, y)

    # Create report
    checker.create_report()

    print("\n" + "="*50)
    print("✅ OVERFITTING ANALYSIS COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    main()
