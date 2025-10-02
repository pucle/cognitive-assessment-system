#!/usr/bin/env python3
"""
Simple Regression Improvement - Focus on Data Quality First
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SimpleRegressionImprovement:
    """Simple regression with robust data preprocessing"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = float('inf')

    def load_and_clean_data(self, csv_path: str) -> tuple:
        """Load and thoroughly clean the data"""
        print("🔄 Loading and cleaning data...")

        # Load data
        df = pd.read_csv(csv_path)
        print(f"Original data shape: {df.shape}")

        # Replace problematic values
        df = df.replace(['NA', '-Inf', 'Inf', ''], np.nan)

        # Convert to numeric where possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                df.drop(col, axis=1, inplace=True)

        # Keep only numeric columns with reasonable missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_pct = df[numeric_cols].isnull().mean()

        # Keep columns with < 50% missing values
        good_cols = missing_pct[missing_pct < 0.5].index.tolist()

        # Always keep mmse if it exists
        if 'mmse' in df.columns and 'mmse' not in good_cols:
            good_cols.append('mmse')

        df = df[good_cols]
        print(f"After cleaning numeric columns: {df.shape}")

        # Handle target variable
        if 'mmse' not in df.columns:
            raise ValueError("MMSE column not found")

        # Remove rows with missing MMSE
        df = df.dropna(subset=['mmse'])
        print(f"After removing missing MMSE: {df.shape}")

        # Prepare features and target
        feature_cols = [col for col in df.columns if col != 'mmse']
        X = df[feature_cols]
        y = df['mmse'].values

        print(f"Target variable range: {y.min():.1f} - {y.max():.1f}")
        print(f"Feature matrix shape: {X.shape}")

        return X, y

    def create_simple_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create simple but useful features"""
        print("🔧 Creating simple features...")

        X_clean = X.copy()

        # Fill missing values with median
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())

        # Keep only columns that are not constant
        varying_cols = []
        for col in X_clean.columns:
            if X_clean[col].std() > 0.01:  # Not constant
                varying_cols.append(col)

        X_clean = X_clean[varying_cols]
        print(f"Features after removing constants: {X_clean.shape}")

        # Add simple statistical features if we have enough columns
        if len(X_clean.columns) > 3:
            X_clean['feature_mean'] = X_clean.mean(axis=1)
            X_clean['feature_std'] = X_clean.std(axis=1)

        return X_clean

    def get_simple_models(self) -> dict:
        """Get simple but effective models"""
        return {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=0.01, random_state=42),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

    def create_model_pipeline(self, model):
        """Create simple pipeline"""
        return Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(score_func=mutual_info_regression, k='all')),
            ('regressor', model)
        ])

    def evaluate_model(self, name: str, model, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Evaluate model with cross-validation"""
        print(f"🧪 Evaluating {name}...")

        try:
            pipeline = self.create_model_pipeline(model)

            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X, y,
                cv=5, scoring='neg_mean_absolute_error',
                n_jobs=-1
            )

            mae_scores = -cv_scores

            # Train on full data for final metrics
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)

            results = {
                'mae_mean': float(np.mean(mae_scores)),
                'mae_std': float(np.std(mae_scores)),
                'r2_full': float(r2_score(y, y_pred)),
                'rmse_full': float(np.sqrt(mean_squared_error(y, y_pred))),
                'mae_full': float(mean_absolute_error(y, y_pred))
            }

            print(f"  {name}: MAE={results['mae_mean']:.3f}±{results['mae_std']:.3f}, R²={results['r2_full']:.3f}")
            # Track best model
            if results['mae_mean'] < self.best_score:
                self.best_score = results['mae_mean']
                self.best_model = pipeline

            return results

        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            return {'error': str(e)}

    def run_improvement_analysis(self, csv_path: str):
        """Run the complete improvement analysis"""
        print("="*60)
        print("SIMPLE REGRESSION IMPROVEMENT ANALYSIS")
        print("="*60)

        # Load and clean data
        X_raw, y = self.load_and_clean_data(csv_path)

        # Create features
        X = self.create_simple_features(X_raw)

        # Get models
        models = self.get_simple_models()

        # Evaluate all models
        print("\n🎯 Evaluating models...")
        results = {}

        for name, model in models.items():
            results[name] = self.evaluate_model(name, model, X, y)

        # Filter successful results
        successful_results = {k: v for k, v in results.items()
                            if isinstance(v, dict) and 'mae_mean' in v}

        # Create summary
        if successful_results:
            print("\n" + "="*60)
            print("📊 MODEL PERFORMANCE SUMMARY")
            print("="*60)

            # Sort by MAE
            sorted_results = sorted(successful_results.items(),
                                  key=lambda x: x[1]['mae_mean'])

            print("\n🏆 Model Performance (Sorted by MAE):")
            print("-" * 50)

            for name, metrics in sorted_results:
                print(f"{name:15}: MAE={metrics['mae_mean']:.2f}, R²={metrics['r2_full']:.3f}")
            # Best model
            best_name = sorted_results[0][0]
            best_metrics = sorted_results[0][1]

            print(f"\n🎯 BEST MODEL: {best_name}")
            print(f"R² Score: {best_metrics['r2_full']:.3f}")
            print(f"MAE: {best_metrics['mae_mean']:.2f}")
            print(f"RMSE: {best_metrics['rmse_full']:.2f}")

            # Compare with baseline (mean prediction)
            baseline_mae = np.mean(np.abs(y - np.mean(y)))
            print(f"Baseline MAE (mean prediction): {baseline_mae:.2f}")
            improvement = (baseline_mae - best_metrics['mae_mean']) / baseline_mae * 100
            print(f"Improvement over baseline: {improvement:.1f}%")
        else:
            print("❌ No successful models")

        # Create simple visualization
        self.create_simple_visualization(successful_results)

        print("\n✅ Analysis completed!")
        return results

    def create_simple_visualization(self, results: dict):
        """Create simple bar chart comparison"""
        if not results:
            return

        model_names = list(results.keys())
        mae_scores = [results[m]['mae_mean'] for m in model_names]
        r2_scores = [results[m]['r2_full'] for m in model_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # MAE comparison
        bars1 = ax1.bar(model_names, mae_scores, color='skyblue', alpha=0.8)
        ax1.set_title('Mean Absolute Error by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('MAE', fontsize=12)
        ax1.set_xlabel('Models', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, score in zip(bars1, mae_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    '.2f', ha='center', va='bottom', fontsize=10)

        # R² comparison
        bars2 = ax2.bar(model_names, r2_scores, color='coral', alpha=0.8)
        ax2.set_title('R² Score by Model', fontsize=14, fontweight='bold')
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.set_xlabel('Models', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, score in zip(bars2, r2_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.3f', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('simple_regression_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📈 Simple comparison saved as 'simple_regression_comparison.png'")

def main():
    """Main function"""
    print("🔬 Starting Simple Regression Improvement...")

    # Run analysis
    improver = SimpleRegressionImprovement()
    results = improver.run_improvement_analysis('backend/dx-mmse.csv')

    print("\n🎉 Simple regression improvement completed!")
    print("📁 Check 'simple_regression_comparison.png' for results")

if __name__ == "__main__":
    main()
