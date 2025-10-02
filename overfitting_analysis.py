#!/usr/bin/env python3
"""
Overfitting Analysis for Cognitive Assessment Models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class OverfittingAnalysis:
    """Analyze overfitting in the improved models"""

    def __init__(self):
        self.results = {}
        self.dataset_info = {}

    def load_and_clean_data(self, csv_path: str):
        """Load and clean the dataset"""
        try:
            df = pd.read_csv(csv_path)

            # Convert all columns to numeric, replace non-numeric with NaN
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove columns with >50% missing values
            missing_pct = df.isnull().sum() / len(df)
            df = df.drop(columns=missing_pct[missing_pct > 0.5].index)

            # Drop rows with missing MMSE
            target_col = 'mmse' if 'mmse' in df.columns else 'MMSE'
            df = df.dropna(subset=[target_col])

            # Separate features and target
            if 'mmse' in df.columns:
                X = df.drop('mmse', axis=1)
                y = df['mmse']
            elif 'MMSE' in df.columns:
                X = df.drop('MMSE', axis=1)
                y = df['MMSE']
            else:
                raise ValueError("MMSE/mmse column not found")

            print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"MMSE range: {y.min():.1f} - {y.max():.1f}")

            self.dataset_info = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'mmse_range': (y.min(), y.max()),
                'mmse_mean': y.mean(),
                'mmse_std': y.std()
            }

            return X, y

        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def create_simple_features(self, X):
        """Create simple feature engineering"""
        # Remove constant features
        X = X.loc[:, X.nunique() > 1]

        # Fill missing values with median
        X = X.fillna(X.median())

        return X

    def get_models(self):
        """Get models for overfitting analysis"""
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01)
        }
        return models

    def analyze_overfitting(self, X, y, model_name, model, cv_folds=5):
        """Analyze overfitting for a specific model"""
        print(f"\n🔍 Analyzing overfitting for {model_name}...")

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(score_func=mutual_info_regression, k='all')),
            ('model', model)
        ])

        # 1. Cross-validation scores
        cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds,
                                  scoring='r2', n_jobs=-1)
        cv_mae_scores = cross_val_score(pipeline, X, y, cv=cv_folds,
                                      scoring='neg_mean_absolute_error', n_jobs=-1)

        # 2. Full dataset training
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        train_r2 = r2_score(y, y_pred)
        train_mae = mean_absolute_error(y, y_pred)

        # 3. Learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            pipeline, X, y, cv=cv_folds,
            scoring='r2', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        # Calculate overfitting metrics
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()
        cv_mae_mean = -cv_mae_scores.mean()  # Convert back from negative

        overfitting_r2 = train_r2 - cv_r2_mean
        overfitting_mae = train_mae - cv_mae_mean

        # Variance analysis
        variance_ratio = cv_r2_std / cv_r2_mean if cv_r2_mean != 0 else float('inf')

        results = {
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std,
            'cv_mae_mean': cv_mae_mean,
            'train_r2': train_r2,
            'train_mae': train_mae,
            'overfitting_r2': overfitting_r2,
            'overfitting_mae': overfitting_mae,
            'variance_ratio': variance_ratio,
            'learning_curve': {
                'train_sizes': train_sizes,
                'train_scores_mean': train_scores.mean(axis=1),
                'val_scores_mean': val_scores.mean(axis=1)
            }
        }

        self.results[model_name] = results

        print(f"  CV R²: {cv_r2_mean:.3f} ± {cv_r2_std:.3f}")
        print(f"  Train R²: {train_r2:.3f}")
        print(f"  Overfitting gap: {overfitting_r2:.3f}")
        print(f"  Variance ratio: {variance_ratio:.3f}")

        return results

    def create_overfitting_report(self):
        """Create comprehensive overfitting report"""
        print("\n" + "="*80)
        print("🔍 OVERFITTING ANALYSIS REPORT")
        print("="*80)

        # Dataset summary
        print("\n📊 DATASET SUMMARY:")
        print("-" * 40)
        print(f"• Samples: {self.dataset_info['n_samples']}")
        print(f"• Features: {self.dataset_info['n_features']}")
        print(f"• MMSE Range: {self.dataset_info['mmse_range'][0]:.1f} - {self.dataset_info['mmse_range'][1]:.1f}")
        print(f"• Samples per feature: {self.dataset_info['n_samples']/self.dataset_info['n_features']:.1f}")

        # Overfitting assessment criteria
        print("\n🎯 OVERFITTING ASSESSMENT CRITERIA:")
        print("-" * 50)
        print("• Overfitting gap (Train - CV) > 0.2: POTENTIAL OVERFITTING")
        print("• Variance ratio (std/mean) > 0.3: HIGH VARIANCE")
        print("• Samples/feature < 10: RISK OF OVERFITTING")
        print("• Learning curves converge: GOOD GENERALIZATION")

        # Model analysis
        print("\n🏆 MODEL OVERFITTING ANALYSIS:")
        print("-" * 50)

        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            print(f"  • CV R²: {result['cv_r2_mean']:.3f} ± {result['cv_r2_std']:.3f}")
            print(f"  • Train R²: {result['train_r2']:.3f}")
            print(f"  • Overfitting gap: {result['overfitting_r2']:.3f}")
            print(f"  • Variance ratio: {result['variance_ratio']:.3f}")
            print(f"  • CV MAE: {result['cv_mae_mean']:.3f}")
            # Assessment
            assessment = self.assess_overfitting_risk(result)
            print(f"  • Assessment: {assessment}")

        # Overall conclusion
        print("\n" + "="*80)
        print("🎯 OVERALL CONCLUSION:")
        print("-" * 50)

        best_model = max(self.results.items(),
                        key=lambda x: x[1]['cv_r2_mean'])

        conclusion = self.generate_conclusion(best_model)
        print(conclusion)

    def assess_overfitting_risk(self, result):
        """Assess overfitting risk for a model"""
        gap = result['overfitting_r2']
        variance = result['variance_ratio']
        samples_per_feature = self.dataset_info['n_samples'] / self.dataset_info['n_features']

        risk_level = "LOW RISK"
        reasons = []

        if gap > 0.2:
            risk_level = "MODERATE RISK"
            reasons.append("large train-CV gap")
        elif gap > 0.1:
            risk_level = "LOW RISK"
            reasons.append("small train-CV gap")

        if variance > 0.3:
            risk_level = "HIGH RISK" if risk_level == "MODERATE RISK" else "MODERATE RISK"
            reasons.append("high variance")

        if samples_per_feature < 10:
            risk_level = "HIGH RISK"
            reasons.append("few samples per feature")

        reason_text = f" ({', '.join(reasons)})" if reasons else ""
        return f"{risk_level}{reason_text}"

    def generate_conclusion(self, best_model):
        """Generate overall conclusion"""
        model_name, result = best_model

        conclusion = f"""
🎯 BEST MODEL: {model_name}
   • Cross-validation R²: {result['cv_r2_mean']:.3f}
   • Overfitting gap: {result['overfitting_r2']:.3f}
   • Assessment: {self.assess_overfitting_risk(result)}

📋 CONCLUSION:
"""

        # Dataset size assessment
        samples_per_feature = self.dataset_info['n_samples'] / self.dataset_info['n_features']

        if samples_per_feature >= 20:
            conclusion += "✅ SUFFICIENT DATA: Good sample-to-feature ratio minimizes overfitting risk.\n"
        elif samples_per_feature >= 10:
            conclusion += "⚠️  MODERATE DATA: Adequate sample size with some overfitting risk.\n"
        else:
            conclusion += "❌ INSUFFICIENT DATA: High risk of overfitting due to limited samples.\n"

        # Model performance assessment
        if result['cv_r2_mean'] > 0.8 and result['overfitting_r2'] < 0.1:
            conclusion += "✅ EXCELLENT GENERALIZATION: High CV performance with minimal overfitting.\n"
        elif result['cv_r2_mean'] > 0.7 and result['overfitting_r2'] < 0.2:
            conclusion += "✅ GOOD GENERALIZATION: Solid performance with acceptable overfitting.\n"
        else:
            conclusion += "⚠️  POOR GENERALIZATION: Either low performance or significant overfitting.\n"

        # Clinical relevance
        if result['cv_mae_mean'] <= 4.0:
            conclusion += "🏥 CLINICAL GRADE: MAE ≤ 4.0 points on MMSE scale (acceptable for clinical use).\n"
        elif result['cv_mae_mean'] <= 5.0:
            conclusion += "⚕️  CLINICAL ACCEPTABLE: MAE ≤ 5.0 points (usable with caution).\n"
        else:
            conclusion += "❌ CLINICAL CONCERNS: MAE > 5.0 points (needs improvement for clinical use).\n"

        return conclusion

    def run_analysis(self, csv_path: str):
        """Run complete overfitting analysis"""
        print("="*80)
        print("🔬 OVERFITTING ANALYSIS FOR COGNITIVE ASSESSMENT MODELS")
        print("="*80)

        # Load data
        X, y = self.load_and_clean_data(csv_path)
        if X is None:
            return

        # Create features
        X = self.create_simple_features(X)

        # Get models
        models = self.get_models()

        # Analyze each model
        for model_name, model in models.items():
            self.analyze_overfitting(X, y, model_name, model)

        # Create report
        self.create_overfitting_report()

        print("\n" + "="*80)
        print("✅ OVERFITTING ANALYSIS COMPLETED!")
        print("📊 Results show model generalization capabilities")
        print("="*80)

def main():
    """Main function"""
    analyzer = OverfittingAnalysis()
    analyzer.run_analysis('backend/dx-mmse.csv')

if __name__ == "__main__":
    main()
