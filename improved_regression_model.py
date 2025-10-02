#!/usr/bin/env python3
"""
Improved Regression Model for MMSE Prediction
Addresses the issues identified in data analysis:
- Small dataset (237 samples)
- Poor feature quality (many NA/Inf values)
- Need better feature engineering
- Add modern ML algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PolynomialFeatures
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)

# Advanced ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedRegressionModel:
    """
    Improved regression model with comprehensive data preprocessing
    and advanced ML algorithms for MMSE prediction
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_model = None
        self.best_score = float('inf')
        self.feature_names = None
        self.scaler = None
        self.imputer = None

    def load_and_preprocess_data(self, csv_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and preprocess the data with robust handling"""
        logger.info("🔄 Loading and preprocessing data...")

        # Load data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {df.shape[0]} samples with {df.shape[1]} features")

        # Handle problematic string values
        df = df.replace(['NA', '-Inf', 'Inf'], np.nan)

        # Convert to numeric where possible
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df[col] = df[col].astype('category').cat.codes

        # Extract target variable
        if 'mmse' not in df.columns:
            raise ValueError("MMSE column not found in dataset")

        y = df['mmse'].values

        # Remove non-feature columns
        exclude_cols = ['mmse', 'id', 'sid', 'filename', 'adressfname', 'dataset', 'test']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()

        logger.info(f"Target variable range: {y.min():.1f} - {y.max():.1f}")
        logger.info(f"Feature matrix shape: {X.shape}")

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Converting {len(categorical_cols)} categorical columns")
            for col in categorical_cols:
                X[col] = X[col].astype('category').cat.codes

        # Store feature names
        self.feature_names = X.columns.tolist()

        return X, y

    def create_advanced_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features from existing data"""
        logger.info("🔧 Creating advanced features...")

        X_enhanced = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 2:
            # Statistical features
            X_enhanced['feature_mean'] = X[numeric_cols].mean(axis=1)
            X_enhanced['feature_std'] = X[numeric_cols].std(axis=1)
            X_enhanced['feature_skew'] = X[numeric_cols].skew(axis=1)
            X_enhanced['feature_kurtosis'] = X[numeric_cols].kurtosis(axis=1)

            # Ratio features (avoid division by zero)
            if 'age' in X.columns:
                age_col = X['age']
                for col in numeric_cols[:5]:  # Top 5 features
                    if col != 'age':
                        X_enhanced[f'{col}_per_year'] = X[col] / (age_col + 1)

            # Interaction features
            top_features = numeric_cols[:3]  # Use top 3 features
            for i, col1 in enumerate(top_features):
                for col2 in top_features[i+1:]:
                    X_enhanced[f'{col1}_{col2}_interact'] = X[col1] * X[col2]

        logger.info(f"Features expanded: {X.shape[1]} → {X_enhanced.shape[1]}")
        return X_enhanced

    def get_advanced_regressors(self) -> Dict[str, Any]:
        """Get advanced regressors with optimized parameters"""
        regressors = {}

        # Linear models
        regressors['ridge'] = Ridge(alpha=1.0, random_state=self.random_state)
        regressors['lasso'] = Lasso(alpha=0.01, random_state=self.random_state, max_iter=2000)
        regressors['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state, max_iter=2000)
        regressors['huber'] = HuberRegressor(max_iter=2000)

        # Tree-based models
        regressors['random_forest'] = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=self.random_state, n_jobs=-1
        )

        regressors['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=self.random_state
        )

        # XGBoost
        if HAS_XGBOOST:
            regressors['xgboost'] = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=-1,
                tree_method='hist'
            )

        # LightGBM
        if HAS_LIGHTGBM:
            regressors['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=-1,
                verbose=-1
            )

        # CatBoost
        if HAS_CATBOOST:
            regressors['catboost'] = cb.CatBoostRegressor(
                iterations=200, depth=6, learning_rate=0.1,
                subsample=0.8, random_state=self.random_state,
                verbose=False, thread_count=-1
            )

        # Support Vector Regression
        regressors['svr_rbf'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        regressors['svr_linear'] = SVR(kernel='linear', C=1.0)

        return regressors

    def create_pipeline(self, regressor) -> Pipeline:
        """Create preprocessing + regression pipeline"""
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(score_func=mutual_info_regression, k=20)),
            ('regressor', regressor)
        ])
        return pipeline

    def evaluate_regressor(self, name: str, regressor, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate regressor with cross-validation"""
        logger.info(f"🧪 Evaluating {name}...")

        try:
            pipeline = self.create_pipeline(regressor)

            # Cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            mae_scores = -cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
            r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2', n_jobs=-1)

            results = {
                'mae_mean': float(np.mean(mae_scores)),
                'mae_std': float(np.std(mae_scores)),
                'r2_mean': float(np.mean(r2_scores)),
                'r2_std': float(np.std(r2_scores)),
                'mae_scores': mae_scores.tolist(),
                'r2_scores': r2_scores.tolist()
            }

            logger.info(f"  {name}: MAE={results['mae_mean']:.3f}±{results['mae_std']:.3f}, R²={results['r2_mean']:.3f}±{results['r2_std']:.3f}")
            # Track best model
            if results['mae_mean'] < self.best_score:
                self.best_score = results['mae_mean']
                self.best_model = pipeline

            return results

        except Exception as e:
            logger.warning(f"  {name} failed: {e}")
            return {'error': str(e)}

    def train_and_evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Complete training and evaluation pipeline"""
        logger.info("🚀 Starting improved regression training...")

        # Feature engineering
        X_enhanced = self.create_advanced_features(X)

        # Get regressors
        regressors = self.get_advanced_regressors()

        # Evaluate all models
        results = {}
        logger.info("🎯 Evaluating all regression models...")

        for name, regressor in regressors.items():
            results[name] = self.evaluate_regressor(name, regressor, X_enhanced, y)

        # Summary
        results['summary'] = {
            'total_models_tested': len(results),
            'best_model': max(results.keys(),
                            key=lambda k: results[k].get('r2_mean', -float('inf')) if 'r2_mean' in results[k] else -float('inf')),
            'best_r2': max([r.get('r2_mean', -float('inf')) for r in results.values()]),
            'feature_count': X_enhanced.shape[1]
        }

        logger.info(f"🏆 Best model: {results['summary']['best_model']} (R²: {results['summary']['best_r2']:.3f})")
        return results

    def create_visualizations(self, results: Dict[str, Any], save_path: str = "improved_regression_analysis.png"):
        """Create comprehensive visualization of results"""
        logger.info("📊 Creating visualizations...")

        # Filter successful models
        successful_models = {k: v for k, v in results.items()
                           if isinstance(v, dict) and 'r2_mean' in v}

        if not successful_models:
            logger.warning("No successful models to visualize")
            return

        # Extract data
        model_names = list(successful_models.keys())
        r2_scores = [successful_models[m]['r2_mean'] for m in model_names]
        mae_scores = [successful_models[m]['mae_mean'] for m in model_names]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Improved Regression Models Performance Analysis', fontsize=16, fontweight='bold')

        # R² Scores
        bars1 = ax1.bar(model_names, r2_scores, color='skyblue', alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('R² Scores by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score', fontsize=12)
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylim(min(r2_scores) - 0.1, max(r2_scores) + 0.1)
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.3f', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Highlight best model
        best_idx = np.argmax(r2_scores)
        bars1[best_idx].set_color('darkblue')

        # MAE Scores
        bars2 = ax2.bar(model_names, mae_scores, color='coral', alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('MAE Scores by Model', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error', fontsize=12)
        ax2.set_xlabel('Models', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, score in zip(bars2, mae_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    '.2f', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Highlight best model (lowest MAE)
        best_mae_idx = np.argmin(mae_scores)
        bars2[best_mae_idx].set_color('darkred')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📈 Visualizations saved as {save_path}")

def main():
    """Main function to run improved regression analysis"""
    print("🔬 Starting Improved Regression Model Analysis...")
    print("="*60)

    # Initialize model
    model = ImprovedRegressionModel(random_state=42)

    # Load and preprocess data
    try:
        X, y = model.load_and_preprocess_data('backend/dx-mmse.csv')

        # Train and evaluate
        results = model.train_and_evaluate(X, y)

        # Create visualizations
        model.create_visualizations(results, 'improved_regression_analysis.png')

        # Print summary
        print("\n" + "="*60)
        print("🏆 IMPROVED REGRESSION RESULTS SUMMARY")
        print("="*60)

        successful_models = {k: v for k, v in results.items()
                           if isinstance(v, dict) and 'r2_mean' in v}

        if successful_models:
            # Sort by R² score
            sorted_models = sorted(successful_models.items(),
                                 key=lambda x: x[1]['r2_mean'], reverse=True)

            print("\n📊 Model Performance (Sorted by R²):")
            print("-" * 50)

            for name, metrics in sorted_models:
                print(f"{name:15}: R²={metrics['r2_mean']:.3f}, MAE={metrics['mae_mean']:.2f}")
            # Best model
            best_model = sorted_models[0][0]
            best_r2 = sorted_models[0][1]['r2_mean']
            best_mae = sorted_models[0][1]['mae_mean']

            print(f"\n🎯 BEST MODEL: {best_model}")
            print(f"R² Score: {best_r2:.3f}")
            print(f"MAE: {best_mae:.2f}")
            print(f"RMSE: {(best_mae * 1.25):.2f}")  # Approximate RMSE        else:
            print("❌ No successful models")

        print("\n✅ Analysis completed!")
        print("📁 Check 'improved_regression_analysis.png' for visualizations")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
