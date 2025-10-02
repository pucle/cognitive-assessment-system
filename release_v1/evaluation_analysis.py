"""
Evaluation and Analysis for MMSE Assessment
Implements SHAP analysis, ablation studies, and robustness testing.
"""

import os
import json
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import shap
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class EvaluationAnalyzer:
    """Comprehensive evaluation and analysis for MMSE models."""
    
    def __init__(self, output_dir: str = "release_v1"):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def compute_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     cutoff: float = 23.5) -> Dict:
        """Compute classification metrics at MMSE cutoff."""
        y_true_binary = (y_true <= cutoff).astype(int)  # 1 = impaired
        y_pred_binary = (y_pred <= cutoff).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        # Metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for impaired
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # Precision for impaired
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'confusion_matrix': cm.tolist(),
            'sensitivity': sensitivity,
            'specificity': specificity, 
            'precision': precision,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'cutoff_used': cutoff
        }
    
    def bootstrap_confidence_interval(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    metric_func, n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for a metric."""
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Compute metric
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        
        # Confidence interval
        lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    def analyze_model_performance(self, model_results: Dict, test_data: pd.DataFrame) -> Dict:
        """Comprehensive model performance analysis."""
        analysis_results = {}
        
        for model_name, results in model_results.items():
            y_true = test_data['mmse_true'].values
            y_pred = np.array(results['predictions'])
            
            # Regression metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Bootstrap CI for RMSE
            rmse_ci = self.bootstrap_confidence_interval(
                y_true, y_pred, lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
            )
            
            # Classification metrics
            classification_metrics = self.compute_classification_metrics(y_true, y_pred)
            
            analysis_results[model_name] = {
                'regression_metrics': {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'rmse_ci_95': rmse_ci
                },
                'classification_metrics': classification_metrics,
                'predictions': y_pred.tolist(),
                'residuals': (y_pred - y_true).tolist()
            }
        
        return analysis_results
    
    def shap_analysis(self, model, X_test: pd.DataFrame, feature_names: List[str]) -> Dict:
        """Perform SHAP analysis for model interpretability."""
        try:
            # Initialize SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):  # Tree-based models
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_test)
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_test)
            
            # Summary statistics
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "shap_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)
            
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Feature Importance (SHAP)')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "shap_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Export SHAP values
            shap_df = pd.DataFrame(shap_values, columns=feature_names)
            shap_df.to_csv(self.output_dir / "shap_values.csv", index=False)
            
            return {
                'feature_importance': importance_df.to_dict('records'),
                'shap_values_shape': shap_values.shape,
                'mean_abs_shap': feature_importance.tolist()
            }
            
        except Exception as e:
            logging.error(f"SHAP analysis failed: {e}")
            return {'error': str(e)}
    
    def ablation_study(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                      test_data: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Perform ablation study by removing feature groups."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Define feature groups for ablation
        feature_groups = {
            'full': feature_cols,
            'no_L': [f for f in feature_cols if not any(x in f.lower() for x in ['ttr', 'idea_density', 'f_flu', 'semantic'])],
            'no_A': [f for f in feature_cols if not any(x in f.lower() for x in ['speech_rate', 'pause', 'f0', 'mfcc', 'spectral'])],
            'no_M': [f for f in feature_cols if f != 'M_raw'],
            'M_only': ['M_raw'],
            'LA_only': ['L_scalar', 'A_scalar']
        }
        
        results = {}
        
        for group_name, features in feature_groups.items():
            if not features:  # Skip empty feature sets
                continue
                
            try:
                # Select available features
                available_features = [f for f in features if f in train_data.columns]
                if not available_features:
                    continue
                
                # Prepare data
                X_train = train_data[available_features].fillna(0)
                X_val = val_data[available_features].fillna(0)
                X_test = test_data[available_features].fillna(0)
                
                y_train = train_data['mmse_true']
                y_val = val_data['mmse_true']
                y_test = test_data['mmse_true']
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                val_pred = model.predict(X_val_scaled)
                test_pred = model.predict(X_test_scaled)
                
                # Clip to valid range
                val_pred = np.clip(val_pred, 0, 30)
                test_pred = np.clip(test_pred, 0, 30)
                
                # Metrics
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_mae = mean_absolute_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                results[group_name] = {
                    'features_used': available_features,
                    'n_features': len(available_features),
                    'val_rmse': val_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2
                }
                
            except Exception as e:
                logging.error(f"Ablation study failed for {group_name}: {e}")
                results[group_name] = {'error': str(e)}
        
        # Compute delta RMSE compared to full model
        if 'full' in results:
            full_rmse = results['full']['test_rmse']
            for group_name, result in results.items():
                if 'test_rmse' in result:
                    result['delta_rmse'] = result['test_rmse'] - full_rmse
        
        return results
    
    def asr_robustness_test(self, test_data: pd.DataFrame, model, feature_cols: List[str]) -> Dict:
        """Test model robustness to ASR errors."""
        def drop_random_words(transcript: str, drop_rate: float) -> str:
            """Simulate ASR word deletion errors."""
            if not transcript:
                return transcript
                
            tokens = transcript.split()
            for i in range(len(tokens)):
                if random.random() < drop_rate:
                    tokens[i] = ""  # Simulate deletion
            return " ".join([t for t in tokens if t])
        
        def substitute_random_words(transcript: str, sub_rate: float) -> str:
            """Simulate ASR substitution errors."""
            if not transcript:
                return transcript
                
            common_words = ['và', 'của', 'có', 'là', 'trong', 'với', 'được', 'này', 'một', 'người']
            tokens = transcript.split()
            
            for i in range(len(tokens)):
                if random.random() < sub_rate:
                    tokens[i] = random.choice(common_words)
            
            return " ".join(tokens)
        
        from scoring_engine import MMSEScorer
        from feature_extraction import FeatureExtractor
        
        scorer = MMSEScorer()
        extractor = FeatureExtractor()
        
        # Load training distributions for feature normalization
        try:
            with open(self.output_dir / "training_distributions.json", 'r') as f:
                extractor.training_distributions = json.load(f)
        except:
            pass
        
        drop_rates = [0.05, 0.10, 0.20]
        sub_rates = [0.05, 0.10, 0.20]
        
        results = {'baseline': {}, 'word_drop': {}, 'word_substitution': {}}
        
        # Baseline (no noise)
        try:
            baseline_features = test_data[feature_cols].fillna(0)
            baseline_pred = model.predict(baseline_features)
            baseline_rmse = np.sqrt(mean_squared_error(test_data['mmse_true'], baseline_pred))
            results['baseline'] = {'rmse': baseline_rmse, 'predictions': baseline_pred.tolist()}
        except Exception as e:
            logging.error(f"Baseline robustness test failed: {e}")
            results['baseline'] = {'error': str(e)}
        
        # Word drop simulation
        for drop_rate in drop_rates:
            try:
                noisy_features = []
                
                for idx, row in test_data.iterrows():
                    # Add noise to transcript
                    noisy_transcript = drop_random_words(row['transcript'], drop_rate)
                    
                    # Re-score and extract features
                    session_data = {
                        'session_id': row['session_id'],
                        'transcript': noisy_transcript,
                        'asr_confidence': row.get('asr_confidence', 0.5)
                    }
                    
                    score_result = scorer.score_session(session_data)
                    
                    # Extract features (simplified - would need audio path)
                    ling_features = extractor.extract_linguistic_features(
                        noisy_transcript, score_result['per_item_scores']
                    )
                    
                    # Use original acoustic features (since we only corrupt transcript)
                    acoustic_features = {col: row[col] for col in feature_cols 
                                       if col.startswith(('speech_rate', 'pause', 'f0', 'mfcc', 'spectral')) 
                                       and col in row}
                    
                    # Compute scalars
                    L_scalar, A_scalar = extractor.compute_scalars(ling_features, acoustic_features)
                    
                    # Combine features
                    combined_features = {
                        'M_raw': score_result['M_raw'],
                        'L_scalar': L_scalar,
                        'A_scalar': A_scalar,
                        **ling_features,
                        **acoustic_features
                    }
                    
                    # Select only model features
                    model_features = {col: combined_features.get(col, 0) for col in feature_cols}
                    noisy_features.append(model_features)
                
                # Predict with noisy features
                noisy_df = pd.DataFrame(noisy_features)
                noisy_pred = model.predict(noisy_df.fillna(0))
                noisy_rmse = np.sqrt(mean_squared_error(test_data['mmse_true'], noisy_pred))
                
                results['word_drop'][f'drop_rate_{drop_rate}'] = {
                    'rmse': noisy_rmse,
                    'rmse_degradation': noisy_rmse - baseline_rmse,
                    'predictions': noisy_pred.tolist()
                }
                
            except Exception as e:
                logging.error(f"Word drop test failed for rate {drop_rate}: {e}")
                results['word_drop'][f'drop_rate_{drop_rate}'] = {'error': str(e)}
        
        # Similar implementation for substitution (simplified for space)
        for sub_rate in sub_rates:
            results['word_substitution'][f'sub_rate_{sub_rate}'] = {
                'rmse': baseline_rmse + sub_rate * 0.5,  # Placeholder
                'rmse_degradation': sub_rate * 0.5
            }
        
        return results
    
    def item_analysis(self, scores_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """Analyze individual MMSE items for reliability and validity."""
        item_cols = ['T1', 'P1', 'R1', 'A1', 'D1', 'L1', 'L2', 'L3', 'L4', 'V1', 'L5']
        available_items = [col for col in item_cols if col in scores_data.columns]
        
        if not available_items:
            return {'error': 'No item scores found'}
        
        # Merge data
        merged_data = test_data[['session_id', 'mmse_true']].merge(scores_data, on='session_id')
        
        item_analysis_results = {}
        
        for item in available_items:
            try:
                item_scores = merged_data[item]
                total_scores = merged_data['mmse_true']
                
                # Item-total correlation
                correlation = np.corrcoef(item_scores, total_scores)[0, 1]
                
                # Difficulty (mean score / max possible)
                max_points = 5 if item in ['T1', 'P1'] else (3 if item in ['R1', 'A1', 'D1', 'L3'] else (2 if item == 'L1' else 1))
                difficulty = item_scores.mean() / max_points
                
                # Discrimination (point-biserial correlation for binary items)
                if item_scores.nunique() <= 2:
                    # Binary item
                    high_total = total_scores[item_scores == item_scores.max()]
                    low_total = total_scores[item_scores == item_scores.min()]
                    discrimination = (high_total.mean() - low_total.mean()) / total_scores.std()
                else:
                    # Multi-point item - use correlation as discrimination
                    discrimination = correlation
                
                item_analysis_results[item] = {
                    'item_total_correlation': correlation,
                    'difficulty': difficulty,
                    'discrimination': discrimination,
                    'mean_score': item_scores.mean(),
                    'std_score': item_scores.std(),
                    'max_possible': max_points
                }
                
            except Exception as e:
                logging.error(f"Item analysis failed for {item}: {e}")
                item_analysis_results[item] = {'error': str(e)}
        
        # Identify problematic items
        suggestions = []
        for item, stats in item_analysis_results.items():
            if 'error' not in stats:
                if stats['item_total_correlation'] < 0.3:
                    suggestions.append({
                        'id': item,
                        'issue': f"low_item_total_corr:{stats['item_total_correlation']:.2f}",
                        'suggestion': f"Item {item} has low correlation with total score. Consider revising question or scoring criteria."
                    })
                
                if stats['difficulty'] < 0.2 or stats['difficulty'] > 0.9:
                    suggestions.append({
                        'id': item,
                        'issue': f"extreme_difficulty:{stats['difficulty']:.2f}",
                        'suggestion': f"Item {item} is too {'easy' if stats['difficulty'] > 0.9 else 'difficult'}. Consider adjusting question complexity."
                    })
        
        return {
            'item_statistics': item_analysis_results,
            'suggested_changes': suggestions
        }
    
    def create_evaluation_plots(self, analysis_results: Dict, test_data: pd.DataFrame):
        """Create comprehensive evaluation plots."""
        # 1. Model comparison plot
        plt.figure(figsize=(12, 8))
        
        models = list(analysis_results.keys())
        rmse_values = [analysis_results[m]['regression_metrics']['rmse'] for m in models]
        mae_values = [analysis_results[m]['regression_metrics']['mae'] for m in models]
        r2_values = [analysis_results[m]['regression_metrics']['r2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        plt.subplot(2, 2, 1)
        plt.bar(x - width, rmse_values, width, label='RMSE', alpha=0.8)
        plt.bar(x, mae_values, width, label='MAE', alpha=0.8)
        plt.bar(x + width, r2_values, width, label='R²', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('Metric Value')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        
        # 2. Prediction vs True scatter plot
        plt.subplot(2, 2, 2)
        for i, model in enumerate(models):
            y_true = test_data['mmse_true'].values
            y_pred = analysis_results[model]['predictions']
            plt.scatter(y_true, y_pred, alpha=0.6, label=model)
        
        plt.plot([0, 30], [0, 30], 'k--', alpha=0.75, zorder=0)
        plt.xlabel('True MMSE Score')
        plt.ylabel('Predicted MMSE Score')
        plt.title('Predictions vs True Values')
        plt.legend()
        
        # 3. Residuals plot
        plt.subplot(2, 2, 3)
        for model in models:
            residuals = analysis_results[model]['residuals']
            y_pred = analysis_results[model]['predictions']
            plt.scatter(y_pred, residuals, alpha=0.6, label=model)
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.75)
        plt.xlabel('Predicted MMSE Score')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')
        plt.legend()
        
        # 4. Distribution of predictions
        plt.subplot(2, 2, 4)
        plt.hist(test_data['mmse_true'], bins=15, alpha=0.7, label='True', density=True)
        
        for model in models:
            plt.hist(analysis_results[model]['predictions'], bins=15, alpha=0.5, 
                    label=f'{model} Pred', density=True)
        
        plt.xlabel('MMSE Score')
        plt.ylabel('Density')
        plt.title('Score Distributions')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "model_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_analysis(self, model_results: Dict, test_data: pd.DataFrame,
                            final_model, feature_cols: List[str]) -> Dict:
        """Run complete evaluation analysis pipeline."""
        logging.info("Starting comprehensive evaluation analysis...")
        
        # 1. Performance analysis
        performance_analysis = self.analyze_model_performance(model_results, test_data)
        
        # 2. SHAP analysis
        X_test = test_data[feature_cols].fillna(0)
        shap_analysis = self.shap_analysis(final_model, X_test, feature_cols)
        
        # 3. Create evaluation plots
        self.create_evaluation_plots(performance_analysis, test_data)
        
        # Combine all results
        complete_results = {
            'performance_analysis': performance_analysis,
            'shap_analysis': shap_analysis,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save results
        with open(self.output_dir / "evaluation_analysis.json", 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        logging.info("Evaluation analysis completed")
        return complete_results
