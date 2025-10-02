#!/usr/bin/env python3
"""
MMSE-like Assessment Training Pipeline
Implements comprehensive training pipeline for Vietnamese audio-based cognitive assessment.
"""

import os
import sys
import json
import logging
import argparse
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
import librosa
import soundfile as sf
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb

# NLP libraries
import whisper
from sentence_transformers import SentenceTransformer
from Levenshtein import ratio as levenshtein_ratio

# Visualization and reporting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm

# Optional optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class MMSEAssessmentPipeline:
    """Main pipeline for MMSE assessment training and evaluation."""
    
    def __init__(self, output_dir: str = "release_v1"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "intermediate").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "tests").mkdir(exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Load questions schema
        self.questions = self.load_questions()
        
        # Initialize models
        self.whisper_model = None
        self.sbert_model = None
        self.scaler = StandardScaler()
        
        # Data containers
        self.raw_data = None
        self.train_data = None
        self.val_data = None 
        self.test_data = None
        
        # Model artifacts
        self.final_model = None
        self.model_metadata = {}
        self.weights = {}
        
        logging.info(f"Pipeline initialized with output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup structured JSON logging."""
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        
        # File handler
        file_handler = logging.FileHandler(self.output_dir / "pipeline.log")
        file_handler.setFormatter(log_format)
        
        logging.basicConfig(
            level=logging.INFO,
            handlers=[console_handler, file_handler]
        )
    
    def load_questions(self) -> List[Dict]:
        """Load questions schema from JSON file."""
        questions_path = self.output_dir / "questions.json"
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        # Validate total points (excluding auxiliary F1)
        total_points = sum(q['max_points'] for q in questions if q['max_points'] is not None)
        if total_points != 30:
            raise ValueError(f"Questions total points = {total_points}, expected 30")
            
        logging.info(f"Loaded {len(questions)} questions, total points: {total_points}")
        return questions
    
    def validate_input_data(self, dataset_path: str) -> bool:
        """Validate input dataset and audio files."""
        try:
            # Check CSV exists
            if not os.path.exists(dataset_path):
                self.output_error("missing_file", f"Dataset file not found: {dataset_path}")
                return False
            
            # Load and validate CSV
            df = pd.read_csv(dataset_path)
            required_cols = ['session_id', 'audio_path', 'mmse_true', 'age', 'gender', 'education_years']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.output_error("missing_columns", 
                                f"Missing required columns: {missing_cols}")
                return False
            
            # Check data requirements
            if len(df) < 50:
                self.output_error("insufficient_data",
                                f"Need at least 50 examples, got {len(df)}")
                return False
            
            # Check audio files
            missing_audio = []
            for idx, row in df.iterrows():
                audio_path = row['audio_path']
                if not os.path.isabs(audio_path):
                    # Try relative to dataset directory
                    dataset_dir = os.path.dirname(dataset_path)
                    audio_path = os.path.join(dataset_dir, audio_path)
                
                if not os.path.exists(audio_path):
                    missing_audio.append(row['audio_path'])
            
            if missing_audio:
                self.output_error("missing_audio_file",
                                f"Missing audio files: {missing_audio[:5]}...")  # Show first 5
                return False
            
            logging.info(f"Dataset validation passed: {len(df)} samples")
            return True
            
        except Exception as e:
            self.output_error("validation_failed", str(e))
            return False
    
    def output_error(self, code: str, message: str, details: Dict = None):
        """Output structured error JSON and exit."""
        error_obj = {
            "status": "error",
            "code": code,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        if details:
            error_obj["details"] = details
        
        error_path = self.output_dir / "error.json"
        with open(error_path, 'w') as f:
            json.dump(error_obj, f, indent=2)
        
        logging.error(f"Pipeline failed: {code} - {message}")
        sys.exit(1)
    
    def load_and_split_data(self, dataset_path: str):
        """Load dataset and create train/val/test splits."""
        df = pd.read_csv(dataset_path)
        
        # Create MMSE severity buckets for stratification
        df['mmse_bucket'] = pd.cut(df['mmse_true'], 
                                  bins=[0, 23, 30], 
                                  labels=['impaired', 'normal'],
                                  include_lowest=True)
        
        # First split: train vs test (85% vs 15%)
        train_val, test = train_test_split(
            df, test_size=0.15, 
            stratify=df['mmse_bucket'],
            random_state=RANDOM_SEED
        )
        
        # Second split: train vs val (70% vs 15% of original)
        train, val = train_test_split(
            train_val, test_size=0.176,  # 0.15/0.85 ≈ 0.176
            stratify=train_val['mmse_bucket'],
            random_state=RANDOM_SEED
        )
        
        self.raw_data = df
        self.train_data = train.reset_index(drop=True)
        self.val_data = val.reset_index(drop=True)
        self.test_data = test.reset_index(drop=True)
        
        logging.info(f"Data splits - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        # Save splits
        self.train_data.to_csv(self.output_dir / "intermediate" / "train_split.csv", index=False)
        self.val_data.to_csv(self.output_dir / "intermediate" / "val_split.csv", index=False)
        self.test_data.to_csv(self.output_dir / "intermediate" / "test_split.csv", index=False)
    
    def initialize_models(self):
        """Initialize ASR and NLP models."""
        logging.info("Initializing models...")
        
        # Whisper ASR
        try:
            self.whisper_model = whisper.load_model("large-v2")
            logging.info("Loaded Whisper large-v2 model")
        except Exception as e:
            self.output_error("asr_failed", f"Failed to load Whisper: {e}")
        
        # Sentence transformer for semantic similarity
        try:
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logging.info("Loaded multilingual sentence transformer")
        except Exception as e:
            logging.warning(f"Failed to load sentence transformer: {e}")
            self.sbert_model = None

    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio file using Whisper."""
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Whisper transcription
            result = self.whisper_model.transcribe(
                audio, 
                language='vi',
                word_timestamps=True
            )
            
            # Extract word-level confidences if available
            word_confidences = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        word_confidences.extend([w.get('probability', 0.5) for w in segment['words']])
            
            avg_confidence = np.mean(word_confidences) if word_confidences else 0.5
            
            return {
                'transcript': result['text'].strip(),
                'confidence': avg_confidence,
                'word_timestamps': result.get('segments', []),
                'language': result.get('language', 'vi')
            }
            
        except Exception as e:
            logging.error(f"Transcription failed for {audio_path}: {e}")
            return {
                'transcript': '',
                'confidence': 0.0,
                'word_timestamps': [],
                'language': 'vi'
            }
    
    def process_transcriptions(self, data_split: pd.DataFrame) -> pd.DataFrame:
        """Process transcriptions for a data split."""
        transcripts = []
        
        for idx, row in tqdm(data_split.iterrows(), desc="Transcribing", total=len(data_split)):
            audio_path = row['audio_path']
            if not os.path.isabs(audio_path):
                # Assume relative to current directory or add path resolution logic
                audio_path = os.path.abspath(audio_path)
            
            result = self.transcribe_audio(audio_path)
            
            transcripts.append({
                'session_id': row['session_id'],
                'transcript': result['transcript'],
                'asr_confidence': result['confidence'],
                'asr_engine': 'whisper-large-v2',
                'word_timestamps': json.dumps(result['word_timestamps']),
                'processing_time_seconds': 0  # Could add timing
            })
        
        transcript_df = pd.DataFrame(transcripts)
        return data_split.merge(transcript_df, on='session_id')


    def process_all_sessions(self):
        """Process scoring and feature extraction for all sessions."""
        from scoring_engine import MMSEScorer
        from feature_extraction import FeatureExtractor
        
        scorer = MMSEScorer(self.sbert_model)
        extractor = FeatureExtractor()
        
        # Process each data split
        for split_name, data_split in [('train', self.train_data), 
                                      ('val', self.val_data), 
                                      ('test', self.test_data)]:
            
            logging.info(f"Processing {split_name} split ({len(data_split)} sessions)")
            
            session_scores = []
            session_features = []
            
            for idx, row in data_split.iterrows():
                try:
                    # Score session
                    session_data = {
                        'session_id': row['session_id'],
                        'transcript': row['transcript'],
                        'asr_confidence': row['asr_confidence']
                    }
                    
                    score_result = scorer.score_session(session_data)
                    
                    # Extract features
                    audio_path = row['audio_path']
                    if not os.path.isabs(audio_path):
                        audio_path = os.path.abspath(audio_path)
                    
                    features = extractor.extract_all_features(
                        session_data, audio_path, score_result['per_item_scores']
                    )
                    
                    # Update score result with computed scalars
                    score_result['L_scalar'] = features['L_scalar']
                    score_result['A_scalar'] = features['A_scalar']
                    
                    session_scores.append(score_result)
                    session_features.append({
                        'session_id': row['session_id'],
                        **features
                    })
                    
                except Exception as e:
                    logging.error(f"Processing failed for session {row['session_id']}: {e}")
                    continue
            
            # Save results
            scores_df = pd.DataFrame(session_scores)
            features_df = pd.DataFrame(session_features)
            
            scores_df.to_csv(self.output_dir / "intermediate" / f"{split_name}_scores.csv", index=False)
            features_df.to_csv(self.output_dir / "intermediate" / f"{split_name}_features.csv", index=False)
        
        # Fit distributions on training data for percentile mapping
        train_features = pd.read_csv(self.output_dir / "intermediate" / "train_features.csv")
        extractor.fit_distributions(
            train_features.to_dict('records'),
            train_features.to_dict('records')
        )
        
        # Save distributions
        with open(self.output_dir / "training_distributions.json", 'w') as f:
            json.dump(extractor.training_distributions, f, indent=2)
        
        logging.info("Session processing completed")
    
    def fit_weights_nnls(self) -> Dict[str, float]:
        """Fit MMSE prediction weights using NNLS."""
        from scipy.optimize import nnls
        
        # Load training data
        train_scores = pd.read_csv(self.output_dir / "intermediate" / "train_scores.csv")
        train_data_merged = self.train_data.merge(train_scores, on='session_id')
        
        # Prepare matrices
        M_values = train_data_merged['M_raw'].values
        L_values = train_data_merged['L_scalar'].values
        A_values = train_data_merged['A_scalar'].values
        y_true = train_data_merged['mmse_true'].values
        
        # Build feature matrix
        X = np.column_stack([M_values, L_values * 30.0, A_values * 30.0])
        
        # Fit NNLS
        w_raw, rnorm = nnls(X, y_true)
        w_normalized = w_raw / w_raw.sum() if w_raw.sum() > 0 else w_raw
        
        weights = {
            'w_M': float(w_normalized[0]),
            'w_L': float(w_normalized[1]), 
            'w_A': float(w_normalized[2]),
            'method': 'NNLS_normalized',
            'rnorm': float(rnorm)
        }
        
        # Compute predictions and validation RMSE
        predictions = X.dot(w_raw)
        train_rmse = np.sqrt(mean_squared_error(y_true, predictions))
        
        # Also try OLS and Ridge for comparison
        from sklearn.linear_model import LinearRegression, Ridge
        
        ols = LinearRegression().fit(X, y_true)
        ols_pred = ols.predict(X)
        ols_rmse = np.sqrt(mean_squared_error(y_true, ols_pred))
        
        ridge = Ridge(alpha=1.0).fit(X, y_true)
        ridge_pred = ridge.predict(X)
        ridge_rmse = np.sqrt(mean_squared_error(y_true, ridge_pred))
        
        logging.info(f"Weight fitting results:")
        logging.info(f"NNLS RMSE: {train_rmse:.3f}, weights: {weights}")
        logging.info(f"OLS RMSE: {ols_rmse:.3f}")
        logging.info(f"Ridge RMSE: {ridge_rmse:.3f}")
        
        # Choose best method
        if ols_rmse < train_rmse and ols_rmse < ridge_rmse and all(ols.coef_ >= 0):
            weights = {
                'w_M': float(ols.coef_[0] / ols.coef_.sum()),
                'w_L': float(ols.coef_[1] / ols.coef_.sum()),
                'w_A': float(ols.coef_[2] / ols.coef_.sum()),
                'method': 'OLS_normalized',
                'rnorm': ols_rmse
            }
        elif ridge_rmse < train_rmse and all(ridge.coef_ >= 0):
            weights = {
                'w_M': float(ridge.coef_[0] / ridge.coef_.sum()),
                'w_L': float(ridge.coef_[1] / ridge.coef_.sum()),
                'w_A': float(ridge.coef_[2] / ridge.coef_.sum()),
                'method': 'Ridge_normalized',
                'rnorm': ridge_rmse
            }
        
        self.weights = weights
        return weights
    
    def train_models(self):
        """Train and evaluate multiple models."""
        # Load features and merge with ground truth
        train_features = pd.read_csv(self.output_dir / "intermediate" / "train_features.csv")
        val_features = pd.read_csv(self.output_dir / "intermediate" / "val_features.csv")
        test_features = pd.read_csv(self.output_dir / "intermediate" / "test_features.csv")
        
        train_scores = pd.read_csv(self.output_dir / "intermediate" / "train_scores.csv")
        val_scores = pd.read_csv(self.output_dir / "intermediate" / "val_scores.csv")
        test_scores = pd.read_csv(self.output_dir / "intermediate" / "test_scores.csv")
        
        # Merge data
        train_data = self.train_data.merge(train_features, on='session_id').merge(train_scores, on='session_id')
        val_data = self.val_data.merge(val_features, on='session_id').merge(val_scores, on='session_id')
        test_data = self.test_data.merge(test_features, on='session_id').merge(test_scores, on='session_id')
        
        # Feature selection
        feature_cols = [col for col in train_features.columns 
                       if col not in ['session_id'] and not col.endswith('_details')]
        
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['mmse_true']
        
        X_val = val_data[feature_cols].fillna(0)
        y_val = val_data['mmse_true']
        
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data['mmse_true']
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model configurations
        models = {
            'M_only': {
                'features': ['M_raw'],
                'model': Ridge(alpha=1.0, random_state=RANDOM_SEED)
            },
            'M_plus_LA': {
                'features': ['M_raw', 'L_scalar', 'A_scalar'],
                'model': Ridge(alpha=1.0, random_state=RANDOM_SEED)
            },
            'XGBoost': {
                'features': feature_cols,
                'model': xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    subsample=0.8,
                    random_state=RANDOM_SEED,
                    early_stopping_rounds=50
                )
            },
            'LightGBM': {
                'features': feature_cols,
                'model': lgb.LGBMRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    subsample=0.8,
                    random_state=RANDOM_SEED
                )
            }
        }
        
        results = {}
        
        for model_name, config in models.items():
            logging.info(f"Training {model_name}...")
            
            # Select features
            if config['features'] != feature_cols:
                X_train_model = train_data[config['features']].fillna(0)
                X_val_model = val_data[config['features']].fillna(0)
                X_test_model = test_data[config['features']].fillna(0)
                
                # Scale if needed
                if model_name.startswith('XGB') or model_name.startswith('Light'):
                    scaler_model = StandardScaler()
                    X_train_model = scaler_model.fit_transform(X_train_model)
                    X_val_model = scaler_model.transform(X_val_model)
                    X_test_model = scaler_model.transform(X_test_model)
            else:
                X_train_model = X_train_scaled
                X_val_model = X_val_scaled  
                X_test_model = X_test_scaled
            
            # Train model
            if 'XGB' in model_name:
                config['model'].fit(
                    X_train_model, y_train,
                    eval_set=[(X_val_model, y_val)],
                    verbose=False
                )
            else:
                config['model'].fit(X_train_model, y_train)
            
            # Predictions
            train_pred = config['model'].predict(X_train_model)
            val_pred = config['model'].predict(X_val_model)
            test_pred = config['model'].predict(X_test_model)
            
            # Clip predictions to valid range
            train_pred = np.clip(train_pred, 0, 30)
            val_pred = np.clip(val_pred, 0, 30)
            test_pred = np.clip(test_pred, 0, 30)
            
            # Evaluate
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            results[model_name] = {
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'model': config['model'],
                'features': config['features'],
                'predictions': test_pred.tolist()
            }
            
            logging.info(f"{model_name} - Val RMSE: {val_rmse:.3f}, Test RMSE: {test_rmse:.3f}")
        
        # Select best model based on validation RMSE
        best_model_name = min(results.keys(), key=lambda k: results[k]['val_rmse'])
        self.final_model = results[best_model_name]['model']
        
        logging.info(f"Best model: {best_model_name}")
        
        # Save model and results
        import joblib
        joblib.dump(self.final_model, self.output_dir / "model_MMSE_v1.pkl")
        
        with open(self.output_dir / "model_results.json", 'w') as f:
            # Convert models to serializable format
            serializable_results = {}
            for name, result in results.items():
                serializable_results[name] = {
                    k: v for k, v in result.items() if k != 'model'
                }
            json.dump(serializable_results, f, indent=2)
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMSE Assessment Training Pipeline")
    parser.add_argument("--dataset", required=True, help="Path to raw_dataset.csv")
    parser.add_argument("--output_dir", default="release_v1", help="Output directory")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Set global seed
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Initialize and run pipeline
    pipeline = MMSEAssessmentPipeline(args.output_dir)
    
    # Validate input
    if not pipeline.validate_input_data(args.dataset):
        sys.exit(1)
    
    # Record run start
    start_time = time.time()
    run_info = {
        "command": " ".join(sys.argv),
        "seed": RANDOM_SEED,
        "start_time": datetime.now().isoformat(),
        "dataset_path": args.dataset
    }
    
    try:
        # Main pipeline execution
        logging.info("Starting MMSE training pipeline...")
        
        # Load and split data
        pipeline.load_and_split_data(args.dataset)
        
        # Initialize models
        pipeline.initialize_models()
        
        # Process transcriptions
        logging.info("Processing transcriptions...")
        pipeline.train_data = pipeline.process_transcriptions(pipeline.train_data)
        pipeline.val_data = pipeline.process_transcriptions(pipeline.val_data)
        pipeline.test_data = pipeline.process_transcriptions(pipeline.test_data)
        
        # Save intermediate transcripts
        transcript_cols = ['session_id', 'transcript', 'asr_confidence', 'asr_engine']
        all_transcripts = pd.concat([
            pipeline.train_data[transcript_cols],
            pipeline.val_data[transcript_cols], 
            pipeline.test_data[transcript_cols]
        ])
        all_transcripts.to_csv(pipeline.output_dir / "intermediate" / "transcripts.csv", index=False)
        
        logging.info("Transcription phase completed")
        
        # Process all sessions (scoring + feature extraction)
        pipeline.process_all_sessions()
        
        # Fit NNLS weights
        weights = pipeline.fit_weights_nnls()
        
        # Train and evaluate models
        model_results = pipeline.train_models()
        
        # Create metadata file
        metadata = {
            "model_name": "XGBoost",  # Will be updated based on best model
            "model_version": "v1",
            "weights": weights,
            "scalers": {
                "feature_means": pipeline.scaler.mean_.tolist(),
                "feature_stds": pipeline.scaler.scale_.tolist()
            },
            "feature_list": list(model_results['XGBoost']['features']),
            "training_seed": RANDOM_SEED,
            "train_val_test_split": {"train": 0.70, "val": 0.15, "test": 0.15},
            "date_trained": datetime.now().isoformat(),
            "notes": "Automated MMSE assessment with audio + transcript features"
        }
        
        with open(pipeline.output_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate test predictions CSV
        test_scores = pd.read_csv(pipeline.output_dir / "intermediate" / "test_scores.csv")
        test_features = pd.read_csv(pipeline.output_dir / "intermediate" / "test_features.csv")
        test_merged = pipeline.test_data.merge(test_scores, on='session_id').merge(test_features, on='session_id')
        
        # Apply final weights to compute Score_total
        test_merged['Score_total_raw'] = (
            weights['w_M'] * test_merged['M_raw'] +
            weights['w_L'] * test_merged['L_scalar'] * 30.0 +
            weights['w_A'] * test_merged['A_scalar'] * 30.0
        )
        test_merged['Score_total_rounded'] = test_merged['Score_total_raw'].round()
        
        # Create final predictions DataFrame
        prediction_cols = ['session_id', 'mmse_true', 'Score_total_rounded', 'M_raw', 'L_scalar', 'A_scalar']
        item_cols = ['T1', 'P1', 'R1', 'A1', 'D1', 'L1', 'L2', 'L3', 'L4', 'V1', 'L5']
        feature_cols = ['speech_rate_wpm', 'pause_rate', 'f0_variability']
        
        final_cols = prediction_cols + item_cols + feature_cols
        available_cols = [col for col in final_cols if col in test_merged.columns]
        
        test_predictions = test_merged[available_cols]
        test_predictions = test_predictions.rename(columns={'Score_total_rounded': 'mmse_pred'})
        test_predictions.to_csv(pipeline.output_dir / "test_predictions.csv", index=False)
        
        # Run comprehensive evaluation
        from evaluation_analysis import EvaluationAnalyzer
        
        evaluator = EvaluationAnalyzer(pipeline.output_dir)
        
        # Get test data with all merged features
        test_data_full = pipeline.test_data.merge(test_scores, on='session_id').merge(test_features, on='session_id')
        
        # Run evaluation analysis
        evaluation_results = evaluator.run_complete_analysis(
            model_results, test_data_full, pipeline.final_model, 
            list(model_results['XGBoost']['features'])
        )
        
        # Run ablation study
        ablation_results = evaluator.ablation_study(
            pipeline.train_data.merge(train_scores, on='session_id').merge(train_features, on='session_id'),
            pipeline.val_data.merge(val_scores, on='session_id').merge(val_features, on='session_id'),
            test_data_full,
            list(model_results['XGBoost']['features'])
        )
        
        # Run robustness testing
        robustness_results = evaluator.asr_robustness_test(
            test_data_full, pipeline.final_model, 
            list(model_results['XGBoost']['features'])
        )
        
        # Item analysis
        item_results = evaluator.item_analysis(test_scores, pipeline.test_data)
        
        # Save additional analysis results
        with open(pipeline.output_dir / "ablation_results.json", 'w') as f:
            json.dump(ablation_results, f, indent=2, default=str)
        
        with open(pipeline.output_dir / "robustness_results.json", 'w') as f:
            json.dump(robustness_results, f, indent=2, default=str)
        
        with open(pipeline.output_dir / "item_analysis.json", 'w') as f:
            json.dump(item_results, f, indent=2, default=str)
        
        # Generate questions_suggested_changes.json if needed
        if 'suggested_changes' in item_results and item_results['suggested_changes']:
            with open(pipeline.output_dir / "questions_suggested_changes.json", 'w') as f:
                json.dump(item_results['suggested_changes'], f, indent=2)
        
        # Generate evaluation report
        from report_generator import MMSEReportGenerator
        
        report_generator = MMSEReportGenerator(pipeline.output_dir)
        report_path = report_generator.generate_report(
            analysis_results=evaluation_results,
            model_results=model_results,
            ablation_results=ablation_results,
            robustness_results=robustness_results,
            item_results=item_results
        )
        
        logging.info(f"Evaluation report generated: {report_path}")
        logging.info("Training pipeline completed successfully")
        
    except Exception as e:
        run_info["error"] = str(e)
        logging.error(f"Pipeline failed: {e}")
        raise
    finally:
        # Record completion
        run_info["duration_seconds"] = time.time() - start_time
        run_info["end_time"] = datetime.now().isoformat()
        
        with open(pipeline.output_dir / "run_log.json", 'w') as f:
            json.dump(run_info, f, indent=2)
