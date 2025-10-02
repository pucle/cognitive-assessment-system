"""
Feature Extraction for MMSE Assessment
Implements L_scalar and A_scalar computation with acoustic and linguistic features.
"""

import logging
import numpy as np
import librosa
from scipy import stats
from typing import Dict, List, Tuple, Optional


class FeatureExtractor:
    """Extract linguistic and acoustic features for MMSE assessment."""
    
    def __init__(self):
        self.training_distributions = {}
    
    def extract_linguistic_features(self, transcript: str, per_item_scores: Dict) -> Dict:
        """Extract linguistic features for L_scalar computation."""
        if not transcript:
            return {
                'F_flu': 0.0,
                'TTR': 0.0, 
                'idea_density': 0.0,
                'semantic_similarity_avg': 0.0
            }
        
        words = transcript.lower().split()
        
        # F_flu from F1 task
        F_flu = min(per_item_scores.get('F1', 0) / 15.0, 1.0)
        
        # Type-Token Ratio
        unique_words = set(words)
        TTR = len(unique_words) / len(words) if words else 0.0
        
        # Idea density (simplified: content words / total words)
        # Vietnamese content words heuristic: exclude common function words
        function_words = ['là', 'của', 'và', 'với', 'để', 'có', 'được', 'trong', 'trên', 'dưới']
        content_words = [w for w in words if w not in function_words and len(w) > 2]
        idea_density = len(content_words) / len(words) if words else 0.0
        
        # Semantic similarity (placeholder - would need more sophisticated implementation)
        semantic_similarity_avg = 0.5
        
        return {
            'F_flu': F_flu,
            'TTR': TTR,
            'idea_density': idea_density,
            'semantic_similarity_avg': semantic_similarity_avg,
            'word_count': len(words),
            'unique_word_count': len(unique_words)
        }
    
    def extract_acoustic_features(self, audio_path: str) -> Dict:
        """Extract acoustic features for A_scalar computation."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Speech rate (words per minute) - placeholder
            duration_minutes = len(audio) / sr / 60
            # This would need word timestamps from ASR for accurate calculation
            speech_rate = 120  # Placeholder WPM
            
            # Pause analysis
            # Detect silence segments
            non_silent_intervals = librosa.effects.split(audio, top_db=20)
            total_speech_time = sum(end - start for start, end in non_silent_intervals) / sr
            total_time = len(audio) / sr
            pause_rate = 1 - (total_speech_time / total_time) if total_time > 0 else 0
            
            # F0 variability
            try:
                f0 = librosa.yin(audio, fmin=50, fmax=300, sr=sr)
                f0_clean = f0[f0 > 0]  # Remove unvoiced frames
                f0_variability = np.std(f0_clean) if len(f0_clean) > 0 else 0
                f0_mean = np.mean(f0_clean) if len(f0_clean) > 0 else 150
            except:
                f0_variability = 20  # Default value
                f0_mean = 150
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_stds = np.std(mfccs, axis=1)
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Energy features
            rmse = np.mean(librosa.feature.rms(y=audio))
            
            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            
            features = {
                'speech_rate_wpm': speech_rate,
                'pause_rate': pause_rate,
                'f0_variability': f0_variability,
                'f0_mean': f0_mean,
                'spectral_centroid_mean': spectral_centroid,
                'spectral_bandwidth_mean': spectral_bandwidth,
                'spectral_rolloff_mean': spectral_rolloff,
                'zero_crossing_rate_mean': zero_crossing_rate,
                'rmse_mean': rmse,
                'tempo': tempo,
                'total_duration': total_time,
                'speech_duration': total_speech_time
            }
            
            # Add MFCC features
            for i, (mfcc_mean, mfcc_std) in enumerate(zip(mfcc_means, mfcc_stds)):
                features[f'mfcc_{i+1}_mean'] = mfcc_mean
                features[f'mfcc_{i+1}_std'] = mfcc_std
            
            return features
            
        except Exception as e:
            logging.error(f"Audio feature extraction failed for {audio_path}: {e}")
            # Return default values
            features = {
                'speech_rate_wpm': 120,
                'pause_rate': 0.2,
                'f0_variability': 20,
                'f0_mean': 150,
                'spectral_centroid_mean': 2000,
                'spectral_bandwidth_mean': 1500,
                'spectral_rolloff_mean': 3000,
                'zero_crossing_rate_mean': 0.1,
                'rmse_mean': 0.1,
                'tempo': 120,
                'total_duration': 300,
                'speech_duration': 240
            }
            
            # Add default MFCC features
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = 0.0
                features[f'mfcc_{i+1}_std'] = 1.0
            
            return features
    
    def fit_distributions(self, linguistic_features_list: List[Dict], 
                         acoustic_features_list: List[Dict]):
        """Fit distributions for percentile mapping during training."""
        # Collect values for percentile mapping
        TTR_values = [f['TTR'] for f in linguistic_features_list]
        ID_values = [f['idea_density'] for f in linguistic_features_list]
        F0var_values = [f['f0_variability'] for f in acoustic_features_list]
        
        self.training_distributions = {
            'TTR_values': TTR_values,
            'ID_values': ID_values,
            'F0var_values': F0var_values
        }
        
        logging.info(f"Fitted distributions: TTR range [{min(TTR_values):.3f}, {max(TTR_values):.3f}], "
                    f"ID range [{min(ID_values):.3f}, {max(ID_values):.3f}], "
                    f"F0var range [{min(F0var_values):.3f}, {max(F0var_values):.3f}]")
    
    def compute_scalars(self, linguistic_features: Dict, acoustic_features: Dict) -> Tuple[float, float]:
        """Compute L_scalar and A_scalar from features."""
        
        # L_scalar computation
        F_flu = linguistic_features['F_flu']
        TTR = linguistic_features['TTR']
        idea_density = linguistic_features['idea_density']
        semantic_sim = linguistic_features['semantic_similarity_avg']
        
        # Normalize TTR and idea_density using percentile mapping
        if self.training_distributions:
            TTR_norm = stats.percentileofscore(self.training_distributions['TTR_values'], TTR) / 100
            ID_norm = stats.percentileofscore(self.training_distributions['ID_values'], idea_density) / 100
        else:
            # Fallback to simple normalization
            TTR_norm = min(TTR, 1.0)
            ID_norm = min(idea_density, 1.0)
        
        L_scalar = 0.4 * F_flu + 0.3 * TTR_norm + 0.2 * ID_norm + 0.1 * semantic_sim
        L_scalar = np.clip(L_scalar, 0.0, 1.0)
        
        # A_scalar computation
        speech_rate = acoustic_features['speech_rate_wpm']
        pause_rate = acoustic_features['pause_rate']
        f0_var = acoustic_features['f0_variability']
        
        # Normalize speech rate (60-150 WPM range)
        SR_norm = (np.clip(speech_rate, 60, 150) - 60) / (150 - 60)
        
        # Normalize pause rate (0-0.6 range)
        pause_rate_norm = np.clip(pause_rate, 0.0, 0.6) / 0.6
        Pause_inv = 1.0 - pause_rate_norm
        
        # Normalize F0 variability using percentile mapping
        if self.training_distributions:
            F0var_norm = stats.percentileofscore(self.training_distributions['F0var_values'], f0_var) / 100
        else:
            F0var_norm = min(f0_var / 50, 1.0)  # Normalize by typical max
        
        A_scalar = 0.5 * SR_norm + 0.3 * Pause_inv + 0.2 * F0var_norm
        A_scalar = np.clip(A_scalar, 0.0, 1.0)
        
        return L_scalar, A_scalar
    
    def extract_all_features(self, session_data: Dict, audio_path: str, 
                            per_item_scores: Dict) -> Dict:
        """Extract all features for a session."""
        # Linguistic features
        ling_features = self.extract_linguistic_features(
            session_data['transcript'], per_item_scores
        )
        
        # Acoustic features
        acoustic_features = self.extract_acoustic_features(audio_path)
        
        # Compute scalars
        L_scalar, A_scalar = self.compute_scalars(ling_features, acoustic_features)
        
        # Combine all features
        all_features = {
            **ling_features,
            **acoustic_features,
            'L_scalar': L_scalar,
            'A_scalar': A_scalar
        }
        
        return all_features
