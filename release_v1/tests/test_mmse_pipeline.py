"""
Unit tests for MMSE Assessment Pipeline
Tests core functionality including scoring, feature extraction, and utilities.
"""

import pytest
import numpy as np
import pandas as pd
import json
import os
import tempfile
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scoring_engine import MMSEScorer
from feature_extraction import FeatureExtractor
from encryption import AudioEncryption


class TestMMSEScorer:
    """Test MMSE scoring engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.scorer = MMSEScorer()
    
    def test_levenshtein_ratio(self):
        """Test Levenshtein ratio computation."""
        # Exact match
        assert self.scorer.levenshtein_ratio_normalized("hello", "hello") == 1.0
        
        # No match
        assert self.scorer.levenshtein_ratio_normalized("", "") == 1.0
        assert self.scorer.levenshtein_ratio_normalized("hello", "") == 0.0
        
        # Partial match
        ratio = self.scorer.levenshtein_ratio_normalized("hello", "helo")
        assert 0.7 < ratio < 1.0
    
    def test_match_word(self):
        """Test word matching with fuzzy logic."""
        # Exact match
        assert self.scorer.match_word("bút", "bút") == True
        
        # Case insensitive
        assert self.scorer.match_word("BÚT", "bút") == True
        
        # Fuzzy match (should pass with high similarity)
        assert self.scorer.match_word("but", "bút") == True
        
        # No match
        assert self.scorer.match_word("hello", "world") == False
    
    def test_parse_date_response(self):
        """Test date parsing functionality."""
        # Complete date response
        response = "14 tháng 9 năm 2025, thứ hai, buổi sáng"
        result = self.scorer.parse_date_response(response)
        
        assert result['day'] == True
        assert result['month'] == True
        assert result['year'] == True
        assert result['weekday'] == True
        assert result['time_of_day'] == True
        
        # Partial response
        response = "sáng"
        result = self.scorer.parse_date_response(response)
        assert result['time_of_day'] == True
        assert result['day'] == False
    
    def test_parse_number_sequence(self):
        """Test serial 7s parsing."""
        # Correct sequence
        response = "93 86 79 72 65"
        expected = [93, 86, 79, 72, 65]
        result = self.scorer.parse_number_sequence(response, expected)
        
        assert all(result)  # All should be True
        
        # Sequence with tolerance
        response = "93 86 80 72 65"  # 80 instead of 79 (within ±1)
        result = self.scorer.parse_number_sequence(response, expected)
        assert result[2] == True  # Should pass with tolerance
    
    def test_score_item_T1(self):
        """Test time orientation scoring."""
        # Complete response
        response = "14 tháng 9 năm 2025, thứ hai, buổi sáng"
        result = self.scorer.score_item("T1", response, 0.8)
        
        assert result['score'] == 5  # All components present
        assert result['error_flag'] == False
    
    def test_score_item_R1(self):
        """Test registration scoring."""
        # All words correct
        response = "bút bàn hoa"
        result = self.scorer.score_item("R1", response, 0.8)
        
        assert result['score'] == 3
        
        # Partial words
        response = "bút table hoa"  # Mix of correct and incorrect
        result = self.scorer.score_item("R1", response, 0.8)
        assert result['score'] == 2  # Should get 2/3
    
    def test_score_session(self):
        """Test complete session scoring."""
        session_data = {
            'session_id': 'test_001',
            'transcript': 'Test transcript with various responses.',
            'asr_confidence': 0.85
        }
        
        result = self.scorer.score_session(session_data)
        
        assert 'session_id' in result
        assert 'per_item_scores' in result
        assert 'M_raw' in result
        assert isinstance(result['M_raw'], (int, float))
        assert 0 <= result['M_raw'] <= 30


class TestFeatureExtractor:
    """Test feature extraction functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = FeatureExtractor()
    
    def test_extract_linguistic_features(self):
        """Test linguistic feature extraction."""
        transcript = "Hôm nay là ngày đẹp trời với nắng nhẹ và gió mát"
        per_item_scores = {'F1': 10}  # Fluency score
        
        features = self.extractor.extract_linguistic_features(transcript, per_item_scores)
        
        assert 'F_flu' in features
        assert 'TTR' in features
        assert 'idea_density' in features
        assert 0 <= features['F_flu'] <= 1
        assert 0 <= features['TTR'] <= 1
        assert 0 <= features['idea_density'] <= 1
    
    def test_extract_acoustic_features_mock(self):
        """Test acoustic feature extraction with mock audio."""
        # This would need a real audio file for full testing
        # For now, test the error handling
        features = self.extractor.extract_acoustic_features("nonexistent.wav")
        
        # Should return default values on error
        assert 'speech_rate_wpm' in features
        assert 'pause_rate' in features
        assert 'f0_variability' in features
        assert features['speech_rate_wpm'] == 120  # Default value
    
    def test_compute_scalars(self):
        """Test L_scalar and A_scalar computation."""
        linguistic_features = {
            'F_flu': 0.8,
            'TTR': 0.6,
            'idea_density': 0.7,
            'semantic_similarity_avg': 0.5
        }
        
        acoustic_features = {
            'speech_rate_wpm': 120,
            'pause_rate': 0.2,
            'f0_variability': 25
        }
        
        L_scalar, A_scalar = self.extractor.compute_scalars(linguistic_features, acoustic_features)
        
        assert 0 <= L_scalar <= 1
        assert 0 <= A_scalar <= 1
    
    def test_fit_distributions(self):
        """Test distribution fitting for percentile mapping."""
        # Mock training data
        linguistic_features = [
            {'TTR': 0.5, 'idea_density': 0.4},
            {'TTR': 0.6, 'idea_density': 0.5},
            {'TTR': 0.7, 'idea_density': 0.6}
        ]
        
        acoustic_features = [
            {'f0_variability': 20},
            {'f0_variability': 25},
            {'f0_variability': 30}
        ]
        
        self.extractor.fit_distributions(linguistic_features, acoustic_features)
        
        assert 'TTR_values' in self.extractor.training_distributions
        assert 'ID_values' in self.extractor.training_distributions
        assert 'F0var_values' in self.extractor.training_distributions


class TestAudioEncryption:
    """Test audio encryption functionality."""
    
    def test_key_generation(self):
        """Test encryption key generation."""
        encryptor = AudioEncryption()
        
        assert len(encryptor.key) == 32  # 256 bits = 32 bytes
        assert isinstance(encryptor.key, bytes)
    
    def test_from_password(self):
        """Test key derivation from password."""
        password = "test_password_123"
        salt = b"test_salt_16byte"
        
        encryptor = AudioEncryption.from_password(password, salt)
        
        assert len(encryptor.key) == 32
        
        # Same password + salt should generate same key
        encryptor2 = AudioEncryption.from_password(password, salt)
        assert encryptor.key == encryptor2.key
    
    def test_encrypt_decrypt_cycle(self):
        """Test encrypt/decrypt cycle with temporary files."""
        encryptor = AudioEncryption()
        
        # Create temporary test data
        test_data = b"This is test audio data" * 1000  # Simulate audio file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "test_input.wav")
            encrypted_path = os.path.join(tmpdir, "test_encrypted.enc")
            decrypted_path = os.path.join(tmpdir, "test_decrypted.wav")
            
            # Write test data
            with open(input_path, "wb") as f:
                f.write(test_data)
            
            # Encrypt
            assert encryptor.encrypt_file(input_path, encrypted_path) == True
            assert os.path.exists(encrypted_path)
            
            # Decrypt
            assert encryptor.decrypt_file(encrypted_path, decrypted_path) == True
            assert os.path.exists(decrypted_path)
            
            # Verify data integrity
            with open(decrypted_path, "rb") as f:
                decrypted_data = f.read()
            
            assert decrypted_data == test_data
    
    def test_key_save_load(self):
        """Test key save and load functionality."""
        encryptor1 = AudioEncryption()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = os.path.join(tmpdir, "test_key.bin")
            
            # Save key
            assert encryptor1.save_key(key_path) == True
            assert os.path.exists(key_path)
            
            # Load key
            encryptor2 = AudioEncryption.load_key(key_path)
            assert encryptor2 is not None
            assert encryptor1.key == encryptor2.key


class TestUtilities:
    """Test utility functions."""
    
    def test_questions_json_validity(self):
        """Test questions.json file structure."""
        questions_path = Path(__file__).parent.parent / "questions.json"
        
        if questions_path.exists():
            with open(questions_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            assert isinstance(questions, list)
            assert len(questions) > 0
            
            # Check required fields
            for q in questions:
                assert 'id' in q
                assert 'domain' in q
                assert 'question_text' in q
                assert 'max_points' in q or q['max_points'] is None  # F1 is auxiliary
            
            # Check total points (excluding auxiliary)
            total_points = sum(q['max_points'] for q in questions if q['max_points'] is not None)
            assert total_points == 30
    
    def test_nnls_weights_function(self):
        """Test NNLS weight fitting function."""
        from scipy.optimize import nnls
        
        # Mock data
        np.random.seed(42)
        n_samples = 100
        
        M_values = np.random.uniform(15, 25, n_samples)
        L_values = np.random.uniform(0.4, 0.9, n_samples)
        A_values = np.random.uniform(0.3, 0.8, n_samples)
        
        # Create target with known weights
        true_w_M, true_w_L, true_w_A = 0.6, 0.3, 0.1
        y_true = true_w_M * M_values + true_w_L * (L_values * 30) + true_w_A * (A_values * 30)
        
        # Add some noise
        y_true += np.random.normal(0, 0.5, n_samples)
        
        # Fit weights
        X = np.column_stack([M_values, L_values * 30, A_values * 30])
        w_raw, rnorm = nnls(X, y_true)
        w_normalized = w_raw / w_raw.sum()
        
        # Check that weights are reasonable
        assert len(w_normalized) == 3
        assert np.abs(w_normalized.sum() - 1.0) < 1e-10
        assert all(w >= 0 for w in w_normalized)
        
        # Weights should be close to true values (with some tolerance due to noise)
        assert np.abs(w_normalized[0] - true_w_M) < 0.1
        assert np.abs(w_normalized[1] - true_w_L) < 0.1
        assert np.abs(w_normalized[2] - true_w_A) < 0.1


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
