"""
MMSE Scoring Engine
Implements detailed scoring logic for Vietnamese MMSE-like assessment.
"""

import re
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from Levenshtein import ratio as levenshtein_ratio


class MMSEScorer:
    """Scoring engine for MMSE items with fuzzy matching and semantic similarity."""
    
    def __init__(self, sbert_model=None):
        self.sbert_model = sbert_model
        self.questions = self.load_questions()
    
    def load_questions(self) -> List[Dict]:
        """Load questions from JSON file."""
        try:
            with open('release_v1/questions.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error("questions.json not found")
            return []
    
    def levenshtein_ratio_normalized(self, a: str, b: str) -> float:
        """Compute normalized Levenshtein ratio."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return levenshtein_ratio(a.lower().strip(), b.lower().strip())
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence transformers."""
        if not self.sbert_model or not text1 or not text2:
            return 0.0
        
        try:
            embeddings = self.sbert_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logging.warning(f"Semantic similarity failed: {e}")
            return 0.0
    
    def match_word(self, predicted: str, target: str) -> bool:
        """Check if predicted word matches target using fuzzy and semantic matching."""
        if not predicted or not target:
            return False
        
        # Exact match
        if predicted.lower().strip() == target.lower().strip():
            return True
        
        # Fuzzy match (Levenshtein >= 0.8)
        if self.levenshtein_ratio_normalized(predicted, target) >= 0.80:
            return True
        
        # Semantic match (>= 0.7)
        if self.semantic_similarity(predicted, target) >= 0.70:
            return True
        
        return False
    
    def parse_date_response(self, response: str) -> Dict[str, bool]:
        """Parse date response and check components."""
        response = response.lower().strip()
        results = {
            'day': False,
            'month': False, 
            'year': False,
            'weekday': False,
            'time_of_day': False
        }
        
        # Day (1-31)
        day_match = re.search(r'\b([1-3]?[0-9])\b', response)
        if day_match:
            day = int(day_match.group(1))
            if 1 <= day <= 31:
                results['day'] = True
        
        # Month (1-12 or month names)
        month_patterns = [
            r'\b(1[0-2]|[1-9])\b',  # Numeric months
            r'\b(tháng\s*(1[0-2]|[1-9]))\b',  # "tháng X"
            r'\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mười một|mười hai)\b'  # Written numbers
        ]
        if any(re.search(pattern, response) for pattern in month_patterns):
            results['month'] = True
        
        # Year (20xx)
        year_match = re.search(r'\b(20[0-9]{2})\b', response)
        if year_match:
            results['year'] = True
        
        # Weekday
        weekdays = ['chủ nhật', 'thứ hai', 'thứ ba', 'thứ tư', 'thứ năm', 'thứ sáu', 'thứ bảy']
        if any(day in response for day in weekdays):
            results['weekday'] = True
        
        # Time of day
        time_patterns = ['sáng', 'chiều', 'tối', 'đêm', 'buổi sáng', 'buổi chiều', 'buổi tối']
        if any(time in response for time in time_patterns):
            results['time_of_day'] = True
        
        return results
    
    def parse_number_sequence(self, response: str, expected: List[int]) -> List[bool]:
        """Parse number sequence (serial 7s) with ±1 tolerance."""
        # Extract numbers from response
        numbers = re.findall(r'\b(\d+)\b', response)
        numbers = [int(n) for n in numbers]
        
        results = []
        for i, expected_num in enumerate(expected):
            if i < len(numbers):
                actual = numbers[i]
                # Allow ±1 tolerance
                if abs(actual - expected_num) <= 1:
                    results.append(True)
                else:
                    results.append(False)
            else:
                results.append(False)
        
        return results
    
    def score_item(self, item_id: str, response: str, confidence: float) -> Dict:
        """Score individual MMSE item."""
        question = next((q for q in self.questions if q['id'] == item_id), None)
        if not question:
            return {'score': 0, 'confidence': confidence, 'error_flag': True, 'details': {}}
        
        score = 0
        error_flag = confidence < 0.6
        details = {}
        
        if item_id == 'T1':  # Time orientation
            date_results = self.parse_date_response(response)
            score = sum(date_results.values())
            details = date_results
            
        elif item_id == 'P1':  # Place orientation
            # Check for hierarchical location components
            response_lower = response.lower()
            
            # Country
            if 'việt nam' in response_lower or 'vn' in response_lower:
                score += 1
            
            # Province/State  
            provinces = ['hà nội', 'hồ chí minh', 'đà nẵng', 'hải phòng', 'cần thơ']
            if any(prov in response_lower for prov in provinces):
                score += 1
            
            # City/District
            if any(word in response_lower for word in ['quận', 'huyện', 'thành phố', 'tp']):
                score += 1
            
            # District/Specific
            if any(word in response_lower for word in ['phường', 'xã', 'khu', 'đường']):
                score += 1
            
            # Specific place
            if any(word in response_lower for word in ['nhà', 'bệnh viện', 'phòng khám', 'văn phòng']):
                score += 1
                
        elif item_id == 'R1':  # Registration
            target_words = ['bút', 'bàn', 'hoa']
            response_words = response.lower().split()
            
            for target in target_words:
                if any(self.match_word(word, target) for word in response_words):
                    score += 1
                    
        elif item_id == 'A1':  # Attention (serial 7s)
            expected_sequence = [93, 86, 79, 72, 65]
            results = self.parse_number_sequence(response, expected_sequence)
            score = sum(results)
            details = {'sequence_correct': results}
            
        elif item_id == 'D1':  # Recall
            target_words = ['bút', 'bàn', 'hoa']
            response_words = response.lower().split()
            
            for target in target_words:
                if any(self.match_word(word, target) for word in response_words):
                    score += 1
                    
        elif item_id == 'L1':  # Naming (pen)
            response_lower = response.lower()
            
            # Name match
            if self.match_word(response_lower, 'bút'):
                score += 1
            
            # Function description
            if any(word in response_lower for word in ['viết', 'ghi', 'chép']):
                score += 1
                
        elif item_id == 'L2':  # Repetition
            target_sentence = "hôm nay trời nắng nhẹ"
            
            # Levenshtein similarity
            lev_sim = self.levenshtein_ratio_normalized(response, target_sentence)
            sem_sim = self.semantic_similarity(response, target_sentence)
            
            if lev_sim >= 0.85 or sem_sim >= 0.85:
                score = 1
            details = {'levenshtein': lev_sim, 'semantic': sem_sim}
            
        elif item_id == 'L3':  # 3-step command
            response_lower = response.lower()
            steps = ['cầm', 'chạm', 'xong']
            
            for step in steps:
                if step in response_lower:
                    score += 1
                    
        elif item_id == 'L4':  # Sentence construction
            # Simple heuristic for subject-verb detection
            response_lower = response.lower()
            
            # Look for common subjects and verbs
            subjects = ['hôm nay', 'trời', 'tôi', 'chúng ta']
            verbs = ['nắng', 'mưa', 'lạnh', 'ấm', 'đẹp', 'xấu']
            
            has_subject = any(subj in response_lower for subj in subjects)
            has_verb = any(verb in response_lower for verb in verbs)
            
            if has_subject and has_verb and len(response.split()) >= 3:
                score = 1
                
        elif item_id == 'V1':  # Clock (visuospatial)
            response_lower = response.lower()
            
            # Check for "2", "hai", "số 2"
            if any(pattern in response_lower for pattern in ['2', 'hai', 'số 2']):
                score = 1
                
        elif item_id == 'L5':  # Naming (chair)
            if self.match_word(response, 'ghế'):
                score = 1
                
        elif item_id == 'F1':  # Fluency (auxiliary)
            # Count unique fruit names
            fruits = ['xoài', 'chuối', 'táo', 'cam', 'chanh', 'nho', 'dưa', 'đu đủ', 
                     'dâu', 'mít', 'măng cụt', 'chôm chôm', 'vải', 'nhãn']
            response_words = response.lower().split()
            
            unique_fruits = set()
            for word in response_words:
                for fruit in fruits:
                    if self.match_word(word, fruit):
                        unique_fruits.add(fruit)
            
            # Score is count (not bounded to max_points for auxiliary)
            score = len(unique_fruits)
            details = {'unique_fruits': list(unique_fruits)}
        
        return {
            'score': min(score, question['max_points']) if question['max_points'] else score,
            'confidence': confidence,
            'error_flag': error_flag,
            'details': details
        }
    
    def score_session(self, session_data: Dict) -> Dict:
        """Score complete session and return structured results."""
        transcript = session_data['transcript']
        confidence = session_data.get('asr_confidence', 0.5)
        
        # For this implementation, we'll assume the transcript contains responses
        # to all questions in sequence. In practice, you'd need to segment the transcript.
        
        # Simplified: split transcript into segments (this would need proper implementation)
        segments = transcript.split('.')  # Very basic segmentation
        
        per_item_scores = {}
        item_confidences = {}
        item_details = {}
        
        for i, question in enumerate(self.questions):
            item_id = question['id']
            
            # Get corresponding segment (simplified)
            if i < len(segments):
                response = segments[i].strip()
            else:
                response = ""
            
            result = self.score_item(item_id, response, confidence)
            per_item_scores[item_id] = result['score']
            item_confidences[item_id] = result['confidence']
            item_details[item_id] = result['details']
        
        # Calculate M_raw (excluding F1)
        M_raw = sum(score for item_id, score in per_item_scores.items() 
                   if item_id != 'F1')
        
        # Placeholder for L and A scalars (will be computed later with features)
        L_scalar = 0.5  # Will be computed in extract_features
        A_scalar = 0.5  # Will be computed in extract_features
        
        return {
            'session_id': session_data['session_id'],
            'per_item_scores': per_item_scores,
            'item_confidences': item_confidences,
            'item_details': item_details,
            'M_raw': M_raw,
            'L_scalar': L_scalar,
            'A_scalar': A_scalar,
            'Score_total_raw': M_raw,  # Will be updated after weight fitting
            'Score_total_rounded': round(M_raw)
        }
