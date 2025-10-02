#!/usr/bin/env python3
"""
Test script for MMSE v2.0 integration
Tests the complete pipeline from backend service to frontend integration.
"""

import os
import sys
import requests
import json
import tempfile
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BACKEND_BASE_URL = "http://localhost:5001"
FRONTEND_BASE_URL = "http://localhost:3000"

def test_backend_health():
    """Test if backend is running."""
    try:
        response = requests.get(f"{BACKEND_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Backend health check passed")
            return True
        else:
            logger.error(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Backend not accessible: {e}")
        return False

def test_mmse_model_info():
    """Test MMSE model info endpoint."""
    try:
        response = requests.get(f"{BACKEND_BASE_URL}/api/mmse/model-info", timeout=10)
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            model_info = data['data']
            logger.info("✅ MMSE model info retrieved successfully")
            logger.info(f"   Model available: {model_info.get('model_available')}")
            logger.info(f"   Scorer available: {model_info.get('scorer_available')}")
            logger.info(f"   Feature extractor available: {model_info.get('feature_extractor_available')}")
            logger.info(f"   Model version: {model_info.get('model_version')}")
            return True
        else:
            logger.error(f"❌ MMSE model info failed: {data.get('error')}")
            return False
    except Exception as e:
        logger.error(f"❌ MMSE model info error: {e}")
        return False

def test_mmse_questions():
    """Test MMSE questions endpoint."""
    try:
        response = requests.get(f"{BACKEND_BASE_URL}/api/mmse/questions", timeout=10)
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            questions = data['data']['questions']
            total_points = data['data']['total_points']
            logger.info(f"✅ MMSE questions retrieved: {len(questions)} questions, {total_points} total points")
            
            # Validate questions structure
            required_fields = ['id', 'domain', 'question_text', 'max_points']
            for q in questions[:3]:  # Check first 3 questions
                if all(field in q for field in required_fields):
                    logger.info(f"   ✓ Question {q['id']}: {q['domain']} ({q['max_points']} pts)")
                else:
                    logger.warning(f"   ⚠ Question {q.get('id', 'unknown')} missing required fields")
            
            return total_points == 30
        else:
            logger.error(f"❌ MMSE questions failed: {data.get('error')}")
            return False
    except Exception as e:
        logger.error(f"❌ MMSE questions error: {e}")
        return False

def create_test_audio():
    """Create a simple test audio file."""
    try:
        import soundfile as sf
        
        # Create a simple synthetic audio (sine wave)
        duration = 5.0  # seconds
        sample_rate = 16000
        frequency = 440.0  # A note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise to make it more realistic
        noise = 0.05 * np.random.randn(len(audio))
        audio = audio + noise
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, audio, sample_rate)
        
        logger.info(f"✅ Created test audio file: {temp_file.name}")
        return temp_file.name
        
    except ImportError:
        logger.warning("⚠ soundfile not available, using dummy file")
        # Create a dummy file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.write(b'dummy audio data')
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logger.error(f"❌ Error creating test audio: {e}")
        return None

def test_mmse_transcription():
    """Test MMSE transcription endpoint."""
    audio_file = create_test_audio()
    if not audio_file:
        return False
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'audio': f}
            response = requests.post(
                f"{BACKEND_BASE_URL}/api/mmse/transcribe",
                files=files,
                timeout=30
            )
        
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            transcription = data['data']
            logger.info("✅ MMSE transcription successful")
            logger.info(f"   Transcript: {transcription.get('transcript', 'N/A')[:100]}...")
            logger.info(f"   Confidence: {transcription.get('confidence', 0):.2f}")
            logger.info(f"   Language: {transcription.get('language', 'N/A')}")
            return True
        else:
            logger.error(f"❌ MMSE transcription failed: {data.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ MMSE transcription error: {e}")
        return False
    finally:
        # Clean up
        try:
            os.unlink(audio_file)
        except:
            pass

def test_mmse_assessment():
    """Test full MMSE assessment endpoint."""
    audio_file = create_test_audio()
    if not audio_file:
        return False
    
    try:
        # Prepare test data
        patient_info = {
            "name": "Test Patient",
            "age": 65,
            "gender": "female",
            "education_years": 12,
            "notes": "Test assessment"
        }
        
        with open(audio_file, 'rb') as f:
            files = {'audio': f}
            data = {
                'session_id': 'test_session_001',
                'patient_info': json.dumps(patient_info)
            }
            response = requests.post(
                f"{BACKEND_BASE_URL}/api/mmse/assess",
                files=files,
                data=data,
                timeout=60  # Longer timeout for full assessment
            )
        
        result = response.json()
        
        if response.status_code == 200 and result.get('success'):
            assessment = result['data']
            logger.info("✅ MMSE assessment successful")
            logger.info(f"   Session ID: {assessment.get('session_id')}")
            logger.info(f"   Status: {assessment.get('status')}")
            
            if 'mmse_scores' in assessment:
                scores = assessment['mmse_scores']
                logger.info(f"   Final Score: {scores.get('final_score', 'N/A')}/30")
                logger.info(f"   Raw Score: {scores.get('M_raw', 'N/A')}/30")
                logger.info(f"   L-Scalar: {scores.get('L_scalar', 'N/A'):.3f}")
                logger.info(f"   A-Scalar: {scores.get('A_scalar', 'N/A'):.3f}")
            
            if 'cognitive_status' in assessment:
                status = assessment['cognitive_status']
                logger.info(f"   Cognitive Status: {status.get('status', 'N/A')}")
                logger.info(f"   Risk Level: {status.get('risk_level', 'N/A')}")
                logger.info(f"   Confidence: {status.get('confidence', 0):.1%}")
            
            return True
        else:
            logger.error(f"❌ MMSE assessment failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ MMSE assessment error: {e}")
        return False
    finally:
        # Clean up
        try:
            os.unlink(audio_file)
        except:
            pass

def test_frontend_accessibility():
    """Test if frontend is accessible."""
    try:
        response = requests.get(f"{FRONTEND_BASE_URL}", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Frontend accessible")
            return True
        else:
            logger.error(f"❌ Frontend not accessible: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Frontend not accessible: {e}")
        return False

def test_mmse_v2_page():
    """Test if MMSE v2 page is accessible."""
    try:
        response = requests.get(f"{FRONTEND_BASE_URL}/mmse-v2", timeout=10)
        if response.status_code == 200:
            logger.info("✅ MMSE v2 page accessible")
            return True
        else:
            logger.error(f"❌ MMSE v2 page not accessible: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ MMSE v2 page error: {e}")
        return False

def run_all_tests():
    """Run all integration tests."""
    logger.info("🚀 Starting MMSE v2.0 Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Backend Health", test_backend_health),
        ("MMSE Model Info", test_mmse_model_info),
        ("MMSE Questions", test_mmse_questions),
        ("MMSE Transcription", test_mmse_transcription),
        ("MMSE Assessment", test_mmse_assessment),
        ("Frontend Accessibility", test_frontend_accessibility),
        ("MMSE v2 Page", test_mmse_v2_page),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! MMSE v2.0 integration is working correctly.")
    else:
        logger.warning(f"⚠ {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
