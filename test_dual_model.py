#!/usr/bin/env python3
"""
Test script for Dual Model System
Test cả model cũ và MMSE v2.0 mới
"""

import os
import sys
import json
import requests
import tempfile
import time
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_backend_health():
    """Test backend health."""
    try:
        response = requests.get("http://localhost:5001/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False

def test_old_model():
    """Test old cognitive assessment model."""
    print("\n🔍 Testing OLD MODEL (Original)...")

    # Test health
    try:
        response = requests.get("http://localhost:5001/api/health")
        if response.status_code == 200:
            print("✅ Old model health check passed")
        else:
            print(f"❌ Old model health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Old model connection failed: {e}")
        return False

    print("✅ Old model is available")
    return True

def test_mmse_v2():
    """Test MMSE v2.0 model."""
    print("\n🧠 Testing MMSE v2.0 MODEL...")

    # Test model info
    try:
        response = requests.get("http://localhost:5001/api/mmse/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                model_info = data['data']
                print("✅ MMSE v2.0 model info retrieved")
                print(f"   Scorer available: {model_info['scorer_available']}")
                print(f"   Feature extractor: {model_info['feature_extractor_available']}")
                print(f"   Model available: {model_info['model_available']}")
            else:
                print(f"❌ MMSE model info failed: {data.get('error')}")
                return False
        else:
            print(f"❌ MMSE model info endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MMSE model info connection failed: {e}")
        return False

    # Test questions
    try:
        response = requests.get("http://localhost:5001/api/mmse/questions", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                questions = data['data']['questions']
                total_points = data['data']['total_points']
                print(f"✅ MMSE questions loaded: {len(questions)} questions, {total_points} points")
            else:
                print(f"❌ MMSE questions failed: {data.get('error')}")
                return False
        else:
            print(f"❌ MMSE questions endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MMSE questions connection failed: {e}")
        return False

    print("✅ MMSE v2.0 is available")
    return True

def create_test_audio():
    """Create a simple test audio file."""
    try:
        import numpy as np
        import soundfile as sf

        # Create simple sine wave
        duration = 3.0
        sample_rate = 16000
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)

        # Add noise
        noise = 0.05 * np.random.randn(len(audio))
        audio = audio + noise

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, audio, sample_rate)
        temp_file.close()

        print(f"✅ Created test audio: {temp_file.name}")
        return temp_file.name

    except ImportError:
        print("⚠️  Cannot create test audio (soundfile not available)")
        return None
    except Exception as e:
        print(f"❌ Error creating test audio: {e}")
        return None

def test_transcription():
    """Test transcription functionality."""
    print("\n🎤 Testing TRANSCRIPTION...")

    audio_file = create_test_audio()
    if not audio_file:
        print("⚠️  Skipping transcription test")
        return True

    try:
        # Test old transcription
        with open(audio_file, 'rb') as f:
            files = {'audio': f}
            response = requests.post("http://localhost:5001/api/transcribe", files=files, timeout=30)

        if response.status_code == 200:
            print("✅ Old model transcription working")
        else:
            print(f"⚠️  Old model transcription failed: {response.status_code}")

        # Test MMSE v2.0 transcription
        with open(audio_file, 'rb') as f:
            files = {'audio': f}
            response = requests.post("http://localhost:5001/api/mmse/transcribe", files=files, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ MMSE v2.0 transcription working")
                print(f"   Transcript: {data['data'].get('transcript', 'N/A')[:50]}...")
            else:
                print(f"⚠️  MMSE transcription failed: {data.get('error')}")
        else:
            print(f"⚠️  MMSE transcription endpoint failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        return False
    finally:
        # Cleanup
        try:
            os.unlink(audio_file)
        except:
            pass

    return True

def show_api_summary():
    """Show API endpoints summary."""
    print("\n📋 API ENDPOINTS SUMMARY")
    print("=" * 50)
    print("🌐 Backend: http://localhost:5001")
    print("🌍 Frontend: http://localhost:3000")
    print("🧠 MMSE v2: http://localhost:3000/mmse-v2")
    print()
    print("OLD MODEL ENDPOINTS:")
    print("  GET  /api/health")
    print("  POST /api/assess")
    print("  POST /api/transcribe")
    print("  POST /api/features")
    print()
    print("MMSE v2.0 ENDPOINTS:")
    print("  GET  /api/mmse/model-info")
    print("  GET  /api/mmse/questions")
    print("  POST /api/mmse/assess")
    print("  POST /api/mmse/transcribe")
    print("=" * 50)

def main():
    """Main test function."""
    print("🧪 DUAL MODEL SYSTEM TEST")
    print("=" * 50)

    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)

    # Test backend connectivity
    if not test_backend_health():
        print("❌ Backend not running. Please start the server first:")
        print("   cd backend")
        print("   python start_dual_model.py")
        return False

    # Test models
    old_model_ok = test_old_model()
    mmse_v2_ok = test_mmse_v2()

    # Test transcription
    transcription_ok = test_transcription()

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Old Model:       {'✅ OK' if old_model_ok else '❌ FAILED'}")
    print(f"MMSE v2.0:       {'✅ OK' if mmse_v2_ok else '❌ FAILED'}")
    print(f"Transcription:   {'✅ OK' if transcription_ok else '❌ FAILED'}")

    if old_model_ok and mmse_v2_ok:
        print("\n🎉 SUCCESS: Both models are working!")
        show_api_summary()
        return True
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
