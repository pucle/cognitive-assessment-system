#!/usr/bin/env python3
"""
Complete setup validation script
Validate tất cả components của dual model system
"""

import os
import sys
import json
import subprocess
import requests
import time
from pathlib import Path

def check_files():
    """Check all required files exist."""
    print("📁 Checking files...")

    required_files = [
        "backend/app.py",
        "backend/cognitive_assessment_ml.py",
        "backend/services/mmse_assessment_service.py",
        "backend/services/scoring_engine.py",
        "backend/services/feature_extraction.py",
        "backend/questions.json",
        "release_v1/train_pipeline.py",
        "release_v1/questions.json",
        "frontend/package.json",
        "frontend/app/(main)/menu/page.tsx",
        "frontend/components/mmse-v2-assessment.tsx"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All required files present")
    return True

def check_dependencies():
    """Check Python dependencies."""
    print("\n🐍 Checking Python dependencies...")

    dependencies = [
        "flask",
        "sentence_transformers",
        "cryptography",
        "python_levenshtein",
        "torch",
        "transformers",
        "numpy",
        "pandas"
    ]

    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep.replace("_", ""))
            print(f"✅ {dep}")
        except ImportError:
            missing_deps.append(dep)
            print(f"❌ {dep}")

    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print("Run: pip install " + " ".join(missing_deps))
        return False

    print("✅ All Python dependencies available")
    return True

def check_backend_setup():
    """Check backend environment setup."""
    print("\n🔧 Checking backend setup...")

    backend_dir = Path("backend")

    # Check virtual environment
    venv_paths = [
        backend_dir / ".ok" / "Scripts" / "activate.bat",
        backend_dir / ".venv" / "Scripts" / "activate.bat"
    ]

    venv_found = False
    for venv_path in venv_paths:
        if venv_path.exists():
            print(f"✅ Virtual environment found: {venv_path.parent.parent}")
            venv_found = True
            break

    if not venv_found:
        print("⚠️  No virtual environment found")
        return False

    # Check MMSE service
    try:
        sys.path.insert(0, str(backend_dir))
        from services.mmse_assessment_service import get_mmse_service
        service = get_mmse_service()
        model_info = service.get_model_info()

        print("✅ MMSE service initialized")
        print(f"   Scorer: {model_info['scorer_available']}")
        print(f"   Questions: {len(service.get_questions())}")

        return True

    except Exception as e:
        print(f"❌ MMSE service failed: {e}")
        return False

def check_frontend_setup():
    """Check frontend setup."""
    print("\n⚛️ Checking frontend setup...")

    frontend_dir = Path("frontend")

    if not (frontend_dir / "package.json").exists():
        print("❌ Frontend package.json not found")
        return False

    # Check if node_modules exists
    if (frontend_dir / "node_modules").exists():
        print("✅ Frontend dependencies installed")
    else:
        print("⚠️  Frontend dependencies not installed")
        print("Run: cd frontend && npm install")

    # Check MMSE component
    mmse_component = frontend_dir / "components" / "mmse-v2-assessment.tsx"
    if mmse_component.exists():
        print("✅ MMSE v2.0 component found")
    else:
        print("❌ MMSE v2.0 component missing")
        return False

    return True

def test_api_endpoints():
    """Test API endpoints."""
    print("\n🌐 Testing API endpoints...")

    endpoints = [
        ("GET", "http://localhost:5001/api/health", "Old model health"),
        ("GET", "http://localhost:5001/api/mmse/model-info", "MMSE model info"),
        ("GET", "http://localhost:5001/api/mmse/questions", "MMSE questions")
    ]

    working_endpoints = 0

    for method, url, description in endpoints:
        try:
            if method == "GET":
                response = requests.get(url, timeout=5)
            elif method == "POST":
                response = requests.post(url, timeout=5)

            if response.status_code == 200:
                print(f"✅ {description}: {url}")
                working_endpoints += 1
            else:
                print(f"❌ {description}: HTTP {response.status_code}")

        except Exception as e:
            print(f"❌ {description}: {str(e)}")

    if working_endpoints == len(endpoints):
        print("✅ All API endpoints working")
        return True
    else:
        print(f"⚠️  {working_endpoints}/{len(endpoints)} endpoints working")
        return False

def check_models():
    """Check model files."""
    print("\n🤖 Checking models...")

    model_files = [
        "release_v1/model_MMSE_v1.pkl",
        "release_v1/model_metadata.json",
        "backend/demo_model/feature_names.pkl"
    ]

    found_models = 0
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ {model_file}")
            found_models += 1
        else:
            print(f"⚠️  {model_file} not found")

    if found_models >= 2:
        print("✅ Sufficient models available")
        return True
    else:
        print("⚠️  Some models missing - may affect functionality")
        return True  # Not critical failure

def generate_report(results):
    """Generate validation report."""
    print("\n" + "="*60)
    print("📊 VALIDATION REPORT")
    print("="*60)

    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")

    successful_components = sum(results.values())
    total_components = len(results)

    print(f"\nOverall: {successful_components}/{total_components} components validated")

    if successful_components == total_components:
        print("\n🎉 ALL COMPONENTS VALIDATED SUCCESSFULLY!")
        print("\n🚀 You can now start the system:")
        print("   python start_complete_system.py")
        print("   # or")
        print("   run_dual_model_system.bat")

    elif successful_components >= total_components - 1:
        print("\n⚠️  Minor issues detected but system should work")
        print("Run the startup script to begin")

    else:
        print("\n❌ Critical issues detected")
        print("Please fix the issues above before proceeding")

    return successful_components == total_components

def main():
    """Main validation function."""
    print("🔍 COGNITIVE ASSESSMENT SYSTEM VALIDATION")
    print("="*60)
    print("This script validates your complete dual model setup")
    print("="*60)

    # Run all checks
    checks = [
        ("Files", check_files),
        ("Dependencies", check_dependencies),
        ("Backend Setup", check_backend_setup),
        ("Frontend Setup", check_frontend_setup),
        ("Models", check_models)
    ]

    results = {}

    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name.upper()} {'='*20}")
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results[check_name] = False

    # Test API if backend seems ready
    if results.get("Backend Setup", False):
        print(f"\n{'='*20} API ENDPOINTS {'='*20}")
        try:
            results["API Endpoints"] = test_api_endpoints()
        except Exception as e:
            print(f"❌ API test failed: {e}")
            results["API Endpoints"] = False

    # Generate final report
    success = generate_report(results)

    print("\n" + "="*60)
    print("💡 NEXT STEPS:")
    print("="*60)

    if success:
        print("1. Start the system:")
        print("   python start_complete_system.py")
        print("   # or")
        print("   run_dual_model_system.bat")
        print("")
        print("2. Access the application:")
        print("   Main app: http://localhost:3000")
        print("   MMSE v2:  http://localhost:3000/mmse-v2")
        print("")
        print("3. Test the APIs:")
        print("   curl http://localhost:5001/api/health")
        print("   curl http://localhost:5001/api/mmse/model-info")
    else:
        print("1. Fix the issues shown above")
        print("2. Re-run validation: python validate_setup.py")
        print("3. Check detailed logs for errors")
        print("4. Refer to README_DUAL_MODEL.md for troubleshooting")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
