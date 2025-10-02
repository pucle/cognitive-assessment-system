#!/usr/bin/env python3
"""
Complete System Startup Script
Khởi động toàn bộ hệ thống dual model một cách tự động
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def run_command(cmd, cwd=None, shell=True):
    """Run command and return success."""
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def setup_backend():
    """Setup backend environment."""
    print("🔧 Setting up backend...")

    backend_dir = Path("backend")

    # Check if .ok environment exists
    ok_env = backend_dir / ".ok" / "Scripts" / "activate.bat"
    if ok_env.exists():
        print("✅ Found existing .ok environment")
        return True

    # Create new environment if needed
    print("📦 Creating new virtual environment...")
    success, stdout, stderr = run_command("python -m venv .ok", cwd=backend_dir)
    if not success:
        print(f"❌ Failed to create environment: {stderr}")
        return False

    print("✅ Virtual environment created")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📥 Installing dependencies...")

    backend_dir = Path("backend")

    # Activate environment and install
    activate_cmd = ".ok\\Scripts\\activate && "
    install_cmd = f"{activate_cmd} pip install python-Levenshtein sentence-transformers fpdf2 cryptography"

    success, stdout, stderr = run_command(install_cmd, cwd=backend_dir)
    if not success:
        print(f"❌ Failed to install dependencies: {stderr}")
        return False

    print("✅ Dependencies installed")
    return True

def copy_mmse_files():
    """Copy MMSE v2.0 files to backend."""
    print("📋 Copying MMSE files...")

    files_to_copy = [
        ("release_v1/questions.json", "backend/questions.json"),
        ("release_v1/scoring_engine.py", "backend/services/scoring_engine.py"),
        ("release_v1/feature_extraction.py", "backend/services/feature_extraction.py"),
        ("release_v1/encryption.py", "backend/services/encryption.py"),
        ("release_v1/mmse_assessment_service.py", "backend/services/mmse_assessment_service.py")
    ]

    for src, dst in files_to_copy:
        if os.path.exists(src):
            # Create destination directory if needed
            dst_dir = os.path.dirname(dst)
            os.makedirs(dst_dir, exist_ok=True)

            # Copy file
            with open(src, 'rb') as fsrc:
                with open(dst, 'wb') as fdst:
                    fdst.write(fsrc.read())
            print(f"✅ Copied {src} -> {dst}")
        else:
            print(f"⚠️  Source file not found: {src}")

    return True

def test_setup():
    """Test if setup is complete."""
    print("🧪 Testing setup...")

    backend_dir = Path("backend")

    # Test with activated environment
    test_cmd = ".ok\\Scripts\\activate && python quick_test.py"
    success, stdout, stderr = run_command(test_cmd, cwd=backend_dir)

    if success:
        print("✅ Setup test passed")
        print(stdout)
        return True
    else:
        print("❌ Setup test failed")
        print(stderr)
        return False

def start_backend():
    """Start backend server."""
    print("🚀 Starting backend server...")

    backend_dir = Path("backend")

    # Start backend in background
    start_cmd = ".ok\\Scripts\\activate && python start_dual_model.py"

    try:
        process = subprocess.Popen(
            start_cmd,
            shell=True,
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait a bit for server to start
        time.sleep(5)

        # Check if process is still running
        if process.poll() is None:
            print("✅ Backend server started successfully")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Backend server failed to start")
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            return False

    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False

def start_frontend():
    """Start frontend development server."""
    print("🌐 Starting frontend server...")

    frontend_dir = Path("frontend")

    # Start frontend in background
    try:
        process = subprocess.Popen(
            "npm run dev",
            shell=True,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for frontend to start
        time.sleep(10)

        # Check if process is still running
        if process.poll() is None:
            print("✅ Frontend server started successfully")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Frontend server failed to start")
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            return False

    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return False

def open_browser():
    """Open browser to the application."""
    print("🌍 Opening browser...")

    try:
        # Try to open main app first
        webbrowser.open("http://localhost:3000")
        time.sleep(2)

        # Then open MMSE v2.0 page
        webbrowser.open("http://localhost:3000/mmse-v2")
        print("✅ Browser opened")
        return True
    except Exception as e:
        print(f"⚠️  Could not open browser: {e}")
        return False

def main():
    """Main startup function."""
    print("🎯 COGNITIVE ASSESSMENT DUAL MODEL SYSTEM")
    print("=" * 50)
    print("This script will:")
    print("1. Setup backend environment")
    print("2. Install dependencies")
    print("3. Copy MMSE v2.0 files")
    print("4. Test setup")
    print("5. Start backend server")
    print("6. Start frontend server")
    print("7. Open browser")
    print("=" * 50)

    # Confirm
    response = input("\nDo you want to continue? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Setup cancelled.")
        return

    # Setup steps
    steps = [
        ("Setup backend environment", setup_backend),
        ("Install dependencies", install_dependencies),
        ("Copy MMSE files", copy_mmse_files),
        ("Test setup", test_setup),
        ("Start backend server", start_backend),
        ("Start frontend server", start_frontend),
        ("Open browser", open_browser)
    ]

    success_count = 0
    for step_name, step_func in steps:
        print(f"\n[STEP] {step_name}...")
        if step_func():
            success_count += 1
        else:
            print(f"❌ {step_name} failed")
            break

    # Summary
    print("\n" + "=" * 50)
    print("📊 STARTUP SUMMARY")
    print("=" * 50)
    print(f"Completed steps: {success_count}/{len(steps)}")

    if success_count >= 5:  # At least backend, frontend, and browser
        print("\n🎉 SYSTEM STARTUP SUCCESSFUL!")
        print("\n🌐 Available URLs:")
        print("   Main App:     http://localhost:3000")
        print("   MMSE v2.0:    http://localhost:3000/mmse-v2")
        print("   Backend API:  http://localhost:5001")
        print("\n📊 API Endpoints:")
        print("   Old Model:    POST /api/assess")
        print("   MMSE v2.0:    POST /api/mmse/assess")
        print("\n🧪 Test commands:")
        print("   curl http://localhost:5001/api/health")
        print("   curl http://localhost:5001/api/mmse/model-info")
        print("\n✅ You can now use both models!")

    else:
        print(f"\n⚠️  Only {success_count} steps completed.")
        print("Please check the errors above and try again.")
        print("\nManual startup commands:")
        print("cd backend && .ok\\Scripts\\activate && python start_dual_model.py")
        print("cd frontend && npm run dev")

if __name__ == "__main__":
    main()
