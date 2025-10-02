#!/usr/bin/env python3
"""
Deployment Testing Script for Cognitive Assessment System
Tests all components after deployment to ensure everything works
"""

import os
import sys
import requests
import time
import json
from typing import Dict, Any, Tuple
import subprocess

class DeploymentTester:
    """Comprehensive deployment tester"""

    def __init__(self):
        self.backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
        self.frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        self.database_url = os.getenv('DATABASE_URL')
        self.results = {}

    def log(self, message: str, status: str = 'INFO'):
        """Log with timestamp"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {status}: {message}")

    def test_endpoint(self, url: str, method: str = 'GET',
                     expected_status: int = 200,
                     timeout: int = 30,
                     json_data: Dict = None) -> Tuple[bool, Dict[str, Any]]:
        """Test an HTTP endpoint"""
        try:
            self.log(f"Testing {method} {url}")

            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=json_data, timeout=timeout)
            else:
                return False, {'error': f'Unsupported method: {method}'}

            if response.status_code == expected_status:
                try:
                    data = response.json()
                    self.log(f"✅ {url} - Status: {response.status_code}")
                    return True, data
                except:
                    self.log(f"✅ {url} - Status: {response.status_code} (non-JSON response)")
                    return True, {'text': response.text}
            else:
                self.log(f"❌ {url} - Status: {response.status_code}")
                return False, {
                    'status_code': response.status_code,
                    'response': response.text[:500]
                }

        except requests.exceptions.RequestException as e:
            self.log(f"❌ {url} - Error: {str(e)}", 'ERROR')
            return False, {'error': str(e)}

    def test_backend_health(self) -> bool:
        """Test backend health endpoints"""
        self.log("Testing backend health endpoints...")

        endpoints = [
            (f"{self.backend_url}/api/health", 200),
            (f"{self.backend_url}/api/health/live", 200),
            (f"{self.backend_url}/api/health/ready", 200),
        ]

        all_passed = True
        for url, expected_status in endpoints:
            success, _ = self.test_endpoint(url, expected_status=expected_status)
            if not success:
                all_passed = False

        self.results['backend_health'] = all_passed
        return all_passed

    def test_backend_api(self) -> bool:
        """Test backend API functionality"""
        self.log("Testing backend API functionality...")

        endpoints = [
            (f"{self.backend_url}/api/mmse/questions", 200),
            (f"{self.backend_url}/api/status", 200),
        ]

        all_passed = True
        for url, expected_status in endpoints:
            success, response = self.test_endpoint(url, expected_status=expected_status)
            if not success:
                all_passed = False
            elif 'questions' in url and success:
                # Validate questions response structure
                if not isinstance(response.get('data', {}).get('questions'), list):
                    self.log("❌ Questions endpoint returned invalid structure", 'ERROR')
                    all_passed = False

        self.results['backend_api'] = all_passed
        return all_passed

    def test_frontend_accessibility(self) -> bool:
        """Test frontend accessibility"""
        self.log("Testing frontend accessibility...")

        try:
            response = requests.get(self.frontend_url, timeout=30)

            if response.status_code == 200:
                self.log("✅ Frontend homepage accessible")
                self.results['frontend_accessibility'] = True
                return True
            else:
                self.log(f"❌ Frontend homepage returned status {response.status_code}", 'ERROR')
                self.results['frontend_accessibility'] = False
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"❌ Frontend accessibility test failed: {str(e)}", 'ERROR')
            self.results['frontend_accessibility'] = False
            return False

    def test_database_connection(self) -> bool:
        """Test database connection"""
        self.log("Testing database connection...")

        if not self.database_url:
            self.log("⚠️ DATABASE_URL not set, skipping database tests")
            self.results['database_connection'] = None
            return True

        try:
            import psycopg2

            conn = psycopg2.connect(self.database_url)
            conn.close()

            self.log("✅ Database connection successful")
            self.results['database_connection'] = True
            return True

        except ImportError:
            self.log("⚠️ psycopg2 not available, skipping database tests")
            self.results['database_connection'] = None
            return True
        except Exception as e:
            self.log(f"❌ Database connection failed: {str(e)}", 'ERROR')
            self.results['database_connection'] = False
            return False

    def test_environment_variables(self) -> bool:
        """Test that required environment variables are set"""
        self.log("Testing environment variables...")

        required_vars = [
            'OPENAI_API_KEY',
            'GEMINI_API_KEY',
            'SECRET_KEY'
        ]

        optional_vars = [
            'DATABASE_URL',
            'BLOB_READ_WRITE_TOKEN',
            'SENTRY_DSN'
        ]

        missing_required = []
        missing_optional = []

        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)

        for var in optional_vars:
            if not os.getenv(var):
                missing_optional.append(var)

        if missing_required:
            self.log(f"❌ Missing required environment variables: {', '.join(missing_required)}", 'ERROR')
            self.results['environment_variables'] = False
            return False

        if missing_optional:
            self.log(f"⚠️ Missing optional environment variables: {', '.join(missing_optional)}")

        self.log("✅ Required environment variables are set")
        self.results['environment_variables'] = True
        return True

    def test_external_services(self) -> bool:
        """Test external service connectivity"""
        self.log("Testing external services...")

        services_ok = True

        # Test OpenAI API
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                client.models.list()
                self.log("✅ OpenAI API accessible")
            except Exception as e:
                self.log(f"❌ OpenAI API test failed: {str(e)}", 'ERROR')
                services_ok = False
        else:
            self.log("⚠️ OPENAI_API_KEY not set")

        # Test Gemini API
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                genai.list_models()
                self.log("✅ Gemini API accessible")
            except Exception as e:
                self.log(f"❌ Gemini API test failed: {str(e)}", 'ERROR')
                services_ok = False
        else:
            self.log("⚠️ GEMINI_API_KEY not set")

        self.results['external_services'] = services_ok
        return services_ok

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests"""
        self.log("🚀 Starting comprehensive deployment tests")
        self.log("=" * 60)

        # Test configuration
        config_ok = self.test_environment_variables()

        if not config_ok:
            self.log("❌ Configuration tests failed. Aborting other tests.", 'ERROR')
            return self.generate_report()

        # Test infrastructure
        database_ok = self.test_database_connection()

        # Test backend
        backend_health_ok = self.test_backend_health()
        backend_api_ok = self.test_backend_api()

        # Test frontend
        frontend_ok = self.test_frontend_accessibility()

        # Test external services
        services_ok = self.test_external_services()

        # Generate final report
        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': {
                'backend_url': self.backend_url,
                'frontend_url': self.frontend_url,
                'database_configured': bool(self.database_url),
            },
            'test_results': self.results,
            'summary': {}
        }

        # Calculate summary
        total_tests = len([r for r in self.results.values() if r is not None])
        passed_tests = len([r for r in self.results.values() if r is True])
        skipped_tests = len([r for r in self.results.values() if r is None])

        report['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests - skipped_tests,
            'skipped_tests': skipped_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }

        # Overall status
        if report['summary']['success_rate'] >= 90:
            report['status'] = '✅ EXCELLENT'
        elif report['summary']['success_rate'] >= 75:
            report['status'] = '✅ GOOD'
        elif report['summary']['success_rate'] >= 50:
            report['status'] = '⚠️ FAIR'
        else:
            report['status'] = '❌ POOR'

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print formatted test report"""
        print("\n" + "="*60)
        print("📊 DEPLOYMENT TEST REPORT")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Status: {report['status']}")
        print()

        print("🌐 Environment:")
        print(f"  Backend: {report['environment']['backend_url']}")
        print(f"  Frontend: {report['environment']['frontend_url']}")
        print(f"  Database: {'✅ Configured' if report['environment']['database_configured'] else '❌ Not configured'}")
        print()

        print("🧪 Test Results:")
        for test_name, result in report['test_results'].items():
            if result is True:
                status = "✅ PASSED"
            elif result is False:
                status = "❌ FAILED"
            else:
                status = "⚠️ SKIPPED"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
        print()

        summary = report['summary']
        print("📈 Summary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Skipped: {summary['skipped_tests']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print()

        if summary['success_rate'] >= 75:
            print("🎉 Deployment tests completed successfully!")
            print("Your Cognitive Assessment System is ready for production.")
        else:
            print("⚠️ Some tests failed. Please review the issues above.")
            print("Check the RUNBOOK.md for troubleshooting guidance.")

        print("="*60)

def main():
    """Main function"""
    print("Cognitive Assessment System - Deployment Tester")
    print("==============================================")

    # Allow overriding URLs via command line
    if len(sys.argv) > 1:
        os.environ['BACKEND_URL'] = sys.argv[1]
    if len(sys.argv) > 2:
        os.environ['FRONTEND_URL'] = sys.argv[2]

    tester = DeploymentTester()
    report = tester.run_comprehensive_test()
    tester.print_report(report)

    # Exit with appropriate code
    success_rate = report['summary']['success_rate']
    if success_rate >= 75:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()
