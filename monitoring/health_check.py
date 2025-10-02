# Health Check Module for Cognitive Assessment System
# Comprehensive health monitoring for production deployments

import os
import time
import psutil
import logging
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
import requests
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checker for the application"""

    def __init__(self):
        self.start_time = time.time()
        self.last_health_check = None
        self.health_check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))

    def get_system_health(self) -> Dict[str, Any]:
        """Get system-level health metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
                'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'disk_free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None,
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {'error': str(e)}

    def check_database_connection(self) -> Tuple[bool, str]:
        """Check database connection health"""
        try:
            import psycopg2

            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return False, "DATABASE_URL not configured"

            # Parse connection string
            conn = psycopg2.connect(database_url)

            # Test query
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 as health_check")
                result = cursor.fetchone()

            conn.close()

            if result and result[0] == 1:
                return True, "Database connection healthy"
            else:
                return False, "Database query failed"

        except ImportError:
            return False, "psycopg2 not available"
        except Exception as e:
            return False, f"Database connection error: {str(e)}"

    def check_ml_models(self) -> Tuple[bool, str]:
        """Check ML model loading status"""
        try:
            # Import ML dependencies (handle gracefully)
            try:
                import torch
                import joblib
                import os
            except ImportError as e:
                return False, f"ML dependencies not available: {e}"

            # Check model files exist
            model_dir = os.getenv('MODEL_PATH', './models')
            required_files = ['model.pkl', 'scaler.pkl']

            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(model_dir, file)):
                    missing_files.append(file)

            if missing_files:
                return False, f"Missing model files: {', '.join(missing_files)}"

            # Try loading models (quick test)
            try:
                model = joblib.load(os.path.join(model_dir, 'model.pkl'))
                scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

                # Quick inference test
                test_input = [[65, 1000, 500, 10, 150]]  # Sample features
                scaled_input = scaler.transform(test_input)
                prediction = model.predict(scaled_input)

                if len(prediction) == 1:
                    return True, "ML models loaded and functional"
                else:
                    return False, "ML model prediction failed"

            except Exception as e:
                return False, f"Model loading error: {str(e)}"

        except Exception as e:
            return False, f"ML health check error: {str(e)}"

    def check_external_services(self) -> Dict[str, Tuple[bool, str]]:
        """Check external service dependencies"""
        services = {}

        # OpenAI API
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                # Simple API test (doesn't cost tokens)
                client.models.list()
                services['openai'] = (True, "OpenAI API accessible")
            except Exception as e:
                services['openai'] = (False, f"OpenAI API error: {str(e)}")
        else:
            services['openai'] = (False, "OpenAI API key not configured")

        # Google Gemini API
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                # List models (free operation)
                genai.list_models()
                services['gemini'] = (True, "Gemini API accessible")
            except Exception as e:
                services['gemini'] = (False, f"Gemini API error: {str(e)}")
        else:
            services['gemini'] = (False, "Gemini API key not configured")

        # Vercel Blob (if configured)
        blob_token = os.getenv('BLOB_READ_WRITE_TOKEN')
        if blob_token:
            try:
                from vercel.blob import list
                # Test blob access
                blobs = list()
                services['vercel_blob'] = (True, "Vercel Blob accessible")
            except Exception as e:
                services['vercel_blob'] = (False, f"Vercel Blob error: {str(e)}")

        return services

    def check_disk_space(self) -> Tuple[bool, str]:
        """Check available disk space"""
        try:
            stat = os.statvfs('/')
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            min_free_gb = float(os.getenv('MIN_DISK_FREE_GB', '1.0'))

            if free_gb < min_free_gb:
                return False, f"Low disk space: {free_gb:.1f}GB free (minimum: {min_free_gb}GB)"
            else:
                return True, f"Disk space OK: {free_gb:.1f}GB free"

        except Exception as e:
            return False, f"Disk space check error: {str(e)}"

    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics"""
        try:
            metrics = {
                'uptime_seconds': time.time() - self.start_time,
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                'environment': os.getenv('FLASK_ENV', 'unknown'),
                'process_id': os.getpid(),
                'thread_count': len(psutil.Process().threads()),
            }

            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics['memory_rss_mb'] = memory_info.rss / 1024 / 1024
            metrics['memory_vms_mb'] = memory_info.vms / 1024 / 1024

            return metrics

        except Exception as e:
            logger.error(f"Error getting application metrics: {e}")
            return {'error': str(e)}

    def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        self.last_health_check = datetime.now()

        health_data = {
            'timestamp': self.last_health_check.isoformat(),
            'status': 'healthy',
            'checks': {},
            'system': self.get_system_health(),
            'application': self.get_application_metrics(),
        }

        # Individual health checks
        checks = {
            'database': self.check_database_connection(),
            'ml_models': self.check_ml_models(),
            'disk_space': self.check_disk_space(),
        }

        # External services
        external_checks = self.check_external_services()
        checks.update(external_checks)

        # Evaluate overall health
        failed_checks = []
        warning_checks = []

        for check_name, (is_healthy, message) in checks.items():
            health_data['checks'][check_name] = {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'message': message
            }

            if not is_healthy:
                failed_checks.append(check_name)

                # Critical failures
                if check_name in ['database', 'ml_models']:
                    health_data['status'] = 'unhealthy'
                else:
                    # Non-critical failures -> warning status
                    if health_data['status'] == 'healthy':
                        health_data['status'] = 'warning'

        # Performance warnings
        system = health_data['system']
        if system.get('memory_percent', 0) > 85:
            health_data['status'] = 'warning'
            warning_checks.append('high_memory_usage')

        if system.get('cpu_percent', 0) > 90:
            health_data['status'] = 'warning'
            warning_checks.append('high_cpu_usage')

        # Summary
        health_data['summary'] = {
            'total_checks': len(checks),
            'failed_checks': len(failed_checks),
            'warning_checks': len(warning_checks),
            'failed_check_names': failed_checks,
            'warning_check_names': warning_checks,
        }

        logger.info(f"Health check completed: {health_data['status']} "
                   f"({len(failed_checks)} failed, {len(warning_checks)} warnings)")

        return health_data

# Flask health check endpoint
def create_health_endpoint(app):
    """Create health check endpoint for Flask app"""

    @app.route('/api/health')
    def health_check():
        """Comprehensive health check endpoint"""
        try:
            checker = HealthChecker()
            health_data = checker.comprehensive_health_check()

            # Return appropriate HTTP status
            status_code = 200
            if health_data['status'] == 'unhealthy':
                status_code = 503  # Service Unavailable
            elif health_data['status'] == 'warning':
                status_code = 200  # OK but with warnings

            return health_data, status_code

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }, 500

    @app.route('/api/health/live')
    def liveness_check():
        """Simple liveness probe"""
        return {'status': 'alive', 'timestamp': datetime.now().isoformat()}

    @app.route('/api/health/ready')
    def readiness_check():
        """Readiness probe with basic checks"""
        try:
            # Quick database check
            db_healthy, db_message = HealthChecker().check_database_connection()

            status = 'ready' if db_healthy else 'not ready'
            status_code = 200 if db_healthy else 503

            return {
                'status': status,
                'database': db_message,
                'timestamp': datetime.now().isoformat()
            }, status_code

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 503

# Export for use in main application
__all__ = ['HealthChecker', 'create_health_endpoint']
