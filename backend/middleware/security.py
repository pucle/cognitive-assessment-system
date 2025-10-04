# Security Middleware for Cognitive Assessment Backend
# Production-grade security features

from flask import Flask, request, abort, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import re
import time
import logging
from functools import wraps
from typing import Dict, Any, Optional
import hashlib
import hmac

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Comprehensive security middleware for Flask application"""

    def __init__(self, app: Flask):
        self.app = app
        self.setup_cors()
        self.setup_rate_limiting()
        self.setup_security_headers()
        self.setup_request_validation()
        self.setup_logging()

    def setup_cors(self):
        """Configure CORS with production settings"""
        origins = self.app.config.get('CORS_ORIGINS', ['http://localhost:3000'])

        CORS(self.app,
             origins=origins,
             methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
             allow_headers=['Content-Type', 'Authorization', 'X-API-Key'],
             expose_headers=['X-RateLimit-Remaining', 'X-RateLimit-Reset'],
             max_age=86400,  # 24 hours
             supports_credentials=True
        )

        logger.info(f"✅ CORS configured for origins: {origins}")

    def setup_rate_limiting(self):
        """Setup rate limiting to prevent abuse"""
        limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["60 per minute", "1000 per hour"],
            storage_uri=self.app.config.get('RATE_LIMIT_STORAGE_URL'),
            strategy="fixed-window"
        )

        # Stricter limits for sensitive endpoints
        limiter.limit("10 per minute")(self.app.view_functions.get('transcribe_audio'))
        limiter.limit("5 per minute")(self.app.view_functions.get('cognitive_assessment'))
        limiter.limit("20 per hour")(self.app.view_functions.get('save_recording'))

        logger.info("✅ Rate limiting configured")

    def setup_security_headers(self):
        """Add security headers to all responses"""
        @self.app.after_request
        def add_security_headers(response):
            # Prevent MIME type sniffing
            response.headers['X-Content-Type-Options'] = 'nosniff'

            # Prevent clickjacking
            response.headers['X-Frame-Options'] = 'DENY'

            # XSS protection
            response.headers['X-XSS-Protection'] = '1; mode=block'

            # HSTS (only for HTTPS)
            if request.is_secure:
                response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

            # Referrer policy
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

            # Permissions policy for sensitive features
            response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(self), camera=()'

            # Content Security Policy (basic)
            csp = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
            response.headers['Content-Security-Policy'] = csp

            return response

        logger.info("✅ Security headers configured")

    def setup_request_validation(self):
        """Validate incoming requests"""
        @self.app.before_request
        def validate_request():
            # Skip validation for health checks and static files
            if request.path in ['/api/health', '/favicon.ico'] or request.path.startswith('/static/'):
                return

            # Validate Content-Type for POST/PUT requests
            if request.method in ['POST', 'PUT'] and request.path.startswith('/api/'):
                content_type = request.headers.get('Content-Type', '')
                if not (content_type.startswith('application/json') or
                       content_type.startswith('multipart/form-data')):
                    abort(400, "Invalid Content-Type. Expected application/json or multipart/form-data")

            # Check for malicious patterns in request data
            self._validate_request_data()

            # Log suspicious activities
            self._log_suspicious_activity()

    def _validate_request_data(self):
        """Validate request data for malicious content"""
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'document\.cookie',
            r'localStorage',
            r'sessionStorage',
            r'XMLHttpRequest',
            r'fetch\s*\(',
            r'<iframe',
            r'<object',
            r'<embed',
            r'union\s+select',
            r'drop\s+table',
            r'--',
            r'/\*.*\*/',
            r'xp_cmdshell',
            r'exec\s*\('
        ]

        # Check URL parameters
        for key, value in request.args.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE | re.DOTALL):
                        logger.warning(f"🚨 Malicious pattern detected in URL parameter: {key}")
                        abort(400, "Malicious input detected")

        # Check form data
        for key, value in request.form.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE | re.DOTALL):
                        logger.warning(f"🚨 Malicious pattern detected in form data: {key}")
                        abort(400, "Malicious input detected")

        # Check JSON data
        try:
            json_data = request.get_json(silent=True)
            if json_data:
                self._validate_json_data(json_data, dangerous_patterns)
        except Exception as e:
            logger.warning(f"Error parsing JSON data: {e}")

    def _validate_json_data(self, data: Any, patterns: list):
        """Recursively validate JSON data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    for pattern in patterns:
                        if re.search(pattern, value, re.IGNORECASE | re.DOTALL):
                            logger.warning(f"🚨 Malicious pattern detected in JSON field: {key}")
                            abort(400, "Malicious input detected")
                elif isinstance(value, (dict, list)):
                    self._validate_json_data(value, patterns)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._validate_json_data(item, patterns)

    def _log_suspicious_activity(self):
        """Log potentially suspicious activities"""
        client_ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        path = request.path

        # Log API calls with suspicious user agents
        suspicious_uagents = ['sqlmap', 'nmap', 'nikto', 'dirbuster', 'gobuster']
        if any(agent in user_agent.lower() for agent in suspicious_uagents):
            logger.warning(f"🚨 Suspicious User-Agent detected: {user_agent} from {client_ip} accessing {path}")

        # Log rapid successive requests (basic rate monitoring)
        current_time = time.time()
        if not hasattr(g, 'last_request_time'):
            g.last_request_time = current_time
        else:
            time_diff = current_time - g.last_request_time
            if time_diff < 0.1:  # Less than 100ms between requests
                logger.warning(f"🚨 Rapid requests detected from {client_ip}: {time_diff}s between requests")
            g.last_request_time = current_time

    def setup_logging(self):
        """Setup security-focused logging"""
        # Log all authentication events
        @self.app.before_request
        def log_auth_events():
            if 'authorization' in request.headers:
                # Don't log the actual token, just that auth was provided
                logger.info(f"🔐 Authenticated request to {request.path} from {request.remote_addr}")

        # Log all file uploads
        @self.app.before_request
        def log_file_uploads():
            if request.files:
                file_info = {name: f"{file.filename} ({file.content_length} bytes)"
                           for name, file in request.files.items()}
                logger.info(f"📁 File upload to {request.path}: {file_info}")

    def create_api_key_middleware(self):
        """Create API key authentication middleware"""
        def require_api_key(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                api_key = request.headers.get('X-API-Key')
                expected_key = self.app.config.get('API_TOKEN')

                if not api_key or not expected_key:
                    abort(401, "API key required")

                # Use constant-time comparison to prevent timing attacks
                if not hmac.compare_digest(api_key, expected_key):
                    logger.warning(f"🚨 Invalid API key attempt from {request.remote_addr}")
                    abort(401, "Invalid API key")

                return f(*args, **kwargs)
            return decorated_function

        return require_api_key

    def create_jwt_middleware(self):
        """Create JWT authentication middleware"""
        def require_jwt(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    abort(401, "JWT token required")

                token = auth_header.split(' ')[1]
                # JWT validation logic would go here
                # For now, just check if token exists
                if not token:
                    abort(401, "Invalid JWT token")

                return f(*args, **kwargs)
            return decorated_function

        return require_jwt

    def create_file_upload_validator(self):
        """Create file upload validation middleware"""
        def validate_file_upload(allowed_extensions: set, max_size: int = 10*1024*1024):
            def decorator(f):
                @wraps(f)
                def decorated_function(*args, **kwargs):
                    if not request.files:
                        abort(400, "No file uploaded")

                    for file in request.files.values():
                        # Check file extension
                        if '.' not in file.filename:
                            abort(400, "File must have extension")

                        extension = file.filename.rsplit('.', 1)[1].lower()
                        if extension not in allowed_extensions:
                            abort(400, f"File extension '{extension}' not allowed")

                        # Check file size
                        file.seek(0, 2)  # Seek to end
                        size = file.tell()
                        file.seek(0)  # Reset to beginning

                        if size > max_size:
                            abort(400, f"File size {size} exceeds maximum {max_size}")

                    return f(*args, **kwargs)
                return decorated_function
            return decorator

        return validate_file_upload

# Utility functions for easy use
def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import current_app, request, abort
        import hmac

        api_key = request.headers.get('X-API-Key')
        expected_key = current_app.config.get('API_TOKEN')

        if not api_key or not expected_key:
            abort(401, "API key required")

        if not hmac.compare_digest(api_key, expected_key):
            abort(401, "Invalid API key")

        return f(*args, **kwargs)
    return decorated_function

def validate_file_upload(allowed_extensions: set, max_size: int = 10*1024*1024):
    """Decorator to validate file uploads"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, abort

            if not request.files:
                abort(400, "No file uploaded")

            for file in request.files.values():
                if '.' not in file.filename:
                    abort(400, "File must have extension")

                extension = file.filename.rsplit('.', 1)[1].lower()
                if extension not in allowed_extensions:
                    abort(400, f"File extension '{extension}' not allowed")

                file.seek(0, 2)
                size = file.tell()
                file.seek(0)

                if size > max_size:
                    abort(400, f"File size {size} exceeds maximum {max_size}")

            return f(*args, **kwargs)
        return decorated_function
    return decorator

logger.info("✅ Security middleware module loaded")
