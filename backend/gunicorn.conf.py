# Gunicorn configuration for Cognitive Assessment Backend
# Production-ready server configuration

import multiprocessing
import os

# Server socket
bind = os.getenv('HOST', '0.0.0.0') + ':' + os.getenv('PORT', '8000')
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings
timeout = 300  # 5 minutes for ML processing
keepalive = 10
graceful_timeout = 30

# Logging
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
accesslog = os.getenv('ACCESS_LOG', '-')
errorlog = os.getenv('ERROR_LOG', '-')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'cognitive_assessment'

# Server mechanics
preload_app = True
pidfile = '/tmp/gunicorn.pid'
user = os.getenv('APP_USER', 'appuser')
group = os.getenv('APP_GROUP', 'appuser')
tmp_upload_dir = None

# SSL (if needed)
keyfile = os.getenv('SSL_KEY_FILE')
certfile = os.getenv('SSL_CERT_FILE')

# Application
wsgi_module = 'app:app'
pythonpath = '/app'

# Development overrides
if os.getenv('FLASK_ENV') == 'development':
    workers = 1
    loglevel = 'debug'
    reload = True
    reload_extra_files = ['*.py', '*.html', '*.css', '*.js']

# Production optimizations
if os.getenv('FLASK_ENV') == 'production':
    # Disable debug mode
    loglevel = 'warning'
    # Enable statsd metrics (optional)
    statsd_host = os.getenv('STATSD_HOST')
    statsd_prefix = os.getenv('STATSD_PREFIX', 'cognitive_assessment')

    # Memory optimization
    worker_tmp_dir = '/dev/shm'
