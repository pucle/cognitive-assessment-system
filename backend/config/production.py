# Production Configuration for Cognitive Assessment Backend
# Secure, optimized settings for production deployment

import os
from pathlib import Path

class ProductionConfig:
    """Production configuration with security and performance optimizations"""

    # ================================
    # Flask Configuration
    # ================================
    FLASK_ENV = 'production'
    DEBUG = False
    TESTING = False

    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))

    # Secret key (MUST be set in environment)
    SECRET_KEY = os.getenv('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable is required")

    # JWT settings
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 30

    # ================================
    # Database Configuration
    # ================================
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is required")

    # Database connection pool
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 10,
        'max_overflow': 20,
        'echo': False
    }

    # ================================
    # AI/ML Configuration
    # ================================
    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Google Gemini
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', GEMINI_API_KEY)

    # Transcription settings
    ENABLE_PAID_TRANSCRIPTION = os.getenv('ENABLE_PAID_TRANSCRIPTION', 'true').lower() == 'true'
    TRANSCRIPTION_BUDGET_LIMIT = float(os.getenv('TRANSCRIPTION_BUDGET_LIMIT', '5.00'))

    # Vietnamese ASR Model
    VI_ASR_MODEL = os.getenv('VI_ASR_MODEL', 'nguyenvulebinh/wav2vec2-large-vietnamese-250h')

    # ================================
    # File Storage Configuration
    # ================================
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_DIR = BASE_DIR / 'uploads'
    STORAGE_DIR = BASE_DIR / 'storage'
    RECORDINGS_DIR = BASE_DIR / 'recordings'
    LOGS_DIR = BASE_DIR / 'logs'

    # Create directories if they don't exist
    for directory in [UPLOAD_DIR, STORAGE_DIR, RECORDINGS_DIR, LOGS_DIR]:
        directory.mkdir(exist_ok=True, parents=True)

    # File paths
    UPLOAD_PATH = os.getenv('UPLOAD_PATH', str(UPLOAD_DIR))
    STORAGE_PATH = os.getenv('STORAGE_PATH', str(STORAGE_DIR))
    RECORDINGS_PATH = os.getenv('RECORDINGS_PATH', str(RECORDINGS_DIR))

    # File size limits
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    MAX_AUDIO_FILE_SIZE = 50 * 1024 * 1024   # 50MB

    # Allowed file extensions
    ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'webm', 'ogg', 'flac'}
    ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

    # ================================
    # Caching Configuration
    # ================================
    CACHE_DIR = os.getenv('CACHE_DIR', '/tmp/cognitive_cache')
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))  # 1 hour
    AUDIO_CACHE_SIZE = int(os.getenv('AUDIO_CACHE_SIZE', '100'))
    TEXT_CACHE_SIZE = int(os.getenv('TEXT_CACHE_SIZE', '500'))

    # Redis (if available)
    REDIS_URL = os.getenv('REDIS_URL')
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL)
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)

    # ================================
    # ML Model Configuration
    # ================================
    MODEL_PATH = os.getenv('MODEL_PATH', str(BASE_DIR / 'models'))
    SVM_MODEL_PATH = os.getenv('SVM_MODEL_PATH', str(BASE_DIR / 'models' / 'svm_model.pkl'))
    SCALER_PATH = os.getenv('SCALER_PATH', str(BASE_DIR / 'models' / 'scaler.pkl'))

    # Model performance settings
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    ML_TIMEOUT = 300  # 5 minutes
    BATCH_SIZE = 32

    # ================================
    # Security Configuration
    # ================================
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'https://your-frontend.vercel.app').split(',')

    # Rate limiting
    RATE_LIMIT_DEFAULT = "60/minute"
    RATE_LIMIT_STORAGE_URL = REDIS_URL

    # Session settings
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour

    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }

    # ================================
    # Logging Configuration
    # ================================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'WARNING').upper()
    LOG_FILE = os.getenv('LOG_FILE', str(LOGS_DIR / 'backend.log'))
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5

    # External logging (optional)
    SENTRY_DSN = os.getenv('SENTRY_DSN')

    # ================================
    # Monitoring Configuration
    # ================================
    # Health check settings
    HEALTH_CHECK_INTERVAL = 30
    HEALTH_CHECK_TIMEOUT = 10

    # Metrics
    ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'false').lower() == 'true'
    METRICS_PORT = int(os.getenv('METRICS_PORT', '9090'))

    # ================================
    # Email Configuration (Optional)
    # ================================
    MAIL_SERVER = os.getenv('SMTP_HOST')
    MAIL_PORT = int(os.getenv('SMTP_PORT', 587))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.getenv('SMTP_USER')
    MAIL_PASSWORD = os.getenv('SMTP_PASS')
    MAIL_DEFAULT_SENDER = os.getenv('EMAIL_USER', 'noreply@yourapp.com')

    # ================================
    # Feature Flags
    # ================================
    ENABLE_ADVANCED_ANALYTICS = os.getenv('ENABLE_ADVANCED_ANALYTICS', 'true').lower() == 'true'
    ENABLE_REAL_TIME_PROCESSING = os.getenv('ENABLE_REAL_TIME_PROCESSING', 'false').lower() == 'true'
    ENABLE_MODEL_EXPLAINABILITY = os.getenv('ENABLE_MODEL_EXPLAINABILITY', 'true').lower() == 'true'

    # ================================
    # External Services
    # ================================
    # Vercel Blob (for file storage)
    BLOB_READ_WRITE_TOKEN = os.getenv('BLOB_READ_WRITE_TOKEN')

    # API tokens
    API_TOKEN = os.getenv('API_TOKEN', 'change-me-token')

    # ================================
    # Validation
    # ================================
    def __init__(self):
        """Validate configuration on initialization"""
        required_vars = [
            'SECRET_KEY',
            'DATABASE_URL',
            'OPENAI_API_KEY',
            'GEMINI_API_KEY'
        ]

        missing_vars = []
        for var in required_vars:
            if not getattr(self, var, None):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Validate database URL format
        if not self.DATABASE_URL.startswith(('postgresql://', 'postgres://')):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection string")

        print("✅ Production configuration loaded successfully")
