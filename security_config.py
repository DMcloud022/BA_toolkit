"""Security configuration for BA_toolkit."""

# Security settings
SECURITY_CONFIG = {
    # File upload settings
    'UPLOAD_SETTINGS': {
        'MAX_UPLOAD_SIZE': 100 * 1024 * 1024,  # 100MB total
        'MAX_FILES_PER_REQUEST': 5,
        'ALLOWED_FILE_TYPES': {
            'text': ['.txt', '.md'],
            'document': ['.pdf', '.docx'],
            'spreadsheet': ['.csv', '.xlsx', '.xls'],
            'data': ['.json', '.xml', '.yaml', '.yml', '.parquet'],
            'database': ['.db', '.sqlite', '.sqlite3']
        }
    },
    
    # URL settings
    'URL_SETTINGS': {
        'ALLOWED_SCHEMES': ['http', 'https'],
        'BLOCKED_HOSTS': [
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
            '::1'
        ],
        'MAX_URL_LENGTH': 2083,  # Standard maximum URL length
        'REQUEST_TIMEOUT': 30,  # seconds
    },
    
    # Content security settings
    'CONTENT_SECURITY': {
        'MAX_TEXT_LENGTH': 1000000,  # characters
        'BLOCKED_PATTERNS': [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
        ],
        'HTML_TAGS_WHITELIST': [
            'p', 'br', 'b', 'i', 'u', 'em', 'strong', 'a', 'h1', 'h2', 'h3',
            'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote', 'code', 'pre'
        ]
    },
    
    # Rate limiting
    'RATE_LIMITS': {
        'MAX_REQUESTS_PER_MINUTE': 60,
        'MAX_FILE_UPLOADS_PER_MINUTE': 10,
        'MAX_TEXT_ANALYSES_PER_MINUTE': 20,
        'MAX_URL_ANALYSES_PER_MINUTE': 15
    },
    
    # Temporary file settings
    'TEMP_FILE_SETTINGS': {
        'MAX_AGE': 3600,  # seconds
        'CLEANUP_INTERVAL': 300,  # seconds
        'MAX_TEMP_SIZE': 1024 * 1024 * 1024,  # 1GB
    },
    
    # Logging settings
    'LOGGING': {
        'ENABLE_SECURITY_LOGGING': True,
        'LOG_LEVEL': 'INFO',
        'LOG_FILE': 'security.log',
        'MAX_LOG_SIZE': 10 * 1024 * 1024,  # 10MB
        'BACKUP_COUNT': 5
    }
}

# Security headers
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';",
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Referrer-Policy': 'strict-origin-when-cross-origin'
} 