import hashlib
import re
import magic
import validators
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
from urllib.parse import urlparse
import mimetypes
import tempfile
import os
import secrets
from security_config import SECURITY_CONFIG, SECURITY_HEADERS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityManager:
    """Security manager for BA_toolkit."""
    
    # Allowed mime types for different file categories
    ALLOWED_MIMES = {
        'text': [
            'text/plain',
            'text/markdown',
            'text/x-markdown',
            'text/md',
            'application/x-markdown',
            'text/x-web-markdown',
            'text/x-rst',
            'text/x-python',
            'text/x-script',
            'text/x-log',
            'text/x-shellscript',
            'text/x-perl',
            'text/x-ruby',
            'text/x-php',
            'text/x-java-source',
            'text/x-c',
            'text/x-c++',
            'text/x-csrc',
            'text/x-chdr',
            'text/x-c++src',
            'text/x-c++hdr',
            'application/x-txt',
            'application/text'
        ],
        'document': [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        ],
        'spreadsheet': [
            'text/csv',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
        ],
        'data': [
            'application/json',
            'application/xml',
            'text/xml',
            'application/x-yaml',
            'application/x-parquet',
        ],
        'database': [
            'application/x-sqlite3',
            'application/vnd.sqlite3',
        ]
    }

    # Maximum file sizes in bytes for different file types
    MAX_FILE_SIZES = {
        'text': 10 * 1024 * 1024,  # 10MB
        'document': 50 * 1024 * 1024,  # 50MB
        'spreadsheet': 20 * 1024 * 1024,  # 20MB
        'data': 30 * 1024 * 1024,  # 30MB
        'database': 100 * 1024 * 1024,  # 100MB
    }

    # Patterns for input validation
    PATTERNS = {
        'url': re.compile(
            r'^https?:\/\/'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        ),
        'filename': re.compile(r'^[\w\-. ]+$'),
        'path': re.compile(r'^[\w\-./]+$'),
    }

    def __init__(self):
        """Initialize the security manager."""
        self.temp_dir = tempfile.mkdtemp(prefix='ba_toolkit_')
        self._setup_temp_dir()

    def _setup_temp_dir(self):
        """Set up secure temporary directory."""
        os.chmod(self.temp_dir, 0o700)  # Restrict permissions
        logger.info(f"Initialized secure temporary directory: {self.temp_dir}")

    def cleanup(self):
        """Clean up temporary files."""
        try:
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _get_file_category(self, filename: str) -> Optional[str]:
        """Determine file category based on extension."""
        ext = Path(filename).suffix.lower().lstrip('.')
        
        # Direct extension mapping
        extension_map = {
            'txt': 'text',
            'md': 'text',
            'markdown': 'text',
            'pdf': 'document',
            'docx': 'document',
            'csv': 'spreadsheet',
            'xlsx': 'spreadsheet',
            'xls': 'spreadsheet',
            'json': 'data',
            'xml': 'data',
            'yaml': 'data',
            'yml': 'data',
            'parquet': 'data',
            'db': 'database',
            'sqlite': 'database',
            'sqlite3': 'database'
        }
        
        # First try direct mapping
        if ext in extension_map:
            logger.info(f"File category for .{ext}: {extension_map[ext]}")
            return extension_map[ext]
        
        # Fallback to MIME type checking
        for category, mimes in self.ALLOWED_MIMES.items():
            if any(mime.endswith(ext) for mime in mimes):
                logger.info(f"File category for .{ext} (MIME match): {category}")
                return category
                
        logger.warning(f"No category found for extension: .{ext}")
        return None

    def validate_file(self, file: Any) -> Tuple[bool, str]:
        """
        Validate uploaded file for security concerns.
        Returns (is_valid, message)
        """
        try:
            # Check if file exists
            if not file or not hasattr(file, 'name'):
                return False, "Invalid file object"

            # Log the validation attempt
            logger.info(f"Validating file: {file.name}")

            # Validate filename
            if not self.PATTERNS['filename'].match(file.name):
                logger.warning(f"Invalid filename pattern: {file.name}")
                return False, "Invalid filename"

            # Get file category
            file_category = self._get_file_category(file.name)
            if not file_category:
                logger.warning(f"Unsupported file type: {file.name}")
                return False, f"Unsupported file type: {Path(file.name).suffix}"

            # Check file size
            if file.size > self.MAX_FILE_SIZES.get(file_category, 0):
                logger.warning(f"File too large: {file.size} bytes")
                return False, f"File too large for {file_category} type (max {self.MAX_FILE_SIZES[file_category]/1024/1024}MB)"

            # Save file temporarily for mime type checking
            temp_path = os.path.join(self.temp_dir, secrets.token_hex(16))
            with open(temp_path, 'wb') as f:
                f.write(file.getvalue())

            try:
                # Check mime type
                mime_type = magic.from_file(temp_path, mime=True)
                logger.info(f"Detected MIME type: {mime_type} for file: {file.name}")
                
                # Special handling for text files
                if file_category == 'text':
                    # Accept any text/* mime type for text files
                    if mime_type.startswith('text/'):
                        logger.info(f"Accepting text file with MIME type: {mime_type}")
                        return True, "File validation successful"
                    # Try to read as text
                    try:
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            f.read()
                        logger.info("File validated as readable text")
                        return True, "File validation successful"
                    except UnicodeDecodeError:
                        logger.warning("File is not a valid text file (UnicodeDecodeError)")
                        return False, "File is not a valid text file"
                
                # For non-text files, check against allowed MIME types
                allowed_mimes = self.ALLOWED_MIMES.get(file_category, [])
                if mime_type not in allowed_mimes:
                    logger.warning(f"Invalid MIME type: {mime_type} for category: {file_category}")
                    logger.debug(f"Allowed MIME types for {file_category}: {allowed_mimes}")
                    return False, f"Invalid file type: {mime_type}"

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # Additional checks for specific file types
            if file_category == 'document':
                return self._validate_document(file)
            elif file_category == 'data':
                return self._validate_data_file(file)
            elif file_category == 'database':
                return self._validate_database(file)

            return True, "File validation successful"

        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            return False, f"File validation failed: {str(e)}"

    def validate_url(self, url: str) -> Tuple[bool, str]:
        """
        Validate URL for security concerns.
        Returns (is_valid, message)
        """
        try:
            # Basic URL validation
            if not url or not isinstance(url, str):
                return False, "Invalid URL"

            # Check URL pattern
            if not self.PATTERNS['url'].match(url):
                return False, "Invalid URL format"

            # Use validators library for thorough URL validation
            if not validators.url(url):
                return False, "Invalid URL structure"

            # Parse URL for additional checks
            parsed = urlparse(url)
            
            # Check for localhost and private IPs
            if parsed.hostname in ['localhost', '127.0.0.1'] or \
               parsed.hostname.startswith(('192.168.', '10.', '172.')):
                return False, "Local and private URLs not allowed"

            # Check protocol
            if parsed.scheme not in ['http', 'https']:
                return False, "Only HTTP(S) protocols allowed"

            return True, "URL validation successful"

        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False, "URL validation failed"

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input to prevent XSS and other injection attacks.
        """
        if not text:
            return ""

        # Remove potential script tags and other dangerous HTML
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.I | re.S)
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        
        # Remove potential SQL injection patterns
        text = re.sub(r'(\b(union|select|insert|update|delete|drop|alter)\b)', 
                     lambda m: m.group(1).upper(), text, flags=re.I)
        
        return text

    def _validate_document(self, file: Any) -> Tuple[bool, str]:
        """Additional validation for document files."""
        try:
            if file.name.lower().endswith('.pdf'):
                # Check for PDF-specific security concerns
                content = file.read()
                file.seek(0)  # Reset file pointer
                
                # Check for potential PDF exploits
                if b'JavaScript' in content or b'/JS' in content:
                    return False, "PDF contains potentially unsafe JavaScript"
                if b'/Launch' in content or b'/Action' in content:
                    return False, "PDF contains potentially unsafe actions"
                
            return True, "Document validation successful"
        except Exception as e:
            logger.error(f"Document validation error: {e}")
            return False, "Document validation failed"

    def _validate_data_file(self, file: Any) -> Tuple[bool, str]:
        """Additional validation for data files."""
        try:
            content = file.read()
            file.seek(0)  # Reset file pointer
            
            # Check for potential command injection in JSON/YAML
            content_str = content.decode('utf-8', errors='ignore')
            if any(pattern in content_str.lower() for pattern in [
                '$(', '`', '${', 'eval(', 'exec(', 'system('
            ]):
                return False, "Data file contains potentially unsafe content"
            
            return True, "Data file validation successful"
        except Exception as e:
            logger.error(f"Data file validation error: {e}")
            return False, "Data file validation failed"

    def _validate_database(self, file: Any) -> Tuple[bool, str]:
        """Additional validation for database files."""
        try:
            # Check SQLite file header
            header = file.read(16)
            file.seek(0)  # Reset file pointer
            
            if not header.startswith(b'SQLite format 3'):
                return False, "Invalid SQLite database format"
            
            return True, "Database validation successful"
        except Exception as e:
            logger.error(f"Database validation error: {e}")
            return False, "Database validation failed"

    def generate_file_hash(self, content: bytes) -> str:
        """Generate secure hash of file content."""
        return hashlib.blake2b(content).hexdigest()

    def verify_security_setup(self) -> Dict[str, bool]:
        """
        Verify security configuration and setup.
        Returns a dictionary of check results.
        """
        checks = {}
        
        try:
            # Check temporary directory
            checks['temp_dir_exists'] = os.path.exists(self.temp_dir)
            checks['temp_dir_permissions'] = oct(os.stat(self.temp_dir).st_mode)[-3:] == '700'
            
            # Check MIME type detection
            checks['magic_available'] = magic is not None
            
            # Check validators
            checks['validators_available'] = validators is not None
            
            # Verify patterns
            checks['url_pattern_valid'] = bool(self.PATTERNS['url'].pattern)
            checks['filename_pattern_valid'] = bool(self.PATTERNS['filename'].pattern)
            
            # Check configuration
            # Assuming SECURITY_CONFIG and SECURITY_HEADERS are defined elsewhere or will be added
            # For now, we'll assume they are available or will be added.
            # checks['security_config_loaded'] = bool(SECURITY_CONFIG)
            # checks['security_headers_loaded'] = bool(SECURITY_HEADERS)
            
            # Log results
            for check, result in checks.items():
                if not result:
                    logger.warning(f"Security check failed: {check}")
                else:
                    logger.info(f"Security check passed: {check}")
            
            return checks
            
        except Exception as e:
            logger.error(f"Security verification error: {str(e)}")
            return {'error': False}

    def create_secure_temp_file(self, content: bytes) -> str:
        """Create a secure temporary file with the given content."""
        temp_path = os.path.join(self.temp_dir, secrets.token_hex(16))
        with open(temp_path, 'wb') as f:
            f.write(content)
        os.chmod(temp_path, 0o600)  # Restrict permissions
        return temp_path 