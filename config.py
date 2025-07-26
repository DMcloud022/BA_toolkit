import os
from dotenv import load_dotenv
from pathlib import Path

# Try to load environment variables from .env file
env_path = Path('.') / '.env'
print(f"Looking for .env file at: {env_path.absolute()}")
print(f".env file exists: {env_path.exists()}")

# Load from .env file if it exists
if env_path.exists():
    load_dotenv(env_path)
    print("Loaded .env file")
else:
    print("No .env file found, trying system environment variables")

# Access the environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# If not found, provide helpful error message
if not GROQ_API_KEY:
    print("\n" + "="*50)
    print("GROQ_API_KEY not found!")
    print("Please ensure you have either:")
    print("1. A .env file in the project root with: GROQ_API_KEY=your_key_here")
    print("2. Set the environment variable in your system")
    print("="*50)
    
    # Try to continue without API key for basic functionality
    GROQ_API_KEY = None

# Configuration constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_FILE_TYPES = {
    'text': ['txt', 'md'],
    'document': ['pdf', 'docx'],
    'spreadsheet': ['csv', 'xlsx', 'xls'],
    'data': ['json', 'xml', 'yaml', 'yml', 'parquet'],
    'database': ['db', 'sqlite', 'sqlite3']
}

# Cache settings
CACHE_ENABLED = True
CACHE_DURATION = 3600  # 1 hour in seconds
MAX_CACHE_ENTRIES = 100

# Analysis settings
MIN_WORDS_FOR_ANALYSIS = 50
MAX_WORDS_FOR_SUMMARY = 1000
CORRELATION_THRESHOLD = 0.7
MAX_CATEGORICAL_VALUES = 50

# AI Model settings
DEFAULT_MODEL = "llama3-8b-8192"
MAX_TOKENS = 4096

# Visualization settings
DEFAULT_FIGURE_SIZE = (10, 6)
MAX_CATEGORIES_IN_PLOT = 20
PLOT_STYLE = 'seaborn-v0_8' 