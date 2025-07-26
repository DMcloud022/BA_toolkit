# Analysis_toolkit

A toolkit for business analysis tasks.

```bash
# Create and activate virtual environment
# On Linux/macOS
python3 -m venv venv && source venv/bin/activate

# On Windows
python -m venv venv && .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.sample .env  # On Linux/macOS
copy .env.sample .env  # On Windows

# Edit .env and replace placeholder values

# Run the application
streamlit run main.py
