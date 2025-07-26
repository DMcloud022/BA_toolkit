import streamlit as st
import pandas as pd
import logging
import warnings
from typing import Dict, Any, Optional, Tuple
from pypdf import PdfReader
from functions import (
    analyze_website,
    analyze_document,
    analyze_excel,
    analyze_database,
    analyze_json,
    analyze_xml,
    analyze_parquet,
    analyze_yaml,
    analyze_text,
    generate_visualizations
)

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure specific loggers
logging.getLogger('pdfminer').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)

# Sentiment analysis constants
NEGATIVE_INDICATORS = [
    "not", "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't", "won't", "wouldn't",
    "can't", "couldn't", "shouldn't", "won't",
    "bad", "poor", "unhappy", "disappointed", "unfortunate",
    "problem", "issue", "fail", "error", "bug",
    "not happy", "not satisfied", "not good"
]

import config
import re
import nltk
from textblob import TextBlob
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from string import punctuation
from security import SecurityManager
from security_config import SECURITY_CONFIG, SECURITY_HEADERS
import streamlit.components.v1 as components
import os
import validators
from datetime import datetime
from pathlib import Path

# Initialize security manager after other configurations
security_manager = SecurityManager()
security_checks = security_manager.verify_security_setup()

if not all(security_checks.values()):
    logger.error("Security setup verification failed!")
    failed_checks = [check for check, result in security_checks.items() if not result]
    logger.error(f"Failed checks: {failed_checks}")

# Add cleanup on exit
import atexit
atexit.register(security_manager.cleanup)

# Download required NLTK data
def setup_nltk():
    """Set up NLTK resources with proper error handling."""
    logger.info("Setting up NLTK resources...")
    
    required_resources = {
        'punkt': 'tokenizers/punkt',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
        'words': 'corpora/words',
        'stopwords': 'corpora/stopwords',
        'omw-1.4': 'corpora/omw-1.4',
        'maxent_ne_chunker_tab': 'chunkers/maxent_ne_chunker_tab'
    }
    
    for resource_name, resource_path in required_resources.items():
        try:
            nltk.data.find(resource_path)
            logger.debug(f"NLTK resource already available: {resource_name}")
        except LookupError:
            try:
                logger.info(f"Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, quiet=True)
                logger.info(f"Successfully downloaded NLTK resource: {resource_name}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource_name}: {str(e)}")
                if resource_name == 'maxent_ne_chunker_tab':
                    logger.info("Will use alternative NER approach")
                else:
                    logger.warning(f"Missing NLTK resource may affect functionality: {resource_name}")

    logger.info("NLTK setup completed")

# Initialize NLTK
try:
    setup_nltk()
except Exception as e:
    logger.error(f"Error during NLTK setup: {str(e)}")
    logger.info("Continuing with limited NLTK functionality")


def main():
    st.set_page_config(
        page_title="Analysis Toolkit", 
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Remove unused session state setup
    if 'security_headers_set' not in st.session_state:
        for header, value in SECURITY_HEADERS.items():
            st.session_state[f"_{header}"] = value
        st.session_state.security_headers_set = True

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .nav-button-active {
            background-color: #ff4b4b;
            color: white;
        }
        .nav-button-inactive {
            background-color: #f0f2f6;
            color: #31333F;
        }
        .nav-button:hover {
            opacity: 0.8;
        }
        .stButton>button {
            width: 100%;
        }
        .upload-section {
            padding: 2rem;
            border-radius: 10px;
            border: 2px dashed #f0f2f6;
            margin-bottom: 2rem;
        }
        .url-section {
            padding: 2rem;
            border-radius: 10px;
            background-color: #f8f9fa;
            margin-bottom: 2rem;
        }
        .info-box {
            padding: 1.5rem;
            border-radius: 8px;
            background-color: #f8f9fa;
            margin: 1rem 0;
        }
        .metric-container {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.title("📊 Business Analysis Toolkit")
    st.markdown("*Powerful analytics for your business data and web content created by Daniel Florencio*")
    st.markdown("---")

    # Navigation using buttons in main area
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Get current page from session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'upload'

    # Navigation buttons
    with col1:
        if st.button("📁 File Analysis", 
                    key="nav_upload",
                    help="Upload and analyze files (CSV, Excel, PDF, etc.)",
                    use_container_width=True):
            st.session_state.current_page = 'upload'
            
    with col2:
        if st.button("🌐 Website Analysis",
                    key="nav_website",
                    help="Analyze any website URL",
                    use_container_width=True):
            st.session_state.current_page = 'website'
            
    with col3:
        if st.button("📝 Text Analyzer",
                    key="nav_text_analyzer",
                    help="Analyze text content and get insights",
                    use_container_width=True):
            st.session_state.current_page = 'text_analyzer'
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Display current page
    if st.session_state.current_page == "upload":
        handle_file_upload()
    elif st.session_state.current_page == "website":
        handle_website_analysis()
    elif st.session_state.current_page == "text_analyzer":
        text_analyzer()

def handle_file_upload():
    """Display the file upload page with enhanced UI."""
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("📁 File Analysis")
        st.markdown("Upload your files for comprehensive analysis")
        
        # Show supported file types in a cleaner way
        with st.expander("ℹ️ Supported File Types"):
            for category, extensions in config.SUPPORTED_FILE_TYPES.items():
                st.markdown(f"**{category.title()}**")
                st.markdown(f"_{', '.join(extensions)}_")
    
    # File uploader with improved styling
    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=list(sum(config.SUPPORTED_FILE_TYPES.values(), [])),
        help="Drag and drop or click to upload"
    )

    if uploaded_file:
        try:
            # Security validation
            is_valid, message = security_manager.validate_file(uploaded_file)
            if not is_valid:
                st.error(f"❌ Security validation failed: {message}")
                return
            
            # Create secure temporary file
            temp_file = security_manager.create_secure_temp_file(uploaded_file.getvalue())
            
            # Display file information
            display_file_info(uploaded_file)
            
            # Process and analyze file
            with st.spinner("🔄 Analyzing file..."):
                analysis_results = process_file(uploaded_file)
                display_analysis_results(analysis_results, uploaded_file.name)
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"File analysis error: {str(e)}")
        finally:
            # Cleanup any temporary files
            if 'temp_file' in locals():
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {e}")

def validate_uploaded_file(file) -> Dict[str, Any]:
    """Validate uploaded file for size, format, and content."""
    try:
        # Check file size
        if file.size > config.MAX_FILE_SIZE:
            return {
                "error": f"File too large. Maximum size is {config.MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
            }
        
        # Check if file is empty
        if file.size == 0:
            return {"error": "File is empty"}
        
        # Validate file extension
        file_ext = file.name.lower().split('.')[-1] if '.' in file.name else ''
        valid_extension = False
        file_category = None
        
        for category, extensions in config.SUPPORTED_FILE_TYPES.items():
            if file_ext in extensions:
                valid_extension = True
                file_category = category
                break
        
        if not valid_extension:
            return {"error": f"Unsupported file type: .{file_ext}"}
        
        # Additional validation based on file type
        if file_category == 'spreadsheet':
            try:
                # Try to read first few rows to validate format
                if file_ext == 'csv':
                    pd.read_csv(file, nrows=5)
                else:
                    pd.read_excel(file, nrows=5)
            except Exception as e:
                return {"error": f"Invalid spreadsheet format: {str(e)}"}
                
        elif file_category == 'document':
            if file_ext == 'pdf':
                try:
                    # Verify PDF is readable using pypdf
                    reader = PdfReader(file)
                    if len(reader.pages) == 0:
                        return {"error": "PDF file appears to be empty"}
                    # Reset file pointer for later use
                    file.seek(0)
                except Exception as e:
                    return {"error": f"Invalid PDF format: {str(e)}"}
                    
        elif file_category == 'data':
            if file_ext == 'json':
                try:
                    json.load(file)
                    file.seek(0)  # Reset file pointer
                except Exception:
                    return {"error": "Invalid JSON format"}
            elif file_ext in ['yaml', 'yml']:
                try:
                    yaml.safe_load(file)
                    file.seek(0)
                except Exception:
                    return {"error": "Invalid YAML format"}
        
        return {"success": True, "category": file_category}
        
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        return {"error": "Failed to validate file"}

def display_file_info(file):
    """Display comprehensive file information."""
    try:
        # Basic file info
        file_ext = file.name.lower().split('.')[-1] if '.' in file.name else 'unknown'
        file_category = next(
            (cat for cat, exts in config.SUPPORTED_FILE_TYPES.items() 
             if file_ext in exts),
            "Unknown"
        )
        
        # Create metrics
        st.info("📋 **File Information**")
        col1, col2, col3, col4 = st.columns(4)
        
        # Size formatting
        size_kb = file.size / 1024
        if size_kb < 1024:
            size_str = f"{size_kb:.1f} KB"
        else:
            size_mb = size_kb / 1024
            size_str = f"{size_mb:.1f} MB"
        
        col1.metric("Filename", file.name)
        col2.metric("Size", size_str)
        col3.metric("Type", file_ext.upper())
        col4.metric("Category", file_category.title())
        
        # Additional file details
        if file.type:
            st.info("📄 **File Details**")
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.write(f"**MIME Type:** {file.type}")
                st.write(f"**Extension:** .{file_ext}")
            
            with details_col2:
                st.write(f"**Category:** {file_category.title()}")
                st.write("**Processing Mode:** " + 
                        ("Binary" if file_ext in ['pdf', 'xlsx', 'docx'] else "Text"))
        
    except Exception as e:
        logger.error(f"Error displaying file info: {str(e)}")
        st.warning("⚠️ Some file information could not be displayed")

def process_file(uploaded_file):
    """Process uploaded file with enhanced error handling and validation."""
    try:
        filename = uploaded_file.name.lower()
        file_type = uploaded_file.type
        
        # Document files
        if any(filename.endswith(ext) for ext in config.SUPPORTED_FILE_TYPES['document']) or \
           file_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            with st.status("📄 Processing document...") as status:
                status.write("Extracting text...")
                if filename.endswith('.pdf'):
                    # Handle PDF files with pypdf
                    try:
                        reader = PdfReader(uploaded_file)
                        text = ""
                        status.write(f"Reading {len(reader.pages)} pages...")
                        for i, page in enumerate(reader.pages, 1):
                            status.write(f"Processing page {i}/{len(reader.pages)}")
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
                        
                        if not text.strip():
                            return {
                                "error": "No readable text found in PDF. The file might be scanned or protected.",
                                "suggestions": [
                                    "Check if the PDF contains actual text (not scanned images)",
                                    "Ensure the PDF is not password-protected",
                                    "Try converting scanned PDFs using OCR software first"
                                ]
                            }
                        
                        # Analyze the extracted text
                        result = analyze_text(text)
                        status.update(label="✅ PDF processed", state="complete")
                        return result
                    except Exception as e:
                        logger.error(f"PDF processing error: {str(e)}")
                        return {"error": f"Failed to process PDF: {str(e)}"}
                else:
                    # Handle other document types
                    result = analyze_document(uploaded_file)
                    status.update(label="✅ Document processed", state="complete")
                    return result
        
        # Spreadsheet files
        elif any(filename.endswith(ext) for ext in config.SUPPORTED_FILE_TYPES['spreadsheet']) or \
             file_type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            with st.status("📊 Processing spreadsheet...") as status:
                status.write("Reading data...")
                result = analyze_excel(uploaded_file)
                status.update(label="✅ Spreadsheet processed", state="complete")
                return result
        
        # Database files
        elif any(filename.endswith(ext) for ext in config.SUPPORTED_FILE_TYPES['database']):
            with st.status("🗄️ Processing database...") as status:
                status.write("Connecting to database...")
                result = analyze_database(uploaded_file)
                status.update(label="✅ Database processed", state="complete")
                return result
        
        # Data files
        elif any(filename.endswith(ext) for ext in config.SUPPORTED_FILE_TYPES['data']):
            with st.status("🔍 Processing data file...") as status:
                status.write("Parsing data...")
                if filename.endswith('.json'):
                    result = analyze_json(uploaded_file)
                elif filename.endswith(('.xml', '.html')):
                    result = analyze_xml(uploaded_file)
                elif filename.endswith('.parquet'):
                    result = analyze_parquet(uploaded_file)
                elif filename.endswith(('.yaml', '.yml')):
                    result = analyze_yaml(uploaded_file)
                status.update(label="✅ Data file processed", state="complete")
                return result
        
        return {"error": f"Unsupported file type: {filename}"}
        
    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        return {"error": f"Failed to process file: {str(e)}"}

def handle_website_analysis():
    """Display the website analysis page with enhanced UI."""
    st.header("🌐 Website Analysis")
    st.markdown("Enter a website URL to analyze its content and structure")
    
    # URL input with better styling
    url = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com",
        help="Enter a complete URL including http:// or https://"
    )

    if url:
        # Security validation
        is_valid, message = security_manager.validate_url(url)
        if not is_valid:
            st.error(f"❌ Security validation failed: {message}")
            return
        
        # Show analysis in progress
        with st.spinner("🔄 Analyzing website..."):
            try:
                analysis_results = analyze_website(url)
                display_website_results(analysis_results)
            except Exception as e:
                st.error(f"❌ Website analysis failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_analysis_results(results, filename):
    """Display comprehensive analysis results for all file types."""
    if "error" in results:
        st.error(f"❌ {results['error']}")
        if "suggestions" in results:
            st.info("💡 Suggestions:")
            for suggestion in results["suggestions"]:
                st.write(f"- {suggestion}")
        return

    st.success("✅ Analysis completed successfully!")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Statistics", "🤖 AI Insights", "📉 Visualizations"])
    
    with tab1:
        display_overview_tab(results, filename)
    
    with tab2:
        display_statistics_tab(results)
    
    with tab3:
        display_ai_insights_tab(results)
    
    with tab4:
        display_visualizations_tab(results)

def display_overview_tab(results, filename):
    """Display overview information for all file types."""
    st.subheader(f"📄 File: {filename}")
    
    # Basic info based on file type
    file_ext = filename.lower().split('.')[-1]
    
    # Common metrics for all file types
    col1, col2, col3 = st.columns(3)
    
    if "shape" in results:
        # Spreadsheet/Database metrics
        col1.metric("Rows", results["shape"][0])
        col2.metric("Columns", results["shape"][1])
    if "memory_usage_mb" in results:
        col3.metric("Memory Usage", f"{results['memory_usage_mb']} MB")
    elif "word_count" in results:
        # Text-based file metrics
        col1.metric("Words", results["word_count"])
        col2.metric("Sentences", results.get("sentence_count", 0))
        col3.metric("Paragraphs", results.get("paragraph_count", 0))
    elif "content_length" in results:
        # Web/HTML content metrics
        col1.metric("Characters", results["content_length"])
        col2.metric("Words", len(results.get("text_sample", "").split()))
        if "links_analysis" in results:
            col3.metric("Total Links", results["links_analysis"].get("total_links", 0))
    
    # File-specific information
    if file_ext in ['json', 'yaml', 'yml']:
        display_structured_data_overview(results)
    elif file_ext in ['xml', 'html']:
        display_markup_overview(results)
    elif file_ext == 'pdf':
        display_pdf_overview(results)
    elif file_ext in ['csv', 'xlsx', 'xls']:
        display_spreadsheet_overview(results)
    elif file_ext in ['db', 'sqlite', 'sqlite3']:
        display_database_overview(results)

def display_structured_data_overview(results):
    """Display overview for JSON/YAML files."""
    if "structure" in results:
        st.info("🔍 Data Structure")
        structure = results["structure"]
        
        if isinstance(structure, dict):
            col1, col2 = st.columns(2)
            with col1:
                if "type" in structure:
                    st.write(f"**Type:** {structure['type']}")
                if "keys" in structure:
                    st.write("**Top-level Keys:**")
                    for key in structure["keys"][:5]:
                        st.write(f"- {key}")
            
            with col2:
                if "nested_objects" in structure:
                    st.write(f"**Nested Objects:** {structure['nested_objects']}")
                if "key_count" in structure:
                    st.write(f"**Total Keys:** {structure['key_count']}")
        
        # Show sample if available
        if "sample" in results:
            with st.expander("👀 View Data Sample"):
                st.code(results["sample"], language=results.get("file_type", "").lower())

def display_markup_overview(results):
    """Display overview for XML/HTML files."""
    if "element_counts" in results:
        st.info("🏗️ Document Structure")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Most Common Elements:**")
            for element, count in results["element_counts"].items():
                st.write(f"- {element}: {count}")
        
        with col2:
            if "total_elements" in results:
                st.metric("Total Elements", results["total_elements"])
            if "unique_elements" in results:
                st.metric("Unique Elements", results["unique_elements"])

def display_pdf_overview(results):
    """Display overview for PDF files."""
    if "text_stats" in results:
        st.info("📑 Document Statistics")
        stats = results["text_stats"]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Characters", stats["characters"])
        col2.metric("Unique Words", stats["unique_words"])
        col3.metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")
        
        # Display readability metrics if available
        if "readability" in results:
            st.info("📚 Readability")
            read_col1, read_col2 = st.columns(2)
            read_col1.metric("Flesch Reading Ease", 
                           f"{results['readability'].get('flesch_reading_ease', 0):.1f}")
            read_col2.metric("Est. Reading Time", 
                           f"{results['readability'].get('estimated_reading_time', 0):.1f} min")

def display_spreadsheet_overview(results):
    """Display overview for spreadsheet files."""
    if "columns" in results:
        st.info("📋 Column Information")
        cols_df = pd.DataFrame({
            'Column': results["columns"],
            'Data Type': [results["dtypes"].get(col, 'Unknown') for col in results["columns"]],
            'Null Count': [results.get("null_counts", {}).get(col, 0) for col in results["columns"]],
            'Null %': [f"{results.get('null_percentage', {}).get(col, 0):.1f}%" for col in results["columns"]]
        })
        st.dataframe(cols_df, use_container_width=True)

def display_database_overview(results):
    """Display overview for database files."""
    if "database_info" in results:
        st.info("🗄️ Database Structure")
        st.metric("Total Tables", results["database_info"].get("total_tables", 0))
        
        # Display table information
        for key, value in results.items():
            if key.startswith("table_") and isinstance(value, dict):
                with st.expander(f"📋 {key.replace('table_', '')}"):
                    if "shape" in value:
                        st.write(f"**Rows:** {value['shape'][0]}")
                        st.write(f"**Columns:** {value['shape'][1]}")
                    if "columns" in value:
                        st.write("**Columns:**")
                        for col in value["columns"]:
                            st.write(f"- {col}")

def display_statistics_tab(results):
    """Display statistical information for all file types."""
    # Spreadsheet/Database statistics
    if "summary_stats" in results:
        st.subheader("📊 Summary Statistics")
        stats_df = pd.DataFrame(results["summary_stats"])
        st.dataframe(stats_df, use_container_width=True)
    
        if "correlation" in results and "high_correlations" in results["correlation"]:
            high_corr = results["correlation"]["high_correlations"]
            if high_corr:
                st.subheader("🔗 High Correlations (>0.7)")
                corr_df = pd.DataFrame(list(high_corr.items()), 
                                     columns=['Variables', 'Correlation'])
                st.dataframe(corr_df, use_container_width=True)
    
    # Text-based statistics
    elif "text_stats" in results:
        st.subheader("📝 Text Statistics")
        stats = results["text_stats"]
        
        # Create multiple columns for different metric categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("📊 Basic Metrics")
            st.metric("Characters", stats["characters"])
            st.metric("Words", stats["words"])
            st.metric("Sentences", stats["sentences"])
        
        with col2:
            st.info("📈 Averages")
            st.metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")
            st.metric("Avg Sentence Length", f"{stats['avg_sentence_length']:.1f}")
        
        with col3:
            st.info("📚 Complexity")
            st.metric("Unique Words", stats["unique_words"])
            if "readability" in results:
                st.metric("Reading Ease", 
                         f"{results['readability'].get('flesch_reading_ease', 0):.1f}")
    
    # Structured data statistics
    elif "structure" in results:
        st.subheader("🔍 Structure Analysis")
        if isinstance(results["structure"], dict):
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("📊 Size Metrics")
                if "key_count" in results["structure"]:
                    st.metric("Total Keys", results["structure"]["key_count"])
                if "nested_objects" in results["structure"]:
                    st.metric("Nested Objects", results["structure"]["nested_objects"])
            
            with col2:
                st.info("🏗️ Composition")
                if "type" in results["structure"]:
                    st.metric("Root Type", results["structure"]["type"])
                if "length" in results["structure"]:
                    st.metric("Array Length", results["structure"]["length"])

def display_visualizations_tab(results):
    """Display visualizations for all file types."""
    st.subheader("📉 Data Visualizations")
    
    with st.spinner("🎨 Generating visualizations..."):
        # For spreadsheet/database data
        if "shape" in results and "columns" in results:
            visualizations = generate_visualizations(results)
            
            if visualizations:
                for title, fig in visualizations.items():
                    st.subheader(title.replace('_', ' ').title())
                    st.pyplot(fig, use_container_width=True)
        
        # For text-based content
        elif "text_stats" in results:
            # Word frequency visualization
            if "frequent_words" in results:
                st.info("📊 Word Frequency")
                freq_data = pd.DataFrame(list(results["frequent_words"].items()), 
                                       columns=['Word', 'Frequency'])
                if not freq_data.empty:
                    import altair as alt
                    chart = alt.Chart(freq_data).mark_bar().encode(
                        x=alt.X('Word:N', sort='-y'),
                        y=alt.Y('Frequency:Q'),
                        tooltip=['Word', 'Frequency']
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)
            
            # Text statistics visualization
            stats = results["text_stats"]
            st.info("📈 Text Metrics")
            
            # Create metrics visualization
            metrics_data = pd.DataFrame([
                {"Metric": "Characters", "Value": stats["characters"]},
                {"Metric": "Words", "Value": stats["words"]},
                {"Metric": "Sentences", "Value": stats["sentences"]},
                {"Metric": "Unique Words", "Value": stats["unique_words"]}
            ])
            
            import altair as alt
            metrics_chart = alt.Chart(metrics_data).mark_bar().encode(
                x=alt.X('Metric:N'),
                y=alt.Y('Value:Q'),
                color=alt.Color('Metric:N', legend=None),
                tooltip=['Metric', 'Value']
            ).properties(height=200)
            st.altair_chart(metrics_chart, use_container_width=True)
        
        # For structured data (JSON/YAML)
        elif "structure" in results:
            st.info("🏗️ Structure Visualization")
            if isinstance(results["structure"], dict):
                # Create structure visualization
                structure_data = []
                if "keys" in results["structure"]:
                    for key in results["structure"]["keys"]:
                        structure_data.append({"Element": "Key", "Name": key})
                if "nested_objects" in results["structure"]:
                    structure_data.append({
                        "Element": "Nested Objects",
                        "Count": results["structure"]["nested_objects"]
                    })
                
                if structure_data:
                    import altair as alt
                    structure_df = pd.DataFrame(structure_data)
                    if "Name" in structure_df.columns:
                        chart = alt.Chart(structure_df).mark_bar().encode(
                            x=alt.X('Name:N'),
                            y=alt.Y('count():Q'),
                            tooltip=['Name', 'count()']
                        ).properties(height=200)
                        st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No visualizations available for this data type")

def display_website_results(results):
    """Display comprehensive website analysis results."""
    if "error" in results:
        st.error(f"❌ {results['error']}")
        return
    
    st.success("✅ Website analysis completed!")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Overview",
        "📝 Content",
        "🔗 Links",
        "📊 Analytics",
        "🤖 AI Insights"
    ])
    
    with tab1:
        st.subheader("Website Overview")
        
        # Technical Information
        st.info("🔧 Technical Details")
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        tech_col1.metric("Status Code", results.get("status_code", "Unknown"))
        tech_col2.metric("Encoding", results.get("encoding", "Unknown"))
        tech_col3.metric("Server", results.get("server", "Unknown"))
        
        # Technologies Used
        if results.get('technologies'):
            st.info("💻 Technologies Detected")
            for tech in results['technologies']:
                st.write(f"- {tech}")
        
        # Security Information
        if results.get('security'):
            st.info("🔒 Security Headers")
            for header, value in results['security'].items():
                st.write(f"**{header}:** {value}")
        
        # Metadata
        st.info("📋 Basic Metadata")
        meta_col1, meta_col2 = st.columns(2)
        
        with meta_col1:
            st.write("**Title:**", results.get("title", "No title found"))
            st.write("**Description:**", results.get("description", "No description found"))
            st.write("**Author:**", results.get("author", "No author found"))
            st.write("**Language:**", results.get("language", "Unknown"))
        
        with meta_col2:
            st.write("**Keywords:**", results.get("keywords", "No keywords found"))
            if results.get("favicon"):
                st.write("**Favicon:**", f"[![Favicon]({results['favicon']})]({results['favicon']})")
        
        # Social Media Metadata
        if results.get("og_data") or results.get("twitter_data"):
            st.info("🌐 Social Media Metadata")
            
            social_col1, social_col2 = st.columns(2)
            
            with social_col1:
                if results.get("og_data"):
                    st.write("**OpenGraph Data:**")
                    for key, value in results["og_data"].items():
                        if key == "image" and value:
                            try:
                                st.image(value, width=200, caption="OG Image")
                            except Exception:
                                st.write(f"**{key.title()}:** {value}")
                        else:
                            st.write(f"**{key.title()}:** {value}")
            
            with social_col2:
                if results.get("twitter_data"):
                    st.write("**Twitter Card Data:**")
                    for key, value in results["twitter_data"].items():
                        if key == "image" and value:
                            try:
                                st.image(value, width=200, caption="Twitter Image")
                            except Exception:
                                st.write(f"**{key.title()}:** {value}")
                        else:
                            st.write(f"**{key.title()}:** {value}")
        
        # Schema.org Data
        if results.get("schema_data"):
            with st.expander("🔍 Schema.org Data"):
                st.json(results["schema_data"])
    
    with tab2:
        st.subheader("Content Analysis")
        
        # Content Metrics
        st.info("📊 Content Overview")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        metrics_col1.metric("Words", results.get("word_count", 0))
        metrics_col2.metric("Sentences", results.get("sentence_count", 0))
        metrics_col3.metric("Paragraphs", results.get("paragraph_count", 0))
        metrics_col4.metric("Text Density", f"{results.get('text_density', 0):.1%}")
        
        # Heading Structure
        if results.get('headings'):
            st.info("📑 Document Structure")
            headings = results['headings']
            
            # Heading counts
            head_col1, head_col2, head_col3 = st.columns(3)
            head_col1.metric("Total Headings", headings['count'].get('total', 0))
            head_col2.metric("Main Headings (H1)", headings['count'].get('h1', 0))
            head_col3.metric("Subheadings", 
                           sum(headings['count'].get(f'h{i}', 0) for i in range(2, 7)))
            
            # Show heading hierarchy
            with st.expander("🔍 View Heading Hierarchy"):
                for level in range(1, 7):
                    tag = f'h{level}'
                    if headings['structure'].get(tag):
                        st.write(f"**{tag.upper()}** ({len(headings['structure'][tag])})")
                        for heading in headings['structure'][tag]:
                            st.write("  " * (level-1) + "•", heading)
        
        # Content Structure
        if results.get('content_structure'):
            st.info("🏗️ Page Elements")
            structure = results['content_structure']
            
            # Interactive elements
            inter_col1, inter_col2, inter_col3 = st.columns(3)
            inter_col1.metric("Forms", structure.get('forms', 0))
            inter_col2.metric("Buttons", structure.get('interactive', {}).get('buttons', 0))
            inter_col3.metric("Input Fields", structure.get('interactive', {}).get('inputs', 0))
            
            # Media elements
            media_col1, media_col2 = st.columns(2)
            media_col1.metric("Images", structure.get('images', 0))
            media_col2.metric("Videos", structure.get('videos', 0))
            
            # Lists and Tables
            list_col1, list_col2, list_col3 = st.columns(3)
            list_col1.metric("Ordered Lists", structure['lists'].get('ordered', 0))
            list_col2.metric("Unordered Lists", structure['lists'].get('unordered', 0))
            list_col3.metric("Tables", structure.get('tables', 0))
            
            # Content composition
            if structure.get('composition'):
                st.write("\n**Content Composition:**")
                comp = structure['composition']
                st.write(f"- {'✓' if comp.get('text_heavy') else '✗'} Text Heavy")
                st.write(f"- {'✓' if comp.get('media_rich') else '✗'} Media Rich")
                st.write(f"- {'✓' if comp.get('interactive') else '✗'} Interactive")
        
        # Language Analysis
        if results.get('language_metrics'):
            st.info("📝 Language Analysis")
            lang = results['language_metrics']
            
            # Text Patterns
            patterns_col1, patterns_col2, patterns_col3 = st.columns(3)
            if lang.get('patterns'):
                patterns = lang['patterns']
                patterns_col1.metric("Questions", patterns.get('questions', 0))
                patterns_col2.metric("Exclamations", patterns.get('exclamations', 0))
                patterns_col3.metric("Numbers", patterns.get('numbers', 0))
            
            # Word Complexity
            if lang.get('complexity'):
                complexity = lang['complexity']
                comp_col1, comp_col2 = st.columns(2)
                comp_col1.metric("Avg Word Length", 
                               f"{complexity.get('avg_word_length', 0):.1f}")
                comp_col2.metric("Unique Words", 
                               f"{complexity.get('unique_words_ratio', 0):.1%}")
                
                # Word length distribution
                st.write("\n**Word Length Distribution:**")
                st.write(f"- Long Words (>6 chars): {complexity.get('long_words_ratio', 0):.1%}")
                st.write(f"- Very Long Words (>10 chars): {complexity.get('very_long_words_ratio', 0):.1%}")
        
        # Readability
        if results.get('readability'):
            st.info("📚 Readability Metrics")
            read_col1, read_col2 = st.columns(2)
            read_col1.metric("Flesch Reading Ease", 
                           f"{results['readability'].get('flesch_reading_ease', 0):.1f}")
            read_col2.metric("Est. Reading Time", 
                           f"{results['readability'].get('estimated_reading_time', 0):.1f} min")
            
            # Interpret Flesch score
            flesch_score = results['readability'].get('flesch_reading_ease', 0)
            if flesch_score > 90:
                st.success("Very Easy to Read - Suitable for 5th graders")
            elif flesch_score > 80:
                st.success("Easy to Read - Suitable for 6th graders")
            elif flesch_score > 70:
                st.info("Fairly Easy to Read - Suitable for 7th graders")
            elif flesch_score > 60:
                st.info("Standard - Suitable for 8th & 9th graders")
            elif flesch_score > 50:
                st.warning("Fairly Difficult - Suitable for 10th to 12th graders")
            elif flesch_score > 30:
                st.warning("Difficult - Suitable for college students")
            else:
                st.error("Very Difficult - Suitable for college graduates")
        
        # Word Frequency
        if results.get('frequent_words'):
            st.info("📊 Most Common Words")
            freq_data = pd.DataFrame(list(results["frequent_words"].items()), 
                                   columns=['Word', 'Frequency'])
            if not freq_data.empty:
                # Use Altair for better chart handling
                import altair as alt
                chart = alt.Chart(freq_data).mark_bar().encode(
                    x=alt.X('Word:N', sort='-y'),
                    y=alt.Y('Frequency:Q'),
                    tooltip=['Word', 'Frequency']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
        
        # Content Sample
        if results.get('text_sample'):
            with st.expander("📄 View Content Sample"):
                st.text_area("First portion of content:", 
                            value=results["text_sample"],
                            height=200,
                            disabled=True)
    
    with tab3:
        st.subheader("Links Analysis")
        
        if results.get('links_analysis'):
            links = results['links_analysis']
            
            # Link Overview
            st.info("🔗 Link Overview")
            link_col1, link_col2, link_col3, link_col4 = st.columns(4)
            link_col1.metric("Total Links", links.get("total_links", 0))
            link_col2.metric("Internal Links", links.get("internal_links", 0))
            link_col3.metric("External Links", links.get("external_links", 0))
            link_col4.metric("Social Links", links.get("social_links", 0))
            
            # Link Distribution Chart
            st.info("📊 Link Distribution")
            data = {
                'Category': ['Internal', 'External', 'Social', 'Resources'],
                'Count': [
                    links.get("internal_links", 0),
                    links.get("external_links", 0),
                    links.get("social_links", 0),
                    links.get("resource_links", 0)
                ]
            }
            df = pd.DataFrame(data)
            df = df[df['Count'] > 0]  # Filter out zero values
            
            if not df.empty:
                # Use Altair for pie chart
                import altair as alt
                pie = alt.Chart(df).mark_arc().encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(field="Category", type="nominal"),
                    tooltip=['Category', 'Count']
                ).properties(width=400, height=400)
                st.altair_chart(pie, use_container_width=True)
            
            # Unique Domains
            if links.get("unique_domains"):
                st.info("🌐 Connected Domains")
                domains = links["unique_domains"]
                if isinstance(domains, (list, set)) and domains:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"Connected to {len(domains)} unique domains:")
                        for domain in sorted(domains):
                            st.write(f"- {domain}")
                    with col2:
                        # Create a small chart showing top domains
                        domain_counts = {}
                        for domain in domains:
                            tld = domain.split('.')[-1]
                            domain_counts[tld] = domain_counts.get(tld, 0) + 1
                        
                        # Show top TLDs chart
                        tld_data = pd.DataFrame(list(domain_counts.items()), 
                                              columns=['TLD', 'Count'])
                        tld_data = tld_data.sort_values('Count', ascending=False).head(5)
                        
                        if not tld_data.empty:
                            tld_chart = alt.Chart(tld_data).mark_bar().encode(
                                x=alt.X('TLD:N', sort='-y'),
                                y='Count:Q',
                                tooltip=['TLD', 'Count']
                            ).properties(height=200)
                            st.altair_chart(tld_chart, use_container_width=True)
            
            # Resource Links
            if links.get("resource_links", 0) > 0:
                st.info("📁 Downloadable Resources")
                st.metric("Resource Files", links["resource_links"])
    
    with tab4:
        st.subheader("Analytics & Insights")
        
        # Content Quality Metrics
        if results.get('readability') and results.get('content_structure'):
            st.info("📊 Content Quality Score")
            
            # Calculate quality scores
            readability_score = min(100, results['readability'].get('flesch_reading_ease', 0))
            
            structure = results.get('content_structure', {})
            has_headings = results.get('headings', {}).get('count', {}).get('total', 0) > 0
            has_images = structure.get('images', 0) > 0
            has_lists = sum(structure.get('lists', {}).values()) > 0
            
            structure_score = sum([
                20 if has_headings else 0,
                20 if has_images else 0,
                20 if has_lists else 0,
                20 if structure.get('tables', 0) > 0 else 0,
                20 if structure.get('interactive', {}).get('buttons', 0) > 0 else 0
            ])
            
            # Display scores
            score_col1, score_col2 = st.columns(2)
            score_col1.metric("Readability Score", f"{readability_score:.0f}/100")
            score_col2.metric("Structure Score", f"{structure_score:.0f}/100")
            
            # Overall quality gauge chart
            overall_score = (readability_score + structure_score) / 2
            quality_color = (
                "🟢" if overall_score >= 80 else
                "🟡" if overall_score >= 60 else
                "🔴"
            )
            st.write(f"\n**Overall Quality: {quality_color} {overall_score:.0f}/100**")
            
            # Quality breakdown
            st.write("\n**Quality Checklist:**")
            st.write(f"- {'✓' if has_headings else '✗'} Proper heading structure")
            st.write(f"- {'✓' if has_images else '✗'} Contains images")
            st.write(f"- {'✓' if has_lists else '✗'} Uses lists for content organization")
            st.write(f"- {'✓' if readability_score > 60 else '✗'} Easy to read")
            st.write(f"- {'✓' if structure.get('interactive', {}).get('buttons', 0) > 0 else '✗'} Interactive elements")
        
        # Content Patterns
        if results.get('language_metrics'):
            st.info("📈 Content Patterns")
            lang = results['language_metrics']
            
            if lang.get('patterns'):
                patterns = lang['patterns']
                
                # Create pattern metrics
                pattern_col1, pattern_col2, pattern_col3 = st.columns(3)
                pattern_col1.metric("Questions", patterns.get('questions', 0))
                pattern_col2.metric("URLs", patterns.get('urls', 0))
                pattern_col3.metric("Email Addresses", patterns.get('emails', 0))
                
                # Create pattern visualization
                pattern_data = pd.DataFrame([
                    {"Type": k, "Count": v}
                    for k, v in patterns.items()
                    if v > 0
                ])
                
                if not pattern_data.empty:
                    pattern_chart = alt.Chart(pattern_data).mark_bar().encode(
                        x=alt.X('Type:N', sort='-y'),
                        y='Count:Q',
                        tooltip=['Type', 'Count']
                    ).properties(height=200)
                    st.altair_chart(pattern_chart, use_container_width=True)
        
        # Word Complexity Analysis
        if results.get('language_metrics', {}).get('complexity'):
            st.info("📚 Word Complexity Analysis")
            complexity = results['language_metrics']['complexity']
            
            # Create complexity metrics
            comp_col1, comp_col2 = st.columns(2)
            comp_col1.metric("Average Word Length", 
                           f"{complexity.get('avg_word_length', 0):.1f} chars")
            comp_col2.metric("Unique Words", 
                           f"{complexity.get('unique_words_ratio', 0):.1%}")
            
            # Create word length distribution chart
            dist_data = pd.DataFrame([
                {"Category": "Short Words", 
                 "Percentage": 1 - complexity.get('long_words_ratio', 0)},
                {"Category": "Long Words (>6 chars)", 
                 "Percentage": complexity.get('long_words_ratio', 0) - 
                             complexity.get('very_long_words_ratio', 0)},
                {"Category": "Very Long Words (>10 chars)", 
                 "Percentage": complexity.get('very_long_words_ratio', 0)}
            ])
            
            dist_chart = alt.Chart(dist_data).mark_bar().encode(
                x=alt.X('Category:N'),
                y=alt.Y('Percentage:Q', axis=alt.Axis(format='%')),
                tooltip=['Category', alt.Tooltip('Percentage:Q', format='.1%')]
            ).properties(height=200)
            st.altair_chart(dist_chart, use_container_width=True)
    
    with tab5:
        st.subheader("AI Analysis")
        if results.get('ai_analysis'):
            st.info("🤖 AI-Generated Insights")
            st.markdown(results["ai_analysis"])
            
            # Add AI-generated recommendations
            st.info("💡 Recommendations")
            
            # Content recommendations
            if results.get('readability'):
                flesch_score = results['readability'].get('flesch_reading_ease', 0)
                if flesch_score < 60:
                    st.write("- Consider simplifying the language for better readability")
                if flesch_score < 30:
                    st.write("- The content might be too complex for general audience")
            
            # Structure recommendations
            if results.get('headings'):
                if not results['headings'].get('has_proper_structure'):
                    st.write("- Consider adding a clear H1 heading for better SEO")
                if results['headings']['count'].get('total', 0) < 3:
                    st.write("- Add more headings to better organize the content")
            
            # Link recommendations
            if results.get('links_analysis'):
                links = results['links_analysis']
                if links.get('internal_links', 0) < links.get('external_links', 0):
                    st.write("- Consider adding more internal links for better site navigation")
                if links.get('social_links', 0) == 0:
                    st.write("- Consider adding social media links for better engagement")
            
            # SEO recommendations
            if not results.get('description') or results.get('description') == "No description found":
                st.write("- Add a meta description for better SEO")
            if not results.get('keywords') or results.get('keywords') == "No keywords found":
                st.write("- Consider adding meta keywords for better indexing")
        else:
            st.warning("AI analysis not available for this website.")

def display_ai_insights_tab(results):
    """Display AI-generated insights for all file types."""
    if "ai_analysis" in results:
        st.subheader("🤖 AI Analysis")
        st.markdown(results["ai_analysis"])
        
        # Add file-type specific insights
        if "text_stats" in results:
            st.info("📝 Content Insights")
            stats = results["text_stats"]
            
            # Generate insights based on metrics
            insights = []
            if stats["avg_word_length"] > 6:
                insights.append("The text contains many complex words")
            if stats["avg_sentence_length"] > 20:
                insights.append("Sentences are generally long")
            if stats["unique_words"] / stats["words"] > 0.5:
                insights.append("The vocabulary is quite diverse")
            
            for insight in insights:
                st.write(f"- {insight}")
        
        elif "shape" in results:
            st.info("📊 Data Insights")
            if "null_percentage" in results:
                null_cols = sum(1 for v in results["null_percentage"].values() if v > 0)
                if null_cols > 0:
                    st.write(f"- {null_cols} columns contain missing values")
            
            if "correlation" in results and "high_correlations" in results["correlation"]:
                corr_count = len(results["correlation"]["high_correlations"])
                if corr_count > 0:
                    st.write(f"- Found {corr_count} strong correlations between variables")
    else:
        st.info("AI analysis not available for this file type")

def text_analyzer():
    """Display the text analyzer interface with comprehensive analysis features."""
    st.header("📝 Text Analysis")
    st.markdown("Analyze text content for insights, sentiment, and key information")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Direct Text Input", "Text File Upload"],
        horizontal=True,
        help="Choose how you want to input your text for analysis"
    )
    
    text_content = None
    
    if input_method == "Direct Text Input":
        # Add a clear description
        st.markdown("#### Enter Text for Analysis")
        st.markdown("Type or paste your text below and click 'Analyze' to get insights.")
        
        # Add placeholder text with example
        placeholder_text = (
            "Example: I'm really excited about this new project! "
            "The team has been working hard, and we've made significant progress. "
            "However, there are still some challenges we need to address."
        )
        
        text_content = st.text_area(
            "Text Input",
            height=200,
            placeholder=placeholder_text,
            help="Enter at least 10 words for better analysis"
        )
        
        # Add character count and input validation
        if text_content:
            char_count = len(text_content)
            word_count = len(text_content.split())
            
            # Show input statistics
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📝 Characters: {char_count}")
            with col2:
                st.info(f"📚 Words: {word_count}")
            
            # Input validation
            if word_count < 3:
                st.warning("⚠️ Please enter more text for meaningful analysis (at least 3 words)")
                text_content = None
        
    else:  # Text File Upload
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt', 'md'],
            help="Upload a text file for analysis. Supported formats: TXT, MD"
        )
        
        if uploaded_file:
            try:
                # Security validation
                is_valid, message = security_manager.validate_file(uploaded_file)
                if not is_valid:
                    st.error(f"❌ Security validation failed: {message}")
                    return
                    
                text_content = uploaded_file.getvalue().decode('utf-8')
                # Sanitize file content
                text_content = security_manager.sanitize_text(text_content)
                
                # Show file info
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                st.info(f"📝 File size: {len(text_content):,} characters")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
    
    if text_content and text_content.strip():
        # Analysis options
        st.markdown("---")
        st.subheader("Analysis Options")
        
        # Use columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            do_sentiment = st.checkbox(
                "Sentiment Analysis", 
                value=True,
                help="Analyze the emotional tone and sentiment of the text"
            )
        with col2:
            do_keywords = st.checkbox(
                "Keyword Extraction", 
                value=True,
                help="Extract key terms, phrases, and topics"
            )
        with col3:
            do_summary = st.checkbox(
                "Text Summarization", 
                value=True,
                help="Generate a summary and analyze text complexity"
            )
        
        # Analysis button with loading state
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing your text..."):
                # Add progress indication
                progress_bar = st.progress(0)
                
                # Basic text statistics
                progress_bar.progress(20)
                stats = analyze_text_stats(text_content)
                
                # Display basic metrics
                progress_bar.progress(40)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Words", stats["word_count"])
                col2.metric("Sentences", stats["sentence_count"])
                col3.metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")
                col4.metric("Reading Time", f"{stats['reading_time']:.1f} min")
                
                # Create tabs for different analyses
                progress_bar.progress(60)
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Overview",
                    "😊 Sentiment",
                    "🔑 Keywords",
                    "📝 Summary"
                ])
                
                with tab1:
                    display_text_overview(stats, text_content)
                
                progress_bar.progress(70)
                with tab2:
                    if do_sentiment:
                        display_sentiment_analysis(text_content)
                
                progress_bar.progress(80)
                with tab3:
                    if do_keywords:
                        display_keyword_analysis(text_content)
                
                progress_bar.progress(90)
                with tab4:
                    if do_summary:
                        display_text_summary(text_content)
                
                # Complete the progress bar
                progress_bar.progress(100)
                st.success("✨ Analysis completed!")
                
                # Add a divider
                st.markdown("---")
                
                # Add helpful tip
                st.info(
                    "💡 **Tip:** Check all tabs above for detailed analysis results. "
                    "Each tab provides different insights about your text."
                )
        else:
            # Show instruction when no analysis is running
            st.info("👆 Click 'Analyze Text' to start the analysis")

def analyze_text_stats(text):
    """Analyze basic text statistics."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "char_count": len(text),
        "avg_word_length": sum(len(word) for word in words) / max(len(words), 1),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "reading_time": len(words) / 200,  # Assuming 200 words per minute
        "unique_words": len(set(words)),
        "unique_ratio": len(set(words)) / max(len(words), 1)
    }

def display_text_overview(stats, text):
    """Display comprehensive text overview."""
    st.subheader("Text Overview")
    
    # Combine these columns into a single info section
    st.info("📊 Text Statistics")
    col1, col2 = st.columns(2)  # Reduce from 4 columns to 2
    
    with col1:
        st.write(f"**Characters:** {stats['char_count']}")
        st.write(f"**Words:** {stats['word_count']}")
        st.write(f"**Sentences:** {stats['sentence_count']}")
        
    with col2:
        st.write(f"**Unique Words:** {stats['unique_words']}")
        st.write(f"**Avg Sentence Length:** {stats['avg_sentence_length']:.1f} words")
        st.write(f"**Vocabulary Diversity:** {stats['unique_ratio']:.1%}")
        
    with col2:
        st.info("📈 Word Length Distribution")
        words = text.split()
        lengths = [len(word) for word in words]
        
        if lengths:
            import altair as alt
            df = pd.DataFrame({'length': lengths})
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('length:Q', bin=True, title='Word Length'),
                y=alt.Y('count()', title='Frequency')
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)

def display_sentiment_analysis(text):
    """Display sentiment analysis using TextBlob with enhanced accuracy."""
    st.subheader("Sentiment Analysis")
    
    try:
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get overall sentiment
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Enhanced sentiment categorization with more nuanced thresholds
        if polarity > 0.3:
            sentiment = "positive"
            confidence = min(abs(polarity) * 100, 100)
        elif polarity < -0.1:  # More sensitive to negative sentiment
            sentiment = "negative"
            confidence = min(abs(polarity) * 100, 100)
        else:
            # Check for negative indicators in the text
            text_lower = text.lower()
            if any(indicator in text_lower for indicator in NEGATIVE_INDICATORS):
                sentiment = "negative"
                confidence = 70  # Moderate confidence for pattern-based detection
            else:
                sentiment = "neutral"
                confidence = 60
        
        # Display results with visual elements
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment indicator
            sentiment_color = {
                'positive': '🟢',
                'negative': '🔴',
                'neutral': '🟡'
            }.get(sentiment, '🟡')
            
            st.metric("Overall Sentiment", 
                     f"{sentiment_color} {sentiment.title()}", 
                     f"Confidence: {confidence:.1f}%")
            
            # Subjectivity interpretation
            subjectivity_label = (
                "Very Objective" if subjectivity < 0.2 else
                "Somewhat Objective" if subjectivity < 0.4 else
                "Mixed" if subjectivity < 0.6 else
                "Somewhat Subjective" if subjectivity < 0.8 else
                "Very Subjective"
            )
            
            st.metric("Subjectivity", 
                     subjectivity_label,
                     f"{subjectivity:.1%}")
        
        with col2:
            # Analyze sentence-level sentiment with context
            sentences = blob.sentences
            sentence_sentiments = []
            
            for sent in sentences[:5]:  # Analyze up to 5 sentences
                sent_polarity = sent.sentiment.polarity
                
                # Check for negation in the sentence
                sent_text = str(sent).lower()
                has_negation = any(neg in sent_text for neg in NEGATIVE_INDICATORS)
                
                # Adjust polarity based on negation
                if has_negation and sent_polarity >= 0:
                    sent_polarity = -abs(sent_polarity) - 0.1  # Make it negative
                
                sentence_sentiments.append((str(sent), sent_polarity))
            
            st.write("**Key Phrases:**")
            for sent, sent_polarity in sentence_sentiments:
                if abs(sent_polarity) > 0.1:  # Show more subtle sentiments
                    emoji = "📈" if sent_polarity > 0.1 else "📉"
                    intensity = (
                        "Strong" if abs(sent_polarity) > 0.5 else
                        "Moderate" if abs(sent_polarity) > 0.3 else
                        "Slight"
                    )
                    st.write(f"{emoji} *{intensity} {sent_polarity > 0 and 'Positive' or 'Negative'}: {sent}*")
        
        # Detailed analysis
        st.markdown("#### Detailed Analysis")
        
        # Generate more nuanced analysis
        analysis_points = []
        
        # Overall sentiment analysis
        if sentiment == "negative":
            analysis_points.append(f"The text expresses {intensity.lower()} negative sentiment overall.")
        elif sentiment == "positive":
            analysis_points.append(f"The text expresses {intensity.lower()} positive sentiment overall.")
        else:
            analysis_points.append("The text expresses a neutral or mixed sentiment overall.")
        
        # Subjectivity analysis
        analysis_points.append(f"The content is {subjectivity_label.lower()} in nature, "
                             f"indicating {'more personal opinions and emotions' if subjectivity > 0.5 else 'more factual and objective content'}.")
        
        # Emotional indicators
        if abs(polarity) > 0.5:
            analysis_points.append("Strong emotional indicators are present in the text.")
        elif abs(polarity) > 0.3:
            analysis_points.append("Moderate emotional indicators are present in the text.")
        else:
            analysis_points.append("The emotional tone is subtle or neutral.")
        
        # Context analysis
        if any(sent_polarity < -0.1 for _, sent_polarity in sentence_sentiments):
            analysis_points.append("Some negative sentiments were detected in specific phrases.")
        
        st.write(" ".join(analysis_points))
                
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        logger.error(f"Sentiment analysis error: {str(e)}")

def display_keyword_analysis(text):
    """Display keyword and key phrase extraction using NLTK."""
    st.subheader("Key Terms and Phrases")
    
    try:
        # Tokenize and clean text
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english') + list(punctuation))
        words = [word for word in words if word not in stop_words and word.isalnum()]
        
        # Get word frequency
        word_freq = Counter(words)
        
        # Get named entities with error handling
        entities = {}
        try:
            # Check if required resources are available
            nltk.data.find('chunkers/maxent_ne_chunker_tab/english_ace_multiclass/')
            chunks = ne_chunk(pos_tag(word_tokenize(text)))
            entities = {
                'PERSON': [],
                'ORGANIZATION': [],
                'GPE': []  # Geo-Political Entities
            }
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    if chunk.label() in entities:
                        entities[chunk.label()].append(' '.join([c[0] for c in chunk]))
        except (LookupError, Exception) as ner_error:
            # Use alternative approach for named entities
            tagged_words = pos_tag(word_tokenize(text))
            entities = {
                'NAMES': [],
                'ORGANIZATIONS': [],
                'LOCATIONS': []
            }
            
            # Simple rule-based NER
            for i, (word, tag) in enumerate(tagged_words):
                if tag.startswith('NNP'):  # Proper noun
                    if i > 0 and tagged_words[i-1][1].startswith('NNP'):
                        # Combine consecutive proper nouns
                        entities['NAMES'].append(f"{tagged_words[i-1][0]} {word}")
                    else:
                        entities['NAMES'].append(word)
            
            logger.debug("Using alternative NER approach")
        
        # Get key phrases (using POS patterns)
        tagged = pos_tag(word_tokenize(text))
        phrases = []
        for i in range(len(tagged)-1):
            if (tagged[i][1].startswith('JJ') and tagged[i+1][1].startswith('NN')) or \
               (tagged[i][1].startswith('NN') and tagged[i+1][1].startswith('NN')):
                phrases.append(f"{tagged[i][0]} {tagged[i+1][0]}")
        
        # Display results in organized sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("🎯 Main Keywords")
            top_words = word_freq.most_common(7)
            for word, count in top_words:
                st.write(f"• {word} ({count})")
            
            st.info("🔤 Key Phrases")
            for phrase in phrases[:5]:
                st.write(f"• {phrase}")
        
        with col2:
            if entities:
                st.info("📍 Named Entities")
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        st.write(f"**{entity_type}:**")
                        for entity in set(entity_list):
                            st.write(f"• {entity}")
            
            st.info("📊 Word Frequency")
            freq_data = pd.DataFrame(top_words, columns=['word', 'count'])
            if not freq_data.empty:
                import altair as alt
                chart = alt.Chart(freq_data).mark_bar().encode(
                    x=alt.X('count:Q', title='Frequency'),
                    y=alt.Y('word:N', sort='-x', title='Word'),
                    tooltip=['word', 'count']
                ).properties(height=min(len(freq_data) * 25, 200))
                st.altair_chart(chart, use_container_width=True)
        
        # Topics section (using simple clustering of frequent words)
        topics = []
        for word, _ in word_freq.most_common(10):
            related_words = [w for w, c in word_freq.items() 
                           if w != word and (w.startswith(word) or word.startswith(w))]
            if related_words:
                topics.append(f"{word} ({', '.join(related_words[:2])})")
        
        if topics:
            st.info("📚 Main Topics")
            cols = st.columns(min(len(topics), 3))
            for i, topic in enumerate(topics[:3]):
                cols[i].markdown(f"**{topic}**")
                
    except Exception as e:
        st.error(f"Error in keyword extraction: {str(e)}")
        logger.error(f"Keyword extraction error: {str(e)}")

def display_text_summary(text):
    """Display text summarization using extractive summarization."""
    st.subheader("Text Summary")
    
    try:
        # Tokenize the text
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english') + list(punctuation))
        words = [word for word in words if word not in stop_words and word.isalnum()]
        
        # Calculate word frequency
        word_freq = Counter(words)
        
        # Calculate sentence scores based on word frequency
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_freq[word]
                    else:
                        sentence_scores[sentence] += word_freq[word]
        
        # Get summary sentences
        summary_sentences = sorted(
            sentence_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        # Analyze complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)
        unique_words_ratio = len(set(words)) / len(words)
        
        if avg_word_length > 6 or avg_sentence_length > 20:
            complexity = "ADVANCED"
        elif avg_word_length > 5 or avg_sentence_length > 15:
            complexity = "INTERMEDIATE"
        else:
            complexity = "BASIC"
        
        # Display results
        st.info("📝 Summary")
        summary_text = " ".join(sent for sent, _ in summary_sentences)
        st.write(summary_text)
        
        # Main points and key elements
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("🎯 Key Elements")
            points = [
                f"Contains {len(sentences)} sentences",
                f"Uses {len(set(words))} unique words",
                f"Average sentence length: {avg_sentence_length:.1f} words"
            ]
            for point in points:
                st.write(f"• {point}")
        
        with col2:
            st.info("💡 Content Analysis")
            takeaways = [
                f"Vocabulary diversity: {unique_words_ratio:.1%}",
                f"Average word length: {avg_word_length:.1f} characters",
                f"Text complexity: {complexity.title()}"
            ]
            for takeaway in takeaways:
                st.write(f"• {takeaway}")
        
        # Style analysis
        st.info("✍️ Writing Style")
        style_col1, style_col2 = st.columns(2)
        
        with style_col1:
            # Analyze writing style
            style_analysis = []
            if avg_sentence_length > 20:
                style_analysis.append("Uses complex, detailed sentences")
            elif avg_sentence_length < 10:
                style_analysis.append("Uses concise, direct sentences")
            else:
                style_analysis.append("Uses balanced sentence structure")
            
            if unique_words_ratio > 0.8:
                style_analysis.append("Rich, varied vocabulary")
            elif unique_words_ratio < 0.4:
                style_analysis.append("Consistent, focused vocabulary")
            else:
                style_analysis.append("Moderate vocabulary range")
            
            st.write(" ".join(style_analysis))
        
        with style_col2:
            complexity_icon = {
                'BASIC': '🟢',
                'INTERMEDIATE': '🟡',
                'ADVANCED': '🔴'
            }.get(complexity, '⚪')
            st.write(f"**Complexity Level:** {complexity_icon} {complexity}")
            
            # Determine target audience based on complexity
            audience = {
                'BASIC': "General audience",
                'INTERMEDIATE': "Informed readers",
                'ADVANCED': "Expert/Academic audience"
            }.get(complexity)
            st.write(f"**Target Audience:** {audience}")
                
    except Exception as e:
        st.error(f"Error in text summarization: {str(e)}")

if __name__ == "__main__":
    main() 