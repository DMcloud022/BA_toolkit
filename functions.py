import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import sqlite3
from io import StringIO, BytesIO
import docx2txt
from pypdf import PdfReader
from typing import Dict, Any, Optional, List
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import config
import chardet
import json
import xml.etree.ElementTree as ET
import pyarrow.parquet as pq
import yaml
from groq import Groq
import warnings
from pathlib import Path
import logging
import hashlib
from datetime import datetime, timedelta
from functools import lru_cache
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

__all__ = [
    'analyze_website',
    'analyze_document',
    'analyze_excel',
    'analyze_database',
    'analyze_json',
    'analyze_xml',
    'analyze_parquet',
    'analyze_yaml',
    'analyze_text',
    'generate_visualizations'
]

def initialize_groq_client():
    """Initialize Groq client with better error handling."""
    try:
        if config.GROQ_API_KEY:
            return Groq(api_key=config.GROQ_API_KEY)
        else:
            logger.warning("Groq API key not found. AI analysis will be limited.")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return None

# Initialize the Groq client
client = initialize_groq_client()

def safe_groq_request(prompt: str, max_retries: int = 3) -> str:
    """Safely make a request to Groq API with retries and fallback analysis."""
    if not client:
        # Provide basic analysis without API
        return generate_basic_analysis(prompt)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=config.DEFAULT_MODEL,
                max_tokens=config.MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq API attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Fallback to basic analysis on final retry
                return generate_basic_analysis(prompt)
    
    return generate_basic_analysis(prompt)

def generate_basic_analysis(prompt: str) -> str:
    """Generate basic analysis without using AI API."""
    try:
        # Extract key metrics from the prompt
        metrics = extract_metrics_from_prompt(prompt)
        
        # Generate insights based on available metrics
        insights = []
        
        if "readability_score" in metrics:
            score = metrics["readability_score"]
            if score > 60:
                insights.append("The content has good readability.")
            else:
                insights.append("The content might be difficult to read for some audiences.")
        
        if "word_count" in metrics:
            count = metrics["word_count"]
            if count < 300:
                insights.append("The content is relatively short.")
            elif count > 1000:
                insights.append("The content is comprehensive.")
        
        if "headings" in metrics and metrics["headings"] > 0:
            insights.append("The content is well-structured with proper headings.")
        
        if "links" in metrics and metrics["links"] > 0:
            insights.append("The content contains useful links for navigation and references.")
        
        # Add general recommendations
        recommendations = [
            "Consider reviewing the content for clarity and conciseness.",
            "Ensure the content structure follows web accessibility guidelines.",
            "Regular content updates can help maintain relevance.",
            "Consider adding more visual elements if appropriate."
        ]
        
        # Combine insights and recommendations
        analysis = "Content Analysis:\n\n"
        analysis += "Key Insights:\n- " + "\n- ".join(insights) + "\n\n"
        analysis += "Recommendations:\n- " + "\n- ".join(recommendations)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error generating basic analysis: {e}")
        return "Basic content analysis is currently unavailable."

def extract_metrics_from_prompt(prompt: str) -> Dict[str, Any]:
    """Extract metrics from the analysis prompt."""
    metrics = {}
    
    try:
        # Extract word count
        word_count_match = re.search(r"word_count\":\s*(\d+)", prompt)
        if word_count_match:
            metrics["word_count"] = int(word_count_match.group(1))
        
        # Extract readability score
        readability_match = re.search(r"flesch_reading_ease\":\s*([\d.]+)", prompt)
        if readability_match:
            metrics["readability_score"] = float(readability_match.group(1))
        
        # Extract heading count
        headings_match = re.search(r"total\":\s*(\d+)", prompt)
        if headings_match:
            metrics["headings"] = int(headings_match.group(1))
        
        # Extract link count
        links_match = re.search(r"total_links\":\s*(\d+)", prompt)
        if links_match:
            metrics["links"] = int(links_match.group(1))
            
    except Exception as e:
        logger.warning(f"Error extracting metrics from prompt: {e}")
    
    return metrics

def get_cache_key(url: str) -> str:
    """Generate a cache key for a URL."""
    return hashlib.md5(url.encode()).hexdigest()

def analyze_website(url: str) -> Dict[str, Any]:
    """Analyze website content with comprehensive error handling and robust parsing."""
    try:
        # Add timeout and headers for better reliability
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Use session for better connection handling
        with requests.Session() as session:
            session.headers.update(headers)
            response = session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Detect encoding
            if response.encoding is None:
                response.encoding = response.apparent_encoding
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Basic metadata extraction
            metadata = extract_metadata(soup, url)
            
            # Content analysis
            content_analysis = analyze_content(soup)
            
            # Links analysis
            links_analysis = analyze_links(soup, url)
            
            # Combine all analyses
            analysis_results = {
                **metadata,
                **content_analysis,
                **links_analysis,
                "url": url,
                "status_code": response.status_code,
                "encoding": response.encoding,
                "content_type": response.headers.get('content-type', 'Unknown'),
                "server": response.headers.get('server', 'Unknown'),
            }
            
            # Generate AI analysis prompt
            ai_prompt = f"""Analyze this website content and provide insights about:
1. Content Quality and Structure
2. SEO and Metadata
3. User Experience
4. Technical Implementation
5. Recommendations for Improvement

Website URL: {url}
Content Stats:
- Words: {content_analysis.get('word_count', 0)}
- Readability Score: {content_analysis.get('readability', {}).get('flesch_reading_ease', 0)}
- Headings: {content_analysis.get('headings', {}).get('count', {}).get('total', 0)}
- Links: {links_analysis.get('links_analysis', {}).get('total_links', 0)}
- Images: {content_analysis.get('content_structure', {}).get('images', 0)}

Title: {metadata.get('title', 'No title')}
Description: {metadata.get('description', 'No description')}"""

            # Get AI analysis
            analysis_results["ai_analysis"] = safe_groq_request(ai_prompt)
            
            return analysis_results
            
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - website took too long to respond"}
    except requests.exceptions.TooManyRedirects:
        return {"error": "Too many redirects - could not reach final destination"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error - unable to reach website"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"Website analysis error: {e}")
        return {"error": f"Failed to analyze website: {str(e)}"}

def extract_metadata(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Extract comprehensive metadata from webpage."""
    metadata = {
        "title": "No title found",
        "description": "No description found",
        "keywords": "No keywords found",
        "author": "No author found",
        "language": "Unknown",
        "favicon": None,
        "og_data": {},
        "twitter_data": {},
        "schema_data": {},
        "security": {},
        "technologies": set(),
    }
    
    try:
        # Title with fallbacks
        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()
        else:
            og_title = soup.find('meta', property='og:title')
            if og_title:
                metadata["title"] = og_title.get('content', '').strip()
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            property = meta.get('property', '').lower()
            content = meta.get('content', '').strip()
            
            if name in ['description', 'keywords', 'author', 'language'] or \
               property in ['og:description', 'og:site_name']:
                metadata[name] = content
            elif name == 'generator':
                metadata['technologies'].add(f"CMS: {content}")
            elif name in ['robots', 'googlebot']:
                metadata['technologies'].add(f"SEO: {content}")
            
            # Security headers
            if name in ['content-security-policy', 'x-frame-options', 'x-xss-protection']:
                metadata['security'][name] = content
        
        # OpenGraph data
        og_tags = ['title', 'description', 'image', 'type', 'site_name', 'locale']
        for tag in og_tags:
            og_meta = soup.find('meta', property=f'og:{tag}')
            if og_meta:
                metadata["og_data"][tag] = og_meta.get('content', '').strip()
        
        # Twitter Card data
        twitter_tags = ['card', 'site', 'creator', 'title', 'description', 'image']
        for tag in twitter_tags:
            twitter_meta = soup.find('meta', attrs={'name': f'twitter:{tag}'})
            if twitter_meta:
                metadata["twitter_data"][tag] = twitter_meta.get('content', '').strip()
        
        # Schema.org data
        for schema in soup.find_all(['script', 'meta'], attrs={'type': 'application/ld+json'}):
            try:
                if schema.string:
                    schema_data = json.loads(schema.string)
                    if isinstance(schema_data, dict):
                        metadata["schema_data"].update(schema_data)
            except json.JSONDecodeError:
                continue
        
        # Favicon with multiple fallbacks
        favicon_links = [
            ('icon', 'shortcut icon', 'apple-touch-icon'),
            ('image/x-icon', 'image/png', 'image/jpeg')
        ]
        for rel_values, types in zip(*favicon_links):
            for rel, type_ in zip(rel_values.split(), types.split()):
                favicon = soup.find('link', rel=rel, type=type_)
                if favicon and favicon.get('href'):
                    favicon_url = favicon['href']
                    if not favicon_url.startswith(('http://', 'https://')):
                        favicon_url = urljoin(url, favicon_url)
                    metadata["favicon"] = favicon_url
                    break
            if metadata["favicon"]:
                break
        
        # Detect technologies
        # JavaScript frameworks
        if soup.find('script', src=lambda x: x and 'react' in x.lower()):
            metadata['technologies'].add('Framework: React')
        if soup.find('script', src=lambda x: x and 'vue' in x.lower()):
            metadata['technologies'].add('Framework: Vue.js')
        if soup.find('script', src=lambda x: x and 'angular' in x.lower()):
            metadata['technologies'].add('Framework: Angular')
        
        # Analytics
        if soup.find('script', src=lambda x: x and 'google-analytics' in x.lower()):
            metadata['technologies'].add('Analytics: Google Analytics')
        if soup.find('script', src=lambda x: x and 'gtag' in x.lower()):
            metadata['technologies'].add('Analytics: Google Tag Manager')
        
        # Convert technologies set to sorted list
        metadata['technologies'] = sorted(list(metadata['technologies']))
        
        return metadata
        
    except Exception as e:
        logger.warning(f"Error extracting metadata: {e}")
        return metadata

def analyze_content(soup: BeautifulSoup) -> Dict[str, Any]:
    """Analyze webpage content with enhanced features."""
    try:
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
            element.decompose()
        
        # Extract main content with better targeting
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_=lambda x: x and any(c in x.lower() for c in ['content', 'main', 'article'])) or
            soup
        )
        
        # Get text content
        text = main_content.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Basic text analysis
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        # Enhanced content analysis
        analysis = {
            "content_length": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 2),
            "avg_sentence_length": round(len(text) / max(len(sentences), 1), 2),
            "text_density": round(len(text) / (len(text) + len(str(soup))), 3),
            "headings": analyze_headings(soup),
            "content_structure": analyze_structure(soup),
            "language_metrics": analyze_language(text),
            "text_sample": text[:config.MAX_WORDS_FOR_SUMMARY] + "..." if len(text) > config.MAX_WORDS_FOR_SUMMARY else text
        }
        
        # Word frequency and readability
        analysis["frequent_words"] = analyze_word_frequency(words)
        analysis["readability"] = calculate_readability(text)
        
        return analysis
        
    except Exception as e:
        logger.warning(f"Error analyzing content: {e}")
        return {"error": "Failed to analyze content"}

def analyze_headings(soup: BeautifulSoup) -> Dict[str, Any]:
    """Analyze heading structure of the webpage."""
    try:
        headings = {'h1': [], 'h2': [], 'h3': [], 'h4': [], 'h5': [], 'h6': []}
        heading_count = {'total': 0}
        
        for level in range(1, 7):
            tag = f'h{level}'
            elements = soup.find_all(tag)
            headings[tag] = [h.get_text().strip() for h in elements]
            heading_count[tag] = len(elements)
            heading_count['total'] += len(elements)
        
        return {
            'structure': headings,
            'count': heading_count,
            'has_proper_structure': heading_count.get('h1', 0) == 1
        }
    except Exception as e:
        logger.warning(f"Error analyzing headings: {e}")
        return {}

def analyze_structure(soup: BeautifulSoup) -> Dict[str, Any]:
    """Analyze the structural elements of the webpage."""
    try:
        structure = {
            'lists': {
                'ordered': len(soup.find_all('ol')),
                'unordered': len(soup.find_all('ul')),
                'definition': len(soup.find_all('dl'))
            },
            'tables': len(soup.find_all('table')),
            'forms': len(soup.find_all('form')),
            'images': len(soup.find_all('img')),
            'videos': len(soup.find_all(['video', 'iframe[src*="youtube"], iframe[src*="vimeo"]'])),
            'interactive': {
                'buttons': len(soup.find_all('button')),
                'inputs': len(soup.find_all('input')),
                'links': len(soup.find_all('a'))
            }
        }
        
        # Calculate content type ratios
        total_elements = sum(structure['lists'].values()) + structure['tables'] + \
                        structure['forms'] + structure['images'] + structure['videos']
        
        if total_elements > 0:
            structure['composition'] = {
                'text_heavy': structure['lists']['ordered'] + structure['lists']['unordered'] > total_elements * 0.3,
                'media_rich': (structure['images'] + structure['videos']) > total_elements * 0.3,
                'interactive': structure['forms'] + structure['interactive']['buttons'] > total_elements * 0.2
            }
        
        return structure
    except Exception as e:
        logger.warning(f"Error analyzing structure: {e}")
        return {}

def analyze_language(text: str) -> Dict[str, Any]:
    """Analyze language patterns and complexity."""
    try:
        # Basic patterns
        patterns = {
            'questions': len(re.findall(r'\?', text)),
            'exclamations': len(re.findall(r'!', text)),
            'numbers': len(re.findall(r'\d+', text)),
            'urls': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            'emails': len(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)),
            'parentheses': len(re.findall(r'\(.*?\)', text))
        }
        
        # Word complexity
        words = text.split()
        long_words = sum(1 for w in words if len(w) > 6)
        very_long_words = sum(1 for w in words if len(w) > 10)
        
        complexity = {
            'avg_word_length': round(sum(len(w) for w in words) / max(len(words), 1), 2),
            'long_words_ratio': round(long_words / max(len(words), 1), 3),
            'very_long_words_ratio': round(very_long_words / max(len(words), 1), 3),
            'unique_words_ratio': round(len(set(words)) / max(len(words), 1), 3)
        }
        
        return {
            'patterns': patterns,
            'complexity': complexity
        }
    except Exception as e:
        logger.warning(f"Error analyzing language: {e}")
        return {}

def analyze_links(soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
    """Analyze webpage links."""
    try:
        from urllib.parse import urljoin, urlparse
        
        links = soup.find_all('a', href=True)
        link_analysis = {
            "total_links": len(links),
            "internal_links": 0,
            "external_links": 0,
            "social_links": 0,
            "resource_links": 0,
            "unique_domains": set()
        }
        
        base_domain = urlparse(base_url).netloc
        
        for link in links:
            try:
                href = link['href']
                if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                    continue
                    
                if not href.startswith(('http://', 'https://')):
                    href = urljoin(base_url, href)
                
                parsed = urlparse(href)
                domain = parsed.netloc
                
                if not domain:
                    continue
                
                # Add to unique domains
                link_analysis["unique_domains"].add(domain)
                
                # Categorize link
                if domain == base_domain:
                    link_analysis["internal_links"] += 1
                elif any(social in domain for social in ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com']):
                    link_analysis["social_links"] += 1
                elif href.endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar')):
                    link_analysis["resource_links"] += 1
                else:
                    link_analysis["external_links"] += 1
            except Exception as e:
                logger.debug(f"Error processing link: {e}")
                continue
        
        # Convert set to list for JSON serialization
        link_analysis["unique_domains"] = sorted(list(link_analysis["unique_domains"]))
        
        return {"links_analysis": link_analysis}
        
    except Exception as e:
        logger.warning(f"Error analyzing links: {e}")
        return {"error": "Failed to analyze links"}

def analyze_word_frequency(words: List[str], top_n: int = 10) -> Dict[str, int]:
    """Analyze word frequency excluding common words."""
    try:
        # Common English stop words to exclude
        stop_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 
                         'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'])
        
        # Clean and filter words
        word_freq = {}
        for word in words:
            word = word.lower().strip('.,!?";()[]{}')
            if len(word) > 2 and word not in stop_words and word.isalnum():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top N frequent words
        sorted_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        # Ensure we have at least one item
        if not sorted_freq:
            return {"no_significant_words": 1}
            
        return sorted_freq
        
    except Exception as e:
        logger.warning(f"Error analyzing word frequency: {e}")
        return {"error_analyzing": 1}

def calculate_readability(text: str) -> Dict[str, float]:
    """Calculate readability metrics."""
    try:
        # Count sentences, words, and syllables
        sentences = max(1, len(re.split(r'[.!?]+', text)))
        words = max(1, len(text.split()))
        syllables = max(1, sum(count_syllables(word) for word in text.split()))
        
        # Calculate Flesch Reading Ease score
        flesch_score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        flesch_score = max(0, min(100, flesch_score))  # Clamp between 0 and 100
        
        # Calculate reading time (assuming 200 words per minute)
        reading_time = max(0.1, words / 200)  # Minimum 0.1 minutes
        
        return {
            "flesch_reading_ease": round(flesch_score, 2),
            "estimated_reading_time": round(reading_time, 2)
        }
        
    except Exception as e:
        logger.warning(f"Error calculating readability: {e}")
        return {
            "flesch_reading_ease": 0,
            "estimated_reading_time": 0
        }

def count_syllables(word: str) -> int:
    """Estimate number of syllables in a word."""
    try:
        word = word.lower().strip('.,!?')
        count = 0
        vowels = 'aeiouy'
        prev_char_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_is_vowel:
                count += 1
            prev_char_is_vowel = is_vowel
        
        # Handle some common cases
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if count == 0:
            count = 1
            
        return count
        
    except Exception:
        return 1

def analyze_document(file) -> Dict[str, Any]:
    """Analyze document with improved file type detection and error handling."""
    try:
        file_content = file.getvalue()
        
        # Detect file type more reliably
        if file.name.lower().endswith('.txt') or file.type == "text/plain":
            # Detect encoding for text files
            detected = chardet.detect(file_content)
            encoding = detected.get('encoding', 'utf-8')
            text = file_content.decode(encoding, errors='ignore')
            
        elif file.name.lower().endswith('.docx') or file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(BytesIO(file_content))
            
        elif file.name.lower().endswith('.pdf') or file.type == "application/pdf":
            try:
                # Use pypdf for better PDF handling
                reader = PdfReader(BytesIO(file_content))
                text = ""
                total_pages = len(reader.pages)
                
                # Extract text from each page
                for i, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                    logger.info(f"Processed PDF page {i}/{total_pages}")
                
                if not text.strip():
                    return {
                        "error": "No readable text found in PDF. The file might be scanned or protected.",
                        "suggestions": [
                            "Check if the PDF contains actual text (not scanned images)",
                            "Ensure the PDF is not password-protected",
                            "Try converting scanned PDFs using OCR software first"
                        ]
                    }
                
            except Exception as e:
                logger.error(f"PDF processing error: {str(e)}")
                return {
                    "error": f"Failed to process PDF: {str(e)}",
                    "suggestions": [
                        "Check if the file is a valid PDF",
                        "Ensure the file is not corrupted",
                        "Try saving the PDF with a different PDF viewer"
                    ]
                }
                
        elif file.name.lower().endswith('.json') or file.type == "application/json":
            return analyze_json(file)
            
        elif file.name.lower().endswith(('.xml', '.html', '.htm')) or file.type in ["application/xml", "text/xml", "text/html"]:
            return analyze_xml(file)
            
        elif file.name.lower().endswith('.parquet') or file.type == "application/x-parquet":
            return analyze_parquet(file)
            
        elif file.name.lower().endswith(('.yaml', '.yml')) or file.type == "text/yaml":
            return analyze_yaml(file)
            
        else:
            return {"error": f"Unsupported file type: {file.type}"}
            
        return analyze_text(text)
        
    except Exception as e:
        # Consolidate error handling
        error_msg = str(e)
        logger.error(f"Document analysis error: {error_msg}")
        return {
            "error": f"Failed to analyze document: {error_msg}",
            "suggestions": [
                "Check if the file is valid and not corrupted",
                "Ensure the file format is supported",
                "Try saving the file with a different application"
            ]
        }

def analyze_excel(file) -> Dict[str, Any]:
    """Analyze Excel/CSV files with improved encoding detection and error handling."""
    try:
        file_content = file.getvalue()
        
        # Simplify file type checking
        is_csv = file.name.lower().endswith('.csv') or file.type == "text/csv"
        is_excel = file.name.lower().endswith(('.xlsx', '.xls'))
        
        if is_csv:
            # CSV handling
            detected = chardet.detect(file_content)
            encoding = detected.get('encoding', 'utf-8')
            df = pd.read_csv(BytesIO(file_content), encoding=encoding)
        elif is_excel:
            # Excel handling
            engine = 'openpyxl' if file.name.lower().endswith('.xlsx') else 'xlrd'
            df = pd.read_excel(BytesIO(file_content), engine=engine)
        else:
            return {"error": f"Unsupported spreadsheet format: {file.name}"}
            
        return analyze_dataframe(df, file.name)
        
    except Exception as e:
        logger.error(f"Excel analysis error: {e}")
        return {"error": f"Failed to analyze Excel file: {str(e)}"}

def analyze_database(file) -> Dict[str, Any]:
    """Analyze SQLite database with improved error handling."""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        
        conn = sqlite3.connect(temp_path)
        
        try:
            # Get all tables
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table';", 
                conn
            )
            
            if tables.empty:
                return {"error": "No tables found in database"}
            
            results = {"database_info": {"total_tables": len(tables)}}
            
            for table_name in tables['name']:
                try:
                    df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
                    results[f"table_{table_name}"] = analyze_dataframe(df, table_name)
                except Exception as e:
                    results[f"table_{table_name}"] = {"error": f"Failed to analyze table: {str(e)}"}
                    
        finally:
            conn.close()
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
            
        return results
        
    except Exception as e:
        logger.error(f"Database analysis error: {e}")
        return {"error": f"Failed to analyze database: {str(e)}"}

def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text content with improved metrics."""
    if not text or not text.strip():
        return {"error": "Empty or invalid text content"}
    
    try:
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Basic metrics
        words = text.split()
        word_count = len(words)
        sentence_count = len(re.findall(r'[.!?]+', text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Word frequency analysis
        word_freq = pd.Series([word.lower().strip('.,!?";()[]{}') for word in words if len(word) > 2])
        frequent_words = word_freq.value_counts().head(10)

        # AI analysis
        prompt = f"""Analyze the following text and provide insights about:
1. Main topics and themes
2. Tone and sentiment
3. Key information
4. Writing style

Text (first 1000 characters):
{text[:1000]}..."""

        ai_analysis = safe_groq_request(prompt)

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_words_per_sentence": round(word_count / max(sentence_count, 1), 2),
            "frequent_words": frequent_words.to_dict(),
            "summary": text[:500] + "..." if len(text) > 500 else text,
            "ai_analysis": ai_analysis,
            "text_stats": {
                "characters": len(text),
                "words": word_count,
                "unique_words": len(set(words)),
                "sentences": sentence_count,
                "paragraphs": paragraph_count,
                "avg_word_length": round(sum(len(word) for word in words) / max(len(words), 1), 2),
                "avg_sentence_length": round(word_count / max(sentence_count, 1), 2)
            }
        }
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        return {"error": f"Failed to analyze text: {str(e)}"}

def analyze_json(file) -> Dict[str, Any]:
    """Analyze JSON files with improved structure detection."""
    try:
        content = file.getvalue()
        data = json.loads(content.decode('utf-8'))
        
        def analyze_json_structure(obj, path="root"):
            """Recursively analyze JSON structure."""
            if isinstance(obj, dict):
                return {
                    "type": "object",
                    "keys": list(obj.keys())[:10],  # Limit to first 10 keys
                    "key_count": len(obj.keys()),
                    "nested_objects": sum(1 for v in obj.values() if isinstance(v, dict))
                }
            elif isinstance(obj, list):
                return {
                    "type": "array",
                    "length": len(obj),
                    "item_types": list(set(type(item).__name__ for item in obj[:10]))
                }
            else:
                return {"type": type(obj).__name__, "value": str(obj)[:100]}
        
        structure = analyze_json_structure(data)
        
        return {
            "file_type": "JSON",
            "structure": structure,
            "size_bytes": len(content),
            "sample": json.dumps(data, indent=2)[:1000] + "..." if len(json.dumps(data)) > 1000 else json.dumps(data, indent=2)
        }
        
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        logger.error(f"JSON analysis error: {e}")
        return {"error": f"Failed to analyze JSON: {str(e)}"}

def analyze_xml(file) -> Dict[str, Any]:
    """Analyze XML files with improved structure detection."""
    try:
        content = file.getvalue()
        root = ET.fromstring(content)
        
        # Extract XML structure information
        elements = {}
        for elem in root.iter():
            tag = elem.tag
            if tag not in elements:
                elements[tag] = 0
            elements[tag] += 1
        
        return {
            "file_type": "XML",
            "root_element": root.tag,
            "total_elements": sum(elements.values()),
            "unique_elements": len(elements),
            "element_counts": dict(sorted(elements.items(), key=lambda x: x[1], reverse=True)[:10]),
            "sample": ET.tostring(root, encoding='unicode')[:1000] + "..."
        }
        
    except ET.ParseError as e:
        return {"error": f"Invalid XML format: {str(e)}"}
    except Exception as e:
        logger.error(f"XML analysis error: {e}")
        return {"error": f"Failed to analyze XML: {str(e)}"}

def analyze_parquet(file) -> Dict[str, Any]:
    """Analyze Parquet files."""
    try:
        table = pq.read_table(BytesIO(file.getvalue()))
        df = table.to_pandas()
        return analyze_dataframe(df, file.name)
    except Exception as e:
        logger.error(f"Parquet analysis error: {e}")
        return {"error": f"Failed to analyze Parquet file: {str(e)}"}

def analyze_yaml(file) -> Dict[str, Any]:
    """Analyze YAML files with improved structure detection."""
    try:
        content = file.getvalue()
        data = yaml.safe_load(content)
        
        def count_yaml_structure(obj):
            """Count YAML structure elements."""
            if isinstance(obj, dict):
                return {"dicts": 1, "lists": 0, "values": sum(count_yaml_structure(v) for v in obj.values() if v is not None)}
            elif isinstance(obj, list):
                return {"dicts": 0, "lists": 1, "values": sum(count_yaml_structure(item) for item in obj if item is not None)}
            else:
                return {"dicts": 0, "lists": 0, "values": 1}
        
        structure = count_yaml_structure(data) if data else {"dicts": 0, "lists": 0, "values": 0}
        
        return {
            "file_type": "YAML",
            "structure": structure,
            "top_level_keys": list(data.keys())[:10] if isinstance(data, dict) else [],
            "sample": yaml.dump(data, default_flow_style=False)[:1000] + "..."
        }
        
    except yaml.YAMLError as e:
        return {"error": f"Invalid YAML format: {str(e)}"}
    except Exception as e:
        logger.error(f"YAML analysis error: {e}")
        return {"error": f"Failed to analyze YAML: {str(e)}"}

def analyze_dataframe(df: pd.DataFrame, filename: str = "dataset") -> Dict[str, Any]:
    """Comprehensive dataframe analysis with improved error handling."""
    try:
        if df.empty:
            return {"error": "Dataset is empty"}
        
        results = {
            "filename": filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        }

        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results["summary_stats"] = df[numeric_cols].describe().to_dict()
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_info = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                unique_vals = df[col].nunique()
                categorical_info[col] = {
                    "unique_count": unique_vals,
                    "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    "sample_values": df[col].dropna().unique()[:5].tolist()
                }
            results["categorical_analysis"] = categorical_info

        # Time series analysis
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) == 0:
            # Try to detect date columns
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_cols = [col]
                        break
                    except:
                        continue

        if len(date_cols) > 0:
            date_col = date_cols[0]
            df_copy = df.copy()
            df_copy = df_copy.set_index(pd.to_datetime(df_copy[date_col]))
            
            # Time series resampling
            if len(df_copy) > 1:
                try:
                    results["time_series"] = {
                        "date_range": {
                            "start": str(df_copy.index.min()),
                            "end": str(df_copy.index.max()),
                            "duration_days": (df_copy.index.max() - df_copy.index.min()).days
                        }
                    }
                    
                    # ARIMA forecasting for numeric columns
                    if len(numeric_cols) > 0:
                        forecasts = {}
                        for col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                            try:
                                series = df_copy[col].dropna()
                                if len(series) > 10:  # Need sufficient data for ARIMA
                                    model = ARIMA(series, order=(1,1,1))
                                    fit_model = model.fit()
                                    forecast = fit_model.forecast(steps=10)
                                    forecasts[col] = forecast.tolist()
                            except Exception as e:
                                logger.warning(f"ARIMA forecast failed for {col}: {e}")
                                continue
                        
                        if forecasts:
                            results["forecasts"] = forecasts
                            
                except Exception as e:
                    logger.warning(f"Time series analysis failed: {e}")

        # Correlation analysis
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr()
                # Remove self-correlations and get top correlations
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                corr_matrix = corr_matrix.mask(mask)
                
                # Find high correlations
                high_corr = corr_matrix.abs().stack().sort_values(ascending=False)
                high_corr = high_corr[high_corr > 0.7].head(10)
                
                results["correlation"] = {
                    "matrix": corr_matrix.to_dict(),
                    "high_correlations": high_corr.to_dict() if not high_corr.empty else {}
                }
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")

        # Simple predictive analysis
        if len(numeric_cols) > 1:
            try:
                target = numeric_cols[-1]
                features = numeric_cols[:-1]
                
                # Remove rows with NaN values
                analysis_df = df[list(features) + [target]].dropna()
                
                if len(analysis_df) > 10:  # Need sufficient data
                    X = analysis_df[features]
                    y = analysis_df[target]
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results["predictive_analysis"] = {
                        "target": target,
                        "features": features.tolist(),
                        "mse": round(mse, 4),
                        "r2": round(r2, 4),
                        "coefficients": dict(zip(features, np.round(model.coef_, 4)))
                    }
            except Exception as e:
                logger.warning(f"Predictive analysis failed: {e}")

        # AI-powered analysis
        sample_data = df.head(5).to_string()
        prompt = f"""Analyze this dataset and provide business insights:

Dataset: {filename}
Shape: {df.shape[0]} rows, {df.shape[1]} columns
Columns: {', '.join(df.columns.tolist())}

Sample data:
{sample_data}

Provide insights about:
1. Data quality and completeness
2. Potential business value
3. Recommended analysis approaches
4. Data anomalies or patterns"""

        results["ai_analysis"] = safe_groq_request(prompt)

        return results
        
    except Exception as e:
        logger.error(f"Dataframe analysis error: {e}")
        return {"error": f"Failed to analyze dataset: {str(e)}"}

def generate_visualizations(data: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """Generate visualizations with improved error handling and modern styling."""
    visualizations = {}
    
    # Set modern style
    plt.style.use('seaborn-v0_8')
    
    try:
        # Summary statistics visualization
        if "summary_stats" in data:
            fig, ax = plt.subplots(figsize=(12, 6))
            df_stats = pd.DataFrame(data["summary_stats"])
            df_stats.loc['mean'].plot(kind='bar', ax=ax, color='skyblue', alpha=0.8)
            ax.set_title('Mean Values by Column', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            visualizations['summary_stats'] = fig

        # Correlation heatmap
        if "correlation" in data and "matrix" in data["correlation"]:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_df = pd.DataFrame(data["correlation"]["matrix"])
            mask = np.triu(np.ones_like(corr_df, dtype=bool))
            sns.heatmap(corr_df, mask=mask, annot=True, cmap='RdYlBu_r', 
                       center=0, fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            visualizations['correlation_heatmap'] = fig

        # Null values visualization
        if "null_percentage" in data:
            null_data = {k: v for k, v in data["null_percentage"].items() if v > 0}
            if null_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                pd.Series(null_data).plot(kind='bar', ax=ax, color='salmon', alpha=0.8)
                ax.set_title('Missing Data Percentage by Column', fontsize=14, fontweight='bold')
                ax.set_ylabel('Percentage Missing (%)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                visualizations['missing_data'] = fig

        # Time series visualization
        if "time_series" in data and "date_range" in data["time_series"]:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f"Time Series Data Available\n"
                             f"Date Range: {data['time_series']['date_range']['start']} to "
                             f"{data['time_series']['date_range']['end']}\n"
                             f"Duration: {data['time_series']['date_range']['duration_days']} days",
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            ax.set_title('Time Series Information', fontsize=14, fontweight='bold')
            ax.axis('off')
            visualizations['time_series_info'] = fig

        # Forecasts visualization
        if "forecasts" in data:
            fig, ax = plt.subplots(figsize=(12, 6))
            for col, forecast in data["forecasts"].items():
                ax.plot(range(len(forecast)), forecast, marker='o', label=f"{col} Forecast")
            ax.set_title('Forecasting Results', fontsize=14, fontweight='bold')
            ax.set_xlabel('Future Time Periods')
            ax.set_ylabel('Predicted Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            visualizations['forecasts'] = fig

        # Feature importance visualization
        if "predictive_analysis" in data and "coefficients" in data["predictive_analysis"]:
            coeffs = data["predictive_analysis"]["coefficients"]
            fig, ax = plt.subplots(figsize=(10, 6))
            pd.Series(coeffs).plot(kind='barh', ax=ax, color='lightgreen', alpha=0.8)
            ax.set_title('Feature Importance (Regression Coefficients)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Coefficient Value')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            visualizations['feature_importance'] = fig

    except Exception as e:
        logger.error(f"Visualization generation error: {e}")
        # Return a simple error visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Visualization Error:\n{str(e)}", 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax.set_title('Visualization Error', fontsize=14, fontweight='bold')
        ax.axis('off')
        visualizations['error'] = fig

    return visualizations 