import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import sqlite3
from io import StringIO, BytesIO
import docx2txt
from pypdf import PdfReader  # Updated from PyPDF2
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize the Groq client with error handling
try:
    if config.GROQ_API_KEY:
        client = Groq(api_key=config.GROQ_API_KEY)
        print("✅ Groq client initialized successfully")
    else:
        client = None
        print("⚠️ Groq client not initialized - API key not found")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    client = None

def safe_groq_request(prompt: str, max_retries: int = 3) -> str:
    """Safely make a request to Groq API with retries."""
    if not client:
        return "AI analysis unavailable - Please set your GROQ_API_KEY in the .env file"
    
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
                return f"AI analysis unavailable - API error: {str(e)}"
    
    return "AI analysis unavailable"

def analyze_website(url: str) -> Dict[str, Any]:
    """Analyze website content with improved error handling."""
    try:
        # Add timeout and headers for better reliability
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()  # Clean up whitespace
        
        # Extract metadata with better error handling
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"
        
        description_tag = soup.find('meta', attrs={'name': 'description'}) or \
                         soup.find('meta', attrs={'property': 'og:description'})
        description = description_tag.get('content', 'No description found') if description_tag else 'No description found'
        
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        keywords = keywords_tag.get('content', 'No keywords found') if keywords_tag else 'No keywords found'

        # Analyze the text content
        text_analysis = analyze_text(text)
        
        return {
            "title": title,
            "description": description,
            "keywords": keywords,
            "url": url,
            "content_length": len(text),
            **text_analysis
        }
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - website took too long to respond"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error - unable to reach website"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"Website analysis error: {e}")
        return {"error": f"Failed to analyze website: {str(e)}"}

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
            reader = PdfReader(BytesIO(file_content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
                
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
        logger.error(f"Document analysis error: {e}")
        return {"error": f"Failed to analyze document: {str(e)}"}

def analyze_excel(file) -> Dict[str, Any]:
    """Analyze Excel/CSV files with improved encoding detection and error handling."""
    try:
        file_content = file.getvalue()
        
        if file.name.lower().endswith('.csv') or file.type == "text/csv":
            # Detect encoding for CSV files
            detected = chardet.detect(file_content)
            encoding = detected.get('encoding', 'utf-8')
            
            try:
                df = pd.read_csv(BytesIO(file_content), encoding=encoding)
            except UnicodeDecodeError:
                # Fallback encodings
                for fallback_encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(BytesIO(file_content), encoding=fallback_encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise UnicodeDecodeError("Unable to decode CSV file")
                    
        elif file.name.lower().endswith(('.xlsx', '.xls')):
            # Use openpyxl engine for better compatibility
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
        "ai_analysis": ai_analysis
    }

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