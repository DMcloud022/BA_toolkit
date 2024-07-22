import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import sqlite3
from io import StringIO, BytesIO
import docx2txt
import PyPDF2
from typing import Dict, Any
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
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import streamlit as st

# Initialize the Groq client
client = Groq(api_key=config.GROQ_API_KEY)

def analyze_website(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        
        # Extract metadata
        title = soup.title.string if soup.title else "No title found"
        description = soup.find('meta', attrs={'name': 'description'})
        description = description['content'] if description else "No description found"
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        keywords = keywords['content'] if keywords else "No keywords found"

        # Analyze the text content
        text_analysis = analyze_text(text)
        
        # Use Groq for enhanced analysis
        prompt = f"Analyze the following text result and provide insights and analysis for technical and non-technical user:\n\n{text[:1000]}..."
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        ai_analysis = response.choices[0].message.content

        return {
            "title": title,
            "description": description,
            "keywords": keywords,
            "word_count": text_analysis['word_count'],
            "sentence_count": text_analysis['sentence_count'],
            "frequent_words": text_analysis['frequent_words'],
            "summary": text_analysis['summary'],
            "ai_analysis": ai_analysis
        }
    except requests.RequestException as e:
        return {"error": f"Failed to fetch website: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to analyze website: {str(e)}"}

def analyze_document(file) -> Dict[str, Any]:
    try:
        if file.type == "text/plain":
            text = StringIO(file.getvalue().decode("utf-8")).read()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(BytesIO(file.getvalue()))
        elif file.type == "application/pdf":
            reader = PyPDF2.PdfReader(BytesIO(file.getvalue()))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        elif file.type == "application/json":
            return analyze_json(file)
        elif file.type == "application/xml":
            return analyze_xml(file)
        elif file.type == "application/x-parquet":
            return analyze_parquet(file)
        elif file.type == "text/yaml":
            return analyze_yaml(file)
        else:
            return {"error": "Unsupported file type"}
        return analyze_text(text)
    except Exception as e:
        return {"error": f"Failed to analyze document: {str(e)}"}

def analyze_excel(file) -> Dict[str, Any]:
    try:
        # Detect encoding for CSV files
        if file.type == "text/csv":
            raw_data = file.getvalue()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            df = pd.read_csv(BytesIO(raw_data), encoding=encoding)
        else:
            df = pd.read_excel(BytesIO(file.getvalue()))
        return analyze_dataframe(df)
    except Exception as e:
        return {"error": f"Failed to analyze Excel file: {str(e)}"}

def analyze_database(file) -> Dict[str, Any]:
    try:
        conn = sqlite3.connect(file.name)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        results = {}
        for table in tables['name']:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            results[table] = analyze_dataframe(df)
        return results
    except Exception as e:
        return {"error": f"Failed to analyze database: {str(e)}"}

def analyze_text(text: str) -> Dict[str, Any]:
    word_count = len(text.split())
    sentence_count = len(re.findall(r'\w+[.!?]', text))
    frequent_words = pd.Series(text.lower().split()).value_counts().head(10)

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [w for w in word_tokens if w.isalnum() and w not in stop_words]
    frequent_words = pd.Series(filtered_words).value_counts().head(10)

    # Use TextBlob for sentiment analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    # Use Groq for enhanced analysis
    prompt = f"Analyze the following text result and provide insights and analysis for technical and non-technical user:\n\n{text[:1000]}..."
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    ai_analysis = response.choices[0].message.content

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "frequent_words": frequent_words.to_dict(),
        "summary": text[:500] + "..." if len(text) > 500 else text,
        "sentiment": sentiment,
        "ai_analysis": ai_analysis
    }

# Analyze JSON files
def analyze_json(file) -> Dict[str, Any]:
    try:
        data = json.load(BytesIO(file.getvalue()))
        # Basic JSON structure analysis
        num_keys = len(data.keys())
        return {
            "json_keys_count": num_keys,
            "json_sample": json.dumps(data, indent=2)
        }
    except json.JSONDecodeError as e:
        return {"error": f"Failed to decode JSON: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to analyze JSON: {str(e)}"}

# Analyze XML files
def analyze_xml(file) -> Dict[str, Any]:
    try:
        tree = ET.parse(BytesIO(file.getvalue()))
        root = tree.getroot()
        # Extract some XML structure details
        elements = [elem.tag for elem in root.iter()]
        return {
            "xml_elements": elements[:10],  # Sample first 10 elements
            "xml_structure": ET.tostring(root, encoding='utf-8').decode()
        }
    except ET.ParseError as e:
        return {"error": f"Failed to parse XML: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to analyze XML: {str(e)}"}

# Analyze Parquet files
def analyze_parquet(file) -> Dict[str, Any]:
    try:
        table = pq.read_table(BytesIO(file.getvalue()))
        df = table.to_pandas()
        return analyze_dataframe(df)  # Reuse analyze_dataframe function
    except Exception as e:
        return {"error": f"Failed to analyze Parquet file: {str(e)}"}

# Analyze YAML files
def analyze_yaml(file) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(BytesIO(file.getvalue()))
        # Basic YAML structure analysis
        num_keys = len(data.keys())
        return {
            "yaml_keys_count": num_keys,
            "yaml_sample": yaml.dump(data)
        }
    except yaml.YAMLError as e:
        return {"error": f"Failed to parse YAML: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to analyze YAML: {str(e)}"}
    
def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "summary_stats": df.describe(include='all').to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
    }

    # Time series analysis
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        date_col = date_cols[0]
        df['date'] = pd.to_datetime(df[date_col])
        df.set_index('date', inplace=True)

        time_periods = ['D', 'W', 'M', 'Y']
        time_series_data = {}
        for period in time_periods:
            resampled = df.resample(period).mean()
            if not resampled.empty:
                time_series_data[period] = resampled.to_dict()

        results["time_series"] = time_series_data
            
        # ARIMA forecasting
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            forecasts = {}
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                model = ARIMA(df[col], order=(1,1,1))
                fit_model = model.fit()
                forecast = fit_model.forecast(steps=30)  # Forecast next 30 periods
                forecasts[col] = forecast.tolist()
            results["forecasts"] = forecasts

        # Detect outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outliers = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_range = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
                outliers[col] = df[(df[col] < outlier_range[0]) | (df[col] > outlier_range[1])].shape[0]
            results["outliers"] = outliers

    # Correlation analysis for numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        results["correlation"] = corr_matrix.to_dict()

    # Predictive analysis
    if len(num_cols) > 1:
        target = num_cols[-1]
        features = num_cols[:-1]
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results["predictive_analysis"] = {
            "target": target,
            "features": features.tolist(),
            "mse": mse,
            "r2": r2,
            "coefficients": dict(zip(features, model.coef_))
        }
    results["data_validation"] = validate_data(df)
    
    # Use Groq for enhanced analysis
    prompt = f"Analyze the following text result and provide insights and analysis for technical and non-technical user:\n\n{df.head().to_string()}\n\nSummary statistics:\n{df.describe().to_string()}"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    
    results["ai_analysis"] = response.choices[0].message.content
    
    return results

def generate_groq_insights(data: Dict[str, Any]) -> Dict[str, str]:
    prompt = f"Analyze the following text result and provide insights and analysis for technical and non-technical user:\n\n{data}"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    return {"groq_insights": response.choices[0].message.content}

def generate_future_predictions(data: Dict[str, Any]) -> Dict[str, str]:
    prompt = f"Based on the following data, predict future trends and recommendation to improve base on the context:\n\n{data}"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    return {"future_predictions": response.choices[0].message.content}

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    validation_results = {}
    
    # Check for missing values
    missing_values = df.isnull().sum()
    validation_results["missing_values"] = missing_values[missing_values > 0].to_dict()
    
    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    validation_results["duplicate_rows"] = duplicate_rows
    
    # Check for constant columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    validation_results["constant_columns"] = constant_columns
    
    # Check for high cardinality in categorical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    high_cardinality = {col: df[col].nunique() for col in cat_columns if df[col].nunique() > 100}
    validation_results["high_cardinality"] = high_cardinality
    
    return validation_results


def generate_visualizations(data):
    visualizations = {}

    try:
        # Example: Line plot for dataframe data
        if 'dataframe' in data:
            df = pd.DataFrame(data['dataframe'])
            plt.figure(figsize=(12, 6))
            plt.plot(df[df.columns[0]], df[df.columns[1]], marker='o')
            plt.title('Line Plot')
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[1])
            plt.grid(True)
            fig = plt.gcf()
            visualizations['line_plot'] = fig
            plt.close(fig)

        if 'dataframe' in data:
            df = pd.DataFrame(data['dataframe'])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                plt.figure(figsize=(12, 6))
                df[numeric_cols].boxplot()
                plt.title('Box Plot of Numeric Columns')
                plt.ylabel('Value')
                plt.xticks(rotation=45)
                fig = plt.gcf()
                visualizations['box_plot'] = fig
                plt.close(fig)

        # Add a scatter matrix for numeric columns
        if 'dataframe' in data:
            df = pd.DataFrame(data['dataframe'])
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4 columns
            if len(numeric_cols) > 1:
                fig = sns.pairplot(df[numeric_cols])
                visualizations['scatter_matrix'] = fig
                plt.close(fig)

        # Example: Bar plot for summary statistics
        if 'summary_stats' in data:
            df = pd.DataFrame(data['summary_stats']).T
            plt.figure(figsize=(12, 6))
            df['mean'].plot(kind='bar')
            plt.title('Bar Plot of Summary Statistics')
            plt.xlabel('Column')
            plt.ylabel('Mean Value')
            plt.xticks(rotation=45)
            plt.grid(True)
            fig = plt.gcf()
            visualizations['summary_stats'] = fig
            plt.close(fig)

        # Example: Heatmap for correlation matrix
        if 'correlation' in data:
            corr_matrix = pd.DataFrame(data['correlation'])
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
            plt.title('Correlation Heatmap')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.grid(True)
            fig = plt.gcf()
            visualizations['correlation_heatmap'] = fig
            plt.close(fig)

        # Example: Time series plot
        if 'time_series' in data:
            plt.figure(figsize=(14, 8))
            for period, series_data in data['time_series'].items():
                for column, values in series_data.items():
                    plt.plot(values.keys(), values.values(), label=f"{column} ({period})")
            plt.legend()
            plt.title('Time Series Analysis')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            plt.grid(True)
            fig = plt.gcf()
            visualizations['time_series'] = fig
            plt.close(fig)

        # Example: Forecast plot
        if 'forecasts' in data:
            plt.figure(figsize=(14, 8))
            for col, forecast in data['forecasts'].items():
                plt.plot(range(len(forecast)), forecast, label=f"{col} Forecast")
            plt.legend()
            plt.title('Forecasts')
            plt.xlabel('Future Time Periods')
            plt.ylabel('Predicted Value')
            plt.grid(True)
            fig = plt.gcf()
            visualizations['forecasts'] = fig
            plt.close(fig)

        # Example: Feature importance plot
        if 'predictive_analysis' in data:
            coeffs = data['predictive_analysis']['coefficients']
            plt.figure(figsize=(12, 6))
            plt.bar(coeffs.keys(), coeffs.values())
            plt.title('Feature Importance in Predictive Model')
            plt.xlabel('Feature')
            plt.ylabel('Coefficient')
            plt.xticks(rotation=45)
            plt.grid(True)
            fig = plt.gcf()
            visualizations['feature_importance'] = fig
            plt.close(fig)

    except Exception as e:
        st.error(f"Failed to generate visualizations: {str(e)}")

    return visualizations