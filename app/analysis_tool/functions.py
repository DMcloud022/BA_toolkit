import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import sqlite3
from io import StringIO
import docx2txt
import PyPDF2
from typing import Dict, Any
import re

def analyze_website(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        return analyze_text(text)
    except requests.RequestException as e:
        return {"error": f"Failed to fetch website: {str(e)}"}

def analyze_document(file) -> Dict[str, Any]:
    try:
        if file.type == "text/plain":
            text = StringIO(file.getvalue().decode("utf-8")).read()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(file)
        elif file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        else:
            return {"error": "Unsupported file type"}
        return analyze_text(text)
    except Exception as e:
        return {"error": f"Failed to analyze document: {str(e)}"}

def analyze_excel(file) -> Dict[str, Any]:
    try:
        df = pd.read_excel(file) if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" else pd.read_csv(file)
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
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "frequent_words": frequent_words.to_dict(),
        "summary": text[:500] + "..." if len(text) > 500 else text
    }

def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "summary_stats": df.describe().to_dict(),
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

    # Correlation analysis for numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        results["correlation"] = corr_matrix.to_dict()

    return results

def generate_visualizations(data: Dict[str, Any]) -> Dict[str, plt.Figure]:
    visualizations = {}

    if "summary_stats" in data:
        # Bar plot of column means
        fig, ax = plt.subplots()
        means = {k: v['mean'] for k, v in data["summary_stats"].items() if 'mean' in v}
        sns.barplot(x=list(means.keys()), y=list(means.values()), ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title("Mean Values by Column")
        visualizations["mean_values"] = fig

    if "correlation" in data:
        # Heatmap of correlation matrix
        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(data["correlation"]), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        visualizations["correlation_heatmap"] = fig

    if "time_series" in data:
        # Line plot of time series data
        fig, ax = plt.subplots()
        for period, series_data in data["time_series"].items():
            for column, values in series_data.items():
                ax.plot(values.keys(), values.values(), label=f"{column} ({period})")
        ax.legend()
        ax.set_title("Time Series Analysis")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        visualizations["time_series"] = fig

    return visualizations