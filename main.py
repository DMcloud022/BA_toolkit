import streamlit as st
import pandas as pd
from functions import (
    analyze_website,
    analyze_document,
    analyze_excel,
    analyze_database,
    analyze_json,
    analyze_xml,
    analyze_parquet,
    analyze_yaml,
    generate_visualizations
)
import config

def main():
    st.set_page_config(
        page_title="Business Analysis Toolkit", 
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Business Analysis Toolkit")
    st.markdown("---")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    menu_options = {
        "📁 Upload File": "upload",
        "🌐 Analyze Website": "website",
        "📊 Dashboard": "dashboard"
    }
    
    choice = st.sidebar.selectbox("Choose an Option", list(menu_options.keys()))
    selected_option = menu_options[choice]

    if selected_option == "upload":
        handle_file_upload()
    elif selected_option == "website":
        handle_website_analysis()
    elif selected_option == "dashboard":
        show_dashboard()

def handle_file_upload():
    st.header("📁 File Analysis")
    
    # File uploader with improved file type detection
    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=list(sum(config.SUPPORTED_FILE_TYPES.values(), [])),
        help="Supported formats: CSV, Excel, PDF, Word, JSON, XML, YAML, Parquet, SQLite"
    )

    if uploaded_file:
        # Display file information
        file_info = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB",
            "File Type": uploaded_file.type or "Unknown"
        }
        
        st.info("📋 **File Information**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Filename", file_info["Filename"])
        col2.metric("Size", file_info["File Size"])
        col3.metric("Type", file_info["File Type"])

        # Check file size
        if uploaded_file.size > config.MAX_FILE_SIZE:
            st.error(f"File too large. Maximum size is {config.MAX_FILE_SIZE / 1024 / 1024:.0f}MB")
            return

        # Process file based on type
        with st.spinner("🔄 Analyzing file..."):
            try:
                analysis_results = process_file(uploaded_file)
                display_analysis_results(analysis_results, uploaded_file.name)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

def process_file(uploaded_file):
    """Process uploaded file based on its type."""
    filename = uploaded_file.name.lower()
    file_type = uploaded_file.type
    
    # Document files
    if any(filename.endswith(ext) for ext in config.SUPPORTED_FILE_TYPES['document']) or \
       file_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        return analyze_document(uploaded_file)
    
    # Text files
    elif any(filename.endswith(ext) for ext in config.SUPPORTED_FILE_TYPES['text']) or \
         file_type == "text/plain":
        return analyze_document(uploaded_file)
    
    # Spreadsheet files
    elif any(filename.endswith(ext) for ext in config.SUPPORTED_FILE_TYPES['spreadsheet']) or \
         file_type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        return analyze_excel(uploaded_file)
    
    # Database files
    elif any(filename.endswith(ext) for ext in config.SUPPORTED_FILE_TYPES['database']):
        return analyze_database(uploaded_file)
    
    # Data files
    elif any(filename.endswith(ext) for ext in config.SUPPORTED_FILE_TYPES['data']):
        if filename.endswith('.json'):
            return analyze_json(uploaded_file)
        elif filename.endswith(('.xml', '.html')):
            return analyze_xml(uploaded_file)
        elif filename.endswith('.parquet'):
            return analyze_parquet(uploaded_file)
        elif filename.endswith(('.yaml', '.yml')):
            return analyze_yaml(uploaded_file)
    
    return {"error": f"Unsupported file type: {file_type}"}

def handle_website_analysis():
    st.header("🌐 Website Analysis")
    
    url = st.text_input(
        "Enter Website URL", 
        placeholder="https://example.com",
        help="Enter a complete URL including http:// or https://"
    )

    if url:
        if not url.startswith(('http://', 'https://')):
            st.warning("⚠️ Please include http:// or https:// in the URL")
            return
            
        if st.button("🔍 Analyze Website", type="primary"):
            with st.spinner("🔄 Analyzing website..."):
                try:
                    analysis_results = analyze_website(url)
                    display_website_results(analysis_results)
                    
                except Exception as e:
                    st.error(f"❌ Website analysis failed: {str(e)}")

def display_analysis_results(results, filename):
    """Display comprehensive analysis results."""
    if "error" in results:
        st.error(f"❌ {results['error']}")
        return

    # Main metrics
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
    """Display overview information."""
    st.subheader(f"📄 File: {filename}")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    if "shape" in results:
        col1.metric("Rows", results["shape"][0])
        col2.metric("Columns", results["shape"][1])
        
    if "memory_usage_mb" in results:
        col3.metric("Memory Usage", f"{results['memory_usage_mb']} MB")
    elif "word_count" in results:
        col1.metric("Words", results["word_count"])
        if "sentence_count" in results:
            col2.metric("Sentences", results["sentence_count"])
    
    # Column information
    if "columns" in results:
        st.subheader("📋 Columns")
        cols_df = pd.DataFrame({
            'Column': results["columns"],
            'Data Type': [results["dtypes"].get(col, 'Unknown') for col in results["columns"]],
            'Null Count': [results.get("null_counts", {}).get(col, 0) for col in results["columns"]],
            'Null %': [f"{results.get('null_percentage', {}).get(col, 0):.1f}%" for col in results["columns"]]
        })
        st.dataframe(cols_df, use_container_width=True)

def display_statistics_tab(results):
    """Display statistical information."""
    if "summary_stats" in results:
        st.subheader("📊 Summary Statistics")
        stats_df = pd.DataFrame(results["summary_stats"])
        st.dataframe(stats_df, use_container_width=True)
    
    if "correlation" in results and "high_correlations" in results["correlation"]:
        high_corr = results["correlation"]["high_correlations"]
        if high_corr:
            st.subheader("🔗 High Correlations (>0.7)")
            corr_df = pd.DataFrame(list(high_corr.items()), columns=['Variables', 'Correlation'])
            st.dataframe(corr_df, use_container_width=True)
    
    if "predictive_analysis" in results:
        st.subheader("🎯 Predictive Analysis Results")
        pa = results["predictive_analysis"]
        col1, col2 = st.columns(2)
        col1.metric("R² Score", f"{pa['r2']:.3f}")
        col2.metric("MSE", f"{pa['mse']:.3f}")
        
        st.write("**Feature Coefficients:**")
        coeff_df = pd.DataFrame(list(pa["coefficients"].items()), 
                               columns=['Feature', 'Coefficient'])
        st.dataframe(coeff_df, use_container_width=True)

def display_ai_insights_tab(results):
    """Display AI-generated insights."""
    if "ai_analysis" in results and results["ai_analysis"]:
        st.subheader("🤖 AI-Powered Analysis")
        st.write(results["ai_analysis"])
    else:
        st.info("AI analysis not available for this file type.")

def display_visualizations_tab(results):
    """Display visualizations."""
    st.subheader("📉 Data Visualizations")
    
    with st.spinner("🎨 Generating visualizations..."):
        visualizations = generate_visualizations(results)
        
    if visualizations:
        for title, fig in visualizations.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("No visualizations available for this data type.")

def display_website_results(results):
    """Display website analysis results."""
    if "error" in results:
        st.error(f"❌ {results['error']}")
        return
    
    st.success("✅ Website analysis completed!")
    
    # Website metadata
    st.subheader("🏷️ Website Metadata")
    col1, col2 = st.columns(2)
    col1.write(f"**Title:** {results.get('title', 'N/A')}")
    col1.write(f"**URL:** {results.get('url', 'N/A')}")
    col2.write(f"**Content Length:** {results.get('content_length', 0)} characters")
    
    if results.get('description', 'No description found') != 'No description found':
        st.write(f"**Description:** {results['description']}")
    
    # Text analysis
    if "word_count" in results:
        st.subheader("📊 Content Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Words", results["word_count"])
        col2.metric("Sentences", results.get("sentence_count", 0))
        col3.metric("Avg Words/Sentence", results.get("avg_words_per_sentence", 0))
    
    # Frequent words
    if "frequent_words" in results:
        st.subheader("🔤 Most Frequent Words")
        freq_df = pd.DataFrame(list(results["frequent_words"].items()), 
                              columns=['Word', 'Frequency'])
        st.dataframe(freq_df, use_container_width=True)
    
    # AI analysis
    if "ai_analysis" in results:
        st.subheader("🤖 AI Analysis")
        st.write(results["ai_analysis"])

def show_dashboard():
    """Show dashboard with usage statistics."""
    st.header("📊 Dashboard")
    st.info("Dashboard functionality coming soon!")
    
    # Placeholder for future dashboard features
    st.subheader("📈 Usage Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Files Analyzed", "0")
    col2.metric("Websites Analyzed", "0") 
    col3.metric("Total Insights", "0")
    col4.metric("API Calls", "0")

if __name__ == "__main__":
    main() 