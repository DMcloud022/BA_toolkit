import streamlit as st
from functions import (
    analyze_website,
    analyze_document,
    analyze_excel,
    analyze_database,
    analyze_json,
    analyze_xml,
    analyze_parquet,
    analyze_yaml,
    generate_visualizations,
    generate_groq_insights,
    generate_future_predictions
)
import pandas as pd
import plotly.express as px

def main():
    st.set_page_config(page_title="Business Analysis Toolkit", layout="wide")
    st.title("Data Analysis Toolkit")

    menu = ["Analyze File", "Analyze Website"]
    choice = st.sidebar.selectbox("Choose an Option", menu)
    

    if choice == "Analyze File":
        file_analysis()
    elif choice == "Analyze Website":
        website_analysis()

def file_analysis():
    uploaded_file = st.file_uploader("Choose a file", type=[
        "csv", "xlsx", "txt", "pdf", "docx", "sqlite",
        "json", "xml", "parquet", "yaml"
    ])

    if uploaded_file:
        try:
            analysis_results = process_file(uploaded_file)
            if analysis_results:
                display_results(analysis_results)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def process_file(uploaded_file):
    file_type = uploaded_file.type
    if file_type in ["text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/pdf"]:
        return analyze_document(uploaded_file)
    elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]:
        return analyze_excel(uploaded_file)
    elif file_type == "application/x-sqlite3":
        return analyze_database(uploaded_file)
    elif file_type == "application/json":
        return analyze_json(uploaded_file)
    elif file_type == "application/xml":
        return analyze_xml(uploaded_file)
    elif file_type == "application/x-parquet":
        return analyze_parquet(uploaded_file)
    elif file_type == "text/yaml":
        return analyze_yaml(uploaded_file)
    else:
        st.error("Unsupported file type")
        return None

def website_analysis():
    st.subheader("Website Analysis")
    url = st.text_input("Enter URL", placeholder="https://example.com")

    if url and st.button("Analyze Website"):
        with st.spinner("Analyzing website..."):
            try:
                analysis_results = analyze_website(url)
                if analysis_results:
                    display_results(analysis_results, is_website=True)
            except Exception as e:
                st.error(f"An error occurred during website analysis: {str(e)}")

def display_results(results, is_website=False):
    if "error" in results:
        st.error(results["error"])
        return

    st.subheader("Analysis Results")

    if is_website:
        display_website_metadata(results)

    display_text_analysis(results)
    display_data_analysis(results)
    display_visualizations(results)
    display_ai_analysis(results)
    display_insights_and_predictions(results)

def display_website_metadata(results):
    st.write("### Website Metadata")
    st.write(f"**Title:** {results.get('title', 'N/A')}")
    st.write(f"**Description:** {results.get('description', 'N/A')}")
    st.write(f"**Keywords:** {results.get('keywords', 'N/A')}")

def display_text_analysis(results):
    if 'word_count' in results or 'sentence_count' in results:
        st.write("### Text Analysis Results")
        st.write(f"**Word Count:** {results.get('word_count', 'N/A')}")
        st.write(f"**Sentence Count:** {results.get('sentence_count', 'N/A')}")

    if 'frequent_words' in results:
        st.write("### Frequent Words")
        fig = px.bar(x=list(results['frequent_words'].keys()), y=list(results['frequent_words'].values()),
                     labels={'x': 'Word', 'y': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)

    if 'summary' in results:
        st.write("### Text Summary")
        st.write(results['summary'])

def display_data_analysis(results):
    if 'shape' in results:
        st.write(f"Dataset shape: {results['shape'][0]} rows, {results['shape'][1]} columns")

    if 'summary_stats' in results:
        st.write("### Summary Statistics")
        st.dataframe(pd.DataFrame(results['summary_stats']))

    if 'null_counts' in results:
        st.write("### Null Value Counts")
        st.dataframe(pd.DataFrame.from_dict(results['null_counts'], orient='index', columns=['Count']))

def display_visualizations(results):
    st.write("### Data Visualizations")
    visualizations = generate_visualizations(results)
    if visualizations:
        for title, fig in visualizations.items():
            st.write(f"#### {title.replace('_', ' ').title()}")
            st.pyplot(fig)
    else:
        st.info("No visualizations available.")

def display_ai_analysis(results):
    if 'ai_analysis' in results:
        st.write("### AI Analysis")
        st.write(results['ai_analysis'])

def display_insights_and_predictions(results):
    insights = generate_groq_insights(results)
    predictions = generate_future_predictions(results)

    st.write("### Insights")
    st.write("#### Business Insights")
    st.write(insights['groq_insights'])
    st.write("#### Future Predictions and Recommendations")
    st.write(predictions['future_predictions'])

if __name__ == "__main__":
    main()