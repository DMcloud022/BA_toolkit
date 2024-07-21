import streamlit as st
import functions
import pandas as pd
import matplotlib.pyplot as plt

def render_ui():
    st.set_page_config(page_title="Toyota Motors Philippines Data Analysis System", layout="wide")

    st.title("Toyota Motors Philippines Data Analysis System")

    menu = ["Upload File", "Analyze Website", "Analyze Document", "Analyze Excel", "Analyze Database"]
    choice = st.sidebar.selectbox("Select Function", menu)

    if choice == "Upload File":
        upload_file()
    elif choice == "Analyze Website":
        analyze_website()
    elif choice == "Analyze Document":
        analyze_document()
    elif choice == "Analyze Excel":
        analyze_excel()
    elif choice == "Analyze Database":
        analyze_database()

def upload_file():
    st.subheader("Upload File")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'txt', 'docx', 'pdf', 'db'])
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        st.success("File Successfully Uploaded")

def analyze_website():
    st.subheader("Analyze Website")
    url = st.text_input("Enter URL")
    if url:
        results = functions.analyze_website(url)
        display_results(results)

def analyze_document():
    st.subheader("Analyze Document")
    uploaded_file = st.file_uploader("Choose a document", type=['txt', 'docx', 'pdf'])
    if uploaded_file is not None:
        results = functions.analyze_document(uploaded_file)
        display_results(results)

def analyze_excel():
    st.subheader("Analyze Excel")
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        results = functions.analyze_excel(uploaded_file)
        display_results(results)

def analyze_database():
    st.subheader("Analyze Database")
    db_file = st.file_uploader("Upload SQLite database file", type=['db'])
    if db_file is not None:
        results = functions.analyze_database(db_file)
        for table, table_results in results.items():
            st.subheader(f"Table: {table}")
            display_results(table_results)

def display_results(results):
    if "error" in results:
        st.error(results["error"])
        return

    st.write("## Data Analysis Report")

    if "shape" in results:
        st.write(f"Dataset contains {results['shape'][0]} rows and {results['shape'][1]} columns.")

    if "summary_stats" in results:
        st.write("### Summary Statistics")
        st.dataframe(pd.DataFrame(results["summary_stats"]))

    if "null_counts" in results:
        st.write("### Null Value Counts")
        st.dataframe(pd.DataFrame.from_dict(results["null_counts"], orient='index', columns=['Count']))

    if "word_count" in results:
        st.write(f"Word count: {results['word_count']}")
        st.write(f"Sentence count: {results['sentence_count']}")

    if "frequent_words" in results:
        st.write("### Most Frequent Words")
        st.dataframe(pd.DataFrame.from_dict(results["frequent_words"], orient='index', columns=['Count']))

    st.write("## Data Visualization")
    visualizations = functions.generate_visualizations(results)
    for title, fig in visualizations.items():
        st.write(f"### {title.replace('_', ' ').title()}")
        st.pyplot(fig)

    st.write("## Conclusion and Summary")
    if "summary" in results:
        st.write(results["summary"])

    st.write("## Valuable Insights and Future Suggestions")
    st.write("1. Consider performing deeper analysis on columns with high variance.")
    st.write("2. Investigate any outliers in the numerical columns.")
    st.write("3. For future analysis, consider collecting more granular data to improve insights.")
    if "time_series" in results:
        st.write("4. The presence of time series data suggests potential for forecasting and trend analysis.")
    if "correlation" in results:
        st.write("5. Explore the relationships between highly correlated variables for potential insights.")