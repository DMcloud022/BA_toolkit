import streamlit as st
import functions
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def render_ui():
    st.set_page_config(page_title="Data Analysis System", layout="wide")

    # Custom CSS for improved styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #e50000;
        color: white;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff0000;
    }
    .stSelectbox, .stTextInput, .stFileUploader {
        background-color: #ffffff;
        border-radius: 5px;
    }
    .stTextInput>input {
        border: 1px solid #e50000;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        image = Image.open("logo.png")
        st.image(image, width=100)
    with col2:
        st.title("Data Analysis System")

    menu = ["Upload File", "Analyze Website", "Analyze Document", "Analyze Excel", "Analyze Database"]
    choice = st.sidebar.selectbox("Select Function", menu, key="sidebar")

    # Add icons to menu items
    menu_icons = {
        "Upload File": "📁",
        "Analyze Website": "🌐",
        "Analyze Document": "📄",
        "Analyze Excel": "📊",
        "Analyze Database": "🗄️"
    }

    for item in menu:
        if st.sidebar.button(f"{menu_icons[item]} {item}"):
            choice = item

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
    st.subheader("📁 Upload File")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'txt', 'docx', 'pdf', 'db'])
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        st.success("File Successfully Uploaded")

def analyze_website():
    st.subheader("🌐 Analyze Website")
    url = st.text_input("Enter URL")
    if url:
        with st.spinner("Analyzing website..."):
            results = functions.analyze_website(url)
        display_results(results)

def analyze_document():
    st.subheader("📄 Analyze Document")
    uploaded_file = st.file_uploader("Choose a document", type=['txt', 'docx', 'pdf'])
    if uploaded_file is not None:
        with st.spinner("Analyzing document..."):
            results = functions.analyze_document(uploaded_file)
        display_results(results)

def analyze_excel():
    st.subheader("📊 Analyze Excel")
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        with st.spinner("Analyzing Excel file..."):
            results = functions.analyze_excel(uploaded_file)
        display_results(results)

def analyze_database():
    st.subheader("🗄️ Analyze Database")
    db_file = st.file_uploader("Upload SQLite database file", type=['db'])
    if db_file is not None:
        with st.spinner("Analyzing database..."):
            results = functions.analyze_database(db_file)
        for table, table_results in results.items():
            st.subheader(f"Table: {table}")
            display_results(table_results)

def display_results(results):
    if "error" in results:
        st.error(results["error"])
        return

    st.write("## 📊 Data Analysis Report")

    if "shape" in results:
        st.info(f"Dataset contains {results['shape'][0]} rows and {results['shape'][1]} columns.")

    if "summary_stats" in results:
        st.write("### 📈 Summary Statistics")
        st.dataframe(pd.DataFrame(results["summary_stats"]).T)

    if "null_counts" in results:
        st.write("### 🔍 Null Value Counts")
        st.dataframe(pd.DataFrame.from_dict(results["null_counts"], orient='index', columns=['Count']))

    if "word_count" in results:
        st.write(f"**Word Count:** {results['word_count']}")
        st.write(f"**Sentence Count:** {results['sentence_count']}")

    if "frequent_words" in results:
        st.write("### 🔤 Most Frequent Words")
        st.dataframe(pd.DataFrame.from_dict(results["frequent_words"], orient='index', columns=['Count']))

    st.write("## 📉 Data Visualization")
    visualizations = functions.generate_visualizations(results)
    for title, fig in visualizations.items():
        st.write(f"### {title.replace('_', ' ').title()}")
        st.pyplot(fig)

    if "predictive_analysis" in results:
        st.write("## 🔮 Predictive Analysis")
        pa = results["predictive_analysis"]
        st.write(f"**Target Variable:** {pa['target']}")
        st.write(f"**Features:** {', '.join(pa['features'])}")
        st.write(f"**Mean Squared Error:** {pa['mse']:.4f}")
        st.write(f"**R-squared Score:** {pa['r2']:.4f}")

    st.write("## 📝 Conclusion and Summary")
    if "ai_analysis" in results:
        st.write(results["ai_analysis"])
    else:
        st.write("AI-powered analysis is not available for this data.")

    st.write("## 💡 Valuable Insights and Future Suggestions")
    st.write("1. Consider performing deeper analysis on columns with high variance.")
    st.write("2. Investigate any outliers in the numerical columns.")
    st.write("3. For future analysis, consider collecting more granular data to improve insights.")
    if "time_series" in results:
        st.write("4. The presence of time series data suggests potential for forecasting and trend analysis.")
    if "correlation" in results:
        st.write("5. Explore the relationships between highly correlated variables for potential insights.")
    if "predictive_analysis" in results:
        st.write("6. The predictive model shows promise. Consider refining it with more advanced techniques or additional features.")

    # Add hover effects to buttons and interactive elements
    st.markdown("""
    <style>
    .stButton>button:hover {
        background-color: #ff0000;
        transition: all 0.3s ease;
    }
    .stSelectbox:hover {
        border-color: #e50000;
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)
