import streamlit as st
import functions
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

def render_ui():
    st.set_page_config(page_title="Data Analysis System", layout="wide")

    # Custom CSS for improved styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #e50000;
        color: white;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff0000;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stSelectbox, .stTextInput, .stFileUploader {
        background-color: #ffffff;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border: 1px solid #e50000;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
    }
    .css-1d391kg {
        padding-top: 3rem;
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
    menu_icons = {
        "Upload File": "📁", "Analyze Website": "🌐", "Analyze Document": "📄",
        "Analyze Excel": "📊", "Analyze Database": "🗄️"
    }

    with st.sidebar:
        st.subheader("Navigation")
        for item in menu:
            if st.button(f"{menu_icons[item]} {item}", key=item):
                st.session_state.choice = item

    choice = st.session_state.get('choice', "Upload File")

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
        st.json(file_details)
        st.success("File Successfully Uploaded")

def analyze_website():
    st.subheader("🌐 Analyze Website")
    url = st.text_input("Enter URL")
    if url and st.button("Analyze"):
        with st.spinner("Analyzing website..."):
            results = functions.analyze_website(url)
        display_results(results)

def analyze_document():
    st.subheader("📄 Analyze Document")
    uploaded_file = st.file_uploader("Choose a document", type=['txt', 'docx', 'pdf'])
    if uploaded_file is not None and st.button("Analyze"):
        with st.spinner("Analyzing document..."):
            results = functions.analyze_document(uploaded_file)
        display_results(results)

def analyze_excel():
    st.subheader("📊 Analyze Excel")
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'csv'])
    if uploaded_file is not None and st.button("Analyze"):
        with st.spinner("Analyzing Excel file..."):
            results = functions.analyze_excel(uploaded_file)
        display_results(results)

def analyze_database():
    st.subheader("🗄️ Analyze Database")
    db_file = st.file_uploader("Upload SQLite database file", type=['db'])
    if db_file is not None and st.button("Analyze"):
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
        df = pd.DataFrame(results["summary_stats"]).T
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns),
                        fill_color='#e50000',
                        align='left',
                        font=dict(color='white', size=12)),
            cells=dict(values=[df[col] for col in df.columns],
                       fill_color='lavender',
                       align='left'))
        ])
        st.plotly_chart(fig, use_container_width=True)

    if "null_counts" in results:
        st.write("### 🔍 Null Value Counts")
        df = pd.DataFrame.from_dict(results["null_counts"], orient='index', columns=['Count'])
        fig = px.bar(df, x=df.index, y='Count', title='Null Value Counts')
        fig.update_layout(xaxis_title='Column', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    if "word_count" in results:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Word Count", results['word_count'])
        with col2:
            st.metric("Sentence Count", results['sentence_count'])

    if "frequent_words" in results:
        st.write("### 🔤 Most Frequent Words")
        df = pd.DataFrame.from_dict(results["frequent_words"], orient='index', columns=['Count'])
        fig = px.bar(df, x=df.index, y='Count', title='Most Frequent Words')
        fig.update_layout(xaxis_title='Word', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    st.write("## 📉 Data Visualization")
    visualizations = functions.generate_visualizations(results)
    for title, fig in visualizations.items():
        st.write(f"### {title.replace('_', ' ').title()}")
        st.pyplot(fig)

    if "predictive_analysis" in results:
        st.write("## 🔮 Predictive Analysis")
        pa = results["predictive_analysis"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{pa['mse']:.4f}")
        with col2:
            st.metric("R-squared Score", f"{pa['r2']:.4f}")
        st.write(f"**Target Variable:** {pa['target']}")
        st.write(f"**Features:** {', '.join(pa['features'])}")

    st.write("## 📝 Conclusion and Summary")
    if "ai_analysis" in results:
        st.info(results["ai_analysis"])
    else:
        st.info("AI-powered analysis is not available for this data.")

    st.write("## 💡 Valuable Insights and Future Suggestions")
    insights = [
        "Consider performing deeper analysis on columns with high variance.",
        "Investigate any outliers in the numerical columns.",
        "For future analysis, consider collecting more granular data to improve insights."
    ]
    if "time_series" in results:
        insights.append("The presence of time series data suggests potential for forecasting and trend analysis.")
    if "correlation" in results:
        insights.append("Explore the relationships between highly correlated variables for potential insights.")
    if "predictive_analysis" in results:
        insights.append("The predictive model shows promise. Consider refining it with more advanced techniques or additional features.")
    
    for i, insight in enumerate(insights, 1):
        st.write(f"{i}. {insight}")

    # Add interactivity to elements
    st.markdown("""
    <script>
    const elements = document.querySelectorAll('.stButton>button, .stSelectbox, .stTextInput>div>div>input');
    elements.forEach(el => {
        el.addEventListener('mouseover', () => {
            el.style.transform = 'scale(1.05)';
            el.style.transition = 'all 0.3s ease';
        });
        el.addEventListener('mouseout', () => {
            el.style.transform = 'scale(1)';
        });
    });
    </script>
    """, unsafe_allow_html=True)

