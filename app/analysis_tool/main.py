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
    generate_visualizations
)

def main():
    st.title("Business Analysis Toolkit")

    menu = ["Upload File", "Analyze Website"]
    choice = st.sidebar.selectbox("Choose an Option", menu)

    if choice == "Upload File":
        uploaded_file = st.file_uploader("Choose a file", type=[
            "csv", "xlsx", "txt", "pdf", "docx", "sqlite",
            "json", "xml", "parquet", "yaml"
        ])

        if uploaded_file:
            try:
                # Process various file types
                analysis_results = None
                if uploaded_file.type in ["text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/pdf"]:
                    analysis_results = analyze_document(uploaded_file)
                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]:
                    analysis_results = analyze_excel(uploaded_file)
                elif uploaded_file.type == "application/x-sqlite3":
                    analysis_results = analyze_database(uploaded_file)
                elif uploaded_file.type == "application/json":
                    analysis_results = analyze_json(uploaded_file)
                elif uploaded_file.type == "application/xml":
                    analysis_results = analyze_xml(uploaded_file)
                elif uploaded_file.type == "application/x-parquet":
                    analysis_results = analyze_parquet(uploaded_file)
                elif uploaded_file.type == "text/yaml":
                    analysis_results = analyze_yaml(uploaded_file)
                else:
                    st.error("Unsupported file type")
                    return

                if "error" in analysis_results:
                    st.error(analysis_results["error"])
                else:
                    st.write("### Analysis Results")
                    st.write(analysis_results)

                    st.write("### Data Visualizations")
                    visualizations = generate_visualizations(analysis_results)
                    if visualizations:
                        for title, fig in visualizations.items():
                            st.write(f"#### {title.replace('_', ' ').title()}")
                            st.pyplot(fig)
                    else:
                        st.warning("No visualizations available.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif choice == "Analyze Website":
        st.subheader("Website Analysis")
        url = st.text_input("Enter URL", placeholder="https://example.com")

        if url:
            if st.button("Analyze Website"):
                with st.spinner("Analyzing website..."):
                    try:
                        analysis_results = analyze_website(url)

                        if "error" in analysis_results:
                            st.error(analysis_results["error"])
                        else:
                            # Display website metadata and AI-enhanced analysis
                            st.write("### Website Metadata")
                            st.write(f"**Title:** {analysis_results.get('title', 'N/A')}")
                            st.write(f"**Description:** {analysis_results.get('description', 'N/A')}")
                            st.write(f"**Keywords:** {analysis_results.get('keywords', 'N/A')}")
                            
                            st.write("### Text Analysis Results")
                            st.write(f"**Word Count:** {analysis_results.get('word_count', 'N/A')}")
                            st.write(f"**Sentence Count:** {analysis_results.get('sentence_count', 'N/A')}")
                            
                            st.write("### Frequent Words")
                            st.write(analysis_results.get("frequent_words", {}))

                            st.write("### Text Summary")
                            st.write(analysis_results.get("summary", "No summary available."))
                            
                            st.write("### AI Analysis")
                            st.write(analysis_results.get("ai_analysis", "No AI analysis available."))

                            # Generate and display visualizations
                            st.write("### Data Visualizations")
                            visualizations = generate_visualizations(analysis_results)
                            if visualizations:
                                for title, fig in visualizations.items():
                                    st.write(f"#### {title.replace('_', ' ').title()}")
                                    st.pyplot(fig)
                            else:
                                st.warning("No visualizations available.")

                    except Exception as e:
                        st.error(f"An error occurred during website analysis: {str(e)}")

if __name__ == "__main__":
    main()
