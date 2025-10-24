import streamlit as st
import datahelper
from datahelper import read_csv_any_encoding
import pandas as pd

# ------------------------------------------------------------
# Function to get an answer based on the question
# ------------------------------------------------------------
def get_answer_to_question(question, df):
    question = question.lower()
    if "average closing price" in question:
        return f"The average closing price is {df['Close'].mean():.2f}"
    elif "highest price" in question:
        return f"The highest price recorded is {df['High'].max():.2f}"
    elif "lowest price" in question:
        return f"The lowest price recorded is {df['Low'].min():.2f}"
    elif "trading volume" in question:
        return f"The average trading volume is {df['Volume'].mean():.2f}"
    elif "trend" in question and "price" in question:
        return "Trend analysis can be visualized through a line chart showing closing prices over time."
    elif "correlation" in question and "price" in question:
        return f"The correlation between opening and closing prices is {df['Open'].corr(df['Close']):.3f}"
    else:
        return "Answer not available for this question."

# ------------------------------------------------------------
# Initialize session state
# ------------------------------------------------------------
if "dataload" not in st.session_state:
    st.session_state.dataload = False

if "questions" not in st.session_state:
    st.session_state.questions = ["", "", ""]

def activate_dataload():
    """Activate data load state"""
    st.session_state.dataload = True

# ------------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="LLM-Based Executive Summary Generator 🤖", layout="wide")

st.image("./image/banner2.png", width="stretch")
st.title("🤖 LLM-Based Executive Summary Generator for CSV Datasets")
st.divider()

# ------------------------------------------------------------
# Sidebar: Load CSV
# ------------------------------------------------------------
st.sidebar.subheader("📂 Load Your Data")
st.sidebar.divider()

loaded_file = st.sidebar.file_uploader("Choose your CSV data file", type="csv")
load_data_btn = st.sidebar.button("Load Dataset", on_click=activate_dataload, use_container_width=True)

# ------------------------------------------------------------
# Layout Columns
# ------------------------------------------------------------
col_prework, col_gap, col_interaction = st.columns([4, 0.5, 7])

# ------------------------------------------------------------
# When data is loaded
# ------------------------------------------------------------
if st.session_state.dataload and loaded_file is not None:
    
    @st.cache_data
    def summarize():
        loaded_file.seek(0)
        return datahelper.summarize_csv(filename=loaded_file)

    data_summary = summarize()

    # Use robust CSV reader
    df = read_csv_any_encoding(loaded_file)

    # --------------------------------------------------------
    # Left Column: Summary of Data
    # --------------------------------------------------------
    with col_prework:
        st.info("📊 Data Summary")
        st.subheader("🔹 Sample of Data")
        st.dataframe(data_summary["initial_data_sample"], width="stretch")
        st.divider()

        st.subheader("🔹 Feature Descriptions")
        st.markdown(data_summary["column_descriptions"])
        st.divider()

        st.subheader("🔹 Missing Values")
        st.write(data_summary["missing_values"])
        st.divider()

        st.subheader("🔹 Duplicate Rows")
        st.write(data_summary["duplicate_values"])
        st.divider()

        st.subheader("🔹 Statistical Overview")
        st.dataframe(data_summary["essential_metrics"], width="stretch")

    # --------------------------------------------------------
    # Right Column: Interactive Analysis
    # --------------------------------------------------------
    with col_interaction:
        st.info("🧠 Interactive Analysis")

        feature_to_analyze = st.text_input("Enter a feature/column name to analyze:")
        examine_btn = st.button("Examine Feature")
        st.divider()

        @st.cache_data
        def explore_variable(data_file, variable):
            data_file.seek(0)
            dataframe = datahelper.get_dataframe(filename=data_file)
            
            if variable and variable in dataframe.columns:
                st.bar_chart(data=dataframe, y=[variable])
                st.divider()
            else:
                st.warning(f"The feature '{variable}' was not found in the dataset.")
                return

            # AI Trend Analysis
            data_file.seek(0)
            trend_response = datahelper.analyze_trend(filename=data_file, variable=variable)
            st.success(trend_response)

        if feature_to_analyze and examine_btn:
            explore_variable(data_file=loaded_file, variable=feature_to_analyze)

        # ----------------------------------------------------
        # Report Generation
        # ----------------------------------------------------
        st.subheader("📝 Generate Executive Summary Report")

        report_title = st.text_input("Enter report title:", "Data Analysis Summary Report")
        st.divider()

        st.subheader("Ask Questions About Your Data")

        # Show and allow editing of initial 3 questions
        for idx in range(3):
            st.session_state.questions[idx] = st.text_input(
                f"Question {idx + 1}",
                value=st.session_state.questions[idx],
                key=f"question_{idx + 1}",
            )

        if st.button("➕ Add Question"):
            st.session_state.questions.append("")

        for idx, question in enumerate(st.session_state.questions[3:], start=4):
            st.session_state.questions[idx] = st.text_input(
                f"Question {idx}",
                value=question,
                key=f"question_{idx}",
            )

        st.divider()

        # ----------------------------------------------------
        # Generate the Report
        # ----------------------------------------------------
        if st.button("🚀 Generate Report"):
            report_content = f"Title: {report_title}\n\n"

            for question in st.session_state.questions:
                if question.strip():
                    answer = get_answer_to_question(question, df)
                    report_content += f"Q: {question}\nA: {answer}\n\n"

            st.text_area("Generated Executive Summary", report_content, height=300)
            st.download_button(
                label="⬇️ Download Report",
                data=report_content,
                file_name="executive_summary.txt",
                mime="text/plain",
            )

else:
    st.info("📥 Please upload and load a CSV file to begin.")
