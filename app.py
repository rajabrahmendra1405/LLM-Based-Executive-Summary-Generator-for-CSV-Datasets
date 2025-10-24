import streamlit as st
import pandas as pd
import datahelper
from datahelper import read_csv_any_encoding

# ------------------------------------------------------------
# Helper Function for Simple Statistical Q&A
# ------------------------------------------------------------
def get_answer_to_question(question, df):
    question = question.lower()
    try:
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
    except Exception:
        return "Unable to compute this question — column may not exist."

# ------------------------------------------------------------
# Initialize Streamlit Session State
# ------------------------------------------------------------
if "dataload" not in st.session_state:
    st.session_state.dataload = False
if "questions" not in st.session_state:
    st.session_state.questions = ["", "", ""]

def activate_dataload():
    st.session_state.dataload = True

# ------------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="LLM-Based Executive Summary Generator 🤖", layout="wide")
st.image("./image/banner2.png", width="stretch")
st.title("🤖 LLM-Based Executive Summary Generator for CSV Datasets")
st.divider()

# ------------------------------------------------------------
# Sidebar File Upload
# ------------------------------------------------------------
st.sidebar.subheader("📂 Load Your Data")
st.sidebar.divider()
loaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
load_btn = st.sidebar.button("Load Dataset", on_click=activate_dataload, use_container_width=True)

col_left, _, col_right = st.columns([4, 0.5, 7])

# ------------------------------------------------------------
# Load and Process CSV
# ------------------------------------------------------------
if st.session_state.dataload and loaded_file is not None:

    @st.cache_data
    def summarize():
        loaded_file.seek(0)
        return datahelper.summarize_csv(filename=loaded_file)

    data_summary = summarize()
    df = read_csv_any_encoding(loaded_file)

    # --------------------------------------------------------
    # LEFT PANEL: Dataset Summary
    # --------------------------------------------------------
    with col_left:
        st.info("📊 Data Summary")
        st.subheader("🔹 Data Preview")
        st.dataframe(data_summary["initial_data_sample"], width="stretch")
        st.divider()

        st.subheader("🔹 Column Descriptions")
        st.markdown(data_summary["column_descriptions"])
        st.divider()

        st.subheader("🔹 Missing Values")
        st.write(data_summary["missing_values"])
        st.divider()

        st.subheader("🔹 Duplicate Rows")
        st.write(data_summary["duplicate_values"])
        st.divider()

        st.subheader("🔹 Statistical Summary")
        st.dataframe(data_summary["essential_metrics"], width="stretch")

    # --------------------------------------------------------
    # RIGHT PANEL: Interactive Features
    # --------------------------------------------------------
    with col_right:
        st.info("🧠 Interactive Analysis")
        feature = st.text_input("Enter a column name to analyze:")
        analyze_btn = st.button("Examine Feature")
        st.divider()

        @st.cache_data
        def explore_feature(file_obj, variable):
            file_obj.seek(0)
            df_local = datahelper.get_dataframe(file_obj)
            if variable in df_local.columns:
                st.bar_chart(df_local, y=[variable])
                file_obj.seek(0)
                st.success(datahelper.analyze_trend(file_obj, variable))
            else:
                st.warning(f"The column '{variable}' was not found in the dataset.")

        if analyze_btn and feature:
            explore_feature(loaded_file, feature)

        # ----------------------------------------------------
        # Q&A Section
        # ----------------------------------------------------
        st.subheader("💬 Ask Questions About Your Data")
        for i in range(3):
            st.session_state.questions[i] = st.text_input(
                f"Question {i + 1}",
                value=st.session_state.questions[i],
                key=f"q_{i}",
            )
        if st.button("➕ Add Another Question"):
            st.session_state.questions.append("")

        for i, q in enumerate(st.session_state.questions[3:], start=4):
            st.session_state.questions[i] = st.text_input(
                f"Question {i}",
                value=q,
                key=f"q_extra_{i}",
            )

        st.divider()

        # ----------------------------------------------------
        # Generate Executive Summary Report
        # ----------------------------------------------------
        report_title = st.text_input("Enter report title:", "Executive Summary Report")
        if st.button("🚀 Generate Report"):
            report = f"Title: {report_title}\n\n"
            for q in st.session_state.questions:
                if q.strip():
                    ans = get_answer_to_question(q, df)
                    report += f"Q: {q}\nA: {ans}\n\n"

            st.text_area("Generated Executive Summary", report, height=300)
            st.download_button(
                "⬇️ Download Report",
                data=report,
                file_name="executive_summary.txt",
                mime="text/plain"
            )

else:
    st.info("📥 Please upload a CSV file and click 'Load Dataset' to begin.")
