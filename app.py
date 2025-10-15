import streamlit as st
import datahelper
import pandas as pd

# Function to get an answer based on the question
def get_answer_to_question(question, df):
    # Example logic for common questions
    if "average closing price" in question.lower():
        return df["Close"].mean()
    elif "highest price" in question.lower():
        return df["High"].max()
    elif "lowest price" in question.lower():
        return df["Low"].min()
    elif "trading volume" in question.lower():
        return df["Volume"].mean()
    elif "trend" in question.lower() and "price" in question.lower():
        return "Trend analysis can be done with a line plot."
    elif "correlation" in question.lower() and "price" in question.lower():
        return df["Open"].corr(df["Close"])
    else:
        return "Answer not available for this question."

# Initialize session state for data load status
if "dataload" not in st.session_state:
    st.session_state.dataload = False

# Initialize session state for dynamic questions
if "questions" not in st.session_state:
    st.session_state.questions = ["", "", ""]  # Initialize with 3 empty questions

def activate_dataload():
    """Activate data load state"""
    st.session_state.dataload = True

# Set page configuration for Streamlit
st.set_page_config(page_title="Data Analyzer 🤖", layout="wide")

# Display header with an image
st.image("./image/banner2.png", use_container_width=True)
st.title("🤖 LLM Agent Data Analyzer")
st.divider()

# Sidebar for loading data
st.sidebar.subheader("Load your data")
st.sidebar.divider()

# File uploader widget for CSV files
loaded_file = st.sidebar.file_uploader("Choose your CSV data", type="csv")

# Button to load data and trigger processing
load_data_btn = st.sidebar.button(
    label="Load", on_click=activate_dataload, use_container_width=True
)

# Layout for displaying data and analysis results
col_prework, col_dummy, col_interaction = st.columns([4, 1, 7])

if st.session_state.dataload:
    # Cache function to summarize the CSV data
    @st.cache_data
    def summarize():
        loaded_file.seek(0)  # Reset file pointer to the beginning
        data_summary = datahelper.summarize_csv(filename=loaded_file)  # Get summary data
        return data_summary

    # Retrieve the summarized data
    data_summary = summarize()

    # Load the dataset into a DataFrame
    df = pd.read_csv(loaded_file)

    with col_prework:
        st.info("Data Summary")
        st.subheader("Sample of Data")
        st.write(data_summary["initial_data_sample"])
        st.divider()

        st.subheader("Features of Data")
        st.write(data_summary["column_descriptions"])
        st.divider()

        st.subheader("Missing Values in Data")
        st.write(data_summary["missing_values"])
        st.divider()

        st.subheader("Duplicate Values in Data")
        st.write(data_summary["duplicate_values"])  # Fixed typo here
        st.divider()

        st.subheader("Summary Statistics of Data")
        st.write(data_summary["essential_metrics"])

    with col_dummy:
        st.empty()

    with col_interaction:
        st.info("Interaction Section")

        # Feature to analyze (first section of interaction)
        feature_to_analyze = st.text_input(label="Which feature do you want to analyze?", key="feature_to_analyze")
        examine_btn = st.button("Examine Feature")  # Trigger visualization of the feature
        st.divider()

        # Visualize the feature when 'Examine' button is clicked
        @st.cache_data
        def explore_variable(data_file, variable):
            data_file.seek(0)  # Reset file pointer
            dataframe = datahelper.get_dataframe(filename=data_file)  # Get full dataframe
            
            if variable and variable in dataframe.columns:  # Check if variable exists in dataframe
                st.bar_chart(data=dataframe, y=[variable])  # Bar chart for selected feature
                st.divider()
            else:
                st.warning(f"The feature '{variable}' was not found in the dataset.")

            # Analyze trend for the selected variable
            data_file.seek(0)  # Reset file pointer
            trend_response = datahelper.analyze_trend(
                filename=loaded_file, variable=variable
            )
            st.success(trend_response)
            return

        # Trigger explore variable if button clicked or variable provided
        if feature_to_analyze and examine_btn:
            explore_variable(data_file=loaded_file, variable=feature_to_analyze)

        # Line break
        st.write("")  # For visual spacing

        # Title input for the analysis report
        report_title = st.text_input("Please enter a title for your analysis report", "Health Care Data Analysis Report", key="report_title")
        st.write("")  # Line break for visual spacing

        # Analysis Questions input
        st.subheader("Analysis Questions")
        
        # Display the first 3 questions (stored in session state)
        for idx in range(3):
            st.session_state.questions[idx] = st.text_input(f"Question {idx + 1}", value=st.session_state.questions[idx], key=f"question_{idx + 1}")

        st.write("")  # Line break for visual spacing

        # Add Question button
        if st.button("Add Question"):
            st.session_state.questions.append("")  # Add an empty string to the list for the new question

        # Display any dynamically added questions (after the initial 3)
        for idx, question in enumerate(st.session_state.questions[3:], start=4):
            st.session_state.questions[idx] = st.text_input(f"Question {idx}", value=question, key=f"question_{idx}")

        st.write("")  # Line break for visual spacing

        # Button to generate the analysis report
        generate_report_btn = st.button("Generate Analysis Report")

        # Function to generate and provide download link for the report
        if generate_report_btn:
            report_content = f"Title: {report_title}\n\n"
            
            # Generate the report for only the questions entered
            for i, question in enumerate(st.session_state.questions):
                if question.strip():  # Only process non-empty questions
                    answer = get_answer_to_question(question, df)  # Get the answer for each question
                    report_content += f"{question}\nAnswer: {answer}\n\n"  # Add question and answer

            # Create the analysis report
            st.text_area("Generated Analysis Report", report_content, height=300)

            # Provide a download option for the report
            st.download_button(
                label="Download Report",
                data=report_content,
                file_name="analysis_report.txt",
                mime="text/plain"
            )

        st.write("")  # Line break for visual spacing
