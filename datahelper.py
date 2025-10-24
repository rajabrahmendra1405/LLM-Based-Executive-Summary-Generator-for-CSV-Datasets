import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)

# ------------------------------------------------------------
# Load environment variables (.env for local, st.secrets for Streamlit Cloud)
# ------------------------------------------------------------
load_dotenv()

# Try to load from Streamlit secrets first (Cloud), else .env (Local)
api_key = None

# Try from Streamlit Cloud secrets
try:
    api_key = st.secrets["groq"]["api_key"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

# Validate API key
if not api_key:
    raise ValueError(
        "Groq API key not found. Please add it to Streamlit secrets (in Cloud) "
        "or define it in your local .env file."
    )

# ------------------------------------------------------------
# Initialize Groq LLM (LLaMA 3.3 70B)
# ------------------------------------------------------------
try:
    llm_groq = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0
    )
except Exception as e:
    raise ValueError(f"Error initializing Groq client: {str(e)}")

selected_llm = llm_groq

# ------------------------------------------------------------
# Summarize CSV Data
# ------------------------------------------------------------
def summarize_csv(filename):
    """Generate a high-level summary of the uploaded CSV file."""
    df = pd.read_csv(filename, low_memory=False)

    pandas_agent = create_pandas_dataframe_agent(
        llm=selected_llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    data_summary = {
        "initial_data_sample": df.head(),
        "column_descriptions": pandas_agent.run(
            "Create a markdown table describing each column name and its meaning."
        ),
        "missing_values": pandas_agent.run(
            "Report if there are any missing values and how many. "
            "Respond like: 'There are X missing values in total.'"
        ),
        "duplicate_values": pandas_agent.run(
            "Report if there are any duplicate rows and how many. "
            "Respond like: 'There are X duplicate rows in total.'"
        ),
        "essential_metrics": df.describe(include='all')
    }

    return data_summary


# ------------------------------------------------------------
# Get DataFrame
# ------------------------------------------------------------
def get_dataframe(filename):
    """Return the loaded DataFrame."""
    try:
        return pd.read_csv(filename, low_memory=False)
    except Exception as e:
        raise ValueError(f"Error loading DataFrame from {filename}: {str(e)}")


# ------------------------------------------------------------
# Analyze Trend
# ------------------------------------------------------------
def analyze_trend(filename, variable):
    """Interpret trend of a specific variable or column."""
    try:
        df = pd.read_csv(filename, low_memory=False)
        pandas_agent = create_pandas_dataframe_agent(
            llm=selected_llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )

        return pandas_agent.run(
            f"Provide a short executive interpretation of trends in the column '{variable}'. "
            "Consider the dataset rows as chronological and provide data-driven reasoning."
        )
    except Exception as e:
        raise ValueError(f"Error analyzing trend for column '{variable}': {str(e)}")


# ------------------------------------------------------------
# Answer Natural-Language Question
# ------------------------------------------------------------
def ask_question(filename, question):
    """Answer any natural-language question about the dataset."""
    try:
        df = pd.read_csv(filename, low_memory=False)
        pandas_agent = create_pandas_dataframe_agent(
            llm=selected_llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )
        return pandas_agent.run(question)
    except Exception as e:
        raise ValueError(f"Error answering question: {str(e)}")
