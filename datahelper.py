import os
import pandas as pd
import streamlit as st
import chardet
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)

# ------------------------------------------------------------
# Universal CSV Reader (Automatic Encoding Detection + Safe UTF-8 Fallback)
# ------------------------------------------------------------
def read_csv_any_encoding(file_obj):
    """
    Reads a CSV file safely with automatic encoding detection.
    Works for both Streamlit uploaded files and file paths.
    If detection fails, it gracefully falls back to UTF-8 with errors='replace'.
    """
    try:
        if hasattr(file_obj, "read"):
            # Streamlit uploaded file
            raw_bytes = file_obj.read()
            detected = chardet.detect(raw_bytes)
            encoding = detected.get("encoding") or "utf-8"
            file_obj.seek(0)
            try:
                return pd.read_csv(file_obj, encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                file_obj.seek(0)
                return pd.read_csv(file_obj, encoding="utf-8", errors="replace", low_memory=False)
        else:
            # Local file path
            with open(file_obj, "rb") as f:
                raw_bytes = f.read()
                detected = chardet.detect(raw_bytes)
                encoding = detected.get("encoding") or "utf-8"
            try:
                return pd.read_csv(file_obj, encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                return pd.read_csv(file_obj, encoding="utf-8", errors="replace", low_memory=False)
    except Exception as e:
        raise ValueError(f"Unable to read CSV file due to encoding issue: {e}")

# ------------------------------------------------------------
# Load environment variables (.env for local, st.secrets for Streamlit Cloud)
# ------------------------------------------------------------
load_dotenv()

# Try loading Groq API key (Streamlit Cloud first, then local)
try:
    api_key = st.secrets["groq"]["api_key"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "Groq API key not found. Please add it to Streamlit secrets (Cloud) "
        "or define GROQ_API_KEY in your local .env file."
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
    df = read_csv_any_encoding(filename)
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
        return read_csv_any_encoding(filename)
    except Exception as e:
        raise ValueError(f"Error loading DataFrame from {filename}: {e}")

# ------------------------------------------------------------
# Analyze Trend
# ------------------------------------------------------------
def analyze_trend(filename, variable):
    """Interpret trend of a specific variable or column."""
    try:
        df = read_csv_any_encoding(filename)
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
        raise ValueError(f"Error analyzing trend for '{variable}': {e}")

# ------------------------------------------------------------
# Answer Natural-Language Question
# ------------------------------------------------------------
def ask_question(filename, question):
    """Answer any natural-language question about the dataset."""
    try:
        df = read_csv_any_encoding(filename)
        pandas_agent = create_pandas_dataframe_agent(
            llm=selected_llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )
        return pandas_agent.run(question)
    except Exception as e:
        raise ValueError(f"Error answering question: {e}")
