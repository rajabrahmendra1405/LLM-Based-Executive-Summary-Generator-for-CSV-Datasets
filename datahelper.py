import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)

# ------------------------------------------------------------
# Load environment variables (.env must contain GROQ_API_KEY)
# ------------------------------------------------------------
load_dotenv()

# Fetch the API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is loaded correctly
if not api_key:
    raise ValueError("API key not found. Please check the .env file.")

# ------------------------------------------------------------
# Initialize Groq LLM (Free – uses LLaMA 3.3 70B)
# ------------------------------------------------------------
try:
    llm_groq = ChatGroq(
        model="llama-3.3-70b-versatile",  # Set the desired model
        api_key=api_key,  # Pass the API key from environment
        temperature=0  # Set temperature for randomness (0 for deterministic output)
    )
except Exception as e:
    raise ValueError(f"Error initializing Groq client: {str(e)}")

# Set the selected LLM for the rest of the app
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

    data_summary = {}

    # 1️⃣ Sample of the data
    data_summary["initial_data_sample"] = df.head()

    # 2️⃣ Column Descriptions
    data_summary["column_descriptions"] = pandas_agent.run(
        "Create a markdown table describing each column name and its meaning."
    )

    # 3️⃣ Missing Values
    data_summary["missing_values"] = pandas_agent.run(
        "Report if there are any missing values and how many. "
        "Respond like: 'There are X missing values in total.'"
    )

    # 4️⃣ Duplicate Values
    data_summary["duplicate_values"] = pandas_agent.run(
        "Report if there are any duplicate rows and how many. "
        "Respond like: 'There are X duplicate rows in total.'"
    )

    # 5️⃣ Essential Metrics
    data_summary["essential_metrics"] = df.describe(include='all')

    return data_summary
# ------------------------------------------------------------
# Function to get the DataFrame (for use in other analyses)
# ------------------------------------------------------------
def get_dataframe(filename):
    """Return the loaded DataFrame."""
    try:
        df = pd.read_csv(filename, low_memory=False)
        return df
    except Exception as e:
        raise ValueError(f"Error loading DataFrame from {filename}: {str(e)}")

# ------------------------------------------------------------
# Analyze Trend of a Specific Variable/Column in the Dataset
# ------------------------------------------------------------
def analyze_trend(filename, variable):
    """Interpret trend of a specific variable or column."""
    try:
        # Load the dataset
        df = pd.read_csv(filename, low_memory=False)

        # Create the pandas agent for analysis
        pandas_agent = create_pandas_dataframe_agent(
            llm=selected_llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )

        # Run the trend analysis
        trend_response = pandas_agent.run(
            f"Provide a short executive interpretation of trends in the column '{variable}'. "
            "Consider the dataset rows as chronological and provide data-driven reasoning."
        )

        return trend_response

    except Exception as e:
        raise ValueError(f"Error analyzing trend for column '{variable}': {str(e)}")

# ------------------------------------------------------------
# Answer Any Natural-Language Question About the Dataset
# ------------------------------------------------------------
def ask_question(filename, question):
    """Answer any natural-language question about the dataset."""
    try:
        # Load the dataset
        df = pd.read_csv(filename, low_memory=False)

        # Create the pandas agent for question answering
        pandas_agent = create_pandas_dataframe_agent(
            llm=selected_llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )

        # Get the AI response to the question
        AI_response = pandas_agent.run(question)
        return AI_response

    except Exception as e:
        raise ValueError(f"Error answering question: {str(e)}")
