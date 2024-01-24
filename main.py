import pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

def main():
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Talk CSV to me")
    st.header("Pandas Agents 🐼🕵️")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")

    if csv_file is not None:

        df = pd.read_csv(csv_file)

        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )

        user_question = st.text_input("Ask me a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="🐼 I'm thinking..."):
                st.write(agent.run(user_question))


if __name__ == "__main__":
    main()
