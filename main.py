import streamlit as st
import pandas as pd
import os

from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler

def main():
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Talk CSV to me")
    st.header("Agent Panda 🕵🐼️")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")

    if csv_file is not None:

        df = pd.read_csv(csv_file)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me a question about your CSV"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                reduce_k_below_max_tokens=True,
                agent_executor_kwargs={"handle_parsing_errors": True}
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = agent.run(prompt, callbacks=[callback])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.success(response)

if __name__ == "__main__":
    main()
