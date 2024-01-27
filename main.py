import streamlit as st
import pandas as pd
import os
import langchain

from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, ConversationalChatAgent, AgentExecutor
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

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
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferWindowMemory(
            chat_memory=msgs,
            return_messages=True,
            k=5,
            memory_key="chat_history",
            output_key="output"
        )

        PREFIX = """
            You are working with a pandas dataframe in Python. The name of the dataframe is `df`.

            This is the result of `print(df.head())`:
            {df}
            Begin!
            {chat_history}
            Question: {input}
            {agent_scratchpad}
            """

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me a question about your CSV"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            csv_agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                prefix=PREFIX,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                reduce_k_below_max_tokens=True,
                max_iterations=5,
                early_stopping_method='generate',
                agent_executor_kwargs={
                    "handle_parsing_errors": True,
                    "memory": memory,
                    "input_variables": ["df", "input", "chat_history", "agent_scratchpad"],
                },
            )

            csv_tool = Tool(
                return_intermediate_steps=False,
                name="csv",
                func=csv_agent.run,
                description="""This tool is useful when you need to answer questions about data stored in a pandas dataframe."""
            )

            tools = [csv_tool]

            chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools, return_intermediate_steps=False)

            executor = AgentExecutor.from_agent_and_tools(
                agent=chat_agent,
                tools=tools,
                memory=memory,
                return_intermediate_steps=False,
                handle_parsing_errors=True,
                verbose=False,
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                #remove this callback line if you dont want to display thoughts and actions in the ui and remove from executor
                callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

                #to debug the response uncomment both debug lines
                #langchain.debug=True
                response = executor.run(prompt, callbacks=[callback])
                #langchain.debug=False
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.success(response)

                #uncomment to view the history
                #st.expander("View the message contents in session state").json(st.session_state.langchain_messages)


if __name__ == "__main__":
    main()
