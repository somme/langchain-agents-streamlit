Talk to your own CSV files
===============

This is a simple app that allows you to talk to your own CSV files in a conversation.
Uses Langchain, pandas dataframes, Streamlit and OpenAi.

**Features**

* chat history
* basic memory using `ConversationBufferWindowMemory`

### Requirements

install poetry virtual environment manager and package manager

```
curl -sSL https://install.python-poetry.org | python3 -
```

### Setup

add .env with your OPENAI_API_KEY=XXX

```
poetry install --no-root
```

### Run

```
poetry shell

streamlit run main.py
```
