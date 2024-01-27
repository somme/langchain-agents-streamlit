Talk to your own CSV files
===============

This is a conversational AI bot that allows you to talk to your own CSV files.
Uses Langchain, pandas dataframes, agents, Streamlit and OpenAi.

### Features

* Chat history
* Basic memory using `ConversationBufferWindowMemory`

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
