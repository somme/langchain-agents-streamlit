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
