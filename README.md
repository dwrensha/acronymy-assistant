# Acronymy Assistant

Acronymy Assistant is an interactive tool for composing backronyms.

## How to run

Set up a virtual environment:

```
virtualenv venv
source venv/bin/activate

```

Download the dependecies:

```
pip install aiohttp
pip install 'transformers[torch]'
```

Start the server:
```
python server.py
```

Then point a web browser to http://localhost:8080/index.html .
