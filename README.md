# Acronymy Assistant

Acronymy Assistant is an interactive tool for composing backronyms.

It was created as an attempt to speed up progress on https://acronymy.net/ .

Here's what it looks like:

<img style="border:1px" src="screenshot.png" height="300px">

And here is a quick video explanation of how it works:

[<img src="http://img.youtube.com/vi/LjOHnXRIp4Y/0.jpg" height="240px">](http://youtu.be/LjOHnXRIp4Y)


## Features

### prompt
Choose a prompt template and edit it to your liking. `$WORD` will get replaced
by the current word, and `$DEF` will get replaced by the definition.
If there is no `$DEF`, then the definition will be added at the end of the prompt.

### temperature
Higher temperature flattens the probability distributions, i.e. makes
Acronymy Assistant more likely to choose weirder tokens.

### remplacement_prob

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

To see commandline options (e.g. specifying which model to use), do this:
```
python server.py --help
```
