# Gradio: a page and an API from one function

Gradio is the **server part** of our stack. You write one Python function. Gradio makes two things from it:

| Output | For | Example |
|---|---|---|
| A web page | People | Upload an image, see the result |
| An API | Programs | The Cloudflare app sends an image, gets JSON |

## How `predict` becomes an endpoint

From `hackathon/docs/model_app/app.py`:

```python
def predict(image_b64: str) -> dict:
    """Score one image. Send a base64-encoded PNG or JPEG."""
    ...
    return result            # a dictionary: it becomes JSON

gr.api(predict, api_name="predict")
```

| In the code | On the server |
|---|---|
| `gr.api(predict, api_name="predict")` | Endpoint `/gradio_api/call/predict` |
| Function arguments, in order | The `data` list in the request |
| The returned dictionary | The JSON in the answer |
| The docstring | The description in the API page (`/?view=api`) |

Rules:
- Use simple input types: text (base64 or JSON text), numbers, lists.
- Return a dictionary with simple values (numbers, text, booleans, lists).
- Change the model, **not** the shape of `predict`. Other parts of the system depend on it.

## Keep the same preprocessing as in training

The server must prepare the input in exactly the same way as the training notebook. If not, the scores are wrong, and no error shows.

| Check | Training | Server |
|---|---|---|
| Image size | for example 28 x 28 | same |
| Colour | greyscale | same |
| Pixel range | 0 to 1 | same |
| Normalisation (mean, std) | as in training | same |
| Column order (tables) | as in training | same |

Test: give the server one training sample. The score must match the notebook.

## Call it

Replace `<url>` with `http://127.0.0.1:7860` (local) or `https://<service-name>.onrender.com`.

**1. Python, `gradio_client` (easiest)**

```python
from gradio_client import Client
client = Client("<url>")
print(client.predict(image_b64, api_name="/predict"))
```

**2. Python, plain HTTP (two steps)**

```python
import requests, json
r = requests.post("<url>/gradio_api/call/predict", json={"data": [image_b64]})
event_id = r.json()["event_id"]
r = requests.get(f"<url>/gradio_api/call/predict/{event_id}")
print(r.text)       # the result is in the line after "event: complete"
```

**3. curl**

```bash
EVENT=$(curl -s -X POST <url>/gradio_api/call/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["<base64 image>"]}' | python3 -c "import sys,json; print(json.load(sys.stdin)['event_id'])")
curl -N <url>/gradio_api/call/predict/$EVENT
```

## Run locally

```bash
pip install -r requirements.txt
python app.py        # open http://127.0.0.1:7860
```

## Common problems

| Problem | Fix |
|---|---|
| The answer says `event: error` | Read the line after it. Often the input has the wrong type. |
| Version differences | Pin `gradio==6.32.0` (the class version). Examples change between versions. |
| First call after a pause is slow | Render was asleep (see `RENDER.md`) |
