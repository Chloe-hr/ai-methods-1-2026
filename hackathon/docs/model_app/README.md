# Hackathon example: autoencoder anomaly detector (model server)

Class example for the **AI Methods I hackathon** (IAAC, MaAI 2026/27). It shows the shape every team's model server needs:
- a trained model;
- a page for people;
- an API for programs;
- a README that explains the autoencoder.

It runs on your computer (`python app.py`) and online as a free **Render** web service.

## The autoencoder

| | |
|---|---|
| **Type** | Convolutional autoencoder: 2 strided convolutions → 16-number code → 2 transposed convolutions |
| **Trained on** | Fashion-MNIST, **normal classes only**: T-shirt/top, pullover, coat, shirt (8,054 images of 28 × 28 px) |
| **Use** | Anomaly detection. The model rebuilds tops well; anything else (shoes, bags) is rebuilt badly. |
| **Score** | Reconstruction error: the mean squared difference between the input and its reconstruction |
| **Threshold** | 95th percentile of the score on normal validation images, stored in `anomaly_ae_config.json` |
| **Test result** | ROC AUC about 0.96, precision 0.93, recall 0.79 (footwear and bags as anomalies) |
| **Format** | ONNX (`anomaly_ae.onnx`), run with ONNX Runtime: no PyTorch needed on the server |
| **Notebook** | `hackathon/docs/01_autoencoders.ipynb` in the course repository |

**Input:** any image. It is converted to greyscale, resized to 28 × 28 and scaled to 0–1. Images with a light background are inverted, because the training images show a light object on a dark background.

**Output:** the score, the threshold, `is_anomaly`, the score as a multiple of the threshold, and the reconstruction.

## Run it locally

```bash
pip install -r requirements.txt
python app.py            # then open http://127.0.0.1:7860
```

## Put it online (Render, free)

1. Push this folder to its own GitHub repository.
2. On [render.com](https://render.com), sign in with GitHub, choose **New → Blueprint** and select the repository. `render.yaml` creates a free Python web service:
   - build command: `pip install -r requirements.txt`
   - start command: `python app.py`
3. After the build, the server answers at `https://<service-name>.onrender.com`.

A free service **sleeps after 15 minutes without requests**. The next request wakes it up, which takes about a minute.

## Call it from a program

```python
from gradio_client import Client
import base64

client = Client("https://<service-name>.onrender.com")
image_b64 = base64.b64encode(open("examples/sneaker.png", "rb").read()).decode()
print(client.predict(image_b64, api_name="/predict"))
```

Over plain HTTP, which is what the Cloudflare app does:
1. `POST /gradio_api/call/predict` with `{"data": ["<base64 image>"]}` returns an `event_id`.
2. `GET /gradio_api/call/predict/<event_id>` returns the result after `event: complete`.

## Files

| File | Contents |
|---|---|
| `app.py` | The preprocessing, the page and the API |
| `anomaly_ae.onnx` | The model (architecture and weights) |
| `anomaly_ae.pt` | The same weights for PyTorch (not used by the server) |
| `anomaly_ae_config.json` | Settings and threshold |
| `examples/` | Test images |
| `requirements.txt`, `.python-version`, `render.yaml` | What Render needs to build and start the server |

## Limitations

The model was trained on 28 px images of clothes. It is a teaching example, not a product detector. Its scores on real photos are only meaningful for images that look like Fashion-MNIST.
