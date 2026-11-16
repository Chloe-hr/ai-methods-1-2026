"""Model server: anomaly detection with a convolutional autoencoder (Gradio + ONNX Runtime).

The page has a form for people (upload an image, see the result) and an API
for programs (`/predict`: send a base64 image, get JSON back). The Cloudflare
app of notebook 2 uses the API. Runs locally (`python app.py`) and on Render.

The model is the ONNX file exported in notebook 1: it contains the architecture
AND the weights, so this file needs no PyTorch and no copy of the model class.
"""
import base64
import io
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "anomaly_ae_config.json").read_text())
SIZE = CONFIG["image_size"]
THRESHOLD = CONFIG["threshold"]

session = ort.InferenceSession(HERE / "anomaly_ae.onnx")
INPUT_NAME = session.get_inputs()[0].name


def preprocess(image: Image.Image):
    # The SAME preprocessing as in training: greyscale, 28 x 28, pixels 0-1, light object on dark background
    grey = image.convert("L").resize((SIZE, SIZE))
    x = np.asarray(grey, dtype=np.float32) / 255.0
    inverted = bool(x.mean() > 0.5)          # a photo on a white background: flip it
    if inverted:
        x = 1.0 - x
    return x[None, None], inverted           # shape (1, 1, 28, 28): one image, one channel


def to_image(array, scale=8):
    # A 28 x 28 array (0-1) -> an enlarged PIL image to display
    img = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    return img.resize((SIZE * scale, SIZE * scale), Image.NEAREST)


def analyse(image: Image.Image):
    x, inverted = preprocess(image)
    rebuilt = session.run(None, {INPUT_NAME: x})[0]
    error_map = ((rebuilt - x) ** 2)[0, 0]
    score = float(error_map.mean())
    return {
        "score": round(score, 5),
        "threshold": round(THRESHOLD, 5),
        "is_anomaly": score > THRESHOLD,
        "score_vs_threshold": round(score / THRESHOLD, 2),
        "inverted_input": inverted,
    }, rebuilt[0, 0], error_map


# ---- For people: the web form ------------------------------------------------
def analyse_for_page(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    result, rebuilt, error_map = analyse(image)
    verdict = "Anomaly" if result["is_anomaly"] else "Normal"
    return {verdict: 1.0}, result, to_image(rebuilt), to_image(error_map / max(error_map.max(), 1e-8))


# ---- For programs: base64 image in, JSON out ---------------------------------
def predict(image_b64: str) -> dict:
    """Score one image. Send a base64-encoded PNG or JPEG (a 'data:image/...;base64,' prefix is allowed)."""
    if "," in image_b64[:100]:
        image_b64 = image_b64.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    result, rebuilt, _ = analyse(image)
    buffer = io.BytesIO()
    to_image(rebuilt, scale=4).save(buffer, format="PNG")
    result["reconstruction_png_base64"] = base64.b64encode(buffer.getvalue()).decode()
    return result


with gr.Blocks(title="Autoencoder anomaly detector") as demo:
    gr.Markdown(
        "# Autoencoder anomaly detector\n"
        f"Trained only on **{', '.join(CONFIG['normal_classes'])}** (Fashion-MNIST). "
        "Anything else is rebuilt badly and gets a high reconstruction error. "
        f"Above the threshold ({THRESHOLD:.4f}) the image is flagged as an anomaly."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Input image")
            button = gr.Button("Analyse", variant="primary")
            gr.Examples(sorted(str(p) for p in (HERE / "examples").glob("*.png")), inputs=image_in)
        with gr.Column():
            verdict_out = gr.Label(label="Verdict")
            details_out = gr.JSON(label="Details")
            with gr.Row():
                rebuilt_out = gr.Image(label="Reconstruction")
                error_out = gr.Image(label="Error map (bright = rebuilt badly)")
    button.click(analyse_for_page, inputs=image_in, outputs=[verdict_out, details_out, rebuilt_out, error_out],
                 api_visibility="private")
    gr.api(predict, api_name="predict")

if __name__ == "__main__":
    # Render (and most hosts) tell the app which port to use in the PORT variable; locally it is 7860
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
