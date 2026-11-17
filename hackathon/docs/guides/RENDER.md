# Render: put the model server online

Render runs a Python server from a GitHub repository. You use it on Tuesday for your model server.

## Facts (checked 17 November 2026)

| Topic | Fact |
|---|---|
| Price | Free plan, no credit card |
| Sleep | The free service sleeps after 15 minutes without requests. It wakes in about 1 minute. |
| Hours | 750 free instance hours for each workspace for each month |
| Memory | 512 MB. Use ONNX Runtime, not PyTorch. |
| Disk | No persistent disk. Files are lost on restart. Store data in D1, R2 or Supabase. |
| Port | The app must listen on `0.0.0.0` and on the port in `$PORT` |
| Python | New services use Python 3.14.10. Pin your version (see below). |
| Region | Fixed when you create the service. The default is Oregon. Choose **Frankfurt**. |
| Deploy | Auto-deploy on each push to the branch |

## Deploy with a Blueprint

1. Put the server files at the **root** of a GitHub repository (see the checks below).
2. Open [render.com](https://render.com). Sign in with GitHub.
3. Choose **New**, then **Blueprint**. Select the repository.
4. Render reads `render.yaml`. Check the service name and the region. Click **Apply**.
5. Open the **Logs** tab. Wait for "live". This takes a few minutes.
6. Open `https://<service-name>.onrender.com`. The page must load.
7. Test the API (see `GRADIO.md`). Write the URL in your `README.md`.

## The render.yaml file

```yaml
services:
  - type: web
    name: team3-facade-model
    runtime: python
    plan: free
    region: frankfurt
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    healthCheckPath: /
```

| Field | Meaning |
|---|---|
| `type: web` | A server that answers HTTP requests |
| `name` | Name of the service. It becomes part of the URL. Use lowercase and hyphens. |
| `runtime: python` | Render installs Python |
| `plan: free` | The free plan |
| `region` | Where the server runs. Cannot change later. Use `frankfurt`. |
| `buildCommand` | Runs once for each deploy. Installs the packages. |
| `startCommand` | Starts the server |
| `healthCheckPath` | Render calls this path to check that the server is alive |

Pin Python: create `.python-version` at the repo root with one line, for example `3.12`. Or set the `PYTHON_VERSION` environment variable.

## Before you deploy

- [ ] `app.py`, the model file and `requirements.txt` are at the repo root (or `render.yaml` points to them)
- [ ] `render.yaml` is at the repo root
- [ ] `.python-version` exists
- [ ] `requirements.txt` has pinned versions and **no** `torch`
- [ ] `app.py` uses `server_name="0.0.0.0"` and `server_port=int(os.environ.get("PORT", 7860))`
- [ ] The server works on your computer: `python app.py`
- [ ] The model file is smaller than 100 MB (GitHub blocks larger files)
- [ ] No secrets in the repository

## If it fails

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails | A package cannot install | Read the build log (the first red line). Pin a version that exists. |
| `ModuleNotFoundError` at start | A package is missing in `requirements.txt` | Add it. Push. |
| Error about a Python version or a wheel | Render uses Python 3.14, the package does not | Add `.python-version` with `3.12` |
| "No open ports detected" | The app listens on `127.0.0.1` or on a fixed port | Use `0.0.0.0` and `$PORT` |
| `FileNotFoundError` for the model | File name case differs (`Model.onnx` and `model.onnx`). Linux is case sensitive. | Use the exact same name in code and in Git |
| Works on the page, error on the API | The input has a different shape or type than in training | Use the same preprocessing as in training. Print the input shape. |
| Model not in the repo, or push rejected | The model is larger than 100 MB | Make the model smaller, or download it from Hugging Face at start |
| "Out of memory" | PyTorch or a big model in 512 MB | Export to ONNX. Remove `torch`. |
| First request is slow or times out | The server was asleep | Wait 1 minute. Open the URL before the demo. |

## Repositories in the iaac-maai organization

Render needs access to the repository. For an organization repository, an **owner** of `iaac-maai` must install the **Render GitHub app** on the organization. Ask the course maintainers. For a personal repository, this is not needed.

## Friday

Open your server URL before 10:00. During the demo, one request every few minutes keeps it awake.
