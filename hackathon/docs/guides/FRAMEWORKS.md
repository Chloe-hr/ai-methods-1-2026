# Choose your tools

**Rule: start with the simplest tool that works.** Change to a bigger tool only when the simple tool cannot do the job. "Our week" marks what we teach and support.

## Model in Python

| Tool | Makes | Use when | Our week |
|---|---|---|---|
| **Gradio** | A web page and an API from one function | You want a quick test page and an API | **Yes** |
| FastAPI | An API only | You build your own frontend and want full control of the routes | No |
| Streamlit | A dashboard | The goal is charts and sliders for people, not an API | No |

## Frontend (the app)

| Tool | Use when | Our week |
|---|---|---|
| **HTML + JavaScript** | One or two screens. No build step. | **Yes** |
| React + Vite | Several screens, shared state, many components | No (only if you know it) |

## Hosting

| Service | Hosts | Use when | Our week |
|---|---|---|---|
| **Cloudflare Pages** | Static files (HTML, JS, CSS) and functions | The app page | **Yes** |
| **Render** | A Python server | The model server | **Yes** |
| Cloudflare Workers | Small functions at the edge | Only logic, no page | Used through Pages functions |

A Python model needs a Python server (Render). Cloudflare cannot run it.

## Data

| Service | Stores | Use when | Our week |
|---|---|---|---|
| **D1** | Rows in tables (SQL) | Results, history, users | **Yes** |
| **R2** | Files (images, models) | Uploaded images | **Yes** |
| Supabase | A Postgres database, login, files | You need real user accounts | No |

Rule: **facts go in the database, files go in object storage.** The database holds the file key, not the file.

## Our week at a glance

```
Browser (HTML + JS)
  -> Cloudflare Pages + functions  -> D1 (rows), R2 (images)
        -> Render (Gradio + ONNX model)
```

## How to decide

| Question | If yes |
|---|---|
| Can a plain HTML page do it? | Use HTML. Skip React. |
| Does the model need Python packages? | Render with Gradio |
| Do you only need a chart page? | Streamlit |
| Do you need exact control of the API routes? | FastAPI |
| Do you store only a few rows? | D1 |
| Do you need user sign-up and email login? | Supabase (more work) |

## For your AI agent

Tell your agent which tools you chose. Write the choice in `docs/DECISIONS.md` (see `PROJECT_STRUCTURE.md`). Without this, the agent may add React, FastAPI and Supabase "to be safe". Ask it to keep the stack small.
