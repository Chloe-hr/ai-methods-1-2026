# Hackathon guides

Short guides for the hackathon week. Each guide is one page. A coding agent (Claude Code, Cursor) can read a guide and follow it.

**Tip:** tell your agent: "Read `RENDER.md` and deploy my model server."

## Which guide, when

| Order | Guide | Use it to | When |
|---|---|---|---|
| 1 | [`FRAMEWORKS.md`](FRAMEWORKS.md) | Choose tools for the model server, the app, hosting and data | Monday, before you build |
| 2 | [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | Set up your repository and a `docs/` folder for plans | Monday |
| 3 | [`GRADIO.md`](GRADIO.md) | Turn your `predict` function into a page and an API | Tuesday |
| 4 | [`RENDER.md`](RENDER.md) | Put the model server online | Tuesday |
| 5 | [`MCP.md`](MCP.md) | Let AI agents call your model as a tool (optional) | Thursday |

## The stack of the week

| Part | Tool | Day |
|---|---|---|
| Model file | ONNX, on Hugging Face | Monday |
| Model server | Gradio on Render | Tuesday |
| App and data | Cloudflare Pages, D1, R2 | Wednesday |
| Agent access (optional) | MCP | Thursday |

## Rules for all guides

- Start with the simplest tool that works.
- Never put a secret (password, API key, token) in Git.
- Change one thing at a time. Test it. Then continue.
- If a step fails, read the log first. Then ask your agent, and paste the log.

## Where the examples are

- Model server: `hackathon/docs/model_app/`
- App: `hackathon/docs/cloudflare_app/`
- Deployment notebook: `hackathon/docs/02_deployment.ipynb`

## A complete example repository

A model server ready to deploy, with step-by-step instructions for GitHub and Render: https://github.com/SerjoschDuering/serjoscha-model-server
