# Project structure

A clear structure helps you and your AI agent. Add a folder only when you need it.

## Model server

**Simple (start here)**

```
02_model_server/
├── app.py                  # preprocessing, predict, page
├── anomaly_ae.onnx         # the model
├── config.json             # threshold, image size, class names
├── requirements.txt
├── .python-version
├── render.yaml
├── examples/               # test inputs
└── README.md               # URL, API example, what the model does
```

**Bigger (when `app.py` is longer than about 200 lines)**

```
02_model_server/
├── app.py                  # only the Gradio page and the API
├── model.py                # load the model, run it
├── preprocessing.py        # the SAME steps as in training
├── config.json
├── tests/test_predict.py
└── ...                     # the same files as above
```

## Web app

**Simple**

```
03_app/
├── public/                 # index.html, login.html, style.css
├── functions/api/          # one file per endpoint
├── schema.sql              # the D1 tables
├── wrangler.toml
└── README.md
```

**Bigger (React + Vite)**

```
03_app/
├── src/
│   ├── components/
│   ├── pages/
│   └── api.js              # all calls to the backend in one file
├── functions/api/
├── public/
├── package.json
└── wrangler.toml
```

## When to split a file

| Sign | Action |
|---|---|
| The file is longer than about 200 lines | Split by job (model, preprocessing, page) |
| You copy the same code twice | Make one function |
| The agent changes things you did not ask for | Smaller files, clearer names |

Three files are enough for most hackathon projects.

## A docs/ folder for planning

Put planning documents in `docs/`. Humans and AI agents read them **first**.

```
docs/
├── PRD.md          # what we build, for whom, what is out of scope
├── PLAN.md         # steps in order, with a checkbox for each
└── DECISIONS.md    # each decision: date, choice, reason
```

| File | Contents | Length |
|---|---|---|
| `PRD.md` | Problem, user, one-sentence product, input and output of the model, must-have and nice-to-have | 1 page |
| `PLAN.md` | Tasks by day. Mark done tasks. | 1 page |
| `DECISIONS.md` | For example: "17 Nov: Gradio on Render. Reason: the class example works." | A list |

Tell your agent at the start of each session: "Read `docs/PRD.md`, `docs/PLAN.md` and `docs/DECISIONS.md` first. Update `PLAN.md` when a task is done."

## Naming

| Item | Rule | Example |
|---|---|---|
| Repository | `hackathon-team<K>-<topic>` | `hackathon-team3-facade-defects` |
| Folders and files | lowercase, no spaces. Day folders start with a number. | `01_autoencoder/` |
| Model files | name and version | `facade_ae_v1.onnx` |

Use the exact same file name in code and in Git. Linux servers are case sensitive.

## .gitignore and secrets

Never commit secrets. A secret in Git is public for ever, even after you delete it.

| Ignore | Why |
|---|---|
| `.env`, `.dev.vars` | Passwords and keys |
| `node_modules/` | Installed packages |
| `.venv/`, `__pycache__/` | Python files made on your computer |
| `.wrangler/` | Local Cloudflare data |
| `*.pt`, big datasets | Too large. Use Hugging Face. |

| Secret | Where it goes |
|---|---|
| App password | `.dev.vars` locally. `npx wrangler pages secret put APP_PASSWORD` online. |
| API key for Render | Render dashboard, **Environment** tab |
| Anything else | Never in code, README or notebooks. Rotate it if it leaked. |

## Checklist

- [ ] `docs/` has `PRD.md`, `PLAN.md`, `DECISIONS.md`
- [ ] No secret in any file. Check with `git grep -i password`.
