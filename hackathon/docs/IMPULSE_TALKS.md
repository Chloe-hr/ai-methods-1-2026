# Impulse talks

Every team gives **one impulse talk** of **3 to 5 minutes**, at the start of a class. Then we discuss it together.

Each topic is a **curious question**: how does that work? Find a real example, open it up, and explain what happens inside. You do not need the real code. A sound guess is fine, if you say it is a guess.

---

## Topics

Choose **one** topic on Monday. The topic fixes your day. The example is a starting point: you can bring your own.

### Tuesday · models
| Topic | The question | Example to start from |
|---|---|---|
| **Latent Space** | How does "find something similar" work? | Visual search ("buildings that look like this one"), or your own find |
| **Generative AI** | How does a model make a new design variant? | Image generators for facades or massing studies, or your own find |
| **Anomaly Detection** | How does a system spot the odd one out? | Strange readings in building energy data, or your own find |

### Wednesday · apps
| Topic | The question | Example to start from |
|---|---|---|
| **App Anatomy** | What is inside a small everyday app? | A to-do app or a calendar app: what runs where? |
| **Cloud Services** | Where do small apps live? | Cloudflare, Vercel, Supabase, Render: what each one is for, with one app built on it |
| **Data Storage** | Where do my photos and data go? | A photo app: files in object storage, facts in a database |

### Thursday · products
| Topic | The question | Example to start from |
|---|---|---|
| **Agent Skills** | What is inside a skill? | Compare **two small skills** (not a huge one). Show the folder structure, `SKILL.md` and its header, scripts, references, and how the agent finds and uses them |
| **Sign in with Google** | What happens between the click and the logged-in page? | Any app with a "Sign in with Google" button |
| **Passwords** | How does an app keep my password safe? | What an app stores instead of your password, and what a leak exposes |
| **1,000 users at once** | What breaks when 1,000 people open my app at the same time? | Your own hackathon app: sleeping servers, limits, costs |

For **Agent Skills**: public skills are on GitHub (search for "SKILL.md"). Pick two that do something fun or useful, and compare how they are built.

---

## Structure: 3 to 5 slides, 3 to 5 minutes

| # | Slide | What to show |
|---|---|---|
| 1 | **The question** | Your topic as a question, and the team name |
| 2 | **The use case** | One real example. Pictures, not text. A second example only if it is short |
| 3 | **How it works inside** | Your diagram (see below). Walk through it step by step |
| 4 | **What it means for us** | How this connects to the hackathon and to your own app |
| 5 | **One question for the class** | Starts the discussion |

### How to reverse-engineer
1. What does the user do, and what do they see?
2. Which parts are involved? (browser, server, model, database, another service)
3. In what order do they talk to each other?
4. What is stored, and where? What is gone after a reload?
5. If there is AI: what goes in, and what comes out?

---

## Diagrams

Use the two diagram types from class. One is enough; both are better.

| Diagram | What it shows | How to draw it |
|---|---|---|
| **System diagram** | What the thing is made of | One box per part (browser, server, model, database). Arrows between them, labelled with what flows (image, vector, JSON, password) |
| **Sequence diagram** | Who talks to whom, in what order | One column per part, time runs **down**. Each arrow is one request or one answer. Answers are dashed. Mark what is stored |

Tools: Google Slides shapes, Miro, Excalidraw, or a Mermaid diagram (ask your AI tool to write it).

---

## Format and hand-in

- **Google Slides.** Share as **"anyone with the link can view"**, so we can open it on the class computer.
- Put the link in your team repository: `impulse/LINK.md`.
- **Due:** at the start of the class in which you present.
- **3 to 5 minutes.** Pictures and diagrams, no text walls. One idea per slide.
- Name your sources on the slide where you use them.
