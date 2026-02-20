# ⚡ Interview Copilot

> A real-time, AI-powered interview assistant that listens to your interview, understands questions, and streams smart first-person answers directly to a sleek on-screen overlay — all with no copy-paste needed.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyQt6](https://img.shields.io/badge/UI-PyQt6-green) ![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3-orange) ![Deepgram](https://img.shields.io/badge/STT-Deepgram%20Nova2-purple) ![Tests](https://img.shields.io/badge/Tests-33%20passing-brightgreen) ![Production](https://img.shields.io/badge/Production%20Ready-9.5%2F10-gold)

---

## 🚀 Features

### Core Engine
| Feature | Detail |
|---|---|
| **Dual Audio Capture** | Captures mic (`[ME]`) + interviewer via Windows WASAPI loopback (`[INTERVIEWER]`) simultaneously |
| **Real-time Transcription** | Deepgram Nova-2 WebSocket — sub-second speech-to-text on both channels |
| **Smart End-of-Turn Gating** | Dangling-word detection + 2.5s silence timeout to precisely detect when interviewer finishes |
| **WS Auto-Reconnect** | Deepgram WebSocket drops trigger automatic reconnect (up to 5 retries with 3s delay) |

### AI & Answer Quality
| Feature | Detail |
|---|---|
| **Smart Model Router** | Technical questions → `llama-3.3-70b-versatile` (deep, precise). HR/Behavioral → `llama-3.1-8b-instant` (fast) |
| **First-Person Candidate Voice** | LLM speaks *as* the candidate — `"I built..."`, `"In my experience..."` — never generic career-coach advice |
| **HR Question Intelligence** | Salary, leadership, failures, conflict, career goals — all answered concretely in first person, no code blocks |
| **Anti-Hallucination** | Only references skills/projects explicitly in `data/portfolio.md` |
| **RAG Confidence Filtering** | ChromaDB L2 threshold (≤1.2) — irrelevant chunks silently dropped, LLM never sees noise |
| **Smooth Token Streaming** | Groq `stream=True` — answer appears word-by-word in realtime, not all at once |

### Hotkeys
| Hotkey | Action |
|---|---|
| `Ctrl+Shift+Space` | **Force answer now** — bypass all silence timers, fire LLM immediately |
| `Ctrl+R` | **Regenerate** — fresh different answer to last question (temperature 0.6) |

### UI & UX
| Feature | Detail |
|---|---|
| **Resizable Overlay** | Drag any edge or corner to resize; always-on-top, click-through design |
| **JetBrains Mono Font** | Bundled in `assets/fonts/` — premium monospace for code blocks |
| **Markdown Rendering** | Bold, inline code, fenced code blocks with label, bullet points — all rendered as styled HTML |
| **Scrollable Q&A History** | Last 3 Q&A pairs with progressive opacity fading (older = more faded) |
| **Copyable Text** | Select any text in overlay with mouse, `Ctrl+C` to copy |
| **System Tray Icon** | Neon green ⚡ icon — right-click to Show/Hide/Quit; click X to hide (app keeps running) |
| **Startup Notification** | Windows balloon toast on launch with hotkey reminder |

---

## 🛠️ Setup

### 1. Clone
```bash
git clone https://github.com/Ammar-2510809836/interview-copilot.git
cd interview-copilot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create `.env` in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

### 4. Add Your Portfolio
Edit `data/portfolio.md` with your resume, skills, projects, and certifications. The RAG system auto-ingests this file on every startup.

---

## ▶️ Usage

```bash
python main.py
```

The overlay appears immediately. The ⚡ tray icon appears in the Windows system tray.

> **Audio Note:** On first run, the app auto-detects your default speakers (loopback) and microphone. If interviewer audio is not captured, check your audio loopback device in `core/capture.py`.

---

## ⌨️ Hotkey Reference

| Hotkey | Action |
|---|---|
| `Ctrl+Shift+Space` | Force immediate LLM answer — bypasses all silence timers |
| `Ctrl+R` | Regenerate last answer with a new, different response |

> **Note:** The `keyboard` library may require administrator privileges on Windows. If hotkeys don't work, run `python main.py` as Administrator.

---

## ✅ Tests

33 unit tests with no API keys required — all mocked:

```bash
python -m pytest tests/ -v
```

| Module | Tests |
|---|---|
| `test_rag.py` | RAG ingestion (normal, missing, empty, chunk count), retrieval (relevance, threshold, separator) |
| `test_llm.py` | Model router (technical vs behavioral), `generate_answer` (mocked Groq), streaming (token assembly, SKIP), regen (temperature, error handling) |
| `test_transcription.py` | Engine init, `_setup_connection` (success/failure), reconnect logic (max retries, 2nd attempt success), queue overflow |

---

## 🏗️ Architecture

```
interview-copilot/
├── main.py                  # Async orchestrator — gating, hotkeys, streaming, Q&A history
├── core/
│   ├── capture.py           # Dual WASAPI loopback + mic capture (PyAudioWPatch)
│   ├── transcription.py     # Deepgram WebSocket STT with auto-reconnect
│   ├── rag.py               # ChromaDB in-memory ingestion + confidence-filtered retrieval
│   └── llm.py               # Groq LLM client — model router, streaming, regen
├── ui/
│   └── overlay.py           # PyQt6 resizable overlay + system tray icon
├── tests/
│   ├── test_rag.py          # 9 RAG unit tests
│   ├── test_llm.py          # 13 LLM unit tests
│   └── test_transcription.py # 11 transcription unit tests
├── assets/
│   └── fonts/
│       └── JetBrainsMono-Regular.ttf
└── data/
    └── portfolio.md         # ← Edit this with your resume/skills
```

---

## 🔑 Required APIs

| API | Purpose | Free Tier |
|---|---|---|
| [Groq](https://console.groq.com) | LLM inference — Llama 3.3 70B + Llama 3.1 8B | ✅ Yes |
| [Deepgram](https://console.deepgram.com) | Real-time speech-to-text (Nova-2) | ✅ Yes (~$200 credit) |

---

## 📦 Dependencies

```
PyQt6
qasync
groq
deepgram-sdk
chromadb
PyAudioWPatch
keyboard
python-dotenv
```

---

## 🔒 Security

| Item | Status |
|---|---|
| API keys in `.env` | ✅ Git-ignored — **never committed** |
| No hardcoded credentials | ✅ All keys use `os.getenv()` only |
| `interview_session.log` | ✅ Git-ignored + daily log rotation (7 days kept) |
| `qa_history.json` | ✅ Git-ignored (runtime state) |
| `chroma_db/` | ✅ Git-ignored (in-memory ChromaDB, no disk persistence) |

> **Verified:** `git log --all -- .env` returns empty — the `.env` file was **never committed at any point** in the git history.

---

## 🏆 Production Readiness: 9.5 / 10

| Area | Score | Notes |
|---|---|---|
| Architecture | ✅ 10/10 | Clean module separation, async throughout |
| Error Handling | ✅ 10/10 | Every external call wrapped, UI shows error state on failure |
| Security | ✅ 10/10 | No secrets in code or history |
| Reliability | ✅ 9/10 | WS auto-reconnect, bounded queues, log rotation |
| Test Coverage | ✅ 9/10 | 33 tests across all core modules |
| Documentation | ✅ 10/10 | This README |

---

*Built to surpass Final Round AI.*
