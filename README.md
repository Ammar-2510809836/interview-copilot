# ⚡ Interview Copilot

> A real-time, AI-powered interview assistant that listens to your interview, understands questions, and streams smart first-person answers directly to a sleek on-screen overlay — all with no copy-paste needed.

![Python](https://img.shields.io/badge/Python-3.12%E2%80%933.13-blue) ![PyQt6](https://img.shields.io/badge/UI-PyQt6-green) ![Groq](https://img.shields.io/badge/LLM-Groq%20%2B%20Gemini-orange) ![Deepgram](https://img.shields.io/badge/STT-Deepgram%20Nova2-purple) ![Tests](https://img.shields.io/badge/Tests-70%2B%20passing-brightgreen)

---

## 🚀 Features

### Core Engine
| Feature | Detail |
|---|---|
| **Dual Audio Capture** | Captures mic (`[ME]`) + interviewer via Windows WASAPI loopback (`[INTERVIEWER]`) simultaneously |
| **Real-time Transcription** | Deepgram Nova-2 WebSocket — sub-second speech-to-text on both channels |
| **Pause-Based Turn Detection** | Measures the **silence gap since the interviewer's last word** (not total talk time), tuned for TTS interviewers like micro1's *Zara*. Fine-grained polling fires right at the pause threshold (~2.5s) instead of overshooting. Dangling-word / completeness heuristics avoid cutting off multi-part questions |
| **Mid-Answer Interruption Handling** | If the interviewer speaks while the copilot is streaming, the stale answer is discarded and listening resumes |
| **WS Auto-Reconnect** | Deepgram WebSocket drops trigger automatic reconnect (up to 5 retries with 3s delay) |

### AI & Answer Quality
| Feature | Detail |
|---|---|
| **Natural Spoken Delivery** | Answers are flowing, first-person spoken prose — contractions, confident, no bullet-point "robot" cadence. Tuned to sound like a real person, not a document being read aloud (`ANSWER_STYLE=spoken`, default) |
| **Instant Thinking-Bridge** | A natural opener ("Right, so the way I'd approach this…") appears in <100ms the moment the interviewer stops — so you start talking immediately while the real answer streams underneath. Never evaluative ("good question" is banned) |
| **Smart Model Router** | Technical questions → `llama-3.3-70b-versatile` (deep, precise). HR/Behavioral → `llama-3.1-8b-instant` (fast) |
| **Multi-Provider Fallback** | Primary Groq/Gemini with automatic fallback across a second Groq key and same-provider models on rate-limit (429) or empty responses — rate limits are per-org, so two keys = two token pools |
| **Lean Prompt / `INTERVIEW_MODE`** | Default ~780-token system prompt to stay well under Groq's free-tier TPM cap. `INTERVIEW_MODE=technical` restores the full AWS/Linux/Terraform/coding templates (~1,500 tokens) only when you need them |
| **First-Person Candidate Voice** | LLM speaks *as* the candidate — `"I built..."`, `"In my experience..."` — never generic career-coach advice |
| **HR Question Intelligence** | Salary, leadership, failures, conflict, career goals — all answered concretely in first person, no code blocks |
| **Anti-Hallucination** | Only references skills/projects explicitly in `data/portfolio.md` / `data/interview_context.md` |
| **RAG Confidence Filtering** | ChromaDB distance + keyword rerank — irrelevant chunks silently dropped, LLM never sees noise |
| **Smooth Token Streaming** | `stream=True` — answer appears word-by-word in realtime, not all at once |

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
> **⚠️ Python Version Requirement:** You **must** use Python 3.12 or 3.13. Python 3.14+ is effectively unsupported by the `pyaudio` community wheels on Windows and will fail to install unless you manually install C++ build tools.

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create `.env` in the project root:
```env
# Required
GROQ_API_KEY=your_groq_api_key_here
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# Optional — resilience & tuning
GROQ_BACKUP_API_KEY=second_groq_key_here   # different org = a second TPM pool for 429 fallback
GEMINI_API_KEY=your_gemini_key_here         # alternative provider (set LLM_PROVIDER=gemini to use)
LLM_PROVIDER=groq                           # groq (default) | gemini | nvidia
ANSWER_STYLE=spoken                         # spoken (default, natural prose) | standard (structured)
INTERVIEW_MODE=general                      # general (lean prompt, default) | technical (full detail)
```

> **Tip:** Leave `INTERVIEW_MODE` unset (lean) for behavioral/support interviews to stay under Groq's 6,000 TPM free-tier cap. Set `INTERVIEW_MODE=technical` for engineering/DevOps interviews to restore the detailed technical answer templates.

### 4. Add Your Context
- `data/portfolio.md` — your resume, skills, projects, certifications (your fixed background).
- `data/interview_context.md` — **per-interview** tailoring (target role, tone, focus areas). Swap this before each interview. Both are auto-ingested into RAG on startup.

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

70+ unit tests, no API keys required — all mocked:

```bash
python -m pytest tests/ -v

# If 'deepgram' or 'chromadb' aren't installed, skip the modules that need them:
python -m pytest tests/ --ignore=tests/test_transcription.py -q
```

| Module | Tests |
|---|---|
| `test_llm.py` | Model router, provider response parsing (Groq vs Gemini), streaming (token assembly, SKIP, stream-close on interruption), turn classification, multi-key/model fallback, lean vs technical prompt, spoken style |
| `test_turn_detection.py` | Pause-based decisions (complete/ambiguous/incomplete/force), short follow-ups, fine-grained poll timing |
| `test_bridge_lines.py` | Thinking-bridge pool routing, anti-repeat, no evaluative phrasing |
| `test_rag.py` | RAG ingestion (normal, missing, empty, chunk count), retrieval (relevance, threshold, separator) |
| `test_transcription.py` | Engine init, `_setup_connection` (success/failure), reconnect logic, queue overflow *(needs `deepgram`)* |

---

## 🏗️ Architecture

```
interview-copilot/
├── main.py                  # Async orchestrator — gating, hotkeys, streaming, Q&A history
├── core/
│   ├── capture.py           # Dual WASAPI loopback + mic capture (PyAudioWPatch)
│   ├── transcription.py     # Deepgram WebSocket STT with auto-reconnect
│   ├── rag.py               # ChromaDB in-memory ingestion + confidence-filtered retrieval
│   ├── turn_detection.py    # Pure pause/turn logic — silence-gap thresholds, poll interval
│   ├── bridge_lines.py      # Instant thinking-bridge openers (pure, no API call)
│   └── llm.py               # Multi-provider LLM client — router, streaming, fallback, prompts
├── ui/
│   └── overlay.py           # PyQt6 resizable overlay + system tray icon
├── tests/
│   ├── test_rag.py             # RAG unit tests
│   ├── test_llm.py             # LLM client unit tests
│   ├── test_turn_detection.py  # Pause/turn detection unit tests
│   ├── test_bridge_lines.py    # Thinking-bridge unit tests
│   └── test_transcription.py   # Transcription unit tests
├── assets/
│   └── fonts/
│       └── JetBrainsMono-Regular.ttf
└── data/
    ├── portfolio.md            # ← Your resume/skills (fixed background)
    └── interview_context.md    # ← Per-interview tailoring (role, tone, focus)
```

---

## 🔑 Required APIs

| API | Purpose | Free Tier |
|---|---|---|
| [Groq](https://console.groq.com) | LLM inference — Llama 3.3 70B + Llama 3.1 8B | ✅ Yes (30 RPM / 6K TPM on 8B) |
| [Deepgram](https://console.deepgram.com) | Real-time speech-to-text (Nova-2) | ✅ Yes (~$200 credit) |
| [Gemini](https://aistudio.google.com) | Optional alternative LLM provider | ✅ Yes |

---

## 📦 Dependencies

```
PyQt6
qasync
groq
openai            # OpenAI-compatible client (NVIDIA NIM + fallbacks)
google-genai      # optional — Gemini provider
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
| `chroma_db/` | ✅ Git-ignored (in-memory ChromaDB, no disk persistence) |

> **Verified:** `git log --all -- .env` returns empty — the `.env` file was **never committed at any point** in the git history.

---

## 🏆 Production Readiness:

| Area | Score | Notes |
|---|---|---|
| Architecture | ✅ Clean module separation, async throughout |
| Error Handling | ✅  Every external call wrapped, UI shows error state on failure |
| Security | ✅  No secrets in code or history |
| Reliability | ✅ WS auto-reconnect, bounded queues, log rotation |
| Test Coverage | ✅ 70+ tests across all core modules |
| Documentation | ✅ This README |

---

*Built to surpass Final Round AI.*
