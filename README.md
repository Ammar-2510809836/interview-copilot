# ⚡ Interview Copilot

> A real-time AI-powered interview assistant that listens, understands, and streams smart answers — directly to your screen as you speak.

---

## 🚀 Features

### Core Engine
| Feature | Description |
|---|---|
| **Dual Audio Capture** | Captures both your mic (`[ME]`) and interviewer's voice via system loopback (`[INTERVIEWER]`) simultaneously |
| **Real-time Transcription** | Deepgram WebSocket API — sub-second speech-to-text on dual channels |
| **Smart End-of-Turn Gating** | Semantic gatekeeper with comma-aware, dangling word detection + 1.5s silence cooldown to know exactly when the interviewer finishes speaking |
| **RAG-Powered Context** | ChromaDB auto-ingests your `data/portfolio.md` on startup — answers only reference skills you actually have |

### AI & Answer Quality
| Feature | Description |
|---|---|
| **Model Router** | Technical questions → `llama-3.3-70b-versatile` (deep answers). Behavioral/HR → `llama-3.1-8b-instant` (fast responses) |
| **First-Person Candidate Voice** | LLM speaks *as* the candidate — "I built...", "In my experience..." — never generic career-coach advice |
| **HR Question Intelligence** | Salary, leadership, failures, career goals, weaknesses — all answered concisely and concretely, no code blocks |
| **Anti-Hallucination** | Only references projects/skills explicitly stated in portfolio context |
| **RAG Confidence Filtering** | ChromaDB L2 distance threshold (≤1.2) — irrelevant chunks are silently dropped so the LLM isn't polluted with noise |
| **Smooth Token Streaming** | Groq `stream=True` — answer appears word-by-word instantly, not all at once |

### Hotkeys
| Hotkey | Action |
|---|---|
| `Ctrl+Shift+Space` | **Force answer now** — bypasses all timers, fires LLM immediately |
| `Ctrl+R` | **Regenerate** — generates a fresh, different answer to the last question (temperature 0.6) |

### UI & UX
| Feature | Description |
|---|---|
| **Resizable Overlay** | Drag any edge or corner to resize — no more fixed window size |
| **Scrollable Q&A History** | Last 3 Q&A pairs visible with progressive opacity fading (older = more faded) |
| **Copyable Text** | Click-drag to select any text in the overlay, `Ctrl+C` to copy |
| **JetBrains Mono Font** | Bundled in `assets/fonts/` — renders code blocks with a premium monospace font |
| **System Tray Icon** | Neon green ⚡ icon in Windows system tray — right-click to Show/Hide/Quit. Click X to hide (not quit) |
| **Markdown Rendering** | Bold keywords, inline code, fenced code blocks with syntax label, bullet points — all rendered as rich HTML |
| **Startup Notification** | Windows balloon toast on launch showing active hotkeys |

---

## 🛠️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/interview-copilot.git
cd interview-copilot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

### 4. Add Your Portfolio
Edit `data/portfolio.md` with your resume, projects, skills, and certifications. The RAG system auto-ingests this on every startup.

---

## ▶️ Usage

```bash
python main.py
```

The overlay appears immediately. The tray icon ⚡ appears in the system tray.

> **Note:** On first run, you may need to check your audio loopback device index in `core/capture.py` if the interviewer audio isn't being captured.

---

## ⌨️ Hotkey Reference

| Hotkey | Action |
|---|---|
| `Ctrl+Shift+Space` | Force immediate LLM answer (bypasses silence timers) |
| `Ctrl+R` | Regenerate last answer with a fresh take |

---

## 🏗️ Architecture

```
interview-copilot/
├── main.py                  # Async orchestrator — gating logic, hotkeys, main loop
├── core/
│   ├── capture.py           # Dual audio capture (PyAudioWPatch loopback + mic)
│   ├── transcription.py     # Deepgram WebSocket streaming transcription
│   ├── rag.py               # ChromaDB ingestion + similarity-filtered retrieval
│   └── llm.py               # Groq LLM client — model router, streaming, regen
├── ui/
│   └── overlay.py           # PyQt6 resizable overlay + system tray
├── assets/
│   └── fonts/
│       └── JetBrainsMono-Regular.ttf
└── data/
    └── portfolio.md         # Your resume/portfolio — edit this!
```

---

## 🔑 Required APIs

| API | Purpose | Free Tier |
|---|---|---|
| [Groq](https://console.groq.com) | LLM inference (Llama 3.3 70B / Llama 3.1 8B) | ✅ Yes |
| [Deepgram](https://console.deepgram.com) | Real-time speech-to-text | ✅ Yes ($200 credit) |

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

- API keys are stored in `.env` (git-ignored — never committed)
- `interview_session.log` and `chroma_db/` are also git-ignored

---

*Built to surpass Final Round AI.*
