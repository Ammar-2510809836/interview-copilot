# Interview Copilot

An AI-powered interview assistant that provides real-time, context-aware advice during online interviews.

## Features
- **Real-time Dual Audio Capture**: Captures both your microphone and the interviewer's voice (loopback) simultaneously.
- **Lightning Fast Transcription**: Uses Deepgram's WebSocket API for sub-second speech-to-text.
- **Smart End-of-Turn Gating**: Employs an intelligent, rule-based semantic gatekeeper (comma-aware, dangling word detection) to perfectly identify when the interviewer stops speaking, handling complex compound questions flawlessly.
- **RAG-Powered Advice**: Uses ChromaDB to ingest your resume/portfolio, ensuring the AI only recommends skills you actually possess.
- **Anti-Hallucination Framework**: Strict LLM prompts force concise, bulleted, and contextually accurate answers without inventing false technical experience.
- **Teleprompter UI**: A sleek, non-intrusive, always-on-top PyQt6 overlay that presents advice in a professional teleprompter format.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   DEEPGRAM_API_KEY=your_deepgram_api_key
   ```
4. Place your resume/portfolio content in `data/portfolio.md` (the RAG system will auto-ingest this on startup).

## Usage
Run the main application:
```bash
python main.py
```
*Note: Depending on your system, you may need to configure the correct audio loopback device index in `core/capture.py`.*

## Architecture
- **core/capture.py**: Captures system audio and microphone using PyAudioWPatch.
- **core/transcription.py**: Streams audio to Deepgram via WebSockets.
- **core/rag.py**: Manages ChromaDB document chunking and retrieval.
- **core/llm.py**: Interfaces with Groq (Llama 3.3 70b) with strict formatting/anti-hallucination rules.
- **ui/overlay.py**: PyQt6 translucent window that renders the advice.
- **main.py**: The central async orchestrator that ties all components together and runs the smart gating logic.
