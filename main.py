# IMPORTANT: chromadb/onnxruntime must be imported BEFORE any PyQt6/qasync modules on Windows
# to prevent DLL load order conflicts that cause "onnxruntime not installed" errors.
import chromadb
import onnxruntime

import sys
import re
import time
import json
import asyncio
import logging
import threading
import qasync
import keyboard
from dotenv import load_dotenv

from core.capture import CaptureEngine
from core.transcription import TranscriptionEngine
from core.rag import RAGManager
from core.llm import LLMClient
import logging.handlers
from ui.overlay import UIOverlay, create_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure background session logger — rotates daily, keeps 7 days
session_logger = logging.getLogger("session_recorder")
session_logger.setLevel(logging.INFO)
file_handler = logging.handlers.TimedRotatingFileHandler(
    "interview_session.log",
    when="midnight",
    backupCount=7,
    encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
session_logger.addHandler(file_handler)

# Thread-safe events for global hotkeys
manual_trigger_event = threading.Event()  # Ctrl+Shift+Space
regen_trigger_event = threading.Event()   # Ctrl+R

async def process_transcripts(transcription_engine, rag_manager, llm_client, ui_overlay):
    transcript_history = []
    
    interviewer_accumulator = []
    me_accumulator = []
    
    last_question = ""
    last_advice = ""

    # Load qa_history from previous session (persists across restarts)
    QA_HISTORY_FILE = "qa_history.json"
    try:
        with open(QA_HISTORY_FILE, "r", encoding="utf-8") as f:
            qa_history = [tuple(pair) for pair in json.load(f)]
        logger.info(f"Loaded {len(qa_history)} Q&A pair(s) from previous session.")
    except (FileNotFoundError, json.JSONDecodeError):
        qa_history = []

    def _save_qa_history():
        """Persist qa_history to disk so it survives restarts."""
        try:
            with open(QA_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(qa_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not save qa_history: {e}")

    
    def markdown_to_html(text: str) -> str:
        """Convert LLM markdown output to styled HTML for the QLabel."""
        try:
            lines = text.split('\n')
            html_lines = []
            in_code_block = False
            code_lang = ""
            code_lines = []

            for line in lines:
                # --- Fenced code block handling ---
                if line.strip().startswith("```"):
                    if not in_code_block:
                        in_code_block = True
                        code_lang = line.strip()[3:].strip()
                        code_lines = []
                    else:
                        # End of code block — render it
                        code_content = "\n".join(code_lines).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                        label = f"<span style='color:#aaaaaa; font-size:11px;'>{code_lang}</span><br>" if code_lang else ""
                        code_font = ui_overlay.CODE_FONT
                        html_lines.append(
                            f"<div style='background:#0d1117; border-left:3px solid #00fa9a; "
                            f"border-radius:5px; padding:10px 12px; margin:6px 0; "
                            f"font-family:\"{code_font}\",Consolas,monospace; "
                            f"font-size:12.5px; line-height:1.5; color:#c9d1d9;'>"
                            f"{label}{code_content}</div>"
                        )
                        in_code_block = False
                        code_lang = ""
                    continue

                if in_code_block:
                    code_lines.append(line)
                    continue

                # --- Inline formatting ---
                line = re.sub(r'\*\*(.+?)\*\*', r"<b style='color:#ffffff;'>\1</b>", line)
                line = re.sub(r'`([^`]+)`', r"<code style='background:#1a1a2e; color:#00fa9a; font-family:Consolas,monospace; padding:1px 4px; border-radius:3px;'>\1</code>", line)

                # --- Bullet points: • or - at start ---
                stripped = line.strip()
                if stripped.startswith("•") or (stripped.startswith("-") and len(stripped) > 2):
                    content = stripped.lstrip("•- ").strip()
                    html_lines.append(
                        f"<div style='margin:3px 0 3px 8px; color:#cccccc;'>"
                        f"<span style='color:#00fa9a; font-weight:bold;'>▸</span>&nbsp;{content}</div>"
                    )
                elif stripped == "":
                    html_lines.append("<br>")
                else:
                    html_lines.append(f"<span style='color:#cccccc;'>{line}</span><br>")

            return "".join(html_lines)
        except Exception as e:
            logger.warning(f"markdown_to_html failed: {e}")
            return f"<span style='color:#cccccc;'>{text}</span>"

    def update_ui():
        try:
            html = ""

            # Render historical Q&A pairs (faded) — oldest first
            for i, (hist_q, hist_a) in enumerate(qa_history):
                opacity = 0.35 + (i / max(len(qa_history), 1)) * 0.35
                fade = f"opacity:{opacity:.2f};"
                advice_html = markdown_to_html(hist_a)
                html += (
                    f"<div style='{fade} margin-bottom:6px;'>"
                    f"<div style='border-left:2px solid #336644; padding:4px 8px; border-radius:3px;'>"
                    f"<span style='color:#607060; font-size:10px; font-weight:bold;'>PREV QUESTION</span><br>"
                    f"<span style='color:#aaaaaa; font-size:12px;'>{hist_q}</span></div>"
                    f"<div style='padding:4px 8px; color:#888888; font-size:12px;'>{advice_html}</div>"
                    f"</div>"
                    f"<hr style='border:none; border-top:1px solid #2a2a2a; margin:4px 0;'>"
                )

            # Render current active Q&A (full brightness)
            if last_question:
                html += (
                    f"<div style='background:#1e2a1e; border-left:3px solid #00fa9a; "
                    f"padding:6px 10px; border-radius:4px; margin-bottom:8px;'>"
                    f"<span style='color:#aaaaaa; font-size:11px; font-weight:bold;'>INTERVIEWER</span><br>"
                    f"<b style='color:#ffffff; font-size:14px;'>{last_question}</b></div>"
                )

            if last_advice:
                advice_html = markdown_to_html(last_advice)
                html += (
                    f"<div style='margin-top:4px;'>"
                    f"<span style='color:#00fa9a; font-size:11px; font-weight:bold;'>⚡ COPILOT</span><br>"
                    f"{advice_html}</div>"
                )

            ui_overlay.update_text(html)
        except Exception as e:
            logger.warning(f"update_ui failed: {e}")
            ui_overlay.update_text("<span style='color:#ff6b6b;'>⚠ UI render error</span>")


    # Set of words that indicate a sentence is cut off mid-thought
    # If the accumulated text ends with one of these, keep listening
    DANGLING_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'so', 'if', 'when', 'while',
        'because', 'since', 'although', 'with', 'without', 'using', 'by',
        'for', 'from', 'to', 'in', 'on', 'at', 'of', 'about', 'into',
        'through', 'during', 'before', 'after', 'between', 'under', 'over',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'has', 'have', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'might', 'may', 'can',
        'that', 'which', 'who', 'whom', 'whose', 'where', 'what',
        'we', 'they', 'i', 'you', 'he', 'she', 'it', 'our', 'their', 'my',
        'this', 'these', 'those', 'then', 'than', 'also', 'very',
        'not', 'just', 'only', 'even', 'still', 'already',
        'like', 'such', 'each', 'every', 'both', 'either', 'neither',
        'regarding', 'concerning', 'including', 'especially', 'specifically',
    }

    # Track when the interviewer started speaking for max-wait safety valve
    interviewer_start_time = None

    while True:
        try:
            # --- MANUAL HOTKEY TRIGGER ---
            # Check if Ctrl+Shift+Space was pressed. If so, bypass all gates immediately.
            if manual_trigger_event.is_set():
                manual_trigger_event.clear()
                if interviewer_accumulator or transcript_history:
                    try:
                        q_text = " ".join(interviewer_accumulator) if interviewer_accumulator else last_question
                        if not q_text:
                            for msg in reversed(transcript_history):
                                if msg.startswith("[INTERVIEWER]"):
                                    q_text = msg.replace("[INTERVIEWER]: ", "").strip()
                                    break
                        logger.info(f"Hotkey: Manual trigger fired! Forcing LLM call.")
                        interviewer_accumulator.clear()
                        interviewer_start_time = None
                        last_advice = "<i style='color:#888888'>Copilot Thinking (Manual)...</i>"
                        update_ui()
                        context = rag_manager.retrieve_context(q_text)
                        answer = await llm_client.generate_answer_regen(transcript_history, context, last_question=q_text)
                        answer_clean = answer.strip() if answer else ""
                        if answer_clean and not answer_clean.startswith("Error"):
                            session_logger.info(f"[COPILOT ADVICE (MANUAL)]:\n{answer_clean}\n" + "="*50)
                            last_advice = answer_clean
                        else:
                            last_advice = "<i style='color:#888888'>(No answer generated)</i>"
                    except Exception as e:
                        logger.error(f"Manual trigger failed: {e}")
                        last_advice = "<i style='color:#ff6b6b'>⚠ Manual trigger error — check logs</i>"
                    update_ui()

                continue

            # --- REGENERATION HOTKEY (Ctrl+R) ---
            # Re-run the LLM on the last question with a fresh answer.
            if regen_trigger_event.is_set():
                regen_trigger_event.clear()
                if last_question or transcript_history:
                    try:
                        logger.info("Hotkey: Ctrl+R fired! Regenerating last answer...")
                        last_advice = "<i style='color:#888888'>Regenerating...</i>"
                        update_ui()
                        q_text = last_question if last_question else "(use recent conversation context)"
                        context = rag_manager.retrieve_context(q_text)
                        answer = await llm_client.generate_answer_regen(transcript_history, context, last_question=last_question)
                        answer_clean = answer.strip() if answer else ""
                        if answer_clean and answer_clean != "SKIP" and not answer_clean.startswith("Error"):
                            session_logger.info(f"[COPILOT ADVICE (REGEN)]:\n{answer_clean}\n" + "="*50)
                            last_advice = answer_clean
                            if qa_history:
                                qa_history[-1] = (last_question, answer_clean)
                                _save_qa_history()  # Persist to disk
                        else:
                            last_advice = "<i style='color:#888888'>(No regenerated answer)</i>"
                    except Exception as e:
                        logger.error(f"Regen failed: {e}")
                        last_advice = "<i style='color:#ff6b6b'>⚠ Regeneration error — check logs</i>"
                    update_ui()
                continue

            try:
                # Wait for incoming transcriptions.
                # 2.5s timeout for fast responsiveness, with rule-based gating below.
                tag, sentence = await asyncio.wait_for(transcription_engine.text_queue.get(), timeout=2.5)
                
                # Log everything to background session file
                session_logger.info(f"{tag}: {sentence}")
                
                # Append to persistent history for the LLM context sliding window
                message = f"{tag}: {sentence}"
                transcript_history.append(message)
                if len(transcript_history) > 30:
                    transcript_history.pop(0)

                if tag == "[INTERVIEWER]":
                    if me_accumulator:
                        # User stopped speaking, Interviewer is talking again. 
                        # Clear old advice and user speech to make room for new question.
                        me_accumulator.clear()
                        last_advice = ""
                        interviewer_accumulator.clear()
                        last_question = ""
                        
                    interviewer_accumulator.append(sentence)
                    last_question = " ".join(interviewer_accumulator)
                    last_advice = "<i style='color:#888888'>Listening to interviewer...</i>"
                    update_ui()
                    
                    # Start the max-wait timer when the interviewer first speaks
                    if interviewer_start_time is None:
                        interviewer_start_time = time.time()
                    
                elif tag == "[ME]":
                    me_accumulator.append(sentence)
                    interviewer_start_time = None  # Reset timer when user speaks
                    # We no longer call update_ui() here so user voice is hidden from the overlay

            except asyncio.TimeoutError:
                # 2.5 seconds of silence detected.
                # If the interviewer was the last one to speak, check if their thought is complete!
                if interviewer_accumulator:
                    q_text = " ".join(interviewer_accumulator)
                    time_waited = time.time() - interviewer_start_time if interviewer_start_time else 0
                    
                    # --- RULE-BASED END-OF-TURN DETECTION ---
                    words = q_text.strip().split()
                    
                    # Gate 1: Minimum word count - no real question is under 4 words
                    if len(words) < 4 and time_waited < 8.0:
                        logger.info(f"Rule Gate: Too few words ({len(words)}). Continuing to listen...")
                        continue
                    
                    # Gate 2: Comma-aware - speaker is listing items, not done yet
                    if q_text.strip().endswith(',') and time_waited < 8.0:
                        logger.info(f"Rule Gate: Trailing comma detected. Continuing to listen...")
                        continue
                    
                    # Gate 3: Dangling word check - last word is a preposition/conjunction
                    last_word = words[-1].lower().rstrip('.,!?;:') if words else ""
                    if last_word in DANGLING_WORDS and time_waited < 8.0:
                        logger.info(f"Rule Gate: Dangling word '{last_word}' detected. Continuing to listen...")
                        continue
                    
                    if time_waited >= 8.0:
                        logger.info(f"Rule Gate: Max wait (8s) reached. Forcing answer generation.")
                    else:
                        # --- COOLDOWN WINDOW ---
                        # All gates passed, but wait 1.5s more to see if interviewer keeps talking.
                        # This catches compound questions delivered as rapid-fire short sentences.
                        try:
                            tag2, sentence2 = await asyncio.wait_for(
                                transcription_engine.text_queue.get(), timeout=1.5
                            )
                            # Someone spoke during cooldown!
                            session_logger.info(f"{tag2}: {sentence2}")
                            message2 = f"{tag2}: {sentence2}"
                            transcript_history.append(message2)
                            if len(transcript_history) > 30:
                                transcript_history.pop(0)
                                
                            if tag2 == "[INTERVIEWER]":
                                # Interviewer is still going! Accumulate and restart the loop.
                                interviewer_accumulator.append(sentence2)
                                last_question = " ".join(interviewer_accumulator)
                                update_ui()
                                logger.info("Cooldown: Interviewer still speaking. Continuing to listen...")
                                continue
                            elif tag2 == "[ME]":
                                me_accumulator.append(sentence2)
                                # User started speaking during cooldown — generate answer NOW
                                logger.info("Cooldown: User started speaking. Generating answer immediately.")
                        except asyncio.TimeoutError:
                            # 1.5s of confirmed silence after gates passed — commit to answer
                            logger.info("Cooldown: Confirmed silence. Generating answer.")
                        
                    interviewer_accumulator.clear() # Clear so we don't re-trigger
                    interviewer_start_time = None  # Reset timer
                    
                    last_advice = "<i style='color:#888888'>Copilot Thinking...</i>"
                    update_ui()
                    
                    # Fetch RAG
                    context = rag_manager.retrieve_context(q_text)
                    
                    # --- STREAMING GENERATION ---
                    # Show raw plain text while streaming (markdown parsing only at the end)
                    answer_clean = ""
                    skipped = False
                    batch_buffer = ""
                    BATCH_CHARS = 8

                    async for token in llm_client.generate_answer_stream(transcript_history, context):
                        if token == "__SKIP__":
                            skipped = True
                            break
                        answer_clean += token
                        batch_buffer += token
                        # Update UI with raw text every ~8 chars or at sentence boundaries
                        if len(batch_buffer) >= BATCH_CHARS or token in ".!?\n":
                            # Show raw text during streaming — fast, no markdown parsing
                            ui_overlay.update_text(
                                f"<span style='color:#aaaaaa; font-size:11px; font-weight:bold;'>⚡ COPILOT (streaming...)</span><br>"
                                f"<span style='color:#cccccc; font-size:14px; font-family:Segoe UI;'>{answer_clean.replace('<','&lt;').replace('>','&gt;')}</span>"
                            )
                            batch_buffer = ""
                            await asyncio.sleep(0.03)  # Give Qt time to repaint

                    if skipped:
                        logger.info("LLM skipped conversational filler.")
                        last_advice = "<i style='color:#888888'>(Skipped conversational filler)</i>"
                    elif answer_clean and not answer_clean.startswith("Error"):
                        logger.info(f"LLM Advice generated.")
                        session_logger.info(f"[COPILOT ADVICE]:\n{answer_clean}\n" + "="*50)
                        last_advice = answer_clean  # Full markdown render happens in update_ui()
                        # Push completed Q&A to scrollable history (keep last 3)
                        qa_history.append((last_question, answer_clean))
                        if len(qa_history) > 3:
                            qa_history.pop(0)
                        _save_qa_history()  # Persist to disk
                    else:
                        logger.info(f"LLM skipped conversational filler.")
                        last_advice = "<i style='color:#888888'>(Skipped conversational filler)</i>"
                        
                    update_ui()  # Final render with full markdown formatting


                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in conversation processing: {e}")

def main():
    """
    Wires all modules together async via QEventLoop.
    """
    load_dotenv()
    
    # Register global hotkeys — requires keyboard library (may need admin on Windows)
    try:
        keyboard.add_hotkey('ctrl+shift+space', lambda: manual_trigger_event.set())
        logger.info("Hotkey registered: Ctrl+Shift+Space = Manual LLM Trigger")
        keyboard.add_hotkey('ctrl+r', lambda: regen_trigger_event.set())
        logger.info("Hotkey registered: Ctrl+R = Regenerate Last Answer")
    except Exception as e:
        logger.warning(
            f"Hotkey registration failed: {e}\n"
            "Try running as Administrator, or install: pip install keyboard\n"
            "Hotkeys (Ctrl+Shift+Space, Ctrl+R) will be DISABLED this session."
        )
    
    app = create_app()
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    ui_overlay = UIOverlay()

    # Initialize RAG & LLM
    rag_manager = RAGManager("data/portfolio.md")
    rag_manager.ingest_portfolio()
    llm_client = LLMClient()
    
    # Initialize Queues & Core Engines
    # maxsize caps prevent unbounded memory growth if LLM/processing is slow
    audio_queue = asyncio.Queue(maxsize=200)
    text_queue = asyncio.Queue(maxsize=100)
    
    capture_engine = CaptureEngine()
    transcription_engine = TranscriptionEngine()
    
    async def run_async_tasks():
        capture_engine.start_capture(audio_queue)
        t_task = asyncio.create_task(transcription_engine.transcribe_stream(audio_queue, text_queue))
        p_task = asyncio.create_task(process_transcripts(transcription_engine, rag_manager, llm_client, ui_overlay))
        
        try:
            # Hangs execution allowing async operations and GUI to function side by side
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            capture_engine.stop_capture()
            t_task.cancel()
            p_task.cancel()

    try:
        with loop:
            loop.run_until_complete(run_async_tasks())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    finally:
        # Crucial to kill any lingering PyQt background threads/windows on Windows
        app.quit()
        sys.exit(0)

if __name__ == "__main__":
    main()
