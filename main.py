# IMPORTANT: chromadb/onnxruntime must be imported BEFORE any PyQt6/qasync modules on Windows
# to prevent DLL load order conflicts that cause "onnxruntime not installed" errors.
import chromadb
import onnxruntime

import os
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

def get_run_dir():
    """Returns the external directory where the user launched the app/exe."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

run_dir = get_run_dir()

# Ensure logs and history write to the user directory, not the internal PyInstaller temp path
file_handler = logging.handlers.TimedRotatingFileHandler(
    os.path.join(run_dir, "interview_session.log"),
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

    # --- Conversation Memory State ---
    conversation_state = {
        "current_question_type": None,  # behavioral, technical, coding
        "previous_question": "",
        "previous_answer": "",
        "conversation_summary": "",
    }

    def is_followup_question(current_text: str, previous_question: str) -> bool:
        """
        Detect if current text is a follow-up to the previous question.
        Uses heuristics like follow-up keywords and contextual references.
        """
        if not previous_question or not current_text:
            return False

        text_lower = current_text.lower().strip()

        # Follow-up indicator words/phrases
        followup_starters = [
            "but", "what if", "how about", "why", "can you", "could you",
            "would you", "do you", "how would", "what about", "what do you think",
            "follow up", "following up", "going back", "referring to",
            "you mentioned", "as you said", "that approach", "that solution",
            "elaborate", "explain more", "tell me more", "expand on"
        ]

        # Check for follow-up starters
        for starter in followup_starters:
            if text_lower.startswith(starter) or f" {starter} " in text_lower:
                logger.info(f"Follow-up detected: starts with '{starter}'")
                return True

        # Check if it's a short question after a long answer (likely follow-up)
        word_count = len(text_lower.split())
        if word_count < 8 and conversation_state["previous_answer"]:
            logger.info(f"Follow-up detected: short question ({word_count} words) with previous answer")
            return True

        # Check for pronoun references to previous context
        reference_words = ["it", "that", "this", "those", "they", "them"]
        first_words = text_lower.split()[:3]
        if any(w in reference_words for w in first_words) and conversation_state["previous_answer"]:
            logger.info("Follow-up detected: pronoun reference to previous context")
            return True

        return False

    def update_conversation_summary(question: str, answer: str, q_type: str):
        """Update the running conversation summary."""
        summary_parts = []

        if q_type == "behavioral":
            # Extract the topic from question
            summary_parts.append(f"Discussed: {question[:60]}...")
        elif q_type == "technical":
            summary_parts.append(f"Covered technical topic: {question[:50]}...")
        elif q_type == "coding":
            summary_parts.append(f"Solved coding problem: {question[:50]}...")
        else:
            summary_parts.append(f"Q: {question[:50]}...")

        # Keep only last 2-3 topics in summary (truncate to ~200 chars)
        current = conversation_state["conversation_summary"]
        new_entry = " | ".join(summary_parts)

        if current:
            # Append but keep reasonable length
            combined = f"{current} | {new_entry}"
            if len(combined) > 300:
                # Keep last part
                combined = combined[-300:]
                # Try to start at a clean boundary
                if " | " in combined:
                    combined = combined.split(" | ", 1)[1]
            conversation_state["conversation_summary"] = combined
        else:
            conversation_state["conversation_summary"] = new_entry

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
    
    # Tracks the last full question that was answered, so if the interviewer
    # continues speaking right after we answered, we can prepend the context
    last_answered_question = ""
    last_answer_time = 0  # time.time() when the last answer was generated

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

                        # Classify and detect follow-up
                        q_type = await llm_client.classify_question(q_text)
                        is_followup = is_followup_question(q_text, conversation_state["previous_question"])

                        context = rag_manager.retrieve_context(q_text)
                        answer = await llm_client.generate_answer_regen(
                            transcript_history, context, last_question=q_text,
                            question_type=q_type, conversation_summary=conversation_state["conversation_summary"]
                        )
                        answer_clean = answer.strip() if answer else ""
                        if answer_clean and not answer_clean.startswith("Error"):
                            session_logger.info(f"[COPILOT ADVICE (MANUAL)]:\n{answer_clean}\n" + "="*50)
                            last_advice = answer_clean
                            transcript_history.append(f"[COPILOT]: {answer_clean}")
                            if len(transcript_history) > 30:
                                transcript_history.pop(0)
                            # Update conversation state
                            conversation_state["current_question_type"] = q_type
                            conversation_state["previous_question"] = q_text
                            conversation_state["previous_answer"] = answer_clean
                            update_conversation_summary(q_text, answer_clean, q_type)
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

                        # Use existing question type if available
                        q_type = conversation_state.get("current_question_type")
                        if not q_type:
                            q_type = await llm_client.classify_question(q_text)

                        context = rag_manager.retrieve_context(q_text)
                        answer = await llm_client.generate_answer_regen(
                            transcript_history, context, last_question=last_question,
                            question_type=q_type, conversation_summary=conversation_state["conversation_summary"]
                        )
                        answer_clean = answer.strip() if answer else ""
                        if answer_clean and answer_clean != "SKIP" and not answer_clean.startswith("Error"):
                            session_logger.info(f"[COPILOT ADVICE (REGEN)]:\n{answer_clean}\n" + "="*50)
                            last_advice = answer_clean
                            transcript_history.append(f"[COPILOT]: {answer_clean}")
                            if len(transcript_history) > 30:
                                transcript_history.pop(0)
                            # Update previous answer
                            conversation_state["previous_answer"] = answer_clean
                        else:
                            last_advice = "<i style='color:#888888'>(No regenerated answer)</i>"
                    except Exception as e:
                        logger.error(f"Regen failed: {e}")
                        last_advice = "<i style='color:#ff6b6b'>⚠ Regeneration error — check logs</i>"
                    update_ui()
                continue

            try:
                # Wait for incoming transcriptions.
                # 3.5s timeout — long enough for natural interviewer pauses between sentences.
                tag, sentence = await asyncio.wait_for(transcription_engine.text_queue.get(), timeout=3.5)
                
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
                    
                    # --- CONTINUATION DETECTION ---
                    # If the interviewer starts speaking again within 8 seconds of our
                    # last answer, they were probably still asking the same question
                    # and we jumped the gun. Prepend the previous question as context.
                    if (last_answered_question 
                        and not interviewer_accumulator 
                        and (time.time() - last_answer_time) < 8.0):
                        logger.info("Continuation detected: Interviewer resumed after premature answer. Prepending previous context.")
                        interviewer_accumulator.append(last_answered_question)
                        last_answered_question = ""  # Only do this once
                        
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
                    if len(words) < 4 and time_waited < 12.0:
                        logger.info(f"Rule Gate: Too few words ({len(words)}). Continuing to listen...")
                        continue
                    
                    # Gate 2: Comma-aware - speaker is listing items, not done yet
                    if q_text.strip().endswith(',') and time_waited < 12.0:
                        logger.info(f"Rule Gate: Trailing comma detected. Continuing to listen...")
                        continue
                    
                    # Gate 3: Dangling word check - last word is a preposition/conjunction
                    last_word = words[-1].lower().rstrip('.,!?;:') if words else ""
                    if last_word in DANGLING_WORDS and time_waited < 12.0:
                        logger.info(f"Rule Gate: Dangling word '{last_word}' detected. Continuing to listen...")
                        continue
                    
                    if time_waited >= 12.0:
                        logger.info(f"Rule Gate: Max wait (12s) reached. Forcing answer generation.")
                    else:
                        # --- COOLDOWN WINDOW ---
                        # All gates passed, but wait 3.0s more to see if interviewer keeps talking.
                        # This catches compound questions and natural pauses between sentences.
                        try:
                            tag2, sentence2 = await asyncio.wait_for(
                                transcription_engine.text_queue.get(), timeout=3.0
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
                            # 3.0s of confirmed silence after gates passed — commit to answer
                            logger.info("Cooldown: Confirmed silence. Generating answer.")
                        
                    # Save the full question before clearing, so we can use it as
                    # context if the interviewer continues speaking (continuation detection)
                    last_answered_question = q_text
                    last_answer_time = time.time()
                    interviewer_accumulator.clear() # Clear so we don't re-trigger
                    interviewer_start_time = None  # Reset timer

                    # --- QUESTION CLASSIFICATION & FOLLOW-UP DETECTION ---
                    q_type = await llm_client.classify_question(q_text)
                    is_followup = is_followup_question(q_text, conversation_state["previous_question"])

                    if is_followup:
                        logger.info(f"Detected follow-up question to: '{conversation_state['previous_question'][:50]}...'")

                    last_advice = "<i style='color:#888888'>Copilot Thinking...</i>"
                    update_ui()

                    # Fetch RAG with conversation history
                    context = rag_manager.retrieve_context(q_text, conversation_history=transcript_history)

                    # --- STREAMING GENERATION ---
                    answer_clean = ""
                    skipped = False
                    batch_buffer = ""
                    BATCH_CHARS = 8

                    # Show typing indicator
                    ui_overlay.set_typing_indicator(True)

                    async for token in llm_client.generate_answer_stream(
                        transcript_history, context,
                        question_type=q_type,
                        is_followup=is_followup,
                        conversation_summary=conversation_state["conversation_summary"]
                    ):
                        if token == "__SKIP__":
                            skipped = True
                            break
                        answer_clean += token
                        batch_buffer += token
                        # Update UI with raw text every ~8 chars or at sentence boundaries
                        if len(batch_buffer) >= BATCH_CHARS or token in ".!?\n":
                            # Show raw text during streaming with question type
                            ui_overlay.update_text(
                                answer_clean.replace('<','&lt;').replace('>','&gt;'),
                                question_type=q_type if q_type else "generic",
                                is_streaming=True
                            )
                            batch_buffer = ""
                            await asyncio.sleep(0.03)  # Give Qt time to repaint

                    # Hide typing indicator
                    ui_overlay.set_typing_indicator(False)

                    if skipped:
                        logger.info("LLM skipped conversational filler.")
                        last_advice = "<i style='color:#888888'>(Skipped conversational filler)</i>"
                    elif answer_clean and not answer_clean.startswith("Error"):
                        logger.info(f"LLM Advice generated.")
                        session_logger.info(f"[COPILOT ADVICE]:\n{answer_clean}\n" + "="*50)
                        last_advice = answer_clean
                        transcript_history.append(f"[COPILOT]: {answer_clean}")
                        if len(transcript_history) > 30:
                            transcript_history.pop(0)

                        # Update conversation state
                        conversation_state["current_question_type"] = q_type
                        conversation_state["previous_question"] = q_text
                        conversation_state["previous_answer"] = answer_clean
                        update_conversation_summary(q_text, answer_clean, q_type)
                    else:
                        logger.info(f"LLM skipped conversational filler.")
                        last_advice = "<i style='color:#888888'>(Skipped conversational filler)</i>"

                    # Final render with full markdown formatting
                    ui_overlay.update_text(last_advice, question_type=q_type if q_type else "generic", is_streaming=False)


                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in conversation processing: {e}")

def main():
    """
    Wires all modules together async via QEventLoop.
    """
    # Load .env file from the user's run directory (so clients can supply their own keys)
    run_dir = get_run_dir()
    load_dotenv(os.path.join(run_dir, ".env"))
    
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
    # Look for portfolio.md in the data folder adjacent to the executable
    portfolio_path = os.path.join(run_dir, "data", "portfolio.md")
    
    # Ensure data folder exists natively if missing (so RAG doesn't crash if user didn't make folder)
    os.makedirs(os.path.dirname(portfolio_path), exist_ok=True)
    if not os.path.exists(portfolio_path):
        with open(portfolio_path, "w", encoding="utf-8") as f:
            f.write("# Enter your resume or portfolio skills here!\n")
            
    rag_manager = RAGManager(portfolio_path)
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
