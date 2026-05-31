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
from core.turn_detection import decide_turn_action, recommended_wait_timeout
from core.bridge_lines import pick_bridge_line
import logging.handlers
from ui.overlay import UIOverlay, create_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure background session logger â€” rotates daily, keeps 7 days
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
    manual_question_queue = asyncio.Queue(maxsize=25)
    loop = asyncio.get_running_loop()

    def enqueue_manual_question(question: str):
        def put_question():
            try:
                manual_question_queue.put_nowait(question)
            except asyncio.QueueFull:
                logger.warning("Manual question queue full; dropped pasted question.")

        loop.call_soon_threadsafe(put_question)

    ui_overlay.signals.manual_question_submitted.connect(enqueue_manual_question)

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
                        # End of code block â€” render it
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

                # --- Bullet points: â€¢ or - at start ---
                stripped = line.strip()
                if stripped.startswith("â€¢") or (stripped.startswith("-") and len(stripped) > 2):
                    content = stripped.lstrip("â€¢- ").strip()
                    html_lines.append(
                        f"<div style='margin:3px 0 3px 8px; color:#cccccc;'>"
                        f"<span style='color:#00fa9a; font-weight:bold;'>&rsaquo;</span>&nbsp;{content}</div>"
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
                    f"<span style='color:#00fa9a; font-size:11px; font-weight:bold;'>âš¡ COPILOT</span><br>"
                    f"{advice_html}</div>"
                )

            ui_overlay.update_text(html)
        except Exception as e:
            logger.warning(f"update_ui failed: {e}")
            ui_overlay.update_text("<span style='color:#ff6b6b;'>âš  UI render error</span>")


    async def answer_manual_question(q_text: str):
        """Answer a pasted question using the same context and style as live interview audio."""
        nonlocal last_question, last_advice

        q_text = (q_text or "").strip()
        if not q_text:
            return

        last_question = q_text
        last_advice = "<i style='color:#888888'>Copilot Thinking (Chat)...</i>"
        transcript_history.append(f"[INTERVIEWER]: {q_text}")
        if len(transcript_history) > 30:
            transcript_history.pop(0)
        session_logger.info(f"[CHAT QUESTION]: {q_text}")
        update_ui()

        q_type = await llm_client.classify_question(q_text)
        is_followup = is_followup_question(q_text, conversation_state["previous_question"])

        context = rag_manager.retrieve_context(q_text, conversation_history=transcript_history)
        answer_clean = ""
        skipped = False
        batch_buffer = ""
        BATCH_CHARS = 8

        ui_overlay.set_typing_indicator(True)
        try:
            chat_max_tokens = int(os.getenv("CHAT_MAX_TOKENS", "1400"))
        except ValueError:
            chat_max_tokens = 1400
        async for token in llm_client.generate_answer_stream(
            transcript_history,
            context,
            question_type=q_type,
            is_followup=is_followup,
            conversation_summary=conversation_state["conversation_summary"],
            max_tokens_override=chat_max_tokens
        ):
            if token == "__SKIP__":
                skipped = True
                break
            answer_clean += token
            batch_buffer += token
            if len(batch_buffer) >= BATCH_CHARS or token in ".!?\n":
                ui_overlay.update_text(
                    answer_clean.replace('<', '&lt;').replace('>', '&gt;'),
                    question_type=q_type if q_type else "generic",
                    is_streaming=True
                )
                batch_buffer = ""
                await asyncio.sleep(0.03)

        ui_overlay.set_typing_indicator(False)

        if skipped:
            last_advice = "<i style='color:#888888'>(Skipped conversational filler)</i>"
        elif answer_clean.startswith("Error"):
            last_advice = f"<span style='color:#ff6b6b'>Ã¢Å¡Â  {answer_clean}</span>"
        elif answer_clean:
            answer_clean = answer_clean.strip()
            session_logger.info(f"[COPILOT ADVICE (CHAT)]:\n{answer_clean}\n" + "="*50)
            last_advice = answer_clean
            transcript_history.append(f"[COPILOT]: {answer_clean}")
            if len(transcript_history) > 30:
                transcript_history.pop(0)
            conversation_state["current_question_type"] = q_type
            conversation_state["previous_question"] = q_text
            conversation_state["previous_answer"] = answer_clean
            update_conversation_summary(q_text, answer_clean, q_type)
        else:
            last_advice = "<i style='color:#888888'>(No answer generated)</i>"

        update_ui()

    # Track when the interviewer started speaking for max-wait safety valve
    interviewer_start_time = None
    # Track the interviewer's most recent word, so pauses are measured as the
    # silence gap since the last chunk rather than total turn duration.
    last_speech_time = None

    # Tracks the last full question that was answered, so if the interviewer
    # continues speaking right after we answered, we can prepend the context
    last_answered_question = ""
    last_answer_time = 0  # time.time() when the last answer was generated

    while True:
        try:
            # --- MANUAL CHAT INPUT ---
            # Pasted questions from the overlay use the same answer pipeline as live audio.
            try:
                pasted_question = manual_question_queue.get_nowait()
            except asyncio.QueueEmpty:
                pasted_question = None
            if pasted_question:
                await answer_manual_question(pasted_question)
                continue

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
                        last_speech_time = None
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
                        last_advice = "<i style='color:#ff6b6b'>âš  Manual trigger error â€” check logs</i>"
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
                        last_advice = "<i style='color:#ff6b6b'>âš  Regeneration error â€” check logs</i>"
                    update_ui()
                continue

            try:
                # Wait for incoming transcriptions. Once interviewer text exists,
                # use adaptive timeouts so complete questions answer quickly while
                # dangling fragments still get a longer continuation window.
                pending_question = " ".join(interviewer_accumulator)
                silence_timeout = recommended_wait_timeout(
                    pending_question,
                    conversation_state["previous_question"],
                )
                transcript_task = asyncio.create_task(transcription_engine.text_queue.get())
                manual_task = asyncio.create_task(manual_question_queue.get())
                done, pending = await asyncio.wait(
                    {transcript_task, manual_task},
                    timeout=silence_timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()

                if not done:
                    raise asyncio.TimeoutError

                if manual_task in done:
                    await answer_manual_question(manual_task.result())
                    continue

                tag, sentence = transcript_task.result()
                
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
                    # Refresh the silence clock on every interviewer chunk so the
                    # pause is measured from their last word, not the turn start.
                    last_speech_time = time.time()

                elif tag == "[ME]":
                    me_accumulator.append(sentence)
                    interviewer_start_time = None  # Reset timer when user speaks
                    last_speech_time = None
                    # We no longer call update_ui() here so user voice is hidden from the overlay

            except asyncio.TimeoutError:
                # Adaptive silence detected. If the interviewer was the last one
                # to speak, check if their thought is complete.
                if interviewer_accumulator:
                    q_text = " ".join(interviewer_accumulator)
                    # Silence gap since the interviewer's last word — this is the
                    # signal for "have they paused long enough", not turn length.
                    pause = time.time() - last_speech_time if last_speech_time else 0

                    decision = decide_turn_action(
                        q_text,
                        pause,
                        conversation_state["previous_question"],
                    )
                    logger.info(
                        "Turn decision: reason=%s answer=%s force=%s pause=%.2fs",
                        decision.reason,
                        decision.should_answer,
                        decision.force,
                        pause,
                    )

                    if not decision.should_answer:
                        logger.info("Turn decision: continuing to listen.")
                        continue

                    if decision.force:
                        logger.info("Turn decision: max wait reached. Forcing answer generation.")
                        turn = {
                            "intent": "new_question",
                            "should_answer_now": True,
                            "confidence": 1.0,
                            "clean_question": q_text,
                        }
                    elif decision.reason == "ambiguous_pause":
                        turn = await llm_client.classify_turn(
                            q_text,
                            transcript_history=transcript_history,
                            previous_question=conversation_state["previous_question"],
                            was_answering=False
                        )
                        logger.info(
                            "Turn classifier: intent=%s answer_now=%s confidence=%.2f",
                            turn.get("intent"),
                            turn.get("should_answer_now"),
                            turn.get("confidence", 0.0)
                        )
                    else:
                        turn = {
                            "intent": "followup" if decision.reason == "short_followup" else "new_question",
                            "should_answer_now": True,
                            "confidence": 0.9,
                            "clean_question": q_text,
                        }

                    if not turn.get("should_answer_now", False):
                        intent = turn.get("intent")
                        if intent == "filler":
                            logger.info("Turn classifier: conversational filler. Clearing interviewer buffer.")
                            interviewer_accumulator.clear()
                            interviewer_start_time = None
                            last_speech_time = None
                            last_question = ""
                            last_advice = "<i style='color:#888888'>(Listening)</i>"
                            update_ui()
                        else:
                            logger.info("Turn classifier: interviewer turn is incomplete. Continuing to listen.")
                        continue

                    q_text = turn.get("clean_question") or q_text

                    # Save the full question before clearing, so we can use it as
                    # context if the interviewer continues speaking (continuation detection)
                    last_answered_question = q_text
                    last_answer_time = time.time()
                    interviewer_accumulator.clear() # Clear so we don't re-trigger
                    interviewer_start_time = None  # Reset timer
                    last_speech_time = None

                    # --- QUESTION CLASSIFICATION & FOLLOW-UP DETECTION ---
                    q_type = await llm_client.classify_question(q_text)
                    is_followup = (
                        turn.get("intent") in {"followup", "continuation", "interruption"}
                        or is_followup_question(q_text, conversation_state["previous_question"])
                    )

                    if is_followup:
                        logger.info(f"Detected follow-up question to: '{conversation_state['previous_question'][:50]}...'")

                    # Instant thinking-bridge: show a natural opener the moment the
                    # interviewer stops, so the candidate can start talking while the
                    # real answer generates. The spoken prompt emits no opener of its
                    # own, so the streamed text continues this line as one turn.
                    bridge_line = pick_bridge_line(q_text, is_followup)
                    ui_overlay.update_text(
                        bridge_line,
                        question_type=q_type if q_type else "generic",
                        is_streaming=True,
                    )

                    # Fetch RAG with conversation history
                    context = rag_manager.retrieve_context(q_text, conversation_history=transcript_history)

                    # --- STREAMING GENERATION ---
                    # Seed with the bridge so streamed tokens append after it.
                    answer_clean = bridge_line + " "
                    skipped = False
                    interrupted = False
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
                        # Zara-style AI interviewers can interrupt while the copilot is
                        # still streaming. Drain any queued transcript immediately and
                        # stop answering the stale question if the interviewer speaks.
                        while True:
                            try:
                                tag_live, sentence_live = transcription_engine.text_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

                            session_logger.info(f"{tag_live}: {sentence_live}")
                            message_live = f"{tag_live}: {sentence_live}"
                            transcript_history.append(message_live)
                            if len(transcript_history) > 30:
                                transcript_history.pop(0)

                            if tag_live == "[INTERVIEWER]":
                                logger.info("Interruption detected while streaming. Stopping current answer.")
                                interrupted = True
                                interviewer_accumulator.clear()
                                interviewer_accumulator.append(sentence_live)
                                last_question = sentence_live
                                last_advice = "<i style='color:#888888'>Listening to interviewer...</i>"
                                interviewer_start_time = time.time()
                                last_speech_time = time.time()
                                break
                            elif tag_live == "[ME]":
                                me_accumulator.append(sentence_live)

                        if interrupted:
                            break

                        if token == "__SKIP__":
                            skipped = True
                            break
                        answer_clean += token
                        batch_buffer += token
                        # Update UI with raw text every ~8 chars or at sentence boundaries
                        if len(batch_buffer) >= BATCH_CHARS or token in ".!?\n":
                            # Pass raw text — update_text/format_structured_answer
                            # html.escape()s it. Pre-escaping here double-escaped any
                            # answer containing < or > (e.g. code, List<int>).
                            ui_overlay.update_text(
                                answer_clean,
                                question_type=q_type if q_type else "generic",
                                is_streaming=True
                            )
                            batch_buffer = ""
                            await asyncio.sleep(0.03)  # Give Qt time to repaint

                    # Hide typing indicator
                    ui_overlay.set_typing_indicator(False)

                    if interrupted:
                        logger.info("Discarded partial answer because interviewer interrupted.")
                        last_advice = "<i style='color:#888888'>Listening to interviewer...</i>"
                        update_ui()
                        continue
                    elif skipped:
                        logger.info("LLM skipped conversational filler.")
                        # Replace the already-shown bridge with a cleared listening
                        # state so no dangling opener line is left on the overlay.
                        last_advice = "<i style='color:#888888'>(Listening)</i>"
                    elif answer_clean.startswith("Error"):
                        logger.error(f"LLM Error detected: {answer_clean}")
                        last_advice = f"<span style='color:#ff6b6b'>âš  {answer_clean}</span>"
                    elif answer_clean:
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
                        logger.info(f"LLM returned empty response.")
                        last_advice = "<i style='color:#888888'>(No answer generated)</i>"

                    # Final render with full markdown formatting
                    update_ui()


                    
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
    
    # Register global hotkeys â€” requires keyboard library (may need admin on Windows)
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

    interview_context_path = os.path.join(run_dir, "data", "interview_context.md")
    if not os.path.exists(interview_context_path):
        with open(interview_context_path, "w", encoding="utf-8") as f:
            f.write(
                "# Interview Context\n\n"
                "Add target role, company, job description, preferred answer style, "
                "technical focus areas, and interview-specific notes here. The copilot "
                "will retrieve this alongside portfolio.md.\n"
            )
            
    rag_manager = RAGManager(portfolio_path)
    rag_manager.ingest_portfolio()
    llm_client = LLMClient()
    
    # Initialize Queues & Core Engines
    # maxsize increased to handle bursts and prevent QueueFull errors
    audio_queue = asyncio.Queue(maxsize=1000)
    text_queue = asyncio.Queue(maxsize=500)
    
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
