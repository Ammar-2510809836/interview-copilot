# IMPORTANT: chromadb/onnxruntime must be imported BEFORE any PyQt6/qasync modules on Windows
# to prevent DLL load order conflicts that cause "onnxruntime not installed" errors.
import chromadb
import onnxruntime

import sys
import asyncio
import logging
import qasync
from dotenv import load_dotenv

from core.capture import CaptureEngine
from core.transcription import TranscriptionEngine
from core.rag import RAGManager
from core.llm import LLMClient
from ui.overlay import UIOverlay, create_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure background session logger for full history
session_logger = logging.getLogger("session_recorder")
session_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("interview_session.log", mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
session_logger.addHandler(file_handler)

async def process_transcripts(transcription_engine, rag_manager, llm_client, ui_overlay):
    transcript_history = []
    
    interviewer_accumulator = []
    me_accumulator = []
    
    last_question = ""
    last_advice = ""
    
    def update_ui():
        # Create a structured layout using basic HTML
        html = ""
        if last_question:
            html += f"<b style='color:#ffffff'>Q: {last_question}</b><br><br>"
            
        if last_advice:
            # Always show copilot's advice once it's available
            advice_html = last_advice.replace('\n', '<br>')
            html += f"<span style='color:#00fa9a'><b>[COPILOT]:</b><br>{advice_html}</span><br><br>"
            
        ui_overlay.update_text(html)

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
    import time
    interviewer_start_time = None

    while True:
        try:
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
                    
                    # Generate Answer
                    answer = await llm_client.generate_answer(transcript_history, context)
                    answer_clean = answer.strip() if answer else ""
                    
                    if answer_clean and answer_clean != "SKIP" and not answer_clean.startswith("Error"):
                        logger.info(f"LLM Advice generated.")
                        # Log the full technical advice to the background file
                        session_logger.info(f"[COPILOT ADVICE]:\n{answer_clean}\n" + "="*50)
                        
                        # Parse out markdown bold since simple rich text QLabel handles <b> better
                        answer_clean = answer_clean.replace("**", "<b>").replace("**", "</b>")
                        last_advice = answer_clean
                    else:
                        logger.info(f"LLM skipped conversational filler.")
                        last_advice = "<i style='color:#888888'>(Skipped conversational filler)</i>"
                        
                    update_ui()
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in conversation processing: {e}")

def main():
    """
    Wires all modules together async via QEventLoop.
    """
    load_dotenv()
    
    app = create_app()
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    ui_overlay = UIOverlay()

    # Initialize RAG & LLM
    rag_manager = RAGManager("data/portfolio.md")
    rag_manager.ingest_portfolio()
    llm_client = LLMClient()
    
    # Initialize Queues & Core Engines
    audio_queue = asyncio.Queue()
    text_queue = asyncio.Queue()
    
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
