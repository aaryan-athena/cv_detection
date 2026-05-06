import threading
import queue
import pythoncom
import win32com.client

_tts_queue = queue.Queue()
_count_lock = threading.Lock()
_active_count = 0


def _tts_worker():
    """Single persistent thread that owns the SAPI COM object."""
    global _active_count
    pythoncom.CoInitialize()          # required: initialise COM for this thread
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Rate = 1                  # -10 (slow) to 10 (fast)
    speaker.Volume = 100

    while True:
        text, done_event = _tts_queue.get()
        try:
            speaker.Speak(text)
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            with _count_lock:
                _active_count -= 1
            if done_event:
                done_event.set()
            _tts_queue.task_done()


_worker = threading.Thread(target=_tts_worker, daemon=True)
_worker.start()


def speak(text: str, blocking: bool = False):
    global _active_count
    print(f"\n[SPEECH] {text}\n")
    with _count_lock:
        _active_count += 1
    done_event = threading.Event() if blocking else None
    _tts_queue.put((text, done_event))
    if blocking and done_event:
        done_event.wait()


def speak_sync(text: str):
    speak(text, blocking=True)


def is_speaking() -> bool:
    with _count_lock:
        return _active_count > 0
