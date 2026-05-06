import speech_recognition as sr

_recognizer = None


def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = sr.Recognizer()
        _recognizer.energy_threshold = 300
        _recognizer.dynamic_energy_threshold = True
        _recognizer.pause_threshold = 0.8
    return _recognizer


def listen_for_command(timeout: int = 7) -> str | None:
    r = _get_recognizer()
    try:
        with sr.Microphone() as source:
            print("[VOICE] Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            print("[VOICE] Listening...")
            audio = r.listen(source, timeout=timeout, phrase_time_limit=5)

        text = r.recognize_google(audio).lower()
        print(f"[VOICE] Heard: '{text}'")
        return text

    except sr.WaitTimeoutError:
        print("[VOICE] Timeout — no speech detected.")
        return None
    except sr.UnknownValueError:
        print("[VOICE] Could not understand speech.")
        return None
    except sr.RequestError as e:
        print(f"[VOICE] Recognition service error: {e}")
        return None
    except Exception as e:
        print(f"[VOICE] Unexpected error: {e}")
        return None


def parse_command(text: str) -> str | None:
    """Return '1' (object), '2' (color), '3' (OCR), or None if unrecognised."""
    if not text:
        return None
    t = text.lower()

    # Check color first to avoid "color detection" matching the generic "detection" branch
    if any(w in t for w in ["color", "colour"]):
        return "2"
    if any(w in t for w in ["text", "ocr", "read", "extract", "word", "letter", "writing"]):
        return "3"
    if any(w in t for w in ["object", "detect", "item", "thing", "indoor", "outdoor"]):
        return "1"

    return None
