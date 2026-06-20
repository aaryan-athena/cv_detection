import cv2
import numpy as np
import sys
import time
import threading

from detector import detect_objects
from color_detector import detect_colors
from ocr import extract_text
from speech import speak, speak_sync, is_speaking
from voice_input import listen_for_command, parse_command

WINDOW = "CV Detection App"
BTN_RADIUS = 50

# Shared state (written by voice thread, read by render loop)
app_state = "idle"       # idle | listening | processing | result
result_frame = None
current_frame = None
frame_lock = threading.Lock()
display_dims = [720, 1280]   # [h, w] — updated each frame for mouse hit-test

MODE_NAMES = {"1": "object detection", "2": "color detection", "3": "text extraction"}

STATE_BAR_COLOR = {
    "idle":       (50,  50,  50),
    "preparing":  (0,   80, 160),
    "listening":  (0,   0,  180),
    "processing": (0,  130, 220),
    "result":     (0,  130,   0),
}
STATE_LABEL = {
    "idle":       "Click MIC button (or press V) to activate voice assistant",
    "preparing":  "Preparing microphone  —  please wait...",
    "listening":  "LISTENING  —  say: object detection / color detection / text extraction",
    "processing": "PROCESSING  —  please wait...",
    "result":     "Result shown above  —  click MIC or press V to go again",
}
BTN_COLOR = {
    "idle":       (90,  90,  90),
    "preparing":  (0,   80, 160),
    "listening":  (0,   0,  200),
    "processing": (0,  140, 255),
    "result":     (0,  140,   0),
}
BTN_LABEL = {
    "idle":       "MIC",
    "preparing":  "...",
    "listening":  " ON",
    "processing": " . .",
    "result":     "MIC",
}


def _btn_center(w, h):
    return (w // 2, h - BTN_RADIUS - 10)


def _in_button(x, y, w, h):
    bx, by = _btn_center(w, h)
    return (x - bx) ** 2 + (y - by) ** 2 <= BTN_RADIUS ** 2


def _draw_ui(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Status bar at top
    bar_color = STATE_BAR_COLOR.get(app_state, (50, 50, 50))
    cv2.rectangle(out, (0, 0), (w, 42), bar_color, -1)
    cv2.putText(out, STATE_LABEL.get(app_state, ""), (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

    # Mic button
    bx, by = _btn_center(w, h)
    btn_col = BTN_COLOR.get(app_state, (90, 90, 90))
    cv2.circle(out, (bx, by), BTN_RADIUS, btn_col, -1)
    cv2.circle(out, (bx, by), BTN_RADIUS, (255, 255, 255), 2)
    lbl = BTN_LABEL.get(app_state, "MIC")
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(out, lbl, (bx - tw // 2, by + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return out


def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        h, w = display_dims
        if app_state in ("idle", "result") and _in_button(x, y, w, h):
            threading.Thread(target=_voice_flow, daemon=True).start()


def _voice_flow():
    global app_state, result_frame

    app_state = "preparing"
    speak_sync("Preparing microphone.")
    time.sleep(0.1)

    def _on_mic_ready():
        global app_state
        app_state = "listening"
        print("[VOICE] Mic ready — now listening.")

    command_text = listen_for_command(timeout=7, on_ready=_on_mic_ready)

    if not command_text:
        speak_sync("No command heard. Please try again.")
        app_state = "idle"
        return

    mode = parse_command(command_text)
    if not mode:
        speak_sync(
            f"I heard '{command_text}' but could not match a command. "
            "Please say object detection, color detection, or text extraction."
        )
        app_state = "idle"
        return

    speak_sync(f"Starting {MODE_NAMES[mode]}. Capturing image now.")
    app_state = "processing"

    with frame_lock:
        captured = current_frame.copy() if current_frame is not None else None

    if captured is None:
        speak_sync("Camera not ready. Please try again.")
        app_state = "idle"
        return

    print(f"\n[APP] Running {MODE_NAMES[mode]}...")
    if mode == "1":
        result_frame = detect_objects(captured)
    elif mode == "2":
        result_frame = detect_colors(captured)
    elif mode == "3":
        result_frame = extract_text(captured)

    app_state = "result"

    # Wait for detector's TTS to finish before returning to idle
    time.sleep(0.5)
    while is_speaking():
        time.sleep(0.2)

    app_state = "idle"


def run():
    global current_frame, display_dims

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 800)
    cv2.setMouseCallback(WINDOW, _mouse_callback)

    speak("Welcome. Click the microphone button or press V to start.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        with frame_lock:
            current_frame = frame.copy()

        # Choose what to render
        if app_state == "result" and result_frame is not None:
            base = result_frame.copy()
        else:
            base = frame.copy()

        display = _draw_ui(base)
        display_dims[0], display_dims[1] = display.shape[:2]

        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("v"), ord("V")) and app_state in ("idle", "result"):
            threading.Thread(target=_voice_flow, daemon=True).start()

    cap.release()
    cv2.destroyAllWindows()
    print("App closed.")


if __name__ == "__main__":
    run()
