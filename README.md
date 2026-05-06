# CV Detection — Voice-Assisted Computer Vision for the Visually Impaired

A real-time computer vision application controlled entirely by voice, designed to assist visually impaired users. Point a webcam at any scene, activate the voice assistant, speak a command, and the app captures the image, runs the selected analysis, and reads the results aloud.

---

## Features

| Mode | Voice Command | What it does |
|---|---|---|
| Object Detection | *"object detection"* | Detects up to 80 objects, labels each as Indoor or Outdoor |
| Color Detection | *"color detection"* | Finds the 5 dominant colors in the scene with percentages |
| Text Extraction (OCR) | *"text extraction"* | Reads all text visible in the image, in correct reading order |

All results are **printed to the terminal** and **spoken aloud** via Windows SAPI (no internet required for TTS).

---

## Project Structure

```
cv_detection/
├── main.py             # App entry point — camera loop, UI, voice flow state machine
├── detector.py         # YOLOv8 object detection with Indoor/Outdoor classification
├── color_detector.py   # Dominant color detection using K-Means clustering (OpenCV)
├── ocr.py              # OCR via EasyOCR — extracts and sorts text in reading order
├── speech.py           # Thread-safe TTS engine using Windows SAPI (win32com)
├── voice_input.py      # Microphone listener and voice command parser (SpeechRecognition)
├── requirements.txt    # Python dependencies
├── setup.bat           # One-click dependency installer
├── .gitignore
└── README.md
```

### File Details

**`main.py`**
Entry point. Manages the OpenCV camera window, renders the on-screen microphone button, handles mouse clicks and keyboard shortcuts, and runs the voice assistant flow in a background thread. State machine: `idle → listening → processing → result → idle`.

**`detector.py`**
Loads YOLOv8n (COCO, 80 classes) via Ultralytics. Each detected object is classified as Indoor or Outdoor using two hard-coded sets. Bounding boxes are drawn on the frame with colour coding — green for indoor, blue-orange for outdoor. Results are spoken via `speech.py`.

**`color_detector.py`**
Converts the frame to RGB, runs OpenCV K-Means clustering (`k=5`) on a random sample of pixels, maps each cluster centre to a named colour via HSV range lookup, and renders a colour swatch panel below the image. Results are spoken.

**`ocr.py`**
Flips the captured frame to correct orientation, runs EasyOCR (English), sorts detected words into reading order (top-to-bottom, left-to-right by bounding box), draws polygon outlines around each word, and renders extracted text in a panel below the image. Full text is spoken.

**`speech.py`**
Thread-safe TTS wrapper around Windows SAPI (`win32com` / `SAPI.SpVoice`). A single persistent worker thread owns the COM object (initialised with `pythoncom.CoInitialize()`). All `speak()` calls push text onto a `queue.Queue`; the worker processes them sequentially. Supports both non-blocking (`speak()`) and blocking (`speak_sync()`) calls, and exposes `is_speaking()` so the voice flow can wait before resetting state.

**`voice_input.py`**
Wraps `SpeechRecognition` with Google's free API. Records from the default microphone, adjusts for ambient noise, and returns transcribed text. `parse_command()` maps free-form speech to one of three modes by keyword matching.

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd cv_detection
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Note — `pyaudio` on Windows:** If `pip install pyaudio` fails, use:
> ```powershell
> pip install pipwin
> pipwin install pyaudio
> ```

### 4. Run

```powershell
python main.py
```

YOLOv8n weights (`yolov8n.pt`, ~6 MB) and EasyOCR models (~100 MB) are downloaded automatically on first run.

---

## How to Use

1. The app opens a camera window with a **MIC** button at the bottom centre.
2. Click the **MIC** button or press **`V`** to activate the voice assistant.
3. The button turns red and the app says *"Listening. Say object detection, color detection, or text extraction."*
4. Speak your command.
5. The app captures the current frame, runs the selected analysis, and **speaks the results**.
6. Once the result is fully read out, the MIC button resets to grey — ready for the next command.
7. Press **`Q`** or **`ESC`** to quit.

### Accepted Voice Commands

| You say | Mode triggered |
|---|---|
| "object detection", "detect", "object", "item", "indoor", "outdoor" | Object Detection |
| "color detection", "color", "colour" | Color Detection |
| "text extraction", "text", "OCR", "read", "extract", "word", "writing" | OCR |

---

## Object Detection — All 80 COCO Classes

YOLOv8n is trained on the [COCO dataset](https://cocodataset.org/) (80 classes). Every detected class is labelled as **Indoor**, **Outdoor**, or **Both** (person appears everywhere).

### Indoor (34 classes)

| Category | Classes |
|---|---|
| Furniture & rooms | chair, couch, bed, dining table, toilet, potted plant |
| Electronics | tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator |
| Kitchen & food | bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake |
| Personal items | tie, handbag, suitcase |
| Decor & misc | book, clock, vase, scissors, teddy bear, hair drier, toothbrush |
| Pets | cat, dog |

### Outdoor (28 classes)

| Category | Classes |
|---|---|
| Vehicles | bicycle, car, motorcycle, airplane, bus, train, truck, boat |
| Street infrastructure | traffic light, fire hydrant, stop sign, parking meter, bench |
| Wild & farm animals | bird, horse, sheep, cow, elephant, bear, zebra, giraffe |
| Sports & recreation | frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket |
| Outdoor accessories | backpack, umbrella |

### Both / Untagged (1 class)

| Class | Reason |
|---|---|
| person | Appears in all environments — not tagged to avoid misleading output |

---

## Color Detection — Named Colors

The K-Means algorithm finds the 5 most dominant colors. Each cluster is mapped to a name via HSV range lookup:

`Red · Orange · Yellow · Green · Cyan · Blue · Purple · Pink · White · Black · Gray · Brown`

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture, image processing, UI rendering |
| `ultralytics` | YOLOv8 object detection |
| `easyocr` | Optical character recognition |
| `pywin32` | Windows SAPI text-to-speech (`win32com`) |
| `SpeechRecognition` | Microphone input and speech-to-text |
| `pyaudio` | Audio backend for SpeechRecognition |
| `Pillow` | Image format support |
| `numpy` | Array operations |
| `scikit-learn` | (Transitive dependency) |

---

## Requirements

- Windows 10 / 11
- Python 3.10+
- Webcam
- Microphone
- Internet connection (for voice recognition via Google Speech API)
