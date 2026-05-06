import cv2
import numpy as np
from ultralytics import YOLO
from speech import speak

model = None

INDOOR_OBJECTS = {
    # Furniture & rooms
    "chair", "couch", "bed", "dining table", "toilet", "potted plant",
    # Electronics
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    # Kitchen & food
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    # Personal items
    "tie", "handbag", "suitcase",
    # Decor & misc
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    # Pets (commonly kept indoors)
    "cat", "dog",
}

OUTDOOR_OBJECTS = {
    # Vehicles
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    # Street infrastructure
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    # Animals (wild / farm)
    "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # Sports & recreation
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    # Accessories used outdoors
    "backpack", "umbrella",
}


def load_model():
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO("yolov8n.pt")
        print("Model loaded.")
    return model


def classify_environment(label):
    label_lower = label.lower()
    if label_lower in INDOOR_OBJECTS:
        return "Indoor"
    elif label_lower in OUTDOOR_OBJECTS:
        return "Outdoor"
    return "Unknown"


def detect_objects(image: np.ndarray) -> np.ndarray:
    m = load_model()
    results = m(image, verbose=False)[0]

    annotated = image.copy()
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = m.names[cls_id]
        env = classify_environment(label)

        if conf < 0.4:
            continue

        detections.append((label, conf, env))

        color = (0, 200, 0) if env == "Indoor" else (200, 100, 0) if env == "Outdoor" else (150, 150, 150)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} ({env}) {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if not detections:
        cv2.putText(annotated, "No objects detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print("No objects detected.")
        speak("No objects detected.")
    else:
        summary = f"Detected: {len(detections)} object(s)"
        cv2.putText(annotated, summary, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        print(f"\n--- Object Detection Results ({len(detections)} object(s)) ---")
        parts = []
        for label, conf, env in detections:
            line = f"  {label} — {env} object, confidence {conf:.0%}"
            print(line)
            parts.append(f"{label}, {env}")
        print("---\n")

        speech_text = f"Detected {len(detections)} object{'s' if len(detections) > 1 else ''}. " + ". ".join(parts)
        speak(speech_text)

    return annotated
