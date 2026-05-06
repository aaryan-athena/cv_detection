import cv2
import numpy as np
from speech import speak


COLOR_RANGES_HSV = [
    ("Red",     (0, 70, 50),   (10, 255, 255),  (0, 0, 220)),
    ("Red",     (170, 70, 50), (180, 255, 255),  (0, 0, 220)),
    ("Orange",  (11, 70, 50),  (25, 255, 255),   (0, 100, 255)),
    ("Yellow",  (26, 70, 50),  (34, 255, 255),   (0, 210, 255)),
    ("Green",   (35, 40, 40),  (85, 255, 255),   (0, 180, 0)),
    ("Cyan",    (86, 40, 40),  (100, 255, 255),  (200, 200, 0)),
    ("Blue",    (101, 40, 40), (130, 255, 255),  (220, 80, 0)),
    ("Purple",  (131, 40, 40), (160, 255, 255),  (180, 0, 180)),
    ("Pink",    (161, 40, 100),(169, 255, 255),  (180, 100, 220)),
    ("White",   (0, 0, 180),   (180, 40, 255),   (230, 230, 230)),
    ("Black",   (0, 0, 0),     (180, 255, 50),   (30, 30, 30)),
    ("Gray",    (0, 0, 51),    (180, 40, 179),   (128, 128, 128)),
    ("Brown",   (10, 60, 20),  (20, 200, 150),   (42, 75, 139)),
]


def _pixel_color_name(hsv_pixel):
    best = "Unknown"
    best_score = -1

    for entry in COLOR_RANGES_HSV:
        name = entry[0]
        lo = np.array(entry[1])
        hi = np.array(entry[2])
        if np.all(hsv_pixel >= lo) and np.all(hsv_pixel <= hi):
            return name

    return best


def _dominant_colors_kmeans(image_rgb, k=5):
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 5000:
        idx = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    counts = np.bincount(labels.flatten())
    order = np.argsort(-counts)
    return [(centers[i], counts[i] / len(labels)) for i in order]


def detect_colors(image: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    dominant = _dominant_colors_kmeans(image_rgb, k=5)

    annotated = image.copy()
    h, w = annotated.shape[:2]

    panel_h = 120
    panel = np.ones((panel_h, w, 3), dtype=np.uint8) * 40

    swatch_w = w // len(dominant)
    for i, (center_rgb, pct) in enumerate(dominant):
        bgr = (int(center_rgb[2]), int(center_rgb[1]), int(center_rgb[0]))
        hsv_px = cv2.cvtColor(np.uint8([[center_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        color_name = _pixel_color_name(hsv_px)

        x0 = i * swatch_w
        x1 = x0 + swatch_w
        cv2.rectangle(panel, (x0, 0), (x1, 60), bgr, -1)

        label = f"{color_name}"
        pct_label = f"{pct:.0%}"
        cv2.putText(panel, label, (x0 + 4, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, pct_label, (x0 + 4, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    annotated = np.vstack([annotated, panel])

    cv2.putText(annotated, "Dominant Colors (top 5)", (10, h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1)

    print("\n--- Color Detection Results ---")
    color_parts = []
    seen = []
    for center_rgb, pct in dominant:
        hsv_px = cv2.cvtColor(np.uint8([[center_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        color_name = _pixel_color_name(hsv_px)
        print(f"  {color_name}: {pct:.0%}")
        if color_name not in seen:
            seen.append(color_name)
            color_parts.append(f"{color_name} at {pct:.0%}")
    print("---\n")

    speech_text = "Dominant colors detected are: " + ", ".join(color_parts)
    speak(speech_text)

    return annotated
