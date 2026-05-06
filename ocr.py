import cv2
import numpy as np
from speech import speak

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        print("Loading EasyOCR model (first run downloads ~100 MB)...")
        _reader = easyocr.Reader(["en"], gpu=False)
        print("EasyOCR ready.")
    return _reader


def _sort_by_reading_order(results, line_threshold=15):
    """Sort OCR results top-to-bottom, left-to-right using bounding box top-left coords."""
    def top_left(r):
        bbox = r[0]
        return (min(pt[1] for pt in bbox), min(pt[0] for pt in bbox))

    sorted_results = sorted(results, key=top_left)

    lines = []
    for item in sorted_results:
        y = min(pt[1] for pt in item[0])
        placed = False
        for line in lines:
            line_y = min(pt[1] for pt in line[0][0])
            if abs(y - line_y) <= line_threshold:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    ordered = []
    for line in lines:
        line.sort(key=lambda r: min(pt[0] for pt in r[0]))
        ordered.extend(line)

    return ordered


def extract_text(image: np.ndarray) -> np.ndarray:
    reader = _get_reader()
    image = cv2.flip(image, 1)
    results = reader.readtext(image)

    results = [r for r in results if r[2] >= 0.3]
    results = _sort_by_reading_order(results)

    annotated = image.copy()
    h, w = annotated.shape[:2]

    extracted_lines = []
    for (bbox, text, conf) in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 100), thickness=2)
        cv2.putText(annotated, f"{text} ({conf:.0%})",
                    (pts[0][0], pts[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        extracted_lines.append(text)

    panel_lines = extracted_lines if extracted_lines else ["(no text detected)"]
    line_h = 22
    panel_h = max(80, (len(panel_lines) + 2) * line_h)
    panel = np.ones((panel_h, w, 3), dtype=np.uint8) * 30

    cv2.putText(panel, "Extracted Text:", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1)
    for idx, line in enumerate(panel_lines[:15]):
        cv2.putText(panel, line[:80], (10, 20 + (idx + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    if len(panel_lines) > 15:
        cv2.putText(panel, f"... ({len(panel_lines) - 15} more lines)", (10, panel_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    annotated = np.vstack([annotated, panel])

    full_text = " ".join(extracted_lines)
    print("\n--- OCR Results ---")
    print(full_text if full_text else "(no text detected)")
    print("---\n")

    speech_text = full_text if full_text else "No text detected in the image."
    speak(speech_text)

    return annotated
