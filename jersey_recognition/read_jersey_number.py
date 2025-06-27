import cv2
import pytesseract
import os

def preprocess_for_ocr(image):
    """
    Preprocess jersey crop for better OCR recognition
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2))  # Enlarge for better OCR

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(resized)

    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def read_jersey_number(frame, bbox, frame_id=0, player_id=0, save_debug=False):
    """
    Extracts and reads jersey number using the full bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    # Clamp coordinates to image dimensions
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    jersey_crop = frame[y1:y2, x1:x2]

    if jersey_crop.size == 0:
        return ""

    if save_debug:
        os.makedirs("debug", exist_ok=True)
        debug_path = f"debug/jersey_crop_f{frame_id}_p{player_id}.jpg"
        cv2.imwrite(debug_path, jersey_crop)

    processed = preprocess_for_ocr(jersey_crop)

    # OCR config
    config = '--psm 6 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(processed, config=config)

    # Keep only digits
    jersey_number = ''.join(filter(str.isdigit, text.strip()))

    if jersey_number:
        os.makedirs("data", exist_ok=True)
        with open("data/jersey_numbers.csv", "a") as f:
            f.write(f"{frame_id},{player_id},{jersey_number}\n")

    return jersey_number
