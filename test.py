import cv2
from jersey_recognition.read_jersey_number import read_jersey_number

# Load the image
image_path = "/Users/sarthaktyagi/Desktop/mlinterviewGuide/soccerAnalysisSystem/data/testimg.png"
frame = cv2.imread(image_path)

# Example bounding box: [x1, y1, x2, y2] (you need to set this manually for now)
# You can use any image viewer (like mac Preview or cv2 selectROI) to get coordinates
bbox = [100, 50, 250, 300]  # <-- UPDATE THIS based on your image

# Call the OCR function
jersey_number = read_jersey_number(frame, bbox, frame_id=0, player_id=1, save_debug=True)

print(f"Detected Jersey Number: {jersey_number}")
