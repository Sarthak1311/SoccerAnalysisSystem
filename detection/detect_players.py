import cv2
from ultralytics import YOLO

# Load the model once globally
model = YOLO("yolov5s.pt")

def detect_players(video_path, conf=0.4):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run prediction
        results = model.predict(frame, conf=conf)

        # Get boxes and class IDs
        for box in results[0].boxes:
            cls_id = int(box.cls[0])  # class id
            class_name = model.names[cls_id]  # class label
            if class_name != "person":
                continue  # only show players

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Player Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
