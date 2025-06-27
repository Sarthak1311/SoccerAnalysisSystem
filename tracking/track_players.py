import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from jersey_recognition.read_jersey_number import read_jersey_number

# Load YOLOv5 model
model = YOLO("yolov5s.pt")

# Initialize DeepSort
tracker = DeepSort(max_age=30)

def track_players(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=0.4)
        detections = results[0].boxes.data  # shape: [N, 6] (x1, y1, x2, y2, conf, class)
        person_detections = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) == 0:  # person class
                person_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = tracker.update_tracks(person_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            jersey_number = read_jersey_number(frame, (x1, y1, x2, y2), frame_id, track_id, save_debug=True)


            label = f"ID:{track_id} #{jersey_number}" if jersey_number else f"ID:{track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
