
# Soccer Analysis System

A computer vision-based system that automatically detects and tracks soccer players in a match video, reads their jersey numbers using OCR, and stores analytics data for further tactical analysis.

---

## Features

- 👥 **Player Detection & Tracking** using YOLOv5 + Deep SORT
- 🔢 **Jersey Number Recognition** via Tesseract OCR
- 📊 **Real-Time Match Analysis** with unique player IDs
- 💾 **CSV Logging** of jersey numbers for each frame
- 🔍 **Debug Visualizations** for crop inspections and OCR tuning

---

##  How It Works

1. **Detect Players:** YOLOv5 detects players as `person` class.
2. **Track Them:** Deep SORT assigns a consistent ID to each player across frames.
3. **Read Jersey Numbers:** Tesseract OCR extracts digits from bounding boxes.
4. **Save Analysis:** Logs frame ID, player ID, and jersey number to a CSV.

---

## 🔧 Setup Instructions

### 1. Clone & Install Requirements

git clone https://github.com/your-username/soccerAnalysisSystem.git
cd soccerAnalysisSystem
pip install -r requirements.txt

### Potential Use Cases
1. Tactical player heatmaps based on movement
2. Player performance analytics via jersey number
3. Automated highlight generation
4. Real-time scoreboard overlay systems

## Future Improvements
1. Pose estimation for more accurate jersey region detection
2. Better OCR preprocessing tuned for soccer videos
3. Automatic team detection (home/away color)
4. Web dashboard for visualizing analytics

## Author
Sarthak Tyagi
📍 Gurgaon, India
ML Engineer | CV & NLP | GitHub