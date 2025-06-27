import logging.config
import os 
from  pathlib import Path
import logging

logging.basicConfig(filename="project_setup_logs",
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

files = [
    "data/data1.txt",
    "detection/detect_players.py",
    "tracking/track_players.py",
    "jersey_recognition/read_jersey_number.py",
    "action/detect_action.py",
    "utils/video_utils.py",
    "requirement.txt",
    "README.md"
]

for file_path in files:
    file_path = Path(file_path)
    filedir,filename = os.path.split(file_path)

    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"creating directory {filedir} for file: {filename} ")

    if (not os.path.exists(file_path) or os.path.getsize(file_path)==0):
        with open(file_path,'w') as file:
            logging.info(f"creating the file :{filename}")
            pass

    else:
        logging.info(f"the file :{filename} already exists")
