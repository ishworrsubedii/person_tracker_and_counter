"""
Created By: ishwor subedi
Date: 2024-08-22
"""
import cv2
from ultralytics import YOLO
from datetime import datetime
import os


class PersonTracker:
    def __init__(self, model_path, result_dir='results/', tracker_config="bytetrack.yaml", conf=0.5, device='cuda:0',
                 iou=0.5, img_size=(720, 1080)):
        self.model = YOLO(model_path)
        self.result_dir = result_dir
        self.tracker_config = tracker_config
        self.conf = conf
        self.device = device
        self.iou = iou
        self.img_size = img_size

    def create_result_file(self):
        folder_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_file_path = os.path.join(self.result_dir, folder_name + ".txt")
        os.makedirs(self.result_dir, exist_ok=True)
        with open(result_file_path, 'w') as file:
            file.write(folder_name + "\n")
        return result_file_path

    def detect_and_track(self, source, show=True, logger=None):
        result_file = self.create_result_file()
        person_count = 0
        previous_person_count = 0

        results = self.model.track(
            source, show=show, stream=True, tracker=self.tracker_config, conf=self.conf,
            device=self.device, iou=self.iou, classes=[0, 1], stream_buffer=True, imgsz=self.img_size
        )

        save_path = "/home/ishwor/Desktop/people_count/results/image.png"

        for i, result in enumerate(results):
            # Convert result frame to NumPy array
            frame = result.plot()  # YOLOv8 result with boxes drawn

            # Save the frame as image.png (overwrites each time)
            cv2.imwrite(save_path, frame)

            boxes = result.boxes
            try:
                id_count = boxes.id.int().tolist()
                max_id = max(id_count)

                if max_id > person_count:
                    person_count = max_id

                if person_count != previous_person_count:
                    previous_person_count = person_count
                    with open(result_file, 'w') as filewrite:
                        filewrite.write(f"Person count: {person_count}\n")
                        if logger:
                            logger.info(f"Person count: {person_count}")
                        print("Person count=", person_count)

            except Exception:
                pass
