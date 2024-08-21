"""
Created By: ishwor subedi
Date: 2024-08-22
"""

import yaml
from src.ob_detection.person_tracker import PersonTracker
from src.utils.logger import setup_logger


def run_pipeline(params_path='params.yaml'):
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    logger = setup_logger('PersonTrackerLogger', 'tracker.log')

    tracker = PersonTracker(
        model_path=params['model_path'],
        result_dir=params['result_dir'],
        tracker_config=params['tracker_config'],
        conf=params['conf'],
        device=params['device'],
        iou=params['iou'],
        img_size=params['img_size']
    )

    video_source = params['video_source']
    tracker.detect_and_track(video_source, show=params['show'], logger=logger)
