import yaml
from src.ob_detection.zone_person_tracker import ZonePersonTracker


def load_config():
    """Load configuration from params.yaml"""
    with open('params.yaml', 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    try:
        config = load_config()

        tracker = ZonePersonTracker(
            model_path=config['model_path'],  # Your YOLO model path
            zone_points=config['zone_points'],
            conf=config['conf'],
            device=config['device'],
            memory_cleanup_interval=config['memory_cleanup_interval'],
            max_memory_usage=config['mx_memory_usage'],
            iou=config['iou'],
            img_size=tuple(config['img_size']),
            result_dir=config['result_dir'],
            tracker_config=config['zone_tracker_config'],

        )

        if not tracker.is_zone_valid():
            print("❌ Invalid zone configuration!")
            exit(1)

        print("✅ Starting production tracking...")
        print("Press 'q' to quit, 'r' to reset tracking")
        print("Zone area:", tracker.get_zone_area())

        # Process video with error handling - runs for max 24 hours
        success = tracker.process_video_with_zone_display(
            source=config['video_source'],  # Webcam or "rtsp://your-cctv-url"
            show=True,
            max_runtime_hours=24,  # Auto-restart after 24 hours
        )

        if success:
            # Get final statistics
            stats = tracker.get_tracking_stats()
            print("\n✅ Final Statistics:")
            print(f"People currently in zone: {stats.get('current_in_zone', 0)}")
            print(f"Total unique people entered: {stats.get('total_entered', 0)}")
            print(f"Total frames processed: {stats.get('frame_count', 0)}")
            print(f"Total errors: {stats.get('error_count', 0)}")
            print(f"Memory usage: {stats.get('memory_usage', 'N/A')}")
        else:
            print("❌ Processing failed")

    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received")
        if 'tracker' in locals():
            tracker.emergency_stop()
    except Exception as e:
        print(f"❌ Critical error: {e}")
        if 'tracker' in locals():
            tracker.emergency_stop()
