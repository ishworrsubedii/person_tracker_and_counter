"""
Created By: ishwor subedi
Date: 2024-08-22
Zone-based Person Tracker - Production Ready with Error Handling & Memory Management
- Ram monitoring, garbage collection
- Auto restart After 24 hr

"""
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import gc
import psutil
import time
import logging
import traceback

frame_save = False


class ZonePersonTracker:
    def __init__(self, model_path, zone_points, result_dir='results/', tracker_config="bytetrack.yaml",
                 conf=0.5, device='cuda:0', iou=0.5, img_size=(720, 1080),
                 memory_cleanup_interval=300, max_memory_usage=80):
        """
        Initialize tracker with error handling and memory management

        Args:
            memory_cleanup_interval: Seconds between memory cleanups (default: 300 = 5 minutes)
            max_memory_usage: Maximum RAM usage percentage before forced cleanup (default: 80%)
        """
        try:
            self.model = YOLO(model_path)
            self.zone_points = np.array(zone_points, dtype=np.int32)
            self.result_dir = result_dir
            self.tracker_config = tracker_config
            self.conf = conf
            self.device = device
            self.iou = iou
            self.img_size = img_size

            # Memory managementsettings
            self.memory_cleanup_interval = memory_cleanup_interval
            self.max_memory_usage = max_memory_usage
            self.last_cleanup_time = time.time()

            # Track people states - only for zone people
            self.people_in_zone = set()
            self.total_people_entered = set()
            self.entry_exit_log = []
            self.zone_id_mapping = {}
            self.next_zone_id = 1

            # Error tracking
            self.error_count = 0
            self.last_successful_frame = 0
            self.consecutive_errors = 0
            self.max_consecutive_errors = 10

            # Performance tracking
            self.frame_count = 0
            self.processing_times = []

            # Setup logging
            self.setup_logging()

            print("ZonePersonTracker initialized successfully")
            self.logger.info("Tracker initialized with memory management")

        except Exception as e:
            print(f"Error initializing tracker: {e}")
            raise

    def setup_logging(self):
        """Setup comprehensive logging"""
        try:
            os.makedirs(self.result_dir, exist_ok=True)
            log_file = os.path.join(self.result_dir, f"tracker_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("Logging system initialized")
        except Exception as e:
            print(f"Warning: Could not setup logging: {e}")
            self.logger = None

    def log_message(self, level, message):
        """Safe logging with fallback"""
        try:
            if self.logger:
                if level == 'info':
                    self.logger.info(message)
                elif level == 'error':
                    self.logger.error(message)
                elif level == 'warning':
                    self.logger.warning(message)
            print(f"[{level.upper()}] {message}")
        except:
            print(f"[{level.upper()}] {message}")

    def check_memory_usage(self):
        """Monitor and manage memory usage"""
        try:
            # Get current memory usage
            memory_percent = psutil.virtual_memory().percent

            # Log memory usage periodically
            if self.frame_count % 300 == 0:  # Every 300 frames
                self.log_message('info', f"Memory usage: {memory_percent:.1f}%")

            # Force cleanup if memory usage is too high
            if memory_percent > self.max_memory_usage:
                self.log_message('warning', f"High memory usage: {memory_percent:.1f}% - forcing cleanup")
                self.force_memory_cleanup()
                return True

            # Regular cleanup based on time interval
            current_time = time.time()
            if current_time - self.last_cleanup_time > self.memory_cleanup_interval:
                self.cleanup_memory()
                self.last_cleanup_time = current_time
                return True

            return False

        except Exception as e:
            self.log_message('error', f"Error checking memory usage: {e}")
            return False

    def cleanup_memory(self):
        """Regular memory cleanup"""
        try:
            # Limit processing times history
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-500:]

            # Limit entry/exit log size
            if len(self.entry_exit_log) > 10000:
                self.entry_exit_log = self.entry_exit_log[-5000:]

            # Clean up old zone ID mappings if too many
            if len(self.zone_id_mapping) > 1000:
                # Keep only recent mappings (this is aggressive, adjust as needed)
                recent_ids = list(self.zone_id_mapping.keys())[-500:]
                self.zone_id_mapping = {k: v for k, v in self.zone_id_mapping.items() if k in recent_ids}

            # Force garbage collection
            gc.collect()

            memory_after = psutil.virtual_memory().percent
            self.log_message('info', f"Memory cleanup completed. Usage: {memory_after:.1f}%")

        except Exception as e:
            self.log_message('error', f"Error during memory cleanup: {e}")

    def force_memory_cleanup(self):
        """Aggressive memory cleanup for high usage situations"""
        try:
            # More aggressive cleanup
            self.processing_times = self.processing_times[-100:] if self.processing_times else []
            self.entry_exit_log = self.entry_exit_log[-1000:] if self.entry_exit_log else []

            # Force multiple garbage collections
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)

            memory_after = psutil.virtual_memory().percent
            self.log_message('warning', f"Aggressive memory cleanup completed. Usage: {memory_after:.1f}%")

        except Exception as e:
            self.log_message('error', f"Error during aggressive memory cleanup: {e}")

    def create_result_file(self):
        try:
            folder_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            result_file_path = os.path.join(self.result_dir, folder_name + ".txt")
            os.makedirs(self.result_dir, exist_ok=True)
            with open(result_file_path, 'w') as file:
                file.write(f"Zone Person Tracker Results - {folder_name}\n")
                file.write("=" * 50 + "\n")
            return result_file_path
        except Exception as e:
            self.log_message('error', f"Error creating result file: {e}")
            return None

    def is_point_in_zone(self, point):
        """Check if a point is inside the defined zone with error handling"""
        try:
            return cv2.pointPolygonTest(self.zone_points, point, False) >= 0
        except Exception as e:
            self.log_message('error', f"Error in point zone test: {e}")
            return False

    def is_box_in_zone(self, box_coords):
        """Check if a bounding box is inside the zone with error handling"""
        try:
            if len(box_coords) < 4:
                return False
            center_x = (box_coords[0] + box_coords[2]) / 2
            center_y = (box_coords[1] + box_coords[3]) / 2
            return self.is_point_in_zone((center_x, center_y))
        except Exception as e:
            self.log_message('error', f"Error in box zone test: {e}")
            return False

    def draw_zone(self, frame):
        """Draw the detection zone with error handling"""
        try:
            if frame is None or frame.size == 0:
                return False

            cv2.polylines(frame, [self.zone_points], True, (255, 255, 255), 2)

            cv2.putText(frame, "DETECTION ZONE",
                        tuple(self.zone_points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw zone points
            for i, point in enumerate(self.zone_points):
                cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"P{i + 1}", tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return True
        except Exception as e:
            self.log_message('error', f"Error drawing zone: {e}")
            return False

    def get_zone_specific_id(self, yolo_id):
        """Map YOLO tracking ID to zone-specific ID with error handling"""
        try:
            if yolo_id not in self.zone_id_mapping:
                self.zone_id_mapping[yolo_id] = self.next_zone_id
                self.next_zone_id += 1
            return self.zone_id_mapping[yolo_id]
        except Exception as e:
            self.log_message('error', f"Error getting zone ID: {e}")
            return -1

    def safe_file_write(self, file_path, content, mode='a'):
        """Safe file writing with error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(file_path, mode) as f:
                    f.write(content)
                    f.flush()  # Ensure data is written
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    self.log_message('error', f"Failed to write to {file_path} after {max_retries} attempts: {e}")
                else:
                    time.sleep(0.1)  # pause before retry
        return False

    def update_person_tracking(self, current_zone_ids, result_file, total_count_file, logger=None):
        """Update person tracking with comprehensive error handling"""
        try:
            if not current_zone_ids:
                current_zone_ids = set()

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Find new entries and exits
            new_entries = current_zone_ids - self.people_in_zone
            exits = self.people_in_zone - current_zone_ids

            # Update tracking sets
            self.people_in_zone = current_zone_ids.copy()
            self.total_people_entered.update(new_entries)

            # Log changes if any
            if new_entries or exits:
                log_content = f"\n[{current_time}]\n"

                if new_entries:
                    entry_text = f"ENTERED ZONE: {sorted(list(new_entries))}"
                    log_content += f"{entry_text}\n"
                    self.entry_exit_log.append(f"{current_time} - {entry_text}")
                    print(f"✅ {entry_text}")

                if exits:
                    exit_text = f"EXITED ZONE: {sorted(list(exits))}"
                    log_content += f"{exit_text}\n"
                    self.entry_exit_log.append(f"{current_time} - {exit_text}")
                    print(f"❌ {exit_text}")

                # Current status
                status_text = f"Currently in zone: {len(self.people_in_zone)} people {sorted(list(self.people_in_zone))}"
                total_text = f"Total unique people entered: {len(self.total_people_entered)}"

                log_content += f"{status_text}\n{total_text}\n" + "-" * 50 + "\n"

                # Safe file writing
                if result_file:
                    self.safe_file_write(result_file, log_content)

                print(f"📊 {status_text}")
                print(f"📈 {total_text}")

            # Always update total count file with timestamp (every time there's a change)
            if (new_entries or exits) and total_count_file:
                total_content = f"[{current_time}] Total People Entered Zone: {len(self.total_people_entered)} | Currently in Zone: {len(self.people_in_zone)}\n"
                self.safe_file_write(total_count_file, total_content)

        except Exception as e:
            self.log_message('error', f"Error in person tracking update: {e}")

    def process_frame_safely(self, result, frame, person_colors, current_zone_ids):
        """Process a single frame with comprehensive error handling"""
        try:
            if result is None:
                return False, 0

            # Check if frame is valid
            if frame is None or frame.size == 0:
                self.log_message('warning', "Received empty frame, skipping")
                return False, 0

            # Draw zone
            if not self.draw_zone(frame):
                return False, 0

            people_outside_zone = 0

            # Process boxes if they exist
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    try:
                        # Get box coordinates safely
                        if box.xyxy is None or len(box.xyxy) == 0:
                            continue

                        box_coords = box.xyxy[0].cpu().numpy()

                        # Validate box coordinates
                        if len(box_coords) < 4 or any(np.isnan(box_coords)) or any(np.isinf(box_coords)):
                            continue

                        # Check if person is in zone
                        if self.is_box_in_zone(box_coords):
                            # Person is in zone - process normally
                            if box.id is not None:
                                yolo_id = int(box.id.item())
                                zone_id = self.get_zone_specific_id(yolo_id)

                                if zone_id > 0:  # Valid zone ID
                                    current_zone_ids.add(zone_id)

                                    # Assign unique color ONLY for zone people
                                    if zone_id not in person_colors:
                                        hue = (zone_id * 137.5) % 360
                                        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                                        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                                        person_colors[zone_id] = tuple(map(int, bgr[0][0]))

                                    color = person_colors[zone_id]

                                    # Draw bounding box with zone ID
                                    x1, y1, x2, y2 = map(int, box_coords)

                                    # Validate coordinates are within frame bounds
                                    h, w = frame.shape[:2]
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                                    # Draw zone-specific ID
                                    id_text = f"ZONE ID: {zone_id}"
                                    (text_width, text_height), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                                   0.7, 2)

                                    # Ensure text fits within frame
                                    text_y = max(text_height + 10, y1)
                                    cv2.rectangle(frame, (x1, text_y - text_height - 10),
                                                  (x1 + text_width + 10, text_y), color, -1)
                                    cv2.putText(frame, id_text, (x1 + 5, text_y - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            else:
                                # Person in zone but no YOLO ID
                                x1, y1, x2, y2 = map(int, box_coords)
                                h, w = frame.shape[:2]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                cv2.putText(frame, "IN ZONE - NO ID", (x1 + 5, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        else:
                            people_outside_zone += 1

                    except Exception as e:
                        self.log_message('error', f"Error processing individual box: {e}")
                        continue

            return True, people_outside_zone

        except Exception as e:
            self.log_message('error', f"Error processing frame: {e}")
            return False, 0

    def process_video_with_zone_display(self, source, show=True, logger=None, max_runtime_hours=24):
        """
        Process video with comprehensive error handling and memory management

        Args:
            max_runtime_hours: Maximum runtime in hours before automatic restart (default: 24)
        """
        start_time = time.time()
        max_runtime_seconds = max_runtime_hours * 3600

        self.log_message('info', f"Starting video processing from source: {source}")

        # Initialize files
        result_file = self.create_result_file()
        person_colors = {}

        # FPS calculation variables
        fps_start_time = cv2.getTickCount()
        fps_counter = 0
        fps = 0

        # Create log files
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        total_count_file = os.path.join(self.result_dir, f"total_count_with_timestamp_{timestamp}.txt")
        entry_exit_log_file = os.path.join(self.result_dir, f"zone_entry_exit_log_{timestamp}.txt")
        error_log_file = os.path.join(self.result_dir, f"error_log_{timestamp}.txt")

        # Initialize log files
        self.safe_file_write(total_count_file,
                             f"Zone Total Count Log - {timestamp}\nTracks total people entered zone with timestamps\n{'=' * 60}\n",
                             'w')
        self.safe_file_write(entry_exit_log_file, f"Zone Entry/Exit Log - {timestamp}\n{'=' * 35}\n", 'w')
        self.safe_file_write(error_log_file, f"Error Log - {timestamp}\n{'=' * 25}\n", 'w')

        results = None

        try:
            # Initialize YOLO tracking
            results = self.model.track(
                source, show=False, stream=True, tracker=self.tracker_config, conf=self.conf,
                device=self.device, iou=self.iou, classes=[0], stream_buffer=True, imgsz=self.img_size
            )

            self.log_message('info', "YOLO tracking initialized successfully")

        except Exception as e:
            self.log_message('error', f"Failed to initialize YOLO tracking: {e}")
            return False

        try:
            for i, result in enumerate(results):
                frame_start_time = time.time()

                # Check runtime limit
                if time.time() - start_time > max_runtime_seconds:
                    self.log_message('warning', f"Maximum runtime of {max_runtime_hours} hours reached. Stopping.")
                    break

                try:
                    self.frame_count += 1
                    current_zone_ids = set()

                    # Check for None result
                    if result is None:
                        self.log_message('warning', f"Received None result at frame {self.frame_count}")
                        self.consecutive_errors += 1
                        if self.consecutive_errors > self.max_consecutive_errors:
                            self.log_message('error', "Too many consecutive errors. Stopping.")
                            break
                        continue

                    # Get frame safely
                    try:
                        frame = result.orig_img.copy() if result.orig_img is not None else None
                    except Exception as e:
                        self.log_message('warning', f"Error copying frame: {e}")
                        frame = None

                    if frame is None or frame.size == 0:
                        self.log_message('warning', f"Empty or invalid frame at {self.frame_count}")
                        self.consecutive_errors += 1
                        if self.consecutive_errors > self.max_consecutive_errors:
                            self.log_message('error', "Too many consecutive empty frames. Stopping.")
                            break
                        continue

                    if frame_save:
                        try:
                            save_path = os.path.join(self.result_dir, "image.png")
                            cv2.imwrite(save_path, frame)
                            self.log_message('info', f"Frame saved to {save_path}")
                        except Exception as e:
                            self.log_message('error', f"Error saving frame: {e}")

                    # Reset consecutive error counter on successful frame
                    self.consecutive_errors = 0
                    self.last_successful_frame = self.frame_count

                    # Calculate FPS
                    fps_counter += 1
                    if fps_counter % 30 == 0:
                        fps_end_time = cv2.getTickCount()
                        time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                        if time_diff > 0:
                            fps = 30.0 / time_diff
                        fps_start_time = cv2.getTickCount()

                    # Process frame safely
                    success, people_outside_zone = self.process_frame_safely(result, frame, person_colors,
                                                                             current_zone_ids)

                    if not success:
                        self.error_count += 1
                        continue

                    # Update tracking
                    self.update_person_tracking(current_zone_ids, result_file, total_count_file, logger)

                    # Memory management
                    if self.check_memory_usage():
                        self.log_message('info', "Memory cleanup performed")

                    # Log current status every 120 frames (every 4 seconds at 30fps)
                    if fps_counter % 120 == 0:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        status_content = f"[{current_time}] RUNNING - Total Entered: {len(self.total_people_entered)}, Currently in Zone: {len(self.people_in_zone)}, Outside Zone: {people_outside_zone}, Errors: {self.error_count}\n"
                        self.safe_file_write(total_count_file, status_content)

                    # Display information on frame
                    try:
                        self.add_display_info(frame, fps, people_outside_zone)
                    except Exception as e:
                        self.log_message('warning', f"Error adding display info: {e}")

                    # Show frame if requested
                    if show:
                        try:
                            cv2.imshow('Zone-Only Person Tracker - Production', frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                self.log_message('info', "User requested quit")
                                break
                            elif key == ord('r'):  # Reset tracking
                                self.reset_tracking()
                                self.log_message('info', "Tracking reset by user")
                        except Exception as e:
                            self.log_message('warning', f"Error in display: {e}")

                    # Track processing time
                    processing_time = time.time() - frame_start_time
                    self.processing_times.append(processing_time)

                    # Log performance every 1000 frames
                    if self.frame_count % 1000 == 0:
                        avg_processing_time = sum(self.processing_times[-100:]) / min(100, len(self.processing_times))
                        self.log_message('info',
                                         f"Frame {self.frame_count}: Avg processing time: {avg_processing_time:.3f}s, FPS: {fps:.1f}, Errors: {self.error_count}")

                except Exception as e:
                    self.error_count += 1
                    self.consecutive_errors += 1
                    error_msg = f"Error processing frame {self.frame_count}: {str(e)}"
                    self.log_message('error', error_msg)
                    self.safe_file_write(error_log_file,
                                         f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")

                    # If too many consecutive errors, break
                    if self.consecutive_errors > self.max_consecutive_errors:
                        self.log_message('error',
                                         f"Too many consecutive errors ({self.consecutive_errors}). Stopping processing.")
                        break

                    continue

        except Exception as e:
            self.log_message('error', f"Critical error in video processing: {e}")
            self.log_message('error', f"Traceback: {traceback.format_exc()}")

        finally:
            # Cleanup and save final results
            self.cleanup_and_save_results(result_file, total_count_file, entry_exit_log_file, error_log_file,
                                          start_time)

            if show:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass

        return True

    def add_display_info(self, frame, fps, people_outside_zone):
        """Add display information to frame with error handling"""
        try:
            if frame is None or frame.size == 0:
                return

            info_y = 30

            # Current count in zone
            current_count_text = f"IN ZONE: {len(self.people_in_zone)}"
            (text_width, text_height), _ = cv2.getTextSize(current_count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(frame, (10, info_y - text_height - 10), (10 + text_width + 20, info_y + 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, info_y - text_height - 10), (10 + text_width + 20, info_y + 10), (0, 255, 0), 3)
            cv2.putText(frame, current_count_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Total entered zone
            info_y += 60
            total_count_text = f"TOTAL ENTERED: {len(self.total_people_entered)}"
            (text_width, text_height), _ = cv2.getTextSize(total_count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(frame, (10, info_y - text_height - 10), (10 + text_width + 20, info_y + 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, info_y - text_height - 10), (10 + text_width + 20, info_y + 10), (255, 0, 0), 3)
            cv2.putText(frame, total_count_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

            # Additional info
            info_y += 60
            outside_text = f"OUTSIDE: {people_outside_zone} | ERRORS: {self.error_count}"
            cv2.putText(frame, outside_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

            # FPS and frame count
            info_y += 40
            fps_text = f"FPS: {fps:.1f} | FRAME: {self.frame_count}"
            cv2.putText(frame, fps_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Memory usage
            info_y += 35
            try:
                memory_percent = psutil.virtual_memory().percent
                memory_text = f"RAM: {memory_percent:.1f}%"
                cv2.putText(frame, memory_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            except:
                pass

            # Current zone IDs (if any)
            if self.people_in_zone:
                info_y += 35
                ids_text = f"ZONE IDs: {sorted(list(self.people_in_zone))}"
                cv2.putText(frame, ids_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            self.log_message('error', f"Error adding display info: {e}")

    def cleanup_and_save_results(self, result_file, total_count_file, entry_exit_log_file, error_log_file, start_time):
        """Cleanup and save final results"""
        try:
            final_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            runtime_hours = (time.time() - start_time) / 3600

            # Save final summary
            if result_file:
                final_summary = f"""
{'=' * 60}
FINAL SUMMARY - ZONE-ONLY TRACKING [{final_time}]
{'=' * 60}
Total runtime: {runtime_hours:.2f} hours
Total frames processed: {self.frame_count}
Total errors encountered: {self.error_count}
Last successful frame: {self.last_successful_frame}
Total unique people who entered zone: {len(self.total_people_entered)}
People currently in zone: {len(self.people_in_zone)}
Current zone occupants (Zone IDs): {sorted(list(self.people_in_zone))}
Zone ID mappings: {dict(sorted(self.zone_id_mapping.items()))}
Average processing time per frame: {sum(self.processing_times[-1000:]) / min(1000, len(self.processing_times)):.4f}s
"""
                self.safe_file_write(result_file, final_summary)

            # Final entry in total count file
            if total_count_file:
                final_content = f"""
[{final_time}] FINAL SESSION SUMMARY
Runtime: {runtime_hours:.2f} hours | Frames: {self.frame_count} | Errors: {self.error_count}
Total People Entered Zone: {len(self.total_people_entered)}
Currently in Zone: {len(self.people_in_zone)}
{'=' * 60}
"""
                self.safe_file_write(total_count_file, final_content)

            # Save entry/exit log
            if entry_exit_log_file and self.entry_exit_log:
                log_content = "\n".join(self.entry_exit_log) + f"\n\n[{final_time}] Session ended\n"
                self.safe_file_write(entry_exit_log_file, log_content)

            # Final cleanup
            self.force_memory_cleanup()

            self.log_message('info',
                             f"Session completed. Runtime: {runtime_hours:.2f}h, Frames: {self.frame_count}, Errors: {self.error_count}")

        except Exception as e:
            self.log_message('error', f"Error during cleanup: {e}")

    def detect_and_track_in_zone(self, source, show=True, logger=None):
        """Simplified method - calls the main processing method"""
        return self.process_video_with_zone_display(source, show, logger)

    def get_zone_area(self):
        """Calculate the area of the detection zone"""
        try:
            return cv2.contourArea(self.zone_points)
        except Exception as e:
            self.log_message('error', f"Error calculating zone area: {e}")
            return 0

    def is_zone_valid(self):
        """Check if the zone points form a valid polygon"""
        try:
            return len(self.zone_points) >= 3 and self.get_zone_area() > 0
        except Exception as e:
            self.log_message('error', f"Error validating zone: {e}")
            return False

    def get_tracking_stats(self):
        """Get current tracking statistics for zone-only tracking"""
        try:
            return {
                'current_in_zone': len(self.people_in_zone),
                'total_entered': len(self.total_people_entered),
                'current_zone_ids': sorted(list(self.people_in_zone)),
                'all_entered_zone_ids': sorted(list(self.total_people_entered)),
                'zone_id_mappings': dict(sorted(self.zone_id_mapping.items())),
                'frame_count': self.frame_count,
                'error_count': self.error_count,
                'memory_usage': psutil.virtual_memory().percent if psutil else 'N/A'
            }
        except Exception as e:
            self.log_message('error', f"Error getting stats: {e}")
            return {}

    def reset_tracking(self):
        """Reset all tracking data - useful for processing new videos"""
        try:
            self.people_in_zone.clear()
            self.total_people_entered.clear()
            self.entry_exit_log.clear()
            self.zone_id_mapping.clear()
            self.next_zone_id = 1
            self.error_count = 0
            self.frame_count = 0
            self.consecutive_errors = 0
            self.processing_times.clear()

            # Force cleanup
            self.force_memory_cleanup()

            self.log_message('info', "Tracking data reset - ready for new video")
            print("✅ Tracking data reset - ready for new video")
        except Exception as e:
            self.log_message('error', f"Error resetting tracking: {e}")

    def get_system_info(self):
        """Get system information for monitoring"""
        try:
            info = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024 ** 3),
                'disk_usage_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage(
                    'C:').percent,
                'frame_count': self.frame_count,
                'error_count': self.error_count,
                'people_in_zone': len(self.people_in_zone),
                'total_entered': len(self.total_people_entered)
            }
            return info
        except Exception as e:
            self.log_message('error', f"Error getting system info: {e}")
            return {}

    def emergency_stop(self):
        """Emergency stop with cleanup"""
        try:
            self.log_message('warning', "Emergency stop initiated")
            self.cleanup_memory()
            cv2.destroyAllWindows()
            self.log_message('info', "Emergency stop completed")
        except Exception as e:
            self.log_message('error', f"Error during emergency stop: {e}")
