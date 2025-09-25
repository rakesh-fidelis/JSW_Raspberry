# import os
# import time
# import cv2
# import numpy as np
# import threading
# import queue
# import logging
# import torch
# import gc
# import base64
# import json
# import traceback
# import datetime
# import random
# import sys
# import math
# import concurrent.futures
# from concurrent.futures import ThreadPoolExecutor
#
# from flask import Flask, Response, jsonify, render_template, request
# from ultralytics import YOLO
# from etracker import EnhancedBagTracker, LineCounter
# import multiprocessing as mp
# from collections import deque, defaultdict
# import psutil
# from dotenv import load_dotenv
#
#
# # Helper functions for line crossing detection
# def is_crossing_line(point, line):
#     """Check if a point is on the line side"""
#     x, y = point
#     (x1, y1), (x2, y2) = line
#
#     # Calculate line equation: ax + by + c = 0
#     a = y2 - y1
#     b = x1 - x2
#     c = x2 * y1 - x1 * y2
#
#     # Calculate the value of the line equation at the point
#     value = a * x + b * y + c
#
#     # Return the sign of the value (positive or negative)
#     return 1 if value > 0 else -1
#
#
# def calculate_direction(old_point, new_point, line_start, line_end):
#     """Calculate the direction of movement relative to the line"""
#     # Get the line vector
#     line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])
#
#     # Get the movement vector
#     movement_vector = (new_point[0] - old_point[0], new_point[1] - old_point[1])
#
#     # Calculate the cross product for direction
#     cross_product = line_vector[0] * movement_vector[1] - line_vector[1] * movement_vector[0]
#
#     # Determine the direction based on the cross product
#     return "IN" if cross_product > 0 else "OUT"
#
#
# def parse_multi_line_config():
#     """Parse multi-line configuration from environment variables"""
#     lines = []
#     directions = []
#     colors = []
#
#     # Parse multi-line coordinates
#     multi_line_coords = os.getenv("MULTI_LINE_COORDS")
#     if not multi_line_coords:
#         logger.error("MULTI_LINE_COORDS not found in environment file!")
#         return [], [], []
#
#     for line_str in multi_line_coords.split(";"):
#         try:
#             x1, y1, x2, y2 = map(int, line_str.split(","))
#             lines.append(((x1, y1), (x2, y2)))
#         except ValueError:
#             logger.warning(f"Invalid line coordinates: {line_str}")
#
#     # Parse directions
#     multi_directions = os.getenv("MULTI_COUNT_DIRECTIONS")
#     if not multi_directions:
#         logger.error("MULTI_COUNT_DIRECTIONS not found in environment file!")
#         return [], [], []
#
#     for direction in multi_directions.split(";"):
#         directions.append(direction.strip())
#
#     # Parse colors
#     multi_colors = os.getenv("LINE_COLORS")
#     if not multi_colors:
#         logger.error("LINE_COLORS not found in environment file!")
#         return [], [], []
#     for color_str in multi_colors.split(";"):
#         try:
#             b, g, r = map(int, color_str.split(","))
#             colors.append((b, g, r))
#         except ValueError:
#             logger.warning(f"Invalid color: {color_str}")
#             colors.append((0, 0, 255))  # Default to red
#
#     # Ensure all lists have the same length
#     min_length = min(len(lines), len(directions), len(colors))
#     lines = lines[:min_length]
#     directions = directions[:min_length]
#     colors = colors[:min_length]
#
#     return lines, directions, colors
#
#
# # Load environment variables
# env_file = '.env_cam4'
# print(f"Loading environment from: {env_file}")
# load_dotenv(dotenv_path=env_file)
#
# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
# logger = logging.getLogger(__name__)
#
# # Configuration from environment variables - Only cement bag model for optimization
# CEMENT_MODEL_PATH = os.getenv("CEMENT_MODEL_PATH")
# CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD"))
# RTSP_URL = os.getenv("RTSP_URL")
# CAMERA_ID = os.getenv("CAMERA_ID")
# FRAME_SKIP = int(os.getenv("FRAME_SKIP"))  # Process every frame for camera streams
# INFER_IMG_SIZE = int(os.getenv("INFER_IMG_SIZE", 320))  # Inference image size
# MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 1))  # Maximum batch size
#
# # Debug configuration
# DEBUG_DETECTIONS = os.getenv("DEBUG_DETECTIONS", "false").lower() == "true"
# if DEBUG_DETECTIONS:
#     logger.info(f"[{CAMERA_ID}] Detection debugging enabled - detailed logs will be shown")
#     logging.getLogger().setLevel(logging.DEBUG)
#
# # Parse multi-line configuration
# lines, line_directions, line_colors = parse_multi_line_config()
# logger.info(f"Loaded {len(lines)} counting lines with enhanced tracker")
#
# # Global variables for counting and tracking
# bag_in = 0
# bag_out = 0
# counting_active = True
# counting_in_active = True  # Control for IN direction counting
# counting_out_active = True  # Control for OUT direction counting
# tracks_history = defaultdict(list)
# track_movement_state = defaultdict(lambda: {'last_direction': None, 'counted': False})
#
# # Spatial-temporal duplicate prevention
# recent_counts = []  # List of (timestamp, position, direction) for recent counts
# COUNT_COOLDOWN_SECONDS = 3.0  # Prevent counting same area within 3 seconds
# SPATIAL_THRESHOLD = 100  # Minimum distance in pixels to consider different bags
#
# # Multithreading optimization variables
# frame_queue = queue.Queue(maxsize=3)  # Buffer for camera frames
# detection_queue = queue.Queue(maxsize=3)  # Buffer for detection results
# processing_lock = threading.Lock()
# frame_buffer = deque(maxlen=10)  # Rolling buffer for frame history
#
# # Thread pool for parallel processing
# max_workers = min(4, mp.cpu_count())  # Fewer workers for camera streams
# executor = ThreadPoolExecutor(max_workers=max_workers)
#
# # Determine device (GPU if available, else CPU)
# if torch.cuda.is_available():
#     device = 'cuda'
#     torch.cuda.empty_cache()
#     torch.backends.cudnn.benchmark = True
#     free_mem, total_mem = torch.cuda.mem_get_info()
#     logger.info(
#         f'CUDA is available. Using GPU for inference. Free memory: {free_mem / 1024 ** 2:.1f}MB / {total_mem / 1024 ** 2:.1f}MB')
# else:
#     device = 'cpu'
#     logger.info('CUDA is not available. Using CPU for inference.')
#
# # Load only cement bag model for optimization
# try:
#     # Check if model file exists
#     if not os.path.exists(CEMENT_MODEL_PATH):
#         logger.error(f"CEMENT_MODEL_PATH is missing or invalid: {CEMENT_MODEL_PATH}")
#         raise FileNotFoundError(f"CEMENT_MODEL_PATH is missing or invalid: {CEMENT_MODEL_PATH}")
#
#     # Load model with explicit device specification and task type
#     logger.info(f"Loading cement bag model: {CEMENT_MODEL_PATH}")
#     model_cement = YOLO(CEMENT_MODEL_PATH, task='detect')
#
#     # Force model to initialize on GPU to avoid "bn" errors
#     if torch.cuda.is_available():
#         logger.info("CUDA is available, initializing model on GPU")
#         try:
#             # Create a dummy input tensor for initialization
#             dummy_input = torch.zeros((1, 3, 640, 640)).cuda()
#             model_cement.model.cuda()
#             with torch.no_grad():
#                 # Run a dummy inference to initialize all layers including batch normalization
#                 _ = model_cement.model(dummy_input)
#             logger.info("Successfully initialized model on GPU")
#         except Exception as gpu_err:
#             logger.warning(f"GPU initialization warning (non-fatal): {str(gpu_err)}")
#         finally:
#             # Clean up GPU memory
#             torch.cuda.empty_cache()
#             gc.collect()
#
#     logger.info(f"Successfully loaded cement bag model: {CEMENT_MODEL_PATH}")
#
# except Exception as e:
#     logger.error(f"Error loading model: {e}")
#     traceback.print_exc()
#     sys.exit(1)
#
# # Initialize enhanced tracker with optimized parameters for cement bags
# tracker = EnhancedBagTracker(
#     iou_threshold=0.25,  # Lower for deformable bags
#     max_missing=100,  # Hold track ID for 40 frames without detection
#     min_hits=2,  # Confirm tracks faster
#     max_age=50,  # Keep tracks alive longer for better persistence
#     motion_threshold=60  # Allow more movement for bags
# )
#
# # Initialize line counter with counting lines
# line_counter = LineCounter(lines)
#
# # Track history for line counting
# tracks_history = {}
# max_history_length = 50  # Limit history to prevent memory growth
#
# # Track movement direction for each object to prevent duplicate counting
# track_movement_state = {}  # Format: {track_id: {'last_direction': 'IN'/'OUT', 'counted': True/False}}
#
# # Initialize Flask app
# app = Flask(__name__)
#
#
# def create_test_frame():
#     """Create a test frame to show when no camera is connected"""
#     frame = np.zeros((480, 640, 3), dtype=np.uint8)
#     cv2.putText(frame, f"Camera {CAMERA_ID} - Connecting...", (50, 240),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     cv2.putText(frame, "Enhanced Tracker Ready", (150, 280),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     ret, jpeg = cv2.imencode('.jpg', frame)
#     return jpeg.tobytes() if ret else None
#
#
# # Initialize latest_frame with test frame
# latest_frame = create_test_frame()
#
#
# # Optimized detection function for camera streams
# def detect_objects_optimized(frame, frame_number):
#     """Optimized detection function for parallel processing"""
#     try:
#         # Validate frame before detection
#         if frame is None or not isinstance(frame, np.ndarray):
#             logger.error(f"[{CAMERA_ID}] Invalid frame type for detection: {type(frame)}, frame #{frame_number}")
#             return [], frame_number
#
#         if frame.size == 0 or frame.ndim != 3:
#             logger.error(f"[{CAMERA_ID}] Invalid frame dimensions for detection: {frame.shape}, frame #{frame_number}")
#             return [], frame_number
#
#         # Ensure frame is in the correct format for YOLO
#         if frame.dtype != np.uint8:
#             logger.warning(f"[{CAMERA_ID}] Converting frame from {frame.dtype} to uint8, frame #{frame_number}")
#             frame = frame.astype(np.uint8)
#
#         # Run YOLO detection with proper error handling
#         try:
#             # Use a timeout to prevent hanging on problematic frames
#             start_time = time.time()
#
#             # Wrap the model call in a try-except block specifically for "bn" errors
#             try:
#                 # Make sure the model is on the correct device
#                 if torch.cuda.is_available():
#                     model_cement.model.cuda()
#
#                 # Run detection with verbose=False to reduce console output
#                 results_cement = model_cement(frame, conf=CONF_THRESHOLD, verbose=False)
#
#             except RuntimeError as bn_err:
#                 # Handle batch normalization errors by retrying once
#                 if "bn" in str(bn_err).lower():
#                     logger.warning(
#                         f"[{CAMERA_ID}] Batch normalization error detected, retrying with CPU: {str(bn_err)}")
#                     # Try to run on CPU as a fallback
#                     model_cement.model.cpu()
#                     torch.cuda.empty_cache()
#                     results_cement = model_cement(frame, conf=CONF_THRESHOLD, verbose=False)
#                     # Move back to GPU for next frame
#                     if torch.cuda.is_available():
#                         model_cement.model.cuda()
#                 else:
#                     # Re-raise if it's not a bn error
#                     raise
#
#             detection_time = time.time() - start_time
#
#             if detection_time > 1.0:  # Log slow detections
#                 logger.warning(f"[{CAMERA_ID}] Slow detection: {detection_time:.2f}s for frame #{frame_number}")
#
#         except torch.cuda.OutOfMemoryError:
#             # Handle CUDA OOM errors
#             logger.error(f"[{CAMERA_ID}] CUDA out of memory on frame #{frame_number}, clearing cache")
#             torch.cuda.empty_cache()
#             gc.collect()
#             return [], frame_number
#
#         except Exception as yolo_err:
#             logger.error(f"[{CAMERA_ID}] YOLO detection error on frame #{frame_number}: {str(yolo_err)}")
#             return [], frame_number
#
#         # Process detections with detailed error handling
#         detections = []
#         try:
#             if results_cement and len(results_cement) > 0:
#                 for result in results_cement:
#                     if result.boxes is not None and len(result.boxes) > 0:
#                         for box in result.boxes:
#                             try:
#                                 # Safely extract box coordinates
#                                 if box.xyxy.numel() == 0 or box.conf.numel() == 0 or box.cls.numel() == 0:
#                                     continue  # Skip empty boxes
#
#                                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                                 conf = float(box.conf[0].cpu().numpy())
#                                 class_id = int(box.cls[0].cpu().numpy())
#
#                                 # Calculate dimensions and aspect ratio for filtering
#                                 width = int(x2 - x1)
#                                 height = int(y2 - y1)
#                                 aspect_ratio = width / height if height > 0 else 0
#
#                                 # Enhanced filtering to reduce false positives
#                                 if (class_id == 0 and
#                                         conf >= CONF_THRESHOLD and
#                                         width >= 30 and height >= 30 and  # Minimum size
#                                         aspect_ratio >= 0.5 and aspect_ratio <= 2.0):  # Reasonable aspect ratio for bags
#
#                                     # Add to valid detections
#                                     detections.append({
#                                         'bbox': [int(x1), int(y1), int(x2), int(y2)],
#                                         'confidence': conf,
#                                         'class_id': class_id,
#                                         'width': width,
#                                         'height': height,
#                                         'aspect_ratio': aspect_ratio,
#                                         'frame_number': frame_number
#                                     })
#
#                                     if DEBUG_DETECTIONS:
#                                         logger.debug(
#                                             f"[{CAMERA_ID}] Valid detection: class={class_id}, conf={conf:.2f}, "
#                                             f"size={width}x{height}, ratio={aspect_ratio:.2f}")
#                                 elif DEBUG_DETECTIONS and class_id == 0 and conf >= 0.2:
#                                     # Log rejected detections for debugging
#                                     logger.debug(
#                                         f"[{CAMERA_ID}] Rejected detection: class={class_id}, conf={conf:.2f}, "
#                                         f"size={width}x{height}, ratio={aspect_ratio:.2f}")
#                             except Exception as box_err:
#                                 logger.warning(f"[{CAMERA_ID}] Error processing detection box: {str(box_err)}")
#                                 continue
#         except Exception as proc_err:
#             logger.error(f"[{CAMERA_ID}] Error processing detection results: {str(proc_err)}")
#
#         # Log detection summary
#         if len(detections) > 0:
#             logger.debug(f"[{CAMERA_ID}] Found {len(detections)} valid detections in frame #{frame_number}")
#
#         return detections, frame_number
#
#     except Exception as e:
#         logger.error(f"[{CAMERA_ID}] Detection error on frame {frame_number}: {str(e)}")
#         return [], frame_number
#
#
# # Threaded Video Capture class for RTSP streams
# class VideoCaptureThread:
#     def __init__(self, src, width=None, height=None):
#         self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
#         # Set buffer size to 1 to reduce latency
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         # Set FPS to reasonable value
#         self.cap.set(cv2.CAP_PROP_FPS, 15)
#         if width and height:
#             self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#             self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#         self.q = queue.Queue(maxsize=2)
#         self.running = True
#         self.thread = threading.Thread(target=self._reader)
#         self.thread.daemon = True
#         self.thread.start()
#
#     def _reader(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#             # Always drop old frames to prevent buffer buildup
#             while not self.q.empty():
#                 try:
#                     self.q.get_nowait()
#                 except queue.Empty:
#                     break
#             try:
#                 self.q.put(frame, timeout=0.1)
#             except queue.Full:
#                 # Skip frame if queue is full
#                 pass
#
#     def read(self):
#         try:
#             return True, self.q.get(timeout=1.0)
#         except queue.Empty:
#             return False, None
#
#     def release(self):
#         self.running = False
#         if self.thread.is_alive():
#             self.thread.join(timeout=2.0)
#         self.cap.release()
#
#
# def frame_capture_thread():
#     """Dedicated thread for frame capture to reduce latency"""
#     global stream_up
#     cap = None
#     reconnect_delay = 5  # Initial reconnect delay in seconds
#     max_reconnect_delay = 60  # Maximum reconnect delay
#     consecutive_failures = 0
#     max_consecutive_failures = 10  # Reset connection after this many failures
#
#     while True:  # Outer loop for reconnection
#         try:
#             logger.info(f"[{CAMERA_ID}] Starting frame capture thread")
#
#             # Create a modified RTSP URL with explicit transport protocol and timeout settings
#             # Format: rtsp://username:password@ip:port/path?rtsp_transport=tcp&timeout=30000000
#             rtsp_url = RTSP_URL
#             if '?' not in rtsp_url:
#                 rtsp_url += '?'
#             else:
#                 rtsp_url += '&'
#
#             # Add RTSP over TCP for more reliable streaming and explicit timeout
#             rtsp_url += 'rtsp_transport=tcp&timeout=15000000'
#             logger.info(f"[{CAMERA_ID}] Using modified RTSP URL with TCP transport")
#
#             # Set OpenCV environment variables for better H264 decoding
#             os.environ[
#                 'OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'protocol_whitelist;file,rtp,udp,tcp,tls,https,rtsp|fflags;nobuffer|max_delay;500000|reorder_queue_size;0|rtsp_transport;tcp|stimeout;5000000'
#
#             # Use standard OpenCV capture with FFMPEG backend for better stability
#             cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#
#             # Configure capture parameters for better RTSP stability
#             cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Small buffer for reduced latency but better stability
#             cap.set(cv2.CAP_PROP_FPS, 15)  # Request reasonable FPS
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#             # Verify connection
#             if not cap.isOpened():
#                 logger.error(f"[{CAMERA_ID}] Failed to open RTSP stream: {rtsp_url}")
#                 raise ConnectionError(f"Failed to open RTSP stream: {rtsp_url}")
#
#             logger.info(f"[{CAMERA_ID}] Stream connected successfully")
#             stream_up = True
#             frame_count = 0
#             consecutive_failures = 0  # Reset failure counter on successful connection
#             reconnect_delay = 5  # Reset reconnect delay on successful connection
#             last_successful_read = time.time()
#
#             # Inner loop for frame reading
#             while True:
#                 # Check for stream timeout (no successful reads for 30 seconds)
#                 current_time = time.time()
#                 if current_time - last_successful_read > 30:
#                     logger.warning(f"[{CAMERA_ID}] Stream timeout - no frames for 30 seconds")
#                     break
#
#                 ret, frame = cap.read()
#
#                 # Handle frame read failures
#                 if not ret or frame is None:
#                     consecutive_failures += 1
#                     logger.warning(f"[{CAMERA_ID}] Failed to read frame (failure #{consecutive_failures})")
#
#                     # Reset connection after too many consecutive failures
#                     if consecutive_failures >= max_consecutive_failures:
#                         logger.warning(f"[{CAMERA_ID}] Too many consecutive failures, resetting connection")
#                         break
#
#                     time.sleep(0.1)
#                     continue
#
#                 # Reset failure counter on successful frame read
#                 consecutive_failures = 0
#                 last_successful_read = time.time()
#
#                 # Validate frame integrity
#                 if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
#                     logger.warning(f"[{CAMERA_ID}] Invalid frame dimensions: {frame.shape}")
#                     time.sleep(0.1)
#                     continue
#
#                 # Add frame to queue for processing (non-blocking)
#                 try:
#                     frame_queue.put((frame.copy(), frame_count), block=False)
#                     frame_count += 1
#                 except queue.Full:
#                     # Skip frame if queue is full to maintain real-time performance
#                     pass
#
#         except Exception as e:
#             logger.error(f"[{CAMERA_ID}] Frame capture error: {str(e)}")
#             stream_up = False
#
#         finally:
#             # Clean up resources
#             if cap is not None:
#                 cap.release()
#             stream_up = False
#
#             # Implement exponential backoff for reconnection
#             logger.info(f"[{CAMERA_ID}] Reconnecting to stream in {reconnect_delay} seconds...")
#             time.sleep(reconnect_delay)
#             reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)  # Exponential backoff
#
#
# def detection_processing_thread():
#     """Dedicated thread for object detection processing"""
#     global camera_confidence
#
#     while True:
#         try:
#             # Get frame from queue with timeout
#             frame, frame_number = frame_queue.get(timeout=1.0)
#
#             # Validate frame before processing
#             if frame is None or frame.size == 0:
#                 logger.warning(f"[{CAMERA_ID}] Skipping empty frame #{frame_number}")
#                 continue
#
#             # Check for corrupted frame
#             try:
#                 height, width = frame.shape[:2]
#                 if height <= 0 or width <= 0 or frame.ndim != 3:
#                     logger.warning(f"[{CAMERA_ID}] Invalid frame dimensions: {frame.shape} for frame #{frame_number}")
#                     continue
#             except Exception as frame_err:
#                 logger.warning(f"[{CAMERA_ID}] Frame validation error: {str(frame_err)} for frame #{frame_number}")
#                 continue
#
#             # Submit detection task to thread pool with proper error handling
#             try:
#                 future = executor.submit(detect_objects_optimized, frame, frame_number)
#
#                 # Get detection results with timeout
#                 try:
#                     detections, _ = future.result(timeout=1.0)  # Increased timeout for more reliable processing
#
#                     # Add to detection queue for tracking (non-blocking)
#                     try:
#                         detection_queue.put((detections, frame, frame_number), block=False)
#                     except queue.Full:
#                         # Skip if queue is full to maintain real-time performance
#                         logger.debug(f"[{CAMERA_ID}] Detection queue full, skipping frame #{frame_number}")
#                         pass
#
#                 except concurrent.futures._base.TimeoutError:  # Use the correct TimeoutError class
#                     logger.warning(f"[{CAMERA_ID}] Detection timeout for frame #{frame_number}")
#                     future.cancel()  # Try to cancel the hanging task
#
#             except Exception as detect_err:
#                 logger.error(f"[{CAMERA_ID}] Detection submission error: {str(detect_err)} for frame #{frame_number}")
#
#         except queue.Empty:
#             # No frames available, just continue
#             continue
#
#         except Exception as e:
#             logger.error(f"[{CAMERA_ID}] Detection processing error: {str(e)}")
#             time.sleep(0.1)  # Short sleep to prevent CPU spinning
#
#
# def tracking_and_counting_thread():
#     """Dedicated thread for tracking and counting logic"""
#     global bag_in, bag_out, camera_confidence, latest_frame
#
#     fps_local = 0.0
#     fps_start_time = time.time()
#     fps_frame_count = 0
#     cleanup_counter = 0
#
#     while True:
#         try:
#             # Get detection results
#             detections, frame, frame_number = detection_queue.get(timeout=1.0)
#
#             start_time = time.time()
#
#             # Convert detections to format expected by enhanced tracker
#             tracker_detections = []
#             for det in detections:
#                 x1, y1, x2, y2 = det['bbox']
#                 w, h = x2 - x1, y2 - y1
#                 tracker_detections.append((x1, y1, w, h, det['confidence']))
#
#             # Update tracker with detections
#             confirmed_tracks = tracker.update(tracker_detections, frame_number)
#
#             # Update track history for line counting
#             for track_id, bbox in confirmed_tracks.items():
#                 if track_id not in tracks_history:
#                     tracks_history[track_id] = []
#
#                 # Get center point
#                 x, y, w, h = bbox
#                 center = (x + w // 2, y + h // 2)
#                 tracks_history[track_id].append(center)
#
#                 # Limit history length
#                 if len(tracks_history[track_id]) > max_history_length:
#                     tracks_history[track_id] = tracks_history[track_id][-max_history_length:]
#
#             # Direction-based counting logic for each track
#             with processing_lock:
#                 for track_id, track_history in tracks_history.items():
#                     if len(track_history) < 3:
#                         continue  # Need at least 3 points for stable direction
#
#                     # Initialize movement state for new tracks
#                     if track_id not in track_movement_state:
#                         track_movement_state[track_id] = {'last_direction': None, 'counted': False}
#
#                     # Get current movement direction by checking line crossings
#                     current_movement_direction = None
#
#                     # Check any line crossing to determine overall movement direction
#                     for i, (line_start, line_end) in enumerate(lines):
#                         if len(track_history) >= 2:
#                             current_pos = track_history[-1]
#                             previous_pos = track_history[-2]
#
#                             # Check if track crossed this line
#                             current_side = is_crossing_line(current_pos, (line_start, line_end))
#                             previous_side = is_crossing_line(previous_pos, (line_start, line_end))
#
#                             # Line crossing detected
#                             if current_side != previous_side:
#                                 # Calculate movement direction relative to line
#                                 movement_direction = calculate_direction(previous_pos, current_pos, line_start,
#                                                                          line_end)
#                                 line_direction = line_directions[i] if i < len(line_directions) else "left"
#
#                                 # Dynamic counting logic based on env configuration
#                                 # This will work with any direction configuration in .env file
#                                 current_movement_direction = movement_direction  # Use actual movement direction
#
#                                 # Log the movement for debugging
#                                 logger.debug(
#                                     f"[{CAMERA_ID}] Track {track_id} crossed line {i + 1}: {movement_direction} (line_direction: {line_direction})")
#
#                                 break  # Found a crossing, use this direction
#
#                     # Spatial-temporal duplicate prevention
#                     if current_movement_direction and counting_active:
#                         track_state = track_movement_state[track_id]
#                         current_time = time.time()
#
#                         # Get current position
#                         if track_id in confirmed_tracks:
#                             x, y, w, h = confirmed_tracks[track_id]
#                             current_pos = (x + w // 2, y + h // 2)
#
#                             # Check for spatial-temporal duplicates
#                             is_duplicate = False
#                             for count_time, count_pos, count_dir in recent_counts:
#                                 # Remove old counts (older than cooldown period)
#                                 if current_time - count_time > COUNT_COOLDOWN_SECONDS:
#                                     continue
#
#                                 # Check if same direction and close position
#                                 if (count_dir == current_movement_direction and
#                                         abs(current_pos[0] - count_pos[0]) < SPATIAL_THRESHOLD and
#                                         abs(current_pos[1] - count_pos[1]) < SPATIAL_THRESHOLD):
#                                     is_duplicate = True
#                                     logger.debug(
#                                         f"[{CAMERA_ID}] Duplicate detected: Track {track_id} too close to recent count")
#                                     break
#
#                             # Clean up old counts
#                             recent_counts[:] = [(t, p, d) for t, p, d in recent_counts
#                                                 if current_time - t <= COUNT_COOLDOWN_SECONDS]
#
#                             # Count only if not duplicate and direction changed
#                             if (not is_duplicate and
#                                     (track_state['last_direction'] != current_movement_direction or
#                                      track_state['last_direction'] is None)):
#
#                                 if current_movement_direction == "IN" and counting_in_active:
#                                     bag_in += 1
#                                     logger.info(f"[{CAMERA_ID}] Bag IN (Track {track_id})! Total IN: {bag_in}")
#                                 elif current_movement_direction == "OUT" and counting_out_active:
#                                     bag_out += 1
#                                     logger.info(f"[{CAMERA_ID}] Bag OUT (Track {track_id})! Total OUT: {bag_out}")
#                                 else:
#                                     # Log when counting is skipped due to direction control
#                                     if current_movement_direction == "IN" and not counting_in_active:
#                                         logger.debug(
#                                             f"[{CAMERA_ID}] Skipped IN count (Track {track_id}) - IN counting disabled")
#                                     elif current_movement_direction == "OUT" and not counting_out_active:
#                                         logger.debug(
#                                             f"[{CAMERA_ID}] Skipped OUT count (Track {track_id}) - OUT counting disabled")
#
#                                 # Record this count to prevent duplicates
#                                 recent_counts.append((current_time, current_pos, current_movement_direction))
#
#                                 # Update state
#                                 track_state['last_direction'] = current_movement_direction
#                                 track_state['counted'] = True
#
#             # Calculate max confidence
#             max_frame_confidence = 0.0
#             for detection in detections:
#                 conf = detection['confidence']
#                 if conf > max_frame_confidence:
#                     max_frame_confidence = conf
#
#             camera_confidence = max_frame_confidence
#
#             # Draw visualizations on frame
#             frame_vis = frame.copy()
#
#             # Draw detections
#             for detection in detections:
#                 x1, y1, x2, y2 = detection['bbox']
#                 conf = detection['confidence']
#
#                 cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame_vis, f'Bag: {conf:.2f}', (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#
#             # Draw confirmed tracks (without ID display)
#             for track_id, bbox in confirmed_tracks.items():
#                 x, y, w, h = bbox
#                 center = (x + w // 2, y + h // 2)
#
#                 # Draw track center point only
#                 cv2.circle(frame_vis, center, 5, (255, 0, 255), -1)
#
#             # Draw counting lines
#             for i, (line_start, line_end) in enumerate(lines):
#                 line_color = line_colors[i] if i < len(line_colors) else (0, 0, 255)
#                 cv2.line(frame_vis, line_start, line_end, line_color, 1)
#
#                 # Add line number at top of line
#                 line_x = line_start[0]  # Use x-coordinate of line
#                 top_y = 20  # Position at top of frame
#                 cv2.putText(frame_vis, f"{i + 1}", (line_x - 5, top_y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)
#
#                 # Draw direction arrow at bottom of line
#                 direction = line_directions[i] if i < len(line_directions) else "left"
#                 bottom_x = line_start[0]
#                 bottom_y = line_end[1] - 10  # Near bottom of line
#                 arrow_length = 15  # Smaller arrow
#
#                 if direction == "left":
#                     arrow_end = (bottom_x - arrow_length, bottom_y)
#                 elif direction == "right":
#                     arrow_end = (bottom_x + arrow_length, bottom_y)
#                 elif direction == "up":
#                     arrow_end = (bottom_x, bottom_y - arrow_length)
#                 else:  # down
#                     arrow_end = (bottom_x, bottom_y + arrow_length)
#
#                 cv2.arrowedLine(frame_vis, (bottom_x, bottom_y), arrow_end, (0, 255, 255), 1)
#
#             # Calculate FPS
#             fps_frame_count += 1
#             elapsed_time = time.time() - fps_start_time
#             if elapsed_time > 1.0:  # Update FPS every second
#                 fps_local = fps_frame_count / elapsed_time
#                 fps_start_time = time.time()
#                 fps_frame_count = 0
#
#             # Draw UI elements
#             font_scale = 0.6
#             thickness = 2
#
#             cv2.putText(frame_vis, f"Bag In: {bag_in}", (10, 35),
#                         cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
#             cv2.putText(frame_vis, f"Bag Out: {bag_out}", (10, 70),
#                         cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
#             cv2.putText(frame_vis, f"FPS: {fps_local:.2f}", (10, 105),
#                         cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
#             # cv2.putText(frame_vis, f"Tracks: {len(confirmed_tracks)}", (10, 140),
#             #            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), thickness)
#             cv2.putText(frame_vis, f"Camera: {CAMERA_ID}", (10, 140),
#                         cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
#
#             # Encode frame for streaming
#             ret, jpeg = cv2.imencode('.jpg', frame_vis)
#             if ret:
#                 latest_frame = jpeg.tobytes()
#
#             # Periodic cleanup
#             cleanup_counter += 1
#             if cleanup_counter % 100 == 0:
#                 if device == 'cuda':
#                     torch.cuda.empty_cache()
#                 gc.collect()
#
#         except queue.Empty:
#             continue
#         except Exception as e:
#             logger.error(f"[{CAMERA_ID}] Tracking error: {e}")
#             time.sleep(0.1)
#
#
# def detection_loop():
#     """Main detection loop - orchestrates multiple threads for optimal performance"""
#     global stream_up
#
#     logger.info(f"[{CAMERA_ID}] Starting optimized multithreaded detection system")
#
#     # Start all processing threads
#     threads = [
#         threading.Thread(target=frame_capture_thread, daemon=True, name="FrameCapture"),
#         threading.Thread(target=detection_processing_thread, daemon=True, name="Detection"),
#         threading.Thread(target=tracking_and_counting_thread, daemon=True, name="Tracking")
#     ]
#
#     for thread in threads:
#         thread.start()
#
#     try:
#         while True:
#             # Check if all threads are alive
#             alive_threads = [t for t in threads if t.is_alive()]
#             if len(alive_threads) < len(threads):
#                 logger.warning(f"[{CAMERA_ID}] Some threads died, restarting...")
#                 break
#
#             time.sleep(5)  # Check every 5 seconds
#     except KeyboardInterrupt:
#         logger.info(f"[{CAMERA_ID}] Shutting down detection system")
#     except Exception as e:
#         logger.error(f"[{CAMERA_ID}] Detection loop error: {e}")
#     finally:
#         stream_up = False
#         executor.shutdown(wait=False)
#
#
# # Flask routes
# @app.route('/')
# def home():
#     return f"Welcome to the Cement Bag Detection Flask App! Camera ID: {CAMERA_ID}"
#
#
# @app.route('/video_feed')
# def video_feed():
#     def generate():
#         while True:
#             if latest_frame is not None:
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
#             else:
#                 time.sleep(0.1)
#
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# @app.route('/bag_count')
# def get_bag_count():
#     global bag_in, bag_out
#     return jsonify({
#         'bag_in': bag_in,
#         'bag_out': bag_out,
#         'total': bag_in + bag_out
#     })
#
#
# @app.route('/counting')
# def counting():
#     global counting_active, bag_in, bag_out
#     # Toggle counting with ?active=true/false
#     active = request.args.get('active')
#     if active is not None:
#         counting_active = active.lower() == 'true'
#     return jsonify({
#         'counting_active': counting_active,
#         'bag_in': bag_in,
#         'bag_out': bag_out,
#         'total_count': bag_in + bag_out,
#         'camera_id': CAMERA_ID
#     })
#
#
# @app.route('/confidence')
# def get_confidence():
#     global camera_confidence
#     return jsonify({'confidence': camera_confidence})
#
#
# @app.route('/health')
# def health():
#     global stream_up
#     status = "ok" if stream_up else "stream_error"
#     return jsonify({"status": status, "camera_id": CAMERA_ID})
#
#
# @app.route('/reset')
# def reset():
#     global bag_in, bag_out
#     bag_in = 0
#     bag_out = 0
#     return jsonify({
#         'message': 'Counters reset',
#         'bag_in': bag_in,
#         'bag_out': bag_out,
#         'camera_id': CAMERA_ID
#     })
#
#
# @app.route('/reset_in')
# def reset_in():
#     global bag_in
#     bag_in = 0
#     return jsonify({
#         'message': 'Bag IN counter reset',
#         'bag_in': bag_in,
#         'bag_out': bag_out,
#         'camera_id': CAMERA_ID
#     })
#
#
# @app.route('/reset_out')
# def reset_out():
#     global bag_out
#     bag_out = 0
#     return jsonify({
#         'message': 'Bag OUT counter reset',
#         'bag_in': bag_in,
#         'bag_out': bag_out,
#         'camera_id': CAMERA_ID
#     })
#
# # --- Simple manual UI and update endpoint ---
# @app.route('/ui')
# def manual_ui():
#     try:
#         return render_template('manual_count_simple.html')
#     except Exception as e:
#         return f"Error loading UI: {str(e)}"
#
# @app.route('/update_count', methods=['POST'])
# def update_count():
#     try:
#         global bag_in, bag_out
#         data = request.get_json(silent=True) or {}
#         if 'in' in data:
#             bag_in = max(0, int(data['in']))
#         if 'out' in data:
#             bag_out = max(0, int(data['out']))
#         return jsonify({'success': True, 'bag_in': bag_in, 'bag_out': bag_out})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 400
#
#
# @app.route('/counting_lines')
# def counting_lines():
#     global counting_active
#     action = request.args.get('action', '').lower()
#
#     if action == 'enable':
#         counting_active = True
#         message = 'Counting lines enabled'
#     elif action == 'disable':
#         counting_active = False
#         message = 'Counting lines disabled'
#     elif action == 'toggle':
#         counting_active = not counting_active
#         message = f'Counting lines {"enabled" if counting_active else "disabled"}'
#     else:
#         message = 'Current counting lines status'
#
#     return jsonify({
#         'message': message,
#         'counting_active': counting_active,
#         'camera_id': CAMERA_ID
#     })
#
#
# @app.route('/counting_in')
# def counting_in():
#     global counting_in_active
#     action = request.args.get('action', '').lower()
#
#     if action == 'enable':
#         counting_in_active = True
#         message = 'IN direction counting enabled'
#     elif action == 'disable':
#         counting_in_active = False
#         message = 'IN direction counting disabled'
#     elif action == 'toggle':
#         counting_in_active = not counting_in_active
#         message = f'IN direction counting {"enabled" if counting_in_active else "disabled"}'
#     else:
#         message = 'Current IN direction counting status'
#
#     return jsonify({
#         'message': message,
#         'counting_in_active': counting_in_active,
#         'counting_out_active': counting_out_active,  # Include OUT status for reference
#         'camera_id': CAMERA_ID
#     })
#
#
# @app.route('/counting_out')
# def counting_out():
#     global counting_out_active
#     action = request.args.get('action', '').lower()
#
#     if action == 'enable':
#         counting_out_active = True
#         message = 'OUT direction counting enabled'
#     elif action == 'disable':
#         counting_out_active = False
#         message = 'OUT direction counting disabled'
#     elif action == 'toggle':
#         counting_out_active = not counting_out_active
#         message = f'OUT direction counting {"enabled" if counting_out_active else "disabled"}'
#     else:
#         message = 'Current OUT direction counting status'
#
#     return jsonify({
#         'message': message,
#         'counting_in_active': counting_in_active,  # Include IN status for reference
#         'counting_out_active': counting_out_active,
#         'camera_id': CAMERA_ID
#     })
#
#
# @app.route('/status')
# def status():
#     global bag_in, bag_out, counting_active, counting_in_active, counting_out_active, stream_up, camera_confidence
#
#     return jsonify({
#         'camera_id': CAMERA_ID,
#         'stream_status': 'online' if stream_up else 'offline',
#         'counting_active': counting_active,
#         'counting_in_active': counting_in_active,
#         'counting_out_active': counting_out_active,
#         'bag_in': bag_in,
#         'bag_out': bag_out,
#         'total_count': bag_in + bag_out,
#         'confidence': camera_confidence,
#         'timestamp': time.time()
#     })
#
#
# # Start detection loop in background thread
# t = threading.Thread(target=detection_loop)
# t.daemon = True
# t.start()
#
# if __name__ == "__main__":
#     # Get port from environment variable or use default
#     port = int(os.getenv("PORT", 5004))
#     print(f"Starting optimized server on port {port}")
#     logger.info(f"[{CAMERA_ID}] Enhanced tracker with multithreading optimization enabled")
#     app.run(host="0.0.0.0", port=port)
#
"""
Cement Bag Counter — Optimized Build

Suggested .env_cam1:

CEMENT_MODEL_PATH=./best_cement-bags_102.pt
RTSP_URL=rtsp://user:pass@192.168.1.10/stream
CAMERA_ID=CAM1

# Performance knobs
INFER_IMG_SIZE=416           # 352/384/416/480; lower => faster
PROCESS_EVERY_N=3            # process every Nth frame
CAPTURE_RESIZE_W=512         # early downscale at capture
MAX_DET=20                   # cap detections per frame
CONF_THRESHOLD=0.35

# Rendering & streaming
DRAW_OVERLAYS=false          # disable heavy drawing to save CPU
JPEG_QUALITY=70              # 60–75 is good balance

# Tracker & duplicate control
MAX_HISTORY=30               # track points kept per ID
COUNT_COOLDOWN_SECONDS=2.0   # time window for duplicate prevention
SPATIAL_THRESHOLD=80         # px radius for duplicate prevention

# Lines (example)
MULTI_LINE_COORDS=100,50,500,50;100,400,500,400
MULTI_COUNT_DIRECTIONS=up;down
LINE_COLORS=0,0,255;0,255,0
FRAME_SKIP=0                 # keep at 0; we use PROCESS_EVERY_N instead
MAX_BATCH_SIZE=1             # not used; single-frame pipeline
PORT=8000
"""

import os
import time
import cv2
import numpy as np
import threading
import queue
import logging
import torch
import gc
import traceback
import sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO
from etracker import EnhancedBagTracker, LineCounter
import multiprocessing as mp
from collections import deque, defaultdict
from dotenv import load_dotenv

# ----------------------------
# Thread & BLAS limits (CPU savings)
# ----------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)
cv2.setNumThreads(1)

# ----------------------------
# Env & logging
# ----------------------------
env_file = '.env_cam4'
print(f"Loading environment from: {env_file}")
load_dotenv(dotenv_path=env_file)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("cement-counter")

# ----------------------------
# Config
# ----------------------------
CEMENT_MODEL_PATH = os.getenv("CEMENT_MODEL_PATH")
RTSP_URL = os.getenv("RTSP_URL")
CAMERA_ID = os.getenv("CAMERA_ID", "CAM1")

# Performance knobs
INFER_IMG_SIZE     = int(os.getenv("INFER_IMG_SIZE", "416"))
PROCESS_EVERY_N    = int(os.getenv("PROCESS_EVERY_N", "3"))
CAPTURE_RESIZE_W   = int(os.getenv("CAPTURE_RESIZE_W", "512"))
MAX_DET            = int(os.getenv("MAX_DET", "20"))
CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "0.35"))

# Rendering
DRAW_OVERLAYS      = os.getenv("DRAW_OVERLAYS", "true").lower() == "true"
JPEG_QUALITY       = int(os.getenv("JPEG_QUALITY", "70"))

# Tracker & duplicate control
MAX_HISTORY              = int(os.getenv("MAX_HISTORY", "30"))
COUNT_COOLDOWN_SECONDS   = float(os.getenv("COUNT_COOLDOWN_SECONDS", "2.0"))
SPATIAL_THRESHOLD        = int(os.getenv("SPATIAL_THRESHOLD", "80"))

# Legacy/unneeded but retained for compatibility
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "0"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1"))

# ----------------------------
# Line config
# ----------------------------
def parse_multi_line_config():
    lines, directions, colors = [], [], []
    mlc = os.getenv("MULTI_LINE_COORDS")
    mld = os.getenv("MULTI_COUNT_DIRECTIONS")
    mlcolor = os.getenv("LINE_COLORS")

    if not mlc or not mld or not mlcolor:
        logger.error("Counting line env vars missing (MULTI_LINE_COORDS / MULTI_COUNT_DIRECTIONS / LINE_COLORS)")
        return [], [], []

    for s in mlc.split(";"):
        try:
            x1, y1, x2, y2 = map(int, s.split(","))
            lines.append(((x1, y1), (x2, y2)))
        except Exception:
            logger.warning(f"Invalid line coordinates: {s}")

    for s in mld.split(";"):
        directions.append(s.strip())

    for s in mlcolor.split(";"):
        try:
            b, g, r = map(int, s.split(","))
            colors.append((b, g, r))
        except Exception:
            colors.append((0, 0, 255))

    n = min(len(lines), len(directions), len(colors))
    return lines[:n], directions[:n], colors[:n]

lines, line_directions, line_colors = parse_multi_line_config()
logger.info(f"Loaded {len(lines)} counting lines")

# ----------------------------
# Helpers for line-crossing
# ----------------------------
def is_crossing_line(point, line):
    x, y = point
    (x1, y1), (x2, y2) = line
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return 1 if (a * x + b * y + c) > 0 else -1

def calculate_direction(old_point, new_point, line_start, line_end):
    lvx, lvy = line_end[0] - line_start[0], line_end[1] - line_start[1]
    mvx, mvy = new_point[0] - old_point[0], new_point[1] - old_point[1]
    cross = lvx * mvy - lvy * mvx
    return "IN" if cross > 0 else "OUT"

# ----------------------------
# Global state
# ----------------------------
bag_in = 0
bag_out = 0
counting_active = True
counting_in_active = True
counting_out_active = True

tracks_history = defaultdict(list)
track_movement_state = defaultdict(lambda: {'last_direction': None, 'counted': False})
recent_counts = []  # (timestamp, (x,y), dir)

frame_queue = queue.Queue(maxsize=2)        # tiny buffers to bound memory
detection_queue = queue.Queue(maxsize=2)
processing_lock = threading.Lock()

# Thread pool (small)
max_workers = max(1, min(2, mp.cpu_count()))  # keep it small for resource savings
executor = ThreadPoolExecutor(max_workers=max_workers)

camera_confidence = 0.0
stream_up = False

# ----------------------------
# Device
# ----------------------------
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    free_mem, total_mem = torch.cuda.mem_get_info()
    logger.info(f'CUDA ON. Free: {free_mem/1024**2:.0f}MB / {total_mem/1024**2:.0f}MB')
else:
    device = 'cpu'
    logger.info('CUDA OFF. Using CPU.')

# ----------------------------
# Load YOLO model (safe fuse + warmup + fp16)
# ----------------------------
try:
    if not CEMENT_MODEL_PATH or not os.path.exists(CEMENT_MODEL_PATH):
        raise FileNotFoundError(f"CEMENT_MODEL_PATH invalid: {CEMENT_MODEL_PATH}")

    logger.info(f"Loading model: {CEMENT_MODEL_PATH}")
    model_cement = YOLO(CEMENT_MODEL_PATH, task='detect')

    # Try fusion; skip if layers lack .bn (prevents 'bn' crash)
    try:
        model_cement.fuse()
        logger.info("Model fused.")
    except Exception as e:
        logger.warning(f"Skip fuse: {e}")

    use_cuda = torch.cuda.is_available()
    device_index = 0 if use_cuda else "cpu"

    # Warmup with inference_mode & fp16
    try:
        dummy = torch.zeros(1, 3, INFER_IMG_SIZE, INFER_IMG_SIZE, device=("cuda" if use_cuda else "cpu"))
        with torch.inference_mode():
            model_cement.predict(
                source=dummy,
                device=device_index,
                imgsz=INFER_IMG_SIZE,
                half=use_cuda,
                conf=max(0.05, min(CONF_THRESHOLD, 0.9)),
                iou=0.45,
                max_det=MAX_DET,
                verbose=False
            )
        logger.info(f"Warmup OK on {'GPU' if use_cuda else 'CPU'} (fp16={use_cuda})")
    except Exception as e:
        logger.warning(f"Warmup warning: {e}")

except Exception as e:
    logger.error(f"Model load error: {e}")
    logger.error("Traceback:\n%s", traceback.format_exc())
    sys.exit(1)

# ----------------------------
# Tracker & LineCounter
# ----------------------------
tracker = EnhancedBagTracker(
    iou_threshold=0.25,
    max_missing=80,       # slightly lower than before to free memory sooner
    min_hits=2,
    max_age=40,
    motion_threshold=60
)
line_counter = LineCounter(lines)

# ----------------------------
# Flask
# ----------------------------
app = Flask(__name__)

def create_test_frame():
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    cv2.putText(frame, f"{CAMERA_ID} Connecting...", (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return jpeg.tobytes() if ret else None

latest_frame = create_test_frame()

# ----------------------------
# Capture thread (early downscale & tiny buffers)
# ----------------------------
def frame_capture_thread():
    global stream_up
    cap = None
    reconnect_delay = 5
    max_reconnect_delay = 60
    consecutive_failures = 0
    max_consecutive_failures = 10

    while True:
        try:
            logger.info(f"[{CAMERA_ID}] Frame capture starting")

            rtsp_url = RTSP_URL or ""
            if '?' not in rtsp_url:
                rtsp_url += '?'
            else:
                rtsp_url += '&'
            rtsp_url += 'rtsp_transport=tcp&timeout=15000000&buffer_size=1024&max_delay=500000'
            logger.info(f"[{CAMERA_ID}] RTSP over TCP")

            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'protocol_whitelist;file,rtp,udp,tcp,tls,https,rtsp|'
                'fflags;nobuffer|max_delay;500000|reorder_queue_size;0|'
                'rtsp_transport;tcp|stimeout;5000000'
            )

            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 12)  # lower requested FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                raise ConnectionError(f"Failed to open RTSP: {rtsp_url}")

            logger.info(f"[{CAMERA_ID}] Stream connected")
            stream_up = True
            frame_number = 0
            consecutive_failures = 0
            reconnect_delay = 5
            last_ok = time.time()

            while True:
                now = time.time()
                if now - last_ok > 30:
                    logger.warning(f"[{CAMERA_ID}] Stream timeout (>30s no frames)")
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"[{CAMERA_ID}] Too many read failures; reconnecting")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                last_ok = now

                # Early downscale to CAPTURE_RESIZE_W
                if frame.shape[1] > CAPTURE_RESIZE_W:
                    new_h = int(CAPTURE_RESIZE_W * frame.shape[0] / frame.shape[1])
                    frame = cv2.resize(frame, (CAPTURE_RESIZE_W, new_h))

                try:
                    # Keep queue tiny; drop if full
                    frame_queue.put((frame, frame_number), block=False)
                    frame_number += 1
                except queue.Full:
                    pass

        except Exception as e:
            logger.error(f"[{CAMERA_ID}] Capture error: {e}")
            stream_up = False

        finally:
            if cap is not None:
                cap.release()
            stream_up = False
            logger.info(f"[{CAMERA_ID}] Reconnecting in {reconnect_delay}s")
            time.sleep(reconnect_delay)
            reconnect_delay = min(2 * reconnect_delay, max_reconnect_delay)

# ----------------------------
# Detection (skip frames, fp16, inference_mode)
# ----------------------------
def detect_objects_optimized(frame, frame_number):
    try:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
            return [], frame_number

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        use_cuda = torch.cuda.is_available()
        device_index = 0 if use_cuda else "cpu"

        t0 = time.time()
        with torch.inference_mode():
            results = model_cement.predict(
                source=frame,
                device=device_index,
                imgsz=INFER_IMG_SIZE,
                half=use_cuda,
                conf=CONF_THRESHOLD,
                iou=0.45,
                max_det=MAX_DET,
                stream=False,
                verbose=False
            )
        dt = time.time() - t0
        if dt > 1.0:
            logger.warning(f"[{CAMERA_ID}] Slow detection: {dt:.2f}s for frame #{frame_number}")

        detections = []
        if results:
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for b in r.boxes:
                    try:
                        if b.xyxy.numel() == 0 or b.conf.numel() == 0 or b.cls.numel() == 0:
                            continue
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        conf = float(b.conf[0].cpu().numpy())
                        cls = int(b.cls[0].cpu().numpy())
                        w = int(x2 - x1)
                        h = int(y2 - y1)
                        if w <= 0 or h <= 0:
                            continue
                        ar = w / h
                        if cls == 0 and conf >= CONF_THRESHOLD and w >= 30 and h >= 30 and 0.5 <= ar <= 2.0:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': conf,
                                'class_id': cls,
                                'width': w, 'height': h, 'aspect_ratio': ar,
                                'frame_number': frame_number
                            })
                    except Exception:
                        continue

        return detections, frame_number

    except Exception as e:
        logger.error(f"[{CAMERA_ID}] detect error on frame {frame_number}: {e}")
        logger.error("Traceback:\n%s", traceback.format_exc())
        return [], frame_number

# ----------------------------
# Detection processing thread (frame skip + tiny queue)
# ----------------------------
def detection_processing_thread():
    global camera_confidence
    while True:
        try:
            frame, frame_number = frame_queue.get(timeout=1.0)

            # Aggressive frame skipping for resource savings
            if PROCESS_EVERY_N > 1 and (frame_number % PROCESS_EVERY_N != 0):
                continue

            # Submit detection (tiny pool)
            future = executor.submit(detect_objects_optimized, frame, frame_number)
            try:
                detections, _ = future.result(timeout=2.0)
                # Fast confidence aggregation
                camera_confidence = max((d['confidence'] for d in detections), default=0.0)
                try:
                    detection_queue.put((detections, frame, frame_number), block=False)
                except queue.Full:
                    pass
            except concurrent.futures._base.TimeoutError:
                future.cancel()
                logger.warning(f"[{CAMERA_ID}] Detection timeout for frame #{frame_number}")

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"[{CAMERA_ID}] Detection thread error: {e}")
            time.sleep(0.05)

# ----------------------------
# Tracking & counting (minimal drawing)
# ----------------------------
def tracking_and_counting_thread():
    global bag_in, bag_out, latest_frame

    fps_local = 0.0
    fps_start = time.time()
    fps_count = 0
    cleanup_counter = 0

    while True:
        try:
            detections, frame, frame_number = detection_queue.get(timeout=1.0)

            # Convert detections to tracker format
            tracker_dets = []
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                tracker_dets.append((x1, y1, x2 - x1, y2 - y1, d['confidence']))

            confirmed_tracks = tracker.update(tracker_dets, frame_number)

            # Update history with bound length
            for tid, bbox in confirmed_tracks.items():
                x, y, w, h = bbox
                center = (x + w // 2, y + h // 2)
                hist = tracks_history[tid]
                hist.append(center)
                if len(hist) > MAX_HISTORY:
                    del hist[:-MAX_HISTORY]

            # Direction-based counting with duplicate prevention
            now = time.time()
            with processing_lock:
                # Clean recent counts window
                if recent_counts:
                    keep = [(t, p, d) for (t, p, d) in recent_counts if now - t <= COUNT_COOLDOWN_SECONDS]
                    recent_counts[:] = keep

                for tid, hist in tracks_history.items():
                    if len(hist) < 3:
                        continue

                    if tid not in track_movement_state:
                        track_movement_state[tid] = {'last_direction': None, 'counted': False}

                    current_dir = None
                    for i, (ls, le) in enumerate(lines):
                        cur = hist[-1]; prev = hist[-2]
                        if is_crossing_line(cur, (ls, le)) != is_crossing_line(prev, (ls, le)):
                            current_dir = calculate_direction(prev, cur, ls, le)
                            break

                    if current_dir and counting_active:
                        st = track_movement_state[tid]
                        if tid in confirmed_tracks:
                            x, y, w, h = confirmed_tracks[tid]
                            pos = (x + w // 2, y + h // 2)

                            # duplicate?
                            dup = any(
                                (d == current_dir and abs(pos[0]-p[0]) < SPATIAL_THRESHOLD and abs(pos[1]-p[1]) < SPATIAL_THRESHOLD)
                                for (t, p, d) in recent_counts
                            )

                            if not dup and (st['last_direction'] != current_dir or st['last_direction'] is None):
                                if current_dir == "IN" and counting_in_active:
                                    bag_in += 1
                                    logger.info(f"[{CAMERA_ID}] Bag IN (Track {tid}) -> {bag_in}")
                                elif current_dir == "OUT" and counting_out_active:
                                    bag_out += 1
                                    logger.info(f"[{CAMERA_ID}] Bag OUT (Track {tid}) -> {bag_out}")

                                recent_counts.append((now, pos, current_dir))
                                st['last_direction'] = current_dir
                                st['counted'] = True

            # Minimal rendering
            frame_vis = frame if not DRAW_OVERLAYS else frame.copy()
            if DRAW_OVERLAYS:
                # Boxes
                for d in detections:
                    x1, y1, x2, y2 = d['bbox']
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Track centers
                for tid, bbox in confirmed_tracks.items():
                    x, y, w, h = bbox
                    c = (x + w // 2, y + h // 2)
                    cv2.circle(frame_vis, c, 3, (255, 0, 255), -1)

                # Lines
                for i, (ls, le) in enumerate(lines):
                    color = line_colors[i] if i < len(line_colors) else (0, 0, 255)
                    cv2.line(frame_vis, ls, le, color, 1)

                # HUD (very light)
                cv2.putText(frame_vis, f"IN:{bag_in} OUT:{bag_out} FPS:{fps_local:.1f} {CAMERA_ID}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # FPS calc
            fps_count += 1
            dt = time.time() - fps_start
            if dt >= 1.0:
                fps_local = fps_count / dt
                fps_start = time.time()
                fps_count = 0

            # Encode
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            ret, jpeg = cv2.imencode('.jpg', frame_vis, encode_params)
            if ret:
                latest_frame = jpeg.tobytes()

            # Periodic cleanup
            cleanup_counter += 1
            if cleanup_counter % 120 == 0:
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"[{CAMERA_ID}] Tracking error: {e}")
            logger.error("Traceback:\n%s", traceback.format_exc())
            time.sleep(0.05)

# ----------------------------
# Orchestrator
# ----------------------------
def detection_loop():
    logger.info(f"[{CAMERA_ID}] Multithreaded detection starting (optimized)")
    threads = [
        threading.Thread(target=frame_capture_thread, daemon=True, name="FrameCapture"),
        threading.Thread(target=detection_processing_thread, daemon=True, name="Detection"),
        threading.Thread(target=tracking_and_counting_thread, daemon=True, name="Tracking"),
    ]
    for t in threads: t.start()

    try:
        while True:
            if any(not t.is_alive() for t in threads):
                logger.warning(f"[{CAMERA_ID}] A worker died, restarting soon...")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info(f"[{CAMERA_ID}] Shutdown requested")
    except Exception as e:
        logger.error(f"[{CAMERA_ID}] Loop error: {e}")
        logger.error("Traceback:\n%s", traceback.format_exc())
    finally:
        executor.shutdown(wait=False)

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def home():
    return f"Welcome to the Cement Bag Detection Flask App! Camera ID: {CAMERA_ID}"

@app.route('/favicon.ico')
def favicon():
    return "", 204

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            else:
                time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bag_count')
def get_bag_count():
    return jsonify({'bag_in': bag_in, 'bag_out': bag_out, 'total': bag_in + bag_out})

@app.route('/counting')
def counting():
    global counting_active
    active = request.args.get('active')
    if active is not None:
        counting_active = active.lower() == 'true'
    return jsonify({'counting_active': counting_active, 'bag_in': bag_in, 'bag_out': bag_out, 'total_count': bag_in + bag_out, 'camera_id': CAMERA_ID})

@app.route('/confidence')
def get_confidence():
    return jsonify({'confidence': camera_confidence})

@app.route('/health')
def health():
    return jsonify({"status": "ok" if stream_up else "stream_error", "camera_id": CAMERA_ID})

@app.route('/reset')
def reset():
    global bag_in, bag_out
    bag_in = 0; bag_out = 0
    return jsonify({'message': 'Counters reset', 'bag_in': bag_in, 'bag_out': bag_out, 'camera_id': CAMERA_ID})

@app.route('/reset_in')
def reset_in():
    global bag_in
    bag_in = 0
    return jsonify({'message': 'Bag IN counter reset', 'bag_in': bag_in, 'bag_out': bag_out, 'camera_id': CAMERA_ID})

@app.route('/reset_out')
def reset_out():
    global bag_out
    bag_out = 0
    return jsonify({'message': 'Bag OUT counter reset', 'bag_in': bag_in, 'bag_out': bag_out, 'camera_id': CAMERA_ID})

@app.route('/counting_lines')
def counting_lines():
    global counting_active
    action = request.args.get('action', '').lower()
    if action == 'enable':   counting_active = True;  msg = 'Counting lines enabled'
    elif action == 'disable': counting_active = False; msg = 'Counting lines disabled'
    elif action == 'toggle':  counting_active = not counting_active; msg = f'Counting lines {"enabled" if counting_active else "disabled"}'
    else: msg = 'Current counting lines status'
    return jsonify({'message': msg, 'counting_active': counting_active, 'camera_id': CAMERA_ID})

@app.route('/counting_in')
def counting_in():
    global counting_in_active
    action = request.args.get('action', '').lower()
    if action == 'enable':   counting_in_active = True;  msg = 'IN direction counting enabled'
    elif action == 'disable': counting_in_active = False; msg = 'IN direction counting disabled'
    elif action == 'toggle':  counting_in_active = not counting_in_active; msg = f'IN direction counting {"enabled" if counting_in_active else "disabled"}'
    else: msg = 'Current IN direction counting status'
    return jsonify({'message': msg, 'counting_in_active': counting_in_active, 'counting_out_active': counting_out_active, 'camera_id': CAMERA_ID})

@app.route('/counting_out')
def counting_out():
    global counting_out_active
    action = request.args.get('action', '').lower()
    if action == 'enable':   counting_out_active = True;  msg = 'OUT direction counting enabled'
    elif action == 'disable': counting_out_active = False; msg = 'OUT direction counting disabled'
    elif action == 'toggle':  counting_out_active = not counting_out_active; msg = f'OUT direction counting {"enabled" if counting_out_active else "disabled"}'
    else: msg = 'Current OUT direction counting status'
    return jsonify({'message': msg, 'counting_in_active': counting_in_active, 'counting_out_active': counting_out_active, 'camera_id': CAMERA_ID})

@app.route('/status')
def status():
    return jsonify({
        'camera_id': CAMERA_ID,
        'stream_status': 'online' if stream_up else 'offline',
        'counting_active': counting_active,
        'counting_in_active': counting_in_active,
        'counting_out_active': counting_out_active,
        'bag_in': bag_in, 'bag_out': bag_out,
        'total_count': bag_in + bag_out,
        'confidence': camera_confidence,
        'timestamp': time.time()
    })

# ----------------------------
# Boot
# ----------------------------
t = threading.Thread(target=detection_loop, daemon=True)
t.start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5004"))
    print(f"Starting optimized server on port {port}")
    logger.info(f"[{CAMERA_ID}] Optimized pipeline active (N={PROCESS_EVERY_N}, img={INFER_IMG_SIZE}, overlays={DRAW_OVERLAYS})")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
