import cv2
import time
import os
import sys
import numpy as np
import logging
import torch
import gc
import threading
import queue
from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from ultralytics import YOLO
from tracker import Tracker

# Helper functions for line crossing detection
def is_crossing_line(point, line):
    """Check if a point is on the line side"""
    x, y = point
    (x1, y1), (x2, y2) = line
    
    # Calculate line equation: ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    # Calculate the value of the line equation at the point
    value = a * x + b * y + c
    
    # Return the sign of the value (positive or negative)
    return 1 if value > 0 else -1
    
def calculate_direction(old_point, new_point, line_start, line_end):
    """Calculate the direction of movement relative to the line"""
    # Get the line vector
    line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    
    # Get the movement vector
    movement_vector = (new_point[0] - old_point[0], new_point[1] - old_point[1])
    
    # Calculate the cross product for direction
    cross_product = line_vector[0] * movement_vector[1] - line_vector[1] * movement_vector[0]
    
    # Determine the direction based on the cross product
    return "IN" if cross_product > 0 else "OUT"


def get_color_direction_key(color, direction):
    """Create a unique key for color-direction combination"""
    return f"{color[0]}_{color[1]}_{color[2]}_{direction}"


def parse_multi_line_config():
    """Parse multi-line configuration from environment variables"""
    lines = []
    directions = []
    colors = []
    
    # Parse multi-line coordinates
    multi_line_coords = os.getenv("MULTI_LINE_COORDS", "160,0,160,480;240,0,240,480;320,0,320,480;400,0,400,480")
    for line_str in multi_line_coords.split(";"):
        try:
            x1, y1, x2, y2 = map(int, line_str.split(","))
            lines.append(((x1, y1), (x2, y2)))
        except ValueError:
            logger.warning(f"Invalid line coordinates: {line_str}")
    
    # Parse directions
    multi_directions = os.getenv("MULTI_COUNT_DIRECTIONS", "right;right;right;right")
    directions = multi_directions.split(";")
    
    # Parse colors
    multi_colors = os.getenv("LINE_COLORS", "0,0,255;0,255,0;255,0,0;255,255,0")
    for color_str in multi_colors.split(";"):
        try:
            b, g, r = map(int, color_str.split(","))
            colors.append((b, g, r))
        except ValueError:
            logger.warning(f"Invalid color: {color_str}")
            colors.append((0, 0, 255))  # Default to red
    
    # Ensure all lists have the same length
    min_length = min(len(lines), len(directions), len(colors))
    lines = lines[:min_length]
    directions = directions[:min_length]
    colors = colors[:min_length]
    
    return lines, directions, colors

# Load environment variables
env_file = '.env_upload'
print(f"Loading environment from: {env_file}")
load_dotenv(dotenv_path=env_file)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables
CEMENT_MODEL_PATH = os.getenv("CEMENT_MODEL_PATH")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.2))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 1))  # Get frame skip from env
INFER_IMG_SIZE = int(os.getenv("INFER_IMG_SIZE", 320))  # Inference image size
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 1))  # Maximum batch size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Parse multi-line configuration
lines, line_directions, line_colors = parse_multi_line_config()
logger.info(f"Loaded {len(lines)} counting lines with multi-line deduplication")

# Legacy single line support (for backward compatibility)
line_coords_str = os.getenv("LINE_COORDS", "320,0,320,480")
try:
    x1, y1, x2, y2 = map(int, line_coords_str.split(","))
    line_start = (x1, y1)
    line_end = (x2, y2)
except Exception as e:
    logger.error(f"Invalid LINE_COORDS in .env: {line_coords_str}")
    line_start = (320, 0)
    line_end = (320, 480)

# Get line color from env (BGR format)
line_color_str = os.getenv("LINE_COLOR", "0,0,255")
try:
    b, g, r = map(int, line_color_str.split(","))
    LINE_COLOR = (b, g, r)
except Exception as e:
    logger.warning(f"Invalid LINE_COLOR in .env: {line_color_str}. Using default red.")
    LINE_COLOR = (0, 0, 255)  # Default to red

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
bag_count = 0
processing_video = False
video_path = None
fps = 0.0
counting_active = True
camera_confidence = 0.0

# Initialize with a test frame to ensure video feed works
def create_test_frame():
    """Create a test frame to show when no video is being processed"""
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, 'Upload a video to start processing', (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(test_frame, 'Multi-line counting ready', (120, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    ret, jpeg = cv2.imencode('.jpg', test_frame)
    if ret:
        return jpeg.tobytes()
    return None

# Initialize latest_frame with test frame
latest_frame = create_test_frame()

# Multi-line counting and deduplication variables
previous_positions = {}  # Store previous positions for each object
crossed_groups = {}  # Track crossings by color-direction groups to prevent double counting
last_cleanup_time = time.time()  # For periodic cleanup

# Determine device (GPU if available, else CPU)
if torch.cuda.is_available():
    device = 'cuda'
    # Set memory management options
    torch.cuda.empty_cache()
    # Enable memory efficient features
    torch.backends.cudnn.benchmark = True
    # Log available GPU memory
    free_mem, total_mem = torch.cuda.mem_get_info()
    logger.info(f'CUDA is available. Using GPU for inference. Free memory: {free_mem/1024**2:.1f}MB / {total_mem/1024**2:.1f}MB')
else:
    device = 'cpu'
    logger.info('CUDA is not available. Using CPU for inference.')

# Load YOLO model
try:
    if not CEMENT_MODEL_PATH or not os.path.exists(CEMENT_MODEL_PATH):
        logger.error(f"CEMENT_MODEL_PATH is missing or invalid: {CEMENT_MODEL_PATH}")
        raise FileNotFoundError(f"CEMENT_MODEL_PATH is missing or invalid: {CEMENT_MODEL_PATH}")
    
    model_cement = YOLO(CEMENT_MODEL_PATH)
    model_cement.to(device)
    
    # Force garbage collection after model loading
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    logger.info(f"Model loaded successfully from {CEMENT_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Threaded Video Capture class
class VideoCaptureThread:
    def __init__(self, src, width=None, height=None):
        self.cap = cv2.VideoCapture(src)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            
            # Clear the queue to always have the latest frame
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            
            self.q.put((ret, frame))
            time.sleep(0.01)  # Small sleep to prevent CPU overuse

    def read(self):
        if not self.running:
            return False, None
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# Global variables for progress tracking
processing_progress = 0
total_frames = 0
processed_frames = 0

# Video processing function with chunked processing for large files
def process_video():
    global bag_count, latest_frame, processing_video, video_path, processing_progress, total_frames, processed_frames
    
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        processing_video = False
        return
    
    # Reset counters for new video
    bag_count = 0
    processing_progress = 0
    processed_frames = 0
    
    # Reset multi-line counting variables
    global previous_positions, crossed_groups, camera_confidence
    previous_positions = {}
    crossed_groups = {}
    camera_confidence = 0.0
    
    # Get video info first to calculate total frames
    temp_cap = cv2.VideoCapture(video_path)
    if not temp_cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        processing_video = False
        return
        
    total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = temp_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_cap.release()
    
    logger.info(f"Video info: {total_frames} frames, {video_fps} FPS, {video_width}x{video_height} resolution")
    
    # Adjust line coordinates based on video dimensions if needed
    global line_start, line_end
    # Check if we need to adjust the line coordinates for this video
    if video_width > 0 and video_height > 0:
        # If the line coordinates were set for a different resolution, adjust them
        if line_end[0] != video_width:
            # Calculate the relative position of the line (as percentage of width/height)
            line_y_percent = line_start[1] / 640  # Assuming default height is 640
            
            # Apply the percentage to the actual video dimensions
            new_line_y = int(line_y_percent * video_height)
            
            # Update the line coordinates
            line_start = (0, new_line_y)
            line_end = (video_width, new_line_y)
            
            logger.info(f"Adjusted counting line for video dimensions: {line_start} to {line_end}")
    
    # Initialize tracking variables
    tracker = Tracker()
    previous_positions = {}  # Track previous positions for direction detection
    crossed_lines = {}  # Format: {object_id_line_idx: (timestamp, direction)}
    
    # FPS calculation variables
    fps = 0.0
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_update_interval = 1.0  # Update FPS every 1 second
    FPS_SMOOTHING = 0.9  # Smoothing factor for FPS calculation
    
    # Use memory management settings from environment variables
    logger.info(f"Using inference size: {INFER_IMG_SIZE}px, batch size: {MAX_BATCH_SIZE}")
    
    # Use frame skip from environment variable
    frame_skip = FRAME_SKIP
    logger.info(f"Using frame skip: {frame_skip} (from environment variable)")
    
    # Start video processing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        processing_video = False
        return
        
    processing_video = True
    
    logger.info(f"Started processing video: {video_path} with frame skip: {frame_skip} (processing 1 out of every {frame_skip} frames)")
    logger.info(f"Video opened successfully. Total frames: {total_frames}, FPS: {video_fps}")
    
    # Process video in chunks to handle large files
    frame_count = 0
    while processing_video:
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            logger.info("End of video reached")
            processing_video = False
            break
            
        # Update progress
        frame_count += 1
        processed_frames = frame_count
        processing_progress = min(100, int((frame_count / total_frames) * 100))
        
        # Skip frames for large videos to improve performance
        if frame_count % frame_skip != 0:
            continue
            
        # Debug: Log frame processing
        if frame_count % 100 == 0:
            logger.info(f"Processing frame {frame_count}/{total_frames} ({processing_progress}%)")
            
        # Resize frame if it's too large to avoid CUDA memory issues (from working code)
        orig_height, orig_width = frame.shape[:2]
        if orig_width > 1280 or orig_height > 720:
            frame = cv2.resize(frame, (min(orig_width, 1280), min(orig_height, 720)))
            
        # Periodically clean up memory
        if frame_count % 100 == 0:
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # Run YOLO detection with improved parameters based on working code
            results_cement = model_cement(frame, conf=CONF_THRESHOLD)
            
            # Process detections and prepare for tracking (simplified approach from working code)
            detections = []
            detection_info = {}  # Store class and confidence for each detection
            max_frame_confidence = 0.0
            
            # Extract boxes from results - simplified processing like working code
            for r in results_cement[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                max_frame_confidence = max(max_frame_confidence, score)
                
                if score > CONF_THRESHOLD:
                    # Convert to x, y, w, h format for tracker (same as working code)
                    detection_box = [int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)]
                    detections.append(detection_box)
                    
                    # Store detection info for overlay display
                    detection_key = f"{int(x1)}_{int(y1)}_{int(x2)-int(x1)}_{int(y2)-int(y1)}"
                    
                    # Get actual class name from YOLO model
                    try:
                        class_name = model_cement.names[int(class_id)]
                    except (KeyError, IndexError):
                        class_name = f"Class_{int(class_id)}"  # Fallback if class name not found
                    
                    detection_info[detection_key] = {'class': class_name, 'confidence': score}
            
            # Update tracker with new detections (same as working code)
            tracked_objects = tracker.update(detections)
            
            # Update camera confidence
            camera_confidence = max_frame_confidence
            
            # Process tracked objects for multi-line counting
            for obj in tracked_objects:
                x, y, w, h, object_id = obj
                
                # Find matching detection info for this tracked object
                detection_key = f"{x}_{y}_{w}_{h}"
                class_name = "Unknown"
                confidence = 0.0
                if detection_key in detection_info:
                    class_name = detection_info[detection_key]['class']
                    confidence = detection_info[detection_key]['confidence']
                else:
                    # Try to find the closest matching detection
                    for key, info in detection_info.items():
                        key_parts = key.split('_')
                        if len(key_parts) == 4:
                            kx, ky, kw, kh = map(int, key_parts)
                            # Check if this detection is close to the tracked object
                            if abs(x - kx) < 20 and abs(y - ky) < 20:
                                class_name = info['class']
                                confidence = info['confidence']
                                break
                
                # Calculate center of the box
                cx = x + w // 2
                cy = y + h // 2
                center = (cx, cy)
                
                # Check for line crossings if we have previous position
                if object_id in previous_positions:
                    prev_position = previous_positions[object_id]
                    
                    # Check each counting line for crossings
                    for i, (line_start, line_end) in enumerate(lines):
                        line_color = line_colors[i] if i < len(line_colors) else (0, 0, 255)
                        line_direction = line_directions[i] if i < len(line_directions) else "right"
                        
                        # Check if object crossed this line
                        prev_side = is_crossing_line(prev_position, (line_start, line_end))
                        current_side = is_crossing_line(center, (line_start, line_end))
                        
                        if prev_side != current_side:
                            # Object crossed the line
                            movement_direction = calculate_direction(prev_position, center, line_start, line_end)
                            
                            # Smart deduplication: use color-direction group key
                            group_key = get_color_direction_key(line_color, line_direction)
                            object_group_key = f"{object_id}_{group_key}"
                            
                            current_time = time.time()
                            
                            # Check if this object already crossed a line with the same color-direction in the last 1.5 seconds
                            should_count = True
                            if object_group_key in crossed_groups:
                                last_crossing_time, last_direction = crossed_groups[object_group_key]
                                time_since_last = current_time - last_crossing_time
                                
                                # Don't count if:
                                # 1. Same direction within 1.5 seconds (prevents double counting)
                                # 2. Different direction but within 0.5 seconds (prevents oscillation)
                                if (last_direction == movement_direction and time_since_last < 1.5) or \
                                   (last_direction != movement_direction and time_since_last < 0.5):
                                    should_count = False
                            
                            # Count based on direction and deduplication logic
                            count_this = False
                            if should_count:
                                # Check if movement direction matches the expected counting direction
                                if ((line_direction == "right" and movement_direction == "IN") or
                                    (line_direction == "left" and movement_direction == "OUT") or
                                    (line_direction == "up" and movement_direction == "OUT") or
                                    (line_direction == "down" and movement_direction == "IN")):
                                    count_this = True
                            
                            if count_this and counting_active:
                                bag_count += 1
                                crossed_groups[object_group_key] = (current_time, movement_direction)
                                logger.info(f"[UPLOAD] ✅ Bag counted at line {i+1} (Color: {line_color}, Direction: {line_direction})! Total: {bag_count}")
                    
                # Draw bounding box and ID (simplified like working code)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Update previous position for next frame
                previous_positions[object_id] = center
            
            # Draw all counting lines with colors and directions
            for i, (line_start, line_end) in enumerate(lines):
                line_color = line_colors[i] if i < len(line_colors) else (0, 0, 255)
                line_direction = line_directions[i] if i < len(line_directions) else "right"
                
                # Draw the line
                cv2.line(frame, line_start, line_end, line_color, 3)
                
                # Add line number
                mid_point = ((line_start[0] + line_end[0]) // 2, (line_start[1] + line_end[1]) // 2)
                cv2.putText(frame, f'L{i+1}', (mid_point[0] - 10, mid_point[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)
                
                # Add direction arrow
                if line_direction == "right":
                    arrow_end = (line_end[0] + 20, line_end[1])
                    cv2.arrowedLine(frame, line_end, arrow_end, line_color, 2)
                elif line_direction == "left":
                    arrow_end = (line_start[0] - 20, line_start[1])
                    cv2.arrowedLine(frame, line_start, arrow_end, line_color, 2)
                elif line_direction == "down":
                    arrow_end = (line_end[0], line_end[1] + 20)
                    cv2.arrowedLine(frame, line_end, arrow_end, line_color, 2)
                elif line_direction == "up":
                    arrow_end = (line_start[0], line_start[1] - 20)
                    cv2.arrowedLine(frame, line_start, arrow_end, line_color, 2)
            
            # Periodic cleanup of old tracking data
            current_time = time.time()
            global last_cleanup_time
            if current_time - last_cleanup_time > 20:  # Cleanup every 20 seconds
                # Remove old entries from crossed_groups (older than 10 seconds)
                old_keys = [k for k, (timestamp, _) in crossed_groups.items() 
                           if current_time - timestamp > 10]
                for key in old_keys:
                    del crossed_groups[key]
                
                # Remove old entries from previous_positions for objects not seen recently
                # (This is handled by the tracker, but we can clean up manually if needed)
                
                last_cleanup_time = current_time
                
                # Clear GPU cache if using CUDA
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Add text overlay with bag count, FPS, and camera info
            cv2.putText(frame, f'Bag Count: {bag_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Upload Processing', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Confidence: {camera_confidence:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Multi-line counting visualization is already drawn above
            
            # Calculate FPS
            fps_frame_count += 1
            elapsed_time = time.time() - fps_start_time
            
            if elapsed_time > fps_update_interval:
                current_fps = fps_frame_count / elapsed_time
                if fps == 0.0:  # First calculation
                    fps = current_fps
                else:
                    fps = FPS_SMOOTHING * fps + (1.0 - FPS_SMOOTHING) * current_fps
                
                # Reset counters
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Add text overlays
            font_scale = 0.6
            thickness = 2
            
            # Draw Bag Count in red at the top left
            cv2.putText(frame, f"Bag Count: {bag_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (0, 0, 255), thickness)
            
            # Draw FPS in yellow below Bag Count
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (0, 255, 255), thickness)
            
            # Encode frame for streaming
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                latest_frame = jpeg.tobytes()
                # Debug: Log frame encoding success
                if frame_count % 100 == 0:
                    logger.info(f"Frame {frame_count} encoded successfully, size: {len(latest_frame)} bytes")
            else:
                logger.error(f"Failed to encode frame {frame_count}")
            
            # Adaptive sleep to maintain consistent frame rate
            frame_time = time.time() - start_time
            if frame_time < 0.01:  # If processing took less than 10ms
                time.sleep(0.005)  # Sleep for 5ms
                
            # For very large videos, periodically release and reopen the video to prevent memory issues
            if total_frames > 10000 and frame_count % 1000 == 0:
                current_pos = frame_count
                cap.release()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Clean up
    cap.release()
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"Video processing completed. Processed {processed_frames} frames with {bag_count} bags counted.")

# Flask routes
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_path, processing_video
    
    # Stop any current processing
    processing_video = False
    time.sleep(0.5)  # Give time for processing to stop
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        video_path = filepath
        
        # Start processing in a separate thread
        threading.Thread(target=process_video, daemon=True).start()
        
        return jsonify({'success': True, 'filename': filename}), 200
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            else:
                # If no frame is available, send a blank frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                _, jpeg = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)  # ~30 FPS
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_bag_count')
def get_bag_count():
    global bag_count, processing_progress, total_frames, processed_frames
    return jsonify({
        'bag_count': bag_count,
        'progress': processing_progress,
        'total_frames': total_frames,
        'processed_frames': processed_frames,
        'processing': processing_video
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8003))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
