import cv2
import time
from ultralytics import YOLO
from flask import Flask, jsonify, Response, request
import threading
import os
import sys
from dotenv import load_dotenv
import logging
import torch
import queue
import gc
import numpy as np
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

def get_color_group_key(color):
    """Create a unique key for color group (ignoring direction)"""
    return f"{color[0]}_{color[1]}_{color[2]}"

def get_color_direction_key(color, direction):
    """Create a unique key for color-direction combination (legacy function)"""
    return f"{color[0]}_{color[1]}_{color[2]}_{direction}"

# --- Optimized Detection Loop Constants ---
FPS_SMOOTHING = 0.9  # Exponential moving average factor for FPS calculation
FPS_UPDATE_INTERVAL = 1.0  # Update FPS every 1 second

# Load environment variables from .env_transship file
env_file = '.env_transship'
print(f"Loading environment from: {env_file}")
load_dotenv(dotenv_path=env_file)

# Models
CEMENT_MODEL_PATH = os.getenv("CEMENT_MODEL_PATH")
FLEET_MODEL_PATH = os.getenv("FLEET_MODEL_PATH")

RTSP_URL = os.getenv("RTSP_URL")
CAMERA_ID = os.getenv("CAMERA_ID", "401")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.2"))

# Frame skip configuration from environment
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))  # Default to every 2nd frame
print(f"Using frame skip: {FRAME_SKIP} (processing 1 out of every {FRAME_SKIP} frames)")

# Cooldown period configuration from environment
COUNT_COOLDOWN = float(os.getenv("COUNT_COOLDOWN", "1.5"))  # Default to 1.5 seconds
print(f"Using count cooldown: {COUNT_COOLDOWN}s (prevents oscillation counting)")

# Parse multiple line coordinates from env (format: x1,y1,x2,y2;x1,y1,x2,y2;...)
multi_line_coords_str = os.getenv("MULTI_LINE_COORDS", "100,200,540,200;100,300,540,300;100,400,540,400")
try:
    lines = []
    for line_str in multi_line_coords_str.split(";"):
        x1, y1, x2, y2 = map(int, line_str.split(","))
        lines.append(((x1, y1), (x2, y2)))
except Exception as e:
    raise ValueError(f"Invalid MULTI_LINE_COORDS in .env: {multi_line_coords_str}")

# Parse counting directions from env (one per line)
multi_directions_str = os.getenv("MULTI_COUNT_DIRECTIONS", "left;right;down")
try:
    count_directions = []
    for direction in multi_directions_str.split(";"):
        direction = direction.strip().lower()
        if direction not in ["left", "right", "up", "down"]:
            print(f"Invalid direction: {direction}. Using 'left' as default.")
            direction = "left"
        count_directions.append(direction)
except Exception as e:
    print(f"Invalid MULTI_COUNT_DIRECTIONS in .env: {multi_directions_str}")
    count_directions = ["left"]

# Ensure we have directions for all lines
while len(count_directions) < len(lines):
    count_directions.append("left")  # Default to left for missing directions

# Parse line colors from env (format: b,g,r;b,g,r;...)
line_colors_str = os.getenv("LINE_COLORS", "0,0,255;0,255,0;255,0,0")
try:
    line_colors = []
    for color_str in line_colors_str.split(";"):
        b, g, r = map(int, color_str.split(","))
        line_colors.append((b, g, r))
except Exception as e:
    print(f"Invalid LINE_COLORS in .env: {line_colors_str}. Using default colors.")
    line_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Default to red, green, blue

# Ensure we have enough colors for all lines
while len(line_colors) < len(lines):
    line_colors.append((0, 0, 255))  # Default to red for any missing colors

transship_count = 0
object_tracks = {}
# Track which objects have crossed which color-direction groups to prevent double counting
crossed_groups = {}  # Format: {object_id_color_direction: timestamp}

print(f"Loaded {len(lines)} counting lines with colors and directions:")
for i, (line, color, direction) in enumerate(zip(lines, line_colors, count_directions)):
    print(f"  Line {i+1}: {line[0]} to {line[1]}, Color: {color}, Direction: {direction}")
camera_confidence = 0.0
latest_frame = None
stream_up = False
counting_active = True

# Performance monitoring variables
performance_stats = {
    'last_frame_time': 0.0,
    'last_detection_speed': 0.0,
    'avg_fps': 0.0
}
FPS_SMOOTHING = 0.9  # Smoothing factor for FPS calculation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

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

# Load models with error handling
try:
    if not CEMENT_MODEL_PATH or not os.path.exists(CEMENT_MODEL_PATH):
        logger.error(f"CEMENT_MODEL_PATH is missing or invalid: {CEMENT_MODEL_PATH}")
        raise FileNotFoundError(f"CEMENT_MODEL_PATH is missing or invalid: {CEMENT_MODEL_PATH}")
    # Load model with memory-efficient settings
    model_cement = YOLO(CEMENT_MODEL_PATH)
    model_cement.to(device)
    # Force garbage collection after model loading
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
except Exception as e:
    logger.error(f"Failed to load cement model: {e}")
    raise

try:
    if not FLEET_MODEL_PATH or not os.path.exists(FLEET_MODEL_PATH):
        logger.error(f"FLEET_MODEL_PATH is missing or invalid: {FLEET_MODEL_PATH}")
        raise FileNotFoundError(f"FLEET_MODEL_PATH is missing or invalid: {FLEET_MODEL_PATH}")
    # Load model with memory-efficient settings
    model_fleet = YOLO(FLEET_MODEL_PATH)
    model_fleet.to(device)
    # Force garbage collection after model loading
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
except Exception as e:
    logger.error(f"Failed to load fleet model: {e}")
    raise

app = Flask(__name__)

class VideoCaptureThread:
    def __init__(self, src, width=None, height=None):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        # Set buffer size to 1 to reduce latency and prevent buildup
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Set FPS to reasonable value to prevent overload
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = queue.Queue(maxsize=2)  # Slightly larger buffer for stability
        self.running = True
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            # Always drop old frames to prevent buffer buildup
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            try:
                self.q.put(frame, timeout=0.1)
            except queue.Full:
                # Skip frame if queue is full to prevent blocking
                pass

    def read(self):
        try:
            return True, self.q.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.cap.release()
        # Clear remaining frames from queue
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Return a blank frame if no frame is available
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "No video feed available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            else:
                # If no frame is available, yield an empty image
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
            time.sleep(0.03)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global stream_up, camera_confidence
    return jsonify({
        'stream_up': stream_up,
        'camera_id': CAMERA_ID,
        'confidence': float(camera_confidence)
    })

@app.route('/')
def home():
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Transshipment Camera {CAMERA_ID}</title>
        <meta http-equiv="refresh" content="1">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }}
            h1 {{ margin-bottom: 10px; }}
            .video {{ margin-bottom: 20px; }}
            .count {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
            .controls {{ margin: 15px 0; }}
            button {{ padding: 8px 16px; margin: 0 5px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <h1>Transshipment Camera {CAMERA_ID}</h1>
        <div class="video">
            <img src="/video_feed" width="100%" />
        </div>
        <div class="count">Transshipment Count: {transship_count}</div>
        <div class="controls">
            <a href="/counting?active=true"><button>Start Counting</button></a>
            <a href="/counting?active=false"><button>Stop Counting</button></a>
            <a href="/reset_count"><button>Reset Count</button></a>
        </div>
        <div>Status: {'Active' if counting_active else 'Paused'}</div>
    </body>
    </html>
    '''

@app.route('/get_count')
def get_bag_count():
    global transship_count
    return jsonify({'transship_count': transship_count})

@app.route('/get_confidence')
def get_confidence():
    global camera_confidence
    return jsonify({'confidence': camera_confidence})

@app.route('/health')
def health():
    global stream_up
    status_code = 200 if stream_up else 503
    return jsonify({'status': 'ok' if stream_up else 'error'}), status_code

@app.route('/counting')
def counting():
    global counting_active, transship_count
    
    # Handle toggling counting on/off
    active_param = request.args.get('active')
    if active_param is not None:
        counting_active = active_param.lower() == 'true'
    
    return jsonify({
        'counting_active': counting_active,
        'transship_count': transship_count,
        'camera_id': CAMERA_ID
    })

@app.route('/reset_count')
def reset_count():
    global transship_count, bag_line_crossings
    transship_count = 0
    bag_line_crossings = {}
    return jsonify({
        'transship_count': transship_count,
        'camera_id': CAMERA_ID
    })

@app.route('/performance')
def performance():
    global performance_stats
    return jsonify(performance_stats)


# --- Manual count update endpoints (for admin) ---
@app.route('/update_count', methods=['POST'])
def update_count():
    """Set or adjust transshipment count manually.
    Accepts JSON: {"count": <int>} to set absolute, or {"value": <int>} alias, or {"delta": 1|-1} to adjust.
    """
    try:
        global transship_count
        data = request.get_json(silent=True) or {}
        if 'count' in data or 'value' in data:
            new_value = data.get('count', data.get('value', 0))
            transship_count = max(0, int(new_value))
            return jsonify({'success': True, 'transship_count': transship_count})
        if 'delta' in data:
            transship_count = max(0, transship_count + int(data.get('delta', 0)))
            return jsonify({'success': True, 'transship_count': transship_count})
        return jsonify({'success': False, 'error': 'Provide count or delta'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def detection_loop():
    global transship_count, camera_confidence, latest_frame, stream_up, counting_active
    
    # Use threaded video capture
    cap = VideoCaptureThread(RTSP_URL, width=640, height=480)
    stream_up = True
    frame_count = 0
    last_results_cement = None
    last_results_fleet = None
    
    # Initialize tracking variables using the Tracker class
    tracker = Tracker()
    previous_positions = {}  # Track previous positions for direction detection
    object_tracks = {}  # Track objects by ID
    bag_line_crossings = {}  # Track which bags have been counted
    
    # FPS calculation variables
    fps = 0.0
    fps_start_time = time.time()
    fps_frame_count = 0

    # Load YOLO models
    try:
        model_cement = YOLO(CEMENT_MODEL_PATH)
        logger.info(f"[{CAMERA_ID}] Models loaded successfully")
    except Exception as e:
        logger.error(f"[{CAMERA_ID}] Error loading models: {e}")
        return
        
    # Frame skip counter for detection
    frame_count = 0

    # Memory cleanup counter
    cleanup_counter = 0
    
    while True:
        start_time = time.time()
        # Get frame from camera with improved error handling
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning(f"[{CAMERA_ID}] Failed to get frame from camera")
            stream_up = False
            time.sleep(0.5)
            continue
            
        stream_up = True
        frame_count += 1
        
        # Only run detection every N frames
        run_detection = frame_count % FRAME_SKIP == 0  # Run every FRAME_SKIP frame
        
        if run_detection:
            try:
                # Clear CUDA cache to prevent memory leaks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Detect cement bags
                results_cement = model_cement(frame, conf=CONF_THRESHOLD, verbose=False)
                
                # Collect detections for tracking
                detections = []
                detection_info = {}  # Store class and confidence for each detection
                max_frame_confidence = 0.0
                
                # Process detections
                for r in results_cement[0].boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    max_frame_confidence = max(max_frame_confidence, score)
                    
                    if score > CONF_THRESHOLD:
                        # Convert to x, y, w, h format for tracker
                        detection_box = [int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)]
                        detections.append(detection_box)
                        
                        # Store detection info for later use
                        detection_key = f"{int(x1)}_{int(y1)}_{int(x2)-int(x1)}_{int(y2)-int(y1)}"
                        
                        # Get actual class name from YOLO model (improved reliability)
                        try:
                            class_name = model_cement.names[int(class_id)]
                        except (KeyError, IndexError):
                            class_name = f"Class_{int(class_id)}"  # Fallback if class name not found
                        
                        detection_info[detection_key] = {'class': class_name, 'confidence': score}
                        
                        # Draw detection box (before tracking) for debugging
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                
                # Update tracker with new detections
                tracked_objects = tracker.update(detections)
                
                # Process tracked objects for line crossing
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
                        # Try to find the closest matching detection (improved reliability)
                        for key, info in detection_info.items():
                            key_parts = key.split('_')
                            if len(key_parts) == 4:
                                kx, ky, kw, kh = map(int, key_parts)
                                # Check if this detection is close to the tracked object
                                if abs(x - kx) < 20 and abs(y - ky) < 20:
                                    class_name = info['class']
                                    confidence = info['confidence']
                                    break
                    
                    # Get previous position for this object
                    prev_position = previous_positions.get(object_id)
                    center = ((x + w // 2), (y + h // 2))
                    previous_positions[object_id] = center
                    
                    # Skip if no previous position (first detection)
                    if prev_position is None:
                        continue
                    
                    # Check each line for crossing
                    for i, (line_start, line_end) in enumerate(lines):
                        line_color = line_colors[i]
                        line_direction = count_directions[i]
                        
                        # Check if object crossed this line
                        prev_side = is_crossing_line(prev_position, (line_start, line_end))
                        current_side = is_crossing_line(center, (line_start, line_end))
                        
                        if prev_side != current_side:
                            # Object crossed the line, check direction
                            movement_direction = calculate_direction(prev_position, center, line_start, line_end)
                            
                            # Create color group key for deduplication (ignore direction)
                            color_group_key = get_color_group_key(line_color)
                            object_group_key = f"{object_id}_{color_group_key}"
                            
                            # Check if this object has been counted in this color-direction group recently
                            current_time = time.time()
                            count_this = True
                            if object_group_key in crossed_groups:
                                last_cross_time, last_direction = crossed_groups[object_group_key]
                                time_since_last = current_time - last_cross_time
                                # Don't count if it's been less than cooldown period since last crossing
                                # AND the direction is the same (prevent oscillation)
                                if (time_since_last < COUNT_COOLDOWN) and (movement_direction == last_direction):
                                    count_this = False
                                    logger.debug(f"[{CAMERA_ID}] Skipping duplicate count for object {object_id} in color group {color_group_key} (time: {time_since_last:.2f}s)")
                            
                            # For transhipment: always count negatively for any bag movement, regardless of direction
                            if count_this and counting_active:
                                transship_count -= 1  # Always decrement for transhipment
                                crossed_groups[object_group_key] = (current_time, movement_direction)
                                logger.info(f"[{CAMERA_ID}] ⬇️ Transshipment (negative count) at line {i+1} (Color Group: {color_group_key}, Movement: {movement_direction})! Total: {transship_count}")
                    
                    # Draw bounding box with class name and confidence
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                camera_confidence = max_frame_confidence
                
            except Exception as e:
                logger.error(f"[{CAMERA_ID}] Error in detection: {e}")
                import traceback
                traceback.print_exc()
                
        # Draw the counting lines with their respective colors and direction arrows (do this for every frame)
        for i, (line_start, line_end) in enumerate(lines):
            color = line_colors[i] if i < len(line_colors) else (0, 0, 255)  # Default to red if no color specified
            direction = count_directions[i] if i < len(count_directions) else "left"
            
            # Draw the line
            cv2.line(frame, line_start, line_end, color, 1)
            
            # Add line number
            mid_x = (line_start[0] + line_end[0]) // 2
            mid_y = (line_start[1] + line_end[1]) // 2
            cv2.putText(frame, f"{i+1}", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw direction arrow
            arrow_length = 30
            arrow_start = (mid_x, mid_y)
            
            if direction == "left":
                arrow_end = (mid_x - arrow_length, mid_y)
            elif direction == "right":
                arrow_end = (mid_x + arrow_length, mid_y)
            elif direction == "up":
                arrow_end = (mid_x, mid_y - arrow_length)
            elif direction == "down":
                arrow_end = (mid_x, mid_y + arrow_length)
            else:
                arrow_end = (mid_x + arrow_length, mid_y)  # Default to right
                
            cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 1)
        
        font_scale = 0.6
        thickness = 2

        # Draw Transshipment Count in red at the top left
        cv2.putText(frame, f"Transshipment Count: {transship_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                    thickness)

        # Draw FPS in yellow below Transshipment Count
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

        # Draw Camera ID in green below FPS
        cv2.putText(frame, f"Camera: {CAMERA_ID}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                    thickness)
        
        # Update FPS calculation
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        
        # Update FPS every second for more stable reading
        if elapsed_time > FPS_UPDATE_INTERVAL:
            # Calculate actual FPS based on frames processed in the interval
            current_fps = fps_frame_count / elapsed_time
            # Apply exponential moving average for smoother FPS display
            if fps == 0.0:  # First calculation
                fps = current_fps
            else:
                fps = FPS_SMOOTHING * fps + (1.0 - FPS_SMOOTHING) * current_fps
            
            # Reset counters
            fps_start_time = time.time()
            fps_frame_count = 0
        
        # Encode frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            latest_frame = jpeg.tobytes()
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            latest_frame = jpeg.tobytes()
        
        # Periodic memory cleanup to prevent gradual slowdown
        cleanup_counter += 1
        if cleanup_counter % 300 == 0:  # Every 300 frames (~20 seconds at 15 FPS)
            # Clean up old entries from crossed_groups (older than 10 seconds)
            current_time = time.time()
            keys_to_remove = []
            for key, (timestamp, direction) in crossed_groups.items():
                if current_time - timestamp > 10.0:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del crossed_groups[key]
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"[{CAMERA_ID}] Memory cleanup: removed {len(keys_to_remove)} old tracking entries")
        
        # Calculate frame processing time
        frame_time = time.time() - start_time
        
        # Update performance statistics
        performance_stats['last_frame_time'] = frame_time
        performance_stats['last_detection_speed'] = fps
        performance_stats['avg_fps'] = fps
        
        # Adaptive sleep to maintain consistent frame rate without overloading CPU
        if frame_time < 0.01:  # If processing took less than 10ms
            time.sleep(0.005)  # Sleep for 5ms

    cap.release()

# Start detection thread in a daemon thread
t = threading.Thread(target=detection_loop)
t.daemon = True
t.start()
logger.info(f"[{CAMERA_ID}] Detection thread started")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 5003))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
