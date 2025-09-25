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

# Determine which .env file to use
env_file = '.env'  # Default
if len(sys.argv) > 1:
    env_file = sys.argv[1]

# Load environment variables from specified .env file
print(f"Loading environment from: {env_file}")
load_dotenv(dotenv_path=env_file)

# ✅ Models
CEMENT_MODEL_PATH = os.getenv("CEMENT_MODEL_PATH")
FLEET_MODEL_PATH = os.getenv("FLEET_MODEL_PATH")

RTSP_URL = os.getenv("RTSP_URL")
CAMERA_ID = os.getenv("CAMERA_ID")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.2))

# Parse line coordinates from env (format: x1,y1,x2,y2)
line_coords_str = os.getenv("LINE_COORDS", "0,300,640,300")
try:
    x1, y1, x2, y2 = map(int, line_coords_str.split(","))
except Exception as e:
    raise ValueError(f"Invalid LINE_COORDS in .env: {line_coords_str}")

bag_count = 0
object_tracks = {}
line_y = None
y_positions = []

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
FPS_SMOOTHING = 0.9  # Smoothing factor for FPS calculation (0.9 = 90% previous value, 10% new value)

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

# ✅ Load models separately
# Load models with error handling
try:
    if not CEMENT_MODEL_PATH or not os.path.exists(CEMENT_MODEL_PATH):
        logger.error(f"CEMENT_MODEL_PATH is missing or invalid: {CEMENT_MODEL_PATH}")
        raise FileNotFoundError(f"CEMENT_MODEL_PATH is missing or invalid: {CEMENT_MODEL_PATH}")
    # Load model with memory-efficient settings
    model_cement = YOLO(CEMENT_MODEL_PATH)
    model_cement.to(device)
    # Force garbage collection after model loading
    import gc
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
    import gc
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
except Exception as e:
    logger.error(f"Failed to load fleet model: {e}")
    raise

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Cement Bag Detection Flask App!"


@app.route('/bag_count')
def get_bag_count():
    global bag_count
    return jsonify({'bag_count': bag_count})


@app.route('/confidence')
def get_confidence():
    global camera_confidence
    return jsonify({'confidence': camera_confidence})


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            else:
                time.sleep(0.1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    global stream_up
    status = "ok" if stream_up else "stream_error"
    return jsonify({"status": status, "camera_id": CAMERA_ID})


@app.route('/counting')
def counting():
    global counting_active, bag_count
    # Toggle counting with ?active=true/false
    active = request.args.get('active')
    if active is not None:
        counting_active = active.lower() == 'true'
    return jsonify({
        'counting_active': counting_active,
        'bag_count': bag_count
    })


@app.route('/performance')
def performance():
    # Add timestamp to performance stats
    stats = performance_stats.copy()
    stats['timestamp'] = time.time()
    return jsonify(stats)


# --- Threaded Video Capture ---
class VideoCaptureThread:
    def __init__(self, src, width=None, height=None):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
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
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.running = False
        self.cap.release()


# --- Optimized Detection Loop ---
FRAME_SKIP = 3  # Only run detection every 3rd frame to reduce GPU load
INFER_IMG_SIZE = 320  # Lower inference size for speed and reduced memory usage

# Memory management settings
MAX_BATCH_SIZE = 1  # Process one frame at a time
TORCH_DEVICE_MEM_FRACTION = 0.7  # Limit GPU memory usage

# FPS calculation variables
fps_start_time = 0
fps_frame_count = 0
fps_update_interval = 1.0  # Update FPS every 1 second

def detection_loop():
    global bag_count, object_tracks, line_y, y_positions, camera_confidence, latest_frame, stream_up, counting_active

    # Use threaded video capture
    cap = VideoCaptureThread(RTSP_URL, width=INFER_IMG_SIZE, height=INFER_IMG_SIZE)
    stream_up = True
    id_counter = 0
    frame_count = 0
    last_results_cement = None
    last_results_fleet = None
    last_frame_clean = None # Initialize last_frame_clean
    
    # FPS calculation variables
    fps = 0.0
    fps_start_time = time.time()
    fps_frame_count = 0
    
    # Memory management
    import gc
    memory_check_counter = 0

    while True:
        start_time = time.time()
        try:
            frame = cap.read()
        except Exception as e:
            logger.warning(f"⚠️ Video capture error: {e}")
            stream_up = False
            time.sleep(1)
            continue
        frame_count += 1
        if frame is None:
            continue
        # Only run detection every N frames
        if frame_count % FRAME_SKIP == 0:
            try:
                # Memory management - periodically clean up
                memory_check_counter += 1
                if memory_check_counter >= 10:  # Check every 10 detection cycles
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    memory_check_counter = 0
                
                # Run YOLO with memory-efficient settings
                last_results_cement = model_cement(frame, imgsz=INFER_IMG_SIZE, batch=MAX_BATCH_SIZE)
                
                # Only run second model if first one succeeded
                last_results_fleet = model_fleet(frame, imgsz=INFER_IMG_SIZE, batch=MAX_BATCH_SIZE)
                
                # Store a clean copy of the frame (before overlays)
                last_frame_clean = frame.copy()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # Handle OOM error gracefully
                    logger.error(f"CUDA out of memory error: {e}")
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    # Skip this frame
                    continue
                else:
                    # Re-raise if it's not an OOM error
                    raise
        else:
            # Use last detection results, just update the frame
            frame = last_frame_clean.copy() if last_frame_clean is not None else frame
        results_cement = last_results_cement
        results_fleet = last_results_fleet
        new_tracks = {}
        max_frame_confidence = 0.0

        # ---------- Cement bag logic ----------
        if results_cement is not None and line_y is None:
            for box in results_cement[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cy = int((y1 + y2) / 2)
                y_positions.append(cy)
            if len(y_positions) >= 50:
                y_positions.sort()
                median_y = y_positions[len(y_positions) // 2]
                line_y = median_y - 20
                logger.info(f"✅ Dynamic line_y set to: {line_y} for camera {CAMERA_ID}")

        if results_cement is not None:
            for box in results_cement[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue
                if conf > max_frame_confidence:
                    max_frame_confidence = conf
                assigned_id = None
                for obj_id, (prev_cx, prev_cy) in object_tracks.items():
                    if abs(cx - prev_cx) < 80 and abs(cy - prev_cy) < 80:
                        assigned_id = obj_id
                        break
                if assigned_id is None:
                    id_counter += 1
                    assigned_id = id_counter
                new_tracks[assigned_id] = (cx, cy)
                if assigned_id in object_tracks and line_y is not None:
                    prev_cx, prev_cy = object_tracks[assigned_id]
                    if prev_cy < line_y and cy >= line_y:
                        if counting_active:
                            bag_count += 1
                        logger.info(f"[{CAMERA_ID}] ✅ Unloaded! Total: {bag_count}")
                    elif prev_cy > line_y and cy <= line_y:
                        if counting_active:
                            bag_count -= 1
                        logger.info(f"[{CAMERA_ID}] ⬆️ Loaded! Total: {bag_count}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'Bag ID: {assigned_id}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        object_tracks = new_tracks
        camera_confidence = max_frame_confidence
        if line_y is not None:
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
        # ---------- Final frame ----------
        end_time = time.time()
        frame_time = end_time - start_time
        
        # Update FPS calculation
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        
        # Update FPS every second for more stable reading
        if elapsed_time > fps_update_interval:
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
        
        # Draw Bag Count in red at the top left
        cv2.putText(frame, f"Bag Count: {bag_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Draw FPS in yellow below Bag Count
        cv2.putText(frame, f"FPS: {fps:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            latest_frame = jpeg.tobytes()
        
        # Performance monitoring
        performance_stats['last_frame_time'] = frame_time
        performance_stats['last_detection_speed'] = fps
        performance_stats['avg_fps'] = fps
        
        # Adaptive sleep to maintain consistent frame rate without overloading CPU
        # Only sleep if processing was very fast
        if frame_time < 0.01:  # If processing took less than 10ms
            time.sleep(0.005)  # Sleep for 5ms

    cap.release()
    cv2.destroyAllWindows()


t = threading.Thread(target=detection_loop)
t.daemon = True
t.start()

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 5002))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)