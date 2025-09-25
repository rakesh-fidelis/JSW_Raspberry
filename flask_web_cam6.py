import cv2
import time
from ultralytics import YOLO
from flask import Flask, jsonify, Response, request, make_response
import threading
import os
import sys
from dotenv import load_dotenv, dotenv_values
import logging
import torch
import queue
import gc
import numpy as np
from tracker import Tracker
import requests
import json
from dotenv import load_dotenv
# Load env file explicitly (default to .env_cam6). Override via ENV_FILE env var.
ENV_FILE = os.getenv('ENV_FILE', '.env_cam6')
load_dotenv(ENV_FILE)
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import render_template


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


# Determine which .env file to use
env_file = '.env_cam6'  # Default for camera 3

# Load environment variables from specified .env file
print(f"Loading environment from: {env_file}")
load_dotenv(dotenv_path=env_file)

# ✅ Models
CEMENT_MODEL_PATH = os.getenv("CEMENT_MODEL_PATH")
FLEET_MODEL_PATH = os.getenv("FLEET_MODEL_PATH")

RTSP_URL = os.getenv("RTSP_URL")
CAMERA_ID = os.getenv("CAMERA_ID")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.2))

# Parse multiple line coordinates from env (format: x1,y1,x2,y2;x1,y1,x2,y2;...)
multi_line_coords_str = os.getenv("MULTI_LINE_COORDS", "0,200,640,200")
try:
    lines = []
    for line_str in multi_line_coords_str.split(";"):
        x1, y1, x2, y2 = map(int, line_str.split(","))
        lines.append(((x1, y1), (x2, y2)))
except Exception as e:
    raise ValueError(f"Invalid MULTI_LINE_COORDS in .env: {multi_line_coords_str}")

# Parse counting directions from env (one per line)
multi_directions_str = os.getenv("MULTI_COUNT_DIRECTIONS", "left")
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
line_colors_str = os.getenv("LINE_COLORS", "0,0,255")
try:
    line_colors = []
    for color_str in line_colors_str.split(";"):
        b, g, r = map(int, color_str.split(","))
        line_colors.append((b, g, r))
except Exception as e:
    print(f"Invalid LINE_COLORS in .env: {line_colors_str}. Using default colors.")
    line_colors = [(0, 0, 255)]  # Default to red

# Ensure we have colors for all lines
while len(line_colors) < len(lines):
    line_colors.append((0, 0, 255))  # Default to red for missing colors

# Separate counters for directional tracking
bag_in = 0
bag_out = 0
object_tracks = {}
# Track which objects have crossed which color-direction groups to prevent double counting
crossed_groups = {}  # Format: {object_id_color_direction: timestamp}

# print(f"Loaded {len(lines)} counting lines with colors and directions:")
# for i, (line, color, direction) in enumerate(zip(lines, line_colors, count_directions)):
#     print(f"  Line {i + 1}: {line[0]} to {line[1]}, Color: {color}, Direction: {direction}")

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
    logger.info(
        f'CUDA is available. Using GPU for inference. Free memory: {free_mem / 1024 ** 2:.1f}MB / {total_mem / 1024 ** 2:.1f}MB')
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

# Global variables for camera state
bag_in = 0
bag_out = 0
counting_active = False
camera_confidence = 0.0
latest_frame = None
stream_up = True


@app.route('/')
def home():
    return f"Welcome to the Cement Bag Detection Flask App! Camera ID: {CAMERA_ID}"


@app.route('/bag_count')
def get_bag_count():
    # Return live bag count values from global variables
    global bag_in, bag_out
    return jsonify({
        'bag_in': bag_in,
        'bag_out': bag_out,
        'total': bag_in - bag_out
    })


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


@app.route('/control')
def camera_control_direct():
    """Alternative direct route for camera control panel"""
    try:
        return render_template('camera_control.html')
    except Exception as e:
        logger.error(f"Error rendering camera control template: {str(e)}")
        return f"Error loading control panel: {str(e)}"


@app.route('/cam6/control')
def camera_control_panel():
    """Serve the camera control panel UI"""
    try:
        return render_template('camera_control.html')
    except Exception as e:
        logger.error(f"Error rendering camera control template: {str(e)}")
        return f"Error loading control panel: {str(e)}"


@app.route('/cam6/ui')
def manual_count_ui():
    """Serve the manual bag count adjustment UI"""
    try:
        global bag_in, bag_out, counting_active
        return render_template('manual_count_ui.html', 
                             camera_id=CAMERA_ID,
                             bag_in=bag_in,
                             bag_out=bag_out,
                             counting_active=counting_active)
    except Exception as e:
        logger.error(f"Error rendering manual count UI template: {str(e)}")
        return f"Error loading manual count UI: {str(e)}"


@app.route('/admin')
@app.route('/cam6/admin')
def manual_count_admin():
    """Single-page admin for manual updates across cameras without video feed"""
    try:
        return render_template('manual_count_admin.html')
    except Exception as e:
        logger.error(f"Error rendering manual count admin template: {str(e)}")
        return f"Error loading manual admin UI: {str(e)}"


# Simple UI for quick manual updates (uniform with other cameras)
@app.route('/ui')
def simple_manual_ui():
    try:
        return render_template('manual_count_simple.html')
    except Exception as e:
        return f"Error loading UI: {str(e)}"

@app.route('/update_count', methods=['POST'])
def simple_update_count():
    try:
        global bag_in, bag_out
        data = request.get_json(silent=True) or {}
        if 'in' in data:
            bag_in = max(0, int(data['in']))
        if 'out' in data:
            bag_out = max(0, int(data['out']))
        return jsonify({'success': True, 'bag_in': bag_in, 'bag_out': bag_out})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/cam6/control/<camera_id>/counting', methods=['POST'])
def toggle_camera_counting(camera_id):
    """Enable or disable counting for a specific camera"""
    try:
        data = request.json
        enabled = data.get('enabled', False)
        
        # Log the request
        logger.info(f"Toggle counting request for camera {camera_id}: {enabled}")
        
        # For camera 6 (local camera)
        if camera_id == CAMERA_ID:
            global counting_active
            counting_active = enabled
            logger.info(f"Local camera {CAMERA_ID} counting set to {counting_active}")
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'counting_active': counting_active
            })
            
        # For remote cameras
        camera_info = get_camera_info_by_id(camera_id)
        if not camera_info:
            return jsonify({'success': False, 'error': f'Camera {camera_id} not found'}), 404
            
        # Determine the URL based on camera info
        base_url = camera_info['base_url']
        logical_id = camera_info['logical_id']
        
        # Different path for transshipment
        if logical_id == 'tranship':
            url = f"{base_url}/{logical_id}/counting"
        else:
            url = f"{base_url}/{logical_id}/counting"
            
        # Send request to toggle counting
        response = requests.post(
            url,
            json={'enabled': enabled},
            timeout=2.0
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'counting_active': enabled
            })
        else:
            return jsonify({
                'success': False,
                'camera_id': camera_id,
                'error': f'Failed to update remote camera: {response.status_code}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error toggling counting for camera {camera_id}: {str(e)}")
        return jsonify({
            'success': False,
            'camera_id': camera_id,
            'error': str(e)
        }), 500


@app.route('/cam6/control/<camera_id>/update', methods=['POST'])
def update_camera_count(camera_id):
    """Manually update count for a specific camera"""
    try:
        data = request.json
        
        # Check if this is for transshipment camera
        is_transshipment = False
        camera_info = get_camera_info_by_id(camera_id)
        
        if not camera_info:
            return jsonify({'success': False, 'error': f'Camera {camera_id} not found'}), 404
            
        if camera_info.get('logical_id') == 'tranship':
            is_transshipment = True
        
        # For camera 6 (local camera)
        if camera_id == CAMERA_ID:
            global bag_in, bag_out
            
            if is_transshipment:
                count = data.get('count', 0)
                # For transshipment, we only have a single count
                bag_in = count
                bag_out = 0
                logger.info(f"Local transshipment count updated to {count}")
            else:
                direction = data.get('direction')
                value = data.get('value', 0)
                
                if direction == 'in':
                    bag_in = value
                    logger.info(f"Local camera {CAMERA_ID} in count updated to {value}")
                elif direction == 'out':
                    bag_out = value
                    logger.info(f"Local camera {CAMERA_ID} out count updated to {value}")
                else:
                    return jsonify({'success': False, 'error': 'Invalid direction'}), 400
            
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'bag_in': bag_in,
                'bag_out': bag_out
            })
        
        # For remote cameras
        base_url = camera_info['base_url']
        logical_id = camera_info['logical_id']
        
        if is_transshipment:
            # Handle transshipment count update
            count = data.get('count', 0)
            payload = {'count': count}
        else:
            # Handle regular camera in/out count update
            direction = data.get('direction')
            value = data.get('value', 0)
            if direction not in ['in', 'out']:
                return jsonify({'success': False, 'error': 'Invalid direction'}), 400
            payload = {direction: value}

        # Try with and without the url_path prefix to survive proxy configurations
        urls_to_try = [
            f"{base_url}/{logical_id}/update_count",
            f"{base_url}/update_count"
        ]
        last_status = None
        last_text = None
        for url in urls_to_try:
            try:
                response = requests.post(url, json=payload, timeout=2.0)
                last_status = response.status_code
                try:
                    last_text = response.text
                except Exception:
                    last_text = None
                if response.status_code == 200:
                    return jsonify({'success': True, 'camera_id': camera_id, 'updated': payload, 'url': url})
            except requests.RequestException as _:
                continue
        return jsonify({'success': False, 'camera_id': camera_id, 'error': f'remote status {last_status}', 'detail': last_text}), 502
            
    except Exception as e:
        logger.error(f"Error updating count for camera {camera_id}: {str(e)}")
        return jsonify({
            'success': False,
            'camera_id': camera_id,
            'error': str(e)
        }), 500


@app.route('/counting')
def counting():
    # Return live counting values from global variables
    global bag_in, bag_out, counting_active
    active = request.args.get('active')
    return jsonify({
        'counting_active': counting_active,
        'bag_in': bag_in,
        'bag_out': bag_out,
        'total_count': bag_in - bag_out,
        'camera_id': CAMERA_ID
    })


@app.route('/performance')
def performance():
    # Add timestamp to performance stats
    stats = performance_stats.copy()
    stats['timestamp'] = time.time()
    return jsonify(stats)


# --- Daily Summary API (In-Memory) ---
import requests
from datetime import datetime, timedelta

def load_camera_configs() -> dict:
    """Load camera configs strictly from environment variables and per-camera .env files.
    JSON input is ignored by design per requirements.
    Sources in order of priority per camera:
      1) Admin env: CAM_<ID>_DOMAIN, CAM_<ID>_PATH, CAM_<ID>_LOGICAL_ID, CAM_<ID>_NAME, CAM_<ID>_LOCATION, CAM_<ID>_RTSP_URL
      2) Admin env: CAM_BASE_DOMAIN (domain fallback shared by all)
      3) Per-camera env file (.env_camN or .env_transship): PUBLIC_DOMAIN, PUBLIC_PATH, RTSP_URL, PORT, CAMERA_ID
      4) Derived: url_path from camera id (e.g., 302 -> /cam3)
         Derived: domain from LOCAL_BASE_HOST + :PORT if available
    """

    ids = os.getenv('CAM_IDS', '102,202,302,402,502,602,702,Transshipment').split(',')
    ids = [i.strip() for i in ids if i.strip()]

    result = {}
    base_override = os.getenv('CAM_BASE_DOMAIN', '').strip()
    local_base_host = os.getenv('LOCAL_BASE_HOST', '127.0.0.1').strip()
    for cam_id in ids:
        key = cam_id.replace('-', '_')
        domain = os.getenv(f'CAM_{key}_DOMAIN', '') or base_override
        url_path = os.getenv(f'CAM_{key}_PATH', '')
        logical_id = os.getenv(f'CAM_{key}_LOGICAL_ID', '')
        name = os.getenv(f'CAM_{key}_NAME', f'Camera {cam_id}')
        location = os.getenv(f'CAM_{key}_LOCATION', '')
        rtsp_url = os.getenv(f'CAM_{key}_RTSP_URL', '')
        # If RTSP not provided via admin env, read from the per-camera env file
        if not rtsp_url:
            # Determine expected env filename for this camera id
            if cam_id.isdigit():
                # Map 102->.env_cam1, 202->.env_cam2, etc.
                cam_index = cam_id[0]  # first digit denotes camera number (1..7)
                candidate = f".env_cam{cam_index}"
            else:
                candidate = ".env_transship"
            try:
                if os.path.exists(candidate):
                    values = dotenv_values(candidate)
                    rtsp_url = values.get('RTSP_URL', rtsp_url)
                    # Only use per-camera PUBLIC_* if admin env didn't define
                    if not domain:
                        domain = values.get('PUBLIC_DOMAIN', domain)
                    if not url_path:
                        url_path = values.get('PUBLIC_PATH', url_path)
                    # As a last resort, try deriving domain from port
                    derived_from_port = False
                    if not domain:
                        cam_port = values.get('PORT')
                        if cam_port:
                            domain = f"http://{local_base_host}:{cam_port}"
                            derived_from_port = True
                    # As a last resort for path, prefer root when calling by port; else derive from camera id / filename
                    if not url_path:
                        if derived_from_port:
                            url_path = "/"
                        else:
                            if cam_id.isdigit() and len(cam_id) >= 1:
                                url_path = f"/cam{cam_id[0]}"
                            else:
                                url_path = "/tranship"
            except Exception:
                pass
        # Derive url_path by ID if still missing (e.g., 302 -> /cam3; Transshipment -> /tranship)
        if not url_path:
            if cam_id.isdigit() and len(cam_id) >= 1:
                url_path = f"/cam{cam_id[0]}"
            else:
                url_path = "/tranship"
        # Normalize domain (no trailing slash). If missing, will fallback to request base later
        if domain:
            domain = domain.rstrip('/')
        result[cam_id] = {
            'domain': domain,
            'url_path': url_path,
            'name': name,
            'location': location,
            'logical_id': logical_id or (url_path.strip('/')),
            'rtsp_url': rtsp_url
        }

    return result

# Camera configuration for daily summary - loaded from env with fallback
CAMERA_CONFIGS = load_camera_configs()

# BASE_URL will be determined dynamically from the request

def get_camera_info_by_id(camera_id):
    """Get camera information from CAMERA_CONFIGS for a given ID.
    If domain is not configured, default to current request's scheme://host.
    """
    # Normalize transshipment id variants
    normalized_id = camera_id
    if camera_id.lower() in ['transshipment', 'transship', 'tranship']:
        normalized_id = 'Transshipment'

    cfg = CAMERA_CONFIGS.get(normalized_id)
    if not cfg:
        return None
    logical_id = cfg.get('logical_id') or cfg.get('url_path', '').strip('/')
    base_url = cfg.get('domain') or ''
    if not base_url:
        try:
            base_url = f"{request.scheme}://{request.host}"
        except Exception:
            base_url = ''
    return {
        'id': normalized_id,
        'logical_id': logical_id,
        'base_url': base_url
    }


def normalize_camera_id(camera_id: str) -> str:
    """Normalize incoming camera identifier to match keys used in CAMERA_CONFIGS."""
    if not camera_id:
        return camera_id
    lower = camera_id.lower()
    if lower in ['transshipment', 'transship', 'tranship']:
        return 'Transshipment'
    return camera_id


def get_camera_data(camera_id, base_url):
    """Fetch current bag count data from a specific camera with optimized performance"""
    camera_data = {
        "id": camera_id,
        "in": 0,
        "out": 0,
        "total": 0,
        "online": False,
        "confidence": 0.0,
        "counting_active": False
    }
    try:
        # Debug logging for camera 6
        if camera_id == '602':
            logger.info(f"Processing Camera 6: camera_id='{camera_id}', CAMERA_ID='{CAMERA_ID}', match={camera_id == CAMERA_ID}")
        
        # If it's this camera (602), return local data immediately
        if camera_id == '602' or camera_id == CAMERA_ID:
            logger.info(f"Returning local data for camera {camera_id}")
            # Use globals for local camera state
            global bag_in, bag_out, counting_active, camera_confidence
            return {
                'bags_in': bag_in,
                'bags_out': bag_out,
                'total_bags': bag_in - bag_out,  # Net count: IN - OUT
                'counting_active': counting_active,
                'confidence': camera_confidence,
                'status': 'online'
            }
        
        # Get the specific domain and URL path for the camera
        if camera_id not in CAMERA_CONFIGS:
            logger.error(f"Camera {camera_id} not found in configuration")
            return {'status': 'error', 'bags_in': 0, 'bags_out': 0, 'total_bags': 0, 'counting_active': False}
        
        camera_config = CAMERA_CONFIGS[camera_id]
        camera_domain = camera_config['domain']
        url_path = camera_config['url_path']
        
        # Use fresh connection with no caching and shorter timeouts for faster response
        headers = {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        # Use a session for connection pooling and faster requests with aggressive timeouts
        session = requests.Session()
        session.timeout = 1  # Set default timeout to 1 second
        
        # The camera_id is already the actual ID from environment files
        expected_camera_id = camera_id
        
        # Special handling for Transshipment service
        if camera_id == 'Transshipment':
            # Try /get_count endpoint for transshipment
            count_url = f"{camera_domain}{url_path}/get_count"
            logger.info(f"Transshipment: Trying URL {count_url}")
            try:
                count_response = session.get(count_url, timeout=2.5, headers=headers)
                logger.info(f"Transshipment: Response status {count_response.status_code}")
                if count_response.status_code == 200:
                    count_data = count_response.json()
                    transship_count = count_data.get('transship_count', 0)
                    logger.info(f"Transshipment: Got count {transship_count}")
                    
                    # Try to get counting status
                    counting_url = f"{camera_domain}{url_path}/counting"
                    transship_active = False
                    try:
                        counting_response = session.get(counting_url, timeout=1.5, headers=headers)
                        if counting_response.status_code == 200:
                            counting_data = counting_response.json()
                            transship_active = counting_data.get('counting_active', False)
                    except:
                        pass
                    
                    logger.info(f"Transshipment: Returning online status")
                    return {
                        'bags_in': 0,  # Transshipment doesn't have separate in/out
                        'bags_out': 0,
                        'total_bags': transship_count,  # Use transship_count as total
                        'counting_active': transship_active,
                        'confidence': 0.0,
                        'status': 'online'
                    }
                else:
                    logger.warning(f"Transshipment: Bad response status {count_response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Transshipment: Request failed - {e}")
                pass
        
        # Use only /counting endpoint for regular cameras - it has all the data we need
        counting_url = f"{camera_domain}{url_path}/counting"
        
        try:
            counting_response = session.get(counting_url, timeout=2.5, headers=headers)
            
            if counting_response.status_code == 200:
                data = counting_response.json()
                return {
                    'bags_in': data.get('bag_in', 0),
                    'bags_out': data.get('bag_out', 0),
                    'total_bags': data.get('bag_in', 0) - data.get('bag_out', 0),
                    'counting_active': data.get('counting_active', False),
                    'confidence': data.get('confidence', 0.0),
                    'status': 'online'
                }
        except requests.exceptions.RequestException:
            # If counting endpoint fails, camera is likely offline
            pass
        
        # Quick health check with very short timeout (1 second)
        try:
            # Use camera domain + path health endpoint
            health_url = f"{camera_domain}{url_path}/health"
            health_response = session.get(health_url, timeout=1.5, headers=headers)
            
            if health_response.status_code == 200:
                return {
                    'bags_in': 0,
                    'bags_out': 0,
                    'total_bags': 0,
                    'counting_active': False,
                    'status': 'online'  # Camera is online even if counting endpoints fail
                }
        except requests.exceptions.RequestException:
            pass
        
        # All endpoints failed, camera is offline
        return {'status': 'offline', 'bags_in': 0, 'bags_out': 0, 'total_bags': 0, 'counting_active': False}
            
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout when fetching data from {camera_id} - camera may be slow")
        return {'status': 'timeout', 'bags_in': 0, 'bags_out': 0, 'total_bags': 0, 'counting_active': False}
    except requests.exceptions.ConnectionError:
        logger.warning(f"Connection error when fetching data from {camera_id} - camera may be offline")
        return {'status': 'offline', 'bags_in': 0, 'bags_out': 0, 'total_bags': 0, 'counting_active': False}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception for {camera_id}: {e}")
        return {'status': 'offline', 'bags_in': 0, 'bags_out': 0, 'total_bags': 0, 'counting_active': False}
    except Exception as e:
        logger.error(f"Unexpected error for {camera_id}: {e}")
        return {'status': 'error', 'bags_in': 0, 'bags_out': 0, 'total_bags': 0, 'counting_active': False}

def get_camera_data_parallel(camera_info):
    """Helper function for parallel camera data fetching"""
    camera_id, config, base_url = camera_info
    camera_data = get_camera_data(camera_id, base_url)
    return camera_id, {
        'name': config['name'],
        'location': config['location'],
        'data': camera_data
    }

def collect_all_camera_data(base_url):
    """Collect data from all cameras in parallel for faster response"""
    all_data = {}
    
    # Import ThreadPoolExecutor for parallel requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Prepare the list of camera information for parallel processing
    camera_info_list = [(camera_id, config, base_url) for camera_id, config in CAMERA_CONFIGS.items()]
    
    # Use more workers and shorter timeout for faster response
    max_workers = min(len(CAMERA_CONFIGS), 8)  # Optimal worker count
    
    # Execute requests in parallel with overall timeout
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_camera = {executor.submit(get_camera_data_parallel, info): info[0] for info in camera_info_list}
        
        # Process results as they complete - no timeout to avoid TimeoutError
        try:
            for future in as_completed(future_to_camera, timeout=5.0):  # Increased timeout for slower networks
                try:
                    camera_id, camera_data = future.result(timeout=0.5)  # Slightly longer individual timeout
                    all_data[camera_id] = camera_data
                except Exception as e:
                    camera_id = future_to_camera[future]
                    logger.warning(f"Error fetching data for camera {camera_id}: {e}")
                    all_data[camera_id] = {
                        'name': CAMERA_CONFIGS[camera_id]['name'],
                        'location': CAMERA_CONFIGS[camera_id]['location'],
                        'data': {'status': 'offline', 'bags_in': 0, 'bags_out': 0, 'total_bags': 0, 'counting_active': False}
                    }
        except Exception as timeout_error:
            # If we get a timeout, just log it and continue with whatever data we have
            logger.warning(f"Overall timeout in camera data collection: {timeout_error}")
        
        # Handle any remaining futures that didn't complete in time
        for future in future_to_camera:
            if not future.done():
                camera_id = future_to_camera[future]
                future.cancel()
                if camera_id not in all_data:
                    logger.warning(f"Camera {camera_id} did not respond in time - marking as offline")
                    all_data[camera_id] = {
                        'name': CAMERA_CONFIGS[camera_id]['name'],
                        'location': CAMERA_CONFIGS[camera_id]['location'],
                        'data': {'status': 'timeout', 'bags_in': 0, 'bags_out': 0, 'total_bags': 0, 'counting_active': False}
                    }
    
    return all_data

@app.route('/cam6/daily_summary')
def get_daily_summary():
    """Simplified real-time daily summary from all cameras"""
    try:
        LOCATION_MAP = {
            '102': 'Entry Gate',
            '202': 'Cluster 1 and 2',
            '302': 'Cluster 4 and 5',
            '402': 'Cluster 3',
            '502': 'Cluster 4 and 5',
            '602': 'Exit Gate',
            '702': 'Cluster 6',
            'Transshipment': 'Transshipment'
        }
        # Construct base URL dynamically from the request
        scheme = request.scheme
        host = request.host
        base_url = f"{scheme}://{host}"
        
        all_camera_data = collect_all_camera_data(base_url)
        
        # Simplified camera data - only essential information
        cameras = []
        totals = {'in': 0, 'out': 0, 'total': 0}
        system = {'online': 0, 'offline': 0, 'total': len(CAMERA_CONFIGS)}
        
        # Define camera order: Current camera (602) first, then others in numerical order
        camera_order = ['602', '102', '202', '302', '402', '502', '702', 'Transshipment']
        
        # Process cameras in the exact order we want them to appear
        for camera_id in camera_order:
            if camera_id not in all_camera_data:
                continue
                
            camera_info = all_camera_data[camera_id]
            data = camera_info['data']
            is_online = data['status'] == 'online'
            
            # Different structure for Transshipment vs regular cameras
            if camera_id == 'Transshipment':
                # Transshipment only has count, no in/out
                camera = {
                    'id': camera_id,
                    'name': camera_info['name'],
                    'location': LOCATION_MAP.get(camera_id, camera_info['location']),
                    'count': data['total_bags'],  # Use 'count' instead of 'total'
                    'active': data.get('counting_active', False),
                    'online': is_online,
                    'confidence': data.get('confidence', 0.0)
                }
            else:
                # Regular cameras have in/out/total structure
                camera = {
                    'id': camera_id,
                    'name': camera_info['name'],
                    'location': LOCATION_MAP.get(camera_id, camera_info['location']),
                    'in': data['bags_in'],
                    'out': data['bags_out'],
                    'total': data['total_bags'],
                    'active': data.get('counting_active', False),
                    'online': is_online,
                    'confidence': data.get('confidence', 0.0)
                }
            
            cameras.append(camera)
            
            # Update totals and system status
            if is_online:
                system['online'] += 1
                totals['in'] += data['bags_in']
                totals['out'] += data['bags_out']
                totals['total'] += data['total_bags']
            else:
                system['offline'] += 1
        
        # Simplified response structure
        response_data = {
            'timestamp': int(time.time()),  # Unix timestamp for easier processing
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'cameras': cameras,
            'totals': totals,
            'system': system,
            'status': 'online' if system['online'] > 0 else 'offline'
        }
        
        # Real-time headers - no caching
        from flask import make_response
        response = make_response(jsonify(response_data))
        response.headers.update({
            'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Access-Control-Allow-Origin': '*',  # Enable CORS for real-time updates
            'Content-Type': 'application/json'
        })
        
        return response
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in daily summary: {e}")
        logger.error(f"Full traceback: {error_details}")
        return jsonify({
            'error': 'Service temporarily unavailable',
            'error_details': str(e),
            'error_type': type(e).__name__,
            'timestamp': int(time.time()),
            'status': 'error'
        }), 500

# Alternative route for backward compatibility and different URL patterns
@app.route('/daily_summary')
def get_daily_summary_alt():
    """Alternative route for daily summary - same functionality"""
    return get_daily_summary()

# Test route to verify deployment
@app.route('/cam6/test')
@app.route('/test')
def test_deployment():
    """Simple test route to verify updated code is deployed"""
    return jsonify({
        'status': 'success',
        'message': 'Updated code deployed successfully!',
        'timestamp': int(time.time()),
        'version': '2025-08-07-optimized',
        'daily_summary_available': True,
        'camera_configs_loaded': len(CAMERA_CONFIGS),
        'current_camera_id': CAMERA_ID
    })

# Debug route to test basic functionality
@app.route('/cam6/debug')
@app.route('/debug')
def debug_info():
    """Debug endpoint to check configuration and basic functionality"""
    try:
        return jsonify({
            'status': 'success',
            'camera_configs': list(CAMERA_CONFIGS.keys()),
            'current_camera_id': CAMERA_ID,
            'current_counts': {
                'bags_in': bag_in,
                'bags_out': bag_out,
                'total': bag_in - bag_out
            },
            'counting_active': counting_active,
            'timestamp': int(time.time())
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': int(time.time())
        }), 500

@app.route('/cam6/api/live_status')
@app.route('/api/live_status')
def get_live_status():
    """Get current live status from all cameras"""
    try:
        # Construct base URL dynamically from the request
        scheme = request.scheme  # http or https
        host = request.host      # domain:port or IP:port
        base_url = f"{scheme}://{host}"
        
        all_camera_data = collect_all_camera_data(base_url)
        
        live_data = []
        for camera_id, camera_info in all_camera_data.items():
            live_data.append({
                'camera_id': camera_id,
                'camera_name': camera_info['name'],
                'location': camera_info['location'],
                'status': camera_info['data']['status'],
                'bags_in': camera_info['data']['bags_in'],
                'bags_out': camera_info['data']['bags_out'],
                'total_bags': camera_info['data']['total_bags'],
                'counting_active': camera_info['data'].get('counting_active', False)
            })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'cameras': live_data
        })
        
    except Exception as e:
        logger.error(f"Error in live status: {e}")
        return jsonify({'error': 'Failed to collect camera data'}), 500


# --- Admin: counting_lines proxy and bulk disable ---
@app.route('/cam6/control/<camera_id>/counting_lines')
def proxy_counting_lines(camera_id: str):
    """Proxy counting_lines enable/disable to remote cameras.
    Example: /cam6/control/502/counting_lines?action=disable
    """
    try:
        action = request.args.get('action', 'disable').lower()
        if action not in ['enable', 'disable', 'toggle']:
            return jsonify({'success': False, 'error': 'Invalid action'}), 400

        # Resolve remote endpoint from CAMERA_CONFIGS
        if camera_id not in CAMERA_CONFIGS:
            return jsonify({'success': False, 'error': f'Camera {camera_id} not found'}), 404

        camera_cfg = CAMERA_CONFIGS[camera_id]
        domain = camera_cfg['domain']
        url_path = camera_cfg['url_path']

        # Many cameras support /counting_lines; transshipment may differ, skip here
        target_url = f"{domain}{url_path}/counting_lines?action={action}"

        try:
            resp = requests.get(target_url, timeout=1.5, headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            })
            data = None
            try:
                data = resp.json()
            except Exception:
                data = {'status_code': resp.status_code}
            if resp.status_code == 200:
                return jsonify({'success': True, 'camera_id': camera_id, 'remote': data})
            return jsonify({'success': False, 'camera_id': camera_id, 'remote': data}), 502
        except requests.RequestException as re:
            return jsonify({'success': False, 'error': str(re)}), 502
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/cam6/control/disable_all')
def disable_all_counting_lines():
    """Disable counting_lines on all configured cameras."""
    action = request.args.get('action', 'disable').lower()
    results = {}
    for cam_id in CAMERA_CONFIGS.keys():
        try:
            # Transshipment may not support counting_lines; attempt and continue
            domain = CAMERA_CONFIGS[cam_id]['domain']
            url_path = CAMERA_CONFIGS[cam_id]['url_path']
            url = f"{domain}{url_path}/counting_lines?action={action}"
            r = requests.get(url, timeout=1.5)
            try:
                results[cam_id] = {'ok': r.ok, 'status': r.status_code, 'data': r.json()}
            except Exception:
                results[cam_id] = {'ok': r.ok, 'status': r.status_code}
        except Exception as e:
            results[cam_id] = {'ok': False, 'error': str(e)}
    return jsonify({'success': True, 'action': action, 'results': results})


@app.route('/cam6/control/<camera_id>/adjust', methods=['POST'])
@app.route('/control/<camera_id>/adjust', methods=['POST', 'GET'])
def adjust_camera_count(camera_id):
    """Adjust (increment/decrement) count for a specific camera by delta.
    JSON body: { "direction": "in"|"out", "delta": 1|-1 }
    """
    try:
        # Support both POST JSON and GET query params for debugging via browser
        if request.method == 'GET':
            data = {
                'direction': request.args.get('direction'),
                'delta': request.args.get('delta')
            }
        else:
            data = request.get_json(silent=True) or {}
        direction = data.get('direction')
        delta_raw = data.get('delta', 0)
        try:
            delta = int(delta_raw) if delta_raw is not None else 0
        except Exception:
            delta = 0
        if direction not in ['in', 'out']:
            return jsonify({'success': False, 'error': 'Invalid direction'}), 400
        if delta == 0:
            return jsonify({'success': False, 'error': 'Delta must be non-zero'}), 400

        # Normalize camera id (handles transshipment variants)
        camera_id_norm = normalize_camera_id(camera_id)

        # Local camera fast-path
        if camera_id_norm == CAMERA_ID or camera_id_norm == '602':
            global bag_in, bag_out
            if direction == 'in':
                bag_in = max(0, bag_in + delta)
            else:
                bag_out = max(0, bag_out + delta)
            return jsonify({'success': True, 'camera_id': camera_id_norm, 'bag_in': bag_in, 'bag_out': bag_out})

        # Remote cameras: fetch current, compute new value, then send to remote /update_count
        # 1) Get current counts using existing collector
        scheme = request.scheme
        host = request.host
        base_url = f"{scheme}://{host}"
        current = get_camera_data(camera_id_norm, base_url)
        curr_in = int(current.get('bags_in', 0) or 0)
        curr_out = int(current.get('bags_out', 0) or 0)
        if direction == 'in':
            new_value = max(0, curr_in + delta)
        else:
            new_value = max(0, curr_out + delta)

        # 2) Resolve remote endpoint
        camera_info = get_camera_info_by_id(camera_id_norm)
        if not camera_info:
            return jsonify({'success': False, 'error': f'Camera {camera_id_norm} not found'}), 404
        base = camera_info['base_url']
        logical_id = camera_info['logical_id']

        if logical_id in ['tranship', 'Transshipment']:
            # For transshipment, only a single count; adjust total as 'count'
            new_count = max(0, int(current.get('total_bags', 0) or 0) + delta)
            # If base already points directly to port-based instance, avoid logical path prefix
            url = f"{base}/update_count" if base.endswith((':5003', ':5003/')) or base.rstrip('/').endswith(':5003') else f"{base}/{logical_id}/update_count"
            payload = {'count': new_count}
        else:
            url = f"{base}/{logical_id}/update_count"
            payload = {direction: new_value}

        resp = requests.post(url, json=payload, timeout=2.0)
        if resp.status_code == 200:
            return jsonify({'success': True, 'camera_id': camera_id_norm, 'updated': payload})
        return jsonify({'success': False, 'camera_id': camera_id_norm, 'error': f'remote status {resp.status_code}'}), 502
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/cam6/api/system_overview')
def get_system_overview():
    """Get system overview with aggregated statistics"""
    try:
        # Construct base URL dynamically from the request
        scheme = request.scheme  # http or https
        host = request.host      # domain:port or IP:port
        base_url = f"{scheme}://{host}"
        
        all_camera_data = collect_all_camera_data(base_url)
        
        total_in = 0
        total_out = 0
        online_count = 0
        active_counting = 0
        
        camera_statuses = []
        
        for camera_id, camera_info in all_camera_data.items():
            data = camera_info['data']
            
            if data['status'] == 'online':
                online_count += 1
                total_in += data['bags_in']
                total_out += data['bags_out']
                
            if data.get('counting_active', False):
                active_counting += 1
                
            camera_statuses.append({
                'camera_id': camera_id,
                'name': camera_info['name'],
                'status': data['status'],
                'counting_active': data.get('counting_active', False)
            })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'system_totals': {
                'total_bags_in': total_in,
                'total_bags_out': total_out,
                'net_bags': total_in - total_out,
                'total_bags': total_in + total_out
            },
            'system_health': {
                'total_cameras': len(CAMERA_CONFIGS),
                'online_cameras': online_count,
                'offline_cameras': len(CAMERA_CONFIGS) - online_count,
                'cameras_counting': active_counting,
                'system_uptime_percent': round((online_count / len(CAMERA_CONFIGS)) * 100, 1)
            },
            'camera_statuses': camera_statuses
        })
        
    except Exception as e:
        logger.error(f"Error in system overview: {e}")
        return jsonify({'error': 'Failed to collect system data'}), 500


# --- Threaded Video Capture ---
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


# --- Optimized Detection Loop ---
# Get frame skip from environment variable (default: 3)
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 3))
logger.info(f"Using frame skip: {FRAME_SKIP} (processing 1 out of every {FRAME_SKIP} frames)")
INFER_IMG_SIZE = 320  # Lower inference size for speed and reduced memory usage

# Memory management settings
MAX_BATCH_SIZE = 1  # Process one frame at a time
TORCH_DEVICE_MEM_FRACTION = 0.7  # Limit GPU memory usage

# FPS calculation variables
fps_start_time = 0
fps_frame_count = 0
fps_update_interval = 1.0  # Update FPS every 1 second


def detection_loop():
    global camera_confidence, latest_frame, stream_up, bag_in, bag_out, counting_active

    # Use threaded video capture
    cap = VideoCaptureThread(RTSP_URL, width=INFER_IMG_SIZE, height=INFER_IMG_SIZE)
    stream_up = True
    frame_count = 0
    last_results_cement = None
    last_results_fleet = None
    last_frame_clean = None  # Initialize last_frame_clean

    # Initialize tracking variables using the Tracker class
    tracker = Tracker()
    previous_positions = {}  # Track previous positions for direction detection

    # Dictionary to track which lines objects have crossed and in which direction
    # Format: {object_id_line_idx: (timestamp, direction)}
    # crossed_lines = {}

    # FPS calculation variables
    fps = 0.0
    fps_start_time = time.time()
    fps_frame_count = 0

    # Memory management
    import gc
    memory_check_counter = 0
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

                # Run YOLO with memory-efficient settings and confidence threshold
                last_results_cement = model_cement(frame, conf=CONF_THRESHOLD, imgsz=INFER_IMG_SIZE,
                                                   batch=MAX_BATCH_SIZE, verbose=False)

                # Only run second model if first one succeeded
                last_results_fleet = model_fleet(frame, conf=CONF_THRESHOLD, imgsz=INFER_IMG_SIZE, batch=MAX_BATCH_SIZE,
                                                 verbose=False)

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
        max_frame_confidence = 0.0

        # ---------- Cement bag logic ----------
        # We're now using the fixed line from environment variables

        if results_cement is not None:
            try:
                # Process detections and prepare for tracking exactly like in working code
                detections = []
                detection_info = {}  # Store class and confidence for each detection

                # Extract boxes using the same method as in working_code
                for r in results_cement[0].boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    max_frame_confidence = max(max_frame_confidence, score)

                    if score > CONF_THRESHOLD:
                        # Convert to x, y, w, h format for tracker
                        detection_box = [int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)]
                        detections.append(detection_box)

                        # Store detection info for later use
                        detection_key = f"{int(x1)}_{int(y1)}_{int(x2) - int(x1)}_{int(y2) - int(y1)}"

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

                # Process tracked objects
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

                    # Calculate center of the box
                    cx = x + w // 2
                    cy = y + h // 2
                    center = (cx, cy)

                    # Check for line crossings if we have previous position
                    if object_id in previous_positions:
                        prev_position = previous_positions[object_id]
                        current_time = time.time()

                        # Check each line for crossings
                        for i, (line_start, line_end) in enumerate(lines):
                            line_color = line_colors[i]
                            line_direction = count_directions[i]

                            # Check if the object crossed this line
                            prev_side = is_crossing_line(prev_position, (line_start, line_end))
                            current_side = is_crossing_line(center, (line_start, line_end))

                            # Only process if the object has crossed the line
                            if prev_side != current_side:
                                # Determine the movement direction
                                movement_direction = calculate_direction(
                                    prev_position, center, line_start, line_end
                                )

                                # Create color group key for deduplication (ignore direction)
                                # color_group_key = get_color_group_key(line_color)
                                # object_group_key = f"{object_id}_{color_group_key}"

                                # Check if this object has been counted in this color-direction group recently
                                # current_time = time.time()
                                # if object_group_key in crossed_groups:
                                #     last_cross_time, last_direction = crossed_groups[object_group_key]
                                #     time_since_last = current_time - last_cross_time
                                #     # Only count if it's been more than 1.5 seconds since last crossing
                                #     # or the direction has changed
                                #     if (time_since_last < 1.5) and (movement_direction == last_direction):
                                #         continue

                                # Check if the movement direction matches the line's counting direction
                                # count_this = False
                                # if ((line_direction == "left" and movement_direction == "OUT") or
                                #         (line_direction == "right" and movement_direction == "IN") or
                                #         (line_direction == "up" and movement_direction == "OUT") or
                                #         (line_direction == "down" and movement_direction == "IN")):
                                #     count_this = True

                                # if count_this and counting_active:
                                #     # Increment appropriate directional counter
                                #     if movement_direction == "IN":
                                #         bag_in += 1
                                #         direction_text = "IN"
                                #     else:
                                #         bag_out += 1
                                #         direction_text = "OUT"

                                #     crossed_groups[object_group_key] = (current_time, movement_direction)
                                #     total_count = bag_in + bag_out
                                #     logger.info(
                                #         f"[{CAMERA_ID}] ✅ Bag {direction_text} at line {i + 1} (Color Group: {color_group_key})! In: {bag_in}, Out: {bag_out}, Total: {total_count}")

                    # Draw bounding box with class name and confidence
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                    # Indicate if this bag has been counted
                    # crossing_key = f"{object_id}_0"  # Only one line in this case
                    # if crossing_key in crossed_lines:
                    #     cv2.putText(frame, "Counted", (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # Update previous position for next frame
                    previous_positions[object_id] = center
            except Exception as e:
                logger.error(f"Error in detection processing: {e}")
                import traceback
                logger.error(traceback.format_exc())
        # Update camera confidence
        camera_confidence = max_frame_confidence

        # Draw all counting lines with their respective colors and directions
        # for i, (line_start, line_end) in enumerate(lines):
        #     line_color = line_colors[i] if i < len(line_colors) else (0, 0, 255)
        #     line_direction = count_directions[i] if i < len(count_directions) else "left"

        #     # Draw the line
        #     # cv2.line(frame, line_start, line_end, line_color, 2)

        #     # Add line number label
        #     mid_x = (line_start[0] + line_end[0]) // 2
        #     mid_y = (line_start[1] + line_end[1]) // 2
        #     cv2.putText(frame, f"{i + 1}", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)

        #     # Draw an arrow to indicate counting direction
        #     arrow_length = 30
        #     arrow_start = (mid_x, mid_y)
        #     arrow_end = None

        #     if line_direction == "left":
        #         arrow_end = (arrow_start[0] - arrow_length, arrow_start[1])
        #     elif line_direction == "right":
        #         arrow_end = (arrow_start[0] + arrow_length, arrow_start[1])
        #     elif line_direction == "up":
        #         arrow_end = (arrow_start[0], arrow_start[1] - arrow_length)
        #     elif line_direction == "down":
        #         arrow_end = (arrow_start[0], arrow_start[1] + arrow_length)

        #     # if arrow_end:
        #     #     cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 2)
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

        font_scale = 0.6
        thickness = 2

        # Draw Bag Count in red at the top left
        # Group 1: Bag counting information
        cv2.putText(frame, f"Bag In: {bag_in}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, f"Bag Out: {bag_out}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

        # Group 2: System information (large gap from counting group)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

        # Draw Camera ID in green below FPS
        cv2.putText(frame, f"Camera: {CAMERA_ID}", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                    thickness)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            latest_frame = jpeg.tobytes()

        # Periodic memory cleanup to prevent gradual slowdown
        cleanup_counter += 1
        if cleanup_counter % 300 == 0:  # Every 300 frames (~20 seconds at 15 FPS)
            # Clean up old entries from crossed_groups (older than 10 seconds)
            # current_time = time.time()
            # keys_to_remove = []
            # for key, (timestamp, direction) in crossed_groups.items():
            #     if current_time - timestamp > 10.0:
            #         keys_to_remove.append(key)
            # for key in keys_to_remove:
            #     del crossed_groups[key]

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # logger.info(f"[{CAMERA_ID}] Memory cleanup: removed {len(keys_to_remove)} old tracking entries")

        # Update performance statistics
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
    # Set up template folder explicitly
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    print(f"Template directory: {template_dir}")
    print(f"Templates exist: {os.path.exists(template_dir)}")
    if os.path.exists(template_dir):
        print(f"Templates: {os.listdir(template_dir)}")
    app.run(host="0.0.0.0", port=port, debug=True)
