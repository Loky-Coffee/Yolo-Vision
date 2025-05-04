from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import uuid
import numpy as np
from ultralytics import YOLO
import os
import time
from datetime import datetime
import threading
import glob
import shutil
from werkzeug.utils import secure_filename
from queue import Queue, PriorityQueue
from concurrent.futures import ThreadPoolExecutor
import torch
import gc
import aiofiles
import asyncio
import psutil
import platform
import atexit

app = Flask(__name__)

current_model = None
camera = None
last_save_time = 0
save_cooldown = 10
is_saving = False
video_active = True
video_frame_queue = Queue(maxsize=2)

frame_lock = threading.Lock()
model_lock = threading.Lock()

model_info = {
    'hw_status': 'Not started',
    'model_dimensions': None,
    'device': None,
    'processing_mode': None,
    'input_shape': None
}

frame_priority_queue = PriorityQueue(maxsize=10)

detection_executor = ThreadPoolExecutor(max_workers=2)

settings = {
    'conf_threshold': 0.5,
    'camera_width': 1280,
    'camera_height': 720,
    'mask_alpha': 0.5,
    'mask_color': '#FF0000',
    'model_input_width': 640,
    'model_input_height': 640
}

color_cache = {}

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
MODELS_FOLDER = "models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

def cleanup_old_files():
    """Clean up old files, keeping only the latest 20"""
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        files = glob.glob(os.path.join(folder, "*"))
        if len(files) > 20:
            files.sort(key=os.path.getctime)
            for file in files[:-20]:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error deleting {file}: {e}")

def get_hardware_info():
    """Get hardware information"""
    gpu_info = "CPU only"
    cpu_info = platform.processor()
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_info = f"GPU: {gpu_name}"
    
    return cpu_info, gpu_info

def get_camera_with_fallback():
    """Try to open camera with fallback resolutions"""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return None
    
    width = settings['camera_width']
    height = settings['camera_height']
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Requested: {width}x{height}, Actual: {actual_width}x{actual_height}")
    return camera

def cleanup_model():
    """Bereinigt das aktuelle Modell aus dem Speicher"""
    global current_model
    
    if current_model is not None:
        print("Cleaning up model...")
        try:
            if hasattr(current_model, 'model'):
                current_model.model.cpu()
                del current_model.model
            del current_model
            current_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("Model cleanup successful")
        except Exception as e:
            print(f"Error during model cleanup: {e}")

def cleanup_memory():
    """Regular memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleanup performed")

cleanup_timer = None

def start_cleanup_timer():
    """Start repeated memory cleanup"""
    global cleanup_timer
    if cleanup_timer is not None:
        cleanup_timer.cancel()
    cleanup_timer = threading.Timer(300, cleanup_memory_loop)
    cleanup_timer.daemon = True
    cleanup_timer.start()

def cleanup_memory_loop():
    """Memory cleanup loop"""
    cleanup_memory()
    start_cleanup_timer()

start_cleanup_timer()

@app.route("/")
def index():
    models = get_available_models()
    return render_template("index.html", models=models, settings=settings)

@app.route("/update_settings", methods=["POST"])
def update_settings():
    global settings
    
    data = request.json
    
    if 'conf_threshold' in data:
        settings['conf_threshold'] = float(data['conf_threshold'])
    if 'camera_width' in data:
        settings['camera_width'] = int(data['camera_width'])
    if 'camera_height' in data:
        settings['camera_height'] = int(data['camera_height'])
    if 'mask_alpha' in data:
        settings['mask_alpha'] = float(data['mask_alpha'])
    if 'mask_color' in data:
        settings['mask_color'] = data['mask_color']
    if 'model_input_width' in data:
        settings['model_input_width'] = int(data['model_input_width'])
    if 'model_input_height' in data:
        settings['model_input_height'] = int(data['model_input_height'])
    
    return jsonify({"status": "success"})

def hex_to_bgr_cached(hex_color):
    """Cached color conversion"""
    if hex_color not in color_cache:
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        color_cache[hex_color] = (b, g, r)
    return color_cache[hex_color]

def get_available_models():
    """Find all available models in models folder"""
    model_patterns = ["*.pt", "*.onnx", "*.engine"]
    models = []
    for pattern in model_patterns:
        models.extend(glob.glob(os.path.join(MODELS_FOLDER, pattern)))
    return [os.path.basename(m) for m in models]

@app.route("/load_model", methods=["POST"])
def load_model():
    """Load selected model with automatic cleanup"""
    global current_model, camera, model_info, settings
    
    model_index = request.json.get("model_index")
    models = get_available_models()
    
    if model_index < 0 or model_index >= len(models):
        return jsonify({"status": "error", "message": "Invalid model index"})
    
    selected_model = models[model_index]
    model_path = os.path.join(MODELS_FOLDER, selected_model)
    
    with model_lock:
        cleanup_model()
        
        try:
            print(f"Loading new model: {selected_model}")
            current_model = YOLO(model_path)
            
            if torch.cuda.is_available():
                current_model.model.to('cuda')
                model_info['device'] = 'GPU'
                model_info['processing_mode'] = 'GPU Beschleunigung'
            else:
                model_info['device'] = 'CPU'
                model_info['processing_mode'] = 'CPU Verarbeitung'
            
            model_info['hw_status'] = f"Model geladen: {selected_model}"
            
            model_info['input_shape'] = "Not detected"
            found_resolution = False
            
            for size in [416, 640, 1280]:
                try:
                    test_img = np.zeros((size, size, 3), dtype=np.uint8)
                    test_result = current_model.predict(source=test_img, conf=0.0, verbose=False)
                    
                    settings['model_input_width'] = size
                    settings['model_input_height'] = size
                    model_info['input_shape'] = f"{size}x{size}"
                    found_resolution = True
                    break
                except Exception as dim_error:
                    continue
            
            if not found_resolution:
                settings['model_input_width'] = 640
                settings['model_input_height'] = 640
                model_info['input_shape'] = "640x640 (default)"
            
            cpu_info, gpu_info = get_hardware_info()
            model_info['hw_status'] += f"\n{cpu_info}\n{gpu_info}\nDevice: {model_info['device']}"
            
            if camera is None:
                camera = get_camera_with_fallback()
            
            return jsonify({
                "status": "success",
                "model_info": model_info
            })
        except Exception as e:
            model_info['hw_status'] = f"Error loading model: {str(e)}"
            return jsonify({
                "status": "error", 
                "message": f"Error loading model: {str(e)}"
            })

@app.route("/get_model_info", methods=["GET"])
def get_model_info():
    """Get current model information"""
    return jsonify(model_info)

@app.route("/detect_image", methods=["POST"])
def detect_image():
    """Detect objects in uploaded image with segmentation masks"""
    global current_model
    
    if current_model is None:
        return jsonify({"status": "error", "message": "No model loaded!"})
    
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded!"})
    
    file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    result_path = os.path.join(RESULT_FOLDER, filename)
    
    file.save(input_path)
    
    try:
        img = cv2.imread(input_path)
        original_img = img.copy()
        img_resized = cv2.resize(img, (settings['model_input_width'], settings['model_input_height']))
        
        with model_lock:
            results = current_model.predict(source=img_resized, conf=settings['conf_threshold'])
        
        output_img = original_img.copy()
        
        for r in results:
            if r.masks is not None:
                masks_data = r.masks.data.cpu().numpy()
                original_height, original_width = original_img.shape[:2]
                
                for mask in masks_data:
                    mask_resized = cv2.resize(mask, (original_width, original_height))
                    mask_bool = mask_resized.astype(bool)
                    
                    colored_mask = np.zeros_like(output_img)
                    colored_mask[:, :] = hex_to_bgr_cached(settings['mask_color'])
                    
                    output_img[mask_bool] = cv2.addWeighted(
                        output_img[mask_bool], 
                        1 - settings['mask_alpha'], 
                        colored_mask[mask_bool], 
                        settings['mask_alpha'], 
                        0
                    )
            
            if r.boxes is not None:
                boxes_data = r.boxes.xyxy
                if len(boxes_data) > 0:
                    original_height, original_width = original_img.shape[:2]
                    scale_x = original_width / settings['model_input_width']
                    scale_y = original_height / settings['model_input_height']
                    
                    for box, conf, cls in zip(boxes_data, r.boxes.conf, r.boxes.cls):
                        x1 = int(box[0] * scale_x)
                        y1 = int(box[1] * scale_y)
                        x2 = int(box[2] * scale_x)
                        y2 = int(box[3] * scale_y)
                        
                        cv2.rectangle(output_img, (x1, y1), (x2, y2), hex_to_bgr_cached(settings['mask_color']), 2)
                        
                        label = f"Class {int(cls)}: {conf:.2f}"
                        cv2.putText(output_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hex_to_bgr_cached(settings['mask_color']), 2)
        
        cv2.imwrite(result_path, output_img)
        
        cleanup_old_files()
        
        return send_file(result_path, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({"status": "error", "message": f"Detection error: {str(e)}"})
    finally:
        try:
            os.remove(input_path)
        except:
            pass

def detect_and_save(frame):
    """Detect objects with segmentation masks and save result if cooldown elapsed"""
    global last_save_time, is_saving, current_model, model_info
    
    if current_model is None:
        return frame
    
    try:
        with frame_lock:
            original_frame = frame.copy()
            frame_resized = cv2.resize(frame, (settings['model_input_width'], settings['model_input_height']))
            
            with model_lock:
                if current_model is None:
                    return frame
                results = current_model.predict(source=frame_resized, conf=settings['conf_threshold'], verbose=False)
            
            output_frame = original_frame.copy()
            object_detected = False
            
            for r in results:
                if r.masks is not None:
                    masks_data = r.masks.data.cpu().numpy()
                    original_height, original_width = original_frame.shape[:2]
                    
                    for mask in masks_data:
                        mask_resized = cv2.resize(mask, (original_width, original_height))
                        mask_bool = mask_resized.astype(bool)
                        
                        colored_mask = np.zeros_like(output_frame)
                        colored_mask[:, :] = hex_to_bgr_cached(settings['mask_color'])
                        
                        output_frame[mask_bool] = cv2.addWeighted(
                            output_frame[mask_bool], 
                            1 - settings['mask_alpha'], 
                            colored_mask[mask_bool], 
                            settings['mask_alpha'], 
                            0
                        )
                        
                        object_detected = True
                
                if r.boxes is not None:
                    boxes_data = r.boxes.xyxy
                    if len(boxes_data) > 0:
                        original_height, original_width = original_frame.shape[:2]
                        scale_x = original_width / settings['model_input_width']
                        scale_y = original_height / settings['model_input_height']
                        
                        for box, conf, cls in zip(boxes_data, r.boxes.conf, r.boxes.cls):
                            x1 = int(box[0] * scale_x)
                            y1 = int(box[1] * scale_y) 
                            x2 = int(box[2] * scale_x)
                            y2 = int(box[3] * scale_y)
                            
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), hex_to_bgr_cached(settings['mask_color']), 2)
                            label = f"Class {int(cls)}: {conf:.2f}"
                            cv2.putText(output_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hex_to_bgr_cached(settings['mask_color']), 2)
                            object_detected = True
            
            current_time = time.time()
            time_since_last_save = current_time - last_save_time
            
            if object_detected and time_since_last_save >= save_cooldown and not is_saving:
                is_saving = True
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_path = os.path.join(RESULT_FOLDER, f"detected_object_{timestamp}.jpg")
                cv2.imwrite(result_path, output_frame)
                last_save_time = current_time
                is_saving = False
                print(f"Object detected and saved: {result_path}")
                
                cleanup_old_files()
            
            if time_since_last_save < save_cooldown:
                cooldown_text = f"Cooldown: {save_cooldown - int(time_since_last_save)}s"
                cv2.putText(output_frame, cooldown_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(output_frame, "Ready", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return output_frame
    except Exception as e:
        print(f"Detection error: {str(e)}")
        model_info['hw_status'] = f"Error: {str(e)}"
        return frame

def video_capture_thread():
    """Optimized thread for video capture"""
    global camera, frame_priority_queue
    
    while video_active:
        if camera is None:
            time.sleep(0.1)
            continue
            
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
        
        timestamp = time.time()
        try:
            frame_priority_queue.put_nowait((-timestamp, frame))
        except:
            try:
                frame_priority_queue.get_nowait()
            except:
                pass
            frame_priority_queue.put_nowait((-timestamp, frame))

def generate_frames():
    """Optimized generator function for video streaming"""
    global frame_priority_queue
    frame_time = 1/30
    last_frame_time = 0
    
    while video_active:
        current_time = time.time()
        if current_time - last_frame_time < frame_time:
            time.sleep(0.01)
            continue
            
        if frame_priority_queue.empty():
            time.sleep(0.01)
            continue
            
        _, frame = frame_priority_queue.get()
        
        processed_frame = detect_and_save(frame)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame, [
            cv2.IMWRITE_JPEG_QUALITY, 90,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
        frame = buffer.tobytes()
        
        last_frame_time = current_time
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset_cooldown', methods=['POST'])
def reset_cooldown():
    """Manually reset cooldown"""
    global last_save_time
    last_save_time = 0
    return jsonify({"status": "success"})

@app.route('/delete_model/<int:model_index>', methods=['POST'])
def delete_model(model_index):
    """Delete model from models folder"""
    models = get_available_models()
    
    if model_index < 0 or model_index >= len(models):
        return jsonify({"status": "error", "message": "Invalid index"})
    
    try:
        model_name = models[model_index]
        model_path = os.path.join(MODELS_FOLDER, model_name)
        os.remove(model_path)
        return jsonify({
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error deleting: {str(e)}"
        })

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Upload a new model to the server"""
    if 'model' not in request.files:
        return jsonify({"status": "error", "message": "Keine Datei ausgewählt!"})
    
    file = request.files['model']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Keine Datei ausgewählt!"})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(MODELS_FOLDER, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            return jsonify({"status": "error", "message": f"Modell '{filename}' existiert bereits!"})
        
        # Check file extension
        allowed_extensions = {'.pt', '.onnx', '.engine'}
        ext = os.path.splitext(filename)[1].lower()
        
        if ext not in allowed_extensions:
            return jsonify({"status": "error", "message": f"Ungültige Dateityp! Erlaubt sind: {', '.join(allowed_extensions)}"})
        
        try:
            file.save(filepath)
            return jsonify({"status": "success", "message": f"Modell '{filename}' wurde erfolgreich hochgeladen!"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Fehler beim Speichern: {str(e)}"})

@app.route('/rename_model', methods=['POST'])
def rename_model():
    """Rename model"""
    data = request.json
    model_index = data.get('model_index')
    new_name = data.get('new_name')
    
    models = get_available_models()
    
    if model_index < 0 or model_index >= len(models):
        return jsonify({"status": "error", "message": "Invalid index"})
    
    try:
        old_name = models[model_index]
        old_path = os.path.join(MODELS_FOLDER, old_name)
        ext = os.path.splitext(old_name)[1]
        
        if not new_name.endswith(ext):
            new_name += ext
        
        new_path = os.path.join(MODELS_FOLDER, new_name)
        
        if os.path.exists(new_path):
            return jsonify({
                "status": "error", 
                "message": f"Model with name '{new_name}' already exists"
            })
        
        os.rename(old_path, new_path)
        return jsonify({
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error renaming: {str(e)}"
        })

def shutdown_cleanup():
    """Clean up at shutdown"""
    global video_active, camera, current_model, cleanup_timer
    
    print("Shutting down...")
    video_active = False
    time.sleep(0.5)
    
    if camera:
        camera.release()
        camera = None
    
    cleanup_model()
    
    if cleanup_timer:
        cleanup_timer.cancel()
    
    detection_executor.shutdown()

atexit.register(shutdown_cleanup)

if __name__ == "__main__":
    video_thread = threading.Thread(target=video_capture_thread, daemon=True)
    video_thread.start()
    
    try:
        app.run(debug=True, threaded=True)
    finally:
        shutdown_cleanup()