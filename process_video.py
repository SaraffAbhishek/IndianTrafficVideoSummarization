from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import tempfile
import os
import uuid
from flask_cors import CORS
from ultralytics import YOLO
import json
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load custom YOLO model
model = YOLO('best_yolo11.pt')

# Define class names
class_names = [
    'trak', 'cyclist', 'bike', 'tempo', 'car', 'zeep', 'toto',
    'e-rickshaw', 'auto-rickshaw', 'bus', 'van', 'cycle-rickshaw',
    'person', 'taxi'
]

# Helper: road detection mask for context
def detect_road_or_ground(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 30, 200])
    road_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

# Technique 1: Black & White background preserving road context
def bw_mask(frame, selected_object, conf_thresh):
    road_mask = detect_road_or_ground(frame)
    results = model(frame, conf=conf_thresh)
    object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        if label == selected_object:
            cv2.rectangle(object_mask, (x1,y1), (x2,y2), 255, -1)
    final_mask = cv2.bitwise_or(road_mask, object_mask)
    desat = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    desat = cv2.cvtColor(desat, cv2.COLOR_GRAY2BGR)
    result = frame.copy()
    result[final_mask == 0] = desat[final_mask == 0]

    detected = []
    for det in results[0].boxes:
        label = class_names[int(det.cls[0].cpu().item())]
        conf = det.conf[0].cpu().item()
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        if label == selected_object:
            detected.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": float(conf), "label": label})
            cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(result, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    return detected, result

# Technique 2: Complete Blackout (black background, only selected object)
def complete_blackout(frame, selected_object, conf_thresh):
    results = model(frame, conf=conf_thresh)
    result = np.zeros_like(frame)
    detected = []
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        conf = det.conf[0].cpu().item()
        if label == selected_object:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            obj = cv2.bitwise_and(frame, frame, mask=mask)
            result = cv2.add(result, obj)
            detected.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": float(conf), "label": label})
            cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(result, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    return detected, result

# Technique 3: Blur Blackout (blur non-object and preserve context)
def blur_blackout(frame, selected_object, conf_thresh):
    context_classes = [
        'road','lane','street','ground','path','parking',
        'sidewalk','crosswalk','traffic light','traffic sign',
        'highway','bridge','tunnel'
    ]
    results = model(frame, conf=conf_thresh)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    context_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        conf = det.conf[0].cpu().item()
        if label == selected_object and conf > conf_thresh:
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
        if any(cc in label.lower() for cc in context_classes):
            cv2.rectangle(context_mask, (x1,y1), (x2,y2), 255, -1)
    combined = cv2.bitwise_or(mask, context_mask)
    inverse = cv2.bitwise_not(combined)
    blurred = cv2.GaussianBlur(frame, (35,35), 0)
    darkened = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)
    background = cv2.addWeighted(blurred, 0.7, darkened, 0.3, 0)
    result = frame.copy()
    result[inverse == 255] = background[inverse == 255]

    detected = []
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        conf = det.conf[0].cpu().item()
        if label == selected_object:
            detected.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": float(conf), "label": label})
            cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(result, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    return detected, result

# Technique 4: Complete Inpainting (remove background & inpaint)
def complete_inpainting(frame, selected_object, conf_thresh):
    results = model(frame, conf=conf_thresh)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    context_classes = ['road','lane','street','ground','path']
    context_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        if label == selected_object:
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
        if any(cc in label.lower() for cc in context_classes):
            cv2.rectangle(context_mask, (x1,y1), (x2,y2), 255, -1)
    combined = cv2.bitwise_or(cv2.bitwise_not(mask), context_mask)
    inpaint_ns = cv2.inpaint(frame, combined, 3, cv2.INPAINT_NS)
    inpaint_tl = cv2.inpaint(frame, combined, 3, cv2.INPAINT_TELEA)
    result = cv2.addWeighted(inpaint_ns, 0.5, inpaint_tl, 0.5, 0)

    detected = []
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        conf = det.conf[0].cpu().item()
        if label == selected_object:
            detected.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": float(conf), "label": label})
            cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(result, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    return detected, result

# Technique 5: Gaussian Blur background
def gaussian_blur(frame, selected_object, conf_thresh):
    results = model(frame, conf=conf_thresh)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        if label == selected_object:
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
    blurred = cv2.GaussianBlur(frame, (21,21), 0)
    result = frame.copy()
    result[mask == 0] = blurred[mask == 0]

    detected = []
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        conf = det.conf[0].cpu().item()
        if label == selected_object:
            detected.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": float(conf), "label": label})
            cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(result, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    return detected, result

# Technique 6: Object Inpainting (remove non-selected via Telea)
def object_inpainting(frame, selected_object, conf_thresh):
    results = model(frame, conf=conf_thresh)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        if label != selected_object:
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
    result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

    detected = []
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        conf = det.conf[0].cpu().item()
        if label == selected_object:
            detected.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": float(conf), "label": label})
            cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(result, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    return detected, result

# Technique 7: Selective Blur (texture replacement)
def selective_blur(frame, selected_object, conf_thresh):
    results = model(frame, conf=conf_thresh)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        if label != selected_object:
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
    replacement = cv2.medianBlur(frame, 15)
    result = frame.copy()
    result[mask == 255] = replacement[mask == 255]

    detected = []
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        label = class_names[int(det.cls[0].cpu().item())]
        conf = det.conf[0].cpu().item()
        if label == selected_object:
            detected.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": float(conf), "label": label})
            cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(result, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    return detected, result

# Mapping techniques
TECHNIQUES = {
    'Black & White': bw_mask,
    'Complete Blackout': complete_blackout,
    'Blur Blackout': blur_blackout,
    'Complete Inpainting': complete_inpainting,
    'Gaussian Blur': gaussian_blur,
    'Object Inpainting': object_inpainting,
    'Selective Blur': selective_blur
}

# API endpoints
@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available object classes"""
    return jsonify({"classes": class_names})

@app.route('/api/techniques', methods=['GET'])
def get_techniques():
    """Get available masking techniques"""
    return jsonify({"techniques": list(TECHNIQUES.keys())})

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Save uploaded file
    file.save(file_path)
    
    # Get video metadata
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return jsonify({
        "file_id": unique_filename,
        "metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        }
    })

@app.route('/api/process', methods=['POST'])
def process_video():
    """Process video with selected technique and object"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    file_id = data.get('file_id')
    selected_object = data.get('selected_object')
    selected_technique = data.get('selected_technique')
    confidence_threshold = float(data.get('confidence_threshold', 0.5))
    
    if not file_id or not selected_object or not selected_technique:
        return jsonify({"error": "Missing required parameters"}), 400
    
    if selected_technique not in TECHNIQUES:
        return jsonify({"error": f"Unknown technique: {selected_technique}"}), 400
    
    if selected_object not in class_names:
        return jsonify({"error": f"Unknown object class: {selected_object}"}), 400
    
    input_path = os.path.join(UPLOAD_FOLDER, file_id)
    if not os.path.exists(input_path):
        return jsonify({"error": "File not found"}), 404
    
    # Create unique output filename
    output_filename = f"{uuid.uuid4()}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    # Create a job ID for status tracking
    job_id = str(uuid.uuid4())
    
    # Start processing in a separate thread/process
    # For simplicity, we'll just process it here (blocking)
    # In production, you'd use Celery, RQ, or similar
    
    # Process video frame by frame
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    detected_objects = []
    frame_count = 0
    
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with selected technique
        detected, processed_frame = TECHNIQUES[selected_technique](frame, selected_object, confidence_threshold)
        
        if detected:
            detected_objects.extend([{"frame": frame_count, **det} for det in detected])
            out.write(processed_frame)
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    
    # Save detection results
    results_filename = f"{os.path.splitext(output_filename)[0]}_detections.json"
    results_path = os.path.join(OUTPUT_FOLDER, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(detected_objects, f)
    
    return jsonify({
        "status": "success",
        "output_video": output_filename,
        "detections_file": results_filename,
        "stats": {
            "total_frames": total_frames,
            "frames_with_detections": len(set(det["frame"] for det in detected_objects)),
            "total_objects_detected": len(detected_objects),
            "processing_time": processing_time
        }
    })

@app.route('/api/video/<filename>', methods=['GET'])
def get_video(filename):
    """Serve processed video"""
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype='video/mp4')

@app.route('/api/detections/<filename>', methods=['GET'])
def get_detections(filename):
    """Get detection results"""
    results_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(results_path):
        return jsonify({"error": "Detections file not found"}), 404
    
    with open(results_path, 'r') as f:
        detections = json.load(f)
    
    return jsonify({"detections": detections})

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get processing job status"""
    # In a real implementation, you'd use a job queue system
    # For now, we'll just return a mock response
    return jsonify({
        "status": "complete",
        "progress": 100
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)