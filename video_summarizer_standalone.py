import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# Load custom YOLO model
model = YOLO('best_yolo11.pt')

# Define class names
class_names = [
    'trak', 'cyclist', 'bike', 'tempo', 'car', 'zeep', 'toto',
    'e-rickshaw', 'auto-rickshaw', 'bus', 'van', 'cycle-rickshaw',
    'person', 'taxi'
]

# Default confidence threshold
CONFIDENCE_THRESHOLD = 0.5

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
            detected.append((x1,y1,x2,y2, conf, label))
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
            detected.append((x1,y1,x2,y2, conf, label))
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
            detected.append((x1,y1,x2,y2,conf,label))
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
            detected.append((x1,y1,x2,y2,conf,label))
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
            detected.append((x1,y1,x2,y2,conf,label))
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
            detected.append((x1,y1,x2,y2,conf,label))
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
            detected.append((x1,y1,x2,y2,conf,label))
            cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(result, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    return detected, result

# Mapping techniques
tECHNIQUES = {
    'Black & White': bw_mask,
    'Complete Blackout': complete_blackout,
    'Blur Blackout': blur_blackout,
    'Complete Inpainting': complete_inpainting,
    'Gaussian Blur': gaussian_blur,
    'Object Inpainting': object_inpainting,
    'Selective Blur': selective_blur
}

# Streamlit UI
st.title("Unified Object Masking App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])
selected_object = st.selectbox("Select object to isolate", class_names)
selected_tech = st.selectbox("Masking Technique", list(tECHNIQUES.keys()))

# Confidence slider
CONFIDENCE_THRESHOLD = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=1.0,
    value=0.5, step=0.05
)

if uploaded_file and st.button("Process Video"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    progress = st.progress(0)
    frames = []
    count = 0

    # Process video frame by frame
    cap = cv2.VideoCapture(input_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detected, out_frame = tECHNIQUES[selected_tech](frame, selected_object, CONFIDENCE_THRESHOLD)
        if detected:
            frames.append(out_frame)
        count += 1
        progress.progress(min(count/total, 1.0))
    cap.release()

    # Save and display
    if frames:
        output_path = 'output.mp4'
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
        for f in frames:
            writer.write(f)
        writer.release()
        st.success(f"Processed video saved as {output_path}")
        st.video(output_path)
    else:
        st.error(f"No {selected_object} detected in the video.")