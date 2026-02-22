import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO

# Check if CUDA is available and select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model for detection
model = YOLO('yolov8n.pt')

# Object labels supported by YOLOv8
object_labels = [model.names[i] for i in range(len(model.names))]

def create_selective_mask(frame, object_filter, context_preservation=True):
    """
    Create an advanced mask for the selected object with intelligent context preservation
    """
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect objects
    results = model(rgb_frame)[0]
    
    # Create masks
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    context_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Stores selected object bounding boxes
    selected_objects = []
    
    # Enhanced context preservation classes
    context_classes = [
        'road', 'lane', 'street', 'ground', 'path', 'parking', 
        'sidewalk', 'crosswalk', 'traffic light', 'traffic sign', 
        'highway', 'bridge', 'tunnel'
    ]
    
    # Exclusion classes (objects to completely remove)
    exclusion_classes = [
        'person', 'bicycle', 'motorcycle', 'car', 
        'bus', 'train', 'boat', 'skateboard'
    ]
    
    for detection in results.boxes:
        x1, y1, x2, y2 = detection.xyxy[0].numpy().astype(int)
        cls = int(detection.cls[0].item())
        label = model.names[cls]
        confidence = detection.conf[0].item()
        
        # Highlight selected object
        if label == object_filter and confidence > 0.5:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            selected_objects.append((x1, y1, x2, y2))
        
        # Intelligent context preservation
        if context_preservation:
            # Preserve context classes
            if any(context_class in label.lower() for context_class in context_classes):
                cv2.rectangle(context_mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask, context_mask, selected_objects

def process_frame(frame, mask, context_mask, selected_objects, blur_intensity=35, remove_others=False):
    """
    Advanced frame processing with smooth blending and context preservation
    """
    # Create a copy of the frame
    processed_frame = frame.copy()
    
    # Combine masks with preference to object mask
    context_and_object_mask = cv2.bitwise_or(mask, context_mask)
    
    # Create an inverse mask for non-selected regions
    inverse_mask = cv2.bitwise_not(context_and_object_mask)
    
    if remove_others:
        # Completely black out non-selected and non-context regions
        processed_frame[inverse_mask == 255] = [0, 0, 0]
    else:
        # Advanced blurring with different techniques
        # 1. Gaussian blur for smooth background
        blurred_frame = cv2.GaussianBlur(processed_frame, (blur_intensity, blur_intensity), 0)
        
        # 2. Add slight darkening to non-selected regions
        darkened_frame = cv2.addWeighted(processed_frame, 0.3, np.zeros_like(processed_frame), 0.7, 0)
        
        # Blend blurred and darkened frames
        background_frame = cv2.addWeighted(blurred_frame, 0.7, darkened_frame, 0.3, 0)
        
        # Replace non-selected regions with blended background
        processed_frame[inverse_mask == 255] = background_frame[inverse_mask == 255]
    
    # Highlight selected objects with dynamic bounding boxes
    for (x1, y1, x2, y2) in selected_objects:
        # Gradient green bounding box
        for i in range(3):
            thickness = max(1, 3 - i)
            color = (0, 255 - i*50, 0)  # Gradient from bright to dark green
            cv2.rectangle(processed_frame, (x1-i, y1-i), (x2+i, y2+i), color, thickness)
    
    return processed_frame

def process_video(input_path, object_filter, progress_bar, blur_intensity=35, remove_others=False):
    """
    Process the video to focus on selected object with advanced techniques
    """
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create masks and get selected object bounding boxes
        object_mask, context_mask, selected_objects = create_selective_mask(frame, object_filter)
        
        # Check if the object is present in this frame
        if np.any(object_mask):
            # Process the frame with advanced highlighting
            processed_frame = process_frame(
                frame, 
                object_mask, 
                context_mask, 
                selected_objects, 
                blur_intensity=blur_intensity,
                remove_others=remove_others
            )
            output_frames.append(processed_frame)
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    cap.release()
    return output_frames

def save_video(frames, output_path, fps):
    """
    Save processed frames as a video with optimized settings
    """
    if not frames:
        return False
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    return True

# Streamlit UI
def main():
    st.title("Intelligent Object Focus in Video")
    
    st.markdown("""
    ### How to Use
    1. Upload a video file
    2. Select the object you want to focus on
    3. Customize processing options
    4. Click 'Process Video' to generate a highlight
    
    ℹ️ The application will:
    - Detect and highlight your selected object
    - Optionally preserve contextual elements
    - Blur or remove other objects
    """)
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    
    # Custom object selection with description
    object_filter = st.selectbox(
        "Select object to focus on", 
        object_labels, 
        index=object_labels.index('truck') if 'truck' in object_labels else 0,
        help="Choose the object type you want to highlight in the video"
    )
    
    # Advanced processing options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        context_preservation = st.checkbox(
            "Preserve Context", 
            value=True, 
            help="Keep contextual elements like roads and traffic signs"
        )
    
    with col2:
        remove_others = st.checkbox(
            "Remove Other Objects", 
            value=False, 
            help="Completely remove non-context objects instead of blurring"
        )
    
    with col3:
        blur_intensity = st.slider(
            "Background Blur", 
            min_value=1, 
            max_value=51, 
            value=35, 
            step=2,
            help="Intensity of blur for non-selected objects",
            disabled=remove_others
        )
    
    process_button = st.button("Process Video")

    if uploaded_file is not None and process_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            input_path = tmp_file.name

        # Get video properties
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Progress bar
        progress_bar = st.progress(0)

        # Process video
        output_frames = process_video(
            input_path, 
            object_filter, 
            progress_bar, 
            blur_intensity=blur_intensity,
            remove_others=remove_others
        )

        if output_frames:
            output_path = "focused_object_video.mp4"
            if save_video(output_frames, output_path, fps):
                st.success(f"Processed video saved as '{output_path}'")
                st.video(output_path)
            else:
                st.error("Video processing failed.")
        else:
            st.error("No frames matched the selected filter.")

if __name__ == "__main__":
    main()