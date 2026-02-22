import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import torch

class AdvancedObjectIsolator:
    def __init__(self, model_path='yolov8n-seg.pt'):
        # Use YOLOv8 Instance Segmentation model
        self.model = YOLO(model_path)
        
        # Preset parameters
        self.confidence_threshold = 0.5
        self.max_objects_per_frame = 10
        
        # Store class names
        self.class_names = self.model.names

    def remove_other_objects(self, frame, selected_object):
        """
        Remove other objects using instance segmentation masks
        """
        # Detect objects with segmentation
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Create a copy of the frame for modification
        result = frame.copy()
        
        # Create a full black background
        black_background = np.zeros_like(frame)
        
        # Process each detection
        for detection in results[0].boxes:
            # Get class label
            cls = int(detection.cls[0].cpu().item())
            label = self.class_names[cls]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            conf = detection.conf[0].cpu().item()
            
            # If this is the selected object, add to black background
            if label == selected_object:
                # Create mask for this specific object
                object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.rectangle(object_mask, (x1, y1), (x2, y2), 255, -1)
                
                # Extract object with original color
                object_region = cv2.bitwise_and(frame, frame, mask=object_mask)
                black_background = cv2.add(black_background, object_region)
                
                # Add bounding box and label
                cv2.rectangle(black_background, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(black_background, 
                            f"{label}: {conf:.2f}", 
                            (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.45, 
                            (0, 255, 0), 
                            2)
        
        return black_background

    def process_frame(self, frame, selected_object):
        # Resize frame to balance performance and quality
        frame = cv2.resize(frame, (800, 450))
        
        # Process frame to remove other objects
        processed_frame = self.remove_other_objects(frame, selected_object)
        
        # Detect objects for tracking
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Track detected objects of selected type
        detected_objects = []
        for detection in results[0].boxes:
            cls = int(detection.cls[0].cpu().item())
            label = self.class_names[cls]
            
            if label == selected_object:
                conf = detection.conf[0].cpu().item()
                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                
                detected_objects.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return detected_objects, processed_frame

def process_video(isolator, input_path, object_filter, progress_bar):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        try:
            detected_objects, processed_frame = isolator.process_frame(
                frame, 
                object_filter
            )

            # Only keep frames with detected objects
            if detected_objects:
                output_frames.append(processed_frame)

            # Update progress
            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        except Exception as e:
            st.error(f"Error processing frame: {e}")
            break

    cap.release()
    return output_frames

def save_video(frames, output_path, fps):
    if not frames:
        return False
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return True

def main():
    st.title("Advanced Object Isolation in Video")
    
    # Initialize the isolator
    isolator = AdvancedObjectIsolator()

    # Supported objects
    object_labels = list(isolator.class_names.values())

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a video", 
        type=["mp4", "avi", "mov", "mkv"]
    )

    # Object selection
    object_filter = st.selectbox(
        "Select object to isolate", 
        object_labels
    )

    # Process button
    process_button = st.button("Process Video")

    # Configuration sidebar
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1
    )
    max_objects = st.sidebar.slider(
        "Max Objects per Frame", 
        min_value=1, 
        max_value=20, 
        value=10, 
        step=1
    )

    if uploaded_file is not None and process_button:
        # Update isolator settings
        isolator.confidence_threshold = confidence_threshold
        isolator.max_objects_per_frame = max_objects

        # Temporary file handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1]) as tmp_file:
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
            isolator, 
            input_path, 
            object_filter, 
            progress_bar
        )

        # Save and display results
        if output_frames:
            output_path = "output.mp4"
            if save_video(output_frames, output_path, fps):
                st.success(f"Processed video saved as '{output_path}'")
                st.video(output_path)
            else:
                st.error("Video saving failed.")
        else:
            st.error(f"No {object_filter} detected in the video.")

if __name__ == "__main__":
    main()