import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import torch

class AdvancedObjectIsolator:
    def __init__(self, model_path='yolov8n.pt'):
        # Use small YOLO model
        self.model = YOLO(model_path)
        
        # Preset parameters
        self.confidence_threshold = 0.5
        self.max_objects_per_frame = 10

    def create_pixel_replacement_mask(self, frame, selected_object):
        """
        Create a mask for pixel replacement strategy
        """
        # Detect objects
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Create a blank mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for detection in results[0].boxes:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            cls = int(detection.cls[0].cpu().item())
            label = self.model.names[cls]
            
            # If not the selected object, mark for replacement
            if label != selected_object:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask, results

    def replace_pixels(self, frame, selected_object):
        """
        Replace pixels of non-selected objects with textured background
        """
        # Create replacement mask
        mask, results = self.create_pixel_replacement_mask(frame, selected_object)
        
        # Create a copy of the frame for modification
        result = frame.copy()
        
        # Create a textured replacement using median filtering
        replacement = cv2.medianBlur(frame, 15)
        
        # Replace pixels where mask is white (non-selected objects)
        result[mask == 255] = replacement[mask == 255]
        
        # Detect and highlight selected objects
        for detection in results[0].boxes:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            conf = detection.conf[0].cpu().item()
            cls = int(detection.cls[0].cpu().item())
            label = self.model.names[cls]
            
            # Highlight only selected objects
            if label == selected_object:
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result, 
                            f"{label}: {conf:.2f}", 
                            (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.45, 
                            (0, 255, 0), 
                            2)
        
        return result

    def process_frame(self, frame, selected_object):
        # Resize frame to balance performance and quality
        frame = cv2.resize(frame, (800, 450))
        
        # Process frame to replace non-selected objects
        processed_frame = self.replace_pixels(frame, selected_object)
        
        # Detect objects for tracking
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Track detected objects of selected type
        detected_objects = []
        for detection in results[0].boxes:
            cls = int(detection.cls[0].cpu().item())
            label = self.model.names[cls]
            
            if label == selected_object:
                conf = detection.conf[0].cpu().item()
                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                
                detected_objects.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return detected_objects, processed_frame

# Rest of the code remains the same as in the original implementation
# (process_video, save_video, and main() functions)


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
    object_labels = [isolator.model.names[i] for i in range(len(isolator.model.names))]

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