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

def create_mask_and_context(frame, object_filter):
    """
    Create a mask for the selected object and preserve context
    """
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect objects
    results = model(rgb_frame)[0]
    
    # Create a blank mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    context_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Road-like classes to preserve context
    context_classes = ['road', 'lane', 'street', 'ground', 'path']
    
    for detection in results.boxes:
        x1, y1, x2, y2 = detection.xyxy[0].numpy().astype(int)
        cls = int(detection.cls[0].item())
        label = object_labels[cls]
        
        # If the object matches the filter, create a mask
        if label == object_filter:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Preserve context for road-like classes
        if any(context_class in label.lower() for context_class in context_classes):
            cv2.rectangle(context_mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask, context_mask

def inpaint_frame(frame, mask, context_mask):
    """
    Inpaint the frame while preserving context
    """
    # Combine masks, giving priority to context
    combined_mask = cv2.bitwise_or(
        cv2.bitwise_not(mask), 
        context_mask
    )
    
    # Inpaint using both Navier-Stokes and Telea methods
    inpainted_ns = cv2.inpaint(frame, combined_mask, 3, cv2.INPAINT_NS)
    inpainted_telea = cv2.inpaint(frame, combined_mask, 3, cv2.INPAINT_TELEA)
    
    # Blend the two inpainting methods
    inpainted_frame = cv2.addWeighted(inpainted_ns, 0.5, inpainted_telea, 0.5, 0)
    
    return inpainted_frame

def process_video(input_path, object_filter, progress_bar):
    """
    Process the video to focus on selected object
    """
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create masks
        object_mask, context_mask = create_mask_and_context(frame, object_filter)
        
        # Check if the object is present in this frame
        if np.any(object_mask):
            # Inpaint the frame
            processed_frame = inpaint_frame(frame, object_mask, context_mask)
            output_frames.append(processed_frame)
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    cap.release()
    return output_frames

def save_video(frames, output_path, fps):
    """
    Save processed frames as a video
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
    st.title("Selective Object Focus in Video")
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    object_filter = st.selectbox("Select object to focus on", object_labels)
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
        output_frames = process_video(input_path, object_filter, progress_bar)

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