import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw
import time

# --------------------------------------------------
# Load the pre-trained object detection model from TensorFlow Hub.
# This model (SSD MobileNet V2 FPNLite 320x320) is known for reasonable accuracy.
# --------------------------------------------------
@st.cache_resource()
def load_detector():
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
    detector = hub.load(model_url)
    return detector

detector = load_detector()

# --------------------------------------------------
# COCO Labels (90 classes)
# --------------------------------------------------
COCO_LABELS = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --------------------------------------------------
# UI: Mode Selection
# --------------------------------------------------
st.title("🚀 Real-Time Object Detection Demo")
st.write("Select a mode below to use the object detection model.")

mode = st.radio("Select Mode:", ["Image Upload", "Live Feed"])

# --------------------------------------------------
# Mode 1: Image Upload
# --------------------------------------------------
if mode == "Image Upload":
    uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert image to numpy array
        image_np = np.array(image)
        
        # Run detection:
        # The TF Hub model expects a uint8 tensor with shape [1, height, width, 3]
        input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)[tf.newaxis, ...]
        results = detector(input_tensor)
        results = {key: value.numpy() for key, value in results.items()}
        
        detection_scores = results['detection_scores'][0]
        detection_boxes = results['detection_boxes'][0]
        detection_classes = results['detection_classes'][0].astype(np.int32)
        threshold = 0.5
        
        # Draw detections on the image
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        for i, score in enumerate(detection_scores):
            if score > threshold:
                box = detection_boxes[i]  # [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = box
                left = int(xmin * im_width)
                right = int(xmax * im_width)
                top = int(ymin * im_height)
                bottom = int(ymax * im_height)
                draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)
                label = COCO_LABELS[detection_classes[i]]
                draw.text((left, top), label, fill="red")
        st.image(image, caption="Detected Image", use_column_width=True)

# --------------------------------------------------
# Mode 2: Live Feed (Continuous Webcam)
# --------------------------------------------------
elif mode == "Live Feed":
    st.write("Live feed is active. Click 'Stop Live Feed' to end the demo.")
    
    # Placeholder for video frames
    frame_placeholder = st.empty()
    
    # Use session state to control live feed loop
    if 'live_feed' not in st.session_state:
        st.session_state.live_feed = True
        
    # Create a "Stop Live Feed" button outside the loop
    if st.button("Stop Live Feed"):
        st.session_state.live_feed = False
    
    # Open the webcam (device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the webcam.")
    else:
        while st.session_state.live_feed:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera.")
                break
            
            # Convert the frame from BGR (OpenCV format) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection on the current frame
            input_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)[tf.newaxis, ...]
            results = detector(input_tensor)
            results = {key: value.numpy() for key, value in results.items()}
            
            detection_scores = results['detection_scores'][0]
            detection_boxes = results['detection_boxes'][0]
            detection_classes = results['detection_classes'][0].astype(np.int32)
            threshold = 0.5
            
            # Draw detections on the frame
            for i, score in enumerate(detection_scores):
                if score > threshold:
                    box = detection_boxes[i]  # [ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = box
                    im_height, im_width, _ = frame_rgb.shape
                    left = int(xmin * im_width)
                    right = int(xmax * im_width)
                    top = int(ymin * im_height)
                    bottom = int(ymax * im_height)
                    cv2.rectangle(frame_rgb, (left, top), (right, bottom), (255, 0, 0), 2)
                    label = COCO_LABELS[detection_classes[i]]
                    cv2.putText(frame_rgb, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Update the frame placeholder with the processed frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Add a small delay to allow UI updates
            time.sleep(0.05)
        
        cap.release()
        st.write("Live feed stopped.")
