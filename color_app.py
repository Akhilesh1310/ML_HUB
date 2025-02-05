import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🎨 Real-Time Color Tracking")

# Define color ranges in HSV
color_ranges = {
    "Red": [(0, 120, 70), (10, 255, 255)],
    "Green": [(40, 40, 40), (80, 255, 255)],
    "Blue": [(90, 50, 50), (130, 255, 255)],
    "Yellow": [(20, 100, 100), (30, 255, 255)]
}

# Function to detect color
def detect_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    output = image.copy()
    
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output, color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return output

# Upload an image or use webcam
option = st.radio("Choose an option:", ("Upload Image", "Live Camera"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        result = detect_color(image)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

elif option == "Live Camera":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    if not cap.isOpened():
        st.error("Error: Could not access the camera!")
    else:
        stop_button = st.button("Stop Camera")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame!")
                break

            result = detect_color(frame)
            stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)

            if stop_button:
                break

        cap.release()
