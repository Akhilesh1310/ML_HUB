# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Cache model loading to avoid reloading on every run
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('models/object_detection_model.h5')
    return model

model = load_model()

# List of VOC 2012 class names (20 classes)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

st.title("Object Detection Demo")
st.write("Upload an image and let the model detect objects for you!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image: resize to 224x224 (or the expected model input size)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # normalize pixels to [0, 1]
    input_tensor = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Run inference using the loaded model
    # Assuming the model outputs a classification and a bounding box
    pred_class, pred_bbox = model.predict(input_tensor)
    
    # Get predicted class and bounding box.
    # For classification, we take the argmax across the VOC classes.
    class_idx = np.argmax(pred_class, axis=-1)[0]
    predicted_label = VOC_CLASSES[class_idx]
    
    # The bounding box is assumed normalized [0,1] for [x_min, y_min, x_max, y_max]
    bbox = pred_bbox[0]  # shape: (4,)
    
    # Scale the normalized bounding box back to original image dimensions
    orig_width, orig_height = image.size
    x_min = int(bbox[0] * orig_width)
    y_min = int(bbox[1] * orig_height)
    x_max = int(bbox[2] * orig_width)
    y_max = int(bbox[3] * orig_height)
    
    st.write("Predicted Class:", predicted_label)
    st.write("Bounding Box (pixels):", (x_min, y_min, x_max, y_max))
    
    # Draw the bounding box and label on the image
    image_with_box = image.copy()
    draw = ImageDraw.Draw(image_with_box)
    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)
    draw.text((x_min, max(y_min - 15, 0)), predicted_label, fill="red")
    
    st.image(image_with_box, caption="Detected Image", use_column_width=True)
