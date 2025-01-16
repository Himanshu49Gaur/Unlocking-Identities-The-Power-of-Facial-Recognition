from PIL import ImageEnhance, Image
import face_recognition
import tempfile
import streamlit as st
import numpy as np
from pathlib import Path

TOLERANCE = 0.6
TRAINING_FOLDER = Path("training_data")  # Change this to your training folder path

def augment_image(image_pil: Image.Image) -> Image.Image:
    """Applies random augmentations to the input image."""
    enhancer = ImageEnhance.Brightness(image_pil)                               # Apply brightness enhancement
    image_pil = enhancer.enhance(1.2)                                           # Slightly increase brightness
    image_pil = image_pil.rotate(15, expand=True)                               # Apply rotation
    return image_pil

def encode_known_faces(model: str = "hog", tolerance: float = TOLERANCE, min_face_size: int = 50, confidence: float = 0.6) -> dict:
    """Encodes known faces and returns the encodings."""
    names = []
    encodings = []
    image_paths = list(TRAINING_FOLDER.glob("*/*"))
    if not image_paths:
        st.warning("No training images found. Add images to the training folder.")
        return {}
    progress = st.progress(0)
    supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    for idx, filepath in enumerate(image_paths):
        if not filepath.suffix.lower() in supported_formats:
            continue
        name = filepath.parent.name
        try:
            image = face_recognition.load_image_file(filepath)
            image_pil = Image.fromarray(image)
            
            augmented_image = augment_image(image_pil)                            # Apply augmentation
            
            # Save augmented image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                augmented_image.save(temp_file.name)
                augmented_image_path = temp_file.name
            
            # Reload the augmented image for face_recognition
            augmented_image_data = face_recognition.load_image_file(augmented_image_path)
        except Exception as e:
            st.error(f"Error processing image {filepath}: {e}")
            continue
        face_locations = face_recognition.face_locations(augmented_image_data, model=model)
        face_encodings = face_recognition.face_encodings(augmented_image_data, face_locations, num_jitters=2)
        for face_location, encoding in zip(face_locations, face_encodings):
            # Check if the detected face size meets the minimum face size requirement
            top, right, bottom, left = face_location
            face_height = bottom - top
            face_width = right - left
            if face_height >= min_face_size and face_width >= min_face_size:
                names.append(name)
                encodings.append(encoding)
        progress.progress((idx + 1) / len(image_paths))
    progress.empty()
    return {"names": names, "encodings": encodings}
