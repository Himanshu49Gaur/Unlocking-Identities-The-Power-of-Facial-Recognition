def augment_image(image_pil: Image.Image) -> Image.Image:
    """Applies random augmentations to the input image."""
    # Apply brightness enhancement
    enhancer = ImageEnhance.Brightness(image_pil)
    image_pil = enhancer.enhance(1.2)  # Slightly increase brightness

    # Apply rotation (randomly between -15 and 15 degrees)
    image_pil = image_pil.rotate(15, expand=True)

    return image_pil


def encode_known_faces(model: str = "hog", tolerance: float = TOLERANCE) -> dict:
    """Encodes known faces and returns the encodings."""
    names = []
    encodings = []
    image_paths = list(TRAINING_FOLDER.glob("*/*"))

    if not image_paths:
        st.warning("No training images found. Add images to the training folder.")
        return {}

    progress = st.progress(0)

    for idx, filepath in enumerate(image_paths):
        if not filepath.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            continue
        name = filepath.parent.name
        try:
            image = face_recognition.load_image_file(filepath)
            image_pil = Image.fromarray(image)
            
            # Apply augmentation
            augmented_image = augment_image(image_pil)
            
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

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

        progress.progress((idx + 1) / len(image_paths))

    progress.empty()
    return {"names": names, "encodings": encodings}