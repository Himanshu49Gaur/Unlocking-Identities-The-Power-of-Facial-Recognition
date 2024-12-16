def recognize_faces(image_location: str, encodings_location: Path = DEFAULT_ENCODINGS_PATH):
    """Detects faces in the given image and recognizes them."""
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(input_image, model="cnn")
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    if not input_face_locations:
        return None, "No faces detected.", [], 0, 0

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)
    detected_names = []

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "unknown"
        detected_names.append(name)
        _display_face(draw, bounding_box, name)

    del draw
    recognized_faces = sum(1 for name in detected_names if name != "unknown")
    return pillow_image, f"Detected {len(input_face_locations)} face(s).", detected_names, len(input_face_locations), recognized_faces


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]
    return None


def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR, width=BOUNDING_BOX_WIDTH)

    try:
        text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
        draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=BOUNDING_BOX_COLOR)
        draw.text((text_left, text_top), name, fill=TEXT_COLOR)
    except AttributeError:
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom), (left + text_width, bottom + text_height)), fill=BOUNDING_BOX_COLOR)
        draw.text((left, bottom), name, fill=TEXT_COLOR)
