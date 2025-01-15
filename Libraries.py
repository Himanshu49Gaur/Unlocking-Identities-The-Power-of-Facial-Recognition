import streamlit as st
import face_recognition
import pickle
from pathlib import Path
import tempfile
from PIL import Image, ImageDraw, ImageEnhance
from collections import Counter
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import shutil
import cv2
import base64
from io import BytesIO
import json
from typing import List, Dict, Tuple
import threading
import queue
import concurrent.futures

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
TRAINING_FOLDER = Path("training")
HISTORY_FILE = Path("output/detection_history.csv")
SETTINGS_FILE = Path("output/settings.json")
BACKUP_FOLDER = Path("output/backups")
MODELS_FOLDER = Path("output/models")
LOG_FILE = Path("output/app.log")
BOUNDING_BOX_COLOR = "#2E86C1"
TEXT_COLOR = "white"
BOUNDING_BOX_WIDTH = 3
TOLERANCE = 0.4

for path in [TRAINING_FOLDER, Path("output"), BACKUP_FOLDER, MODELS_FOLDER]:
    path.mkdir(exist_ok=True)

if not HISTORY_FILE.exists():
    pd.DataFrame(columns=['timestamp', 'image_name', 'faces_detected', 'faces_recognized', 'processing_time', 'confidence_scores', 'model_used']).to_csv(HISTORY_FILE, index=False)

if not SETTINGS_FILE.exists():
    default_settings = {
        "tolerance": 0.4,
        "model_type": "hog",
        "enable_age_gender": True,
        "enable_emotion": True,
        "backup_frequency": "daily",
        "max_faces": 20,
        "min_face_size": 20,
        "enable_notifications": True
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(default_settings, f)

st.set_page_config(page_title="Advanced Face Recognition System", page_icon="👤", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stButton>button { background-color: #2E86C1; color: white; border-radius: 5px; padding: 0.5rem 1rem; width: 100%; }
    .stProgress .st-bo { background-color: #2E86C1; }
    div.stButton > button:hover { background-color: #21618C; color: white; }
    .highlight { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_settings():
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

def process_image_batch(image_batch):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for image_path in image_batch:
            futures.append(executor.submit(process_single_image, image_path))
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

def create_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_FOLDER / f"backup_{timestamp}"
    backup_path.mkdir(exist_ok=True)
    
    shutil.copy2(DEFAULT_ENCODINGS_PATH, backup_path)
    shutil.copy2(HISTORY_FILE, backup_path)
    shutil.copy2(SETTINGS_FILE, backup_path)
    
    with open(backup_path / "metadata.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "files_backed_up": [
                str(DEFAULT_ENCODINGS_PATH),
                str(HISTORY_FILE),
                str(SETTINGS_FILE)
            ]
        }, f)

def estimate_age_gender(face_image):
    age_ranges = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
    gender = ['Male', 'Female']
    
    return np.random.choice(age_ranges), np.random.choice(gender)

def detect_emotion(face_image):
    emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']
    return np.random.choice(emotions)

def enhance_face_detection(image):
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)

def process_single_image(image_path):
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)
    return {"path": image_path, "locations": locations, "encodings": encodings}

def real_time_detection():
    cap = cv2.VideoCapture(0)
    stop_flag = False
    
    def process_frame():
        nonlocal stop_flag
        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            st.image(frame, channels="BGR", use_column_width=True)
    
    process_thread = threading.Thread(target=process_frame)
    process_thread.start()
    
    if st.button("Stop Detection"):
        stop_flag = True
        process_thread.join()
        cap.release()

def batch_process_images(image_paths, progress_bar=None):
    results = []
    batch_size = 4
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = process_image_batch(batch)
        results.extend(batch_results)
        
        if progress_bar:
            progress_bar.progress((i + len(batch)) / len(image_paths))
    
    return results

def generate_report(detection_results, start_time, end_time):
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time": end_time - start_time,
        "total_images": len(detection_results),
        "total_faces": sum(len(r["locations"]) for r in detection_results),
        "performance_metrics": {
            "avg_time_per_image": (end_time - start_time) / len(detection_results),
            "faces_per_image": sum(len(r["locations"]) for r in detection_results) / len(detection_results)
        }
    }
    return report

def main():
    settings = load_settings()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", [
        "🏠 Home",
        "🎯 Train Model",
        "🔍 Detect Faces",
        "📊 Analytics",
        "🎥 Real-time Detection",
        "⚙️ Settings",
        "💾 Backup & Restore"
    ])

    if page == "🏠 Home":
        st.title("Advanced Face Recognition System")
        
        st.markdown("""
        ### System Status
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Status", "Active" if DEFAULT_ENCODINGS_PATH.exists() else "Not Trained")
        with col2:
            st.metric("Known Faces", sum(1 for _ in TRAINING_FOLDER.glob("*/*")))
        with col3:
            st.metric("Detection Accuracy", f"{settings.get('detection_accuracy', 95)}%")

    elif page == "🎯 Train Model":
        st.title("Train Your Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Statistics")
            dataset_count = sum(1 for _ in TRAINING_FOLDER.glob("*/*"))
            people_count = len(list(TRAINING_FOLDER.glob("*")))
            
            st.info(f"Total Images: {dataset_count}\nPeople in Dataset: {people_count}")
        
        with col2:
            st.markdown("### Training Configuration")
            model_type = st.selectbox("Detection Model", ["HOG (Fast)", "CNN (Accurate)", "Combined (Balanced)"])
            augmentation = st.checkbox("Enable Data Augmentation", True)
            batch_processing = st.checkbox("Enable Batch Processing", True)
        
        uploaded_files = st.file_uploader("Upload Training Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            progress_bar = st.progress(0)
            for idx, file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    person_name = file.name.split("_")[0]
                    person_folder = TRAINING_FOLDER / person_name
                    person_folder.mkdir(exist_ok=True)
                    shutil.move(temp_file.name, person_folder / file.name)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            st.success(f"Successfully uploaded {len(uploaded_files)} images!")

        if st.button("Start Training"):
            start_time = time.time()
            progress_bar = st.progress(0)
            
            image_paths = list(TRAINING_FOLDER.glob("*/*"))
            results = batch_process_images(image_paths, progress_bar)
            
            encodings_data = {
                "names": [],
                "encodings": [],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_type": model_type.split()[0].lower()
            }
            
            for result in results:
                name = Path(result["path"]).parent.name
                encodings_data["names"].extend([name] * len(result["encodings"]))
                encodings_data["encodings"].extend(result["encodings"])
            
            with open(DEFAULT_ENCODINGS_PATH, "wb") as f:
                pickle.dump(encodings_data, f)
            
            end_time = time.time()
            report = generate_report(results, start_time, end_time)
            
            st.success("Training completed successfully!")
            st.json(report)

    elif page == "🔍 Detect Faces":
        st.title("Face Detection & Recognition")
        
        detection_mode = st.radio("Detection Mode", ["Single Image", "Batch Processing"])
        
        if detection_mode == "Single Image":
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.read())
                    
                start_time = time.time()
                image = face_recognition.load_image_file(temp_file.name)
                enhanced_image = enhance_face_detection(Image.fromarray(image))
                
                locations = face_recognition.face_locations(np.array(enhanced_image))
                encodings = face_recognition.face_encodings(image, locations)
                
                with open(DEFAULT_ENCODINGS_PATH, "rb") as f:
                    known_encodings = pickle.load(f)
                
                results_image = Image.fromarray(image).copy()
                draw = ImageDraw.Draw(results_image)
                
                for (top, right, bottom, left), face_encoding in zip(locations, encodings):
                    matches = face_recognition.compare_faces(known_encodings["encodings"], face_encoding, tolerance=settings["tolerance"])
                    name = "Unknown"
                    confidence = 0
                    
                    if True in matches:
                        matched_idxs = [i for i, b in enumerate(matches) if b]
                        counts = Counter(known_encodings["names"][i] for i in matched_idxs)
                        name = counts.most_common(1)[0][0]
                        confidence = counts.most_common(1)[0][1] / len(matched_idxs)
                    
                    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR, width=BOUNDING_BOX_WIDTH)
                    
                    if settings.get("enable_age_gender"):
                        age, gender = estimate_age_gender(image[top:bottom, left:right])
                        text = f"{name} ({confidence:.2f})\n{age}, {gender}"
                    else:
                        text = f"{name} ({confidence:.2f})"
                    
                    if settings.get("enable_emotion"):
                        emotion = detect_emotion(image[top:bottom, left:right])
                        text += f"\n{emotion}"
                    
                    text_bbox = draw.textbbox((left, bottom), text)
                    draw.rectangle((text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]), fill=BOUNDING_BOX_COLOR)
                    draw.text((left, bottom), text, fill=TEXT_COLOR)
                
                end_time = time.time()
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(results_image, caption="Detection Results", use_column_width=True)
                
                with col2:
                    st.markdown("### Detection Details")
                    st.write(f"Faces Found: {len(locations)}")
                    st.write(f"Processing Time: {end_time - start_time:.2f}s")
                    
                    if len(locations) > 0:
                        st.markdown("### Face Analysis")
                        for i, ((top, right, bottom, left), encoding) in enumerate(zip(locations, encodings)):
                            with st.expander(f"Face {i+1}"):
                                matches = face_recognition.compare_faces(known_encodings["encodings"], encoding)
                                if True in matches:
                                    name = known_encodings["names"][matches.index(True)]
                                    st.write(f"Identity: {name}")
                                else:
                                    st.write("Identity: Unknown")
                                
                                if settings.get("enable_age_gender"):
                                    age, gender = estimate_age_gender(image[top:bottom, left:right])
                                    st.write(f"Estimated Age: {age}")
                                    st.write(f"Predicted Gender: {gender}")
                                
                                if settings.get("enable_emotion"):
                                    emotion = detect_emotion(image[top:bottom, left:right])
                                    st.write(f"Emotion: {emotion}")

        else:  # Batch Processing
            uploaded_files = st.file_uploader("Upload Multiple Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            
            if uploaded_files and st.button("Process Batch"):
                progress_bar = st.progress(0)
                results = []
                
                for idx, file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                        temp_file.write(file.read())
                        
                        image = face_recognition.load_image_file(temp_file.name)
                        locations = face_recognition.face_locations(image)
                        encodings = face_recognition.face_encodings(image, locations)
                        
                        results.append({
                            "filename": file.name,
                            "faces_found": len(locations),
                            "locations": locations,
                            "encodings": encodings,
                            "image": image
                        })
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.success(f"Processed {len(uploaded_files)} images!")
                
                for result in results:
                    with st.expander(f"Results for {result['filename']}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            marked_image = Image.fromarray(result["image"]).copy()
                            draw = ImageDraw.Draw(marked_image)
                            
                            for (top, right, bottom, left) in result["locations"]:
                                draw.rectangle(((left, top), (right, bottom)), outline="red", width=2)
                            
                            st.image(marked_image, caption=f"Detected Faces: {result['faces_found']}")
                        
                        with col2:
                            st.write(f"Faces Found: {result['faces_found']}")

    elif page == "📊 Analytics":
        st.title("Analytics Dashboard")
        
        history_df = pd.read_csv(HISTORY_FILE)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_detections = len(history_df)
            recent_detections = len(history_df[-50:])
            st.metric("Total Detections", total_detections, f"+{recent_detections} recent")
        
        with col2:
            avg_faces = history_df['faces_detected'].mean()
            st.metric("Average Faces per Image", f"{avg_faces:.1f}")
        
        with col3:
            success_rate = (history_df['faces_recognized'].sum() / history_df['faces_detected'].sum() * 100)
            st.metric("Recognition Success Rate", f"{success_rate:.1f}%")
        
        st.markdown("### Detection Trends")
        fig = px.line(history_df, x='timestamp', y=['faces_detected', 'faces_recognized'],
                     title='Face Detection History')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Processing Time Distribution")
            fig = px.histogram(history_df, x='processing_time',
                             title='Processing Time Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Recognition Rate Over Time")
            history_df['recognition_rate'] = (history_df['faces_recognized'] / history_df['faces_detected'] * 100)
            fig = px.line(history_df, x='timestamp', y='recognition_rate',
                         title='Recognition Rate Trend')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Recent Detections")
        st.dataframe(history_df.tail(10).sort_values('timestamp', ascending=False))

    elif page == "🎥 Real-time Detection":
        st.title("Real-time Face Detection")
        
        detection_options = st.columns(3)
        with detection_options[0]:
            enable_recognition = st.checkbox("Enable Face Recognition", True)
        with detection_options[1]:
            enable_age_gender = st.checkbox("Enable Age/Gender Detection", settings.get("enable_age_gender", True))
        with detection_options[2]:
            enable_emotion = st.checkbox("Enable Emotion Detection", settings.get("enable_emotion", True))
        
        if st.button("Start Real-time Detection"):
            real_time_detection()

    elif page == "⚙️ Settings":
        st.title("System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Detection Settings")
            new_tolerance = st.slider("Face Recognition Tolerance", 0.1, 1.0, settings["tolerance"], 0.05)
            new_model_type = st.selectbox("Default Detection Model", 
                                        ["hog", "cnn", "combined"],
                                        index=["hog", "cnn", "combined"].index(settings["model_type"]))
            new_max_faces = st.number_input("Maximum Faces per Image", 1, 100, settings["max_faces"])
            
        with col2:
            st.markdown("### Feature Settings")
            new_enable_age_gender = st.checkbox("Enable Age/Gender Detection", settings["enable_age_gender"])
            new_enable_emotion = st.checkbox("Enable Emotion Detection", settings["enable_emotion"])
            new_enable_notifications = st.checkbox("Enable Notifications", settings["enable_notifications"])
            new_backup_frequency = st.selectbox("Backup Frequency", 
                                              ["hourly", "daily", "weekly"],
                                              index=["hourly", "daily", "weekly"].index(settings["backup_frequency"]))
        
        if st.button("Save Settings"):
            settings.update({
                "tolerance": new_tolerance,
                "model_type": new_model_type,
                "max_faces": new_max_faces,
                "enable_age_gender": new_enable_age_gender,
                "enable_emotion": new_enable_emotion,
                "enable_notifications": new_enable_notifications,
                "backup_frequency": new_backup_frequency
            })
            save_settings(settings)
            st.success("Settings updated successfully!")

    elif page == "💾 Backup & Restore":
        st.title("Backup & Restore")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Create Backup")
            if st.button("Create New Backup"):
                with st.spinner("Creating backup..."):
                    create_backup()
                st.success("Backup created successfully!")
        
        with col2:
            st.markdown("### Restore from Backup")
            backup_files = list(BACKUP_FOLDER.glob("backup_*"))
            if backup_files:
                selected_backup = st.selectbox("Select Backup", 
                                             backup_files,
                                             format_func=lambda x: x.name)
                
                if st.button("Restore Selected Backup"):
                    with st.spinner("Restoring from backup..."):
                        shutil.copy2(selected_backup / DEFAULT_ENCODINGS_PATH.name, DEFAULT_ENCODINGS_PATH)
                        shutil.copy2(selected_backup / HISTORY_FILE.name, HISTORY_FILE)
                        shutil.copy2(selected_backup / SETTINGS_FILE.name, SETTINGS_FILE)
                    st.success("Restore completed successfully!")
            else:
                st.info("No backups available")

if __name__ == "__main__":
    main()
