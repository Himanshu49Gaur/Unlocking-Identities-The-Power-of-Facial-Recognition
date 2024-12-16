import streamlit as st
from pathlib import Path
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw, ImageEnhance
import tempfile
import os
from sklearn.model_selection import train_test_split
import time

# Constants
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
TRAINING_FOLDER = Path("training")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
BOUNDING_BOX_WIDTH = 5
TOLERANCE = 0.3

# Directories
TRAINING_FOLDER.mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

# Set up Streamlit layout
st.set_page_config(page_title="Face Detection App", layout="wide")
st.title("Face Detection using HOG and CNN Models")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Train Model", "Detect Faces", "Test Model"])