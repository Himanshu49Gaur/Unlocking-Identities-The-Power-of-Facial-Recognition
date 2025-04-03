# Unlocking-Identities-The-Power-of-Facial-Recognition

A Streamlit-based application for face detection and recognition using **HOG** and **CNN** models. This app allows you to train models on custom datasets, detect faces in uploaded images, and evaluate the model's performance.

## Features
- **Train Model**: Upload images organized by name in folders, and train the model to recognize faces.
- **Detect Faces**: Upload an image to detect and recognize faces with bounding boxes and labels.
- **Test Model**: Evaluate the model's accuracy on a test dataset.
- **Data Augmentation**: Automatically augments training images for better accuracy.
- **Real-time Feedback**: Displays results with face detection time.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Himanshu49Gaur/face-detection-app.git
    cd face-detection-app
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

## Usage
1. **Prepare Dataset**:
   - Create a `training` folder.
   - Add images organized in subfolders by name (e.g., `training/John/image1.jpg`).

2. **Train Model**:
   - Navigate to the **Train Model** section in the app.
   - Upload additional images if needed.
   - Click the **Train Model** button to generate face encodings.

3. **Detect Faces**:
   - Navigate to the **Detect Faces** section.
   - Upload an image for detection.
   - View the results, including bounding boxes and detected names.

4. **Test Model**:
   - Navigate to the **Test Model** section.
   - Click the **Test Model** button to evaluate accuracy on a test dataset.

## File Structure
face-detection-app/ 
│ 

├── app.py # Main Streamlit app file 

├── training/ # Folder for training images

├── output/ # Folder for output files (e.g., encodings.pkl)

├── requirements.txt # Python dependencies 

└── README.md # Project documentation


## Dependencies
- Python 3.8+
- Streamlit
- face_recognition
- scikit-learn
- Pillow
- pickle

Install dependencies using:
```bash
pip install -r requirements.txt
```

How it Works
Training:

1. Extracts face encodings from images in the training dataset.
Augments images to improve recognition accuracy.
Detection:

2. Detects faces in uploaded images.
Matches detected faces with the trained dataset.
Testing:

3. Splits the dataset into training and test sets.
Evaluates accuracy based on correctly recognized faces.

Limitations

1. Accuracy: Dependent on the quality and size of the training dataset.

2. Performance: Face recognition using the CNN model may require more processing power.

Future Enhancements

1. Add support for live video face detection.

2. Implement an option to adjust augmentation parameters.

3. Support for additional face recognition models.

Collaborator : https://github.com/Abhinavmehra2004

License

This project is licensed under the MIT License. See the LICENSE file for details.












