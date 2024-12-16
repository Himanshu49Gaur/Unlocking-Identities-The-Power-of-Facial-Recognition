if page == "Home":
    st.header("Welcome to the Face Detection App!")
    st.markdown("""
    - **Train Model**: Train the model using the existing dataset or by adding additional images.
    - **Detect Faces**: Detect and recognize faces in an unknown image.
    - **Test Model**: Test the accuracy of the model on a test dataset.
    """)

elif page == "Train Model":
    st.header("Train Your Model")
    st.markdown("### Existing Dataset")
    dataset_count = sum(1 for _ in TRAINING_FOLDER.glob("*/*"))
    st.write(f"Number of images in the existing dataset: {dataset_count}")

    st.markdown("### Add New Images")
    uploaded_files = st.file_uploader("Upload images (organized in folders by name):", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = Path(temp_file.name)
                person_name = uploaded_file.name.split("_")[0]
                person_folder = TRAINING_FOLDER / person_name
                person_folder.mkdir(exist_ok=True)
                dest_path = person_folder / uploaded_file.name
                temp_path.rename(dest_path)
        st.success("Uploaded images added to the dataset.")

    if st.button("Train Model"):
        encodings = encode_known_faces()
        with DEFAULT_ENCODINGS_PATH.open(mode="wb") as f:
            pickle.dump(encodings, f)
        st.success("Model trained successfully.")

elif page == "Detect Faces":
    st.header("Detect Faces in an Unknown Image")
    uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])
    
    

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
         temp_file.write(uploaded_file.read())

        # Start timing
        start_time = time.time()

        # Perform face recognition
        result_image, message, detected_names, total_faces, recognized_faces = recognize_faces(temp_file.name)

        # End timing
        end_time = time.time()

        # Calculate time taken
        time_taken = end_time - start_time

        if result_image:
            st.image(result_image, caption="Detected Faces", use_column_width=True)
            st.markdown("### Names Detected")
            for name in detected_names:
                st.write(name)
            st.markdown(f"**Time Taken:** {time_taken:.2f} seconds")  # Display time taken
        else:
            st.info(message)
            st.markdown(f"**Time Taken:** {time_taken:.2f} seconds")  # Display time taken if no faces detected
            
elif page == "Test Model":
    st.header("Test Model Accuracy")
    all_images = list(TRAINING_FOLDER.glob("*/*"))
    if len(all_images) < 2:
        st.warning("Insufficient dataset for testing. Add more images to the training folder.")
    else:
        train_images, test_images = train_test_split(all_images, test_size=0.3, random_state=42)
        st.write(f"Training Images: {len(train_images)}")
        st.write(f"Testing Images: {len(test_images)}")

        if st.button("Test Model"):
            with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
                encodings_data = pickle.load(f)
            
            correctly_recognized = 0
            total_faces = 0
            progress = st.progress(0)
            num_test_images = len(test_images)

            for idx, test_image in enumerate(test_images):
                _, _, detected_names, num_faces, num_recognized = recognize_faces(str(test_image), encodings_location=DEFAULT_ENCODINGS_PATH)
                total_faces += num_faces
                correctly_recognized += num_recognized
                progress.progress((idx + 1) / num_test_images)

            progress.empty()
            accuracy = (correctly_recognized / total_faces) * 100 if total_faces > 0 else 0
            st.success(f"Testing Completed! Accuracy: {accuracy:.2f}%")
            st.write(f"Total Test Faces: {total_faces}")
            st.write(f"Correctly Recognized Faces: {correctly_recognized}")