# Unlocking-Identities-The-Power-of-Facial-Recognition

Do you have a phone that you can unlock with your face? Have you ever wondered how that works? Have you ever wanted to build your own face recognizer? With Python, some data, and a few helper packages, you can create your very own. In this project, you’ll use face detection and face recognition to identify faces in a given image.

To accomplish this feat, you’ll first use face detection, or the ability to find faces in an image. Then, you’ll implement face recognition, which is the ability to identify detected faces in an image. To that end, your program will do three primary tasks:
1.Train a new face recognition model.
2.Validate the model.
3.Test the model.

To build this face recognition application, you won’t need advanced linear algebra, deep machine learning algorithm knowledge, or even any experience with OpenCV, one of the leading Python libraries enabling a lot of computer vision work.
Instead, you should have an intermediate-level understanding of Python. You should be comfortable with:
1.Installing third-party modules with pip
2.Using argparse to create a command-line interface
3.Opening and reading files with pathlib
4.Serializing and deserializing Python objects with pickle

First, create your project and data directories:
PS> mkdir face_recognizer
PS> cd face_recognizer
PS> mkdir output
PS> mkdir training
PS> mkdir validation

Running these commands creates a directory called face_recognizer/, moves to it, then creates the folders output/, training/, and validation/, which you’ll use throughout the project. Now you can create a virtual environment using the tool of your choice.
Before you start installing this project’s dependencies with pip, you’ll need to ensure that you have CMake and a C compiler like gcc installed on your system.

Before you start installing this project’s dependencies with pip, you’ll need to ensure that you have CMake and a C compiler like gcc installed on your system.

To install CMake on Windows, visit the https://cmake.org/download/ page and install the appropriate installer for your system.

You can’t get gcc as a stand-alone download for Windows, but you can install it as a part of the MinGW runtime environment through the Chocolatey package manager with the following command:
PS> choco install mingw

Mow open your favorite text editor to create your requirements.txt file:
dlib==19.24.0
face-recognition==1.3.0
numpy==1.24.2
Pillow==9.4.0


To make Streamlit work we can use the follow command and steps : 
"streamlit run appname.py" command in command line













