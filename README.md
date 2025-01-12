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

Any time you’re working on a Python project that uses external dependencies you’re installing with pip, it’s best to first create a virtual environment:
py -m venv venv\
This command allows the Python launcher for Windows to select an appropriate version of Python to execute. It comes bundled with the official installation and is the most convenient way to execute Python on Windows.
You can bypass the launcher and run the Python executable directly using the python command, but if you haven’t configured the PATH and PATHEXT variables, then you might need to provide the full path:
PS> C:\Users\Name\AppData\Local\Programs\Python\Python312\python -m venv venv\
The system path shown above assumes that you installed Python 3.12 using the Windows installer provided by the Python downloads page. The path to the Python executable on your system might be different. Working with PowerShell, you can find the path using the where.exe python command.
Note: You don’t need to include the backslash (\) at the end of the name of your virtual environment, but it’s a helpful reminder that you’re creating a folder.
This command creates a new virtual environment named venv using Python’s built-in venv module. The first venv that you use in the command specifies the module, and the second venv/ sets the name for your virtual environment. You could name it differently, but calling it venv is a good practice for consistency.

Great! Your project now has its own virtual environment. Generally, before you start to use it, you’ll activate the environment by executing a script that comes with the installation:
PS> venv\Scripts\activate
(venv) PS>

After you’ve created and activated your virtual environment, you can install any external dependencies that you need for your project:
(venv) PS> python -m pip install <package-name>

Once you’re done working with this virtual environment, you can deactivate it:
(venv) PS> deactivate
PS>


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













