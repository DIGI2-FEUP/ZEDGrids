# C++ Module
This folder provides the collaborative monitor C++ API: integrates ZED camera, OpenPose, PCL, and OpenCV (Object Detection)

These files must be put in the folder [`examples/user_code/`](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/user_code) of OpenPose before using CMAKE to build

IN CMAKE:
- Please make sure all the option needed are checked, mainly the DOWNLOAD_BODY_COCO_MODEL 

MUST ADD in Visual Studio:
- C/C++ Additional Include Directories: ZED SDK dependecy freeglut\include (usually in C:\Program Files (x86)\ZED SDK\dependencies\freeglut_x.x\include)
- Linker Additional Library Directories: ZED SDK dependecy freeglut lib (C:\Program Files (x86)\ZED SDK\dependencies\freeglut_x.x\x64)
- Linker Input: add ws2_32.lib
