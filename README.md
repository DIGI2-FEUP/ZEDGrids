# ZEDGrids
Framework for Voxelgrid-based Space Monitoring in Human-Robot (UR5) Collaboration Environments, using ZED camera, OpenPose and PyntCloud.

## Dependencies
### Python Module:
* [Python 3.7](https://www.python.org/downloads/release/python-37/)
* [PyntCloud](https://pyntcloud.readthedocs.io/en/latest/)
### C++ Module:
* [ZED SDK](https://www.stereolabs.com/developers/release/)
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [PCL](http://www.pointclouds.org/downloads/)
* [OpenCV](https://opencv.org/)

## Install:
Clone this repository :
```bash
    git clone https://github.com/DIGI2-FEUP/ZEDGrids
```
- Download and Install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Put the C++ Module files inside the OpenPose folder [`examples/user_code/`](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/user_code)
- Build OpenPose using [CMake](https://cmake.org/download/)
- Install PyntCloud
- Place voxelgrid.py file from the Python Module (replace) in PyntCloud folder [`structures´](https://github.com/daavoo/pyntcloud/tree/master/pyntcloud/structures)

## Usage:
- Run voxel_pc_server.py 
```bash
    python voxel_pc_server.py
```
- THEN Execute zed_openpose_pc.cpp (in Visual Studio)

