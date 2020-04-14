########################################################################
#
# Copyright (c) 2017, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import pandas as pd
from pyntcloud import PyntCloud
import ctypes
from pandas.plotting import scatter_matrix
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
from math import copysign
import numpy as np
import struct
import sys
import time

def getIfromRGB(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    print(red, green, blue)
    RGBint = (red<<16) + (green<<8) + blue
    return RGBint

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
    init_params.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.UNIT_METER  # Use milliliter units (for depth measurements)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD  # Use STANDARD sensing mode

    # Capture 50 images and depth, then stop
    i = 0
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    while i < 1:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.VIEW_LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.MEASURE_DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA)
            pc = np.array(point_cloud.get_data())
            pc1 = np.transpose(pc, (2, 0, 1)).reshape(4, -1)
            pc_py = np.transpose(pc1, (1, 0))
            df = pd.DataFrame(pc_py, columns=['x', 'y', 'z', 'rgb'])
            test = df['rgb']
            r_mat = []
            for x in test:
                s = struct.pack('>f', x)
                i = struct.unpack('>l', s)[0]
                # you can get back the float value by the inverse operations
                pack = ctypes.c_uint32(i).value
                r = (pack & 0x00FF0000) >> 16
                g = (pack & 0x0000FF00) >> 8
                b = (pack & 0x000000FF)
                r_mat.append(r)
                #print(r, g, b)  # prints r,g,b values in the 0-255 range
            print(len(r_mat))
            cloud = PyntCloud(df)
            #print(cloud.points.describe())
            #cloud.plot(backend='matplotlib', use_as_color=['rgb'])
            #voxelgrid_id = cloud.add_structure("voxelgrid", n_x=100, n_y=100, n_z=100)
            #voxelgrid = cloud.structures[voxelgrid_id]
            '''
            x_values = []
            y_values = []
            z_values = []
            for i in range(0,719):
                for j in range(0,1279):
                    x_values.append(pc[i][j][0])
                    y_values.append(pc[i][j][1])
                    z_values.append(pc[i][j][2])'''
            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            '''x = round(image.get_width() / 2)
            y = round(image.get_height() / 2)
            err, point_cloud_value = point_cloud.get_value(x, y)

            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                 point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])

            if not np.isnan(distance) and not np.isinf(distance):
                distance = round(distance)
                print("Distance to Camera at ({0}, {1}): {2} mm\n".format(x, y, distance))
                # Increment the loop
                i = i + 1
            else:
                print("Can't estimate distance at this position, move the camera\n")'''
            i = i + 1
            sys.stdout.flush()
    #print(points_list)
    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
