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

import numpy as np
import struct
import sys
import ctypes
import math
import time
import pyzed.sl as sl
import pandas as pd
from pyntcloud import PyntCloud
import warnings
import mayavi.mlab
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
warnings.filterwarnings("ignore",category =RuntimeWarning)

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
    init_params.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.UNIT_CENTIMETER  # Use milliliter units (for depth measurements)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD  # Use STANDARD sensing mode

    # Capture 50 images and depth, then stop
    i = 0
    point_cloud = sl.Mat()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        while i < 4:
        # A new image is available if grab() returns SUCCESS
            start_time = time.time()

            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA)
            pc = np.array(point_cloud.get_data())
            pc1 = np.transpose(pc, (2, 0, 1)).reshape(4, -1)
            pc_py = np.transpose(pc1, (1, 0))
            df = pd.DataFrame(pc_py, columns=['x', 'y', 'z', 'rgb'])        # with Nan and infinity
            df = df.replace([np.inf, -np.inf], np.nan)                      # inf turn to NaN
            test = df['rgb']
            r_mat = []
            g_mat = []
            b_mat = []
            for x in test:
                    # cast float32 to int so that bitwise operations are possible
                    s = struct.pack('>f', x)
                    ii = struct.unpack('>l', s)[0]
                    # you can get back the float value by the inverse operations
                    pack = ctypes.c_uint32(ii).value
                    r = (pack & 0x00FF0000) >> 16
                    g = (pack & 0x0000FF00) >> 8
                    b = (pack & 0x000000FF)
                    r_mat.append(r)
                    g_mat.append(g)
                    b_mat.append(b)
            df = df.drop(['rgb'], axis=1)
            dr = pd.DataFrame(r_mat, columns=['red'])
            dg = pd.DataFrame(g_mat, columns=['green'])
            db = pd.DataFrame(b_mat, columns=['blue'])
            dat1 = pd.concat([df, dr, dg, db], axis=1)      # with nan
            dat1 = dat1.dropna()
            #dat1_clean.to_csv(r'C:\\Users\\Liliana\\Desktop\\output_rgb.csv', na_rep='NA', sep=',', index=False, quoting=3)

            cloud = PyntCloud(dat1)
            #cloud.plot(backend='matplotlib')

            voxelgrid_id = cloud.add_structure("voxelgrid", n_x=40, n_y=40, n_z=40)
            voxelgrid = cloud.structures[voxelgrid_id]

            i = i + 1
            lapsed_time = time.time() - start_time
            print(lapsed_time)
    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
