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
np.set_printoptions(threshold=np.inf)
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
    init_params.coordinate_units = sl.UNIT.UNIT_METER  # Use milliliter units (for depth measurements)



    cloud = PyntCloud.from_file(r'C:\\Users\\Liliana\\Documents\\points_rgb.txt', sep=" ", header=0, names=["x", "y", "z", "red", "green", "blue"])
    #cloud.plot(backend='matplotlib')

    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=40, n_y=40, n_z=40)
    voxelgrid = cloud.structures[voxelgrid_id]
    binary_feature_vector = voxelgrid.get_feature_vector(mode="binary")      # return a 3D and 1D voxel array with 0s on
                                                                                        # empty voxels and 1s on voxel with 1+ points
    free_3d = np.where(binary_feature_vector == 0)
    occupied_3d = np.where(binary_feature_vector > 0)

    #voxelgrid.plot(d=3, mode="density", output_name='grid_10x10x10.html')
    #voxelgrid.plot(d=3, mode="binary", cmap='plasma', output_name='grid_last2.html')


if __name__ == "__main__":
    main()
