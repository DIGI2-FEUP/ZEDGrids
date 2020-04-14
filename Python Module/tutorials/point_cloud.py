import time
import pyzed.sl as sl
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import warnings
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
    while i < 1:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            start_time = time.time()

            zed.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA)
            pc = np.array(point_cloud.get_data())
            pc1 = np.transpose(pc, (2, 0, 1)).reshape(4, -1)
            pc_py = np.transpose(pc1, (1, 0))
            df = pd.DataFrame(pc_py, columns=['x', 'y', 'z', 'rgb'])        # with Nan and infinity
            df = df.replace([np.inf, -np.inf], np.nan)                      # inf turn to NaN
            dat1 = df.dropna()                                              # drop NaN
            print(dat1)
            #dat1.to_csv(r'C:\\Users\\Liliana\\Desktop\\output_rgb.csv', na_rep='NA', sep=',', index=False, quoting=3)

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
