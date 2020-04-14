import struct
import sys
import socket
import time
from data_secondary_client import *
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from threading import Thread
import pandas as pd
from pyntcloud import PyntCloud
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
warnings.filterwarnings("ignore", category=RuntimeWarning)

HOST = '127.0.0.1'
PORT_POSE = 5588
PORT_PC = 5577
PORT_VALUE = 5599
s = None


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


class ThreeDMapping:
    def __init__(self, *, joints=None, jo=None, value=None, voxelgrid=None):
        self.joints_voxels = joints
        self.joints = jo
        self.value = value
        self.voxelgrids = voxelgrid


    def connect_sock(self, PORT):
        '''
        opens connection to accept clients
        :param PORT: port number: PORT_POSE (5588)
        :return: connection
        '''
        for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
            af, socktype, proto, canonname, sa = res
            try:
                s = socket.socket(af, socktype, proto)
            except socket.error as msg:
                s = None
                continue
            try:
                s.bind(sa)
                s.listen(1)
            except socket.error as msg:
                s.close()
                s = None
                continue
            break
        if s is None:
            print('could not open socket')
            sys.exit(1)
        conn, addr = s.accept()
        print('Connected by', addr)
        return conn

    def rec_values(self):
        '''
        receives points coordinates from C++ program via socket and creates UR5 appropriate coordinates array
        :return: joints_voxels: dataframe [x, y, z] transformed in base UR5 per pose joint
        '''

        runn3 = True
        while runn3:
            conn3 = self.connect_sock(PORT_VALUE)
            imgSize3 = 12
            socket_data3 = b''
            while imgSize3:
                nbytes3 = conn3.recv(imgSize3)                         # receive complete image size
                if not nbytes3:
                    print('not data')
                    break
                socket_data3 += nbytes3  # add to sockData the values received
                imgSize3 -= len(nbytes3)
            buff = struct.unpack("fff", socket_data3)
            if np.isnan(buff[0]):
                print("Can not calculate depth of Object select! Please try again!")
                self.value = None
            else:
                self.value = buff
            if map.value is not None:
                C2 = np.ones((1, 4))
                C2[:, 0:3] = [self.value[0], self.value[1], self.value[2]]  # point given by 3D mapping c++
                C2 = np.dot(T, C2.T).T  # transform C to UR5 base coordinates
                #point_obj = (C2[0][0], C2[0][1], C2[0][2], tcp[3], tcp[4], tcp[5])
                #print("going to point", point_obj)
                #ss.send_move("movel", point_obj, vel=0.1, acc=0.1, wait=True)
        conn3.close()  # close tcp connection

    def rec_pose_values(self, T):
        '''
        receives OpenPose joints via socket and creates joints array
        :param T: (m+1)x(m+1) homogeneous transformation matrix that maps pose joints on to UR5 base
        :return: joints_voxels: dataframe [x, y, z] transformed in base UR5 per pose joint
        (total 18 joints as in Pose COCO - github openpose/doc/output.md), only shows available joints)
        '''
        conn2 = self.connect_sock(PORT_POSE)
        runn2 = True
        while runn2:
            imgSize2 = 288
            socket_data2 = b''
            while imgSize2:
                nbytes2 = conn2.recv(imgSize2)                         # receive complete image size
                if not nbytes2:
                    print('not data')
                    break
                socket_data2 += nbytes2  # add to sockData the values received
                imgSize2 -= len(nbytes2)
            buff = struct.unpack("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", socket_data2)
            joint = np.reshape(buff, (18, 4))
            joint_df = pd.DataFrame(joint, columns=['x', 'y', 'z', 'w'])
            joint = joint_df.drop(['w'], axis=1)
            self.joints = joint
            joint = joint.dropna()
            C = np.ones((len(joint), 4))
            C[:, 0:3] = joint.values
            C = np.dot(T, C.T).T        # transform human joints to UR5 base coordinates
            C = np.delete(C, 3, 1)
            joint_nr = joint.index.values
            C_cent = C * 100  # meters to centimeters
            joint_vox = np.column_stack((C_cent, joint_nr))
            self.joints_voxels = joint_vox
        conn2.close()  # close tcp connection

    def rec_point_cloud(self, max_depth, n_vox, T, plots=True):
        '''
        receives Point_cloud array in 1D [3650400x1] via socket and creates clean pyntcloud voxelgrid
        :param max_depth: maximum depth possible in meters
        :param n_vox: number of voxels per axis - total voxels will be n_vox^3
        :param T: (m+1)x(m+1) homogeneous transformation matrix that maps pointcloud points on to UR5 base
        :param plots : if true plots voxel grid in binary mode on html file
        :return:
        '''
        #sur = send_UR()
        #joints = sur.get_joint_pos()
        #all_ur5 = sur.forward_kin(0, joints, 0)
        #all_human = self.joints_voxels
        conn = self.connect_sock(PORT_PC)
        runn = True
        i = 0
        while runn:
            imgSize = 4665600
            socket_data = b''
            while imgSize:
                nbytes = conn.recv(imgSize)  # receive complete image size
                if not nbytes:
                    break
                socket_data += nbytes  # add to sockData the values received
                imgSize -= len(nbytes)
            socket_data = np.frombuffer(socket_data, np.float32)
            pc = socket_data.reshape(-1, 4)
            df = pd.DataFrame(pc, columns=['x', 'y', 'z', 'rgb'])  # with Nan and infinity
            df = df[df['z'] < max_depth]
            df = df * 100  # meters to centimeters
            df = df.drop(['rgb'], axis=1)
            df = df.replace([np.inf, -np.inf], np.nan)  # inf to NaN
            df = df.dropna()  # drop NaN
            C = np.ones((len(df), 4))
            C[:, 0:3] = df.values
            C = np.dot(T, C.T).T        # transform point cloud to UR5 base coordinates
            C = pd.DataFrame(C, columns=['x', 'y', 'z', 'rgb'])
            C = C.drop(['rgb'], axis=1)
            cloud = PyntCloud(C)
            voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n_vox, n_y=n_vox, n_z=n_vox)
            voxelgrid = cloud.structures[voxelgrid_id]
            self.voxelgrids = voxelgrid
            if plots:
                a = 'a.html'
                voxelgrid.plot(d=3, mode="density", cmap='plasma', output_name=a)
        conn.close()  # close tcp connection



import pickle
if __name__ == "__main__":
    maxdepth = 3       # max depth in meters
    B_cor = np.array([[0.255561, 0.573695, 0.439873], [0.255262, 0.573531, 0.283118], [0.306974, 0.526021, 0.635867],
                      [0.254765, 0.483677, 0.633389], [0.454096, 0.115465, 0.497287]])  # end effector [x,y,z] from UR5
   
    A_cor = np.array(
        [[0.259851, -0.162137, 0.844032], [0.257923, -0.00298342, 0.841735], [0.258585, -0.363162, 0.899228],
         [0.319684, -0.358864, 0.895015], [0.423556, -0.222905, 1.31981]])  # end effector [x,y,z] from point cloud
 

    T, R1, t1 = best_fit_transform(A_cor, B_cor)
    ss = send_UR()
    map = ThreeDMapping()
    t_point = Thread(target=map.rec_values)
    t_pose = Thread(target=map.rec_pose_values, args=(T,))
    t_pc = Thread(target=map.rec_point_cloud, args=(maxdepth, 50, T))
    t_pc.start()
    t_pose.start()
    t_point.start()

