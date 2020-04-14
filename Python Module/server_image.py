import socket
import sys
import cv2
import numpy as np
import struct
from threading import Thread

HOST = '127.0.0.1'
PORT_OP = 5577
PORT_RGB = 5555
PORT_DEPTH = 5566
PORT_POSE = 5588
s = None
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CHANNEL_GRAY = 1            # number of channels for gray image
CHANNEL_RGB = 4             # number of channels for RGB image - 4
CHANNEL_OP = 3              # number of channels for BGR image - 3
CHANNEL_DEPTH = 4           # number of channels for depth image


def rec_rgb():
    '''
    receives RGB frames and plots them
    :return:
    '''
    conn = connect_sock(PORT_RGB)
    runn = True
    while runn:
        shape = (FRAME_HEIGHT, FRAME_WIDTH, CHANNEL_RGB)      # change CHANNEL according to image type wanted
        camera_feed = np.zeros(shape, np.uint8)                  # creates empty numpy array with the image shape
        imgSize = camera_feed.size                               # image size
        socket_data = b''
        result = True
        while imgSize:
            nbytes = conn.recv(imgSize)                         # receive complete image size
            if not nbytes:
                break
            socket_data += nbytes                                  # add to sockData the values received
            imgSize -= len(nbytes)                              # when complete empty imgSize
        if result:
            camera_feed = camera_np(camera_feed, socket_data, 'rgb')  # call function to transform camerafeed in numpy
            cv2.namedWindow("1");  # Create a window for display
            cv2.imshow("1", camera_feed)  # shows image in window
            key = cv2.waitKey(30)
            runn = key
            if key == 27:  # if user clicks escape, window closes
                runn = False
        else:
            runn = False
    conn.close()  # close tcp connection

def rec_depth():
    '''
    receives depth frames and plots them
    :return:
    '''
    conn2 = connect_sock(PORT_DEPTH)
    runn = True
    print("entrei depth")
    while runn:
        shape = (FRAME_HEIGHT, FRAME_WIDTH, CHANNEL_DEPTH)      # change CHANNEL according to image type wanted
        camera_feed = np.zeros(shape, np.uint8)                  # creates empty numpy array with the image shape
        imgSize = camera_feed.size                               # image size
        socket_data = b''
        result = True
        while imgSize:
            nbytes2 = conn2.recv(imgSize)                         # receive complete image size
            if not nbytes2:
                break
            socket_data += nbytes2                                  # add to sockData the values received
            imgSize -= len(nbytes2)                              # when complete empty imgSize
        if result:
            camera_feed_depth = camera_np(camera_feed, socket_data, 'depth')  # call function to transform camerafeed in numpy
            cv2.namedWindow("2")  # Create a window for display
            cv2.imshow("2", camera_feed_depth)  # shows image in window
            key2 = cv2.waitKey(30)
            runn = key2
            if key2 == 27:  # if user clicks escape, window closes
                runn = False
        else:
            runn = False
    conn2.close()  # close tcp connection

def rec_openpose():
    '''
    receives RGB frames and plots them
    :return:
    '''
    conn = connect_sock(PORT_OP)
    runn = True
    while runn:
        shape = (FRAME_HEIGHT, FRAME_WIDTH, CHANNEL_OP)      # change CHANNEL according to image type wanted
        camera_feed = np.zeros(shape, np.uint8)                  # creates empty numpy array with the image shape
        imgSize = camera_feed.size                               # image size
        socket_data = b''
        result = True
        while imgSize:
            nbytes = conn.recv(imgSize)                         # receive complete image size
            if not nbytes:
                break
            socket_data += nbytes                                  # add to sockData the values received
            imgSize -= len(nbytes)                              # when complete empty imgSize
        if result:
            camera_feed = camera_np(camera_feed, socket_data, 'openpose')  # call function to transform camerafeed in numpy
            cv2.namedWindow("1");  # Create a window for display
            cv2.imshow("1", camera_feed)  # shows image in window
            key = cv2.waitKey(30)
            runn = key
            if key == 27:  # if user clicks escape, window closes
                runn = False
        else:
            runn = False
    conn.close()  # close tcp connection

def rec_pose_values():
    conn = connect_sock(PORT_POSE)
    runn = True
    my_list = []
    while runn:
        imgSize = 288
        while imgSize:
            nbytes = conn.recv(288)                         # receive complete image size
            print(len(nbytes))
            lol = struct.unpack("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", nbytes)      #
            joints = np.reshape(lol,(18,4))
            print(joints)
            if not nbytes:
                break
            imgSize -= len(nbytes)                              # when complete empty imgSize


def connect_sock(PORT):
    '''
    opens connection to accept clients
    :param PORT: port number: PORT_RGB (5555) or PORT_DEPTH (5566)
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


def camera_np(camera_feed, socket_data, img_type):
    '''
    :param camera_feed: the empty camera feed
    :param socket_data: data received in the socket
    :param img_type: type of image: can be 'rgb', 'gray' or 'depth'
    :return: camera_feed actualized
    '''
    if img_type == 'rgb':
        k = 4
    elif img_type == 'openpose':       # GBR
        k = 3
    elif img_type == 'gray':
        k = 1
    elif img_type == 'depth':
        k = 4
    width = camera_feed.shape[0]
    height = camera_feed.shape[1]
    socket_data = np.frombuffer(socket_data, np.uint8)
    camera_feed = np.tile(socket_data, 1).reshape((width, height, k))
    return camera_feed


if __name__ == "__main__":
    t1 = Thread(target=rec_depth).start()
    t2 = Thread(target=rec_rgb).start()
    t3 = Thread(target=rec_openpose).start()


