## this is for FPGA demo with only one FPGA and one computer for computer for display
## xxu8@nd.edu

import socket
import threading
import struct
import time
import cv2
import numpy
import argparse
from os import listdir
from os.path import isfile, join
from random import shuffle

FLAGS = None

class Senders_Carame_Object:
    def __init__(self, addr_port=("192.168.1.100", 3000)):
        print 'Senders_Carame_Object init'
        print addr_port
        self.resolution = (640, 360)
        self.img_fps = 30
        self.addr_port = addr_port
        self.Set_Socket(self.addr_port)

    def Set_Socket(self, addr_port):
        print 'Senders_Carame_Object   Set_Socket'
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(addr_port)
        self.server.listen(5)
        print("the process work in the port:%d" % addr_port[1])


def check_option(object, client):
    info = struct.unpack('hh', client.recv(4))  ## 8 or 12
    if info[0] != object.resolution[0] or info[1] != object.resolution[1]:
        print "error: check option fails, received resolution is: " + str(info[0]) + "," + str(info[1])
        return 1
    else:
        return 0


def RT_Image(object, client):
    print 'RT_Image '
    if (check_option(object, client) == 1):
        return

    images = []
    for file in listdir(FLAGS.image_dir):
        if '.jpg' in file:
            images.append(file)
    images.sort()

    camera = cv2.VideoCapture(1)


    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), object.img_fps]
    indexN = 0
    while (1):
        # time.sleep(0.4) ## about 10 fps

        # object.img = cv2.imread(FLAGS.image_dir + '/' + images[indexN%len(images)], cv2.IMREAD_COLOR)
        _, object.img = camera.read()

        indexN = indexN + 1
        object.img = cv2.resize(object.img, object.resolution)
        _, img_encode = cv2.imencode('.jpg', object.img, img_param)
        img_code = numpy.array(img_encode)
        object.img_data = img_code.tostring()
        try:
            client.send(struct.pack("ll", len(object.img_data), indexN) + object.img_data)
            print str(indexN) + ', size of the send img:', len(object.img_data)
            ## wait until the images are processed on FPGAs
            detecRec = struct.unpack("hhhh", client.recv(8))

            cv2.rectangle(object.img, (abs(int(detecRec[0])), abs(int(detecRec[1]))),
                          (abs(int(detecRec[2])), abs(int(detecRec[3]))), (0, 255, 0), 4)

            # cv2.WINDOW_NORMAL makes the output window resizealbe
            cv2.namedWindow("window", cv2.WINDOW_NORMAL)
            # resize the window according to the screen resolution
            cv2.resizeWindow('window', 1920, 1080)

            cv2.imshow("window", object.img)

            k = cv2.waitKey(27)
            if k == 27:
                break

        except:
            camera.release()
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Absolute path to image directory.')

    FLAGS, unparsed = parser.parse_known_args()

    sender = Senders_Carame_Object(("192.168.3.113", 3000))
    print "connection accept"
    client, D_addr = sender.server.accept()
    clientThread = threading.Thread(None, target=RT_Image, args=(sender, client,))
    clientThread.start()





