import socket
import cv2
import threading
import struct
import numpy as np
import sys

sys.path.append("/home/xilinx/jupyter_notebooks/dac_2019_contest/common")
#################### import all libraries and initializations ############
from pynq import Xlnk
from pynq import Overlay
from pynq.mmio import MMIO
from PIL import Image

class FPGA_Connect_Object:
    def __init__(self, addr_port_client=("192.168.1.100", 3000)):
        print('FPGA_Connect_Object init')
        self.resolution = [640, 360]
        self.client_port = addr_port_client

        team_name = 'SystemsETHZ'
        # agent = Agent(team_name)

        interval_time = 0
        xlnk = Xlnk()
        xlnk.xlnk_reset()

        ###########################variable initializing######################
        OVERLAY_PATH = '/home/xilinx/jupyter_notebooks/dac_2019_contest/common/' + team_name + '/ultra96_v04.bit'
        WEIGHTS_FILE_NAME = '/home/xilinx/jupyter_notebooks/dac_2019_contest/common/' + team_name + '/weights_file_v04_demo.txt'

        ###########################change board settings######################

        ###########################download      overlay######################
        overlay = Overlay(OVERLAY_PATH)
        self.dma = overlay.axi_dma_0
        self.nn_ctrl = MMIO(0xA0010000, length=1024)
        ###########################download      weights######################
        self.MINIBATCH_SIZE = 1
        self.height = 176
        self.width = 320
        pixel_bits = 24
        pixels_per_line = 384/pixel_bits
        self.num_lines = int((self.height*self.width)/pixels_per_line)

        self.in_buffer = xlnk.cma_array(shape=(self.MINIBATCH_SIZE*self.num_lines, 64), dtype=np.uint8)
        fire1_num_out_lines = (self.height/4)*(self.width/4)*self.MINIBATCH_SIZE
        self.fire1_out_buffer = xlnk.cma_array(shape=(int(16*fire1_num_out_lines),), dtype=np.uint32)
        fire2_num_out_lines = (self.height/8)*(self.width/8)*self.MINIBATCH_SIZE
        self.fire2_out_buffer = xlnk.cma_array(shape=(int(16*fire2_num_out_lines),), dtype=np.uint32)
        fire3_num_out_lines = (self.height/16)*(self.width/16)*self.MINIBATCH_SIZE
        self.fire3_out_buffer = xlnk.cma_array(shape=(int(16*fire3_num_out_lines),), dtype=np.uint32)
        self.fire4_out_buffer = xlnk.cma_array(shape=(int(16*fire3_num_out_lines),), dtype=np.uint32)
        self.fire5_out_buffer = xlnk.cma_array(shape=(int(16*fire3_num_out_lines),), dtype=np.uint32)
        final_num_lines = int((self.height/16)*(self.width/16))
        self.bndboxes = [xlnk.cma_array(shape=(self.MINIBATCH_SIZE,final_num_lines,16), dtype=np.int32),
                        xlnk.cma_array(shape=(self.MINIBATCH_SIZE,final_num_lines,16), dtype=np.int32),
                        xlnk.cma_array(shape=(self.MINIBATCH_SIZE,final_num_lines,16), dtype=np.int32),
                        xlnk.cma_array(shape=(self.MINIBATCH_SIZE,final_num_lines,16), dtype=np.int32)]
        self.obj_array = np.zeros((self.MINIBATCH_SIZE,final_num_lines))

        NUM_LAYERS = 3+4*4
        weights_file = open(WEIGHTS_FILE_NAME, "r")
        layer = 0
        total_iterations = np.zeros(NUM_LAYERS)
        for line in weights_file:
            if "layer" in line:
                temp = line.split(": ")
                layer = int(temp[1])
            if "total_iterations" in line:
                temp = line.split(": ")
                total_iterations[layer] = int(temp[1])
        weights_file.close()

        weightfactors_length = np.zeros(NUM_LAYERS)
        self.weightsfactors = []
        for i in range(0, NUM_LAYERS):
            weightfactors_length[i] = int(total_iterations[i])
            self.weightsfactors.append( xlnk.cma_array(shape=(int(16*weightfactors_length[i]),), dtype=np.uint32) )
        self.obj_factors = np.zeros(4)
        self.box_factors = np.zeros(4)
            
        index = 0
        weights_file = open(WEIGHTS_FILE_NAME, "r")
        for line in weights_file:
            if "layer" in line:
                temp = line.split(": ")
                layer = int(temp[1])
                index = 0
            elif "total_iterations" not in line:
                if "obj_factor" in line:
                    temp = line.split(' ')
                    self.obj_factors[int(temp[1])] = int(temp[2])
                elif "box_factor" in line:
                    temp = line.split(' ')
                    self.box_factors[int(temp[1])] = int(temp[2])
                else:
                    no0x = line.split('0x')[-1]
                    base = 1
                    while base < len(no0x):
                        part = no0x[-1*(base+8):-1*base]    
                        self.weightsfactors[layer][index*16 + int(base/8)] = int(part, 16)
                        base += 8
                    index += 1

    ## Define transfer functions
    def weightsfactors_transfer(self, weightsfactors):
        self.nn_ctrl.write(0x40, 13)
        self.nn_ctrl.write(0x48, 0)
        self.nn_ctrl.write(0x0, 0) # Reset
        self.nn_ctrl.write(0x0, 1) # Deassert reset
        self.dma.sendchannel.transfer(weightsfactors)
        self.dma.sendchannel.wait()
        
    def fire(self, inbuffer, outbuffer, 
             squeeze_din_w, squeeze_din_h,
             expand_din_w, expand_din_h,
             expand_din_w_afterpool, expand_din_h_afterpool,
             whichfire):
        self.nn_ctrl.write(0x0, 0) # Reset
        self.nn_ctrl.write(0x10, int(squeeze_din_w))
        self.nn_ctrl.write(0x18, int(squeeze_din_h))
        self.nn_ctrl.write(0x20, int(expand_din_w))
        self.nn_ctrl.write(0x28, int(expand_din_h))
        self.nn_ctrl.write(0x30, int(expand_din_w_afterpool))
        self.nn_ctrl.write(0x38, int(expand_din_h_afterpool))
        self.nn_ctrl.write(0x40, whichfire)
        self.nn_ctrl.write(0x48, self.MINIBATCH_SIZE) # set numReps
        self.nn_ctrl.write(0x0, 1) # Deassert reset
        self.dma.recvchannel.transfer(outbuffer)
        self.dma.sendchannel.transfer(inbuffer)

        ###########################end      initializing######################
    def Socket_Connect_Client(self):
        print('FPGA_Connect_Object Socket_Connect')
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client.connect(self.client_port)
        print("As a client, FPGA receives imgs from %s:%d" % (self.client_port[0], self.client_port[1]))

    def ProcessImg(self):
        print('FPGA_Connect_Object begins process imgs')
        self.client.send(struct.pack("hh", self.resolution[0], self.resolution[1]))

        print("debug_FPGA 1")

        while (1):
            recv_data = self.client.recv(16)
            # print("size ", len(recv_data))
            info = struct.unpack("qq", recv_data)

            buf_size = info[0]
            ## receive imgs
            if buf_size:
                try:
                    self.buf = b""
                    temp_buf = self.buf
                    while (buf_size):
                        temp_buf = self.client.recv(buf_size)
                        buf_size -= len(temp_buf)
                        self.buf += temp_buf
                except:
                    pass;

            data = np.fromstring(self.buf, dtype='uint8')

            ## get the imgs and process it
            image = cv2.imdecode(data, 1)

            ###########################call PL to do inference######################
            #detected bounding box coordinates in x1, y1, x2, y2
            result = np.zeros((1, 17))

            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)  
            self.in_buffer[0:self.num_lines,0:48] = np.reshape(image, (self.num_lines, 48))

            self.weightsfactors_transfer(self.weightsfactors[0])
            self.fire(self.in_buffer, self.fire1_out_buffer,\
                self.width/2, self.height/2, self.width/2, self.height/2, self.width/4, self.height/4, 1)
            self.dma.recvchannel.wait()

            self.weightsfactors_transfer(self.weightsfactors[1])
            self.fire(self.fire1_out_buffer, self.fire2_out_buffer,\
                self.width/4, self.height/4, self.width/4, self.height/4, self.width/8, self.height/8, 2)
            self.dma.recvchannel.wait()

            self.weightsfactors_transfer(self.weightsfactors[2])
            self.fire(self.fire2_out_buffer, self.fire3_out_buffer,\
                self.width/8, self.height/8, self.width/8, self.height/8, self.width/16, self.height/16, 3)
            self.dma.recvchannel.wait()

            for t in range(0, 4):
                self.weightsfactors_transfer(self.weightsfactors[3 + t*4])
                self.fire(self.fire3_out_buffer, self.fire4_out_buffer,\
                    self.width/16, self.height/16, self.width/16, self.height/16, self.width/16, self.height/16, 4)
                self.dma.recvchannel.wait()

                self.weightsfactors_transfer(self.weightsfactors[4 + t*4])
                self.fire(self.fire4_out_buffer, self.fire5_out_buffer,
                    self.width/16, self.height/16, self.width/16, self.height/16, self.width/16, self.height/16, 5)
                self.dma.recvchannel.wait()

                self.weightsfactors_transfer(self.weightsfactors[5 + t*4])
                self.fire(self.fire5_out_buffer, self.fire4_out_buffer,\
                    self.width/16, self.height/16, self.width/16, self.height/16, self.width/16, self.height/16, 6)
                self.dma.recvchannel.wait()

                self.weightsfactors_transfer(self.weightsfactors[6 + t*4])
                self.fire(self.fire4_out_buffer, self.bndboxes[t],\
                    self.width/16, self.height/16, self.width/16, self.height/16, self.width/16, self.height/16, 7)
                self.dma.recvchannel.wait()
                
                temp_obj = np.multiply(np.divide(self.bndboxes[t][:,:,4],float((1<<16))), float(self.obj_factors[t]))
                if t == 0:
                    self.obj_array = temp_obj
                else:
                    self.obj_array = np.add(self.obj_array, temp_obj)

            grid_cell = np.argmax(self.obj_array, axis=1)
            result[0,16] = grid_cell[0]
            for t in range(0,4):
                result[0, t*4:(t+1)*4] = self.bndboxes[t][0,grid_cell[0],0:4]

            float_objdetect = result[0,16].astype('float')
            float_bndboxes1 = np.multiply(np.divide(result[0,0:4].astype('float'), float((1<<16))), float(self.box_factors[0]))
            float_bndboxes2 = np.multiply(np.divide(result[0,4:8].astype('float'), float((1<<16))), float(self.box_factors[1]))
            float_bndboxes3 = np.multiply(np.divide(result[0,8:12].astype('float'), float((1<<16))), float(self.box_factors[2]))
            float_bndboxes4 = np.multiply(np.divide(result[0,12:16].astype('float'), float((1<<16))), float(self.box_factors[3]))
            float_bndboxes = float_bndboxes1+float_bndboxes2+float_bndboxes3+float_bndboxes4
            float_bndboxes = np.divide( float_bndboxes, 4.0*float((1 << 22)) )
            
            obj_h = int(float_objdetect/(self.width/16))
            obj_w = int(float_objdetect%(self.width/16))
                
            x1 = int((float_bndboxes[0] + obj_w*16) *(640/self.width))
            y1 = int((float_bndboxes[1] + obj_h*16) *(360/self.height))
            x2 = int((float_bndboxes[2] + obj_w*16) *(640/self.width))
            y2 = int((float_bndboxes[3] + obj_h*16) *(360/self.height))
            ###########################end        PL       call######################

            self.client.send(struct.pack("hhhh", abs(x1), abs(y1), abs(x2), abs(y2)))

    def ProcessInThread(self):
        print('FPGA_Connect_Object   Get_Data')
        showThread = threading.Thread(target=self.ProcessImg)
        showThread.start()

if __name__ == '__main__':
    FPGA = FPGA_Connect_Object(("192.168.3.109", 3000))
    print("connect object created")
    FPGA.Socket_Connect_Client()
    FPGA.ProcessInThread()