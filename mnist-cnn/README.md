# mnist-cnn

Following this guide you will learn:

1. How to use a Tensorpack (https://github.com/tensorpack/tensorpack) script to train a convolutional neural network (CNN) to perform MNIST handwritten digit classification.
2. How to generate weights from the Tensorflow model as C-arrays, to be used in Vivado HLS.
3. How to implement the CNN in Vivado HLS targeted C, using the neural network layers in hls-nn-lib.
4. How to generate RTL using Vivado HLS, and how to generate a final overlay ready to be deployed on PYNQ.
5. How to write a python script to use the CNN that is deployed on the PYNQ to perform handwritten digit classification.

## 1. Training

For training the CNN, you need to first install Tensorflow (https://www.tensorflow.org/install/). Since the neural network that we will train is relatively small, not having a GPU will not hurt much.

If you haven't done so yet, clone the repo and go to the training directory:

1. $ git clone https://github.com/fpgasystems/spooNN
2. $ export PYTHONPATH=/path/to/spooNN/hls-nn-lib/training:$PYTHONPATH
3. $ cd spooNN/mnist-cnn/training/

Have a look at the mnist-cnn.py script. The CNN that we will train will be quantized (1-bit weights and 5-bit activations).

Start the training:
- $ python ./mnist-cnn.py 1
This will take a while. Observe the console output to see how much accuracy is reached on the test set. After 10 epochs, it should be around 98%.

Generate the weights to be used in C and dump them also as a numpy array to weights.npy (replace NAME1 and NAME2 accordingly. NAME1 will be unique, whereas for NAME2 you can select weights from a certain iteration):
- $ python ./mnist-cnn.py 2 --meta ./train_log/mnist-cnn/NAME1.meta --model ./train_log/mnist-cnn/NAME2.data-00000-of-00001 --output weights.npy
This operation creates 2 files: mnist-cnn-config.h and mnist-cnn-params.h. The -config.h file contains layer-wise configuration parameters, for example the size of the convolution kernel, stride etc. The -params.h file contains the weights and activation factors as C arrays.

Perform inference with the weights you dumped. This is useful when you want to compare intermediate results between Tensorflow and C while debugging. Download the MNIST test file from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist:
- $ python ./mnist-cnn.py 0 --testfile path/to/mnist.t --weights ./weights.npy

## 2. HLS implementation and testing

Now that we have trained a network and have the weights as C arrays, we can implement the entire CNN using the layers provided in hls-nn-lib. We can also test the functionality of our CNN entirely, by compiling the design with gcc. To be able to compile this file, you need to have Vivado HLS installed.

1. Take a look at mnist-cnn-1W5A.cpp. This file contains the top function DoCompute with a stream input and a stream output. It also has an argument "numReps", which will be a runtime configurable register on the FPGA, indicating how many images the CNN should process per initiation. There is also a main function in this file, which serves as a C testbench. In the main function we feed input images to the input stream and get results from the output stream, which we can then evaluate for functional correctness.
2. Export XILINX_VIVADO_HLS to point to your installation: $ export XILINX_VIVADO_HLS=/path/to/Xilinx/Vivado/2017.4
3. Compile mnist-cnn-1W5A.cpp with: /path/to/spooNN/mnist-cnn/hls$ make
4. Test the executable with: /path/to/spooNN/mnist-cnn/hls$ ./t_1W5A /path/to/mnist.t

## 3. RTL and bitstream generation

We now have a functionally correct CNN implemented in C, targeting Vivado HLS. 

1. Now we need to use Vivado HLS to generate RTL from the C implementation. There is a script for doing this: /path/to/spooNN/mnist-cnn$ ./scripts/make_IP.sh /path/to/spooNN/mnist-cnn
2. After the RTL is generated it is packaged in an IP, that is located in /path/to/spooNN/mnist-cnn/output/hls_project/sol1/impl/ip/xilinx_com_hls_DoCompute_1_0.zip
3. Create a directory called repo: $ mkdir repo
4. Copy paste the IP into the new directory: /path/to/spooNN/mnist-cnn$ cp /path/to/spooNN/mnist-cnn/output/hls_project/sol1/impl/ip/xilinx_com_hls_DoCompute_1_0.zip ./repo
5. Extract the IP: /path/to/spooNN/mnist-cnn$ unzip ./repo/xilinx_com_hls_DoCompute_1_0.zip

Now, the repo directory can be included in Vivado and the IP we generated can be instantiated.

1. We have a script to generate a final bitstream using this IP: /path/to/spooNN/mnist-cnn$ ./scripts/make_bitstream.sh /path/to/spooNN/mnist-cnn
2. This will take a while (around 30 minutes). After it is finished, find the generated bitstream in /path/to/spooNN/mnist-cnn/output/pynq-vivado/pynq-vivado.runs/impl_1/procsys_wrapper.bit

Besides the bitstream, you also need a .tcl file that describes the overlay. You can generate that by opening the pynq-vivado project with Vivado. Open the block design procsys. Then do File->Export->Export Block Design. This is the same file that is readily available at ./scripts/procsys.tcl

## 4. Deployment on the PYNQ

Now that we have a bitstream and a .tcl file (together, called an overlay), we show how to deploy it on the PYNQ.

1. Follow the guide to bring up you PYNQ (http://pynq.readthedocs.io) and make sure you can connect to it via SSH.
2. Copy the mnist-cnn/deploy directory to the jupyter_notebooks directory on the PYNQ (assuming the static IP 192.168.2.99): $ scp -r /path/to/spooNN/mnist-cnn/deploy xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/
3. Then open a browser window and navigate to http://192.168.2.99:9090, to access jupyter notebooks that are on the PYNQ.
4. Open deploy/mnist-cnn.ipynb and follow the notebook to see how we can use the CNN on the FPGA to perform handwritten digit inference. (overlay.bit and overlay.tcl are the files that are readily generated following the instructions of the previous part.)