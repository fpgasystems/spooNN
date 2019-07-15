
CINCLUDES	+= -I../../hls-nn-lib/
CINCLUDES 	+= -I${XILINX_VIVADO_HLS}/include/
CINCLUDES 	+= -std=c++0x -O1 -Wall -Wno-unknown-pragmas -Wno-unused-variable -Wno-uninitialized `pkg-config --cflags opencv`
CXX			= g++
LDFLAGS		= `pkg-config --libs opencv` -lpthread

all: halfsqueezenet_folded

halfsqueezenet_folded: halfsqueezenet_folded.cpp
	$(CXX) $(CFLAGS) $(CINCLUDES) halfsqueezenet_folded.cpp $(LDFLAGS) -o t