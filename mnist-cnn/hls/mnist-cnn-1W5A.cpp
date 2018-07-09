//*************************************************************************
// Copyright (C) 2018 Kaan Kara - Systems Group, ETH Zurich

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//*************************************************************************


#define TESTBENCH
// #define DEBUG

#define AP_INT_MAX_W 16384

#include "hls-nn-lib.h"
#include "../training/mnist-cnn-config-1W5A.h"
#include "../training/mnist-cnn-params-1W5A.h"

void DoCompute(stream<ap_axis >& in, stream<ap_axis >& out, const unsigned int numReps) {
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS RESOURCE variable=weights0 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights0 complete dim=0
#pragma HLS RESOURCE variable=factorA0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB0 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB0 complete dim=0
#pragma HLS RESOURCE variable=weights1 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights1 complete dim=0
#pragma HLS RESOURCE variable=factorA1 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB1 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB1 complete dim=0
#pragma HLS RESOURCE variable=weights2 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights2 complete dim=0
#pragma HLS RESOURCE variable=factorA2 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB2 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB2 complete dim=0
#pragma HLS RESOURCE variable=weights3 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights3 complete dim=0
#pragma HLS RESOURCE variable=factorA3 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB3 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA3 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB3 complete dim=0
#pragma HLS RESOURCE variable=weights4 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights4 complete dim=0
#pragma HLS RESOURCE variable=factorA4 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB4 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA4 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB4 complete dim=0
#pragma HLS RESOURCE variable=weights5 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights5 complete dim=0
#pragma HLS RESOURCE variable=factorA5 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB5 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA5 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB5 complete dim=0

#pragma HLS DATAFLOW

	// 2 rows of 28 row image per line -> per image 14 lines
	const unsigned int NumLinesPerRep = 14;

	stream<ap_uint<448> > in_stream_extract("in_stream_extract");
	ExtractPixels<448, NumLinesPerRep> (in, in_stream_extract, numReps);

	stream<ap_uint<L0_Cin*L0_Ibit> > in_stream("in_stream");
	ReduceWidth<448, L0_Cin*L0_Ibit, NumLinesPerRep> (in_stream_extract, in_stream, numReps);

#ifdef DEBUG
	Monitor<L0_Din, L0_Cin, L0_Ibit>(in_stream, (char*)"./log/mon_in_stream.log", numReps);
#endif

stream<ap_uint<L0_Cout*L0_Abit> > conv0("conv0");
CONV2D_ACT_NoP<L0_K, L0_S, L0_Din, L0_Cin, L0_Cout, L0_Ibit, L0_Wbit, L0_Mbit, L0_Abit, L0_MVTU_InP, L0_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(in_stream, weights0, factorA0, factorB0, conv0, numReps);
#ifdef DEBUG
Monitor<L0_Din/L0_S, L0_Cout, L0_Abit>(conv0, (char*)"log/mon_conv0.log", numReps);
#endif

stream<ap_uint<L6_Cin*L6_Ibit> > pool0("pool0");
POOL2D_NoP<L6_K, L6_S, L6_Din, L6_Cin, L6_Ibit>
(conv0, pool0, numReps);
#ifdef DEBUG
Monitor<L6_Din/L6_S, L6_Cin, L6_Ibit>(pool0, (char*)"log/mon_pool0.log", numReps);
#endif

stream<ap_uint<L1_Cout*L1_Abit> > conv1("conv1");
CONV2D_ACT_NoP<L1_K, L1_S, L1_Din, L1_Cin, L1_Cout, L1_Ibit, L1_Wbit, L1_Mbit, L1_Abit, L1_MVTU_InP, L1_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(pool0, weights1, factorA1, factorB1, conv1, numReps);
#ifdef DEBUG
Monitor<L1_Din/L1_S, L1_Cout, L1_Abit>(conv1, (char*)"log/mon_conv1.log", numReps);
#endif

stream<ap_uint<L2_Cout*L2_Abit> > conv2("conv2");
CONV2D_ACT_NoP<L2_K, L2_S, L2_Din, L2_Cin, L2_Cout, L2_Ibit, L2_Wbit, L2_Mbit, L2_Abit, L2_MVTU_InP, L2_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(conv1, weights2, factorA2, factorB2, conv2, numReps);
#ifdef DEBUG
Monitor<L2_Din/L2_S, L2_Cout, L2_Abit>(conv2, (char*)"log/mon_conv2.log", numReps);
#endif

stream<ap_uint<L7_Cin*L7_Ibit> > pool1("pool1");
POOL2D_NoP<L7_K, L7_S, L7_Din, L7_Cin, L7_Ibit>
(conv2, pool1, numReps);
#ifdef DEBUG
Monitor<L7_Din/L7_S, L7_Cin, L7_Ibit>(pool1, (char*)"log/mon_pool1.log", numReps);
#endif

stream<ap_uint<L3_Cout*L3_Abit> > conv3("conv3");
CONV2D_ACT_NoP<L3_K, L3_S, L3_Din, L3_Cin, L3_Cout, L3_Ibit, L3_Wbit, L3_Mbit, L3_Abit, L3_MVTU_InP, L3_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(pool1, weights3, factorA3, factorB3, conv3, numReps);
#ifdef DEBUG
Monitor<L3_Din/L3_S, L3_Cout, L3_Abit>(conv3, (char*)"log/mon_conv3.log", numReps);
#endif

stream<ap_uint<L4_OutP*L4_Abit> > dense0("dense0");
DENSE_ACT<L4_Din, L4_Dout, L4_Ibit, L4_Wbit, L4_Mbit, L4_Abit, L4_InP, L4_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(conv3, weights4, factorA4, factorB4, dense0, numReps);

stream<ap_uint<L5_OutP*L5_Mbit> > dense1("dense1");
DENSE_NOACT<L5_Din, L5_Dout, L5_Ibit, L5_Wbit, L5_Mbit, L5_InP, L5_OutP, SCALE_BITS>
(dense0, weights5, dense1, numReps);

	stream<ap_uint<512> > out_nolast("out_nolast");
	AppendZeros<10*L5_Mbit, 512, 1> (dense1, out_nolast, numReps);

	AddLast<1>(out_nolast, out, numReps);
}

// TESTBENCH
#ifdef TESTBENCH
#include "loader.h"
#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
#define NUM_SAMPLES 1

	char* pathToDataset;
	if (argc != 2) {
		cout << "Usage: ./t <pathToDataset>" << endl;
		return 0;
	}
	else {
		pathToDataset = argv[1];
}

	loader load = loader();

	load.load_libsvm_data(pathToDataset, NUM_SAMPLES, 784, 10);
	load.x_normalize(0, 'r');

	// One line will contain 448 useful bits
	const unsigned int data_points_per_line = 448/L0_Ibit;
	// Per image, we need 14 lines
	const unsigned int buffer_size = NUM_SAMPLES*14;
	stream<ap_axis > inputStream;
	
	unsigned int index = 0;
	for (unsigned int i = 0; i < buffer_size; i++) {
		ap_axis temp;
		for (unsigned int j = 0; j < data_points_per_line; j++) {
			temp.data( L0_Ibit*(j+1)-1, L0_Ibit*j ) = ((unsigned int)(load.x[index++]*255.0));
		}
		cout << "inputStream[" << i << "]: " << hex << temp.data << dec << endl;

		inputStream.write(temp);
	}
	
	stream<ap_axis > outputStream;

	DoCompute(inputStream, outputStream, NUM_SAMPLES);


	ap_axis outputBuffer[NUM_SAMPLES];

	for (unsigned int i = 0; i < NUM_SAMPLES; i++) {
		outputBuffer[i] = outputStream.read();
	}
	
	unsigned long MASK = ((long)1 << L5_Mbit) - 1;
	unsigned int count_trues = 0;
	for (unsigned int i = 0; i < NUM_SAMPLES; i++) {
		cout << "outputBuffer[" << i << "]: " << hex << outputBuffer[i].data << dec << endl;
		int max = 0;
		unsigned int prediction = -1;
		for (unsigned int j = 0; j < L5_Dout; j++) {
			int temp = (outputBuffer[i].data >> (j*L5_Mbit)) & MASK;
			temp = temp >> SCALE_BITS;
			cout << "outputBuffer[" << i << "][" << j << "]: " << static_cast<int>(temp) << endl;
			if (temp > max){
				max = temp;
				prediction = j;
			}
		}
		cout << "prediction: " << prediction << endl;
		if (load.y[i*10 + prediction] == 1)
			count_trues++;
	}
	cout << count_trues << " correct out of " << NUM_SAMPLES << endl;
}
#endif
