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
// #define REAL_IMAGE
// #define DEBUG

#define AP_INT_MAX_W 16384

#include "hls-nn-lib.h"
#include "../training/halfsqueezenet-config.h"
#include "../training/halfsqueezenet-params.h"

#define L_K1 1
#define L_K3 3
#define L_S 1
#define L_MAX_Din 58
#define L_MAX_squeeze_Cin 96
#define L_MAX_squeeze_Cout 32
#define L_MAX_expand_Cin 32
#define L_MAX_expand_Cout 96
#define L_Ibit L1_Ibit
#define L_Wbit L1_Wbit
#define L_Mbit L1_Mbit
#define L_Abit L1_Abit

#define squeeze_L_MVTU_InP L1_MVTU_InP
#define squeeze_L_MVTU_OutP L1_MVTU_OutP
#define expand_L_MVTU_InP L2_MVTU_InP
#define expand_L_MVTU_OutP L2_MVTU_OutP

#define USEFUL_LINE_BITS 480
#define LINES_PER_ALL_CHANNELS 1
const unsigned NumLinesPerRep = 3136;

#define LAST_LAYER 7

#define SQUEEZE_WEIGHT_ITERATIONS ((L_MAX_squeeze_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L_MAX_squeeze_Cout/squeeze_L_MVTU_OutP)
#define SQUEEZE_FACTOR_ITERATIONS L_MAX_squeeze_Cout/squeeze_L_MVTU_OutP
#define EXPAND_WEIGHT_ITERATIONS ((L_MAX_expand_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L_MAX_expand_Cout/expand_L_MVTU_OutP)
#define EXPAND_FACTOR_ITERATIONS L_MAX_expand_Cout/expand_L_MVTU_OutP
#define TOTAL_ITERATIONS SQUEEZE_WEIGHT_ITERATIONS+SQUEEZE_FACTOR_ITERATIONS+EXPAND_WEIGHT_ITERATIONS+EXPAND_FACTOR_ITERATIONS

static ap_uint<squeeze_L_MVTU_InP*L_Wbit> squeeze_weights[squeeze_L_MVTU_OutP][SQUEEZE_WEIGHT_ITERATIONS];
static ap_int<L_Mbit> squeeze_factorA[squeeze_L_MVTU_OutP][SQUEEZE_FACTOR_ITERATIONS];
static ap_int<L_Mbit> squeeze_factorB[squeeze_L_MVTU_OutP][SQUEEZE_FACTOR_ITERATIONS];

static ap_uint<expand_L_MVTU_InP*L_Wbit> expand3x3_weights[expand_L_MVTU_OutP][EXPAND_WEIGHT_ITERATIONS];
static ap_int<L_Mbit> expand3x3_factorA[expand_L_MVTU_OutP][EXPAND_FACTOR_ITERATIONS]; 
static ap_int<L_Mbit> expand3x3_factorB[expand_L_MVTU_OutP][EXPAND_FACTOR_ITERATIONS];

template <unsigned LineWidth, unsigned NumLines>
void DemuxStream3 (
	stream<ap_uint<LineWidth> >& in, 
	stream<ap_uint<LineWidth> >& out1, 
	stream<ap_uint<LineWidth> >& out2, 
	stream<ap_uint<LineWidth> >& out3, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp = in.read();
		if (whichFire == 1)
			out1.write(temp);
		else if (whichFire == 2)
			out2.write(temp);
		else
			out3.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void DemuxStream2 (
	stream<ap_uint<LineWidth> >& in, 
	stream<ap_uint<LineWidth> >& out1, 
	stream<ap_uint<LineWidth> >& out2, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp = in.read();
		if (whichFire == LAST_LAYER)
			out2.write(temp);
		else
			out1.write(temp);
	}
}

template <unsigned NumLines>
void DemuxStream2_0 (
	stream<ap_axis >& in, 
	stream<ap_axis >& out1, 
	stream<ap_axis >& out2, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_axis temp = in.read();
		if (whichFire == 1)
			out1.write(temp);
		else
			out2.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void MuxStream3 (
	stream<ap_uint<LineWidth> >& in1, 
	stream<ap_uint<LineWidth> >& in2, 
	stream<ap_uint<LineWidth> >& in3,
	stream<ap_uint<LineWidth> >& out, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp;
		if (whichFire == 1)
			temp = in1.read();
		else if (whichFire == 2)
			temp = in2.read();
		else
			temp = in3.read();
		out.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void MuxStream2 (
	stream<ap_uint<LineWidth> >& in1, 
	stream<ap_uint<LineWidth> >& in2,
	stream<ap_uint<LineWidth> >& out, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp;
		if (whichFire == LAST_LAYER)
			temp = in2.read();
		else
			temp = in1.read();
		out.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void MuxStream2_0 (
	stream<ap_uint<LineWidth> >& in1, 
	stream<ap_uint<LineWidth> >& in2,
	stream<ap_uint<LineWidth> >& out, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp;
		if (whichFire == 1)
			temp = in1.read();
		else
			temp = in2.read();
		out.write(temp);
	}
}

void DoFire(stream<ap_axis >& in, stream<ap_axis >& out,
	const unsigned squeeze_Din, const unsigned squeeze_Cin, const unsigned squeeze_Cout,
	const unsigned expand_Din, const unsigned expand_Din_afterpool, const unsigned expand_Cin, const unsigned expand_Cout,
	const unsigned whichFire, const unsigned numReps,
	const unsigned first_numReps,
	const unsigned conv0_numReps,
	const unsigned other_numReps,
	const unsigned pool1_numReps,
	const unsigned pool2_numReps,
	const unsigned fire5_numReps,
	const unsigned main_out_numReps,
	const unsigned final_out_numReps) 
{
#pragma HLS DATAFLOW
	stream<ap_axis> to_conv0("to_conv0");
	stream<ap_axis> to_fire("to_fire");
	DemuxStream2_0<1>(in, to_conv0, to_fire, whichFire, first_numReps);

// BRANCH 1
	stream<ap_uint<384> > in_stream_extract0("DoCompute.in_stream_extract0");
	ExtractPixels<384, NumLinesPerRep> (to_conv0, in_stream_extract0, conv0_numReps);

	stream<ap_uint<L0_Cin*L0_Ibit> > in_stream("DoCompute.in_stream");
	ReduceWidth<384, L0_Cin*L0_Ibit, NumLinesPerRep> (in_stream_extract0, in_stream, conv0_numReps);
#ifdef DEBUG
	Monitor<L0_Din, L0_Cin, L0_Ibit>(in_stream, (char*)"./log/mon_in_stream_folded.log", conv0_numReps);
#endif
	stream<ap_uint<L0_Cout*L0_Abit> > conv1("conv1");
	CONV2D_ACT_NoP<L0_K, L0_S, L0_Din, L0_Cin, L0_Cout, L0_Ibit, L0_Wbit, L0_Mbit, L0_Abit, L0_MVTU_InP, L0_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
	(in_stream, weights0, factorA0, factorB0, conv1, conv0_numReps);
#ifdef DEBUG
	if (whichFire == 1)
		Monitor<L0_Din/L0_S, L0_Cout, L0_Abit>(conv1, (char*)"log/mon_conv1_folded.log", conv0_numReps);
#endif
	stream<ap_uint<L18_Cin*L18_Ibit> > pool1("pool1");
	POOL2D_NoP<L18_K, L18_S, L18_Din, L18_Cin, L18_Ibit> (conv1, pool1, conv0_numReps);
	stream<ap_uint<L_MAX_squeeze_Cin*L_Ibit> > out_padded("out_padded");
	AppendZeros<L18_Cin*L18_Ibit, L_MAX_squeeze_Cin*L_Ibit, L1_Din*L1_Din> (pool1, out_padded, conv0_numReps);

// BRANCH 2
	stream<ap_uint<USEFUL_LINE_BITS> > in_stream_extract1("DoCompute.in_stream_extract1");
	ExtractPixels<USEFUL_LINE_BITS, LINES_PER_ALL_CHANNELS> (to_fire, in_stream_extract1, other_numReps);
	stream<ap_uint<L_MAX_squeeze_Cin*L_Ibit> > fire_in("fire_in");
	ExpandWidth<USEFUL_LINE_BITS, L_MAX_squeeze_Cin*L_Ibit, 1> (in_stream_extract1, fire_in, other_numReps);


	stream<ap_uint<L_MAX_squeeze_Cin*L_Ibit> > first_out("first_out");
	MuxStream2_0<L_MAX_squeeze_Cin*L_Ibit, 1>(out_padded, fire_in, first_out, whichFire, squeeze_Din*squeeze_Din*numReps);


	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out("fire_out");
	HALFFIRE_ACT_variable<	L_K1, L_S, L_MAX_Din, L_MAX_squeeze_Cin, L_MAX_squeeze_Cout, L_Ibit, L_Wbit, L_Mbit, L_Abit, squeeze_L_MVTU_InP, squeeze_L_MVTU_OutP,
							L_K3, L_S, L_MAX_Din, L_MAX_expand_Cin, L_MAX_expand_Cout, L_Ibit, L_Wbit, L_Mbit, L_Abit, expand_L_MVTU_InP, expand_L_MVTU_OutP,
							SCALE_BITS, FACTOR_SCALE_BITS>
	(first_out, squeeze_weights, squeeze_factorA, squeeze_factorB, expand3x3_weights, expand3x3_factorA, expand3x3_factorB, fire_out, 
	squeeze_Din, /*squeeze_Cin, squeeze_Cout,*/ expand_Din, /*expand_Cin, expand_Cout,*/ numReps);

	

	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out1("fire_out1");
	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out2("fire_out2");
	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out3("fire_out3");
	DemuxStream3<L_MAX_expand_Cout*L_Abit, 1> (fire_out, fire_out1, fire_out2, fire_out3, whichFire, expand_Din*expand_Din*numReps);


	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > pool_out1("pool_out");
	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > pool_out2("pool_out");
	POOL2D_NoP<L19_K, L19_S, L19_Din, L_MAX_expand_Cout, L_Ibit> (fire_out1, pool_out1, pool1_numReps);
	POOL2D_NoP<L20_K, L20_S, L20_Din, L_MAX_expand_Cout, L_Ibit> (fire_out2, pool_out2, pool2_numReps);

	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > pool_out("pool_out");
	MuxStream3<L_MAX_expand_Cout*L_Abit, 1> (pool_out1, pool_out2, fire_out3, pool_out, whichFire, expand_Din_afterpool*expand_Din_afterpool*numReps);

#ifdef DEBUG
	if (whichFire == 1)
		Monitor<L19_Din/L19_S, L19_Cin, L19_Ibit>(pool_out, (char*)"log/mon_pool2_folded.log", numReps);
	else if (whichFire == 2)
		Monitor<L20_Din/L20_S, L20_Cin, L20_Ibit>(pool_out, (char*)"log/mon_pool3_folded.log", numReps);
	else if (whichFire == 5)
		Monitor<L10_Din/L10_S, L10_Cout, L10_Abit>(pool_out, (char*)"log/mon_fire5_folded.log", numReps);
	else if (whichFire == 6)
		Monitor<L12_Din/L12_S, L12_Cout, L12_Abit>(pool_out, (char*)"log/mon_fire6_folded.log", numReps);
	else if (whichFire == 7)
		Monitor<L14_Din/L14_S, L14_Cout, L14_Abit>(pool_out, (char*)"log/mon_fire7_folded.log", numReps);
#endif


	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > main_out("main_out");
	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire5("fire5");
	DemuxStream2<L_MAX_expand_Cout*L_Abit, 1> (pool_out, main_out, fire5, whichFire, expand_Din_afterpool*expand_Din_afterpool*numReps);


// BRANCH 1
	stream<ap_uint<512> > main_out_padded("main_out_padded");
	AppendZeros<USEFUL_LINE_BITS, 512, 1> (main_out, main_out_padded, LINES_PER_ALL_CHANNELS*expand_Din_afterpool*expand_Din_afterpool*main_out_numReps);

// BRANCH 2
	stream<ap_uint<L14_Cout*L14_Abit> > fire5_class("fire5_class");
	stream<ap_uint<L14_Cout*L14_Abit> > fire5_obj("fire5_obj");
#pragma HLS STREAM variable=fire5_obj depth=14*14+48
	DuplicateStreams<L14_Cout*L14_Abit, L15_Din*L15_Din>(fire5, fire5_class, fire5_obj, fire5_numReps);
	stream<ap_uint<L15_Cout*L15_Abit> > conv_class("conv_class");
	CONV2D_1x1_ACT_NoP<L15_Din, L15_Cin, L15_Cout, L15_Ibit, L15_Wbit, L15_Mbit, L15_Abit, L15_MVTU_InP, L15_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
	(fire5_class, weights15, factorA15, factorB15, conv_class, fire5_numReps);
#ifdef DEBUG
	Monitor<L15_Din/L15_S, L15_Cout, L15_Abit>(conv_class, (char*)"log/mon_conv_class_folded.log", fire5_numReps);
#endif
	stream<ap_uint<(L14_Cout+L15_Cout)*L15_Abit> > class_out("class_out");
	ConcatStreams<L14_Cout*L14_Abit, L15_Cout*L15_Abit, L15_Din*L15_Din>(fire5_obj, conv_class, class_out, fire5_numReps);
	stream<ap_uint<(L14_Cout+L15_Cout)*L15_Abit> > class_out_obj("class_out_obj");
	stream<ap_uint<(L14_Cout+L15_Cout)*L15_Abit> > class_out_box("class_out_box");
	DuplicateStreams<(L14_Cout+L15_Cout)*L15_Abit, L16_Din*L16_Din>(class_out, class_out_obj, class_out_box, fire5_numReps);
	stream<ap_uint<L16_Cout*L16_Mbit> > conv_obj("conv_obj");
	CONV2D_1x1_NOACT_NoP<L16_Din, L16_Cin, L16_Cout, L16_Ibit, L16_Wbit, L16_Mbit, L16_MVTU_InP, L16_MVTU_OutP>
	(class_out_obj, weights16, conv_obj, fire5_numReps);
	stream<ap_uint<L17_Cout*L17_Mbit> > conv_box("conv_box");
#pragma HLS STREAM variable=conv_box depth=14*14
	CONV2D_1x1_NOACT_NoP<L17_Din, L17_Cin, L17_Cout, L17_Ibit, L17_Wbit, L17_Mbit, L17_MVTU_InP, L17_MVTU_OutP>
	(class_out_box, weights17, conv_box, fire5_numReps);
	stream<ap_uint<8+L17_Cout*L17_Mbit> > box_prediction("box_prediction");
	ObjDetectSelect<L17_Mbit, L17_Cout*L17_Mbit, L17_Din*L17_Din> (conv_obj, conv_box, box_prediction, fire5_numReps);
	stream<ap_uint<512> > box_prediction_padded("box_prediction_padded");
	AppendZeros<8+L17_Cout*L17_Mbit, 512, 1> (box_prediction, box_prediction_padded, fire5_numReps);

	stream<ap_uint<512> > final_out("final_out");
	MuxStream2<512, 1>(main_out_padded, box_prediction_padded, final_out, whichFire, final_out_numReps);

	AddLast<1> (final_out, out, final_out_numReps);
}

void writeWeightsFactors(stream<ap_axis >& in) {
#pragma HLS DATAFLOW

	stream<ap_uint<512> > squeeze_weights_stream("squeeze_weights_stream");
	stream<ap_uint<512> > squeeze_factors_stream("squeeze_weights_stream");
	stream<ap_uint<512> > expand_weights_stream("squeeze_weights_stream");
	stream<ap_uint<512> > expand_factors_stream("squeeze_weights_stream");

	for (unsigned i = 0; i < TOTAL_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_axis temp_in = in.read();
		if (i < SQUEEZE_WEIGHT_ITERATIONS)
			squeeze_weights_stream.write(temp_in.data);
		else if (i < SQUEEZE_WEIGHT_ITERATIONS + EXPAND_WEIGHT_ITERATIONS)
			expand_weights_stream.write(temp_in.data);
		else if (i < SQUEEZE_WEIGHT_ITERATIONS + EXPAND_WEIGHT_ITERATIONS + SQUEEZE_FACTOR_ITERATIONS)
			squeeze_factors_stream.write(temp_in.data);
		else
			expand_factors_stream.write(temp_in.data);
	}

	for (unsigned i = 0; i < SQUEEZE_WEIGHT_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = squeeze_weights_stream.read();
		for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<squeeze_L_MVTU_InP*L_Wbit> temp = temp_in( (p+1)*squeeze_L_MVTU_InP*L_Wbit-1, p*squeeze_L_MVTU_InP*L_Wbit );
			squeeze_weights[p][i] = temp;
		}
	}

	for (unsigned i = 0; i < EXPAND_WEIGHT_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = expand_weights_stream.read();
		for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<expand_L_MVTU_InP*L_Wbit> temp = temp_in( (p+1)*expand_L_MVTU_InP*L_Wbit-1, p*expand_L_MVTU_InP*L_Wbit );
			expand3x3_weights[p][i] = temp;
		}
	}

	for (unsigned i = 0; i < SQUEEZE_FACTOR_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = squeeze_factors_stream.read();
		for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<2*L_Mbit> temp_factorAB = temp_in( (p+1)*2*L_Mbit-1, p*2*L_Mbit );
			squeeze_factorA[p][i] = temp_factorAB(L_Mbit-1, 0);
			squeeze_factorB[p][i] = temp_factorAB(2*L_Mbit-1, L_Mbit);
		}
	}

	for (unsigned i = 0; i < EXPAND_FACTOR_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = expand_factors_stream.read();
		for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<2*L_Mbit> temp_factorAB = temp_in( (p+1)*2*L_Mbit-1, p*2*L_Mbit );
			expand3x3_factorA[p][i] = temp_factorAB(L_Mbit-1, 0);
			expand3x3_factorB[p][i] = temp_factorAB(2*L_Mbit-1, L_Mbit);
		}
	}
}

void halfsqueezenet(stream<ap_axis >& in, stream<ap_axis >& out,
	const unsigned squeeze_Din, const unsigned squeeze_Cin, const unsigned squeeze_Cout,
	const unsigned squeeze_weight_iterations, const unsigned squeeze_factor_iterations,
	const unsigned expand_Din, const unsigned expand_Din_afterpool, const unsigned expand_Cin, const unsigned expand_Cout,
	const unsigned expand_weight_iterations, const unsigned expand_factor_iterations,
	const unsigned whichFire, const unsigned numReps) {
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=squeeze_Din bundle=control
#pragma HLS INTERFACE s_axilite port=squeeze_Cin bundle=control
#pragma HLS INTERFACE s_axilite port=squeeze_Cout bundle=control
#pragma HLS INTERFACE s_axilite port=squeeze_weight_iterations bundle=control
#pragma HLS INTERFACE s_axilite port=squeeze_factor_iterations bundle=control
#pragma HLS INTERFACE s_axilite port=expand_Din bundle=control
#pragma HLS INTERFACE s_axilite port=expand_Din_afterpool bundle=control
#pragma HLS INTERFACE s_axilite port=expand_Cin bundle=control
#pragma HLS INTERFACE s_axilite port=expand_Cout bundle=control
#pragma HLS INTERFACE s_axilite port=expand_weight_iterations bundle=control
#pragma HLS INTERFACE s_axilite port=expand_factor_iterations bundle=control
#pragma HLS INTERFACE s_axilite port=whichFire bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS RESOURCE variable=weights0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB0 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB0 complete dim=0
#pragma HLS RESOURCE variable=weights15 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA15 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB15 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights15 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA15 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB15 complete dim=0
#pragma HLS RESOURCE variable=weights16 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA16 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB16 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights16 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA16 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB16 complete dim=0
#pragma HLS RESOURCE variable=weights17 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA17 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB17 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights17 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA17 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB17 complete dim=0
	
#pragma HLS RESOURCE variable=squeeze_weights core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=squeeze_factorA core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=squeeze_factorB core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=squeeze_weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=squeeze_factorA complete dim=0
#pragma HLS ARRAY_PARTITION variable=squeeze_factorB complete dim=0

#pragma HLS RESOURCE variable=expand3x3_weights core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=expand3x3_factorA core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=expand3x3_factorB core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=expand3x3_weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=expand3x3_factorA complete dim=0
#pragma HLS ARRAY_PARTITION variable=expand3x3_factorB complete dim=0

	const unsigned first_numReps = (whichFire == 1) ? NumLinesPerRep*numReps : LINES_PER_ALL_CHANNELS*squeeze_Din*squeeze_Din*numReps;
	const unsigned conv0_numReps = (whichFire == 1) ? numReps : 0;
	const unsigned other_numReps = (whichFire != 1) ? squeeze_Din*squeeze_Din*numReps : 0;
	const unsigned pool1_numReps = (whichFire == 1) ? numReps : 0;
	const unsigned pool2_numReps = (whichFire == 2) ? numReps : 0;
	const unsigned fire5_numReps = (whichFire == LAST_LAYER) ? numReps : 0;
	const unsigned main_out_numReps = (whichFire != LAST_LAYER) ? numReps : 0;
	const unsigned final_out_numReps = (whichFire == LAST_LAYER) ? numReps : LINES_PER_ALL_CHANNELS*expand_Din_afterpool*expand_Din_afterpool*numReps;

	if (whichFire == 13) {
		writeWeightsFactors(in);
	}
	else {
		DoFire(in, out,
			squeeze_Din, squeeze_Cin, squeeze_Cout,
			expand_Din, expand_Din_afterpool, expand_Cin, expand_Cout,
			whichFire, numReps,
			first_numReps,
			conv0_numReps,
			other_numReps,
			pool1_numReps,
			pool2_numReps,
			fire5_numReps,
			main_out_numReps,
			final_out_numReps);
	}
}

void II_determiner(stream<ap_axis >& in, stream<ap_axis >& out) {
	halfsqueezenet(in, out, L1_Din, L1_Cin, L1_Cout, 0, 0, L2_Din, L2_Din >> 1, L2_Cin, L2_Cout, 0, 0, 1, 1);
}

// TESTBENCH
#ifdef TESTBENCH

#ifdef REAL_IMAGE
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif

#include <fstream>
#include <iostream>
#include <string>
using namespace std;

int main() {

const unsigned NUM_SAMPLES=1;

#ifdef REAL_IMAGE
	string imagename("test_image1.png");
	Mat im;
	im = imread(imagename.c_str(), IMREAD_COLOR);

	unsigned height = im.rows;
	unsigned width = im.cols;
#else
	unsigned height = 224;
	unsigned width = 224;
#endif
	
	cout << "Image height: " << height << endl;
	cout << "Image width: " << width << endl;

	const unsigned pixel_bits = L0_Ibit*L0_Cin;
	const unsigned pixels_per_line = 384/pixel_bits;
	const unsigned buffer_size = (NUM_SAMPLES*height*width)/pixels_per_line;
	stream<ap_axis > inputStream("inputStream");

	cout << "pixels_per_line: " << pixels_per_line << endl;
	cout << "buffer_size: " << buffer_size << endl;

#ifdef REAL_IMAGE
	uint8_t* pixel_ptr = (uint8_t*)im.data;
	unsigned channels = im.channels();
#else
	uint8_t* pixel_ptr = (uint8_t*)malloc(3*height*width);
	unsigned channels = 3;
	unsigned k = 0;
	for (unsigned y = 0; y < height; y++) {
		for (unsigned x = 0; x < width; x++) {
			for (unsigned c = 0; c < channels; c++) {
				pixel_ptr[y*width*channels + x*channels + c] = (k++)%256;
				// pixel_ptr[y*width*channels + x*channels + c] = 0;
			}			
		}
	}
#endif
	unsigned index = 0;
	unsigned word;

	for (unsigned i = 0; i < NUM_SAMPLES; i++) {
		word = 0;
		ap_axis temp;
		for (unsigned y = 0; y < height; y++) {
			for (unsigned x = 0; x < width; x++) {
				unsigned red = (unsigned)pixel_ptr[y*width*channels + x*channels];
				unsigned green = (unsigned)pixel_ptr[y*width*channels + x*channels + 1];
				unsigned blue = (unsigned)pixel_ptr[y*width*channels + x*channels + 2];
				unsigned rgb = (blue << 16) + (green << 8) + red;

				temp.data(pixel_bits*(word+1)-1, pixel_bits*word) = rgb;

				if (word == pixels_per_line-1) {
					inputStream.write(temp);
					word = 0;
					temp.data = 0;
					index++;
				}
				else
					word++;
			}
		}
	}

#ifndef REAL_IMAGE
	free(pixel_ptr);
#endif

	cout << "index: " << index << endl;
	cout << "word: " << word << endl;

	const unsigned weight_mem_size = ((L_MAX_squeeze_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L_MAX_squeeze_Cout/squeeze_L_MVTU_OutP) 
										+ ((L_MAX_expand_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L_MAX_expand_Cout/expand_L_MVTU_OutP);
	const unsigned factor_mem_size = L_MAX_squeeze_Cout/squeeze_L_MVTU_OutP + 2*(L_MAX_expand_Cout/expand_L_MVTU_OutP);

	stream<ap_axis> weightsfactors_stream[LAST_LAYER];

	unsigned squeeze_weight_iterations[LAST_LAYER];
	unsigned squeeze_factor_iterations[LAST_LAYER];
	unsigned expand_weight_iterations[LAST_LAYER];
	unsigned expand_factor_iterations[LAST_LAYER];

	squeeze_weight_iterations[0] = ((L1_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L1_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[1] = ((L3_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L3_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[2] = ((L5_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L5_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[3] = ((L7_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L7_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[4] = ((L9_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L9_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[5] = ((L11_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L11_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[6] = ((L13_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L13_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[0] = (L1_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[1] = (L3_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[2] = (L5_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[3] = (L7_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[4] = (L9_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[5] = (L11_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[6] = (L13_Cout/squeeze_L_MVTU_OutP);

	expand_weight_iterations[0] = ((L2_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L2_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[1] = ((L4_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L4_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[2] = ((L6_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L6_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[3] = ((L8_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L8_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[4] = ((L10_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L10_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[5] = ((L12_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L12_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[6] = ((L14_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L14_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[0] = (L2_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[1] = (L4_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[2] = (L6_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[3] = (L8_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[4] = (L10_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[5] = (L12_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[6] = (L14_Cout/expand_L_MVTU_OutP);

	ofstream ofs ("weights_file.txt", ofstream::out);
	cout << "Writing weights to ports" << endl;
	for (unsigned layer = 0; layer < LAST_LAYER; layer++) {
		ap_uint<squeeze_L_MVTU_InP*L_Wbit>* squeeze_weights[squeeze_L_MVTU_OutP];
		ap_uint<expand_L_MVTU_InP*L_Wbit>* expand3x3_weights[expand_L_MVTU_OutP];
		ap_int<L_Mbit>* squeeze_factorA[squeeze_L_MVTU_OutP];
		ap_int<L_Mbit>* squeeze_factorB[squeeze_L_MVTU_OutP];
		ap_int<L_Mbit>* expand3x3_factorA[expand_L_MVTU_OutP];
		ap_int<L_Mbit>* expand3x3_factorB[expand_L_MVTU_OutP];
		for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
			if (layer == 0) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights1[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA1[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB1[p];
			}
			else if (layer == 1) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights3[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA3[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB3[p];
			}
			else if (layer == 2) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights5[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA5[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB5[p];
			}
			else if (layer == 3) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights7[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA7[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB7[p];
			}
			else if (layer == 4) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights9[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA9[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB9[p];
			}
			else if (layer == 5) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights11[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA11[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB11[p];
			}
			else if (layer == 6) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights13[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA13[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB13[p];
			}
		}
		for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
			if (layer == 0) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights2[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA2[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB2[p];
			}
			else if (layer == 1) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights4[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA4[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB4[p];
			}
			else if (layer == 2) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights6[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA6[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB6[p];
			}
			else if (layer == 3) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights8[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA8[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB8[p];
			}
			else if (layer == 4) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights10[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA10[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB10[p];
			}
			else if (layer == 5) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights12[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA12[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB12[p];
			}
			else if (layer == 6) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights14[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA14[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB14[p];
			}
		}

		cout << "Allocating ports for layer " << layer << endl;
		ofs << "layer: " << layer << endl;
		ofs << "squeeze_weight_iterations: " << SQUEEZE_WEIGHT_ITERATIONS << endl;
		ofs << "expand_weight_iterations: " << EXPAND_WEIGHT_ITERATIONS << endl;
		ofs << "squeeze_factor_iterations: " << SQUEEZE_FACTOR_ITERATIONS << endl;
		ofs << "expand_factor_iterations: " << EXPAND_FACTOR_ITERATIONS << endl;
		for (unsigned i = 0; i < squeeze_weight_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
				temp.data( (p+1)*squeeze_L_MVTU_InP*L_Wbit-1, p*squeeze_L_MVTU_InP*L_Wbit ) = squeeze_weights[p][i];
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < SQUEEZE_WEIGHT_ITERATIONS - squeeze_weight_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < expand_weight_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
				temp.data( (p+1)*expand_L_MVTU_InP*L_Wbit-1, p*expand_L_MVTU_InP*L_Wbit ) = expand3x3_weights[p][i];
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < EXPAND_WEIGHT_ITERATIONS - expand_weight_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < squeeze_factor_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
				temp.data( 2*p*L_Mbit + L_Mbit-1, 2*p*L_Mbit ) = squeeze_factorA[p][i];
				temp.data( (2*p+1)*L_Mbit + L_Mbit-1, (2*p+1)*L_Mbit ) = squeeze_factorB[p][i];
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < SQUEEZE_FACTOR_ITERATIONS - squeeze_factor_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < expand_factor_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
				temp.data( 2*p*L_Mbit + L_Mbit-1, 2*p*L_Mbit ) = expand3x3_factorA[p][i];
				temp.data( (2*p+1)*L_Mbit + L_Mbit-1, (2*p+1)*L_Mbit ) = expand3x3_factorB[p][i];
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < EXPAND_FACTOR_ITERATIONS - expand_factor_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
	}
	ofs.close();
	cout << "Writing weights complete" << endl;

	stream<ap_axis > empty;
	
	// cout << "weightsfactors_stream[0].nbytes: " << weightsfactors_stream[0].size()*64 << endl;
	// unsigned temp_size = weightsfactors_stream[0].size();
	// for (unsigned j = 0; j < temp_size; j++) {
	// 	ap_axis temp = weightsfactors_stream[0].read();
	// 	cout << hex << temp.data << dec << endl;
	// 	weightsfactors_stream[0].write(temp);
	// }

	stream<ap_axis > outputStream1("outputStream1");
	halfsqueezenet(weightsfactors_stream[0], empty, 0, 0, 0, squeeze_weight_iterations[0], squeeze_factor_iterations[0], 0, 0, 0, 0, expand_weight_iterations[0], expand_factor_iterations[0], 13, 0);
	cout << "inputStream.nbytes: " << inputStream.size()*64 << endl;
	halfsqueezenet(inputStream, outputStream1, L1_Din, L1_Cin, L1_Cout, 0, 0, L2_Din, L2_Din >> 1, L2_Cin, L2_Cout, 0, 0, 1, NUM_SAMPLES);
	cout << "outputStream1.nbytes: " << outputStream1.size()*64 << endl;
	cout << "Done fire1!" << endl;

	// for (unsigned j = 0; j < 1; j++) {
	// 	ap_axis temp = outputStream1.read();
	// 	cout << hex << temp.data << dec << endl;
	// 	outputStream1.write(temp);
	// }

	stream<ap_axis > outputStream2("outputStream2");
	halfsqueezenet(weightsfactors_stream[1], empty, 0, 0, 0, squeeze_weight_iterations[1], squeeze_factor_iterations[1], 0, 0, 0, 0, expand_weight_iterations[1], expand_factor_iterations[1], 13, 0);
	halfsqueezenet(outputStream1, outputStream2, L3_Din, L3_Cin, L3_Cout, 0, 0, L4_Din, L4_Din >> 1, L4_Cin, L4_Cout, 0, 0, 2, NUM_SAMPLES);
	cout << "Done fire2!" << endl;

	// for (unsigned j = 0; j < 1; j++) {
	// 	ap_axis temp = outputStream2.read();
	// 	cout << hex << temp.data << dec << endl;
	// 	outputStream2.write(temp);
	// }

	stream<ap_axis > outputStream3("outputStream3");
	halfsqueezenet(weightsfactors_stream[2], empty, 0, 0, 0, squeeze_weight_iterations[2], squeeze_factor_iterations[2], 0, 0, 0, 0, expand_weight_iterations[2], expand_factor_iterations[2], 13, 0);
	halfsqueezenet(outputStream2, outputStream3, L5_Din, L5_Cin, L5_Cout, 0, 0, L6_Din, L6_Din, L6_Cin, L6_Cout, 0, 0, 3, NUM_SAMPLES);
	cout << "Done fire3!" << endl;

	// for (unsigned j = 0; j < 1; j++) {
	// 	ap_axis temp = outputStream3.read();
	// 	cout << hex << temp.data << dec << endl;
	// 	outputStream3.write(temp);
	// }

	stream<ap_axis > outputStream4("outputStream4");
	halfsqueezenet(weightsfactors_stream[3], empty, 0, 0, 0, squeeze_weight_iterations[3], squeeze_factor_iterations[3], 0, 0, 0, 0, expand_weight_iterations[3], expand_factor_iterations[3], 13, 0);
	halfsqueezenet(outputStream3, outputStream4, L7_Din, L7_Cin, L7_Cout, 0, 0, L8_Din, L8_Din, L8_Cin, L8_Cout, 0, 0, 4, NUM_SAMPLES);
	cout << "Done fire4!" << endl;

	// for (unsigned j = 0; j < 1; j++) {
	// 	ap_axis temp = outputStream4.read();
	// 	cout << hex << temp.data << dec << endl;
	// 	outputStream4.write(temp);
	// }

	stream<ap_axis > outputStream5("outputStream5");
	halfsqueezenet(weightsfactors_stream[4], empty, 0, 0, 0, squeeze_weight_iterations[4], squeeze_factor_iterations[4], 0, 0, 0, 0, expand_weight_iterations[4], expand_factor_iterations[4], 13, 0);
	halfsqueezenet(outputStream4, outputStream5, L9_Din, L9_Cin, L9_Cout, 0, 0, L10_Din, L10_Din, L10_Cin, L10_Cout, 0, 0, 5, NUM_SAMPLES);
	cout << "Done fire5!" << endl;

	// for (unsigned j = 0; j < 1; j++) {
	// 	ap_axis temp = outputStream5.read();
	// 	cout << hex << temp.data << dec << endl;
	// 	outputStream5.write(temp);
	// }

	stream<ap_axis > outputStream6("outputStream6");
	halfsqueezenet(weightsfactors_stream[5], empty, 0, 0, 0, squeeze_weight_iterations[5], squeeze_factor_iterations[5], 0, 0, 0, 0, expand_weight_iterations[5], expand_factor_iterations[5], 13, 0);
	halfsqueezenet(outputStream5, outputStream6, L11_Din, L11_Cin, L11_Cout, 0, 0, L12_Din, L12_Din, L12_Cin, L12_Cout, 0, 0, 6, NUM_SAMPLES);
	cout << "Done fire6!" << endl;

	// for (unsigned j = 0; j < 1; j++) {
	// 	ap_axis temp = outputStream6.read();
	// 	cout << hex << temp.data << dec << endl;
	// 	outputStream6.write(temp);
	// }

	stream<ap_axis > outputStream7("outputStream7");
	halfsqueezenet(weightsfactors_stream[6], empty, 0, 0, 0, squeeze_weight_iterations[6], squeeze_factor_iterations[6], 0, 0, 0, 0, expand_weight_iterations[6], expand_factor_iterations[6], 13, 0);
	halfsqueezenet(outputStream6, outputStream7, L13_Din, L13_Cin, L13_Cout, 0, 0, L14_Din, L14_Din, L14_Cin, L14_Cout, 0, 0, 7, NUM_SAMPLES);
	cout << "Done fire7!" << endl;

	const unsigned num_out_lines = NUM_SAMPLES;

	for (unsigned i = 0; i < num_out_lines; i++) {
		ap_axis temp = outputStream7.read();
		if (i < 5) {
			cout << "data: " << hex << temp.data << dec << endl;
			cout << "keep: " << hex << temp.keep << dec << endl;
			cout << "last: " << temp.last << endl;
		}

		ap_int<L_Mbit> xmin = temp.data(L_Mbit-1, 0);
		ap_int<L_Mbit> ymin = temp.data(L_Mbit*2-1, L_Mbit*1);
		ap_int<L_Mbit> xmax = temp.data(L_Mbit*3-1, L_Mbit*2);
		ap_int<L_Mbit> ymax = temp.data(L_Mbit*4-1, L_Mbit*3);
		unsigned max_index = temp.data(8+L_Mbit*4, L_Mbit*4);

		cout << "xmin: " << xmin << endl;
		cout << "ymin: " << ymin << endl;
		cout << "xmax: " << xmax << endl;
		cout << "ymax: " << ymax << endl;
		cout << "max_index: " << max_index << endl;
	}


	// for (unsigned i = 0; i < num_out_lines; i++) {
	// 	for (unsigned k = 0; k < L16_Din*L16_Din; k++) {
	// 		ap_axis temp = outputStream7.read();
	// 		ap_int<L_Mbit> temp_i = temp.data(L_Mbit-1, 0);
	// 		cout << "conv_obj " << k << ": " << temp_i << endl;
	// 	}
	// }


	cout << "Execution finished!" << endl;

	return 0;
}

#endif
