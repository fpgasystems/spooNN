#define TESTBENCH
// #define REAL_IMAGE
// #define DEBUG

#define IS_DEMO 1

#define AP_INT_MAX_W 16384

#include "hls-nn-lib.h"
#if IS_DEMO == 1
#include "../training/halfsqueezenet-config_demo.h"
#include "../training/halfsqueezenet-params_demo.h"
#else
#include "../training/halfsqueezenet-config.h"
#include "../training/halfsqueezenet-params.h"
#endif

#define IM_WIDTH 320
#define IM_HEIGHT 176
#define PIXELS_PER_LINE 16 // 16*(8bits*3channels) = 384

#define L_K1 1
#define L_K3 3
#define L_S1 1
#define L_S2 2
#define L_MAX_Din_W 162
#define L_MAX_Din_H 90
#define L_MAX_squeeze_Cin 96
#define L_MAX_squeeze_Cout 32
#define L_MAX_expand_Cin 32
#define L_MAX_expand_Cout 96
#define L_Ibit L1_Ibit
#define L_Wbit L1_Wbit
#define L_Mbit L1_Mbit
#define L_Abit L1_Abit
#define L_final_Din_W IM_WIDTH/16
#define L_final_Din_H IM_HEIGHT/16
#define L_class_Cin L39_Cin
#define L_class_Cout L39_Cout
#define L_obj_Cin L40_Cin
#define L_obj_Cout L40_Cout
#define L_box_Cin L41_Cin
#define L_box_Cout L41_Cout
#define L_objbox_Wbit L40_Wbit

#define squeeze_L_MVTU_InP L1_MVTU_InP
#define squeeze_L_MVTU_OutP L1_MVTU_OutP
#define expand_L_MVTU_InP L2_MVTU_InP
#define expand_L_MVTU_OutP L2_MVTU_OutP
#define class_L_MVTU_InP L39_MVTU_InP
#define class_L_MVTU_OutP L39_MVTU_OutP
#define obj_L_MVTU_InP L40_MVTU_InP
#define obj_L_MVTU_OutP L40_MVTU_OutP
#define box_L_MVTU_InP L41_MVTU_InP
#define box_L_MVTU_OutP L41_MVTU_OutP

#define USEFUL_LINE_BITS 480
#define LINES_PER_ALL_CHANNELS 1
const unsigned NumLinesPerRep = (IM_WIDTH*IM_HEIGHT)/PIXELS_PER_LINE;

#define LAST_LAYER 7
#define NUM_LAYERS 3+4*4

#define SQUEEZE_WEIGHT_ITERATIONS ((L_MAX_squeeze_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L_MAX_squeeze_Cout/squeeze_L_MVTU_OutP)
#define SQUEEZE_FACTOR_ITERATIONS L_MAX_squeeze_Cout/squeeze_L_MVTU_OutP
#define EXPAND_WEIGHT_ITERATIONS ((L_MAX_expand_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L_MAX_expand_Cout/expand_L_MVTU_OutP)
#define EXPAND_FACTOR_ITERATIONS L_MAX_expand_Cout/expand_L_MVTU_OutP
#define CLASS_WEIGHT_ITERATIONS ((L_class_Cin*L_K1*L_K1)/class_L_MVTU_InP)*(L_class_Cout/class_L_MVTU_OutP)
#define CLASS_FACTOR_ITERATIONS L_class_Cout/class_L_MVTU_OutP
#define OBJ_WEIGHT_ITERATIONS ((L_obj_Cin*L_K1*L_K1)/obj_L_MVTU_InP)*(L_obj_Cout/obj_L_MVTU_OutP)
#define BOX_WEIGHT_ITERATIONS ((L_box_Cin*L_K1*L_K1)/box_L_MVTU_InP)*(L_box_Cout/box_L_MVTU_OutP)
#define THRESHOLD1 SQUEEZE_WEIGHT_ITERATIONS
#define THRESHOLD2 THRESHOLD1 + EXPAND_WEIGHT_ITERATIONS
#define THRESHOLD3 THRESHOLD2 + 2*SQUEEZE_FACTOR_ITERATIONS
#define THRESHOLD4 THRESHOLD3 + 2*EXPAND_FACTOR_ITERATIONS
#define THRESHOLD5 THRESHOLD4 + CLASS_WEIGHT_ITERATIONS
#define THRESHOLD6 THRESHOLD5 + 2*CLASS_FACTOR_ITERATIONS
#define THRESHOLD7 THRESHOLD6 + OBJ_WEIGHT_ITERATIONS
#define THRESHOLD8 THRESHOLD7 + BOX_WEIGHT_ITERATIONS
#define TOTAL_ITERATIONS THRESHOLD8

static ap_uint<squeeze_L_MVTU_InP*L_Wbit> squeeze_weights[squeeze_L_MVTU_OutP][SQUEEZE_WEIGHT_ITERATIONS];
static ap_int<L_Mbit> squeeze_factorA[squeeze_L_MVTU_OutP][SQUEEZE_FACTOR_ITERATIONS];
static ap_int<L_Mbit> squeeze_factorB[squeeze_L_MVTU_OutP][SQUEEZE_FACTOR_ITERATIONS];

static ap_uint<expand_L_MVTU_InP*L_Wbit> expand3x3_weights[expand_L_MVTU_OutP][EXPAND_WEIGHT_ITERATIONS];
static ap_int<L_Mbit> expand3x3_factorA[expand_L_MVTU_OutP][EXPAND_FACTOR_ITERATIONS]; 
static ap_int<L_Mbit> expand3x3_factorB[expand_L_MVTU_OutP][EXPAND_FACTOR_ITERATIONS];

static ap_uint<class_L_MVTU_InP*L_Wbit> class_weights[class_L_MVTU_OutP][CLASS_WEIGHT_ITERATIONS];
static ap_int<L_Mbit> class_factorA[class_L_MVTU_OutP][CLASS_FACTOR_ITERATIONS];
static ap_int<L_Mbit> class_factorB[class_L_MVTU_OutP][CLASS_FACTOR_ITERATIONS];

static ap_uint<obj_L_MVTU_InP*L_objbox_Wbit> obj_weights[obj_L_MVTU_OutP][OBJ_WEIGHT_ITERATIONS];
static ap_uint<box_L_MVTU_InP*L_objbox_Wbit> box_weights[box_L_MVTU_OutP][BOX_WEIGHT_ITERATIONS];

template <unsigned NumLines>
void DemuxStream2_conv1 (
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
void MuxStream2_fire (
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

template <unsigned LineWidth, unsigned NumLines>
void DemuxStream2_fire (
	stream<ap_uint<LineWidth> >& in,
	stream<ap_uint<LineWidth> >& out1,
	stream<ap_uint<LineWidth> >& out2,
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp = in.read();
		if (whichFire == 1 || whichFire == 2 || whichFire == 3)
			out1.write(temp);
		else
			out2.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void MuxStream2_pool (
	stream<ap_uint<LineWidth> >& in1, 
	stream<ap_uint<LineWidth> >& in2,
	stream<ap_uint<LineWidth> >& out, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp;
		if (whichFire == 1 || whichFire == 2 || whichFire == 3)
			temp = in1.read();
		else
			temp = in2.read();
		out.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void DemuxStream2_pool (
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

template <unsigned LineWidth, unsigned NumLines>
void MuxStream2_final (
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

void DoFire(stream<ap_axis >& in, stream<ap_axis >& out,
	const unsigned squeeze_Din_W, const unsigned squeeze_Din_H,
	const unsigned expand_Din_W, const unsigned expand_Din_H,
	const unsigned whichFire, const unsigned numReps,
	const unsigned first_numReps,
	const unsigned conv1_numReps,
	const unsigned other_numReps,
	const unsigned pool_numReps,
	const unsigned main_out_numReps,
	const unsigned firemain_out_numReps,
	const unsigned firelast_numReps,
	const unsigned final_out_numReps) 
{
#pragma HLS DATAFLOW
	stream<ap_axis> to_conv1("to_conv1");
	stream<ap_axis> to_fire("to_fire");
	DemuxStream2_conv1<1>(in, to_conv1, to_fire, whichFire, first_numReps);

// BRANCH 1
	stream<ap_uint<384> > in_stream_extract1("DoCompute.in_stream_extract1");
	ExtractPixels<384, NumLinesPerRep> (to_conv1, in_stream_extract1, conv1_numReps);

	stream<ap_uint<L0_Cin*L0_Ibit> > in_stream("DoCompute.in_stream");
	ReduceWidth<384, L0_Cin*L0_Ibit, NumLinesPerRep> (in_stream_extract1, in_stream, conv1_numReps);
#ifdef DEBUG
	Monitor_RECT<L0_Din_W, L0_Din_H, L0_Cin, L0_Ibit>(in_stream, (char*)"./log/mon_in_stream_folded.log", conv1_numReps);
#endif
	stream<ap_uint<L0_Cout*L0_Abit> > conv1("conv1");
	CONV2D_ACT_NoP_RECT<L0_K, L0_S, L0_Din_W, L0_Din_H, L0_Cin, L0_Cout, L0_Ibit, L0_Wbit, L0_Mbit, L0_Abit, L0_MVTU_InP, L0_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
	(in_stream, weights0, factorA0, factorB0, conv1, conv1_numReps);
#ifdef DEBUG
	if (whichFire == 1) {
		Monitor_RECT<L0_Din_W/L0_S, L0_Din_H/L0_S, L0_Cout, L0_Abit>(conv1, (char*)"log/mon_conv1_folded.log", conv1_numReps);
	}
#endif
	stream<ap_uint<L0_Cout*L0_Abit> > pool1("pool1");
	POOL2D_NoP_RECT<L_K3, L_S2, L0_Din_W, L0_Din_H, L0_Cout, L0_Abit> (conv1, pool1, conv1_numReps);
#ifdef DEBUG
	if (whichFire == 1) {
		Monitor_RECT<L0_Din_W/L_S2, L0_Din_H/L_S2, L0_Cout, L0_Abit>(pool1, (char*)"log/mon_pool1_folded.log", numReps);
	}
#endif
	stream<ap_uint<L_MAX_squeeze_Cin*L_Ibit> > out_padded("out_padded");
	AppendZeros<L0_Cout*L0_Abit, L_MAX_squeeze_Cin*L_Ibit, L1_Din_W*L1_Din_H> (pool1, out_padded, conv1_numReps);

// BRANCH 2
	stream<ap_uint<USEFUL_LINE_BITS> > in_stream_extract2("DoCompute.in_stream_extract2");
	ExtractPixels<USEFUL_LINE_BITS, LINES_PER_ALL_CHANNELS> (to_fire, in_stream_extract2, other_numReps);
	stream<ap_uint<L_MAX_squeeze_Cin*L_Ibit> > fire_in("fire_in");
	ExpandWidth<USEFUL_LINE_BITS, L_MAX_squeeze_Cin*L_Ibit, 1> (in_stream_extract2, fire_in, other_numReps);

	stream<ap_uint<L_MAX_squeeze_Cin*L_Ibit> > first_out("first_out");
	MuxStream2_fire<L_MAX_squeeze_Cin*L_Ibit, 1>(out_padded, fire_in, first_out, whichFire, squeeze_Din_W*squeeze_Din_H*numReps);

	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out("fire_out");
	HALFFIRE_ACT_variable_RECT<	L_K1, L_S1, L_MAX_Din_W, L_MAX_Din_H, L_MAX_squeeze_Cin, L_MAX_squeeze_Cout, L_Ibit, L_Wbit, L_Mbit, L_Abit, squeeze_L_MVTU_InP, squeeze_L_MVTU_OutP,
								L_K3, L_S1, L_MAX_Din_W, L_MAX_Din_H, L_MAX_expand_Cin, L_MAX_expand_Cout, L_Ibit, L_Wbit, L_Mbit, L_Abit, expand_L_MVTU_InP, expand_L_MVTU_OutP,
								SCALE_BITS, FACTOR_SCALE_BITS>
	(first_out, squeeze_weights, squeeze_factorA, squeeze_factorB, expand3x3_weights, expand3x3_factorA, expand3x3_factorB, fire_out, 
	squeeze_Din_W, squeeze_Din_H, expand_Din_W, expand_Din_H, numReps);


	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out1("fire_out1");
	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out2("fire_out2");
	DemuxStream2_fire<L_MAX_expand_Cout*L_Abit, 1> (fire_out, fire_out1, fire_out2, whichFire, expand_Din_W*expand_Din_H*numReps);

#ifdef DEBUG
	cout << "fire_out1.size(): " << fire_out1.size() << endl;
	cout << "fire_out2.size(): " << fire_out2.size() << endl;
#endif

	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > pool_out("pool_out");
	POOL2D_NoP_variable_RECT<L_K3, L_MAX_Din_W, L_MAX_Din_H, L_MAX_expand_Cout, L_Ibit> (fire_out1, pool_out, expand_Din_W, expand_Din_H, pool_numReps);

#ifdef DEBUG
	cout << "pool_out.size(): " << pool_out.size() << endl;
#endif

	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > main_out("main_out");
	MuxStream2_pool<L_MAX_expand_Cout*L_Abit, 1> (pool_out, fire_out2, main_out, whichFire, main_out_numReps);

#ifdef DEBUG
	cout << "main_out.size(): " << main_out.size() << endl;
	if (whichFire == 1)
		Monitor_RECT<L0_Din_W/4, L0_Din_H/4, L_MAX_expand_Cout, L_Abit>(main_out, (char*)"log/mon_pool2_folded.log", numReps);
	else if (whichFire == 2)
		Monitor_RECT<L0_Din_W/8, L0_Din_H/8, L_MAX_expand_Cout, L_Abit>(main_out, (char*)"log/mon_pool3_folded.log", numReps);
	else if (whichFire == 3)
		Monitor_RECT<L0_Din_W/16, L0_Din_H/16, L_MAX_expand_Cout, L_Abit>(main_out, (char*)"log/mon_pool4_folded.log", numReps);
	else if (whichFire == 5)
		Monitor_RECT<L0_Din_W/16, L0_Din_H/16, L_MAX_expand_Cout, L_Abit>(main_out, (char*)"log/mon_fire5_folded.log", numReps);
	else if (whichFire == 6)
		Monitor_RECT<L0_Din_W/16, L0_Din_H/16, L_MAX_expand_Cout, L_Abit>(main_out, (char*)"log/mon_fire6_folded.log", numReps);
	else if (whichFire == 7)
		Monitor_RECT<L0_Din_W/16, L0_Din_H/16, L_MAX_expand_Cout, L_Abit>(main_out, (char*)"log/mon_fire7_folded.log", numReps);
#endif

	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > main_out1("main_out1");
	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > main_out2("main_out2");
	DemuxStream2_pool<L_MAX_expand_Cout*L_Abit, 1> (main_out, main_out1, main_out2, whichFire, main_out_numReps);

// BRANCH 1
	stream<ap_uint<512> > main_out_padded("main_out_padded");
	AppendZeros<USEFUL_LINE_BITS, 512, 1> (main_out1, main_out_padded, firemain_out_numReps);

// BRANCH 2
	stream<ap_uint<L14_Cout*L14_Abit> > firelast_class("firelast_class");
	stream<ap_uint<L14_Cout*L14_Abit> > firelast_obj("firelast_obj");
#pragma HLS STREAM variable=firelast_obj depth=32*32+48
	DuplicateStreams<L14_Cout*L14_Abit, L_final_Din_W*L_final_Din_H>(main_out2, firelast_class, firelast_obj, firelast_numReps);
	stream<ap_uint<L_class_Cout*L_Abit> > conv_class("conv_class");
	CONV2D_1x1_ACT_NoP_RECT<L_final_Din_W, L_final_Din_H, L_MAX_expand_Cout, L_class_Cout, L_Ibit, L_Wbit, L_Mbit, L_Abit, class_L_MVTU_InP, class_L_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
	(firelast_class, class_weights, class_factorA, class_factorB, conv_class, firelast_numReps);
#ifdef DEBUG
	Monitor_RECT<L_final_Din_W, L_final_Din_H, L_class_Cout, L_Abit>(conv_class, (char*)"log/mon_conv_class_folded.log", firelast_numReps);
#endif
	stream<ap_uint<(L14_Cout+L_class_Cout)*L_Abit> > class_out("class_out");
	ConcatStreams<L14_Cout*L14_Abit, L_class_Cout*L_Abit, L_final_Din_W*L_final_Din_H>(firelast_obj, conv_class, class_out, firelast_numReps);
	stream<ap_uint<(L14_Cout+L_class_Cout)*L_Abit> > class_out_obj("class_out_obj");
	stream<ap_uint<(L14_Cout+L_class_Cout)*L_Abit> > class_out_box("class_out_box");
	DuplicateStreams<(L14_Cout+L_class_Cout)*L_Abit, L_final_Din_W*L_final_Din_H>(class_out, class_out_obj, class_out_box, firelast_numReps);

	stream<ap_uint<L_Mbit> > conv_obj("conv_obj");
	CONV2D_1x1_NOACT_NoP_RECT<L_final_Din_W, L_final_Din_H, L_MAX_expand_Cout+L_class_Cout, L_obj_Cout, L_Ibit, L_objbox_Wbit, L_Mbit, obj_L_MVTU_InP, obj_L_MVTU_OutP>
	(class_out_obj, obj_weights, conv_obj, firelast_numReps);
	stream<ap_uint<L_box_Cout*L_Mbit> > conv_box("conv_box");
#pragma HLS STREAM variable=conv_box depth=32*32
	CONV2D_1x1_NOACT_NoP_RECT<L_final_Din_W, L_final_Din_H, L_MAX_expand_Cout+L_class_Cout, L_box_Cout, L_Ibit, L_objbox_Wbit, L_Mbit, box_L_MVTU_InP, box_L_MVTU_OutP>
	(class_out_box, box_weights, conv_box, firelast_numReps);

	// stream<ap_uint<8+L_box_Cout*L_Mbit> > box_prediction("box_prediction");
	// ObjDetectSelect<L_Mbit, L_box_Cout*L_Mbit, L_final_Din_W*L_final_Din_H> (conv_obj, conv_box, box_prediction, firelast_numReps);
	stream<ap_uint<L_Mbit+L_box_Cout*L_Mbit> > box_prediction("box_prediction");
	ObjDetectOutput<L_Mbit, L_box_Cout*L_Mbit, L_final_Din_W*L_final_Din_H> (conv_obj, conv_box, box_prediction, firelast_numReps);

	stream<ap_uint<512> > box_prediction_padded("box_prediction_padded");
	AppendZeros<L_Mbit+L_box_Cout*L_Mbit, 512, L_final_Din_W*L_final_Din_H> (box_prediction, box_prediction_padded, firelast_numReps);

	stream<ap_uint<512> > final_out("final_out");
	MuxStream2_final<512, 1>(main_out_padded, box_prediction_padded, final_out, whichFire, final_out_numReps);

	AddLast<1> (final_out, out, final_out_numReps);
}

void writeWeightsFactors(stream<ap_axis >& in) {
#pragma HLS DATAFLOW

	stream<ap_uint<512> > squeeze_weights_stream("squeeze_weights_stream");
	stream<ap_uint<512> > squeeze_factors_stream("squeeze_weights_stream");
	stream<ap_uint<512> > expand_weights_stream("squeeze_weights_stream");
	stream<ap_uint<512> > expand_factors_stream("squeeze_weights_stream");
	stream<ap_uint<512> > class_weights_stream("class_weights_stream");
	stream<ap_uint<512> > class_factors_stream("class_factors_stream");
	stream<ap_uint<512> > obj_weights_stream("obj_weights_stream");
	stream<ap_uint<512> > box_weights_stream("box_weights_stream");

	for (unsigned i = 0; i < TOTAL_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_axis temp_in = in.read();
		if (i < THRESHOLD1)
			squeeze_weights_stream.write(temp_in.data);
		else if (i <THRESHOLD2)
			expand_weights_stream.write(temp_in.data);
		else if (i < THRESHOLD3)
			squeeze_factors_stream.write(temp_in.data);
		else if (i < THRESHOLD4)
			expand_factors_stream.write(temp_in.data);
		else if (i < THRESHOLD5)
			class_weights_stream.write(temp_in.data);
		else if (i < THRESHOLD6)
			class_factors_stream.write(temp_in.data);
		else if (i < THRESHOLD7)
			obj_weights_stream.write(temp_in.data);
		else if (i < THRESHOLD8)
			box_weights_stream.write(temp_in.data);
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
		ap_uint<512> temp_inA = squeeze_factors_stream.read();
		ap_uint<512> temp_inB = squeeze_factors_stream.read();
		for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			squeeze_factorA[p][i] = temp_inA( (p+1)*L_Mbit-1, p*L_Mbit );
			squeeze_factorB[p][i] = temp_inB( (p+1)*L_Mbit-1, p*L_Mbit );
		}
	}

	for (unsigned i = 0; i < EXPAND_FACTOR_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_inA = expand_factors_stream.read();
		ap_uint<512> temp_inB = expand_factors_stream.read();
		for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			expand3x3_factorA[p][i] = temp_inA( (p+1)*L_Mbit-1, p*L_Mbit );
			expand3x3_factorB[p][i] = temp_inB( (p+1)*L_Mbit-1, p*L_Mbit );
		}
	}

	for (unsigned i = 0; i < CLASS_WEIGHT_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = class_weights_stream.read();
		for (unsigned p = 0; p < class_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<class_L_MVTU_InP*L_Wbit> temp = temp_in( (p+1)*class_L_MVTU_InP*L_Wbit-1, p*class_L_MVTU_InP*L_Wbit );
			class_weights[p][i] = temp;
		}
	}

	for (unsigned i = 0; i < CLASS_FACTOR_ITERATIONS; i++) {
// #pragma HLS PIPELINE II=1 -> because CLASS_FACTOR_ITERATIONS is 1 and HLS does not like it
		ap_uint<512> temp_inA = class_factors_stream.read();
		ap_uint<512> temp_inB = class_factors_stream.read();
		for (unsigned p = 0; p < class_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			class_factorA[p][i] = temp_inA( (p+1)*L_Mbit-1, p*L_Mbit );
			class_factorB[p][i] = temp_inB( (p+1)*L_Mbit-1, p*L_Mbit );
		}
	}

	for (unsigned i = 0; i < OBJ_WEIGHT_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = obj_weights_stream.read();
		for (unsigned p = 0; p < obj_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<obj_L_MVTU_InP*L_objbox_Wbit> temp = temp_in( (p+1)*obj_L_MVTU_InP*L_objbox_Wbit-1, p*obj_L_MVTU_InP*L_objbox_Wbit );
			obj_weights[p][i] = temp;
		}
	}

	for (unsigned i = 0; i < BOX_WEIGHT_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = box_weights_stream.read();
		for (unsigned p = 0; p < box_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<box_L_MVTU_InP*L_objbox_Wbit> temp = temp_in( (p+1)*box_L_MVTU_InP*L_objbox_Wbit-1, p*box_L_MVTU_InP*L_objbox_Wbit );
			box_weights[p][i] = temp;
		}
	}
}

void halfsqueezenet(stream<ap_axis >& in, stream<ap_axis >& out,
	const unsigned squeeze_Din_W, const unsigned squeeze_Din_H,
	const unsigned expand_Din_W, const unsigned expand_Din_H, 
	const unsigned expand_Din_W_afterpool, const unsigned expand_Din_H_afterpool,
	const unsigned whichFire, const unsigned numReps) {
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=squeeze_Din_W bundle=control
#pragma HLS INTERFACE s_axilite port=squeeze_Din_H bundle=control
#pragma HLS INTERFACE s_axilite port=expand_Din_W bundle=control
#pragma HLS INTERFACE s_axilite port=expand_Din_H bundle=control
#pragma HLS INTERFACE s_axilite port=expand_Din_W_afterpool bundle=control
#pragma HLS INTERFACE s_axilite port=expand_Din_H_afterpool bundle=control
#pragma HLS INTERFACE s_axilite port=whichFire bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS RESOURCE variable=weights0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB0 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB0 complete dim=0

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

#pragma HLS RESOURCE variable=class_weights core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=class_factorA core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=class_factorB core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=class_weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=class_factorA complete dim=0
#pragma HLS ARRAY_PARTITION variable=class_factorB complete dim=0

#pragma HLS RESOURCE variable=obj_weights core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=obj_weights complete dim=0

#pragma HLS RESOURCE variable=box_weights core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=box_weights complete dim=0

	const unsigned first_numReps = (whichFire == 1) ? NumLinesPerRep*numReps : LINES_PER_ALL_CHANNELS*squeeze_Din_W*squeeze_Din_H*numReps;
	const unsigned conv1_numReps = (whichFire == 1) ? numReps : 0;
	const unsigned other_numReps = (whichFire != 1) ? squeeze_Din_W*squeeze_Din_H*numReps : 0;
	const unsigned pool_numReps = (whichFire == 1 || whichFire == 2 || whichFire == 3) ? numReps : 0;
	const unsigned main_out_numReps = LINES_PER_ALL_CHANNELS*expand_Din_W_afterpool*expand_Din_H_afterpool*numReps;
	const unsigned firemain_out_numReps = (whichFire == LAST_LAYER) ? 0 : main_out_numReps;
	const unsigned firelast_numReps = (whichFire == LAST_LAYER) ? numReps : 0;
	// const unsigned final_out_numReps = (whichFire == LAST_LAYER) ? numReps : main_out_numReps;
	const unsigned final_out_numReps = main_out_numReps;

#ifdef DEBUG
	cout << "--------------------------------" << endl;
	cout << "first_numReps: " << first_numReps << endl;
	cout << "conv1_numReps: " << conv1_numReps << endl;
	cout << "other_numReps: " << other_numReps << endl;
	cout << "pool_numReps: " << pool_numReps << endl;
	cout << "main_out_numReps: " << main_out_numReps << endl;
	cout << "firelast_numReps: " << firelast_numReps << endl;
	cout << "final_out_numReps: " << final_out_numReps << endl;
	cout << "--------------------------------" << endl;
#endif

	if (whichFire == 13) {
		writeWeightsFactors(in);
	}
	else {
		DoFire(in, out,
			squeeze_Din_W, squeeze_Din_H,
			expand_Din_W, expand_Din_H,
			whichFire, numReps,
			first_numReps,
			conv1_numReps,
			other_numReps,
			pool_numReps,
			main_out_numReps,
			firemain_out_numReps,
			firelast_numReps,
			final_out_numReps);
	}
}

// TESTBENCH
#ifdef TESTBENCH

#ifdef REAL_IMAGE
// #include <opencv2/core/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
#endif

#include <fstream>
#include <iostream>
#include <string>
using namespace std;

int main() {

const unsigned NUM_SAMPLES=1;

#ifdef REAL_IMAGE
	string imagename("test_image.png");
	Mat im;
	im = imread(imagename.c_str(), CV_LOAD_IMAGE_COLOR);

	unsigned height = im.rows;
	unsigned width = im.cols;
#else
	unsigned height = IM_HEIGHT;
	unsigned width = IM_WIDTH;
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

	stream<ap_axis> weightsfactors_stream[NUM_LAYERS];

	unsigned squeeze_weight_iterations[NUM_LAYERS];
	unsigned squeeze_factor_iterations[NUM_LAYERS];
	unsigned expand_weight_iterations[NUM_LAYERS];
	unsigned expand_factor_iterations[NUM_LAYERS];

	squeeze_weight_iterations[0] = ((L1_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L1_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[1] = ((L3_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L3_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[2] = ((L5_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L5_Cout/squeeze_L_MVTU_OutP);
	for (unsigned i = 0; i < 4; i++) {
		squeeze_weight_iterations[3 + i*4] = ((L7_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L7_Cout/squeeze_L_MVTU_OutP);
		squeeze_weight_iterations[4 + i*4] = ((L9_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L9_Cout/squeeze_L_MVTU_OutP);
		squeeze_weight_iterations[5 + i*4] = ((L11_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L11_Cout/squeeze_L_MVTU_OutP);
		squeeze_weight_iterations[6 + i*4] = ((L13_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L13_Cout/squeeze_L_MVTU_OutP);
	}
	squeeze_factor_iterations[0] = (L1_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[1] = (L3_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[2] = (L5_Cout/squeeze_L_MVTU_OutP);
	for (unsigned i = 0; i < 4; i++) {
		squeeze_factor_iterations[3 + i*4] = (L7_Cout/squeeze_L_MVTU_OutP);
		squeeze_factor_iterations[4 + i*4] = (L9_Cout/squeeze_L_MVTU_OutP);
		squeeze_factor_iterations[5 + i*4] = (L11_Cout/squeeze_L_MVTU_OutP);
		squeeze_factor_iterations[6 + i*4] = (L13_Cout/squeeze_L_MVTU_OutP);
	}
	expand_weight_iterations[0] = ((L2_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L2_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[1] = ((L4_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L4_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[2] = ((L6_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L6_Cout/expand_L_MVTU_OutP);
	for (unsigned i = 0; i < 4; i++) {
		expand_weight_iterations[3 + i*4] = ((L8_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L8_Cout/expand_L_MVTU_OutP);
		expand_weight_iterations[4 + i*4] = ((L10_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L10_Cout/expand_L_MVTU_OutP);
		expand_weight_iterations[5 + i*4] = ((L12_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L12_Cout/expand_L_MVTU_OutP);
		expand_weight_iterations[6 + i*4] = ((L14_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L14_Cout/expand_L_MVTU_OutP);
	}
	expand_factor_iterations[0] = (L2_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[1] = (L4_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[2] = (L6_Cout/expand_L_MVTU_OutP);
	for (unsigned i = 0; i < 4; i++) {
		expand_factor_iterations[3 + i*4] = (L8_Cout/expand_L_MVTU_OutP);
		expand_factor_iterations[4 + i*4] = (L10_Cout/expand_L_MVTU_OutP);
		expand_factor_iterations[5 + i*4] = (L12_Cout/expand_L_MVTU_OutP);
		expand_factor_iterations[6 + i*4] = (L14_Cout/expand_L_MVTU_OutP);
	}

	ofstream ofs ("weights_file.txt", ofstream::out);
	cout << "Writing weights to ports" << endl;
	for (unsigned layer = 0; layer < NUM_LAYERS; layer++) {
		ap_uint<squeeze_L_MVTU_InP*L_Wbit>* squeeze_weights[squeeze_L_MVTU_OutP];
		ap_uint<expand_L_MVTU_InP*L_Wbit>* expand3x3_weights[expand_L_MVTU_OutP];
		ap_int<L_Mbit>* squeeze_factorA[squeeze_L_MVTU_OutP];
		ap_int<L_Mbit>* squeeze_factorB[squeeze_L_MVTU_OutP];
		ap_int<L_Mbit>* expand3x3_factorA[expand_L_MVTU_OutP];
		ap_int<L_Mbit>* expand3x3_factorB[expand_L_MVTU_OutP];
		ap_uint<class_L_MVTU_InP*L_Wbit>* class_weights[class_L_MVTU_OutP];
		ap_int<L_Mbit>* class_factorA[class_L_MVTU_OutP];
		ap_int<L_Mbit>* class_factorB[class_L_MVTU_OutP];
		ap_uint<obj_L_MVTU_InP*L_objbox_Wbit>* obj_weights[obj_L_MVTU_OutP];
		ap_uint<box_L_MVTU_InP*L_objbox_Wbit>* box_weights[box_L_MVTU_OutP];

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
			else if (layer == 7) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights15[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA15[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB15[p];
			}
			else if (layer == 8) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights17[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA17[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB17[p];
			}
			else if (layer == 9) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights19[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA19[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB19[p];
			}
			else if (layer == 10) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights21[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA21[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB21[p];
			}
			else if (layer == 11) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights23[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA23[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB23[p];
			}
			else if (layer == 12) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights25[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA25[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB25[p];
			}
			else if (layer == 13) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights27[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA27[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB27[p];
			}
			else if (layer == 14) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights29[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA29[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB29[p];
			}
			else if (layer == 15) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights31[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA31[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB31[p];
			}
			else if (layer == 16) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights33[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA33[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB33[p];
			}
			else if (layer == 17) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights35[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA35[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB35[p];
			}
			else if (layer == 18) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights37[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA37[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB37[p];
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
			else if (layer == 7) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights16[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA16[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB16[p];
			}
			else if (layer == 8) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights18[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA18[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB18[p];
			}
			else if (layer == 9) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights20[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA20[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB20[p];
			}
			else if (layer == 10) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights22[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA22[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB22[p];
			}
			else if (layer == 11) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights24[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA24[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB24[p];
			}
			else if (layer == 12) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights26[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA26[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB26[p];
			}
			else if (layer == 13) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights28[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA28[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB28[p];
			}
			else if (layer == 14) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights30[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA30[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB30[p];
			}
			else if (layer == 15) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights32[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA32[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB32[p];
			}
			else if (layer == 16) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights34[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA34[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB34[p];
			}
			else if (layer == 17) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights36[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA36[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB36[p];
			}
			else if (layer == 18) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights38[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA38[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB38[p];
			}
		}
		for (unsigned p = 0; p < class_L_MVTU_OutP; p++) {
			if (layer == 6) {
				class_weights[p] = (ap_uint<class_L_MVTU_InP*L_Wbit>*)weights39[p];
				class_factorA[p] = (ap_int<L_Mbit>*)factorA39[p];
				class_factorB[p] = (ap_int<L_Mbit>*)factorB39[p];
			}
			else if (layer == 10) {
				class_weights[p] = (ap_uint<class_L_MVTU_InP*L_Wbit>*)weights42[p];
				class_factorA[p] = (ap_int<L_Mbit>*)factorA42[p];
				class_factorB[p] = (ap_int<L_Mbit>*)factorB42[p];
			}
			else if (layer == 14) {
				class_weights[p] = (ap_uint<class_L_MVTU_InP*L_Wbit>*)weights45[p];
				class_factorA[p] = (ap_int<L_Mbit>*)factorA45[p];
				class_factorB[p] = (ap_int<L_Mbit>*)factorB45[p];
			}
			else if (layer == 18) {
				class_weights[p] = (ap_uint<class_L_MVTU_InP*L_Wbit>*)weights48[p];
				class_factorA[p] = (ap_int<L_Mbit>*)factorA48[p];
				class_factorB[p] = (ap_int<L_Mbit>*)factorB48[p];
			}
		}
		for (unsigned p = 0; p < obj_L_MVTU_OutP; p++) {
			if (layer == 6) {
				obj_weights[p] = (ap_uint<obj_L_MVTU_InP*L_objbox_Wbit>*)weights40[p];
			}
			else if (layer == 10) {
				obj_weights[p] = (ap_uint<obj_L_MVTU_InP*L_objbox_Wbit>*)weights43[p];
			}
			else if (layer == 14) {
				obj_weights[p] = (ap_uint<obj_L_MVTU_InP*L_objbox_Wbit>*)weights46[p];
			}
			else if (layer == 18) {
				obj_weights[p] = (ap_uint<obj_L_MVTU_InP*L_objbox_Wbit>*)weights49[p];
			}
		}
		for (unsigned p = 0; p < box_L_MVTU_OutP; p++) {
			if (layer == 6) {
				box_weights[p] = (ap_uint<box_L_MVTU_InP*L_objbox_Wbit>*)weights41[p];
			}
			else if (layer == 10) {
				box_weights[p] = (ap_uint<box_L_MVTU_InP*L_objbox_Wbit>*)weights44[p];
			}
			else if (layer == 14) {
				box_weights[p] = (ap_uint<box_L_MVTU_InP*L_objbox_Wbit>*)weights47[p];
			}
			else if (layer == 18) {
				box_weights[p] = (ap_uint<box_L_MVTU_InP*L_objbox_Wbit>*)weights50[p];
			}
		}

		cout << "Allocating ports for layer " << layer << endl;
		ofs << "layer: " << layer << endl;
		ofs << "total_iterations: " << TOTAL_ITERATIONS << endl;
		for (unsigned i = 0; i < SQUEEZE_WEIGHT_ITERATIONS; i++) {
			ap_axis temp;
			temp.data = 0;
			if (i < squeeze_weight_iterations[layer]) {
				for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
					temp.data( (p+1)*squeeze_L_MVTU_InP*L_Wbit-1, p*squeeze_L_MVTU_InP*L_Wbit ) = squeeze_weights[p][i];
				}
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < EXPAND_WEIGHT_ITERATIONS; i++) {
			ap_axis temp;
			temp.data = 0;
			if (i < expand_weight_iterations[layer]) {
				for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
					temp.data( (p+1)*expand_L_MVTU_InP*L_Wbit-1, p*expand_L_MVTU_InP*L_Wbit ) = expand3x3_weights[p][i];
				}
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < SQUEEZE_FACTOR_ITERATIONS; i++) {
			ap_axis tempA;
			ap_axis tempB;
			tempA.data = 0;
			tempB.data = 0;
			if (i < squeeze_factor_iterations[layer]) {
				for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
					tempA.data( (p+1)*L_Mbit-1, p*L_Mbit ) = squeeze_factorA[p][i];
					tempB.data( (p+1)*L_Mbit-1, p*L_Mbit ) = squeeze_factorB[p][i];
				}
			}
			weightsfactors_stream[layer].write(tempA);
			weightsfactors_stream[layer].write(tempB);
			ofs << hex << tempA.data << dec << endl;
			ofs << hex << tempB.data << dec << endl;
		}
		for (unsigned i = 0; i < EXPAND_FACTOR_ITERATIONS; i++) {
			ap_axis tempA;
			ap_axis tempB;
			tempA.data = 0;
			tempB.data = 0;
			if (i < expand_factor_iterations[layer]) {
				for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
					tempA.data( (p+1)*L_Mbit-1, p*L_Mbit ) = expand3x3_factorA[p][i];
					tempB.data( (p+1)*L_Mbit-1, p*L_Mbit ) = expand3x3_factorB[p][i];
				}
			}
			weightsfactors_stream[layer].write(tempA);
			weightsfactors_stream[layer].write(tempB);
			ofs << hex << tempA.data << dec << endl;
			ofs << hex << tempB.data << dec << endl;
		}
		
		for (unsigned i = 0; i < CLASS_WEIGHT_ITERATIONS; i++) {
			ap_axis temp;
			temp.data = 0;
			if (layer == 6 || layer == 10 || layer == 14 || layer == 18) {
				for (unsigned p = 0; p < class_L_MVTU_OutP; p++) {
					temp.data( (p+1)*class_L_MVTU_InP*L_Wbit-1, p*class_L_MVTU_InP*L_Wbit ) = class_weights[p][i];
				}
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < CLASS_FACTOR_ITERATIONS; i++) {
			ap_axis tempA;
			ap_axis tempB;
			tempA.data = 0;
			tempB.data = 0;
			if (layer == 6 || layer == 10 || layer == 14 || layer == 18) {
				for (unsigned p = 0; p < class_L_MVTU_OutP; p++) {
					tempA.data( (p+1)*L_Mbit-1, p*L_Mbit ) = class_factorA[p][i];
					tempB.data( (p+1)*L_Mbit-1, p*L_Mbit ) = class_factorB[p][i];
				}
			}
			weightsfactors_stream[layer].write(tempA);
			weightsfactors_stream[layer].write(tempB);
			ofs << hex << tempA.data << dec << endl;
			ofs << hex << tempB.data << dec << endl;
		}
		for (unsigned i = 0; i < OBJ_WEIGHT_ITERATIONS; i++) {
			ap_axis temp;
			temp.data = 0;
			if (layer == 6 || layer == 10 || layer == 14 || layer == 18) {
				for (unsigned p = 0; p < obj_L_MVTU_OutP; p++) {
					temp.data( (p+1)*obj_L_MVTU_InP*L_objbox_Wbit-1, p*obj_L_MVTU_InP*L_objbox_Wbit ) = obj_weights[p][i];
				}
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < BOX_WEIGHT_ITERATIONS; i++) {
			ap_axis temp;
			temp.data = 0;
			if (layer == 6 || layer == 10 || layer == 14 || layer == 18) {
				for (unsigned p = 0; p < box_L_MVTU_OutP; p++) {
					temp.data( (p+1)*box_L_MVTU_InP*L_objbox_Wbit-1, p*box_L_MVTU_InP*L_objbox_Wbit ) = box_weights[p][i];
				}
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
	}
	ofs << "obj_factor 0 " << factorA40[0][0] << endl;
	ofs << "obj_factor 1 " << factorA43[0][0] << endl;
	ofs << "obj_factor 2 " << factorA46[0][0] << endl;
	ofs << "obj_factor 3 " << factorA49[0][0] << endl;
	ofs << "box_factor 0 " << factorA41[0][0] << endl;
	ofs << "box_factor 1 " << factorA44[0][0] << endl;
	ofs << "box_factor 2 " << factorA47[0][0] << endl;
	ofs << "box_factor 3 " << factorA50[0][0] << endl;
	ofs.close();
	cout << "Writing weights complete" << endl;

	stream<ap_axis> empty;
	
	// cout << "weightsfactors_stream[0].nbytes: " << weightsfactors_stream[0].size()*64 << endl;
	// unsigned temp_size = weightsfactors_stream[0].size();
	// for (unsigned j = 0; j < temp_size; j++) {
	// 	ap_axis temp = weightsfactors_stream[0].read();
	// 	cout << hex << temp.data << dec << endl;
	// 	weightsfactors_stream[0].write(temp);
	// }

	stream<ap_axis > outputStream1("outputStream1");
	halfsqueezenet(weightsfactors_stream[0], empty,
					0, 0, 0, 0, 0, 0,
					13, 0);
	cout << "inputStream.nbytes: " << inputStream.size()*64 << endl;
	halfsqueezenet(inputStream, outputStream1,
					L1_Din_W, L1_Din_H, L2_Din_W, L2_Din_H, L2_Din_W >> 1, L2_Din_H >> 1,
					1, NUM_SAMPLES);
	cout << "outputStream1.nbytes: " << outputStream1.size()*64 << endl;
	cout << "Done fire1!" << endl;

	// for (unsigned j = 0; j < 1; j++) {
	// 	ap_axis temp = outputStream1.read();
	// 	cout << hex << temp.data << dec << endl;
	// 	outputStream1.write(temp);
	// }

	stream<ap_axis > outputStream2("outputStream2");
	halfsqueezenet(weightsfactors_stream[1], empty,
					0, 0, 0, 0, 0, 0,
					13, 0);
	halfsqueezenet(outputStream1, outputStream2,
					L3_Din_W, L3_Din_H, L4_Din_W, L4_Din_H, L4_Din_W >> 1, L4_Din_H >> 1,
					2, NUM_SAMPLES);
	cout << "Done fire2!" << endl;

	// for (unsigned j = 0; j < 1; j++) {
	// 	ap_axis temp = outputStream2.read();
	// 	cout << hex << temp.data << dec << endl;
	// 	outputStream2.write(temp);
	// }

	stream<ap_axis > outputStream3("outputStream3");
	halfsqueezenet(weightsfactors_stream[2], empty,
					0, 0, 0, 0, 0, 0,
					13, 0);
	halfsqueezenet(outputStream2, outputStream3,
					L5_Din_W, L5_Din_H, L6_Din_W, L6_Din_H, L6_Din_W >> 1, L6_Din_H >> 1,
					3, NUM_SAMPLES);
	cout << "Done fire3!" << endl;

	// for (unsigned j = 0; j < 1; j++) {
	// 	ap_axis temp = outputStream3.read();
	// 	cout << hex << temp.data << dec << endl;
	// 	outputStream3.write(temp);
	// }

	stream<ap_axis > outputStream7[4];
	for (unsigned i = 0; i < 4; i++) {
		stream<ap_axis > outputStream3_copy("outputStream3_copy");
		unsigned size = outputStream3.size();
		for (unsigned k = 0; k < size; k++) {
			ap_axis temp = outputStream3.read();
			outputStream3_copy.write(temp);
			if (i < 3)
				outputStream3.write(temp);
		}

		stream<ap_axis > outputStream4("outputStream4");
		halfsqueezenet(weightsfactors_stream[3 + i*4], empty,
						0, 0, 0, 0, 0, 0,
						13, 0);
		halfsqueezenet(outputStream3_copy, outputStream4,
						L7_Din_W, L7_Din_H, L8_Din_W, L8_Din_H, L8_Din_W, L8_Din_H,
						4, NUM_SAMPLES);
		cout << "Done fire4!" << endl;

		// for (unsigned j = 0; j < 1; j++) {
		// 	ap_axis temp = outputStream4.read();
		// 	cout << hex << temp.data << dec << endl;
		// 	outputStream4.write(temp);
		// }

		stream<ap_axis > outputStream5("outputStream5");
		halfsqueezenet(weightsfactors_stream[4 + i*4], empty,
						0, 0, 0, 0, 0, 0,
						13, 0);
		halfsqueezenet(outputStream4, outputStream5,
						L9_Din_W, L9_Din_H, L10_Din_W, L10_Din_H, L10_Din_W, L10_Din_H,
						5, NUM_SAMPLES);
		cout << "Done fire5!" << endl;

		// for (unsigned j = 0; j < 1; j++) {
		// 	ap_axis temp = outputStream5.read();
		// 	cout << hex << temp.data << dec << endl;
		// 	outputStream5.write(temp);
		// }

		stream<ap_axis > outputStream6("outputStream6");
		halfsqueezenet(weightsfactors_stream[5 + i*4], empty,
						0, 0, 0, 0, 0, 0,
						13, 0);
		halfsqueezenet(outputStream5, outputStream6,
						L11_Din_W, L11_Din_H, L12_Din_W, L12_Din_H, L12_Din_W, L12_Din_H,
						6, NUM_SAMPLES);
		cout << "Done fire6!" << endl;

		// for (unsigned j = 0; j < 1; j++) {
		// 	ap_axis temp = outputStream6.read();
		// 	cout << hex << temp.data << dec << endl;
		// 	outputStream6.write(temp);
		// }
		// cout << "outputStream6.size(): " << outputStream6.size() << endl;

		halfsqueezenet(weightsfactors_stream[6 + i*4], empty,
						0, 0, 0, 0, 0, 0,
						13, 0);
		halfsqueezenet(outputStream6, outputStream7[i],
						L13_Din_W, L13_Din_H, L14_Din_W, L14_Din_H, L14_Din_W, L14_Din_H,
						7, NUM_SAMPLES);
		cout << "Done fire7!" << endl;
	}

	const unsigned num_out_lines = NUM_SAMPLES;

	ap_int<L_Mbit> xmin_array[4][L_final_Din_W*L_final_Din_H];
	ap_int<L_Mbit> ymin_array[4][L_final_Din_W*L_final_Din_H];
	ap_int<L_Mbit> xmax_array[4][L_final_Din_W*L_final_Din_H];
	ap_int<L_Mbit> ymax_array[4][L_final_Din_W*L_final_Din_H];
	float obj_array[L_final_Din_W*L_final_Din_H];

	for (unsigned i = 0; i < num_out_lines; i++) {

		for (unsigned k = 0; k < 4; k++) {
			for (unsigned j = 0; j < L_final_Din_W*L_final_Din_H; j++) {
				ap_axis temp = outputStream7[k].read();
				xmin_array[k][j] = temp.data(L_Mbit-1, 0);
				ymin_array[k][j] = temp.data(L_Mbit*2-1, L_Mbit*1);
				xmax_array[k][j] = temp.data(L_Mbit*3-1, L_Mbit*2);
				ymax_array[k][j] = temp.data(L_Mbit*4-1, L_Mbit*3);
				ap_int<L_Mbit> temp_obj = temp.data(L_Mbit*5-1, L_Mbit*4);

				if (k == 0) {
					obj_array[j] = ((float)temp_obj/(1 << SCALE_BITS))*(float)factorA40[0][0];
				}
				else if (k == 1) {
					obj_array[j] += ((float)temp_obj/(1 << SCALE_BITS))*(float)factorA43[0][0];
				}
				else if (k == 1) {
					obj_array[j] += ((float)temp_obj/(1 << SCALE_BITS))*(float)factorA46[0][0];
				}
				else if (k == 1) {
					obj_array[j] += ((float)temp_obj/(1 << SCALE_BITS))*(float)factorA49[0][0];
				}
			}
		}

		float temp_max = -100;
		unsigned max_index = 0;
		for (unsigned j = 0; j < L_final_Din_W*L_final_Din_H; j++) {
			if (obj_array[j] > temp_max) {
				temp_max = obj_array[j];
				max_index = j;
			}
		}

		cout << "max_index: " << max_index << endl;

		unsigned obj_h = max_index/(width/16);
		unsigned obj_w = max_index%(width/16);

		cout << "obj_h: " << obj_h << endl;
		cout << "obj_w: " << obj_w << endl;

		float xmin_f = 0;
		float ymin_f = 0;
		float xmax_f = 0;
		float ymax_f = 0;
		float factors[4];
		factors[0] = (float)factorA41[0][0];
		factors[1] = (float)factorA44[0][0];
		factors[2] = (float)factorA47[0][0];
		factors[3] = (float)factorA50[0][0];
		for (unsigned k = 0; k < 4; k++) {
			cout << "xmin_array[" << k << "][max_index]: " << xmin_array[k][max_index] << endl;
			cout << "ymin_array[" << k << "][max_index]: " << ymin_array[k][max_index] << endl;
			cout << "xmax_array[" << k << "][max_index]: " << xmax_array[k][max_index] << endl;
			cout << "ymax_array[" << k << "][max_index]: " << ymax_array[k][max_index] << endl;
			xmin_f += (((float)xmin_array[k][max_index]/(1 << SCALE_BITS))*factors[k])/(1 << FACTOR_SCALE_BITS);
			ymin_f += (((float)ymin_array[k][max_index]/(1 << SCALE_BITS))*factors[k])/(1 << FACTOR_SCALE_BITS);
			xmax_f += (((float)xmax_array[k][max_index]/(1 << SCALE_BITS))*factors[k])/(1 << FACTOR_SCALE_BITS);
			ymax_f += (((float)ymax_array[k][max_index]/(1 << SCALE_BITS))*factors[k])/(1 << FACTOR_SCALE_BITS);
		}
		xmin_f /= 4;
		ymin_f /= 4;
		xmax_f /= 4;
		ymax_f /= 4;

		cout << "output: " << xmin_f << " " << ymin_f << " " << xmax_f << " " << ymax_f << endl;
		
		unsigned xmin = (xmin_f + obj_w*16)*((float)640.0/width);
		unsigned ymin = (ymin_f + obj_h*16)*((float)360.0/height);
		unsigned xmax = (xmax_f + obj_w*16)*((float)640.0/width);
		unsigned ymax = (ymax_f + obj_h*16)*((float)360.0/height);

		cout << "bndboxes: " << xmin << " " << ymin << " " << xmax << " " << ymax << endl;
	}

	cout << "Execution finished!" << endl;

	return 0;
}

#endif
