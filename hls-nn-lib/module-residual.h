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

#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include <iostream>
using namespace std;
#include <assert.h>

#include "layer-conv2d.h"
#include "misc.h"

template <	unsigned first_K,
			unsigned first_S,
			unsigned first_Din,
			unsigned first_Cin,
			unsigned first_Cout,
			unsigned first_Ibit,
			unsigned first_Wbit,
			unsigned first_Mbit,
			unsigned first_Abit,
			unsigned first_MVTU_InP,
			unsigned first_MVTU_OutP,
			
			unsigned second_K,
			unsigned second_S,
			unsigned second_Din,
			unsigned second_Cin,
			unsigned second_Cout,
			unsigned second_Ibit,
			unsigned second_Wbit,
			unsigned second_Mbit,
			unsigned second_Abit,
			unsigned second_MVTU_InP,
			unsigned second_MVTU_OutP,

			unsigned ScaleBits>
void NODIM_INCREASE_RESIDUAL_ACT (
	stream<ap_uint<first_Cin*first_Ibit> >& in,

	const ap_uint<first_MVTU_InP*first_Wbit> first_weights[first_MVTU_OutP][((first_Cin*first_K*first_K)/first_MVTU_InP)*(first_Cout/first_MVTU_OutP)], 
	const ap_int<first_Abit> first_factorA[first_MVTU_OutP][first_Cout/first_MVTU_OutP],
	const ap_int<first_Abit> first_factorB[first_MVTU_OutP][first_Cout/first_MVTU_OutP],

	const ap_uint<second_MVTU_InP*second_Wbit> second_weights[second_MVTU_OutP][((second_Cin*second_K*second_K)/second_MVTU_InP)*(second_Cout/second_MVTU_OutP)], 
	const ap_int<second_Abit> second_factorA[second_MVTU_OutP][second_Cout/second_MVTU_OutP],
	const ap_int<second_Abit> second_factorB[second_MVTU_OutP][second_Cout/second_MVTU_OutP],

	stream<ap_uint<second_Cout*second_Abit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	stream<ap_uint<first_Cin*first_Ibit> > in_out1("res_in_out1");
	stream<ap_uint<first_Cin*first_Ibit> > in_out2("res_in_out2");
const unsigned first_latency = (10+first_Din*first_K);
const unsigned second_latency = (10+second_Din*second_K);
#pragma HLS STREAM variable=in_out2 depth=first_latency+second_latency

	DuplicateStreams<first_Cin*first_Ibit, first_Din*first_Din>(in, in_out1, in_out2, reps);

	stream<ap_uint<first_Cout*first_Abit> > first_out("first_out");
	CONV2D_ACT_NoP<first_K, first_S, first_Din, first_Cin, first_Cout, first_Ibit, first_Wbit, first_Mbit, first_Abit, first_MVTU_InP, first_MVTU_OutP, ScaleBits>
	(in_out1, first_weights, first_factorA, first_factorB, first_out, reps);

	stream<ap_uint<second_Cout*second_Abit> > second_out("second_out");
	CONV2D_ACT_NoP<second_K, second_S, second_Din, second_Cin, second_Cout, second_Ibit, second_Wbit, second_Mbit, second_Abit, second_MVTU_InP, second_MVTU_OutP, ScaleBits>
	(first_out, second_weights, second_factorA, second_factorB, second_out, reps);

	AddStreams<first_Cin*first_Ibit, first_Din*first_Din>(in_out2, second_out, out, reps);
}

template <	unsigned first_K,
			unsigned first_S,
			unsigned first_Din,
			unsigned first_Cin,
			unsigned first_Cout,
			unsigned first_Ibit,
			unsigned first_Wbit,
			unsigned first_Mbit,
			unsigned first_Abit,
			unsigned first_MVTU_InP,
			unsigned first_MVTU_OutP,
			
			unsigned second_K,
			unsigned second_S,
			unsigned second_Din,
			unsigned second_Cin,
			unsigned second_Cout,
			unsigned second_Ibit,
			unsigned second_Wbit,
			unsigned second_Mbit,
			unsigned second_Abit,
			unsigned second_MVTU_InP,
			unsigned second_MVTU_OutP,

			unsigned ScaleBits>
void NODIM_INCREASE_RESIDUAL_ACT_THIN (
	stream<ap_uint<first_Cin*first_Ibit> >& in,

	const ap_uint<first_MVTU_InP*first_Wbit> first_weights[first_MVTU_OutP][((first_Cin*first_K*first_K)/first_MVTU_InP)*(first_Cout/first_MVTU_OutP)], 
	const ap_int<first_Abit> first_factorA[first_MVTU_OutP][first_Cout/first_MVTU_OutP],
	const ap_int<first_Abit> first_factorB[first_MVTU_OutP][first_Cout/first_MVTU_OutP],

	const ap_uint<second_MVTU_InP*second_Wbit> second_weights[second_MVTU_OutP][((second_Cin*second_K*second_K)/second_MVTU_InP)*(second_Cout/second_MVTU_OutP)], 
	const ap_int<second_Abit> second_factorA[second_MVTU_OutP][second_Cout/second_MVTU_OutP],
	const ap_int<second_Abit> second_factorB[second_MVTU_OutP][second_Cout/second_MVTU_OutP],

	stream<ap_uint<second_Cout*second_Abit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	stream<ap_uint<first_Cin*first_Ibit> > in_out1("res_in_out1");
	stream<ap_uint<first_Ibit> > in_out2("res_in_out2");
const unsigned first_latency = first_Cin*(10+first_Din*first_K);
const unsigned second_latency = second_Cin*(10+second_Din*second_K);
#pragma HLS STREAM variable=in_out2 depth=first_latency+second_latency

	DuplicateStreams_ReduceWidth<first_Cin*first_Ibit, first_Ibit, first_Din*first_Din>(in, in_out1, in_out2, reps);

	stream<ap_uint<first_Cout*first_Abit> > first_out("first_out");
	CONV2D_ACT_NoP<first_K, first_S, first_Din, first_Cin, first_Cout, first_Ibit, first_Wbit, first_Mbit, first_Abit, first_MVTU_InP, first_MVTU_OutP, ScaleBits>
	(in_out1, first_weights, first_factorA, first_factorB, first_out, reps);

	stream<ap_uint<second_Cout*second_Abit> > second_out("second_out");
	CONV2D_ACT_NoP<second_K, second_S, second_Din, second_Cin, second_Cout, second_Ibit, second_Wbit, second_Mbit, second_Abit, second_MVTU_InP, second_MVTU_OutP, ScaleBits>
	(first_out, second_weights, second_factorA, second_factorB, second_out, reps);

	AddStreams_ExpandWidth<first_Cin*first_Ibit, first_Ibit, first_Din*first_Din>(second_out, in_out2, out, reps);
}

template <	unsigned first_K,
			unsigned first_S,
			unsigned first_Din,
			unsigned first_Cin,
			unsigned first_Cout,
			unsigned first_Ibit,
			unsigned first_Wbit,
			unsigned first_Mbit,
			unsigned first_Abit,
			unsigned first_MVTU_InP,
			unsigned first_MVTU_OutP,
			
			unsigned second_K,
			unsigned second_S,
			unsigned second_Din,
			unsigned second_Cin,
			unsigned second_Cout,
			unsigned second_Ibit,
			unsigned second_Wbit,
			unsigned second_Mbit,
			unsigned second_Abit,
			unsigned second_MVTU_InP,
			unsigned second_MVTU_OutP,

			unsigned ScaleBits>
void NODIM_INCREASE_RESIDUAL_ACT_NOBRANCH (
	stream<ap_uint<first_Cin*first_Ibit> >& in,

	const ap_uint<first_MVTU_InP*first_Wbit> first_weights[first_MVTU_OutP][((first_Cin*first_K*first_K)/first_MVTU_InP)*(first_Cout/first_MVTU_OutP)], 
	const ap_int<first_Abit> first_factorA[first_MVTU_OutP][first_Cout/first_MVTU_OutP],
	const ap_int<first_Abit> first_factorB[first_MVTU_OutP][first_Cout/first_MVTU_OutP],

	const ap_uint<second_MVTU_InP*second_Wbit> second_weights[second_MVTU_OutP][((second_Cin*second_K*second_K)/second_MVTU_InP)*(second_Cout/second_MVTU_OutP)], 
	const ap_int<second_Abit> second_factorA[second_MVTU_OutP][second_Cout/second_MVTU_OutP],
	const ap_int<second_Abit> second_factorB[second_MVTU_OutP][second_Cout/second_MVTU_OutP],

	stream<ap_uint<second_Cout*second_Abit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	stream<ap_uint<first_Cin*first_Ibit> > in_out2("res_in_out2");
const unsigned first_latency = (10+first_Din*first_K);
const unsigned second_latency = 6;
#pragma HLS STREAM variable=in_out2 depth=first_latency+second_latency

	stream<ap_uint<first_Cout*first_Abit> > first_out("first_out");
	CONV2D_ACT_NoP_residual<first_K, first_S, first_Din, first_Cin, first_Cout, first_Ibit, first_Wbit, first_Mbit, first_Abit, first_MVTU_InP, first_MVTU_OutP, ScaleBits>
	(in, first_weights, first_factorA, first_factorB, first_out, in_out2, reps);

	stream<ap_uint<second_Cout*second_Abit> > second_out("second_out");
	CONV2D_ACT_NoP<second_K, second_S, second_Din, second_Cin, second_Cout, second_Ibit, second_Wbit, second_Mbit, second_Abit, second_MVTU_InP, second_MVTU_OutP, ScaleBits>
	(first_out, second_weights, second_factorA, second_factorB, second_out, reps);

	AddStreams<first_Cin*first_Ibit, first_Din*first_Din>(in_out2, second_out, out, reps);
}

template <	unsigned first_K,
			unsigned first_S,
			unsigned first_Din,
			unsigned first_Cin,
			unsigned first_Cout,
			unsigned first_Ibit,
			unsigned first_Wbit,
			unsigned first_Mbit,
			unsigned first_Abit,
			unsigned first_MVTU_InP,
			unsigned first_MVTU_OutP,
			
			unsigned second_K,
			unsigned second_S,
			unsigned second_Din,
			unsigned second_Cin,
			unsigned second_Cout,
			unsigned second_Ibit,
			unsigned second_Wbit,
			unsigned second_Mbit,
			unsigned second_Abit,
			unsigned second_MVTU_InP,
			unsigned second_MVTU_OutP,

			unsigned pool_K,
			unsigned pool_S,
			unsigned pool_Din,
			unsigned pool_Cin,
			unsigned pool_Ibit,

			unsigned short_K,
			unsigned short_S,
			unsigned short_Din,
			unsigned short_Cin,
			unsigned short_Cout,
			unsigned short_Ibit,
			unsigned short_Wbit,
			unsigned short_Mbit,
			unsigned short_Abit,
			unsigned short_MVTU_InP,
			unsigned short_MVTU_OutP,

			unsigned ScaleBits>
void DIM_INCREASE_RESIDUAL_ACT (
	stream<ap_uint<first_Cin*first_Ibit> >& in,

	const ap_uint<first_MVTU_InP*first_Wbit> first_weights[first_MVTU_OutP][((first_Cin*first_K*first_K)/first_MVTU_InP)*(first_Cout/first_MVTU_OutP)], 
	const ap_int<first_Abit> first_factorA[first_MVTU_OutP][first_Cout/first_MVTU_OutP],
	const ap_int<first_Abit> first_factorB[first_MVTU_OutP][first_Cout/first_MVTU_OutP],

	const ap_uint<second_MVTU_InP*second_Wbit> second_weights[second_MVTU_OutP][((second_Cin*second_K*second_K)/second_MVTU_InP)*(second_Cout/second_MVTU_OutP)], 
	const ap_int<second_Abit> second_factorA[second_MVTU_OutP][second_Cout/second_MVTU_OutP],
	const ap_int<second_Abit> second_factorB[second_MVTU_OutP][second_Cout/second_MVTU_OutP],

	const ap_uint<short_MVTU_InP*short_Wbit> short_weights[short_MVTU_OutP][((short_Cin*short_K*short_K)/short_MVTU_InP)*(short_Cout/short_MVTU_OutP)], 
	const ap_int<short_Abit> short_factorA[short_MVTU_OutP][short_Cout/short_MVTU_OutP],
	const ap_int<short_Abit> short_factorB[short_MVTU_OutP][short_Cout/short_MVTU_OutP],

	stream<ap_uint<second_Cout*second_Abit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	stream<ap_uint<first_Cin*first_Ibit> > in_out1("res_in_out1");
	stream<ap_uint<first_Cin*first_Ibit> > in_out2("res_in_out2");
const unsigned first_latency = (10+first_Din*first_K);
const unsigned pool_latency = (10+pool_Din*pool_K);
#pragma HLS STREAM variable=in_out2 depth=first_latency+pool_latency

	DuplicateStreams<first_Cin*first_Ibit, first_Din*first_Din>(in, in_out1, in_out2, reps);

	// Main branch
	stream<ap_uint<first_Cout*first_Abit> > first_out("first_out");
	CONV2D_ACT_NoP<first_K, first_S, first_Din, first_Cin, first_Cout, first_Ibit, first_Wbit, first_Mbit, first_Abit, first_MVTU_InP, first_MVTU_OutP, ScaleBits>
	(in_out1, first_weights, first_factorA, first_factorB, first_out, reps);

	stream<ap_uint<second_Cout*second_Abit> > second_out("second_out");
	CONV2D_ACT_NoP<second_K, second_S, second_Din, second_Cin, second_Cout, second_Ibit, second_Wbit, second_Mbit, second_Abit, second_MVTU_InP, second_MVTU_OutP, ScaleBits>
	(first_out, second_weights, second_factorA, second_factorB, second_out, reps);

	stream<ap_uint<pool_Cin*pool_Ibit> > pool_out("pool_out");
	POOL2D_NoP<pool_K, pool_S, pool_Din, pool_Cin, pool_Ibit>
	(second_out, pool_out, reps);

	// Short branch
	stream<ap_uint<short_Cout*short_Abit> > short_out("short_out");
	CONV2D_ACT_NoP<short_K, short_S, short_Din, short_Cin, short_Cout, short_Ibit, short_Wbit, short_Mbit, short_Abit, short_MVTU_InP, short_MVTU_OutP, ScaleBits>
	(in_out2, short_weights, short_factorA, short_factorB, short_out, reps);

	AddStreams<pool_Cin*pool_Ibit, (pool_Din/pool_S)*(pool_Din/pool_S)>(pool_out, short_out, out, reps);
}

template <	unsigned first_K,
			unsigned first_S,
			unsigned first_Din,
			unsigned first_Cin,
			unsigned first_Cout,
			unsigned first_Ibit,
			unsigned first_Wbit,
			unsigned first_Mbit,
			unsigned first_Abit,
			unsigned first_MVTU_InP,
			unsigned first_MVTU_OutP,
			
			unsigned second_K,
			unsigned second_S,
			unsigned second_Din,
			unsigned second_Cin,
			unsigned second_Cout,
			unsigned second_Ibit,
			unsigned second_Wbit,
			unsigned second_Mbit,
			unsigned second_Abit,
			unsigned second_MVTU_InP,
			unsigned second_MVTU_OutP,

			unsigned pool_K,
			unsigned pool_S,
			unsigned pool_Din,
			unsigned pool_Cin,
			unsigned pool_Ibit,

			unsigned short_K,
			unsigned short_S,
			unsigned short_Din,
			unsigned short_Cin,
			unsigned short_Cout,
			unsigned short_Ibit,
			unsigned short_Wbit,
			unsigned short_Mbit,
			unsigned short_Abit,
			unsigned short_MVTU_InP,
			unsigned short_MVTU_OutP,

			unsigned ScaleBits>
void DIM_INCREASE_RESIDUAL_ACT_NOBRANCH (
	stream<ap_uint<first_Cin*first_Ibit> >& in,

	const ap_uint<first_MVTU_InP*first_Wbit> first_weights[first_MVTU_OutP][((first_Cin*first_K*first_K)/first_MVTU_InP)*(first_Cout/first_MVTU_OutP)], 
	const ap_int<first_Abit> first_factorA[first_MVTU_OutP][first_Cout/first_MVTU_OutP],
	const ap_int<first_Abit> first_factorB[first_MVTU_OutP][first_Cout/first_MVTU_OutP],

	const ap_uint<second_MVTU_InP*second_Wbit> second_weights[second_MVTU_OutP][((second_Cin*second_K*second_K)/second_MVTU_InP)*(second_Cout/second_MVTU_OutP)], 
	const ap_int<second_Abit> second_factorA[second_MVTU_OutP][second_Cout/second_MVTU_OutP],
	const ap_int<second_Abit> second_factorB[second_MVTU_OutP][second_Cout/second_MVTU_OutP],

	const ap_uint<short_MVTU_InP*short_Wbit> short_weights[short_MVTU_OutP][((short_Cin*short_K*short_K)/short_MVTU_InP)*(short_Cout/short_MVTU_OutP)], 
	const ap_int<short_Abit> short_factorA[short_MVTU_OutP][short_Cout/short_MVTU_OutP],
	const ap_int<short_Abit> short_factorB[short_MVTU_OutP][short_Cout/short_MVTU_OutP],

	stream<ap_uint<second_Cout*second_Abit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	stream<ap_uint<first_Cin*first_Ibit> > in_out2("res_in_out2");
const unsigned first_latency = 6;
const unsigned pool_latency = (10+pool_Din*pool_K);
#pragma HLS STREAM variable=in_out2 depth=first_latency+pool_latency

	// Main branch
	stream<ap_uint<first_Cout*first_Abit> > first_out("first_out");
	CONV2D_ACT_NoP_residual<first_K, first_S, first_Din, first_Cin, first_Cout, first_Ibit, first_Wbit, first_Mbit, first_Abit, first_MVTU_InP, first_MVTU_OutP, ScaleBits>
	(in, first_weights, first_factorA, first_factorB, first_out, in_out2, reps);

	stream<ap_uint<second_Cout*second_Abit> > second_out("second_out");
	CONV2D_ACT_NoP<second_K, second_S, second_Din, second_Cin, second_Cout, second_Ibit, second_Wbit, second_Mbit, second_Abit, second_MVTU_InP, second_MVTU_OutP, ScaleBits>
	(first_out, second_weights, second_factorA, second_factorB, second_out, reps);

	stream<ap_uint<pool_Cin*pool_Ibit> > pool_out("pool_out");
	POOL2D_NoP<pool_K, pool_S, pool_Din, pool_Cin, pool_Ibit>
	(second_out, pool_out, reps);

	// Short branch
	stream<ap_uint<short_Cout*short_Abit> > short_out("short_out");
	CONV2D_ACT_NoP<short_K, short_S, short_Din, short_Cin, short_Cout, short_Ibit, short_Wbit, short_Mbit, short_Abit, short_MVTU_InP, short_MVTU_OutP, ScaleBits>
	(in_out2, short_weights, short_factorA, short_factorB, short_out, reps);

	AddStreams<pool_Cin*pool_Ibit, (pool_Din/pool_S)*(pool_Din/pool_S)>(pool_out, short_out, out, reps);
}