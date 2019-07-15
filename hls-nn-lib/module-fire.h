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

template <	unsigned squeeze_K,
			unsigned squeeze_S,
			unsigned squeeze_Din,
			unsigned squeeze_Cin,
			unsigned squeeze_Cout,
			unsigned squeeze_Ibit,
			unsigned squeeze_Wbit,
			unsigned squeeze_Mbit,
			unsigned squeeze_Abit,
			unsigned squeeze_MVTU_InP,
			unsigned squeeze_MVTU_OutP,
			
			unsigned expand1x1_K,
			unsigned expand1x1_S,
			unsigned expand1x1_Din,
			unsigned expand1x1_Cin,
			unsigned expand1x1_Cout,
			unsigned expand1x1_Ibit,
			unsigned expand1x1_Wbit,
			unsigned expand1x1_Mbit,
			unsigned expand1x1_Abit,
			unsigned expand1x1_MVTU_InP,
			unsigned expand1x1_MVTU_OutP,

			unsigned expand3x3_K,
			unsigned expand3x3_S,
			unsigned expand3x3_Din,
			unsigned expand3x3_Cin,
			unsigned expand3x3_Cout,
			unsigned expand3x3_Ibit,
			unsigned expand3x3_Wbit,
			unsigned expand3x3_Mbit,
			unsigned expand3x3_Abit,
			unsigned expand3x3_MVTU_InP,
			unsigned expand3x3_MVTU_OutP,

			unsigned ScaleBits>
void FIRE_ACT(
	stream<ap_uint<squeeze_Cin*squeeze_Ibit> >& in,

	const ap_uint<squeeze_MVTU_InP*squeeze_Wbit> squeeze_weights[squeeze_MVTU_OutP][((squeeze_Cin*squeeze_K*squeeze_K)/squeeze_MVTU_InP)*(squeeze_Cout/squeeze_MVTU_OutP)],
	const ap_int<squeeze_Mbit> squeeze_factorA[squeeze_MVTU_OutP][squeeze_Cout/squeeze_MVTU_OutP],
	const ap_int<squeeze_Mbit> squeeze_factorB[squeeze_MVTU_OutP][squeeze_Cout/squeeze_MVTU_OutP],

	const ap_uint<expand1x1_MVTU_InP*expand1x1_Wbit> expand1x1_weights[expand1x1_MVTU_OutP][((expand1x1_Cin*expand1x1_K*expand1x1_K)/expand1x1_MVTU_InP)*(expand1x1_Cout/expand1x1_MVTU_OutP)], 
	const ap_int<expand1x1_Mbit> expand1x1_factorA[expand1x1_MVTU_OutP][expand1x1_Cout/expand1x1_MVTU_OutP],
	const ap_int<expand1x1_Mbit> expand1x1_factorB[expand1x1_MVTU_OutP][expand1x1_Cout/expand1x1_MVTU_OutP],

	const ap_uint<expand3x3_MVTU_InP*expand3x3_Wbit> expand3x3_weights[expand3x3_MVTU_OutP][((expand3x3_Cin*expand3x3_K*expand3x3_K)/expand3x3_MVTU_InP)*(expand3x3_Cout/expand3x3_MVTU_OutP)], 
	const ap_int<expand3x3_Mbit> expand3x3_factorA[expand3x3_MVTU_OutP][expand3x3_Cout/expand3x3_MVTU_OutP],
	const ap_int<expand3x3_Mbit> expand3x3_factorB[expand3x3_MVTU_OutP][expand3x3_Cout/expand3x3_MVTU_OutP],

	stream<ap_uint<expand1x1_Cout*expand1x1_Abit+expand3x3_Cout*expand3x3_Abit> >& out, 
	const unsigned reps = 1)
{
	static_assert( expand1x1_Din == expand3x3_Din, "For FIRE module, expand1x1_Din is not equal to expand3x3_Din");

#pragma HLS DATAFLOW

	stream<ap_uint<squeeze_Cout*squeeze_Abit> > squeeze_out("squeeze_out");
	CONV2D_ACT_NoP<squeeze_K, squeeze_S, squeeze_Din, squeeze_Cin, squeeze_Cout, squeeze_Ibit, squeeze_Wbit, squeeze_Mbit, squeeze_Abit, squeeze_MVTU_InP, squeeze_MVTU_OutP, ScaleBits>
	(in, squeeze_weights, squeeze_factorA, squeeze_factorB, squeeze_out, reps);
	
	stream<ap_uint<squeeze_Cout*squeeze_Abit> > squeeze_out1x1("squeeze_out1x1");
	stream<ap_uint<squeeze_Cout*squeeze_Abit> > squeeze_out3x3("squeeze_out3x3");
	DuplicateStreams<squeeze_Cout*squeeze_Abit, squeeze_Din*squeeze_Din>(squeeze_out, squeeze_out1x1, squeeze_out3x3, reps);

	stream<ap_uint<expand1x1_Cout*expand1x1_Abit> > expand_out1x1("expand_out1x1");
#pragma HLS STREAM variable=expand_out1x1 depth=expand3x3_Din*expand3x3_K+48
	CONV2D_ACT_NoP<expand1x1_K, expand1x1_S, expand1x1_Din, expand1x1_Cin, expand1x1_Cout, expand1x1_Ibit, expand1x1_Wbit, expand1x1_Mbit, expand1x1_Abit, expand1x1_MVTU_InP, expand1x1_MVTU_OutP, ScaleBits>
	(squeeze_out1x1, expand1x1_weights, expand1x1_factorA, expand1x1_factorB, expand_out1x1, reps);

	stream<ap_uint<expand3x3_Cout*expand3x3_Abit> > expand_out3x3("expand_out3x3");
	CONV2D_ACT_NoP<expand3x3_K, expand3x3_S, expand3x3_Din, expand3x3_Cin, expand3x3_Cout, expand3x3_Ibit, expand3x3_Wbit, expand3x3_Mbit, expand3x3_Abit, expand3x3_MVTU_InP, expand3x3_MVTU_OutP, ScaleBits>
	(squeeze_out3x3, expand3x3_weights, expand3x3_factorA, expand3x3_factorB, expand_out3x3, reps);

	ConcatStreams<expand1x1_Cout*expand1x1_Abit, expand3x3_Cout*expand3x3_Abit, expand1x1_Din*expand1x1_Din>(expand_out1x1, expand_out3x3, out, reps);
}

template <	unsigned squeeze_K,
			unsigned squeeze_S,
			unsigned squeeze_Din,
			unsigned squeeze_Cin,
			unsigned squeeze_Cout,
			unsigned squeeze_Ibit,
			unsigned squeeze_Wbit,
			unsigned squeeze_Mbit,
			unsigned squeeze_Abit,
			unsigned squeeze_MVTU_InP,
			unsigned squeeze_MVTU_OutP,
			
			unsigned expand1x1_K,
			unsigned expand1x1_S,
			unsigned expand1x1_Din,
			unsigned expand1x1_Cin,
			unsigned expand1x1_Cout,
			unsigned expand1x1_Ibit,
			unsigned expand1x1_Wbit,
			unsigned expand1x1_Mbit,
			unsigned expand1x1_MVTU_InP,
			unsigned expand1x1_MVTU_OutP,

			unsigned expand3x3_K,
			unsigned expand3x3_S,
			unsigned expand3x3_Din,
			unsigned expand3x3_Cin,
			unsigned expand3x3_Cout,
			unsigned expand3x3_Ibit,
			unsigned expand3x3_Wbit,
			unsigned expand3x3_Mbit,
			unsigned expand3x3_MVTU_InP,
			unsigned expand3x3_MVTU_OutP,

			unsigned ScaleBits>
void FIRE_NOACT(
	stream<ap_uint<squeeze_Cin*squeeze_Ibit> >& in,

	const ap_uint<squeeze_MVTU_InP*squeeze_Wbit> squeeze_weights[squeeze_MVTU_OutP][((squeeze_Cin*squeeze_K*squeeze_K)/squeeze_MVTU_InP)*(squeeze_Cout/squeeze_MVTU_OutP)],
	const ap_int<squeeze_Abit> squeeze_factorA[squeeze_MVTU_OutP][squeeze_Cout/squeeze_MVTU_OutP],
	const ap_int<squeeze_Abit> squeeze_factorB[squeeze_MVTU_OutP][squeeze_Cout/squeeze_MVTU_OutP],

	const ap_uint<expand1x1_MVTU_InP*expand1x1_Wbit> expand1x1_weights[expand1x1_MVTU_OutP][((expand1x1_Cin*expand1x1_K*expand1x1_K)/expand1x1_MVTU_InP)*(expand1x1_Cout/expand1x1_MVTU_OutP)], 

	const ap_uint<expand3x3_MVTU_InP*expand3x3_Wbit> expand3x3_weights[expand3x3_MVTU_OutP][((expand3x3_Cin*expand3x3_K*expand3x3_K)/expand3x3_MVTU_InP)*(expand3x3_Cout/expand3x3_MVTU_OutP)], 
	
	stream<ap_uint<expand1x1_Cout*expand1x1_Mbit+expand3x3_Cout*expand3x3_Mbit> >& out, 
	const unsigned reps = 1)
{
	static_assert( expand1x1_Din == expand3x3_Din, "For FIRE module, expand1x1_Din is not equal to expand3x3_Din");

#pragma HLS DATAFLOW

	stream<ap_uint<squeeze_Cout*squeeze_Abit> > squeeze_out("squeeze_out");
	CONV2D_ACT_NoP<squeeze_K, squeeze_S, squeeze_Din, squeeze_Cin, squeeze_Cout, squeeze_Ibit, squeeze_Wbit, squeeze_Mbit, squeeze_Abit, squeeze_MVTU_InP, squeeze_MVTU_OutP, ScaleBits>
	(in, squeeze_weights, squeeze_factorA, squeeze_factorB, squeeze_out, reps);
	
	stream<ap_uint<squeeze_Cout*squeeze_Abit> > squeeze_out1x1("squeeze_out1x1");
	stream<ap_uint<squeeze_Cout*squeeze_Abit> > squeeze_out3x3("squeeze_out3x3");
	DuplicateStreams<squeeze_Cout*squeeze_Abit, squeeze_Din*squeeze_Din>(squeeze_out, squeeze_out1x1, squeeze_out3x3, reps);

	stream<ap_uint<expand1x1_Cout*expand1x1_Mbit> > expand_out1x1("expand_out1x1");
#pragma HLS STREAM variable=expand_out1x1 depth=expand3x3_Din*expand3x3_K+48
	CONV2D_NOACT_NoP<expand1x1_K, expand1x1_S, expand1x1_Din, expand1x1_Cin, expand1x1_Cout, expand1x1_Ibit, expand1x1_Wbit, expand1x1_Mbit, expand1x1_MVTU_InP, expand1x1_MVTU_OutP, ScaleBits>
	(squeeze_out1x1, expand1x1_weights, expand_out1x1, reps);

	stream<ap_uint<expand3x3_Cout*expand3x3_Mbit> > expand_out3x3("expand_out3x3");
	CONV2D_NOACT_NoP<expand3x3_K, expand3x3_S, expand3x3_Din, expand3x3_Cin, expand3x3_Cout, expand3x3_Ibit, expand3x3_Wbit, expand3x3_Mbit, expand3x3_MVTU_InP, expand3x3_MVTU_OutP, ScaleBits>
	(squeeze_out3x3, expand3x3_weights, expand_out3x3, reps);

	ConcatStreams<expand1x1_Cout*expand1x1_Mbit, expand3x3_Cout*expand3x3_Mbit, expand1x1_Din*expand1x1_Din>(expand_out1x1, expand_out3x3, out, reps);
}

template <	unsigned squeeze_K,
			unsigned squeeze_S,
			unsigned squeeze_MAX_Din,
			unsigned squeeze_MAX_Cin,
			unsigned squeeze_MAX_Cout,
			unsigned squeeze_Ibit,
			unsigned squeeze_Wbit,
			unsigned squeeze_Mbit,
			unsigned squeeze_Abit,
			unsigned squeeze_MVTU_InP,
			unsigned squeeze_MVTU_OutP,

			unsigned expand3x3_K,
			unsigned expand3x3_S,
			unsigned expand3x3_MAX_Din,
			unsigned expand3x3_MAX_Cin,
			unsigned expand3x3_MAX_Cout,
			unsigned expand3x3_Ibit,
			unsigned expand3x3_Wbit,
			unsigned expand3x3_Mbit,
			unsigned expand3x3_Abit,
			unsigned expand3x3_MVTU_InP,
			unsigned expand3x3_MVTU_OutP,

			unsigned ScaleBits,
			unsigned FactorScaleBits>
void HALFFIRE_ACT_variable(
	stream<ap_uint<squeeze_MAX_Cin*squeeze_Ibit> >& in,

	const ap_uint<squeeze_MVTU_InP*squeeze_Wbit> squeeze_weights[squeeze_MVTU_OutP][((squeeze_MAX_Cin*squeeze_K*squeeze_K)/squeeze_MVTU_InP)*(squeeze_MAX_Cout/squeeze_MVTU_OutP)],
	const ap_int<squeeze_Mbit> squeeze_factorA[squeeze_MVTU_OutP][squeeze_MAX_Cout/squeeze_MVTU_OutP],
	const ap_int<squeeze_Mbit> squeeze_factorB[squeeze_MVTU_OutP][squeeze_MAX_Cout/squeeze_MVTU_OutP],

	const ap_uint<expand3x3_MVTU_InP*expand3x3_Wbit> expand3x3_weights[expand3x3_MVTU_OutP][((expand3x3_MAX_Cin*expand3x3_K*expand3x3_K)/expand3x3_MVTU_InP)*(expand3x3_MAX_Cout/expand3x3_MVTU_OutP)], 
	const ap_int<expand3x3_Mbit> expand3x3_factorA[expand3x3_MVTU_OutP][expand3x3_MAX_Cout/expand3x3_MVTU_OutP],
	const ap_int<expand3x3_Mbit> expand3x3_factorB[expand3x3_MVTU_OutP][expand3x3_MAX_Cout/expand3x3_MVTU_OutP],

	stream<ap_uint<expand3x3_MAX_Cout*expand3x3_Abit> >& out,

	const unsigned squeeze_Din,
	// const unsigned squeeze_Cin,
	// const unsigned squeeze_Cout,

	const unsigned expand_Din,
	// const unsigned expand_Cin,
	// const unsigned expand_Cout,

	const unsigned reps = 1)
{
#pragma HLS DATAFLOW
	stream<ap_uint<squeeze_MAX_Cout*squeeze_Abit> > squeeze_out("squeeze_out");
	CONV2D_1x1_ACT_NoP_variable<squeeze_MAX_Din, squeeze_MAX_Cin, squeeze_MAX_Cout, squeeze_Ibit, squeeze_Wbit, squeeze_Mbit, squeeze_Abit, squeeze_MVTU_InP, squeeze_MVTU_OutP, ScaleBits, FactorScaleBits>
	(in, squeeze_weights, squeeze_factorA, squeeze_factorB, squeeze_out, squeeze_Din, /*squeeze_Cin, squeeze_Cout,*/ reps);	

	CONV2D_ACT_NoP_variable<expand3x3_K, expand3x3_MAX_Din, expand3x3_MAX_Cin, expand3x3_MAX_Cout, expand3x3_Ibit, expand3x3_Wbit, expand3x3_Mbit, expand3x3_Abit, expand3x3_MVTU_InP, expand3x3_MVTU_OutP, ScaleBits, FactorScaleBits>
	(squeeze_out, expand3x3_weights, expand3x3_factorA, expand3x3_factorB, out, expand_Din, /*expand_Cin, expand_Cout,*/ reps);
}

template <	unsigned squeeze_K,
			unsigned squeeze_S,
			unsigned squeeze_MAX_Din_W,
			unsigned squeeze_MAX_Din_H,
			unsigned squeeze_MAX_Cin,
			unsigned squeeze_MAX_Cout,
			unsigned squeeze_Ibit,
			unsigned squeeze_Wbit,
			unsigned squeeze_Mbit,
			unsigned squeeze_Abit,
			unsigned squeeze_MVTU_InP,
			unsigned squeeze_MVTU_OutP,

			unsigned expand3x3_K,
			unsigned expand3x3_S,
			unsigned expand3x3_MAX_Din_W,
			unsigned expand3x3_MAX_Din_H,
			unsigned expand3x3_MAX_Cin,
			unsigned expand3x3_MAX_Cout,
			unsigned expand3x3_Ibit,
			unsigned expand3x3_Wbit,
			unsigned expand3x3_Mbit,
			unsigned expand3x3_Abit,
			unsigned expand3x3_MVTU_InP,
			unsigned expand3x3_MVTU_OutP,

			unsigned ScaleBits,
			unsigned FactorScaleBits>
void HALFFIRE_ACT_variable_RECT(
	stream<ap_uint<squeeze_MAX_Cin*squeeze_Ibit> >& in,

	const ap_uint<squeeze_MVTU_InP*squeeze_Wbit> squeeze_weights[squeeze_MVTU_OutP][((squeeze_MAX_Cin*squeeze_K*squeeze_K)/squeeze_MVTU_InP)*(squeeze_MAX_Cout/squeeze_MVTU_OutP)],
	const ap_int<squeeze_Mbit> squeeze_factorA[squeeze_MVTU_OutP][squeeze_MAX_Cout/squeeze_MVTU_OutP],
	const ap_int<squeeze_Mbit> squeeze_factorB[squeeze_MVTU_OutP][squeeze_MAX_Cout/squeeze_MVTU_OutP],

	const ap_uint<expand3x3_MVTU_InP*expand3x3_Wbit> expand3x3_weights[expand3x3_MVTU_OutP][((expand3x3_MAX_Cin*expand3x3_K*expand3x3_K)/expand3x3_MVTU_InP)*(expand3x3_MAX_Cout/expand3x3_MVTU_OutP)], 
	const ap_int<expand3x3_Mbit> expand3x3_factorA[expand3x3_MVTU_OutP][expand3x3_MAX_Cout/expand3x3_MVTU_OutP],
	const ap_int<expand3x3_Mbit> expand3x3_factorB[expand3x3_MVTU_OutP][expand3x3_MAX_Cout/expand3x3_MVTU_OutP],

	stream<ap_uint<expand3x3_MAX_Cout*expand3x3_Abit> >& out,

	const unsigned squeeze_Din_W,
	const unsigned squeeze_Din_H,
	// const unsigned squeeze_Cin,
	// const unsigned squeeze_Cout,

	const unsigned expand_Din_W,
	const unsigned expand_Din_H,
	// const unsigned expand_Cin,
	// const unsigned expand_Cout,

	const unsigned reps = 1)
{
#pragma HLS DATAFLOW
	stream<ap_uint<squeeze_MAX_Cout*squeeze_Abit> > squeeze_out("squeeze_out");
	CONV2D_1x1_ACT_NoP_variable_RECT<squeeze_MAX_Din_W, squeeze_MAX_Din_H, squeeze_MAX_Cin, squeeze_MAX_Cout, squeeze_Ibit, squeeze_Wbit, squeeze_Mbit, squeeze_Abit, squeeze_MVTU_InP, squeeze_MVTU_OutP, ScaleBits, FactorScaleBits>
	(in, squeeze_weights, squeeze_factorA, squeeze_factorB, squeeze_out, squeeze_Din_W, squeeze_Din_H, /*squeeze_Cin, squeeze_Cout,*/ reps);

	CONV2D_ACT_NoP_variable_RECT<expand3x3_K, expand3x3_MAX_Din_W, expand3x3_MAX_Din_H, expand3x3_MAX_Cin, expand3x3_MAX_Cout, expand3x3_Ibit, expand3x3_Wbit, expand3x3_Mbit, expand3x3_Abit, expand3x3_MVTU_InP, expand3x3_MVTU_OutP, ScaleBits, FactorScaleBits>
	(squeeze_out, expand3x3_weights, expand3x3_factorA, expand3x3_factorB, out, expand_Din_W, expand_Din_H, /*expand_Cin, expand_Cout,*/ reps);
}