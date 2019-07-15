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

#include "sliding-window-unit.h"
#include "pooling-unit.h"
#include "misc.h"


template <	unsigned K,
			unsigned MAX_Din_W,
			unsigned MAX_Din_H,
			unsigned MAX_Cin,
			unsigned Ibit>
void POOL2D_NoP_variable_RECT(
	stream<ap_uint<MAX_Cin*Ibit> >& in,
	stream<ap_uint<MAX_Cin*Ibit> >& out,
	const unsigned Din_W,
	const unsigned Din_H, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned S = 2;
	const unsigned Dout_W = (Din_W >> 1) + (Din_W&0x1);
	const unsigned Dout_H = (Din_H >> 1) + (Din_H&0x1);
	const unsigned IntermediateDout_W = S*(Dout_W-1) + K;
	const unsigned IntermediateDout_H = S*(Dout_H-1) + K;

#ifdef CONV2_DEBUG
	cout << "Pool Din_W: " << Din_W << endl;
	cout << "Pool Din_H: " << Din_H << endl;
	cout << "Dout_W: " << Dout_W << endl;
	cout << "Dout_H: " << Dout_H << endl;
	cout << "IntermediateDout_W: " << IntermediateDout_W << endl;
	cout << "IntermediateDout_H: " << IntermediateDout_H << endl;
#endif
	const unsigned TopPad = (IntermediateDout_H - Din_H)/2;
	const unsigned BottomPad = (IntermediateDout_H - Din_H) - TopPad;
	const unsigned LeftPad = (IntermediateDout_W - Din_W)/2;
	const unsigned RightPad = (IntermediateDout_W - Din_W) - LeftPad;
#ifdef CONV2_DEBUG
	cout << "TopPad: " << TopPad << endl;
	cout << "BottomPad: " << BottomPad << endl;
	cout << "LeftPad: " << LeftPad << endl;
	cout << "RightPad: " << RightPad << endl;
#endif

	stream<ap_uint<MAX_Cin*Ibit> > samepad_out("samepad_out");
	SAMEPAD_variable_RECT<MAX_Cin, Ibit>(in, samepad_out, TopPad, BottomPad, LeftPad, RightPad, Din_W, Din_H, reps);

	stream<ap_uint<MAX_Cin*Ibit> > swu_out("swu_out");
	SWU_NoP_variable_RECT_S2<K, MAX_Din_W, MAX_Din_H, MAX_Cin, Ibit> (samepad_out, swu_out, IntermediateDout_W, IntermediateDout_H, reps);

	POOL_variable<Ibit, K, MAX_Cin, 1>(swu_out, out, Dout_W*Dout_H, reps);
}

template <	unsigned K,
			unsigned S,
			unsigned Din_W,
			unsigned Din_H,
			unsigned Cin,
			unsigned Ibit>
void POOL2D_NoP_RECT(
	stream<ap_uint<Cin*Ibit> >& in,
	stream<ap_uint<Cin*Ibit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned Dout_W = Din_W/S + (Din_W%S > 0);
	const unsigned Dout_H = Din_H/S + (Din_H%S > 0);
	const unsigned IntermediateDout_W = S*(Dout_W-1) + K;
	const unsigned IntermediateDout_H = S*(Dout_H-1) + K;
#ifdef CONV2_DEBUG
	cout << "Dout_W: " << Dout_W << endl;
	cout << "Dout_H: " << Dout_H << endl;
	cout << "IntermediateDout_W: " << IntermediateDout_W << endl;
	cout << "IntermediateDout_H: " << IntermediateDout_H << endl;
#endif
	const unsigned TopPad = (IntermediateDout_H - Din_H)/2;
	const unsigned BottomPad = (IntermediateDout_H - Din_H) - TopPad;
	const unsigned LeftPad = (IntermediateDout_W - Din_W)/2;
	const unsigned RightPad = (IntermediateDout_W - Din_W) - LeftPad;
#ifdef CONV2_DEBUG
	cout << "TopPad: " << TopPad << endl;
	cout << "BottomPad: " << BottomPad << endl;
	cout << "LeftPad: " << LeftPad << endl;
	cout << "RightPad: " << RightPad << endl;
#endif

	stream<ap_uint<Cin*Ibit> > samepad_out("samepad_out");
	SAMEPAD_RECT<TopPad, BottomPad, LeftPad, RightPad, Din_W, Din_H, Cin, Ibit>(in, samepad_out, reps);

	stream<ap_uint<Cin*Ibit> > swu_out("swu_out");
	SWU_NoP_RECT<K, S, IntermediateDout_W, IntermediateDout_H, Cin, Ibit> (samepad_out, swu_out, reps);

	POOL<Dout_W*Dout_H, Ibit, K, Cin, 1>(swu_out, out, reps);
}

template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Ibit>
void POOL2D_NoP(
	stream<ap_uint<Cin*Ibit> >& in,
	stream<ap_uint<Cin*Ibit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned Dout = Din/S + (Din%S > 0);
	const unsigned IntermediateDout = S*(Dout-1) + K;
#ifdef CONV2_DEBUG
	cout << "Dout: " << Dout << endl;
	cout << "IntermediateDout: " << IntermediateDout << endl;
#endif
	const unsigned TopLeftPad = (IntermediateDout - Din)/2;
	const unsigned BottomRightPad = (IntermediateDout - Din) - TopLeftPad;
#ifdef CONV2_DEBUG
	cout << "TopLeftPad: " << TopLeftPad << endl;
	cout << "BottomRightPad: " << BottomRightPad << endl;
#endif
	
	stream<ap_uint<Cin*Ibit> > samepad_out("samepad_out");
	SAMEPAD<TopLeftPad, BottomRightPad, Din, Cin, Ibit>(in, samepad_out, reps);

	stream<ap_uint<Cin*Ibit> > swu_out("swu_out");

	SWU_NoP<K, S, IntermediateDout, Cin, Ibit> (samepad_out, swu_out, reps);

	POOL<Dout*Dout, Ibit, K, Cin, 1>(swu_out, out, reps);
}

template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Ibit>
void POOL2D_KP(
	stream<ap_uint<Cin*Ibit> >& in,
	stream<ap_uint<Cin*Ibit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned Dout = Din/S + (Din%S > 0);
	const unsigned IntermediateDout = S*(Dout-1) + K;
#ifdef CONV2_DEBUG
	cout << "Dout: " << Dout << endl;
	cout << "IntermediateDout: " << IntermediateDout << endl;
#endif
	const unsigned TopLeftPad = (IntermediateDout - Din)/2;
	const unsigned BottomRightPad = (IntermediateDout - Din) - TopLeftPad;
#ifdef CONV2_DEBUG
	cout << "TopLeftPad: " << TopLeftPad << endl;
	cout << "BottomRightPad: " << BottomRightPad << endl;
#endif
	
	stream<ap_uint<Cin*Ibit> > samepad_out("samepad_out");
	SAMEPAD<TopLeftPad, BottomRightPad, Din, Cin, Ibit>(in, samepad_out, reps);

	stream<ap_uint<K*Cin*Ibit> > swu_out("swu_out");

	SWU_KP<K, S, IntermediateDout, Cin, Ibit> (samepad_out, swu_out, reps);

	POOL<Dout*Dout, Ibit, K, Cin, K>(swu_out, out, reps);
}

template <	unsigned Din,
			unsigned Cin,
			unsigned Ibit>
void GLOBAL_AVG_POOL(
	stream<ap_uint<Cin*Ibit> >& in,
	stream<ap_uint<Cin*Ibit> >& out,
	const unsigned reps = 1)
{
	for (unsigned rep = 0; rep < reps; rep++) {

		ap_uint<Cin*Ibit> accumulator = 0;
		for (unsigned i = 0; i < Din*Din; i++) {
#pragma HLS PIPELINE II=1
			ap_uint<Cin*Ibit> temp = in.read();
			accumulator = accumulator + temp;
		}
		out.write(accumulator);
		
	}
}