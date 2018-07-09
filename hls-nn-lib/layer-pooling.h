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