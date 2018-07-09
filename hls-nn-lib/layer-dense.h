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

#include "matrix-vector-unit.h"

template <	unsigned Din,
			unsigned Dout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void DENSE_ACT(
	stream<ap_uint<InP*Ibit> >& in,
	const ap_uint<InP*Wbit> weights[OutP][(Din/InP)*(Dout/OutP)],
	const ap_int<Mbit> factorA[OutP][Dout/OutP],
	const ap_int<Mbit> factorB[OutP][Dout/OutP],
	stream<ap_uint<OutP*Abit> >& out,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	MVAU_rowfirst<1, Ibit, Wbit, Mbit, Abit, Din, Dout, InP, OutP, ScaleBits, FactorScaleBits>(in, weights, factorA, factorB, out, reps);
}

template <	unsigned Din,
			unsigned Dout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits>
void DENSE_NOACT(
	stream<ap_uint<InP*Ibit> >& in,
	const ap_uint<InP*Wbit> weights[OutP][(Din/InP)*(Dout/OutP)],
	stream<ap_uint<OutP*Mbit> >& out,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	MVU_rowfirst<1, Ibit, Wbit, Mbit, Din, Dout, InP, OutP>(in, weights, out, reps);
}
