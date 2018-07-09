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

// #define POOL_DEBUG

template <	unsigned NumVecs,
			unsigned Ibit,
			unsigned K,
			unsigned Cin,
			unsigned InP>
void POOL(
	stream<ap_uint<InP*Cin*Ibit> >& vec,
	stream<ap_uint<Cin*Ibit> >& out,
	const unsigned reps = 1)
{
	static_assert( (K*K)%InP == 0, "K*K mod InP is not 0" );

	ap_uint<Cin*Ibit> result;
	unsigned wVec = 0;

	for (unsigned rep = 0; rep < reps*NumVecs*(K*K)/InP; rep++) {
#pragma HLS PIPELINE II=1

		if (wVec == 0)
			result = 0;

		ap_uint<InP*Cin*Ibit> tempVec = vec.read();

		for (unsigned c = 0; c < Cin; c++) {
#pragma HLS UNROLL
			for (unsigned p = 0; p < InP; p++) {
				ap_uint<Ibit> temp = tempVec( (p*Cin+c+1)*Ibit-1 , (p*Cin+c)*Ibit );
				
				result( (c+1)*Ibit-1, c*Ibit ) = (temp > result( (c+1)*Ibit-1, c*Ibit )) ? temp : result( (c+1)*Ibit-1, c*Ibit );
			}
		}

		if (wVec == (K*K)/InP-1)
			out.write(result);
		
		if (wVec == (K*K)/InP-1)
			wVec = 0;
		else
			wVec++;
	}
}