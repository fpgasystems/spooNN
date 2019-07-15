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

// #define SWU_DEBUG

template <	unsigned K,
			unsigned S,
			unsigned Din_W,
			unsigned Din_H,
			unsigned Cin,
			unsigned Ibit>
void SWU_NoP_RECT(
	stream<ap_uint<Cin*Ibit> >& in, 
	stream<ap_uint<Cin*Ibit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( (Din_W-K)%S == 0, "(Din_W-K) mod S is not 0");
	static_assert( (Din_H-K)%S == 0, "(Din_H-K) mod S is not 0");
	static_assert( K >= S, "K is not >= than S");

	const unsigned steps = (Din_W-K)/S+1;
	const unsigned line_buffer_size = K*Din_W;
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P

	ap_uint<Cin*Ibit> temp_in;

	ap_uint<1> initial_fill = 0;
	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;

	for (unsigned rep = 0; rep < reps*Din_H; rep++) {

		if (h == Din_H) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0; w < Din_W; w++) {
#pragma HLS PIPELINE II=1
			temp_in = in.read();
			
			unsigned line_buffer_pointer = pointer + w;
			if (line_buffer_pointer >= line_buffer_size) {
				line_buffer_pointer = line_buffer_pointer - line_buffer_size;
			}
#ifdef SWU_DEBUG
			cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
			line_buffer[line_buffer_pointer] = temp_in;
		}

		stride += 1;
		pointer += Din_W;
		if (pointer >= line_buffer_size) {
			pointer = pointer - line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;

			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K); i++ ) {
#pragma HLS PIPELINE II=1
				unsigned read_address = (pointer+s*S) + y*Din_W + x;

				if (read_address >= line_buffer_size)
					read_address = read_address - line_buffer_size;
#ifdef SWU_DEBUG
				cout << "read_address: " << read_address << endl;
#endif
				ap_uint<Cin*Ibit> temp_out = line_buffer[read_address];
				out.write(temp_out);

				if (x == K-1) {
					x = 0;
					if (y == K-1) {
						y = 0;
						if (s == steps-1)
							s = 0;
						else
							s++;
					}
					else
						y++;
				}
				else
					x++;
			}
		}
	}
}

template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Ibit>
void SWU_NoP(
	stream<ap_uint<Cin*Ibit> >& in, 
	stream<ap_uint<Cin*Ibit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( (Din-K)%S == 0, "(Din-K) mod S is not 0");
	static_assert( K >= S, "K is not >= than S");

	const unsigned steps = (Din-K)/S+1;
	const unsigned line_buffer_size = K*Din;
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P

	ap_uint<Cin*Ibit> temp_in;

	ap_uint<1> initial_fill = 0;
	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;

	for (unsigned rep = 0; rep < reps*Din; rep++) {

		if (h == Din) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0; w < Din; w++) {
#pragma HLS PIPELINE II=1
			temp_in = in.read();
			
			unsigned line_buffer_pointer = pointer + w;
			if (line_buffer_pointer >= line_buffer_size)
				line_buffer_pointer = line_buffer_pointer - line_buffer_size;
#ifdef SWU_DEBUG
			cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
			line_buffer[line_buffer_pointer] = temp_in;	
		}

		stride += 1;
		pointer += Din;
		if (pointer >= line_buffer_size) {
			pointer = pointer - line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;
	
			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K); i++ ) {
#pragma HLS PIPELINE II=1
				
				unsigned read_address = (pointer+s*S) + y*Din + x;

				if (read_address >= line_buffer_size)
					read_address = read_address - line_buffer_size;
#ifdef SWU_DEBUG
				cout << "read_address: " << read_address << endl;
#endif
				ap_uint<Cin*Ibit> temp_out = line_buffer[read_address];
				out.write(temp_out);

				if (x == K-1) {
					x = 0;
					if (y == K-1) {
						y = 0;
						if (s == steps-1)
							s = 0;
						else
							s++;
					}
					else
						y++;
				}
				else
					x++;
			}


		}
	}
}

template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Ibit>
void SWU_KP(
	stream<ap_uint<Cin*Ibit> >& in, 
	stream<ap_uint<K*Cin*Ibit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( (Din-K)%S == 0, "(Din-K) mod S is not 0");
	static_assert( K >= S, "K is not >= than S");

	const unsigned steps = (Din-K)/S+1;
	const unsigned line_buffer_size = Din*K;

#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete

	ap_uint<Cin*Ibit> temp_in;

	ap_uint<1> initial_fill = 0;
	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;

	for (unsigned rep = 0; rep < reps*Din; rep++) {

		if (h == Din) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0; w < Din; w++) {
#pragma HLS PIPELINE II=1
			temp_in = in.read();
			
			unsigned line_buffer_pointer = pointer + w;
			if (line_buffer_pointer >= line_buffer_size)
				line_buffer_pointer = line_buffer_pointer - line_buffer_size;
#ifdef SWU_DEBUG
			cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
			line_buffer[line_buffer_pointer] = temp_in;	
		}

		stride += 1;
		pointer += Din;
		if (pointer >= line_buffer_size) {
			pointer = pointer - line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;
	
			unsigned s = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*K; i++ ) {
#pragma HLS PIPELINE II=1
				
				ap_uint<K*Cin*Ibit> temp_out;

				for (unsigned p = 0; p < K; p++) {
#pragma HLS UNROLL
					unsigned read_address = (pointer+s*S) + y*Din + p;

					if (read_address >= line_buffer_size)
						read_address = read_address - line_buffer_size;
#ifdef SWU_DEBUG
					cout << "read_address: " << read_address << endl;
#endif
					temp_out( (p+1)*Cin*Ibit-1, p*Cin*Ibit ) = line_buffer[read_address];
				}
				out.write(temp_out);

				if (y == K-1) {
					y = 0;
					if (s == steps-1)
						s = 0;
					else
						s++;
				}
				else
					y++;
				
			}
		}
		
	}
}

template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Ibit,
			unsigned TopLeftPad, 
			unsigned BottomRightPad>
void SWU_NoP_residual(
	stream<ap_uint<Cin*Ibit> >& in, 
	stream<ap_uint<Cin*Ibit> >& out, 
	stream<ap_uint<Cin*Ibit> >& out_res, 
	const unsigned reps = 1) 
{
	static_assert( (Din-K)%S == 0, "(Din-K) mod S is not 0");
	static_assert( K >= S, "K is not >= than S");

	const unsigned steps = (Din-K)/S+1;
	const unsigned line_buffer_size = K*Din;
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P

	ap_uint<Cin*Ibit> temp_in;

	ap_uint<1> initial_fill = 0;
	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;
	unsigned h_res = 0;
	unsigned w_res = 0;

	for (unsigned rep = 0; rep < reps*Din; rep++) {

		if (h == Din) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0; w < Din; w++) {
#pragma HLS PIPELINE II=1
			temp_in = in.read();
			
			unsigned line_buffer_pointer = pointer + w;
			if (line_buffer_pointer >= line_buffer_size)
				line_buffer_pointer = line_buffer_pointer - line_buffer_size;
#ifdef SWU_DEBUG
			cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
			line_buffer[line_buffer_pointer] = temp_in;	
		}

		stride += 1;
		pointer += Din;
		if (pointer >= line_buffer_size) {
			pointer = pointer - line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;
	
			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K); i++ ) {
#pragma HLS PIPELINE II=1
				
				unsigned read_address = (pointer+s*S) + y*Din + x;

				if (read_address >= line_buffer_size)
					read_address = read_address - line_buffer_size;
#ifdef SWU_DEBUG
				cout << "read_address: " << read_address << endl;
#endif
				ap_uint<Cin*Ibit> temp_out = line_buffer[read_address];
				out.write(temp_out);

				if (x == K-1) {
					x = 0;
					if (y == K-1) {
						y = 0;
						if (s == steps-1)
							s = 0;
						else
							s++;
					}
					else
						y++;
				}
				else
					x++;
			}
		}

		if (initial_fill == 1 && pointer == 0) {
			for (unsigned i = 0; i < line_buffer_size; i++) {
#pragma HLS PIPELINE II=1

				ap_uint<Cin*Ibit> temp_out = line_buffer[i];
				if (h_res >= TopLeftPad && h_res < Din-BottomRightPad && w_res >= TopLeftPad && w_res < Din-BottomRightPad)
					out_res.write(temp_out);

				if (w_res == Din-1) {
					w_res = 0;
					if (h_res == Din-1)
						h_res = 0;
					else
						h_res++;
				}
				else
					w_res++;
			}
		}

	}
}

template <	unsigned K,
			unsigned MAX_Din,
			unsigned MAX_Cin,
			unsigned Ibit>
void SWU_NoP_variable(
	stream<ap_uint<MAX_Cin*Ibit> >& in, 
	stream<ap_uint<MAX_Cin*Ibit> >& out,
	const unsigned Din,
	const unsigned reps = 1) 
{
	const unsigned S = 1;

	const unsigned steps = (Din-K)+1;
	const unsigned current_line_buffer_size = K*Din;
	const unsigned line_buffer_size = K*MAX_Din;
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<MAX_Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P

	ap_uint<MAX_Cin*Ibit> temp_in;

	ap_uint<1> initial_fill = 0;
	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;

	for (unsigned rep = 0; rep < reps*Din; rep++) {

		if (h == Din) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0; w < Din; w++) {
#pragma HLS PIPELINE II=1
			temp_in = in.read();
			
			unsigned line_buffer_pointer = pointer + w;
			if (line_buffer_pointer >= current_line_buffer_size)
				line_buffer_pointer = line_buffer_pointer - current_line_buffer_size;
#ifdef SWU_DEBUG
			cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
			line_buffer[line_buffer_pointer] = temp_in;	
		}

		stride += 1;
		pointer += Din;
		if (pointer >= current_line_buffer_size) {
			pointer = pointer - current_line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < current_line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;
	
			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K); i++ ) {
#pragma HLS PIPELINE II=1
				
				unsigned read_address = (pointer+s*S) + y*Din + x;

				if (read_address >= current_line_buffer_size)
					read_address = read_address - current_line_buffer_size;
#ifdef SWU_DEBUG
				cout << "read_address: " << read_address << endl;
#endif
				ap_uint<MAX_Cin*Ibit> temp_out = line_buffer[read_address];
				out.write(temp_out);

				if (x == K-1) {
					x = 0;
					if (y == K-1) {
						y = 0;
						if (s == steps-1)
							s = 0;
						else
							s++;
					}
					else
						y++;
				}
				else
					x++;
			}


		}
	}
}

template <	unsigned K,
			unsigned MAX_Din_W,
			unsigned MAX_Din_H,
			unsigned MAX_Cin,
			unsigned Ibit>
void SWU_NoP_variable_RECT(
	stream<ap_uint<MAX_Cin*Ibit> >& in, 
	stream<ap_uint<MAX_Cin*Ibit> >& out,
	const unsigned Din_W,
	const unsigned Din_H,
	const unsigned reps = 1) 
{
	const unsigned S = 1;

	const unsigned steps = (Din_W-K)+1;
	const unsigned current_line_buffer_size = K*Din_W;
	const unsigned line_buffer_size = K*MAX_Din_W;
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<MAX_Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P

	ap_uint<MAX_Cin*Ibit> temp_in;

	ap_uint<1> initial_fill = 0;
	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;

	for (unsigned rep = 0; rep < reps*Din_H; rep++) {

		if (h == Din_H) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0; w < Din_W; w++) {
#pragma HLS PIPELINE II=1
			temp_in = in.read();
			
			unsigned line_buffer_pointer = pointer + w;
			if (line_buffer_pointer >= current_line_buffer_size)
				line_buffer_pointer = line_buffer_pointer - current_line_buffer_size;
#ifdef SWU_DEBUG
			cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
			line_buffer[line_buffer_pointer] = temp_in;	
		}

		stride += 1;
		pointer += Din_W;
		if (pointer >= current_line_buffer_size) {
			pointer = pointer - current_line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < current_line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;
	
			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K); i++ ) {
#pragma HLS PIPELINE II=1
				
				unsigned read_address = (pointer+s*S) + y*Din_W + x;

				if (read_address >= current_line_buffer_size)
					read_address = read_address - current_line_buffer_size;
#ifdef SWU_DEBUG
				cout << "read_address: " << read_address << endl;
#endif
				ap_uint<MAX_Cin*Ibit> temp_out = line_buffer[read_address];
				out.write(temp_out);

				if (x == K-1) {
					x = 0;
					if (y == K-1) {
						y = 0;
						if (s == steps-1)
							s = 0;
						else
							s++;
					}
					else
						y++;
				}
				else
					x++;
			}
		}
	}
}

template <	unsigned K,
			unsigned MAX_Din_W,
			unsigned MAX_Din_H,
			unsigned MAX_Cin,
			unsigned Ibit>
void SWU_NoP_variable_RECT_S2(
	stream<ap_uint<MAX_Cin*Ibit> >& in, 
	stream<ap_uint<MAX_Cin*Ibit> >& out,
	const unsigned Din_W,
	const unsigned Din_H,
	const unsigned reps = 1) 
{
	const unsigned S = 2;

	static_assert( K >= S, "K is not >= than S");

	const unsigned steps = ((Din_W-K)>>1)+1;
	const unsigned current_line_buffer_size = K*Din_W;
	const unsigned line_buffer_size = K*MAX_Din_W;
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<MAX_Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P

	ap_uint<MAX_Cin*Ibit> temp_in;

	ap_uint<1> initial_fill = 0;
	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;

	for (unsigned rep = 0; rep < reps*Din_H; rep++) {

		if (h == Din_H) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0; w < Din_W; w++) {
#pragma HLS PIPELINE II=1
			temp_in = in.read();
			
			unsigned line_buffer_pointer = pointer + w;
			if (line_buffer_pointer >= current_line_buffer_size)
				line_buffer_pointer = line_buffer_pointer - current_line_buffer_size;
#ifdef SWU_DEBUG
			cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
			line_buffer[line_buffer_pointer] = temp_in;	
		}

		stride += 1;
		pointer += Din_W;
		if (pointer >= current_line_buffer_size) {
			pointer = pointer - current_line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < current_line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;
	
			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K); i++ ) {
#pragma HLS PIPELINE II=1
				
				unsigned read_address = (pointer+s*S) + y*Din_W + x;

				if (read_address >= current_line_buffer_size)
					read_address = read_address - current_line_buffer_size;
#ifdef SWU_DEBUG
				cout << "read_address: " << read_address << endl;
#endif
				ap_uint<MAX_Cin*Ibit> temp_out = line_buffer[read_address];
				out.write(temp_out);

				if (x == K-1) {
					x = 0;
					if (y == K-1) {
						y = 0;
						if (s == steps-1)
							s = 0;
						else
							s++;
					}
					else
						y++;
				}
				else
					x++;
			}
		}
	}
}