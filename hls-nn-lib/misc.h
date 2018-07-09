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
#include <fstream>
#include <string>
using namespace std;
#include <assert.h>

// #define MISC_DEBUG

struct ap_axis{
	ap_uint<512> data;
	ap_uint<1> last;
	ap_uint<64> keep;
};

template <	unsigned NumLines>
void AddLast(
	stream<ap_uint<512> >& in,
	stream<ap_axis >& out,
	const unsigned reps = 1)
{
	ap_axis temp;
	temp.keep = "0xffffffffffffffff";

	for (unsigned i = 0; i < reps*NumLines-1; i++) {
		temp.data = in.read();
		temp.last = 0;
		out.write(temp);
	}

	temp.data = in.read();
	temp.last = 1;
	out.write(temp);
}

template <	unsigned LineWidth,
			unsigned NumLines>
void Mem2Stream(
	ap_uint<LineWidth> * in,
	stream<ap_uint<LineWidth> >& out,
	const unsigned reps = 1)
{
	for (unsigned i = 0; i < reps*NumLines; i++) {
		out.write(in[i]);
	}
}

template <	unsigned LineWidth,
			unsigned NumLines>
void Stream2Mem(
	stream<ap_uint<LineWidth> >& in,
	ap_uint<LineWidth> * out,
	const unsigned reps = 1)
{
	for (unsigned i = 0; i < reps*NumLines; i++) {
		out[i] = in.read();
	}
}

template <	unsigned StreamW,
			unsigned NumLines>
void StreamCopy(
	stream<ap_uint<StreamW> > & in, 
	stream<ap_uint<StreamW> >& out, 
	const unsigned reps = 1)
{
	ap_uint<StreamW> temp;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
		temp = in.read();
		out.write( temp );
	}
}

template <	unsigned OutStreamW,
			unsigned NumLines>
void ExtractPixels(
	stream<ap_axis > & in, 
	stream<ap_uint<OutStreamW> >& out, 
	const unsigned reps = 1)
{
	ap_axis temp;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
		temp = in.read();
		out.write( temp.data(OutStreamW-1, 0) );
	}
}

template <	unsigned InStreamW,
			unsigned OutStreamW,
			unsigned NumLines>
void AppendZeros(
	stream<ap_uint<InStreamW> >& in, 
	stream<ap_uint<OutStreamW> >& out, 
	const unsigned reps = 1) 
{
	static_assert( InStreamW < OutStreamW, "For AppendZeros in stream is wider than out stream." );

	ap_uint<OutStreamW> buffer;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
		buffer(OutStreamW-1, InStreamW) = 0;
		buffer(InStreamW-1, 0) = in.read();
		out.write(buffer);
	}
}

template <	unsigned InStreamW,
			unsigned OutStreamW,
			unsigned NumLines>
void ReduceWidth(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > & out,
	const unsigned reps = 1)
{
	static_assert( InStreamW%OutStreamW == 0, "For ReduceWidth, InStreamW mod OutStreamW is not 0" );

	const unsigned parts = InStreamW/OutStreamW;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=InStreamW/OutStreamW

		ap_uint<InStreamW> temp_in = in.read();
		for (unsigned p = 0; p < parts; p++) {

			ap_uint<OutStreamW> temp_out = temp_in(OutStreamW-1, 0);
			out.write( temp_out );
			temp_in = temp_in >> OutStreamW;
		}
	}
}

template <	unsigned InStreamW,
			unsigned OutStreamW,
			unsigned NumLines>
void ReduceWidth_variable(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > & out,
	const unsigned usefulInStreamW,
	const unsigned reps = 1)
{
	static_assert( InStreamW%OutStreamW == 0, "For ReduceWidth, InStreamW mod OutStreamW is not 0" );

	const unsigned parts = usefulInStreamW/OutStreamW;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=InStreamW/OutStreamW

		ap_uint<InStreamW> temp_in = in.read();
		for (unsigned p = 0; p < parts; p++) {

			ap_uint<OutStreamW> temp_out = temp_in(OutStreamW-1, 0);
			out.write( temp_out );
			temp_in = temp_in >> OutStreamW;
		}
	}
}

template <	unsigned InStreamW,
			unsigned OutStreamW,
			unsigned NumLines>
void ExpandWidth(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > & out,
	const unsigned reps = 1)
{
	static_assert( OutStreamW%InStreamW == 0, "For ExpandWidth, OutStreamW mod InStreamW is not 0" );

	const unsigned parts = OutStreamW/InStreamW;
	ap_uint<OutStreamW> buffer;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
		
		for (unsigned p = 0; p < parts; p++) {
#pragma HLS PIPELINE II=1
			ap_uint<InStreamW> temp = in.read();
			buffer( (p+1)*InStreamW-1, p*InStreamW ) = temp;
		}
		out.write(buffer);
		
	}
}

template <	unsigned InStreamW,
			unsigned OutStreamW,
			unsigned NumLines>
void ExpandWidth_variable(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > & out,
	const unsigned usefulOutStreamW,
	const unsigned reps = 1)
{
	static_assert( OutStreamW%InStreamW == 0, "For ExpandWidth, OutStreamW mod InStreamW is not 0" );

	const unsigned parts = usefulOutStreamW/InStreamW;
	ap_uint<OutStreamW> buffer;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
		buffer = 0;
		for (unsigned p = 0; p < parts; p++) {
#pragma HLS PIPELINE II=1
			ap_uint<InStreamW> temp = in.read();
			buffer( (p+1)*InStreamW-1, p*InStreamW ) = temp;
		}
		out.write(buffer);
		
	}
}

template <	unsigned InStreamW,
			unsigned NumLines>
void DuplicateStreams(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<InStreamW> > & out1,
	stream<ap_uint<InStreamW> > & out2,
	const unsigned reps = 1)
{
	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=1
		ap_uint<InStreamW> temp = in.read();
		out1.write(temp);
		out2.write(temp);	
	}
}

template <	unsigned InStreamW,
			unsigned OutStreamW2,
			unsigned NumLines>
void DuplicateStreams_ReduceWidth(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<InStreamW> > & out1,
	stream<ap_uint<OutStreamW2> > & out2,
	const unsigned reps = 1)
{
	const unsigned parts = InStreamW/OutStreamW2;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=InStreamW/OutStreamW2

		ap_uint<InStreamW> temp = in.read();
		out1.write(temp);
		
		for (unsigned p = 0; p < parts; p++) {
			ap_uint<OutStreamW2> temp_out = temp(OutStreamW2-1, 0);
			out2.write( temp_out );
			temp = temp >> OutStreamW2;
		}
	}
}

template <	unsigned InStreamW1,
			unsigned InStreamW2,
			unsigned NumLines>
void ConcatStreams(
	stream<ap_uint<InStreamW1> > & in1,
	stream<ap_uint<InStreamW2> > & in2,
	stream<ap_uint<InStreamW2+InStreamW1> > & out,
	const unsigned reps = 1)
{
	ap_uint<InStreamW2+InStreamW1> temp_out;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=1
		ap_uint<InStreamW1> temp1 = in1.read();
		ap_uint<InStreamW2> temp2 = in2.read();
		
		temp_out(InStreamW1-1, 0) = temp1;
		temp_out(InStreamW2+InStreamW1-1, InStreamW1) = temp2;

		out.write(temp_out);
	}
}

template <	unsigned InStreamW1,
			unsigned InStreamW2,
			unsigned NumLines>
void ConcatStreams_variable(
	stream<ap_uint<InStreamW1> > & in1,
	stream<ap_uint<InStreamW2> > & in2,
	stream<ap_uint<InStreamW2+InStreamW1> > & out,
	const unsigned usefulInStreamW1,
	const unsigned usefulInStreamW2,
	const unsigned reps = 1)
{
	ap_uint<InStreamW2+InStreamW1> temp_out = 0;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=1
		temp_out = 0;

		ap_uint<InStreamW1> temp1 = in1.read();
		ap_uint<InStreamW2> temp2 = in2.read();
		
		temp_out(usefulInStreamW1-1, 0) = temp1(usefulInStreamW1-1, 0);
		temp_out(usefulInStreamW2+usefulInStreamW1-1, usefulInStreamW1) = temp2(usefulInStreamW2-1, 0);

		out.write(temp_out);
	}
}


template <	unsigned InStreamW_obj,
			unsigned InStreamW_box,
			unsigned NumLines>
void ObjDetectSelect(
	stream<ap_uint<InStreamW_obj> > & in_obj,
	stream<ap_uint<InStreamW_box> > & in_box,
	stream<ap_uint<8+InStreamW_box> > & out,
	const unsigned reps = 1)
{
	ap_int<InStreamW_obj> temp_max = 0;
	ap_uint<8> max_index = 0;
	ap_uint<8+InStreamW_box> outbuf = 0;

	for (unsigned rep = 0; rep < reps; rep++) {

		temp_max = (1 << (InStreamW_obj-1));
		max_index = 0;
		for (ap_uint<8> i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
			ap_int<InStreamW_obj> temp_obj = in_obj.read();
			if (temp_obj > temp_max) {
				temp_max = temp_obj;
				max_index = i;
			}
		}
		outbuf(7+InStreamW_box, InStreamW_box) = max_index;
		for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
			ap_uint<InStreamW_box> temp_box = in_box.read();
			outbuf(InStreamW_box-1, 0) = temp_box;
			if (i == max_index)
				out.write(outbuf);
		}
	}
}


template <	unsigned StreamW,
			unsigned NumLines>
void AddStreams(
	stream<ap_uint<StreamW> > & in1,
	stream<ap_uint<StreamW> > & in2,
	stream<ap_uint<StreamW> > & out,
	const unsigned reps = 1)
{
	ap_uint<StreamW> temp_out;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=1
		ap_uint<StreamW> temp1 = in1.read();
		ap_uint<StreamW> temp2 = in2.read();
		
		temp_out = temp1 + temp2;

		out.write(temp_out);
	}
}

template <	unsigned StreamW,
			unsigned InStreamW2,
			unsigned NumLines>
void AddStreams_ExpandWidth(
	stream<ap_uint<StreamW> > & in1,
	stream<ap_uint<InStreamW2> > & in2,
	stream<ap_uint<StreamW> > & out,
	const unsigned reps = 1)
{
	const unsigned parts = StreamW/InStreamW2;
	ap_uint<StreamW> buffer;
	ap_uint<StreamW> temp_out;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=StreamW/InStreamW2
		ap_uint<StreamW> temp1 = in1.read();

		for (unsigned p = 0; p < parts; p++) {
			ap_uint<InStreamW2> temp = in2.read();
			buffer( (p+1)*InStreamW2-1, p*InStreamW2 ) = temp;
		}

		temp_out = temp1 + buffer;

		out.write(temp_out);
	}
}

template <	unsigned InStreamW,
			unsigned NumLines>
void DoubleOneStream(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<InStreamW> > & out,
	const unsigned reps = 1)
{
	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=2
		ap_uint<InStreamW> temp = in.read();
		out.write(temp);
		out.write(temp);	
	}
}

template <	unsigned StreamW,
			unsigned NumLines>
void AggregateOneStream(
	stream<ap_uint<StreamW> > & in,
	stream<ap_uint<StreamW> > & out,
	const unsigned reps = 1)
{
	ap_uint<StreamW> temp_out;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=2
		ap_uint<StreamW> temp1 = in.read();
		ap_uint<StreamW> temp2 = in.read();
		
		temp_out = temp1 + temp2;

		out.write(temp_out);
	}
}

#ifdef DEBUG // From the network_main.cpp file

string hexFromInt(int value, unsigned precision) {
	unsigned hex_digits = precision/4;
	if (precision%4 > 0)
		hex_digits += 1;

	if (value < 0)
		value = (1 << precision) + value;

	string result = "";
	for (unsigned d = 0; d < hex_digits; d++) {
		unsigned temp = value & 0xF;
		value = value >> 4;
		stringstream ss;
		ss << hex << temp;
		result = ss.str() + result;
	}
	
	return result;
}
template <	unsigned Din,
			unsigned Cin,
			unsigned Ibit>
void Monitor(
	stream<ap_uint<Cin*Ibit> > & in,
	char* filename,
	unsigned reps = 1)
{
	ofstream fileout(filename);

	for (unsigned rep = 0; rep < reps; rep++) {
#ifdef MISC_DEBUG
		cout << "-----------------------------------" << endl;
#endif
		for (unsigned h = 0; h < Din; h++) {
			for (unsigned w = 0; w < Din; w++) {

				ap_uint<Cin*Ibit> temp = in.read();
				in.write(temp);

				string line = "";
				for (unsigned c = 0; c < Cin; c++) {
					line = hexFromInt( temp( (c+1)*Ibit - 1, c*Ibit ), Ibit ) + "_" + line;
#ifdef MISC_DEBUG
					cout << temp( (c+1)*Ibit - 1, c*Ibit ) << " ";
#endif
				}
				fileout << "0x" << line << endl;
			}
#ifdef MISC_DEBUG
			cout << endl;
#endif
		}
	}
}

#endif

template <	unsigned TopLeftPad,
			unsigned BottomRightPad,
			unsigned Din,
			unsigned Cin,
			unsigned Ibit>
void SAMEPAD(
	stream<ap_uint<Cin*Ibit> >& in,
	stream<ap_uint<Cin*Ibit> >& out,
	const unsigned reps = 1)
{
	const unsigned Dout = (Din+TopLeftPad+BottomRightPad);
	ap_uint<Cin*Ibit> temp_out = 0;

	for (unsigned rep = 0; rep < reps; rep++) {

		for (unsigned h = 0; h < TopLeftPad; h++) {
			for (unsigned s = 0; s < Dout; s++) {
				out.write(0);
			}
		}

		for (unsigned h = 0; h < Din; h++) {

			for ( unsigned s = 0; s < Dout; s++ ) {
#pragma HLS PIPELINE II=1

				if ( (s < TopLeftPad) || (s >= Dout-BottomRightPad) ) {
					temp_out = 0;
				}
				else {
					temp_out = in.read();
				}
				
				out.write(temp_out);
			}
		}

		for (unsigned h = 0; h < BottomRightPad; h++) {
			for (unsigned i = 0; i < Dout; i++) {
				out.write(0);
			}
		}

	}
}

template <	unsigned MAX_Cin,
			unsigned Ibit>
void SAMEPAD_variable(
	stream<ap_uint<MAX_Cin*Ibit> >& in,
	stream<ap_uint<MAX_Cin*Ibit> >& out,
	const unsigned TopLeftPad,
	const unsigned BottomRightPad,
	const unsigned Din,
	const unsigned reps = 1)
{
	const unsigned Dout = (Din+TopLeftPad+BottomRightPad);
	ap_uint<MAX_Cin*Ibit> temp_out = 0;

	for (unsigned rep = 0; rep < reps; rep++) {

		for (unsigned h = 0; h < TopLeftPad; h++) {
			for (unsigned s = 0; s < Dout; s++) {
				out.write(0);
			}
		}

		for (unsigned h = 0; h < Din; h++) {

			for ( unsigned s = 0; s < Dout; s++ ) {
#pragma HLS PIPELINE II=1

				if ( (s < TopLeftPad) || (s >= Dout-BottomRightPad) ) {
					temp_out = 0;
				}
				else {
					temp_out = in.read();
				}
				
				out.write(temp_out);
			}
		}

		for (unsigned h = 0; h < BottomRightPad; h++) {
			for (unsigned i = 0; i < Dout; i++) {
				out.write(0);
			}
		}

	}
}