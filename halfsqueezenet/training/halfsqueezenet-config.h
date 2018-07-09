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


// conv1/Conv2D
// Cycles per IFM: 451584.0
#define L0_K 3
#define L0_S 2
#define L0_Din 224
#define L0_Cin 3
#define L0_Cout 32
#define L0_Ibit 8
#define L0_Wbit 20
#define L0_Mbit 32
#define L0_Abit 5
#define L0_SWU_OutP 1
#define L0_MVTU_InP 3
#define L0_MVTU_OutP 8

// squeeze_conv_fire1/Conv2D
// Cycles per IFM: 100352.0
#define L1_K 1
#define L1_S 1
#define L1_Din 56
#define L1_Cin 32
#define L1_Cout 32
#define L1_Ibit 5
#define L1_Wbit 1
#define L1_Mbit 32
#define L1_Abit 5
#define L1_SWU_OutP 1
#define L1_MVTU_InP 8
#define L1_MVTU_OutP 4

// expand_3x3_conv_fire1/Conv2D
// Cycles per IFM: 338688.0
#define L2_K 3
#define L2_S 1
#define L2_Din 56
#define L2_Cin 32
#define L2_Cout 96
#define L2_Ibit 5
#define L2_Wbit 1
#define L2_Mbit 32
#define L2_Abit 5
#define L2_SWU_OutP 1
#define L2_MVTU_InP 32
#define L2_MVTU_OutP 8

// squeeze_conv_fire2/Conv2D
// Cycles per IFM: 75264.0
#define L3_K 1
#define L3_S 1
#define L3_Din 28
#define L3_Cin 96
#define L3_Cout 32
#define L3_Ibit 5
#define L3_Wbit 1
#define L3_Mbit 32
#define L3_Abit 5
#define L3_SWU_OutP 1
#define L3_MVTU_InP 8
#define L3_MVTU_OutP 4

// expand_3x3_conv_fire2/Conv2D
// Cycles per IFM: 84672.0
#define L4_K 3
#define L4_S 1
#define L4_Din 28
#define L4_Cin 32
#define L4_Cout 96
#define L4_Ibit 5
#define L4_Wbit 1
#define L4_Mbit 32
#define L4_Abit 5
#define L4_SWU_OutP 1
#define L4_MVTU_InP 32
#define L4_MVTU_OutP 8

// squeeze_conv_fire3/Conv2D
// Cycles per IFM: 18816.0
#define L5_K 1
#define L5_S 1
#define L5_Din 14
#define L5_Cin 96
#define L5_Cout 32
#define L5_Ibit 5
#define L5_Wbit 1
#define L5_Mbit 32
#define L5_Abit 5
#define L5_SWU_OutP 1
#define L5_MVTU_InP 8
#define L5_MVTU_OutP 4

// expand_3x3_conv_fire3/Conv2D
// Cycles per IFM: 21168.0
#define L6_K 3
#define L6_S 1
#define L6_Din 14
#define L6_Cin 32
#define L6_Cout 96
#define L6_Ibit 5
#define L6_Wbit 1
#define L6_Mbit 32
#define L6_Abit 5
#define L6_SWU_OutP 1
#define L6_MVTU_InP 32
#define L6_MVTU_OutP 8

// squeeze_conv_fire4/Conv2D
// Cycles per IFM: 18816.0
#define L7_K 1
#define L7_S 1
#define L7_Din 14
#define L7_Cin 96
#define L7_Cout 32
#define L7_Ibit 5
#define L7_Wbit 1
#define L7_Mbit 32
#define L7_Abit 5
#define L7_SWU_OutP 1
#define L7_MVTU_InP 8
#define L7_MVTU_OutP 4

// expand_3x3_conv_fire4/Conv2D
// Cycles per IFM: 21168.0
#define L8_K 3
#define L8_S 1
#define L8_Din 14
#define L8_Cin 32
#define L8_Cout 96
#define L8_Ibit 5
#define L8_Wbit 1
#define L8_Mbit 32
#define L8_Abit 5
#define L8_SWU_OutP 1
#define L8_MVTU_InP 32
#define L8_MVTU_OutP 8

// squeeze_conv_fire5/Conv2D
// Cycles per IFM: 18816.0
#define L9_K 1
#define L9_S 1
#define L9_Din 14
#define L9_Cin 96
#define L9_Cout 32
#define L9_Ibit 5
#define L9_Wbit 1
#define L9_Mbit 32
#define L9_Abit 5
#define L9_SWU_OutP 1
#define L9_MVTU_InP 8
#define L9_MVTU_OutP 4

// expand_3x3_conv_fire5/Conv2D
// Cycles per IFM: 21168.0
#define L10_K 3
#define L10_S 1
#define L10_Din 14
#define L10_Cin 32
#define L10_Cout 96
#define L10_Ibit 5
#define L10_Wbit 1
#define L10_Mbit 32
#define L10_Abit 5
#define L10_SWU_OutP 1
#define L10_MVTU_InP 32
#define L10_MVTU_OutP 8

// squeeze_conv_fire6/Conv2D
// Cycles per IFM: 18816.0
#define L11_K 1
#define L11_S 1
#define L11_Din 14
#define L11_Cin 96
#define L11_Cout 32
#define L11_Ibit 5
#define L11_Wbit 1
#define L11_Mbit 32
#define L11_Abit 5
#define L11_SWU_OutP 1
#define L11_MVTU_InP 8
#define L11_MVTU_OutP 4

// expand_3x3_conv_fire6/Conv2D
// Cycles per IFM: 21168.0
#define L12_K 3
#define L12_S 1
#define L12_Din 14
#define L12_Cin 32
#define L12_Cout 96
#define L12_Ibit 5
#define L12_Wbit 1
#define L12_Mbit 32
#define L12_Abit 5
#define L12_SWU_OutP 1
#define L12_MVTU_InP 32
#define L12_MVTU_OutP 8

// squeeze_conv_fire7/Conv2D
// Cycles per IFM: 18816.0
#define L13_K 1
#define L13_S 1
#define L13_Din 14
#define L13_Cin 96
#define L13_Cout 32
#define L13_Ibit 5
#define L13_Wbit 1
#define L13_Mbit 32
#define L13_Abit 5
#define L13_SWU_OutP 1
#define L13_MVTU_InP 8
#define L13_MVTU_OutP 4

// expand_3x3_conv_fire7/Conv2D
// Cycles per IFM: 21168.0
#define L14_K 3
#define L14_S 1
#define L14_Din 14
#define L14_Cin 32
#define L14_Cout 96
#define L14_Ibit 5
#define L14_Wbit 1
#define L14_Mbit 32
#define L14_Abit 5
#define L14_SWU_OutP 1
#define L14_MVTU_InP 32
#define L14_MVTU_OutP 8

// conv_class/Conv2D
// Cycles per IFM: 2352.0
#define L15_K 1
#define L15_S 1
#define L15_Din 14
#define L15_Cin 96
#define L15_Cout 12
#define L15_Ibit 5
#define L15_Wbit 1
#define L15_Mbit 32
#define L15_Abit 5
#define L15_SWU_OutP 1
#define L15_MVTU_InP 8
#define L15_MVTU_OutP 12

// conv_obj/Conv2D
// Cycles per IFM: 21168.0
#define L16_K 1
#define L16_S 1
#define L16_Din 14
#define L16_Cin 108
#define L16_Cout 1
#define L16_Ibit 5
#define L16_Wbit 24
#define L16_Mbit 32
#define L16_Abit 5
#define L16_SWU_OutP 1
#define L16_MVTU_InP 1
#define L16_MVTU_OutP 1

// conv_box/Conv2D
// Cycles per IFM: 21168.0
#define L17_K 1
#define L17_S 1
#define L17_Din 14
#define L17_Cin 108
#define L17_Cout 4
#define L17_Ibit 5
#define L17_Wbit 24
#define L17_Mbit 32
#define L17_Abit 5
#define L17_SWU_OutP 1
#define L17_MVTU_InP 1
#define L17_MVTU_OutP 4

// pool1/max_pooling2d/MaxPool
// Cycles per IFM: 34832.0
#define L18_K 3
#define L18_S 2
#define L18_Din 112
#define L18_Cin 32
#define L18_Ibit 5
#define L18_SWU_OutP 1

// pool2/max_pooling2d/MaxPool
// Cycles per IFM: 8792.0
#define L19_K 3
#define L19_S 2
#define L19_Din 56
#define L19_Cin 96
#define L19_Ibit 5
#define L19_SWU_OutP 1

// pool3/max_pooling2d/MaxPool
// Cycles per IFM: 2240.0
#define L20_K 3
#define L20_S 2
#define L20_Din 28
#define L20_Cin 96
#define L20_Ibit 5
#define L20_SWU_OutP 1

#define SCALE_BITS 18
#define FACTOR_SCALE_BITS 22

// #pragma HLS RESOURCE variable=weights0 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights0 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA0 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB0 complete dim=1
// #pragma HLS RESOURCE variable=weights1 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights1 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA1 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB1 complete dim=1
// #pragma HLS RESOURCE variable=weights2 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights2 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA2 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB2 complete dim=1
// #pragma HLS RESOURCE variable=weights3 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights3 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA3 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB3 complete dim=1
// #pragma HLS RESOURCE variable=weights4 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights4 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA4 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB4 complete dim=1
// #pragma HLS RESOURCE variable=weights5 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights5 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA5 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB5 complete dim=1
// #pragma HLS RESOURCE variable=weights6 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights6 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA6 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB6 complete dim=1
// #pragma HLS RESOURCE variable=weights7 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights7 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA7 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB7 complete dim=1
// #pragma HLS RESOURCE variable=weights8 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights8 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA8 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB8 complete dim=1
// #pragma HLS RESOURCE variable=weights9 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights9 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA9 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB9 complete dim=1
// #pragma HLS RESOURCE variable=weights10 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights10 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA10 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB10 complete dim=1
// #pragma HLS RESOURCE variable=weights11 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights11 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA11 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB11 complete dim=1
// #pragma HLS RESOURCE variable=weights12 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights12 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA12 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB12 complete dim=1
// #pragma HLS RESOURCE variable=weights13 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights13 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA13 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB13 complete dim=1
// #pragma HLS RESOURCE variable=weights14 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights14 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA14 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB14 complete dim=1
// #pragma HLS RESOURCE variable=weights15 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights15 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA15 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB15 complete dim=1
// #pragma HLS RESOURCE variable=weights16 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights16 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA16 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB16 complete dim=1
// #pragma HLS RESOURCE variable=weights17 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights17 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorA17 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=factorB17 complete dim=1


// stream<ap_uint<L0_Cout*L0_Abit> > conv0("conv0");
// CONV2D<L0_K, L0_S, L0_Din, L0_Cin, L0_Cout, L0_Ibit, L0_Wbit, L0_Mbit, L0_Abit, L0_SWU_OutP, L0_MVTU_InP, L0_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_in_stream, weights0, thresholds0, conv0, numReps);
// #else
// (in_stream, weights0, thresholds0, conv0, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L0_Cout*L0_Abit> > mon_conv0("mon_conv0");
// Monitor<L0_Din/L0_S, L0_Cout, L0_Abit, 1, 0>(conv0, mon_conv0, (char*)"log/mon_conv0.log", numReps);
// #endif

// stream<ap_uint<L1_Cout*L1_Abit> > conv1("conv1");
// CONV2D<L1_K, L1_S, L1_Din, L1_Cin, L1_Cout, L1_Ibit, L1_Wbit, L1_Mbit, L1_Abit, L1_SWU_OutP, L1_MVTU_InP, L1_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv0, weights1, thresholds1, conv1, numReps);
// #else
// (conv0, weights1, thresholds1, conv1, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L1_Cout*L1_Abit> > mon_conv1("mon_conv1");
// Monitor<L1_Din/L1_S, L1_Cout, L1_Abit, 1, 0>(conv1, mon_conv1, (char*)"log/mon_conv1.log", numReps);
// #endif

// stream<ap_uint<L2_Cout*L2_Abit> > conv2("conv2");
// CONV2D<L2_K, L2_S, L2_Din, L2_Cin, L2_Cout, L2_Ibit, L2_Wbit, L2_Mbit, L2_Abit, L2_SWU_OutP, L2_MVTU_InP, L2_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv1, weights2, thresholds2, conv2, numReps);
// #else
// (conv1, weights2, thresholds2, conv2, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L2_Cout*L2_Abit> > mon_conv2("mon_conv2");
// Monitor<L2_Din/L2_S, L2_Cout, L2_Abit, 1, 0>(conv2, mon_conv2, (char*)"log/mon_conv2.log", numReps);
// #endif

// stream<ap_uint<L3_Cout*L3_Abit> > conv3("conv3");
// CONV2D<L3_K, L3_S, L3_Din, L3_Cin, L3_Cout, L3_Ibit, L3_Wbit, L3_Mbit, L3_Abit, L3_SWU_OutP, L3_MVTU_InP, L3_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv2, weights3, thresholds3, conv3, numReps);
// #else
// (conv2, weights3, thresholds3, conv3, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L3_Cout*L3_Abit> > mon_conv3("mon_conv3");
// Monitor<L3_Din/L3_S, L3_Cout, L3_Abit, 1, 0>(conv3, mon_conv3, (char*)"log/mon_conv3.log", numReps);
// #endif

// stream<ap_uint<L4_Cout*L4_Abit> > conv4("conv4");
// CONV2D<L4_K, L4_S, L4_Din, L4_Cin, L4_Cout, L4_Ibit, L4_Wbit, L4_Mbit, L4_Abit, L4_SWU_OutP, L4_MVTU_InP, L4_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv3, weights4, thresholds4, conv4, numReps);
// #else
// (conv3, weights4, thresholds4, conv4, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L4_Cout*L4_Abit> > mon_conv4("mon_conv4");
// Monitor<L4_Din/L4_S, L4_Cout, L4_Abit, 1, 0>(conv4, mon_conv4, (char*)"log/mon_conv4.log", numReps);
// #endif

// stream<ap_uint<L5_Cout*L5_Abit> > conv5("conv5");
// CONV2D<L5_K, L5_S, L5_Din, L5_Cin, L5_Cout, L5_Ibit, L5_Wbit, L5_Mbit, L5_Abit, L5_SWU_OutP, L5_MVTU_InP, L5_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv4, weights5, thresholds5, conv5, numReps);
// #else
// (conv4, weights5, thresholds5, conv5, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L5_Cout*L5_Abit> > mon_conv5("mon_conv5");
// Monitor<L5_Din/L5_S, L5_Cout, L5_Abit, 1, 0>(conv5, mon_conv5, (char*)"log/mon_conv5.log", numReps);
// #endif

// stream<ap_uint<L6_Cout*L6_Abit> > conv6("conv6");
// CONV2D<L6_K, L6_S, L6_Din, L6_Cin, L6_Cout, L6_Ibit, L6_Wbit, L6_Mbit, L6_Abit, L6_SWU_OutP, L6_MVTU_InP, L6_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv5, weights6, thresholds6, conv6, numReps);
// #else
// (conv5, weights6, thresholds6, conv6, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L6_Cout*L6_Abit> > mon_conv6("mon_conv6");
// Monitor<L6_Din/L6_S, L6_Cout, L6_Abit, 1, 0>(conv6, mon_conv6, (char*)"log/mon_conv6.log", numReps);
// #endif

// stream<ap_uint<L7_Cout*L7_Abit> > conv7("conv7");
// CONV2D<L7_K, L7_S, L7_Din, L7_Cin, L7_Cout, L7_Ibit, L7_Wbit, L7_Mbit, L7_Abit, L7_SWU_OutP, L7_MVTU_InP, L7_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv6, weights7, thresholds7, conv7, numReps);
// #else
// (conv6, weights7, thresholds7, conv7, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L7_Cout*L7_Abit> > mon_conv7("mon_conv7");
// Monitor<L7_Din/L7_S, L7_Cout, L7_Abit, 1, 0>(conv7, mon_conv7, (char*)"log/mon_conv7.log", numReps);
// #endif

// stream<ap_uint<L8_Cout*L8_Abit> > conv8("conv8");
// CONV2D<L8_K, L8_S, L8_Din, L8_Cin, L8_Cout, L8_Ibit, L8_Wbit, L8_Mbit, L8_Abit, L8_SWU_OutP, L8_MVTU_InP, L8_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv7, weights8, thresholds8, conv8, numReps);
// #else
// (conv7, weights8, thresholds8, conv8, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L8_Cout*L8_Abit> > mon_conv8("mon_conv8");
// Monitor<L8_Din/L8_S, L8_Cout, L8_Abit, 1, 0>(conv8, mon_conv8, (char*)"log/mon_conv8.log", numReps);
// #endif

// stream<ap_uint<L9_Cout*L9_Abit> > conv9("conv9");
// CONV2D<L9_K, L9_S, L9_Din, L9_Cin, L9_Cout, L9_Ibit, L9_Wbit, L9_Mbit, L9_Abit, L9_SWU_OutP, L9_MVTU_InP, L9_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv8, weights9, thresholds9, conv9, numReps);
// #else
// (conv8, weights9, thresholds9, conv9, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L9_Cout*L9_Abit> > mon_conv9("mon_conv9");
// Monitor<L9_Din/L9_S, L9_Cout, L9_Abit, 1, 0>(conv9, mon_conv9, (char*)"log/mon_conv9.log", numReps);
// #endif

// stream<ap_uint<L10_Cout*L10_Abit> > conv10("conv10");
// CONV2D<L10_K, L10_S, L10_Din, L10_Cin, L10_Cout, L10_Ibit, L10_Wbit, L10_Mbit, L10_Abit, L10_SWU_OutP, L10_MVTU_InP, L10_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv9, weights10, thresholds10, conv10, numReps);
// #else
// (conv9, weights10, thresholds10, conv10, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L10_Cout*L10_Abit> > mon_conv10("mon_conv10");
// Monitor<L10_Din/L10_S, L10_Cout, L10_Abit, 1, 0>(conv10, mon_conv10, (char*)"log/mon_conv10.log", numReps);
// #endif

// stream<ap_uint<L11_Cout*L11_Abit> > conv11("conv11");
// CONV2D<L11_K, L11_S, L11_Din, L11_Cin, L11_Cout, L11_Ibit, L11_Wbit, L11_Mbit, L11_Abit, L11_SWU_OutP, L11_MVTU_InP, L11_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv10, weights11, thresholds11, conv11, numReps);
// #else
// (conv10, weights11, thresholds11, conv11, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L11_Cout*L11_Abit> > mon_conv11("mon_conv11");
// Monitor<L11_Din/L11_S, L11_Cout, L11_Abit, 1, 0>(conv11, mon_conv11, (char*)"log/mon_conv11.log", numReps);
// #endif

// stream<ap_uint<L12_Cout*L12_Abit> > conv12("conv12");
// CONV2D<L12_K, L12_S, L12_Din, L12_Cin, L12_Cout, L12_Ibit, L12_Wbit, L12_Mbit, L12_Abit, L12_SWU_OutP, L12_MVTU_InP, L12_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv11, weights12, thresholds12, conv12, numReps);
// #else
// (conv11, weights12, thresholds12, conv12, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L12_Cout*L12_Abit> > mon_conv12("mon_conv12");
// Monitor<L12_Din/L12_S, L12_Cout, L12_Abit, 1, 0>(conv12, mon_conv12, (char*)"log/mon_conv12.log", numReps);
// #endif

// stream<ap_uint<L13_Cout*L13_Abit> > conv13("conv13");
// CONV2D<L13_K, L13_S, L13_Din, L13_Cin, L13_Cout, L13_Ibit, L13_Wbit, L13_Mbit, L13_Abit, L13_SWU_OutP, L13_MVTU_InP, L13_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv12, weights13, thresholds13, conv13, numReps);
// #else
// (conv12, weights13, thresholds13, conv13, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L13_Cout*L13_Abit> > mon_conv13("mon_conv13");
// Monitor<L13_Din/L13_S, L13_Cout, L13_Abit, 1, 0>(conv13, mon_conv13, (char*)"log/mon_conv13.log", numReps);
// #endif

// stream<ap_uint<L14_Cout*L14_Abit> > conv14("conv14");
// CONV2D<L14_K, L14_S, L14_Din, L14_Cin, L14_Cout, L14_Ibit, L14_Wbit, L14_Mbit, L14_Abit, L14_SWU_OutP, L14_MVTU_InP, L14_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv13, weights14, thresholds14, conv14, numReps);
// #else
// (conv13, weights14, thresholds14, conv14, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L14_Cout*L14_Abit> > mon_conv14("mon_conv14");
// Monitor<L14_Din/L14_S, L14_Cout, L14_Abit, 1, 0>(conv14, mon_conv14, (char*)"log/mon_conv14.log", numReps);
// #endif

// stream<ap_uint<L15_Cout*L15_Abit> > conv15("conv15");
// CONV2D<L15_K, L15_S, L15_Din, L15_Cin, L15_Cout, L15_Ibit, L15_Wbit, L15_Mbit, L15_Abit, L15_SWU_OutP, L15_MVTU_InP, L15_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv14, weights15, thresholds15, conv15, numReps);
// #else
// (conv14, weights15, thresholds15, conv15, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L15_Cout*L15_Abit> > mon_conv15("mon_conv15");
// Monitor<L15_Din/L15_S, L15_Cout, L15_Abit, 1, 0>(conv15, mon_conv15, (char*)"log/mon_conv15.log", numReps);
// #endif

// stream<ap_uint<L16_Cout*L16_Abit> > conv16("conv16");
// CONV2D<L16_K, L16_S, L16_Din, L16_Cin, L16_Cout, L16_Ibit, L16_Wbit, L16_Mbit, L16_Abit, L16_SWU_OutP, L16_MVTU_InP, L16_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv15, weights16, thresholds16, conv16, numReps);
// #else
// (conv15, weights16, thresholds16, conv16, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L16_Cout*L16_Abit> > mon_conv16("mon_conv16");
// Monitor<L16_Din/L16_S, L16_Cout, L16_Abit, 1, 0>(conv16, mon_conv16, (char*)"log/mon_conv16.log", numReps);
// #endif

// stream<ap_uint<L17_Cout*L17_Abit> > conv17("conv17");
// CONV2D<L17_K, L17_S, L17_Din, L17_Cin, L17_Cout, L17_Ibit, L17_Wbit, L17_Mbit, L17_Abit, L17_SWU_OutP, L17_MVTU_InP, L17_MVTU_OutP, SCALE_BITS>
// #ifdef DEBUG
// (mon_conv16, weights17, thresholds17, conv17, numReps);
// #else
// (conv16, weights17, thresholds17, conv17, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L17_Cout*L17_Abit> > mon_conv17("mon_conv17");
// Monitor<L17_Din/L17_S, L17_Cout, L17_Abit, 1, 0>(conv17, mon_conv17, (char*)"log/mon_conv17.log", numReps);
// #endif

// stream<ap_uint<L18_Cin*L18_Ibit> > pool18("pool18");
// POOL2D<L18_K, L18_S, L18_Din, L18_Cin, L18_Ibit, L18_SWU_OutP>
// #ifdef DEBUG
// (mon_conv17, pool18, numReps);
// #else
// (conv17, pool18, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L18_Cin*L18_Ibit> > mon_pool18("mon_pool18");
// Monitor<L18_Din/L18_S, L18_Cin, L18_Ibit, 1, 0>(pool18, mon_pool18, (char*)"log/mon_pool18.log", numReps);
// #endif

// stream<ap_uint<L19_Cin*L19_Ibit> > pool19("pool19");
// POOL2D<L19_K, L19_S, L19_Din, L19_Cin, L19_Ibit, L19_SWU_OutP>
// #ifdef DEBUG
// (mon_pool18, pool19, numReps);
// #else
// (pool18, pool19, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L19_Cin*L19_Ibit> > mon_pool19("mon_pool19");
// Monitor<L19_Din/L19_S, L19_Cin, L19_Ibit, 1, 0>(pool19, mon_pool19, (char*)"log/mon_pool19.log", numReps);
// #endif

// stream<ap_uint<L20_Cin*L20_Ibit> > pool20("pool20");
// POOL2D<L20_K, L20_S, L20_Din, L20_Cin, L20_Ibit, L20_SWU_OutP>
// #ifdef DEBUG
// (mon_pool19, pool20, numReps);
// #else
// (pool19, pool20, numReps);
// #endif
// #ifdef DEBUG
// stream<ap_uint<L20_Cin*L20_Ibit> > mon_pool20("mon_pool20");
// Monitor<L20_Din/L20_S, L20_Cin, L20_Ibit, 1, 0>(pool20, mon_pool20, (char*)"log/mon_pool20.log", numReps);
// #endif
