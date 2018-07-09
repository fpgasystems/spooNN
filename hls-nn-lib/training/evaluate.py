#*************************************************************************
# Copyright (C) 2018 Kaan Kara - Systems Group, ETH Zurich

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#*************************************************************************

import sys
import math
import argparse
import imp
import json
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from tensorpack import TowerContext, logger
from tensorpack.tfutils import varmanip, get_model_loader

VERBOSITY = 1

"""number of activation bits to decide between thresholds and factors"""
THRESHOLD_TO_FACTOR_TRANSITION = 4
"""Maximum input parallelism in matrix vector multiplication (dot product parallelism)"""
MAX_INP = 8
"""Maximum output parallelism in matrix vector multiplication (number of dot product units)"""
MAX_OUTP = 8
"""Number of bits for multiply accumulate registers"""
MAC_BITS = 32
"""Number of bits for full precision weights"""
FIXED_BITS = 20
"""full precision weight scale factor"""
SCALE_BITS = 18
"""full precision weight scale factor with high precision"""
HIGH_PREC_SCALE_BITS = SCALE_BITS+4
"""factor scale factor"""
FACTOR_SCALE_BITS = 22

def formatWeights(basemem, Wbit, InP, OutP, MatrixW, MatrixH, weights):
	Wtype = "const ap_uint<"+str(InP)+"*"+str(Wbit)+"> "
	Wname = "weights"+str(basemem)+"["+str(OutP)+"]["+str(int(MatrixH/InP*MatrixW/OutP))+"] = {\n"
	values = ''
	for i in range(0, len(weights)):
		values += '{' + ', '.join(weights[i]) + '},\n'

	closing = "\n};\n"

	string = Wtype + Wname + values[0:len(values)-2] + closing
	return string

def formatThresholds(basemem, Mbit, NumThresholds, OutP, MatrixW, thresholds):
	Ttype = "const ap_uint<"+str(NumThresholds)+"*"+str(Mbit)+"> "
	Tname = "thresholds"+str(basemem)+"["+str(OutP)+"]["+str(int(MatrixW/OutP))+"] = {\n"
	values = ''
	for i in range(0, len(thresholds)):
		values += '{' + ', '.join(thresholds[i]) + '},\n'

	closing = "\n};\n"

	string = Ttype + Tname + values[0:len(values)-2] + closing
	return string

def formatFactors(name, basemem, Mbit, OutP, MatrixW, factors):
	Ttype = "const ap_int<"+str(Mbit)+"> "
	Tname = str(name)+str(basemem)+"["+str(OutP)+"]["+str(int(MatrixW/OutP))+"] = {\n"
	values = ''
	for i in range(0, len(factors)):
		values += '{' + ', '.join(factors[i]) + '},\n'

	closing = "\n};\n"

	string = Ttype + Tname + values[0:len(values)-2] + closing
	return string

def intFromBitstring(bitstring, precision):
	res = 0
	for i in range(0, len(bitstring)):
		res += int(bitstring[i]) << i

	return res

def hexFromInt(value, precision):
	hex_digits = int(precision/4)
	if precision%4 > 0:
		hex_digits += 1

	if value < 0:
		value = 2**precision + value

	result = ''
	for d in range(0, hex_digits):
		temp = value & 0xF
		value = value >> 4
		result = hex(temp)[2:3] + result

	return result


def computeWeights(W, Wbit, InP, OutP):
	if VERBOSITY == 2:
		print('W.shape: ' + str(W.shape))
		print('W: ' + str(W))

	Wmax = np.amax(W)
	W = np.divide(W, Wmax)

	if VERBOSITY == 2:
		print('Wmax :' + str(Wmax) )
		print('W: ' + str(W))

	if Wbit == 1:
		ones = np.ones(W.shape)
		tempW = np.zeros(W.shape)
		tempW[ np.where( W < 0 ) ] = ones[ np.where( W < 0 ) ]
		W = tempW.astype("int32")
	else:
		if Wbit > FIXED_BITS:
			W = np.multiply(W, 2**HIGH_PREC_SCALE_BITS)
		else:
			W = np.multiply(W, 2**SCALE_BITS)
		W = np.round(W).astype("int32")

		for x in W.flatten():
			if x > 2**(Wbit-1) or x < -2**(Wbit-1):
				print("OVERFLOW in weights!")
				sys.exit()

	if VERBOSITY == 2:
		print('W.shape: ' + str(W.shape))
		print('W: ' + str(W))

	if len(W.shape) == 4:
		W = np.transpose(W, axes=(3, 0, 1, 2))
		W = W.reshape((W.shape[0], W.shape[1]*W.shape[2]*W.shape[3]))

	if VERBOSITY == 2:
		print('W.shape: ' + str(W.shape))
		print('W: ' + str(W))

	MatrixW = W.shape[0]
	MatrixH = W.shape[1]

	M = W.reshape((MatrixW, int(MatrixH/InP), InP))
	if VERBOSITY == 2:
		print('MatrixW: ' + str(MatrixW))
		print('OutP: ' + str(OutP))
		print('MatrixH: ' + str(MatrixH))
		print('InP: ' + str(InP))
		print('M.shape: ' + str(M.shape))
		print('M: ' + str(M))

	res = [['' for j in range(int(MatrixW/OutP*MatrixH/InP)) ] for i in range(OutP)]
	for wVec in range(0, int(MatrixH/InP)):
		for wMat in range(0, int(MatrixW/OutP)):
			for outp in range (0, int(OutP)):

				if Wbit == 1:
					res[outp][wVec*int(MatrixW/OutP)+wMat] = '\"0x' + hexFromInt(intFromBitstring(M[wMat*OutP+outp, wVec], InP), InP) + '\"';
				else:
					for p in range(0, InP):
						res[outp][wVec*int(MatrixW/OutP)+wMat] = hexFromInt(M[wMat*OutP+outp, wVec, p], Wbit) + res[outp][wVec*int(MatrixW/OutP)+wMat]
					res[outp][wVec*int(MatrixW/OutP)+wMat] = '\"0x' + res[outp][wVec*int(MatrixW/OutP)+wMat] + '\"';

	if VERBOSITY == 2:
		print('res ' + str(res) + ' len(res): ' + str(len(res)))

	return res, Wmax

def computeThresholds(MatrixW, OutP, Wmax, bn_beta, bn_gamma, bn_mean, bn_variance, Mbit, Abit):
	steps = 2**Abit - 1

	if VERBOSITY == 2:
		print('Wmax: ' + str(Wmax))
		print('bn_beta: ' + str(bn_beta))
		print('bn_gamma: ' + str(bn_gamma))
		print('bn_mean: ' + str(bn_mean))
		print('bn_variance: ' + str(bn_variance))
		print('steps: ' + str(steps))

	thresholds = np.linspace(0, steps-1, steps)
	thresholds += 0.5
	
	if VERBOSITY == 2:
		print('thresholds: ' + str(thresholds))
	
	temp = np.zeros((MatrixW, steps))
	for s in range(0, steps):
		if bn_beta[0] == 0:
			scaled_thresholds = np.ones(MatrixW)
			scaled_thresholds = np.multiply(scaled_thresholds, thresholds[s])
			scaled_thresholds = np.divide(scaled_thresholds, Wmax)

			scaled_thresholds = np.multiply(scaled_thresholds, 2**SCALE_BITS )

			if VERBOSITY == 2:
				print('scaled_thresholds: ' + str(scaled_thresholds))

			for x in scaled_thresholds:
				if x > 2**(Mbit-1) or x < -2**(Mbit-1):
					print("OVERFLOW in thresholds!")
					print("Value: " + str(x))
					sys.exit()

		else:
			scaled_thresholds = np.subtract( thresholds[s], bn_beta )
			scaled_thresholds = np.multiply( scaled_thresholds, np.sqrt(bn_variance) )
			scaled_thresholds = np.divide( scaled_thresholds, bn_gamma )
			scaled_thresholds = np.add( scaled_thresholds, bn_mean )
			scaled_thresholds = np.divide( scaled_thresholds, Wmax )

			scaled_thresholds = np.multiply(scaled_thresholds, 2**SCALE_BITS )

			if VERBOSITY == 2:
				print('scaled_thresholds: ' + str(scaled_thresholds))

			for x in scaled_thresholds:
				if x > 2**(Mbit-1) or x < -2**(Mbit-1):
					print("OVERFLOW in thresholds!")
					print("Value: " + str(x))
					sys.exit()

		if len(scaled_thresholds) != MatrixW:
			print('Dimension of thresh is not equal to MatrixW!')
			sys.exit()

		temp[:,s] = scaled_thresholds

	thresholds = temp

	if VERBOSITY == 2:
		print('thresholds: ' + str(thresholds))

	index = 0
	hex_thresholds = [['' for j in range(int(MatrixW/OutP)) ] for i in range(OutP)]
	for j in range(0, int(MatrixW/OutP)):
		for i in range(0, OutP):	
			for s in range(0, steps):
				hex_thresholds[i][j] = hexFromInt(int(thresholds[index, s]), Mbit) + hex_thresholds[i][j]
			index += 1
			hex_thresholds[i][j] = '\"0x' + hex_thresholds[i][j] + '\"';

	if VERBOSITY == 2:
		print('hex_thresholds: ' + str(hex_thresholds) )

	return hex_thresholds

def computeFactors(MatrixW, OutP, Wmax, bn_beta, bn_gamma, bn_mean, bn_variance, Mbit):
	if VERBOSITY == 2:
		print('MatrixW: ' + str(MatrixW))
		print('Wmax: ' + str(Wmax))
		print('bn_beta: ' + str(bn_beta.shape) + '\n' + str(bn_beta))
		print('bn_gamma: ' + str(bn_gamma.shape) + '\n' + str(bn_gamma))
		print('bn_mean: ' + str(bn_mean.shape) + '\n' + str(bn_mean))
		print('bn_variance: ' + str(bn_variance.shape) + '\n' + str(bn_variance))

	if bn_beta[0] == 0:
		A = np.round(np.multiply( Wmax, 2**FACTOR_SCALE_BITS )).astype('int32')
		B = 0

		if A > 2**(Mbit-1) or A < -2**(Mbit-1):
			print("OVERFLOW in factorA!")
			sys.exit()
		
		hex_A = [['' for j in range(int(MatrixW/OutP)) ] for i in range(OutP)]
		for j in range(0, int(MatrixW/OutP)):
			for i in range(0, OutP):
				hex_A[i][j] = hexFromInt(A, Mbit) + hex_A[i][j]
				hex_A[i][j] = '\"0x' + hex_A[i][j] + '\"';

		hex_B = [['' for j in range(int(MatrixW/OutP)) ] for i in range(OutP)]
		for j in range(0, int(MatrixW/OutP)):
			for i in range(0, OutP):	
				hex_B[i][j] = hexFromInt(B, Mbit) + hex_B[i][j]
				hex_B[i][j] = '\"0x' + hex_B[i][j] + '\"';

	else:
		A = np.divide( np.multiply(Wmax, bn_gamma), np.sqrt(bn_variance) )
		B = np.subtract( bn_beta, np.divide( np.multiply(bn_mean, bn_gamma), np.sqrt(bn_variance) ) )

		if VERBOSITY == 2:
			print('A: ' + str(A.shape) + '\n' + str(A))
			print('B: ' + str(B.shape) + '\n' + str(B))
			sys.exit()

		A = np.round(np.multiply( A, 2**FACTOR_SCALE_BITS )).astype('int32')
		B = np.round(np.multiply( B, 2**FACTOR_SCALE_BITS )).astype('int32')

		if VERBOSITY == 2:
			print('Scaled A: ' + str(A.shape) + '\n' + str(A))
			print('Scaled B: ' + str(B.shape) + '\n' + str(B))

		for a in A:
			if a > 2**(Mbit-1) or a < -2**(Mbit-1):
				print("OVERFLOW in factorA!")
				sys.exit()
		for b in B:
			if b > 2**(Mbit-1) or b < -2**(Mbit-1):
				print("OVERFLOW in factorB!")
				sys.exit()

		index = 0
		hex_A = [['' for j in range(int(MatrixW/OutP)) ] for i in range(OutP)]
		for j in range(0, int(MatrixW/OutP)):
			for i in range(0, OutP):
				hex_A[i][j] = hexFromInt(A[index], Mbit) + hex_A[i][j]
				hex_A[i][j] = '\"0x' + hex_A[i][j] + '\"';
				index += 1

		index = 0
		hex_B = [['' for j in range(int(MatrixW/OutP)) ] for i in range(OutP)]
		for j in range(0, int(MatrixW/OutP)):
			for i in range(0, OutP):	
				hex_B[i][j] = hexFromInt(B[index], Mbit) + hex_B[i][j]
				hex_B[i][j] = '\"0x' + hex_B[i][j] + '\"';
				index += 1

		if VERBOSITY == 2:
			print('hex_A: ' + str(hex_A) )
			print('hex_B: ' + str(hex_B) )

	return hex_A, hex_B

def genereateHLSparams(layers_list, network_model, path_to_params, fw):
	hlsweights_file_content = ''

	total_weight_mem_usage = 0
	total_thresh_mem_usage = 0
	total_linebuf_mem_usage = 0

	for layer in layers_list:
		if layer['func'] in ['conv_layer', 'fc_layer']:
			print('layer name: ' + layer['name'])
			print('layer[m]:' + layer['m'])
			for m in network_model:
				if str(layer['m']) in m.name:
					layer['m'] = m

			print('layer[m]:' + str(layer['m']) )

			next_layer = None
			index = layers_list.index(layer)+1
			if index < len(layers_list):
				next_layer = layers_list[index]

			if next_layer != None:
				print('Next layer of ' + layer['name'] +  ' is: ' + next_layer['name'])
				to_search = next_layer['name'][0:next_layer['name'].rfind('/')]
				print('to_search: ' + to_search)

				if next_layer['func'] == 'bnorm_layer':
					for m in network_model:
						if to_search in m.name:
							if 'beta' in m.name:
								layer['bn_beta'] = m
							elif 'gamma' in m.name:
								layer['bn_gamma'] = m
							elif 'mean' in m.name:
								layer['bn_mean'] = m
							elif 'variance' in m.name:
								layer['bn_variance'] = m

			print('Generating Weights of layer ' + layer['m'].name)

			if layer['Wbit'] >= FIXED_BITS:
				W = layer['m']
			else:
				W = fw(layer['m'])
			W = W.eval()

			hlsweights_file_content = hlsweights_file_content + "\n"

			if layer['func'] == 'conv_layer':
				InP = layer['MVTU_InP']
				OutP = layer['MVTU_OutP']
				MatrixW = layer['Cout']
				MatrixH = layer['Cin']*layer['K']*layer['K']

				WmemUsage = layer['Wbit']*MatrixW*MatrixH
				TmemUsage = layer['Mbit']*MatrixW*(2**layer['Abit']-1)
				LBUFmemUsage = layer['K']*layer['input'][1]*layer['Cin']*layer['Ibit']
				hlsweights_file_content = hlsweights_file_content + "// weight_mem_usage: " +  str(WmemUsage) + " bits\n"
				hlsweights_file_content = hlsweights_file_content + "// thresh_mem_usage: " +  str(TmemUsage) + " bits\n"
				hlsweights_file_content = hlsweights_file_content + "// linebuf_mem_usage: " + str(LBUFmemUsage) + " bits\n"
				hlsweights_file_content = hlsweights_file_content + "// total_mem_usage: " + str(WmemUsage+TmemUsage+LBUFmemUsage) + " bits\n"
				total_weight_mem_usage += WmemUsage 
				total_thresh_mem_usage += TmemUsage 
				total_linebuf_mem_usage += LBUFmemUsage

			elif layer['func'] == 'fc_layer':
				W = W.transpose()
				InP = layer['InP']
				OutP = layer['OutP']
				MatrixW = layer['output']
				MatrixH = layer['input']

				WmemUsage = layer['Wbit']*MatrixW*MatrixH
				TmemUsage = layer['Mbit']*MatrixW*(2**layer['Abit']-1)
				hlsweights_file_content = hlsweights_file_content + "// weight_mem_usage: " +  str(WmemUsage) + " bits\n"
				hlsweights_file_content = hlsweights_file_content + "// thresh_mem_usage: " +  str(TmemUsage) + " bits\n"
				hlsweights_file_content = hlsweights_file_content + "// total_mem_usage: " + str(WmemUsage+TmemUsage) + " bits\n"
				total_weight_mem_usage += WmemUsage
				total_thresh_mem_usage += TmemUsage


			Weights, Wmax = computeWeights(W, layer['Wbit'], InP, OutP)

			hlsweights_file_content = hlsweights_file_content + formatWeights(layer['basemem'], layer['Wbit'], InP, OutP, MatrixW, MatrixH, Weights)

			if layer['Abit'] < THRESHOLD_TO_FACTOR_TRANSITION:
				if 'bn_beta' in layer:
					if VERBOSITY == 1:
						print('Found following batch normalization of this layer: ')
						print(layer['bn_beta'].name)
						print(layer['bn_gamma'].name)
						print(layer['bn_mean'].name)
						print(layer['bn_variance'].name)
					thresholds = computeThresholds( MatrixW, OutP, Wmax, layer['bn_beta'].eval(), layer['bn_gamma'].eval(), layer['bn_mean'].eval(), layer['bn_variance'].eval(), layer['Mbit'], layer['Abit'])
				else:
					thresholds = computeThresholds( MatrixW, OutP, Wmax, [0], [0], [0], [0], layer['Mbit'], layer['Abit'])

				hlsweights_file_content = hlsweights_file_content + formatThresholds(layer['basemem'], layer['Mbit'], 2**layer['Abit']-1, OutP, MatrixW, thresholds)
			else:
				if 'bn_beta' in layer:
					if VERBOSITY == 1:
						print('Found following batch normalization of this layer: ')
						print(layer['bn_beta'].name)
						print(layer['bn_gamma'].name)
						print(layer['bn_mean'].name)
						print(layer['bn_variance'].name)
					factorA, factorB = computeFactors( MatrixW, OutP, Wmax, layer['bn_beta'].eval(), layer['bn_gamma'].eval(), layer['bn_mean'].eval(), layer['bn_variance'].eval(),  layer['Mbit'])
				else:
					factorA, factorB = computeFactors( MatrixW, OutP, Wmax, [0], [0], [0], [0], layer['Mbit'] )

				hlsweights_file_content = hlsweights_file_content + formatFactors('factorA', layer['basemem'], layer['Mbit'], OutP, MatrixW, factorA)
				hlsweights_file_content = hlsweights_file_content + formatFactors('factorB', layer['basemem'], layer['Mbit'], OutP, MatrixW, factorB)

	f = open(path_to_params, "w")
	f.write("// Total memory usage: " + str(total_weight_mem_usage + total_thresh_mem_usage + total_linebuf_mem_usage) + " bits\n")
	f.write("// total_weight_mem_usage:" + str(total_weight_mem_usage) + " bits\n")
	f.write("// total_thresh_mem_usage:" + str(total_thresh_mem_usage) + " bits\n")
	f.write("// total_linebuf_mem_usage:" + str(total_linebuf_mem_usage) + " bits\n")
	f.write(hlsweights_file_content)
	f.close()

def writeDefine(defineName, defineValue, configFile):
	configFile.write("#define "+str(defineName)+" "+str(defineValue)+"\n")

def generateConfig(layers_list, path_to_config):
	configFile = open(path_to_config, "w")

	current_stream = "in_stream"
	network = "\n"
	pragmas = "\n"

	for layer in layers_list:
		if layer['func'] == "conv_layer":
			layerPrefix = "L"+str(layer['basemem']);

			configFile.write("// " + layer['name'] + "\n")
			configFile.write("// Cycles per IFM: " + str(layer['cycles']) + "\n")

			pragmas += "// #pragma HLS RESOURCE variable=weights" + str(layer['basemem']) + " core=RAM_1P_BRAM\n"
			pragmas += "// #pragma HLS ARRAY_PARTITION variable=weights" + str(layer['basemem']) + " complete dim=0\n"
			if layer['Abit'] < THRESHOLD_TO_FACTOR_TRANSITION:
				pragmas += "// #pragma HLS ARRAY_PARTITION variable=thresholds" + str(layer['basemem']) + " complete dim=0\n"
			else:
				pragmas += "// #pragma HLS RESOURCE variable=factorA" + str(layer['basemem']) + " core=RAM_1P_BRAM\n"
				pragmas += "// #pragma HLS RESOURCE variable=factorB" + str(layer['basemem']) + " core=RAM_1P_BRAM\n"
				pragmas += "// #pragma HLS ARRAY_PARTITION variable=factorA" + str(layer['basemem']) + " complete dim=0\n"
				pragmas += "// #pragma HLS ARRAY_PARTITION variable=factorB" + str(layer['basemem']) + " complete dim=0\n"

			writeDefine(layerPrefix+'_K', layer['K'], configFile)
			writeDefine(layerPrefix+'_S', layer['S'], configFile)
			writeDefine(layerPrefix+'_Din', layer['input'][1], configFile)
			writeDefine(layerPrefix+'_Cin', layer['input'][0], configFile)
			writeDefine(layerPrefix+'_Cout', layer['output'][0], configFile)
			writeDefine(layerPrefix+'_Ibit', layer['Ibit'], configFile)
			writeDefine(layerPrefix+'_Wbit', layer['Wbit'], configFile)
			writeDefine(layerPrefix+'_Mbit', layer['Mbit'], configFile)
			writeDefine(layerPrefix+'_Abit', layer['Abit'], configFile)
			writeDefine(layerPrefix+'_SWU_OutP', layer['SWU_OutP'], configFile)
			writeDefine(layerPrefix+'_MVTU_InP', layer['MVTU_InP'], configFile)
			writeDefine(layerPrefix+'_MVTU_OutP', layer['MVTU_OutP'], configFile)

			current_stream = "conv" + str(layer['basemem'])

			configFile.write("\n")
			
		if layer['func'] == "fc_layer":
			layerPrefix = "L"+str(layer['basemem']);

			configFile.write("// " + layer['name'] + "\n")
			configFile.write("// Cycles per IFM: " + str(layer['cycles']) + "\n")
			
			pragmas += "// #pragma HLS RESOURCE variable=weights" + str(layer['basemem']) + " core=RAM_1P_BRAM\n"
			pragmas += "// #pragma HLS ARRAY_PARTITION variable=weights" + str(layer['basemem']) + " complete dim=0\n"
			if layer['Abit'] < THRESHOLD_TO_FACTOR_TRANSITION:
				pragmas += "// #pragma HLS ARRAY_PARTITION variable=thresholds" + str(layer['basemem']) + " complete dim=0\n"
			else:
				pragmas += "// #pragma HLS RESOURCE variable=factorA" + str(layer['basemem']) + " core=RAM_1P_BRAM\n"
				pragmas += "// #pragma HLS RESOURCE variable=factorB" + str(layer['basemem']) + " core=RAM_1P_BRAM\n"
				pragmas += "// #pragma HLS ARRAY_PARTITION variable=factorA" + str(layer['basemem']) + " complete dim=0\n"
				pragmas += "// #pragma HLS ARRAY_PARTITION variable=factorB" + str(layer['basemem']) + " complete dim=0\n"

			writeDefine(layerPrefix+'_Din', layer['input'], configFile)
			writeDefine(layerPrefix+'_Dout', layer['output'], configFile)
			writeDefine(layerPrefix+'_Ibit', layer['Ibit'], configFile)
			writeDefine(layerPrefix+'_Wbit', layer['Wbit'], configFile)
			writeDefine(layerPrefix+'_Mbit', layer['Mbit'], configFile)
			writeDefine(layerPrefix+'_Abit', layer['Abit'], configFile)
			writeDefine(layerPrefix+'_InP', layer['InP'], configFile)
			writeDefine(layerPrefix+'_OutP', layer['OutP'], configFile)

			current_stream = "dense" + str(layer['basemem'])

			configFile.write("\n")

		if layer['func'] == "maxpool_layer":
			poolPrefix = "L"+str(layer['basemem']);

			configFile.write("// " + layer['name'] + "\n")
			configFile.write("// Cycles per IFM: " + str(layer['cycles']) + "\n")

			writeDefine(poolPrefix+'_K', layer['K'], configFile)
			writeDefine(poolPrefix+'_S', layer['S'], configFile)
			writeDefine(poolPrefix+'_Din', layer['input'][1], configFile)
			writeDefine(poolPrefix+'_Cin', layer['input'][0], configFile)
			writeDefine(poolPrefix+'_Ibit', layer['Ibit'], configFile)
			writeDefine(poolPrefix+'_SWU_OutP', layer['SWU_OutP'], configFile)

			current_stream = "pool" + str(layer['basemem'])

			configFile.write("\n")

	writeDefine("SCALE_BITS", SCALE_BITS, configFile)
	writeDefine("FACTOR_SCALE_BITS", FACTOR_SCALE_BITS, configFile)
	writeDefine("HIGH_PREC_SCALE_BITS", HIGH_PREC_SCALE_BITS, configFile)

	configFile.write(pragmas)
	configFile.write(network)

	configFile.close()

	return layers_list

def generateLayers(session, activation_bits, weight_bits, non_quantized_layers, frequency, FMpS_target):
	important_ops = []

	variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	for v in variables:
		if VERBOSITY == 2:
			print(v.name)
		v_value = session.run(v)
		if VERBOSITY == 2:
			print('shape: ' + str(v_value.shape))
		important_ops.append([v.name[0:len(v.name)-2], 'Variable'])
		if 'conv' in v.name and 'W' in v.name:
			important_ops.append([v.name[0:len(v.name)-4], 'Conv2D'])
		if 'fc' in v.name and 'W' in v.name:
			important_ops.append([v.name[0:len(v.name)-4], 'MatMul'])

	if VERBOSITY == 2:
		print('|---------------------------------------------------------|')
		for io in important_ops:
			print(io)
		print('|---------------------------------------------------------|')

	ops_list = []
	ops = session.graph.get_operations()

	for io in important_ops:
		if VERBOSITY == 2:
			print('|---------------------------------------------------------|')
			print(str(o.outputs))
			print(o.name)
		for o in ops:
			if (io[0] == o.name or (io[0] in o.name and io[1] != 'Variable')) and io[1] in o.op_def.name and 'gradients' not in o.name and 'Inference' not in o.name:
				if VERBOSITY == 1:
					print('|---------------------------------------------------------|')
					print(o.name)
					print(str(o.outputs))
					print(o.op_def.name)
				if 'Conv2D' in o.name:
					ops_list.append([o.name, o.outputs, o.inputs, o.get_attr('strides')])
				else:
					ops_list.append([o.name, o.outputs, o.inputs])

				break

	for o in ops:
		if 'pool' in o.name and 'Pool' in o.op_def.name and 'gradients' not in o.name and 'Inference' not in o.name:
			if VERBOSITY == 1:
				print('|---------------------------------------------------------|')
				print(o.name)
				print(str(o.outputs))
				print(o.op_def.name)
			if 'pool' in o.name:
				ops_list.append([o.name, o.outputs, o.inputs, o.get_attr('strides'), o.get_attr('ksize')])


	max_cycles = 0
	previous_OutP = 0
	previous_variable = None
	basemem = 0;
	last_output = []
	json_out = ''
	layers_list = []
	for o in ops_list:
		params = {}

		if VERBOSITY == 1:
			print('|---------------------------------------------------------|')
			print( 'name: ' + o[0] + ', outputs: ' + str(o[1]) )
			for i in o[2]:
				print('inputs: ' + str(i))
		
		if o[0] in non_quantized_layers:
			this_weight_bits = FIXED_BITS
			if 'obj' in o[0] or 'box' in o[0]:
				this_weight_bits = FIXED_BITS+4
		else:
			this_weight_bits = weight_bits

		if 'W' in o[0]:
			previous_variable = o[0]

		if 'Conv2D' in o[0]:
			name = o[0]
			input_shape = o[2][0].get_shape()
			output_shape = o[1][0].get_shape()
			kernel_shape = o[2][1].get_shape()
			strides = o[3];

			print('|---------------------------------------------------------|')
			print('name: ' + name)
			print('input_shape: ' + str(input_shape) )
			print('output_shape: ' + str(output_shape) )
			print('kernel_shape: ' + str(kernel_shape))
			print('strides: ' + str(strides))

			if basemem == 0:
				input_bits = 8
			else:
				input_bits = int(activation_bits)

			mac_bits = MAC_BITS

			params = {	'func':'conv_layer',
						'input':[int(input_shape[3]), int(input_shape[2]), int(input_shape[1])],
						'Cin':int(input_shape[3]),
						'Ibit':input_bits,
						'Abit':int(activation_bits),
						'Wbit':int(this_weight_bits),
						'Mbit':mac_bits,
						'Cout':int(output_shape[3]),
						'K':int(kernel_shape[0]),
						'S':int(strides[1]),
						'name':name,
						'output':[int(output_shape[3]), int(output_shape[2]), int(output_shape[1])],
						'basemem':basemem}
			layers_list.append(params)
			json_out += json.dumps(params) + ','
			last_output = [int(output_shape[3]), int(output_shape[2]), int(output_shape[1])]
			
			Din = float(params['input'][1])
			Dout = Din/float(params['S'])
			K = float(params['K'])
			Cin = float(params['Cin'])
			Cout = float(params['Cout'])
			Constant = 6;

			SWU_OutP = 1
			print('SWU_OutP: ' + str(SWU_OutP))
			if Cin%MAX_INP == 0:
				MVTU_InP = MAX_INP
			else:
				if Cin == 3:
					MVTU_InP = 3
				else:
					MVTU_InP = 1

			if 'expand' in name:
				MVTU_InP = 32
			elif 'squeeze' in name:
				MVTU_InP = 8
			print('MVTU_InP: ' + str(MVTU_InP))
			if Cout%MAX_OUTP == 0:
				MVTU_OutP = MAX_OUTP
			else:
				MVTU_OutP = Cout

			if 'conv1' in name:
				MVTU_OutP = 8
			elif 'expand' in name:
				MVTU_OutP = 8
			elif 'squeeze' in name:
				MVTU_OutP = 4
			print('MVTU_OutP: ' + str(MVTU_OutP))

			SWU_cycles = Dout*(Din + (Dout*K*K)/SWU_OutP + Constant)
			MVTU_cycles = (Cin*K*K*Cout*Dout*Dout)/(SWU_OutP*MVTU_InP*MVTU_OutP)

			print('Raw SWU_cycles: ' + str(SWU_cycles) )
			print('Raw MVTU_cycles: ' + str(MVTU_cycles) )

			print('--- After mods are 0: ---')
			print('SWU_OutP: ' + str(SWU_OutP))
			print('MVTU_InP: ' + str(MVTU_InP))
			print('MVTU_OutP: ' + str(MVTU_OutP))

			SWU_cycles = Dout*(Din + (Dout*K*K)/SWU_OutP + Constant)
			MVTU_cycles = (Cin*K*K*Cout*Dout*Dout)/(SWU_OutP*MVTU_InP*MVTU_OutP)

			all_cycles = [SWU_cycles, MVTU_cycles]

			print('SWU_cycles: ' + str(SWU_cycles) )
			print('MVTU_cycles: ' + str(MVTU_cycles) )

			print('Target FMpS: ' + str(FMpS_target))
			achieved_FMpS = frequency/max(all_cycles)
			print('Achieved FMpS: ' + str(achieved_FMpS))

			if achieved_FMpS < FMpS_target:
				FMpS_target = achieved_FMpS

			params['SWU_OutP'] = int(SWU_OutP)
			params['MVTU_InP'] = int(MVTU_InP)
			params['MVTU_OutP'] = int(MVTU_OutP)

			basemem += 1

			previous_OutP = Cout
			params['m'] = previous_variable
			print('m: ' + params['m'])

			params['cycles'] = max(all_cycles)
			if (max(all_cycles) > max_cycles):
				max_cycles = max(all_cycles)

		elif 'beta' in o[0]:
			print('|---------------------------------------------------------|')
			name = o[0]
			print('name: ' + name)
			print('input_shape: ' + str(last_output) )
			print('output_shape: ' + str(last_output) )
			
			mac_bits = MAC_BITS

			params = {	'func':'bnorm_layer',
						'input':last_output,
						'Mbit':mac_bits,
						'Ibit':int(activation_bits),
						'Wbit':int(this_weight_bits),
						'name':name,
						'output':last_output}
			layers_list.append(params)
			json_out += json.dumps(params) + ','

		elif 'pool' in o[0]:
			print('|---------------------------------------------------------|')
			name = o[0]
			print('name: ' + name)
			strides = o[3];
			input_shape = o[2][0].get_shape()
			print('input_shape: ' + str(input_shape) )
			output_shape = o[1][0].get_shape()
			print('output_shape: ' + str(output_shape) )
			print('strides: ' + str(strides))
			ksize = o[4];
			print('ksize: ' +str(ksize))

			mac_bits = MAC_BITS

			params = {	'func':'maxpool_layer',
						'Wbit':int(this_weight_bits),
						'Abit':int(activation_bits),
						'Mbit':mac_bits,
						'Ibit':int(activation_bits),
						'input':[int(input_shape[3]), int(input_shape[2]), int(input_shape[1])],
						'Cin':int(input_shape[3]),
						'S':strides[1],
						'K':ksize[1],
						'name':name,
						'output':[int(output_shape[3]), int(output_shape[2]), int(output_shape[1])],
						'basemem':basemem}
			layers_list.append(params)
			json_out += json.dumps(params) + ','
			last_output = [int(output_shape[3]), int(output_shape[2]), int(output_shape[1])]

			Din = float(params['input'][1])
			Dout = Din/float(params['S'])
			K = float(params['K'])
			Cin = float(params['Cin'])
			Constant = 6
			
			SWU_OutP = 1
			print('SWU_OutP: ' + str(SWU_OutP))

			SWU_cycles = Dout*(Din + (Dout*K*K)/SWU_OutP + Constant)
			print('Raw SWU_cycles: ' + str(SWU_cycles) )

			if frequency/SWU_cycles < FMpS_target:
				SWU_OutP = K
			else:
				SWU_OutP = 1

			print('--- After mods are 0: ---')
			print('SWU_OutP: ' + str(SWU_OutP))

			SWU_cycles = Dout*(Din + (Dout*K*K)/SWU_OutP + Constant)

			print('SWU_cycles: ' + str(SWU_cycles) )

			print('Target FMpS: ' + str(FMpS_target))
			achieved_FMpS = frequency/SWU_cycles
			print('Achieved FMpS: ' + str(achieved_FMpS))

			if achieved_FMpS < FMpS_target:
				FMpS_target = achieved_FMpS

			params['SWU_OutP'] = int(SWU_OutP)

			previous_OutP = Cin;

			params['cycles'] = SWU_cycles
			if (SWU_cycles > max_cycles):
				max_cycles = SWU_cycles

			basemem += 1

		elif 'MatMul' in o[0]:
			print('|---------------------------------------------------------|')
			name = o[0]
			print('name: ' + name)
			input_shape = o[2][0].get_shape()
			print('input_shape: ' + str(input_shape) )
			output_shape = o[1][0].get_shape()
			print('output_shape: ' + str(output_shape) )
			
			if basemem == 0:
				input_bits = 8
			else:
				input_bits = int(activation_bits)

			mac_bits = MAC_BITS

			params = {	'func':'fc_layer',
						'name':name,
						'input':int(input_shape[1]),
						'output':int(output_shape[1]),
						'Wbit':int(this_weight_bits),
						'Abit':int(activation_bits),
						'Mbit':mac_bits,
						'Ibit':input_bits,
						'basemem':basemem}
			layers_list.append(params)
			json_out += json.dumps(params) + ','
			last_output = [int(output_shape[1])]

			MatrixH = float(params['input'])
			MatrixW = float(params['output'])

			if previous_OutP > 0:
				InP = previous_OutP
			else:
				InP = math.sqrt(MatrixH)

			OutP = MAX_OUTP
			print('InP: ' + str(InP))
			print('OutP: ' + str(OutP))

			MVTU_cycles = (MatrixH*MatrixW)/(InP*OutP)

			print('Raw MVTU_cycles: ' + str(MVTU_cycles))

			while MatrixH%InP > 0:
				if InP > MatrixH:
					InP = MatrixH
				else:
					InP += 1

			while MatrixW%OutP > 0:
				if OutP > MatrixW:
					OutP = MatrixW
				else:
					OutP += 1

			print('--- After mods are 0: ---')
			print('InP: ' + str(InP))
			print('OutP: ' + str(OutP))

			MVTU_cycles = (MatrixH*MatrixW)/(InP*OutP)

			print('Target FMpS: ' + str(FMpS_target))
			achieved_FMpS = frequency/MVTU_cycles
			print('Achieved FMpS: ' + str(achieved_FMpS))

			if achieved_FMpS < FMpS_target:
				FMpS_target = achieved_FMpS

			params['InP'] = int(InP)
			params['OutP'] = int(OutP)

			basemem += 1

			previous_OutP = OutP
			params['m'] = previous_variable
			print('m: ' + params['m'])

			params['cycles'] = MVTU_cycles
			if (MVTU_cycles > max_cycles):
				max_cycles = MVTU_cycles

	return json_out, layers_list, max_cycles

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--meta', help='metagraph file', required=True)
	parser.add_argument('--model', help='model file', required=True)
	parser.add_argument('--w', help='Give weight bits', required=True)
	parser.add_argument('--a', help='Give activation bits', required=True)
	args = parser.parse_args()

	weight_bits = args.w
	activation_bits = args.a

	with tf.Graph().as_default() as G:
		tf.train.import_meta_graph(args.meta)

		# loading...
		init = get_model_loader(args.model)
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		sess.run(tf.global_variables_initializer())
		init.init(sess)

		with sess.as_default():
			jsonfile = open(args.meta + '.net', 'w')

			json_out, _ = generateLayers(sess, activation_bits, weight_bits)

			jsonfile.write('[')
			jsonfile.write(json_out[:len(json_out)-1])
			jsonfile.write(']')