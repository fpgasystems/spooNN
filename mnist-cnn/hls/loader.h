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

#ifndef _loader
#define _loader

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>

using namespace std;

class loader {
public:
	float* x;
	float* y;

	char x_normalizedToMinus1_1;

	uint32_t numSamples;
	uint32_t numFeatures;
	uint32_t numClasses;
	
	loader();
	~loader();

	void load_libsvm_data(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures, uint32_t _numClasses);
	
	void x_normalize(char toMinus1_1, char rowOrColumnWise);
};

loader::loader() {
	srand(7);

	x = NULL;
	y = NULL;

	numSamples = 0;
	numFeatures = 0;
	numClasses = 0;
}

loader::~loader() {
	if (x != NULL)
		free(x);
	if (y != NULL)
		free(y);
}

void loader::load_libsvm_data(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures, uint32_t _numClasses) {
	cout << "Reading " << pathToFile << endl;

	numSamples = _numSamples;
	numFeatures = _numFeatures;
	numClasses = _numClasses;
	x = (float*)calloc(numSamples*numFeatures, sizeof(float)); 
	y = (float*)calloc(numSamples*numClasses, sizeof(float));

	string line;
	ifstream f(pathToFile);

	uint32_t index = 0;
	if (f.is_open()) {
		while( index < numSamples ) {
			getline(f, line);
			int pos0 = 0;
			int pos1 = 0;
			int pos2 = 0;
			int column = 0;
			//while ( column < numFeatures-1 ) {
			while ( pos2 < (int)line.length()+1 ) {
				if (pos2 == 0) {
					pos2 = line.find(" ", pos1);
					uint32_t temp = stoi(line.substr(pos1, pos2-pos1), NULL);
					y[index*numClasses + temp] = 1.0;
				}
				else {
					pos0 = pos2;
					pos1 = line.find(":", pos1)+1;
					pos2 = line.find(" ", pos1);
					column = stof(line.substr(pos0+1, pos1-pos0-1));
					if (pos2 == -1) {
						pos2 = line.length()+1;
						x[index*numFeatures + column] = stof(line.substr(pos1, pos2-pos1), NULL);
					}
					else
						x[index*numFeatures + column] = stof(line.substr(pos1, pos2-pos1), NULL);
				}
			}
			index++;
		}
		f.close();
	}
	else {
		cout << "Unable to open libsvm file" << endl;
		return;
	}
	
	cout << "numSamples: " << numSamples << endl;
	cout << "numFeatures: " << numFeatures << endl;
}

void loader::x_normalize(char toMinus1_1, char rowOrColumnWise) {
	x_normalizedToMinus1_1 = toMinus1_1;
	if (rowOrColumnWise == 'r') {
		for (uint32_t i = 0; i < numSamples; i++) {
			float _min = numeric_limits<float>::max();
			float _max = numeric_limits<float>::min();
			for (uint32_t j = 1; j < numFeatures; j++) { // Don't normalize bias
				float x_here = x[i*numFeatures + j];
				if (x_here > _max)
					_max = x_here;
				if (x_here < _min)
					_min = x_here;
			}
			float _range = _max - _min;
			if (_range > 0) {
				if (toMinus1_1 == 1) {
					for (uint32_t j = 1; j < numFeatures; j++) { // Don't normalize bias
						x[i*numFeatures + j] = ((x[i*numFeatures + j] - _min)/_range)*2.0-1.0;
					}
				}
				else {
					for (uint32_t j = 1; j < numFeatures; j++) { // Don't normalize bias
						x[i*numFeatures + j] = ((x[i*numFeatures + j] - _min)/_range);
					}
				}
			}
		}
	}
	else {
		for (uint32_t j = 1; j < numFeatures; j++) { // Don't normalize bias
			float _min = numeric_limits<float>::max();
			float _max = numeric_limits<float>::min();
			for (uint32_t i = 0; i < numSamples; i++) {
				float x_here = x[i*numFeatures + j];
				if (x_here > _max)
					_max = x_here;
				if (x_here < _min)
					_min = x_here;
			}
			float _range = _max - _min;
			if (_range > 0) {
				if (toMinus1_1 == 1) {
					for (uint32_t i = 0; i < numSamples; i++) {
						x[i*numFeatures + j] = ((x[i*numFeatures + j] - _min)/_range)*2.0-1.0;
					}
				}
				else {
					for (uint32_t i = 0; i < numSamples; i++) {
						x[i*numFeatures + j] = ((x[i*numFeatures + j] - _min)/_range);
					}
				}
			}
		}
	}
}

#endif