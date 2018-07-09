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

import numpy as np

class loader:
	def __init__(self):
		self.num_samples = 0;
		self.num_features = 0;

		self.a = None;
		self.b = None;

	def load_libsvm_data(self, path_to_file, num_samples, num_features, one_hot, classes):
		self.num_samples = num_samples
		self.num_features = num_features

		self.a = np.zeros((self.num_samples, self.num_features))
		if one_hot == 0:
			self.b = np.zeros(self.num_samples)
		else:
			self.b = np.zeros(( self.num_samples, len(classes) ))

		f = open(path_to_file, 'r')
		for i in range(0, self.num_samples):
			line = f.readline()
			items = line.split()
			if one_hot == 0:
				self.b[i] = float(items[0])
			else:
				self.b[i, classes.index(items[0])] = 1
			for j in range(1, len(items)):
				item = items[j].split(':')
				self.a[i, int(item[0])] = float(item[1])