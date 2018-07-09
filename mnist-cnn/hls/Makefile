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

CINCLUDES	+= -I../../hls-nn-lib/
CINCLUDES 	+= -I${XILINX_VIVADO_HLS}/include/
CINCLUDES 	+= -std=c++0x -Wall -Wno-unknown-pragmas -Wall -Wno-unknown-pragmas -Wno-unused-variable -g
CXX			= g++
LDFLAGS		= -lpthread

all: mnist-cnn-1W5A

mnist-cnn-1W5A:
	$(CXX) $(CFLAGS) $(CINCLUDES) mnist-cnn-1W5A.cpp -o t_1W5A $(LDFLAGS)