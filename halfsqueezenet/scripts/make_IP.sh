#!/bin/bash

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

if [[ $# -ne 1 ]]; then 
	echo "Usage: ./make_hw.sh </path/to/network>"
	exit
fi

PATH_TO_SCRIPTS=$1/scripts
PATH_TO_OUTPUT=$1/output

echo "PATH_TO_SCRIPTS: $PATH_TO_SCRIPTS"
echo "PATH_TO_OUTPUT: $PATH_TO_OUTPUT"

if [ ! -d "$PATH_TO_OUTPUT" ]; then
	mkdir $PATH_TO_OUTPUT
fi

HLS_PROJECT="hls_project"
HLS_SRC="$1/hls"
HLS_NN_LIB="$1/../hls-nn-lib"


cd $PATH_TO_OUTPUT
vivado_hls -f $PATH_TO_SCRIPTS/create_hls.tcl -tclargs $HLS_PROJECT $HLS_SRC $HLS_NN_LIB