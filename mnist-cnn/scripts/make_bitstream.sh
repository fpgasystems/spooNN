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

HW_ROOT=$1

if [ -n "$HW_ROOT" ]; then
	echo "HW_ROOT=$HW_ROOT"

	SCRIPT_DIR="$HW_ROOT/scripts"
	IP_REPO="$HW_ROOT/repo"

	PROJ_NAME="pynq-vivado"
	VIVADO_OUT_DIR="$HW_ROOT/output/$PROJ_NAME"
	VIVADO_SCRIPT=$SCRIPT_DIR/make-vivado-proj.tcl
	vivado -mode batch -notrace -source $VIVADO_SCRIPT -tclargs $PROJ_NAME $VIVADO_OUT_DIR $SCRIPT_DIR $IP_REPO
else
	echo "HW_ROOT is NOT set!"
fi