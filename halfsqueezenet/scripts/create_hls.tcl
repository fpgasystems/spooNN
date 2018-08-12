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

if {$argc != 5} {
	puts "argc ${argc}"
	puts [lindex $argv 0]
	puts [lindex $argv 1]
	puts [lindex $argv 2]
	puts [lindex $argv 3]
	puts [lindex $argv 4]
	puts "Expected <proj_name> </path/to/hls-src> </path/to/hls-nn-lib>"
	exit
}

set proj_name [lindex $argv 2]
set hls_src [lindex $argv 3]
set hls_nn_lib [lindex $argv 4]

puts "proj_name: ${proj_name}"
puts "hls_src: ${hls_src}" 
puts "hls_nn_lib: ${hls_nn_lib}" 

set config_toplevelfxn "halfsqueezenet"
set config_proj_part "xc7z020clg400-1"
set config_clkperiod 12

open_project $proj_name
add_files "${hls_src}/halfsqueezenet_folded.cpp" -cflags "-std=c++0x -I${hls_nn_lib}"
set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

# syntesize and export
create_clock -period $config_clkperiod -name default
csynth_design
export_design -format ip_catalog
exit 0