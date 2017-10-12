Version 1 (Tue 03 Oct 2017 08:43:39 AM CDT)

This document summarize the steps to include LTO. Note that these 
steps are in addition to the regular steps needed to compile 
raptor.

Link time optimization was effectively achieved in Blue Waters.

Step 0: Set the correct environment:
    module switch PrgEnv-cray PrgEnv-gnu
    module unload gcc
    module load gcc/6.3.0 (or other version of your choice)

Step 1: Check linker version (need 2.21 or newer).
    $:~> ld --version
    GNU ld (GNU Binutils; SUSE Linux Enterprise 11) 2.23.1
    Copyright 2012 Free Software Foundation, Inc.
    This program is free software; you may redistribute it under the terms of
    the GNU General Public License version 3 or (at your option) a later version.
    This program has absolutely no warranty.


Step 2: Add the -flto flag:
    in the raptor/cmake/cxx_config.cmake file look for the line
    containing the "CMAKE_CXX_FLAGS" macro. Add the -flto flag as in
    
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -flto")
    
Step 3: Activate the CMAKE_AR, CMAKE_CXX_ARCHIVE_CREATE and CMAKE_CXX_ARCHIVE_FINISH macros:
    in the raptor/cmake/cxx_config.cmake file uncomment the lines including those macros as in
    
    SET(CMAKE_AR  "gcc-ar")
    SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
    SET(CMAKE_CXX_ARCHIVE_FINISH   true)
        
Step 4: Load the Blue Water python suite (as Blue Waters is set today, LTO do not work without this step):
    module load bwpy


It is always important to do speed tests before and after the addition of LTO to verify that it 
is having a positive effect in raptor.
