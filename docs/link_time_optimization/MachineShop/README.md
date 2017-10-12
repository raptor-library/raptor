Version 1 (Tue 03 Oct 2017 09:51:26 AM CDT)

This document summarize the steps to include LTO. Note that these 
steps are in addition to the regular steps needed to compile 
raptor.

Link time optimization was effectively achieved in some systems of
the MachineShop (https://andreask.cs.illinois.edu/MachineShop/UserNotes).
Particularly, porter, dunkel and stout were studied.

Compiler and mpi implementation version:

    porter: 
        gcc (Debian/Linaro 5.4.1-12) 5.4.1 20170820
        mpiexec (OpenRTE) 2.1.1
        
    dunkel and stout: 
        gcc (Debian 7.2.0-5) 7.2.0
        mpiexec (OpenRTE) 2.1.1


Step 1: Check linker version (need 2.21 or newer).
    $:~> ld --version
    GNU ld (GNU Binutils for Debian) 2.29.1
    Copyright (C) 2017 Free Software Foundation, Inc.
    This program is free software; you may redistribute it under the terms of
    the GNU General Public License version 3 or (at your option) a later version.
    This program has absolutely no warranty.
Note: The three systems studied have the same linker version.


Step 2: Add the -flto flag:
    in the raptor/cmake/cxx_config.cmake file look for the line
    containing the "CMAKE_CXX_FLAGS" macro. Add the -flto flag as in
    
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -flto")

Note: After this step, if raptor is compiled dunkel and stout produce and executable
      that effectively includes LTO, with speedups of over 2 compared to no LTO.

Note: The following step was only required for porter. Notice that the compiler version
      in porter is older than the one used in dunkel and stout.
    
Step 3: Activate the CMAKE_AR, CMAKE_CXX_ARCHIVE_CREATE and CMAKE_CXX_ARCHIVE_FINISH macros:
    in the raptor/cmake/cxx_config.cmake file uncomment the lines including those macros as in
    
    SET(CMAKE_AR  "gcc-ar")
    SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
    SET(CMAKE_CXX_ARCHIVE_FINISH   true)
        

It is always important to do speed tests before and after the addition of LTO to verify that it 
is having a positive effect in raptor.
