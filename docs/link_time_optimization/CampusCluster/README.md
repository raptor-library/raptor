Version 1 (Tue 03 Oct 2017 10:28:50 AM CDT )

This document summarize the steps to include LTO. Note that these 
steps are in addition to the regular steps needed to compile 
raptor.

Link time optimization was not achieved for raptor in the campus cluster.

Compiler and mpi implementation version:

    gcc (GCC) 6.2.0
    openMPI: mpiexec (OpenRTE) 2.0.1 and mpich HYDRA build 3.1.4

Step 1: Check linker version (need 2.21 or newer).
    $:~> ld --version
    GNU ld version 2.20.51.0.2-5.47.el6_9.1 20100205
    Copyright 2009 Free Software Foundation, Inc.
    This program is free software; you may redistribute it under the terms of
    the GNU General Public License version 3 or (at your option) a later version.
    This program has absolutely no warranty.

Note: The linker version is older than the one that seems to be required.
      The steps were executed anyway to test its effect.


Step 2: Add the -flto flag:
    in the raptor/cmake/cxx_config.cmake file look for the line
    containing the "CMAKE_CXX_FLAGS" macro. Add the -flto flag as in
    
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -flto")

Note: After this step, the program compiles correctly using either openMPI or MPICH
      but no benefit form LTO is observed.
    
Step 3: Activate the CMAKE_AR, CMAKE_CXX_ARCHIVE_CREATE and CMAKE_CXX_ARCHIVE_FINISH macros:
    in the raptor/cmake/cxx_config.cmake file uncomment the lines including those macros as in
    
    SET(CMAKE_AR  "gcc-ar")
    SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
    SET(CMAKE_CXX_ARCHIVE_FINISH   true)
        
Note: After this step, the program don't compiles. It prints the 
      following error, apparently coming form gcc-ar:
      
    sorry - this program has been built without plugin support
    make[2]: *** [lib/libraptor.a] Error 1
    make[1]: *** [raptor/CMakeFiles/raptor.dir/all] Error 2
    make: *** [all] Error 2

