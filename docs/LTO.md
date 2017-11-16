The information on this page summarizes compiling RAPtor
with link time optimization (LTO) using the gnu compilers (`-flto`).
Link time optimization has been observed to produce significant speed-up.

The procedure for LTO varies from system to system.  Factors such as compiler version, linker version, mpi implementation all
effect the results.  The results are summarized for:

1. Blue Waters
2. Workstations

Important: The `ld` version has an impact in all cases; new versions link automatically.

# Automatic

Procedure for automatic Link Time Optimization (also known as Interprocedural optimization)

1.  Use of cmake version >= 3.9

    The first line of `CMakeLists.txt` located in the project directory (`~/raptor/CMakeLists.txt`) must be:

        cmake_minimum_required(VERSION 3.9)`

    Additionally, the following line also has to be included in this `CMakeLists.txt`:

        include(CheckIPOSupported)

2.  Check if iterprocedural optimizations are allowed (building the RAPtor library) .

    In `CMakeLists.txt` located in the raptor directory (`~/raptor/raptor/CMakeLists.txt`) add the following lines after the definition of the `target_link_libraries` (raptor ....):

        check_ipo_supported(RESULT result)
        if(result)
          set_property(TARGET raptor PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
        endif()

3.  Check if iterprocedural optimizations are allowed for any executable using the raptor library.  For example, for  `test_par_interpolation`:

    In the `CMakeLists.txt` located in the directory containing the program you want to link with the raptor library, (for this example `~/raptor/raptor/ruge_stuben/tests/CMakeLists.txt`) add the following lines after the definition of the `target_link_libraries(test_par_interpolation raptor .....)`:

        check_ipo_supported(RESULT result)
        if(result)
          set_property(TARGET test_par_interpolation PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
        endif()

### Notes:

1.  There is no need to modify `~/raptor/cmake/cxx_config.cmake`

3.  To verify if the correct flags are being send to the compiler you can look at a file called `flags.make`, e.g.: (`~/raptor/build/raptor/CMakeFiles/raptor.dir/flags.make`) and verify if the `-flto` flag (gnu compiler) or the `-ipo` flag (Intel compiler) is there.

# BlueWaters

Link time optimization for BlueWaters.

1.  Set the correct environment:

        module switch PrgEnv-cray PrgEnv-gnu
        module unload gcc
        module load gcc/6.3.0 (or other version of your choice)

2.  Check linker version (need 2.21 or newer).

        $:~> ld --version
        GNU ld (GNU Binutils; SUSE Linux Enterprise 11) 2.23.1
        Copyright 2012 Free Software Foundation, Inc.
        This program is free software; you may redistribute it under the terms of
        the GNU General Public License version 3 or (at your option) a later version.
        This program has absolutely no warranty.


3.  Add the `-flto` flag:
    in `raptor/cmake/cxx_config.cmake` look for the line
    containing the `CMAKE_CXX_FLAGS` macro. Add the `-flto` flag as in

        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -flto")

4.  Activate the `CMAKE_AR`, `CMAKE_CXX_ARCHIVE_CREATE`, and `CMAKE_CXX_ARCHIVE_FINISH` macros:
    in the r`aptor/cmake/cxx_config.cmake` file uncomment the lines including those macros as in

        SET(CMAKE_AR  "gcc-ar")
        SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
        SET(CMAKE_CXX_ARCHIVE_FINISH   true)

5.  Load the Blue Water python suite (as Blue Waters is set today, LTO does not work without this step):

        module load bwpy

# Workstations

Compiler and mpi implementation version:

    porter:
        gcc (Debian/Linaro 5.4.1-12) 5.4.1 20170820
        mpiexec (OpenRTE) 2.1.1

    dunkel and stout:
        gcc (Debian 7.2.0-5) 7.2.0
        mpiexec (OpenRTE) 2.1.1


1.  Check linker version (need 2.21 or newer).

        $:~> ld --version
        GNU ld (GNU Binutils for Debian) 2.29.1
        Copyright (C) 2017 Free Software Foundation, Inc.
        This program is free software; you may redistribute it under the terms of
        the GNU General Public License version 3 or (at your option) a later version.
        This program has absolutely no warranty.


2.  Add the -flto flag:
    in the raptor/cmake/cxx_config.cmake file look for the line
    containing the "CMAKE_CXX_FLAGS" macro. Add the -flto flag as in

        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -flto")

Note: After this step, if raptor is compiled dunkel and stout produce and executable
      that effectively includes LTO, with speedups of over 2 compared to no LTO.

Note: The following step was only required for porter. Notice that the compiler version
      in porter is older than the one used in dunkel and stout.

3. Activate the CMAKE_AR, CMAKE_CXX_ARCHIVE_CREATE and CMAKE_CXX_ARCHIVE_FINISH macros:
    in the raptor/cmake/cxx_config.cmake file uncomment the lines including those macros as in

        SET(CMAKE_AR  "gcc-ar")
        SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
        SET(CMAKE_CXX_ARCHIVE_FINISH   true)
