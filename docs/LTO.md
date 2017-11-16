The information on this page summarizes compiling RAPtor
with link time optimization (LTO) using the gnu compilers (-flto).
Link time optimization has been observed to produce significant speed-up.

The procedure for LTO varies from system to system.  Factors such as compiler version, linker version, mpi implementation all
effect the results.  The results are summarized for:
    1.- Blue Waters
    2.- Campus Cluster (University of Illinois)
    3.- Workstations

Important: The `ld` version has an impact in all cases; new versions link automatically.

# Automatic

Procedure for automatic Link Time Optimization (also known as Interprocedural optimization)

1.  Use of cmake version >= 3.9

    The first line of `CMakeLists.txt` located in the project directory (~/raptor/CMakeLists.txt) must be:

        cmake_minimum_required(VERSION 3.9)`

    Additionally, the following line also has to be included in this `CMakeLists.txt`:

        include(CheckIPOSupported)

2.  Check if iterprocedural optimizations are allowed (building the raptor library) .

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

1.  The procedure described above was tested successfully in dunkel, one of the computers of the MachineShop.

2.  There is no need to modify the ~/raptor/cmake/cxx_config.cmake as stated before.

3.  To verify if the correct flags are being send to the compiler you can look at a file called `flags.make`, e.g.: (`~/raptor/build/raptor/CMakeFiles/raptor.dir/flags.make`) and verify if the `-flto` flag (gnu compiler) or the `-ipo` flag (intel compiler) is there.
