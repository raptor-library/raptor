Procedure for automatic Link Time Optimization (also known as Interprocedural optimization)

1.- Use of cmake version >= 3.9

    The first line of the CMakeLists.txt located in the project directory (~/raptor/CMakeLists.txt) must be:

    cmake_minimum_required(VERSION 3.9)

    Additionally,the following line also has to be included in this CMakeLists.txt:
    
    include(CheckIPOSupported)

2.- Check if iterprocedural optimizations are allowed (building the raptor library) .

    In the CMakeLists.txt located in the raptor directory (~/raptor/raptor/CMakeLists.txt) add the following lines
    after the definition of the target_link_libraries(raptor ....):
    
    check_ipo_supported(RESULT result)
    if(result)
      set_property(TARGET raptor PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
    endif()

3.- Check if iterprocedural optimizations are allowed for any executable using the raptor library. 
    For example, for  test_exxon_reader:
    
    In the CMakeLists.txt located in the directory containing the program you want to link with the raptor libtary, 
    (for this example ~/raptor/raptor/gallery/tests/CMakeLists.txt) add the following lines after the definition 
    of the target_link_libraries(test_exxon_reader raptor .....):
    
    check_ipo_supported(RESULT result)
    if(result)
      set_property(TARGET test_exxon_reader PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
    endif()
    
Notes:

a.- The procedure described above was tested successfully in dunkel, one of the computers of the MachineShop.

b.- There is no need to modify the ~/raptor/cmake/cxx_config.cmake as stated before.

c.- To verify if the correct flags are being send to the compiler you can look at a file called flags.make, 
    e.g.: (~/raptor/build/raptor/CMakeFiles/raptor.dir/flags.make) and verify if the -flto flag (gnu compiler) 
    or the -ipo flag (intel compiler) is there.

