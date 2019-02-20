# from https://cmake.org/Wiki/CMake:How_To_Find_Libraries

find_package(PkgConfig)
pkg_check_modules(MUELU QUIET muelu)


find_path(MUELU_INCLUDE_DIR Epetra_MPIComm.h
    HINTS ${PC_MUELU_INCLUDEDIR} ${PC_MUELU_INCLUDE_DIRS}
          ${MUELU_DIR}/include ${MUELU_DIR} 
          $ENV{MUELU_DIR}/include $ENV{MUELU_DIR})



find_library(MUELU_LIBRARY NAMES muelu-adapters
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU1_LIBRARY NAMES muelu-interface
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU2_LIBRARY NAMES muelu
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU3_LIBRARY NAMES intrepid2
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU4_LIBRARY NAMES teko
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU5_LIBRARY NAMES stratimikos
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU6_LIBRARY NAMES stratimikosbelos
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU7_LIBRARY NAMES stratimikosamesos2
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU8_LIBRARY NAMES stratimikosaztecoo
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU9_LIBRARY NAMES stratimikosamesos
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU10_LIBRARY NAMES stratimikosml
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU11_LIBRARY NAMES stratimikosifpack
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU12_LIBRARY NAMES ifpack2-adapters
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU13_LIBRARY NAMES ifpack2
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU14_LIBRARY NAMES anasazitpetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU15_LIBRARY NAMES ModeLaplace
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU16_LIBRARY NAMES anasaziepetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU17_LIBRARY NAMES anasazi
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU18_LIBRARY NAMES amesos2
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU19_LIBRARY NAMES shylu_nodetacho
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU20_LIBRARY NAMES shylu_nodehts
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU21_LIBRARY NAMES belosxpetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU22_LIBRARY NAMES belostpetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU23_LIBRARY NAMES belosepetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU24_LIBRARY NAMES belos
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU25_LIBRARY NAMES ml
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU26_LIBRARY NAMES ifpack
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU27_LIBRARY NAMES zoltan2
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU28_LIBRARY NAMES amesos
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU29_LIBRARY NAMES galeri-xpetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU30_LIBRARY NAMES galeri-epetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU31_LIBRARY NAMES aztecoo
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU32_LIBRARY NAMES isorropia
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU33_LIBRARY NAMES xpetra-sup
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU34_LIBRARY NAMES xpetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU35_LIBRARY NAMES thyratpetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU36_LIBRARY NAMES thyraepetraext
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU37_LIBRARY NAMES thyraepetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU38_LIBRARY NAMES thyracore
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU39_LIBRARY NAMES trilinosss
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU40_LIBRARY NAMES tpetraext
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU41_LIBRARY NAMES tpetrainout
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU42_LIBRARY NAMES tpetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU43_LIBRARY NAMES kokkostsqr
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU44_LIBRARY NAMES tpetraclassiclinalg
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU45_LIBRARY NAMES tpetraclassicnodeapi
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU46_LIBRARY NAMES tpetraclassic
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU47_LIBRARY NAMES epetraext
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU48_LIBRARY NAMES triutils
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU49_LIBRARY NAMES shards
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU50_LIBRARY NAMES zoltan
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU51_LIBRARY NAMES epetra
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU52_LIBRARY NAMES sacado
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU53_LIBRARY NAMES rtop
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU54_LIBRARY NAMES kokkoskernels
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU55_LIBRARY NAMES teuchoskokkoscomm 
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU56_LIBRARY NAMES teuchoskokkoscompat
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU57_LIBRARY NAMES teuchosremainder
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU58_LIBRARY NAMES teuchosnumerics
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU59_LIBRARY NAMES teuchoscomm
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU60_LIBRARY NAMES teuchosparameterlist
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU61_LIBRARY NAMES teuchosparser
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU62_LIBRARY NAMES teuchoscore
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU63_LIBRARY NAMES kokkosalgorithms
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU64_LIBRARY NAMES kokkoscontainers
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU65_LIBRARY NAMES kokkoscore
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})
find_library(MUELU66_LIBRARY NAMES gtest
             HINTS ${PC_MUELU_LIBDIR} ${PC_MUELU_LIBRARY_DIRS}
             ${MUELU_DIR}/lib ${MUELU_DIR}
             $ENV{MUELU_DIR}/lib $ENV{MUELU_DIR})


set(MUELU_LIBRARIES ${MUELU_LIBRARY} ${MUELU1_LIBRARY} ${MUELU2_LIBRARY}
    ${MUELU3_LIBRARY} ${MUELU4_LIBRARY} ${MUELU5_LIBRARY} ${MUELU6_LIBRARY}
    ${MUELU7_LIBRARY} ${MUELU8_LIBRARY} ${MUELU9_LIBRARY} ${MUELU10_LIBRARY}
    ${MUELU11_LIBRARY} ${MUELU12_LIBRARY} ${MUELU13_LIBRARY} ${MUELU14_LIBRARY}
    ${MUELU15_LIBRARY} ${MUELU16_LIBRARY} ${MUELU17_LIBRARY} ${MUELU18_LIBRARY}
    ${MUELU19_LIBRARY} ${MUELU20_LIBRARY} ${MUELU21_LIBRARY} ${MUELU22_LIBRARY}
    ${MUELU23_LIBRARY} ${MUELU24_LIBRARY} ${MUELU25_LIBRARY} ${MUELU26_LIBRARY}
    ${MUELU27_LIBRARY} ${MUELU28_LIBRARY} ${MUELU29_LIBRARY} ${MUELU30_LIBRARY}
    ${MUELU31_LIBRARY} ${MUELU32_LIBRARY} ${MUELU33_LIBRARY} ${MUELU34_LIBRARY}
    ${MUELU35_LIBRARY} ${MUELU36_LIBRARY} ${MUELU37_LIBRARY} ${MUELU38_LIBRARY}
    ${MUELU39_LIBRARY} ${MUELU40_LIBRARY} ${MUELU41_LIBRARY} ${MUELU42_LIBRARY}
    ${MUELU43_LIBRARY} ${MUELU44_LIBRARY} ${MUELU45_LIBRARY} ${MUELU46_LIBRARY}
    ${MUELU47_LIBRARY} ${MUELU48_LIBRARY} ${MUELU49_LIBRARY} ${MUELU50_LIBRARY}
    ${MUELU51_LIBRARY} ${MUELU52_LIBRARY} ${MUELU53_LIBRARY} ${MUELU54_LIBRARY}
    ${MUELU55_LIBRARY} ${MUELU56_LIBRARY} ${MUELU57_LIBRARY} ${MUELU58_LIBRARY}
    ${MUELU59_LIBRARY} ${MUELU60_LIBRARY} ${MUELU61_LIBRARY} ${MUELU62_LIBRARY}
    ${MUELU63_LIBRARY} ${MUELU64_LIBRARY} ${MUELU65_LIBRARY} ${MUELU66_LIBRARY} )

set(MUELU_INCLUDE_DIRS ${TRILINOS_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MUELU  DEFAULT_MSG
                                  MUELU_LIBRARY MUELU_INCLUDE_DIR)

mark_as_advanced(MUELU_INCLUDE_DIR MUELU_LIBRARY )



