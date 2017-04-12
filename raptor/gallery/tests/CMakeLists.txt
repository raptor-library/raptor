add_executable(test_stencil test_stencil.cpp)
target_link_libraries(test_stencil raptor)
add_test(StencilTest ./test_stencil)

add_executable(test_exxon_reader test_exxon_reader.cpp)
target_link_libraries(test_exxon_reader raptor)
add_test(ExxonReaderTest ./test_exxon_reader)

