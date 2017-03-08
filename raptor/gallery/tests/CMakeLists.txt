add_executable(test_stencil test_stencil.cpp)
target_link_libraries(test_stencil raptor)
add_test(StencilTest ./test_stencil)

