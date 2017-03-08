add_executable(test_amg test_amg.cpp)
target_link_libraries(test_amg raptor)
add_test(AMGTest ./test_amg)

