add_library(fuzzer STATIC fuzz_target.cpp)
install(TARGETS fuzzer DESTINATION lib)