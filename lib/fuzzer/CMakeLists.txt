rlcAddLibrary(fuzzer src/fuzz_target.cpp)
llvm_map_components_to_libnames(llvm_utils_libs Support)
target_link_libraries(fuzzer)

set(LIB_RLC_FUZZER_FILENAME libfuzzer-d.so)
configure_file(include/rlc/fuzzer/Fuzzer.hpp.in include/rlc/fuzzer/Fuzzer.hpp)