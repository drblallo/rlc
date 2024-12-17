rlcAddLibrary(ml src/PPO.cpp)

execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND  python ${CMAKE_SOURCE_DIR}/python/find_site_packages.py  OUTPUT_VARIABLE TORCH_PACKAGES_LOCATION)
target_link_libraries(ml PRIVATE libtorch.so libc10.so)
target_link_directories(ml PRIVATE "${TORCH_PACKAGES_LOCATION}/torch/lib")
target_include_directories(ml PRIVATE "${TORCH_PACKAGES_LOCATION}/torch/include" "${TORCH_PACKAGES_LOCATION}/torch/include/torch/csrc/api/include/")
target_compile_options(ml PUBLIC -fexceptions -frtti -D_GLIBCXX_USE_CXX11_ABI=0)
