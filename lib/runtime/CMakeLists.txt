rlcAddLibrary(runtime STATIC src/Parse.c)
set_property(TARGET runtime PROPERTY C_STANDARD 11)
target_compile_options(runtime PUBLIC)

rlcAddLibraryNoTest(asan_runtime STATIC src/Parse.c)
set_property(TARGET asan_runtime PROPERTY C_STANDARD 11)
target_compile_options(asan_runtime PUBLIC "-fsanitize=address")
