clang -fuse-ld=lld -v -fsanitize=address input asa_runtime.lib 

"C:\\Users\\mofio\\Documents\\rlc-infrastructure\\llvm-install-release\\bin\\lld-link" 
"-out:.\\test\\Output\\alternative_to_string.rl.tmp.exe"
-defaultlib:libcmt 
-defaultlib:oldnames 
"-libpath:C:\\Users\\mofio\\Documents\\rlc-infrastructure\\llvm-install-release\\lib\\clang\\18\\lib\\windows" 
-nologo 
-debug 
-incremental:no 
"C:\\Users\\mofio\\Documents\\rlc-infrastructure\\llvm-install-release\\lib\\clang\\18\\lib\\windows\\clang_rt.asan-x86_64.lib" 
"-wholearchive:C:\\Users\\mofio\\Documents\\rlc-infrastructure\\llvm-install-release\\lib\\clang\\18\\lib\\windows\\clang_rt.asan-x86_64.lib" 
"C:\\Users\\mofio\\Documents\\rlc-infrastructure\\llvm-install-release\\lib\\clang\\18\\lib\\windows\\clang_rt.asan_cxx-x86_64.lib" 
"-wholearchive:C:\\Users\\mofio\\Documents\\rlc-infrastructure\\llvm-install-release\\lib\\clang\\18\\lib\\windows\\clang_rt.asan_cxx-x86_64.lib" 
"input" 
"asan_runtime.lib"

clang -fuse-ld=lld -v input runtime.lib
"C:\\Users\\mofio\\Documents\\rlc-infrastructure\\llvm-install-release\\bin\\lld-link" 
"-out:.\\test\\Output\\alternative_to_string.rl.tmp.exe" 
-defaultlib:libcmt 
-defaultlib:oldnames 
"-libpath:C:\\Users\\mofio\\Documents\\rlc-infrastructure\\llvm-install-release\\lib\\clang\\18\\lib\\windows" 
-nologo 
"input" 
"runtime.lib"
