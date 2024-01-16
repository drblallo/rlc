# RUN: rlc %s -o %t -i %stdlib --fuzzer
# RUN: %t
import fuzzer.cpp_functions
import fuzzer.utils

fun crash_on_five(Int input) -> Int {input != 1247}:
	return 0

act play() -> Play:
	let current = 0
	while current != 7:
	    act subact(Int x) {x > 5, x < 15}
	    current = x

	    actions:
	        act this(Int y, Int z) {z < 3, z > 0, y < 14, y >= 0 }
	        current = y
	        act that(Int a) {a >= 0, a < 10000}
	        crash_on_five(a)


