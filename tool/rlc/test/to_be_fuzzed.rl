# RUN: rlc %s -o %t -i %stdlib --fuzzer
# RUN: %t
import fuzzer.cpp_functions
import fuzzer.utils

act play() -> Play:
	let current = 0
	while current != 7:
	    act subact(Int x) { x < 8 }
	    current = x

	    actions:
	        act this(Int y)
	        current = y
	        act that()

fun main() -> Int:
	return 0
