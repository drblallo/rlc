# RUN: rlc %s -o %t -i %stdlib 
# RUN: %t

ext fun something_extern() -> Int

fun main() -> Int:
	return 0
