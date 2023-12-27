# RUN: rlc %s -o %t -i %stdlib 
# RUN: %t

fun<T> returnDoubleIfInt(T a) -> T:
	if a is Int:
		return a + a
	return a

fun main() -> Int:
	return returnDoubleIfInt(2) - 4
