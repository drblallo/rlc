# RUN: rlc %s -o %t -i %stdlib 
# RUN: %t

enum Asd:
	first	
	second	

fun main() -> Int:
	let asd : Asd
	asd = Asd::second
	return asd.value - 1
