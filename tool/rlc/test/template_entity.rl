# RUN: rlc %s -o %t -i %stdlib 
# RUN: %t

ent<T> TemplatedEntity:
	T asd

fun main() -> Int:
	let var : TemplatedEntity<Int>
	var.asd = 2	
	return var.asd - 2
