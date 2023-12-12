import collections.vector
import fuzzer.cpp_functions

fun init_available_subactions() -> Vector<Int>:
    let res : Vector<Int>
    return res

fun add_available_subaction(Vector<Int> available_subactions, Int subactionID):
    available_subactions.append(subactionID)

fun pick_subaction(Vector<Int> available_subactions) -> Int:
    let index = RLC_Fuzzer_getInput( available_subactions.size)
    return available_subactions.get(index)
