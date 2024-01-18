import collections.vector
import fuzzer.cpp_functions

fun RLC_Fuzzer_init_available_subactions() -> Vector<Int>:
    let res : Vector<Int>
    return res

fun RLC_Fuzzer_add_available_subaction(Vector<Int> available_subactions, Int subactionID):
    available_subactions.append(subactionID)

fun RLC_Fuzzer_pick_subaction(Vector<Int> available_subactions) -> Int:
    let index = RLC_Fuzzer_get_input( available_subactions.size())
    return available_subactions.get(index)
