import collections.vector
import fuzzer.cpp_functions

fun fuzzer_init_available_subactions() -> Vector<Int>:
    let res : Vector<Int>
    return res

fun fuzzer_add_available_subaction(Vector<Int> available_subactions, Int subactionID):
    available_subactions.append(subactionID)

fun fuzzer_pick_subaction(Vector<Int> available_subactions) -> Int:
    let index = fuzzer_get_input( available_subactions.size())
    return available_subactions.get(index)

fun fuzzer_clear_available_subactions(Vector<Int> available_subactions):
    available_subactions.clear()
