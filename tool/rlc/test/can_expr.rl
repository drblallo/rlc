fun func(Int x) -> Int {x < 3}:
    return 0

fun func2() -> Int:
    return 0

fun main() -> Int:
    if can func(4):
        return 1
    if !can func2():
        return 1
    return 0
