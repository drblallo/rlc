# RUN: rlc %s -o %t -i %stdlib --sanitize -O2 -g
# RUN: %t

import string

fun main() -> Int:
    let x : String
    return x.size()


