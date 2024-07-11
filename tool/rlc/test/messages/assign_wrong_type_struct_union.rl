# RUN: rlc %s -o %t -i %stdlib --print-ir-on-failure=false 2>&1 --expect-fail | FileCheck %s

# CHECK: 14:9: error: Could not find matching function assign(Int | Float,B)

cls A:
    Int | Float c

cls B:
    Int b

fun asd(): 
    let a : A
    let b : B
    a.c = b
