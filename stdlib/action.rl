#
#Copyright 2024 Massimo Fioravanti
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#

import collections.vector
import serialization.to_byte_vector
import serialization.print

trait<FrameType, ActionType> ApplicableTo:
    fun apply(ActionType action, FrameType frame)

fun<FrameType, ActionType> can_apply_impl(ActionType action, FrameType frame) -> Bool:
    for alternative of action:
        using Type = type(alternative)
        if action is Type:
            if action is ApplicableTo<FrameType>:
                return can action.apply(frame)
    return false

fun<FrameType, ActionType> apply(ActionType action, FrameType frame) { can_apply_impl(action, frame) }: 
    for alternative of action:
        using Type = type(alternative)
        if action is Type:
            if action is ApplicableTo<FrameType>:
                action.apply(frame)


fun<FrameType, AllActionsVariant> parse_and_execute(FrameType state, AllActionsVariant variant, Vector<Byte> input, Int read_bytes):
    while read_bytes + 8 <= input.size():
        if from_byte_vector(variant, input, read_bytes):
            if can apply(variant, state):
                apply(variant, state)

fun<AllActionsVariant> parse_actions(AllActionsVariant variant, Vector<Byte> input, Int read_bytes) -> Vector<AllActionsVariant>:
    let to_return : Vector<AllActionsVariant>
    while read_bytes + 8 < input.size():
        if from_byte_vector(variant, input, read_bytes):
            to_return.append(variant)
    return to_return

# parses actions taking only one byte and by taking taking the reminder of the parsed number divided by the number of actions, so that no action is ever marked as invalid
fun<AllActionsVariant> parse_action_optimized(AllActionsVariant variant, Vector<Byte> input, Int read_bytes) -> Bool: 
    let num_actions : Byte
    for field of variant:
        num_actions = num_actions + byte(1)

    let counter : Byte
    if !from_byte_vector(counter, input, read_bytes):
        return false

    counter = counter % num_actions

    for field of variant:
        if counter == byte(0):
            using Type = type(field)
            let to_parse : Type
            if !from_byte_vector(to_parse, input, read_bytes):
                return false
            field = to_parse
            return true
        counter = counter - byte(1)

    return false

fun<AllActionsVariant> parse_actions(AllActionsVariant variant, Vector<Byte> input) -> Vector<AllActionsVariant>:
    return parse_actions(variant, input, 0)

fun<FrameType, AllActionsVariant> parse_and_execute(FrameType state, AllActionsVariant variant, Vector<Byte> input):
    parse_and_execute(state, variant, input, 0)


fun<FrameType, AllActionsVariant> gen_python_methods(FrameType state, AllActionsVariant variant):
    let state : FrameType
    let serialized = as_byte_vector(state)
    from_byte_vector(state, serialized)
    let x : AllActionsVariant
    apply(x, state)
    to_string(state)
    to_string(x)
    from_string(x, ""s)
    from_string(state, ""s)
    parse_actions(x, serialized)
    from_byte_vector(x, serialized)
    parse_action_optimized(x, serialized, 0)
    enumerate(x)

trait<T> Enumerable:
    fun enumerate(T obj, Vector<T> output)

fun enumerate(Bool b, Vector<Bool> output):
    output.append(true)
    output.append(false)

fun<T> _enumerate_impl(T obj, Int current_argument, Vector<T> out, Int num_args):
    let counter = 0
    for field of obj:
        if counter == current_argument:
            if field is Enumerable:
                using Type = type(field)
                let vec : Vector<Type>
                field.enumerate(vec)

                let next_current_argument = current_argument + 1
                let is_last_one = next_current_argument == num_args
                let counter_2 = 0
                while counter_2 != vec.size():
                    field = vec.get(counter_2)
                    if is_last_one:
                        out.append(obj)
                    else:
                        _enumerate_impl(obj, next_current_argument, out, num_args)
                    counter_2 = counter_2 + 1
            return
        counter = counter + 1


fun<T> enumerate(T obj) -> Vector<T>:
    let to_return : Vector<T>
    if obj is Enumerable:
        obj.enumerate(to_return)
    else if obj is Alternative:
        for field of obj:
            using Type = type(field)
            let field : Type 
            let alternatives = enumerate(field)
            let counter = 0
            while counter < alternatives.size():
                obj = alternatives.get(counter)
                to_return.append(obj) 
                counter = counter + 1
    else:
        let num_fields = 0
        for field of obj:
            num_fields = num_fields + 1
        if num_fields == 0:
            to_return.append(obj)
            return to_return
        _enumerate_impl(obj, 0, to_return, num_fields)

    return to_return
