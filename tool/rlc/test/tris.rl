# RUN: rlc %s -o %t -i %stdlib 
# RUN: %t

import serialization.to_byte_vector

ent Board:
	Int[9] slots
	Bool playerTurn


fun get(Board b, Int x, Int y) -> Int:
	return b.slots[x + (y*3)]

fun set(Board b, Int x, Int y, Int val): 
	b.slots[x + (y * 3)] = val

fun full(Board b) -> Bool:
	let x = 0

	while x < 3:
		let y = 0
		while y < 3:
			if b.get(x, y) == 0:
				return false
			y = y + 1
		x = x + 1

	return true

fun three_in_a_line_player_row(Board b, Int player_id, Int row) -> Bool:
	return b.get(0, row) == b.get(1, row) and b.get(0, row) == b.get(2, row) and b.get(0, row) == player_id

fun three_in_a_line_player(Board b, Int player_id) -> Bool:
	let x = 0
	while x < 3:
		if b.get(x, 0) == b.get(x, 1) and b.get(x, 0) == b.get(x, 2) and b.get(x, 0) == player_id:
			return true

		if three_in_a_line_player_row(b, player_id, x):
			return true
		x = x + 1

	if b.get(0, 0) == b.get(1, 1) and b.get(0, 0) == b.get(2, 2) and b.get(0, 0) == player_id:
		return true

	if b.get(0, 2) == b.get(1, 1) and b.get(0, 2) == b.get(2, 0) and b.get(0, 2) == player_id:
		return true

	return false

fun current_player(Board board) -> Int:
	return int(board.playerTurn) + 1

fun next_turn(Board board):
	board.playerTurn = !board.playerTurn

act play() -> Tris:
	let board : Board
	let score = 10
	while !full(board):
		act mark(Int x, Int y) {
			x < 3,
			x >= 0,
			y < 3,
			y >= 0,
			board.get(x, y) == 0
		}
		score = score - 1

		board.set(x, y, board.current_player())

		if board.three_in_a_line_player(board.current_player()):
			return

		board.next_turn()

fun gen_printer_parser():
	let state = play()
	let serialized = state.as_byte_vector()
	state.from_byte_vector(serialized)

fun main() -> Int:
	let game = play()
	game.mark(0, 0)
	if game.board.full():
		return 1
	game.mark(1, 0)
	if game.board.full():
		return 2
	game.mark(1, 1)
	if game.board.full():
		return 3
	game.mark(2, 0)
	if game.board.full():
		return 4
	game.mark(2, 2)
	if game.board.full():
		return 5
	return int(game.board.three_in_a_line_player(1)) - 1

