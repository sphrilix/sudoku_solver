def _is_valid_move(sudoku: list[list[int]], pos: (int, int), num: int) -> bool:
    for i in range(9):
        if sudoku[pos[0]][i] == num and pos[1] != i:
            return False

    for j in range(9):
        if sudoku[j][pos[1]] == num and pos[0] != j:
            return False

    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if sudoku[i][j] == num and (i, j) != pos:
                return False

    return True


def _first_empty(sudoku: list[list[int]]) -> (int, int):
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                return i, j
    return None


def _solve_rec(sudoku: list[list[int]]) -> bool:
    empty_indices = _first_empty(sudoku)
    if not empty_indices:
        return True
    row, col = empty_indices
    for i in range(1, 10):
        if _is_valid_move(sudoku, empty_indices, i):
            sudoku[row][col] = i
            if _solve_rec(sudoku):
                return True
            sudoku[row][col] = 0
    return False


def solve(sudoku: list[list[int]]) -> list[list[int]]:
    _solve_rec(sudoku)
    return sudoku
