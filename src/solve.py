def _is_valid_move(sudoku: list[list[int]], pos: (int, int), num: int) -> bool:
    """
    Private function to validating whether a move is valid or not.
    :param sudoku: Current sudoku on which the move is performed
    :param pos: x and y position of the desired move.
    :param num: Number which used in the move
    :return: Returns True if the move is valid according to sudoku rules
    """

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
    """
    Private function for detecting the first empty slot in a given Sudoku.
    :param sudoku: Sudoku in which the first empty slot should be found.
    :return: Returns the x and y position of the first empty slot
    """

    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                return i, j
    return None


def _solve_rec(sudoku: list[list[int]]) -> bool:
    """
    Helper function to solve the sudoku recursively using backpropagation.
    :param sudoku: The sudoku which should be solved
    :return: Returns true if the sudoku is solved.
    """

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
    """
    Solve a given sudoku.
    :param sudoku: The sudoku to be solved.
    :return: Returns the solved Sudoku.
    """

    _solve_rec(sudoku)
    return sudoku
