import cv2
import numpy as np
import pathlib
from preprocess import pre_process_image
from recognize import recognize_sudoku


def main(img_path: pathlib.Path) -> list[list[str]]:
    """
    Main function for recognizing a sudoku image. Just give the path of an image of a sudoku and it will return a 2-dim.
    which represents a sudoku.
    :param img_path: Path of the image.
    :return: Returns a 2-dim. list which represents a sudoku.
    """

    img = cv2.imread(str(img_path), 0)
    return recognize_sudoku(pre_process_image(img))


def print_sudoku(board: list[list]) -> None:
    """
    Just a print function for debugging.
    :param board: Takes a 2-dim. list which represents a sudoku.
    """

    print("-" * 37)
    for i, row in enumerate(board):
        print(("|" + " {}   {}   {} |" * 3).format(*[x if x != 0 else " " for x in row]))
        if i == 8:
            print("-" * 37)
        elif i % 3 == 2:
            print("|" + "---+" * 8 + "---|")
        else:
            print("|" + "   +" * 8 + "   |")


if __name__ == "__main__":
    print_sudoku(main(pathlib.Path("/img.png")))
