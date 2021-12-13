import io
import cv2
import numpy as np
import requests
import json
import re

"""
The API key from https://ocr.space
"""
API_KEY: str = ""

"""
Integer which indicates which engine to be used.
"""
OCR_ENGINE: int = 2


def _img_to_str(img: np.ndarray) -> str:
    """
    Send a request to ocr.space API and return the result.
    :param img: The image to be sent to the API.
    :return: Returns the result of the API request.
    """

    _, compressed_image = cv2.imencode(".jpg", img)
    file_bytes = io.BytesIO(compressed_image)
    payload: json = {"apikey": API_KEY,
                     "ocrengine": OCR_ENGINE}
    files: json = {"temp.jpg": file_bytes}
    resp: requests.Response = requests.post("https://api.ocr.space/parse/image",
                                            data=payload,
                                            files=files)
    return resp.json()['ParsedResults'][0]['ParsedText']


def _concat_numbers(images: list[np.ndarray]) -> (np.ndarray, list):
    """
    Concatenate number to one image to reduce API request.
    :param images: List of images to be concatenated.
    :return: Returns the concatenated image and the indexes of the concatenated images.
    """

    concat: list = []
    indexes: list = []
    for index, img in enumerate(images):

        # Only concatenate images which contains numbers. Therefore a little heuristic only concatenate images where
        # more then the 1500 px are different from the background.
        if (img != img.min()).sum() > 1500:
            concat.append(img)
            indexes.append(index)
    return cv2.hconcat(concat), indexes


def _do_numbers_back(numbers: str, indexes: list) -> list[list[int]]:
    """
    Merge back recognized numbers to their indexes.
    :param numbers: Recognized numbers as a string.
    :param indexes: Indexes of the recognized numbers.
    :return: Returns a 2 dimensional list, which represents the sudoku.
    """

    sudoku: list[list] = [[], [], [], [], [], [], [], [], []]

    # Remove everything which is not a number.
    numbers = re.sub(r"\D", "", numbers)
    for x in range(9):
        for y in range(9):
            if 9 * x + y in indexes:
                sudoku[x].append(int(numbers[indexes.index(9 * x + y)]))
            else:
                sudoku[x].append(0)
    return sudoku


def recognize_sudoku(images: list[np.ndarray]) -> list[list[int]]:
    """
    Recognize a list of images of cell in a sudoku.
    :param images: List of the cells of an image from sudoku.
    :return: Returns a 2-dim. list which represents a sudoku.
    """
    image, indexes = _concat_numbers(images)
    str_of_numbers: str = _img_to_str(image)
    return _do_numbers_back(str_of_numbers, indexes)
