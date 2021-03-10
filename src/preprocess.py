import operator
import cv2
import numpy as np


def _main_feature_extraction(img: np.ndarray, skip_dilate: bool = True) -> np.ndarray:
    """
    Uses a blurring function, adaptive thresholding and dilation to expose
    the main features of an image.
    :param img: The image which should be processed
    :param skip_dilate: Boolean whether dilate should be skipped or not
    :return: Returns a black and white image of the input.
    """

    # Gaussian blur with a kernel size (height, width) of 9.
    # Note that kernel sizes must be positive and odd and the kernel must be
    # square.
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Adaptive threshold using 11 nearest neighbour pixels
    proc = cv2.adaptiveThreshold(
        proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Invert colours, so grid lines have non-zero pixel values.
    proc = cv2.bitwise_not(proc)

    # Dilating the image helps!
    if not skip_dilate:
        # Dilate the image to increase the size of the grid lines.
        kernel = np.array(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8
        )
        proc = cv2.dilate(proc, kernel)

    return proc


def _find_corners_of_largest_polygon(img: np.ndarray) -> list:
    """
    Finds the 4 extreme corners of the largest contour in the image.
    :param img: The image where the 4 extreme corners should be searched.
    :return: The position of the four extreme corners.
    """

    # Find contours
    contours, h = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort by area, descending
    contours = sorted(
        contours, key=cv2.contourArea, reverse=True
    )
    polygon = contours[0]  # Largest image

    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value
    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1),
    )
    top_left, _ = min(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1),
    )
    bottom_left, _ = min(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1),
    )
    top_right, _ = max(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1),
    )

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [
        polygon[top_left][0],
        polygon[top_right][0],
        polygon[bottom_right][0],
        polygon[bottom_left][0],
    ]


def _crop_and_warp(img: np.ndarray, crop_rect: list) -> np.ndarray:
    """
    Crops and warps a rectangular section from an image into a square of
    similar size.
    :param img: The image to be cropped and warped.
    :param crop_rect: The positions of the corners.
    :return: Returns the cropped and warped image.
    """

    # Rectangle described by top left, top right, bottom right and bottom left
    # points.
    top_left, top_right, bottom_right, bottom_left = (
        crop_rect[0],
        crop_rect[1],
        crop_rect[2],
        crop_rect[3],
    )

    # Explicitly set the data type to float32 or `getPerspectiveTransform`
    # will throw an error.
    src = np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype="float32"
    )

    def distance_between(x, y):
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    # Get the longest side in the rectangle
    side = max(
        [
            distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right),
        ]
    )

    # Describe a square with side of the calculated length, this is the new
    # perspective we want to warp to.
    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype="float32",
    )

    # Gets the transformation matrix for skewing the image to fit a square by
    # comparing the 4 before and after points.
    m = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    warp = cv2.warpPerspective(img, m, (int(side), int(side)))
    return warp


def _split_up(img: np.ndarray) -> list:
    """
    Split up the picture in 81 even partial images.
    :param img: The image to be split up.
    :return: Returns a list of 81 even parts of the input Image.
    """

    y_offset: int = int(img.shape[0] / 9)
    x_offset: int = int(img.shape[1] / 9)
    x_threshold: int = 0
    y_threshold: int = 0
    split_board: list = []
    for i in range(9):
        for j in range(9):
            split_board.append(cv2.resize(_remove_board_borders(img[y_threshold: y_threshold + y_offset,
                                                                x_threshold: x_threshold + x_offset]), (64, 64)))
            x_threshold += x_offset
        x_threshold = 0
        y_threshold += y_offset
    return split_board


def _remove_board_borders(img: np.ndarray) -> np.ndarray:
    """
    Removes 20% of the image on each side.
    :param img: The image where the borders should be removed.
    :return: Returns the image with the removed borders.
    """

    y_offset: int = int(img.shape[0] / 5)
    x_offset: int = int(img.shape[1] / 5)
    return img[y_offset: -y_offset, x_offset: -x_offset]


def pre_process_image(img: np.ndarray) -> list[np.ndarray]:
    """
    Preprocesses a image to split up in 81 even partial images, where each represents a cell from the sudoku.
    :param img: The image to split up.
    :return: Return a list of 81 images where each one is a cell from the sudoku.
    """

    feature_extracted_img: np.ndarray = _main_feature_extraction(img)
    return _split_up(_crop_and_warp(feature_extracted_img, _find_corners_of_largest_polygon(feature_extracted_img)))
