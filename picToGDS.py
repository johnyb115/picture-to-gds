# -*- coding: utf-8 -*-
"""Convert an image file to a GDS file
"""

import cv2
import numpy as np
import gdspy

import argparse


def minmax(v):
    if v > 255:
        v = 255
    if v < 0:
        v = 0
    return v


def main(fileName, sizeOfTheCell, layerNum, isDither, threshold_offset, scale, invert=False):
    """Convert an image file (fileName) to a GDS file
    """
    print("Converting an image file to a GDS file..")

    # ---- Load image safely ----
    img_raw = cv2.imread(fileName)
    if img_raw is None:
        raise FileNotFoundError(f"Could not read image from path: {fileName}")

    # Fix scale if the caller passes 0 or negative
    if scale is None or scale <= 0:
        print(f"Warning: invalid scale={scale}, using 1.0 instead.")
        scale = 1.0

    # Read an image file with scaling
    img = cv2.resize(img_raw, dsize=None, fx=scale, fy=scale)

    width = img.shape[1]
    height = img.shape[0]
    print(f"width:{width}")
    print(f"height:{height}")

    # Convert an image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Optional Floyd–Steinberg dithering (operates in-place on gray)
    if isDither:
        # Floyd–Steinberg dithering
        # https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
        for y in range(0, height - 1):
            for x in range(1, width - 1):
                old_p = gray[y, x]
                new_p = np.round(old_p / 255.0) * 255
                gray[y, x] = new_p
                error_p = old_p - new_p
                gray[y, x + 1] = minmax(gray[y, x + 1] + error_p * 7 / 16.0)
                gray[y + 1, x - 1] = minmax(
                    gray[y + 1, x - 1] + error_p * 3 / 16.0
                )
                gray[y + 1, x] = minmax(gray[y + 1, x] + error_p * 5 / 16.0)
                gray[y + 1, x + 1] = minmax(
                    gray[y + 1, x + 1] + error_p * 1 / 16.0
                )

    # --- Thresholding with Otsu + offset---
    # First, compute Otsu threshold on the (possibly dithered) grayscale image
    T_otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # Apply offset from user (can be negative or positive)
    T_effective = T_otsu + threshold_offset
    # Clip to valid range
    T_effective = float(np.clip(T_effective, 0, 255))
    print(f"Otsu threshold: {T_otsu}, offset: {threshold_offset}, effective: {T_effective}")

    # Now apply a standard binary threshold with the adjusted threshold
    _, binaryImage = cv2.threshold(gray, T_effective, 255, cv2.THRESH_BINARY)

    # Fill orthological corner
    for x in range(width - 1):
        for y in range(height - 1):
            if (
                binaryImage[y, x] == 0
                and binaryImage[y + 1, x] == 255
                and binaryImage[y, x + 1] == 255
                and binaryImage[y + 1, x + 1] == 0
            ):
                binaryImage[y + 1, x] = 0
            elif (
                binaryImage[y, x] == 255
                and binaryImage[y + 1, x] == 0
                and binaryImage[y, x + 1] == 0
                and binaryImage[y + 1, x + 1] == 255
            ):
                binaryImage[y + 1, x + 1] = 0

    # Invert image if requested (swap black and white)
    if invert:
        binaryImage = 255 - binaryImage

    # Output image.bmp (preview of the binary pattern)
    cv2.imwrite("image.bmp", binaryImage)

    # The GDSII file is called a library, which contains multiple cells.
    lib = gdspy.GdsLibrary()
    gdspy.current_library = gdspy.GdsLibrary()

    # Geometry must be placed in cells.
    unitCell = lib.new_cell("CELL")
    square = gdspy.Rectangle((0.0, 0.0), (1.0, 1.0), layer=int(layerNum))
    unitCell.add(square)

    grid = lib.new_cell("GRID")
    ys = 0.0
    for y in range(height):
        xs = 0.0
        for x in range(width):
            b = binaryImage[y, x]
            if b == 0:
                grid.add(
                    gdspy.CellReference(
                        unitCell, (xs * sizeOfTheCell, ys * sizeOfTheCell)
                    )
                )

            xs += 1.0
        ys += 1.0

    # Scaling needs to be done to align the minimum feature size to sizeOfTheCell
    top = lib.new_cell("TOP")
    scaleFactor = sizeOfTheCell
    scaledGrid = gdspy.CellReference(grid, (0.0, 0.0), magnification=scaleFactor)

    top.add(scaledGrid)
    lib.write_gds("image.gds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fileName", type=str, help="name of the input image file")
    parser.add_argument(
        "sizeOfTheCell",
        type=float,
        help="size of the unit-cells (minimum width and space) [um]",
    )
    parser.add_argument(
        "layerNum", type=int, help="layer number of the output GDSII file"
    )
    parser.add_argument("--scale", default=1.0, type=float, help="scale")
    parser.add_argument("-d", action="store_true", help="Floyd–Steinberg dithering")
    parser.add_argument(
        "--threshold_offset",
        type=float,
        default=0.0,
        help=(
            "Offset added to the Otsu threshold (negative values make more pixels black, "
            "positive values make more pixels white)."
        ),
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert binary image (black ↔ white) before creating GDS",
    )
    args = parser.parse_args()

    main(
        args.fileName,
        args.sizeOfTheCell,
        args.layerNum,
        args.d,
        args.threshold_offset,
        args.scale,
        args.invert,
    )
