import spectral
import numpy as np
import cv2
from time import time
import os


def find_nearest_band(wavelengths, target_wavelength):
    """
    Find the index of the band with the wavelength closest to the target wavelength.

    Parameters:
    - wavelengths: List or array of wavelengths (as strings or floats).
    - target_wavelength: The target wavelength (float).

    Returns:
    - index: Index of the band closest to the target wavelength.
    """
    wavelengths_float = np.array([float(w) for w in wavelengths])
    return np.argmin(np.abs(wavelengths_float - target_wavelength))


def resize_for_display(image, max_height=900):
    """
    Resize the image to fit within the specified maximum height while maintaining aspect ratio.

    Parameters:
    - image: The input image to be resized.
    - max_height: The maximum height for the resized image.

    Returns:
    - resized: The resized image.
    - scale: The scaling factor applied to the original image.
    """
    h, w = image.shape[:2]
    if h <= max_height:
        return image, 1.0

    scale = max_height / h
    resized = cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def select_coordinates(image, image_path: str, file_name: str):
    """
    Allow the user to select a set of rectangular coordinates on the given image.

    Parameters:
    - image: The input image on which to select the coordinates.
    - image_path: The file path of the image.
    - file_name: A string identifier for saving the coordinates.

    Returns:
    - coordinates: A tuple (x, y, w, h) representing the selected coordinates.
    """
    # Resize image for display if it's too tall
    display_image, scale = resize_for_display(image, max_height=900)

    # Select ROI on resized image
    coordinates = cv2.selectROI(
        file_name.upper(), display_image, fromCenter=False, showCrosshair=True
    )
    cv2.destroyAllWindows()

    # Scale coordinates back to original image dimensions
    scaled_coordinates = (
        int(coordinates[0] / scale),
        int(coordinates[1] / scale),
        int(coordinates[2] / scale),
        int(coordinates[3] / scale),
    )

    top_left = (scaled_coordinates[0], scaled_coordinates[1])
    top_right = (scaled_coordinates[0] + scaled_coordinates[2], scaled_coordinates[1])
    bottom_right = (
        scaled_coordinates[0] + scaled_coordinates[2],
        scaled_coordinates[1] + scaled_coordinates[3],
    )
    bottom_left = (scaled_coordinates[0], scaled_coordinates[1] + scaled_coordinates[3])
    calibration_coordinates = [top_left, top_right, bottom_right, bottom_left]

    # Save in corner format matching notebook format
    save_path = os.path.splitext(image_path)[0] + "_" + file_name + "_coordinates.npy"
    with open(save_path, "wb") as f:
        np.save(f, calibration_coordinates)

    return calibration_coordinates


def get_rois(dark_coords, white_coords):
    """
    Process dark and white calibration coordinates to create a bounding box.

    Parameters:
    - dark_coords: List of corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    - white_coords: List of corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

    Returns:
    - dark_coords: A numpy array of corner coordinates for dark calibration.
    - white_coords: A numpy array of corner coordinates for white calibration.
    - bounding_box: A numpy array representing the bounding box coordinates.
    """

    dark_coords = np.array(dark_coords)
    white_coords = np.array(white_coords)

    # Create bounding box from white top edge and dark bottom edge
    bounding_box = np.array(
        [
            [white_coords[0][0], white_coords[0][1]],  # TL from white
            [white_coords[1][0], white_coords[1][1]],  # TR from white
            [white_coords[3][0], dark_coords[3][1]],  # BL: white x, dark y
            [white_coords[2][0], dark_coords[2][1]],  # BR: white x, dark y
        ]
    )
    return white_coords, dark_coords, bounding_box


def crop_image(image, bounding_box):
    """
    Crop the input image using the provided bounding box.

    Parameters:
    - image: The input image to be cropped.
    - bounding_box: A 2D array of corner coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].

    Returns:
    - cropped_image: The cropped image.
    """
    # Bounding_box format: [[TL_x, TL_y], [TR_x, TR_y], [BR_x, BR_y], [BL_x, BL_y]]
    rows_range = (
        bounding_box[0, 1],
        bounding_box[2, 1],
    )  # Top to Bottom (y coordinates)

    cols_range = (
        bounding_box[0, 0],
        bounding_box[1, 0],
    )  # Left to Right (x coordinates)

    cropped_image = image[
        rows_range[0] : rows_range[1], cols_range[0] : cols_range[1], :
    ]

    return cropped_image


def normalize_image(image):
    """
    Normalize the input image to the range [0, 1].

    Parameters:
    - image: The input image to be normalized.

    Returns:
    - normalized_image: The normalized image.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def normalize_boxes(
    white_calibration_coordinates, dark_calibration_coordinates, bounding_box
):
    """
    Normalize calibration coordinates relative to the bounding box.

    Parameters:
    - white_calibration_coordinates: 3D array of white calibration corner coordinates.
    - dark_calibration_coordinates: 3D array of dark calibration corner coordinates.
    - bounding_box: 2D array of bounding box coordinates.

    Returns:
    - white_calibration_relative_coordinates: Numpy array of normalized white calibration coordinates.
    - dark_calibration_relative_coordinates: Numpy array of normalized dark calibration coordinates.
    """

    white_calibration_coordinates = np.array(white_calibration_coordinates)
    dark_calibration_coordinates = np.array(dark_calibration_coordinates)
    bounding_box = np.array(bounding_box)

    # Flatten to 2D if needed (from [[x, y]] format to [x, y] format)
    if white_calibration_coordinates.ndim == 3:
        white_calibration_coordinates = white_calibration_coordinates.reshape(-1, 2)
    if dark_calibration_coordinates.ndim == 3:
        dark_calibration_coordinates = dark_calibration_coordinates.reshape(-1, 2)

    # Reorder wc and dc to match order (TL, TR, BL, BR)
    reorder_idx = [0, 1, 3, 2]
    white_calibration_coordinates = white_calibration_coordinates[reorder_idx]
    dark_calibration_coordinates = dark_calibration_coordinates[reorder_idx]

    # Offset (top-left of big_rect)
    offset = bounding_box[0]

    # Relative coordinates
    white_calibration_relative_coordinates = white_calibration_coordinates - offset
    dark_calibration_relative_coordinates = dark_calibration_coordinates - offset

    # Force ONLY dark to span full width of big_rect (matches notebook behavior)
    left_x = 0
    right_x = bounding_box[1][0] - bounding_box[0][0]

    dark_calibration_relative_coordinates[0][0] = left_x  # TL
    dark_calibration_relative_coordinates[1][0] = right_x  # TR
    dark_calibration_relative_coordinates[2][0] = left_x  # BL
    dark_calibration_relative_coordinates[3][0] = right_x  # BR

    return white_calibration_relative_coordinates, dark_calibration_relative_coordinates


def replicate_and_extend(image, coordinates, target_shape):
    """
    Replicate and extend the calibration region vertically to match the target shape.
    This matches the notebook implementation (extract_and_replicate).

    Parameters:
    - image: The input image to be replicated and extended.
    - coordinates: The coordinates used for replication (4x2 array).
    - target_shape: The desired shape (height, width, bands) for the output image.

    Returns:
    - replicated_image: The replicated image matching target_shape vertically.
    """
    rows_range = (int(coordinates[0, 1]), int(coordinates[2, 1]))  # Top to Bottom
    cols_range = (int(coordinates[0, 0]), int(coordinates[1, 0]))  # Left to Right

    # Ensure coordinates are within image bounds
    rows_range = (max(0, rows_range[0]), min(image.shape[0], rows_range[1]))
    cols_range = (max(0, cols_range[0]), min(image.shape[1], cols_range[1]))

    # Extract the calibration region
    region = image[rows_range[0] : rows_range[1], cols_range[0] : cols_range[1], :]

    # Replicate vertically to match target height
    height_diff = rows_range[1] - rows_range[0]
    num_repeats = target_shape[0] // height_diff
    replicated_image = np.concatenate([region] * num_repeats, axis=0)

    # Add remaining rows if needed
    if replicated_image.shape[0] < target_shape[0]:
        additional_rows = target_shape[0] - replicated_image.shape[0]
        last_piece = region[:additional_rows, :, :]
        replicated_image = np.concatenate([replicated_image, last_piece], axis=0)

    return replicated_image


def white_dark_calibrate(raw_hdr, white_hdr, dark_hdr, outdir):
    """
    Perform white-dark calibration on a hyperspectral image cube.

    Parameters:
    - raw_hdr: Path to the raw hyperspectral image header file (.hdr)
    - white_hdr: Path to the white reference hyperspectral image header file (.hdr)
    - dark_hdr: Path to the dark reference hyperspectral image header file (.hdr)
    - outdir: Directory to save the calibrated output cube

    Returns:
    - calibrated_cube: The calibrated hyperspectral image cube
    """

    # Load hyperspectral image cubes
    raw_cube = spectral.open_image(raw_hdr).load()
    white_cube = spectral.open_image(white_hdr).load()
    dark_cube = spectral.open_image(dark_hdr).load()

    # Ensure all cubes have the same dimensions
    if raw_cube.shape != white_cube.shape or raw_cube.shape != dark_cube.shape:
        raise ValueError("Input cubes must have the same dimensions")

    # Perform white-dark calibration
    calibrated_cube = (raw_cube - dark_cube) / (white_cube - dark_cube)
    calibrated_cube = np.clip(calibrated_cube, 0, 1)  # Clip values to [0, 1]

    # Save the calibrated cube
    calibrated_hdr = outdir + f"/calibrated_cube_{int(time())}.hdr"
    spectral.envi.save_image(
        calibrated_hdr, calibrated_cube, dtype=np.float32, interleave="bil"
    )

    print(f"Calibrated cube saved to {calibrated_hdr}")
    return calibrated_cube


def white_dark_calibrate_from_rois(raw_hdr, white_roi, dark_roi, outdir):
    """
    Perform white-dark calibration on a hyperspectral image cube using specified ROIs.

    Parameters:
    - raw_hdr: Path to the raw hyperspectral image header file (.hdr)
    - white_roi: List of corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    - dark_roi: List of corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    - outdir: Directory to save the calibrated output cube

    Returns:
    - calibrated: The calibrated hyperspectral image cube
    """
    cube = spectral.open_image(raw_hdr).load()

    white_calibration_coordinates, dark_calibration_coordinates, bounding_box = (
        get_rois(dark_roi, white_roi)
    )
    cropped = crop_image(cube, bounding_box)

    white_calibration_relative_coordinates, dark_calibration_relative_coordinates = (
        normalize_boxes(
            white_calibration_coordinates, dark_calibration_coordinates, bounding_box
        )
    )

    white_calibration_reference = replicate_and_extend(
        cropped, white_calibration_relative_coordinates, cropped.shape
    )
    dark_calibration_reference = replicate_and_extend(
        cropped, dark_calibration_relative_coordinates, cropped.shape
    )

    assert (
        cropped.shape
        == white_calibration_reference.shape
        == dark_calibration_reference.shape
    ), "Shapes of cropped image and calibration references do not match."

    calibrated = (cropped - dark_calibration_reference) / (
        white_calibration_reference - dark_calibration_reference
    )
    calibrated = np.clip(calibrated, 0, 1)

    out_hdr = os.path.join(
        outdir, f"{os.path.basename(raw_hdr)[:-4]}_calibrated_roi.hdr"
    )
    spectral.envi.save_image(
        out_hdr, calibrated, dtype=np.float32, interleave="bil", force=True
    )
    print(f"Calibrated cube saved to {out_hdr}")
    return calibrated
