import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QRadioButton,
    QGroupBox,
    QMessageBox,
    QTextEdit,
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
from preprocessing.calibration import (
    white_dark_calibrate,
    white_dark_calibrate_from_rois,
    select_coordinates,
    find_nearest_band,
    normalize_image,
)


class CalibrationWorker(QThread):
    status = pyqtSignal(str)
    image_ready = pyqtSignal(np.ndarray, list)  # image and wavelengths
    output = pyqtSignal(str)  # for print output

    def __init__(self, raw, white, dark, outdir):
        super().__init__()
        self.raw = raw
        self.white = white
        self.dark = dark
        self.outdir = outdir

    def run(self):
        self.output.emit("Running calibration...\n")
        calibrated = white_dark_calibrate(self.raw, self.white, self.dark, self.outdir)

        # Extract wavelengths from the raw image header
        import spectral

        raw_img = spectral.open_image(self.raw)
        wavelengths = raw_img.metadata.get("wavelength", [])
        wavelengths = [float(w) for w in wavelengths] if wavelengths else []

        self.image_ready.emit(calibrated, wavelengths)
        self.status.emit("Calibration done ✔")
        self.output.emit("Calibration done ✔\n")


class CalibrationWorkerROI(QThread):
    status = pyqtSignal(str)
    image_ready = pyqtSignal(np.ndarray, list)  # image and wavelengths
    output = pyqtSignal(str)  # for print output

    def __init__(self, raw, white_roi, dark_roi, outdir):
        super().__init__()
        self.raw = raw
        self.white_roi = white_roi
        self.dark_roi = dark_roi
        self.outdir = outdir

    def run(self):
        import io
        import sys

        self.output.emit("Running ROI-based calibration...\n")

        # Redirect stdout to capture prints
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            calibrated = white_dark_calibrate_from_rois(
                self.raw, self.white_roi, self.dark_roi, self.outdir
            )
        finally:
            # Capture output and restore stdout
            output_text = sys.stdout.getvalue()
            sys.stdout = old_stdout
            if output_text:
                self.output.emit(output_text)

        # Extract wavelengths from the raw image header
        import spectral

        raw_img = spectral.open_image(self.raw)
        wavelengths = raw_img.metadata.get("wavelength", [])
        wavelengths = [float(w) for w in wavelengths] if wavelengths else []

        self.image_ready.emit(calibrated, wavelengths)
        self.status.emit("Calibration done ✔")
        self.output.emit("Calibration done ✔\n")


class CalibApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FX White–Dark Calibration")

        self.raw_hdr = None
        self.white_hdr = None
        self.dark_hdr = None
        self.white_roi = None
        self.dark_roi = None

        layout = QVBoxLayout()

        # Calibration mode selection
        self.mode_full = QRadioButton("Use WHITE & DARK cubes")
        self.mode_roi = QRadioButton("Select WHITE & DARK regions from RAW")
        self.mode_full.setChecked(True)

        mode_group = QGroupBox("Calibration mode")
        mode_layout = QVBoxLayout()
        mode_layout.addWidget(self.mode_full)
        mode_layout.addWidget(self.mode_roi)
        mode_group.setLayout(mode_layout)

        layout.addWidget(mode_group)

        self.btn_raw = QPushButton("Select RAW cube (.hdr)")
        self.btn_white = QPushButton("Select WHITE cube (.hdr)")
        self.btn_dark = QPushButton("Select DARK cube (.hdr)")

        self.btn_run = QPushButton("Run calibration")
        self.btn_roi = QPushButton("Select WHITE & DARK ROIs from RAW")

        self.btn_raw.clicked.connect(lambda: self.select_file("raw"))
        self.btn_white.clicked.connect(lambda: self.select_file("white"))
        self.btn_dark.clicked.connect(lambda: self.select_file("dark"))
        self.btn_run.clicked.connect(self.run_calibration)
        self.btn_roi.clicked.connect(self.select_rois)

        layout.addWidget(self.btn_raw)
        layout.addWidget(self.btn_white)
        layout.addWidget(self.btn_dark)
        layout.addWidget(self.btn_roi)
        layout.addWidget(self.btn_run)

        # Output text display
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_text)

        self.setLayout(layout)
        self.mode_full.toggled.connect(self.update_mode)
        self.update_mode()

    def update_mode(self):
        """Enable/disable buttons based on selected mode."""
        full = self.mode_full.isChecked()
        self.btn_white.setEnabled(full)
        self.btn_dark.setEnabled(full)
        self.btn_roi.setEnabled(not full)

    def select_file(self, kind):
        """Select HDR file for RAW, WHITE, or DARK."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HDR file", "", "HDR Files (*.hdr)"
        )
        if not path:
            return
        setattr(self, f"{kind}_hdr", path)
        self.append_output(f"{kind.upper()} selected: {path}")

    def select_rois(self):
        """Select WHITE and DARK ROIs from the RAW image."""
        if not self.raw_hdr:
            QMessageBox.warning(self, "Missing RAW", "Select RAW cube first")
            return

        import spectral
        import io
        import sys

        cube = spectral.open_image(self.raw_hdr).load()

        # simple RGB preview
        rgb = cube[:, :, [60, 30, 10]]
        rgb_min = rgb.min()
        rgb_max = rgb.max()
        if rgb_max > rgb_min:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
        else:
            rgb = np.zeros_like(rgb)

        # Capture stdout for select_coordinates calls
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            self.white_roi = select_coordinates(rgb, self.raw_hdr, "white")
            print("White Coordinates: ", self.white_roi)
            self.dark_roi = select_coordinates(rgb, self.raw_hdr, "dark")
            print("Dark Coordinates: ", self.dark_roi)
        finally:
            output_text = sys.stdout.getvalue()
            sys.stdout = old_stdout
            if output_text:
                self.append_output(output_text)

        self.append_output("WHITE & DARK ROIs selected")

    def append_output(self, text):
        """
        Append text to the output display.

        Parameters:
        - text (str): Text to append.

        """
        self.output_text.append(text.rstrip())

    def run_calibration(self):
        """Run the calibration process based on selected mode and inputs."""
        if not self.raw_hdr:
            self.append_output("ERROR: Missing RAW cube")
            return

        # Create calibrated folder if it doesn't exist
        raw_dir = os.path.dirname(self.raw_hdr)
        output_dir = os.path.join(raw_dir, "calibrated")
        os.makedirs(output_dir, exist_ok=True)

        # MODE A: full cubes
        if self.mode_full.isChecked():
            if None in (self.white_hdr, self.dark_hdr):
                self.append_output("ERROR: Missing WHITE/DARK cubes")
                return

            self.worker = CalibrationWorker(
                self.raw_hdr, self.white_hdr, self.dark_hdr, output_dir
            )

        # MODE B: ROI-based
        else:
            if self.white_roi is None or self.dark_roi is None:
                self.append_output("ERROR: Select WHITE & DARK ROIs first")
                return

            self.worker = CalibrationWorkerROI(
                self.raw_hdr, self.white_roi, self.dark_roi, output_dir
            )

        self.worker.status.connect(self.append_output)
        self.worker.image_ready.connect(self.display_calibrated_image)
        self.worker.output.connect(self.append_output)
        self.worker.start()

    def display_calibrated_image(self, image, wavelengths):
        """
        Display the calibrated hyperspectral image as RGB.

        Parameters:
        - image (np.ndarray): The hyperspectral data cube.
        - wavelengths (list): List of wavelength values corresponding to bands.

        Returns:
        - None
        """
        try:
            # Handle empty wavelengths
            if not wavelengths or len(wavelengths) == 0:
                # Fallback to simple RGB using band indices
                rgb = image[:, :, [60, 30, 10]]
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                rgb = (rgb * 255).astype(np.uint8)
                display_image, _ = self.resize_for_display_static(rgb, max_height=900)
                cv2.imshow("Calibrated Image Preview", display_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return

            # Find nearest bands to RGB wavelengths
            red_band_index = find_nearest_band(wavelengths, 660)
            green_band_index = find_nearest_band(wavelengths, 550)
            blue_band_index = find_nearest_band(wavelengths, 450)

            # Extract the RGB bands
            red_band = image[:, :, red_band_index]
            green_band = image[:, :, green_band_index]
            blue_band = image[:, :, blue_band_index]

            # Normalize the bands to the range [0, 1]
            red_normalized = normalize_image(red_band)
            green_normalized = normalize_image(green_band)
            blue_normalized = normalize_image(blue_band)

            # Stack the normalized bands to create an RGB image
            rgb_image = np.stack(
                (red_normalized, green_normalized, blue_normalized), axis=-1
            ).squeeze()

            # Handle NaN/Inf values before converting
            rgb_image = np.nan_to_num(rgb_image, nan=0.0, posinf=1.0, neginf=0.0)
            rgb_image = np.clip(rgb_image, 0.0, 1.0)

            # Convert the RGB image to the range [0, 255]
            rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            rgb_image_bgr = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2BGR)

            # Display using OpenCV
            display_image, _ = self.resize_for_display_static(
                rgb_image_bgr, max_height=900
            )
            cv2.imshow("Calibrated Image Preview", display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            QMessageBox.warning(
                self, "Display Error", f"Failed to display image: {str(e)}"
            )

    @staticmethod
    def resize_for_display_static(image, max_height=900):
        """
        Resize image for display if needed.

        Parameters:
        - image (np.ndarray): Input image.
        - max_height (int): Maximum height for display.

        Returns:
        - resized (np.ndarray): Resized image.
        - scale (float): Scaling factor applied.

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibApp()
    window.show()
    sys.exit(app.exec_())
