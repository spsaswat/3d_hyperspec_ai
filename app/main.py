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
from preprocessing.background_removal import remove_background


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
        self.camera_type = "fx10"  # Default to fx10
        # Initialize with FX10 coordinates by default
        self.white_roi = [(231, 77), (845, 77), (845, 110), (231, 110)]
        self.dark_roi = [(217, 2091), (848, 2091), (848, 2100), (217, 2100)]
        self.calibrated_image = None
        self.calibrated_wavelengths = None

        layout = QVBoxLayout()

        # Camera type selection
        self.cam_fx10 = QRadioButton("FX10 (VNIR)")
        self.cam_fx17 = QRadioButton("FX17 (SWIR)")
        self.cam_fx10.setChecked(True)

        cam_group = QGroupBox("Camera Type")
        cam_layout = QVBoxLayout()
        cam_layout.addWidget(self.cam_fx10)
        cam_layout.addWidget(self.cam_fx17)
        cam_group.setLayout(cam_layout)

        layout.addWidget(cam_group)

        # Calibration mode selection
        self.mode_full = QRadioButton("Use WHITE & DARK cubes")
        self.mode_roi = QRadioButton("Use fixed calibration regions from RAW")
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
        self.btn_remove_bg = QPushButton("Remove plant background")

        self.btn_raw.clicked.connect(lambda: self.select_file("raw"))
        self.btn_white.clicked.connect(lambda: self.select_file("white"))
        self.btn_dark.clicked.connect(lambda: self.select_file("dark"))
        self.btn_run.clicked.connect(self.run_calibration)
        self.btn_remove_bg.clicked.connect(self.remove_background)

        layout.addWidget(self.btn_raw)
        layout.addWidget(self.btn_white)
        layout.addWidget(self.btn_dark)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.btn_remove_bg)

        # Output text display
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_text)

        self.setLayout(layout)
        self.cam_fx10.toggled.connect(self.update_camera_type)
        self.mode_full.toggled.connect(self.update_mode)
        self.update_mode()

    def update_mode(self):
        """Enable/disable buttons based on selected mode."""
        full = self.mode_full.isChecked()
        self.btn_white.setEnabled(full)
        self.btn_dark.setEnabled(full)

    def update_camera_type(self):
        """Update camera type based on radio button selection and load coordinates."""
        if self.cam_fx10.isChecked():
            self.camera_type = "fx10"
            self.white_roi = [(231, 77), (845, 77), (845, 110), (231, 110)]
            self.dark_roi = [(217, 2091), (848, 2091), (848, 2100), (217, 2100)]
            self.append_output("Camera type: FX10 (VNIR)")
            self.append_output("Auto-loaded FX10 calibration coordinates")
        else:
            self.camera_type = "fx17"
            self.white_roi = [(99, 28), (512, 28), (512, 30), (99, 30)]
            self.dark_roi = [(111, 2085), (517, 2085), (517, 2087), (111, 2087)]
            self.append_output("Camera type: FX17 (SWIR)")
            self.append_output("Auto-loaded FX17 calibration coordinates")

    def select_file(self, kind):
        """Select HDR file for RAW, WHITE, or DARK."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HDR file", "", "HDR Files (*.hdr)"
        )
        if not path:
            return
        setattr(self, f"{kind}_hdr", path)
        self.append_output(f"{kind.upper()} selected: {path}")

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
        # Store for later background removal
        self.calibrated_image = image
        self.calibrated_wavelengths = wavelengths

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

    def remove_background(self):
        """Remove plant background from calibrated image based on camera type."""
        if self.calibrated_image is None or self.calibrated_wavelengths is None:
            self.append_output("ERROR: Run calibration first")
            return

        self.append_output(
            f"Removing plant background using {self.camera_type.upper()}..."
        )

        try:
            plant_only, plant_mask, index_map = remove_background(
                self.calibrated_image, self.calibrated_wavelengths, self.camera_type
            )

            # Save the background-removed image
            raw_dir = os.path.dirname(self.raw_hdr)
            no_bg_dir = os.path.join(raw_dir, "no-background")
            os.makedirs(no_bg_dir, exist_ok=True)

            raw_basename = os.path.basename(self.raw_hdr)[:-4]
            out_hdr = os.path.join(no_bg_dir, f"{raw_basename}_no_background.hdr")

            import spectral

            spectral.envi.save_image(out_hdr, plant_only, dtype=np.float32, interleave='bil', force=True)
            self.append_output(f"Background-removed image saved to {out_hdr}")

            # Display the results
            self.display_removal_results(plant_only, plant_mask, index_map)

            self.append_output("Plant background removal complete ✔")

        except Exception as e:
            self.append_output(f"ERROR: Background removal failed: {str(e)}")

    def display_removal_results(self, plant_only, plant_mask, index_map):
        """
        Display plant background removal results.

        Parameters:
        - plant_only: 3D image with background removed
        - plant_mask: 2D boolean mask
        - index_map: 2D map (NDVI or NWACI)
        """
        try:
            # Create a figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Plot 1: Index map (NDVI or NWACI)
            im0 = axes[0].imshow(index_map, cmap="RdYlGn")
            axes[0].set_title(f"{'NDVI' if self.camera_type == 'fx10' else 'NWACI'}")
            axes[0].axis("off")
            plt.colorbar(im0, ax=axes[0])

            # Plot 2: Plant mask
            axes[1].imshow(plant_mask, cmap="gray")
            axes[1].set_title("Plant Mask")
            axes[1].axis("off")

            # Plot 3: Plant pixels only (RGB preview)
            if self.calibrated_wavelengths and len(self.calibrated_wavelengths) > 0:
                red_idx = find_nearest_band(self.calibrated_wavelengths, 660)
                green_idx = find_nearest_band(self.calibrated_wavelengths, 550)
                blue_idx = find_nearest_band(self.calibrated_wavelengths, 450)

                rgb = np.stack(
                    [
                        normalize_image(plant_only[:, :, red_idx]),
                        normalize_image(plant_only[:, :, green_idx]),
                        normalize_image(plant_only[:, :, blue_idx]),
                    ],
                    axis=-1,
                )
                rgb = np.clip(rgb, 0, 1)
            else:
                # Fallback
                rgb = plant_only[:, :, [60, 30, 10]]
                rgb = normalize_image(rgb)

            axes[2].imshow(rgb)
            axes[2].set_title("Plant Pixels Only")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.warning(
                self, "Display Error", f"Failed to display results: {str(e)}"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibApp()
    window.show()
    sys.exit(app.exec_())
