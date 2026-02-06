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
        self.camera_type = "fx10"  # Default to fx10
        # Initialize with FX10 coordinates by default
        self.white_roi = [(165, 77), (900, 77), (900, 110), (165, 110)]
        self.dark_roi = [(165, 2091), (900, 2091), (900, 2100), (165, 2100)]
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

        self.btn_raw = QPushButton("Select RAW cube (.hdr)")
        self.btn_run = QPushButton("Run calibration")
        self.btn_remove_bg = QPushButton("Remove plant background")

        self.btn_raw.clicked.connect(lambda: self.select_file("raw"))
        self.btn_run.clicked.connect(self.run_calibration)
        self.btn_remove_bg.clicked.connect(self.remove_background)

        layout.addWidget(cam_group)
        layout.addWidget(self.btn_raw)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.btn_remove_bg)

        # Output text display
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_text)

        self.setLayout(layout)
        self.cam_fx10.toggled.connect(self.update_camera_type)

    def update_camera_type(self):
        """Update camera type based on radio button selection and load coordinates."""
        if self.cam_fx10.isChecked():
            self.camera_type = "fx10"
            self.white_roi = [(165, 77), (900, 77), (900, 110), (165, 110)]
            self.dark_roi = [(165, 2091), (900, 2091), (900, 2100), (165, 2100)]
            self.append_output("Camera type: FX10 (VNIR)")
            self.append_output("Auto-loaded FX10 calibration coordinates")
        else:
            self.camera_type = "fx17"
            self.white_roi = [(75, 28), (545, 28), (545, 30), (75, 30)]
            self.dark_roi = [(100, 2085), (540, 2085), (540, 2087), (100, 2087)]
            self.append_output("Camera type: FX17 (SWIR)")
            self.append_output("Auto-loaded FX17 calibration coordinates")

    def select_file(self, kind):
        """Select HDR file for RAW cube."""
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
        """Run the calibration process using ROI-based calibration."""
        if not self.raw_hdr:
            self.append_output("ERROR: Missing RAW cube")
            return

        # Create calibrated folder if it doesn't exist
        raw_dir = os.path.dirname(self.raw_hdr)
        output_dir = os.path.join(raw_dir, "calibrated")
        os.makedirs(output_dir, exist_ok=True)

        if self.white_roi is None or self.dark_roi is None:
            self.append_output("ERROR: ROI coordinates not set")
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
        Display the raw and calibrated hyperspectral images as RGB side by side.

        Parameters:
        - image (np.ndarray): The calibrated hyperspectral data cube.
        - wavelengths (list): List of wavelength values corresponding to bands.

        Returns:
        - None
        """
        # Store for later background removal
        self.calibrated_image = image
        self.calibrated_wavelengths = wavelengths

        try:
            # Load raw image for comparison
            import spectral

            raw_img_data = spectral.open_image(self.raw_hdr)
            # Use .load() to get the full array instead of slicing
            raw_image = raw_img_data.load()

            # Safely extract wavelengths from metadata
            raw_wavelengths = []
            try:
                raw_wavelengths_raw = raw_img_data.metadata.get("wavelength", [])
                if raw_wavelengths_raw is not None:
                    # Convert to list if it's array-like
                    if hasattr(raw_wavelengths_raw, "__iter__") and not isinstance(
                        raw_wavelengths_raw, (str, bytes)
                    ):
                        raw_wavelengths = [float(w) for w in raw_wavelengths_raw]
            except (TypeError, ValueError):
                raw_wavelengths = []

            # Extract RGB from raw image
            if raw_wavelengths and len(raw_wavelengths) > 0:
                red_idx = int(find_nearest_band(raw_wavelengths, 660))
                green_idx = int(find_nearest_band(raw_wavelengths, 550))
                blue_idx = int(find_nearest_band(raw_wavelengths, 450))

                raw_r = np.squeeze(
                    normalize_image(raw_image[:, :, red_idx].astype(float))
                )
                raw_g = np.squeeze(
                    normalize_image(raw_image[:, :, green_idx].astype(float))
                )
                raw_b = np.squeeze(
                    normalize_image(raw_image[:, :, blue_idx].astype(float))
                )
                raw_rgb = np.stack([raw_r, raw_g, raw_b], axis=-1)
            else:
                # Fallback: use band indices
                raw_r = np.squeeze(normalize_image(raw_image[:, :, 60].astype(float)))
                raw_g = np.squeeze(normalize_image(raw_image[:, :, 30].astype(float)))
                raw_b = np.squeeze(normalize_image(raw_image[:, :, 10].astype(float)))
                raw_rgb = np.stack([raw_r, raw_g, raw_b], axis=-1)

            # Extract RGB from calibrated image
            if wavelengths and len(wavelengths) > 0:
                red_band_index = int(find_nearest_band(wavelengths, 660))
                green_band_index = int(find_nearest_band(wavelengths, 550))
                blue_band_index = int(find_nearest_band(wavelengths, 450))

                cal_r = np.squeeze(
                    normalize_image(image[:, :, red_band_index].astype(float))
                )
                cal_g = np.squeeze(
                    normalize_image(image[:, :, green_band_index].astype(float))
                )
                cal_b = np.squeeze(
                    normalize_image(image[:, :, blue_band_index].astype(float))
                )
                calibrated_rgb = np.stack([cal_r, cal_g, cal_b], axis=-1)
            else:
                # Fallback: use band indices
                cal_r = np.squeeze(normalize_image(image[:, :, 60].astype(float)))
                cal_g = np.squeeze(normalize_image(image[:, :, 30].astype(float)))
                cal_b = np.squeeze(normalize_image(image[:, :, 10].astype(float)))
                calibrated_rgb = np.stack([cal_r, cal_g, cal_b], axis=-1)

            # Normalize both to [0, 1]
            raw_rgb = np.squeeze(raw_rgb)
            raw_rgb = np.nan_to_num(raw_rgb, nan=0.0, posinf=1.0, neginf=0.0)
            raw_rgb = np.clip(raw_rgb, 0.0, 1.0)

            calibrated_rgb = np.squeeze(calibrated_rgb)
            calibrated_rgb = np.nan_to_num(
                calibrated_rgb, nan=0.0, posinf=1.0, neginf=0.0
            )
            calibrated_rgb = np.clip(calibrated_rgb, 0.0, 1.0)

            # Display side by side using matplotlib
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            axes[0].imshow(raw_rgb)
            axes[0].set_title("Before: Raw Image", fontsize=14, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(calibrated_rgb)
            axes[1].set_title("After: Calibrated Image", fontsize=14, fontweight="bold")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            import traceback

            error_msg = f"Failed to display image: {str(e)}\n{traceback.format_exc()}"
            self.append_output(f"ERROR: {error_msg}")
            QMessageBox.warning(self, "Display Error", error_msg)

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
                self.calibrated_image,
                self.calibrated_wavelengths,
                self.camera_type,
            )

            # Save the background-removed image
            raw_dir = os.path.dirname(self.raw_hdr)
            no_bg_dir = os.path.join(raw_dir, "no-background")
            os.makedirs(no_bg_dir, exist_ok=True)

            raw_basename = os.path.basename(self.raw_hdr)[:-4]
            out_hdr = os.path.join(no_bg_dir, f"{raw_basename}_no_background.hdr")

            import spectral

            spectral.envi.save_image(
                out_hdr, plant_only, dtype=np.float32, interleave="bil", force=True
            )
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

            # Determine index map title
            if self.camera_type == "fx10":
                index_title = "NDVI"
                cmap = "RdYlGn"
            else:
                index_title = "NWACI"
                cmap = "RdYlGn"

            # Plot 1: Index map (NDVI, NWACI, or Green reflectance)
            im0 = axes[0].imshow(index_map, cmap=cmap)
            axes[0].set_title(index_title)
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
