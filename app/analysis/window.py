import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PyQt5.QtWidgets import (
	QWidget,
	QPushButton,
	QLabel,
	QFileDialog,
	QVBoxLayout,
	QHBoxLayout,
	QMessageBox,
)
from PyQt5.QtCore import pyqtSignal
 
import spectral

from .spectrum_plot import cross_calibrate_and_plot
from .spatial_registration import register_fx10_fx17


class AnalysisWindow(QWidget):
	"""Simple analysis window to load FX10 and FX17 HDRs and plot mean spectra.

	- Select FX10 file and FX17 file using two buttons.
	- When both are selected, click "Plot" to show two plots (one per camera).
	"""

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Analysis")
		# Enforce a sensible minimum and initial size
		self.setMinimumSize(300, 200)
		# self.resize(800, 520)
		self.fx10_path = None
		self.fx17_path = None
		self.corrected_fx17_path = None  # Path to saved corrected FX17

		layout = QVBoxLayout()

		# FX10 selector
		h1 = QHBoxLayout()
		self.fx10_btn = QPushButton("Select FX10 HDR")
		self.fx10_btn.clicked.connect(self.select_fx10)
		self.fx10_label = QLabel("No file selected")
		h1.addWidget(self.fx10_btn)
		h1.addWidget(self.fx10_label)

		# FX17 selector
		h2 = QHBoxLayout()
		self.fx17_btn = QPushButton("Select FX17 HDR")
		self.fx17_btn.clicked.connect(self.select_fx17)
		self.fx17_label = QLabel("No file selected")
		h2.addWidget(self.fx17_btn)
		h2.addWidget(self.fx17_label)

		# Plot button
		self.plot_btn = QPushButton("Plot")
		self.plot_btn.setEnabled(False)
		self.plot_btn.clicked.connect(self.plot_spectra)

		# Save Corrected FX17 button
		self.save_fx17_btn = QPushButton("Save Corrected FX17")
		self.save_fx17_btn.setEnabled(False)
		self.save_fx17_btn.clicked.connect(self.save_corrected_fx17)

		# Overlay Preview button
		self.overlay_btn = QPushButton("Overlay Preview")
		self.overlay_btn.setEnabled(False)
		self.overlay_btn.clicked.connect(self.show_overlay_preview)

		# Save Wrap Matrix button
		self.save_warp_btn = QPushButton("Save Wrap Matrix")
		self.save_warp_btn.setEnabled(False)
		self.save_warp_btn.clicked.connect(self.save_wrap_matrix)

		# Save Registered Image button
		self.save_registered_btn = QPushButton("Save Registered Image")
		self.save_registered_btn.setEnabled(False)
		self.save_registered_btn.clicked.connect(self.save_registered_image)

		#storage variable for fx17 corrected varaible
		self.cross_results = None
		self.registration_results = None  # Store wrap matrix and inliers

		layout.addLayout(h1)
		layout.addLayout(h2)
		layout.addWidget(self.plot_btn)
		layout.addWidget(self.save_fx17_btn)
		layout.addWidget(self.overlay_btn)
		layout.addWidget(self.save_warp_btn)
		layout.addWidget(self.save_registered_btn)

		self.setLayout(layout)

	closed = pyqtSignal()

	def closeEvent(self, event):
		# Emit a closed signal so callers (main window) can re-enable buttons
		try:
			self.closed.emit()
		except Exception:
			pass
		super().closeEvent(event)

	def select_fx10(self):
		path, _ = QFileDialog.getOpenFileName(self, "Select FX10 HDR", "", "HDR Files (*.hdr)")
		if not path:
			return
		self.fx10_path = path
		self.fx10_label.setText(os.path.basename(path))
		self._update_plot_enabled()

	def select_fx17(self):
		path, _ = QFileDialog.getOpenFileName(self, "Select FX17 HDR", "", "HDR Files (*.hdr)")
		if not path:
			return
		self.fx17_path = path
		self.fx17_label.setText(os.path.basename(path))
		self._update_plot_enabled()

	def _update_plot_enabled(self):
		self.plot_btn.setEnabled(bool(self.fx10_path and self.fx17_path))

	def _mark_extrema(self, ax, wl, spectrum):
		"""Mark only prominent local minima and maxima with wavelength labels.

		Uses `scipy.signal.find_peaks` with a prominence threshold (10% of
		the spectrum dynamic range) to avoid annotating small fluctuations.
		"""
		# Guard against empty or constant spectra
		if spectrum is None or spectrum.size == 0:
			return
		s = np.nanmin(spectrum)
		e = np.nanmax(spectrum)
		rng = float(e - s)
		if rng <= 0 or not np.isfinite(rng):
			return

		# Prominence threshold as a fraction of dynamic range (increased to 10%)
		prom = max(rng * 0.10, 1e-6)

		# Find prominent maxima and minima
		max_idx, _ = signal.find_peaks(spectrum, prominence=prom)
		min_idx, _ = signal.find_peaks(-spectrum, prominence=prom)

		# Plot maxima (black dots)
		for idx in max_idx:
			# ax.plot(wl[idx], spectrum[idx], "ko", markersize=5)
			ax.annotate(f"{wl[idx]:.2f} nm", xy=(wl[idx], spectrum[idx]),
						xytext=(2, 3), textcoords="offset points",
						ha="left", fontsize=7, color="black", fontweight="bold",
						bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

		# Plot minima (black dots)
		for idx in min_idx:
			# ax.plot(wl[idx], spectrum[idx], "ko", markersize=5)
			ax.annotate(f"{wl[idx]:.2f} nm", xy=(wl[idx], spectrum[idx]),
						xytext=(2, -8), textcoords="offset points",
						ha="left", fontsize=7, color="black", fontweight="bold",
						bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

	def _load_mean_spectrum(self, hdr_path):
		img = spectral.open_image(hdr_path)
		data = img.load()  # H x W x B
		# mean across spatial dims
		mean_spectrum = np.nanmean(data, axis=(0, 1))
		# attempt to read wavelengths metadata
		wls = img.metadata.get("wavelength", None)
		if wls is None:
			# fallback to band indices
			wls = np.arange(mean_spectrum.size)
		else:
			try:
				wls = [float(w) for w in wls]
			except Exception:
				wls = np.arange(mean_spectrum.size)
		return np.array(wls), np.array(mean_spectrum)

	def plot_spectra(self):
		try:
			fx10_wl, fx10_spec = self._load_mean_spectrum(self.fx10_path)
			fx17_wl, fx17_spec = self._load_mean_spectrum(self.fx17_path)

			results = cross_calibrate_and_plot(
				fx10_path=self.fx10_path,
				fx17_path=self.fx17_path
			)

			self.cross_results = results
			self.save_fx17_btn.setEnabled(True)

			fx17_corr_wl = results["fx17_corr_wl"]
			fx17_corr_spec = results["fx17_corr_spec"]

			# Mask ≥ 900nm
			mask = fx17_corr_wl >= 900
			fx17_corr_wl_masked = fx17_corr_wl[mask]
			fx17_corr_spec_masked = fx17_corr_spec[mask]

			fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

			# FX10
			axes[0].plot(fx10_wl, fx10_spec, color="tab:blue")
			axes[0].set_title("FX10 ROI mean")
			axes[0].set_ylabel("Reflectance")
			axes[0].set_xlabel("Wavelength")
			self._mark_extrema(axes[0], fx10_wl, fx10_spec)

			# FX17
			axes[1].plot(fx17_wl, fx17_spec, color="tab:green")
			axes[1].set_title("FX17 original ROI mean")
			axes[1].set_ylabel("Reflectance")
			axes[1].set_xlabel("Wavelength")
			self._mark_extrema(axes[1], fx17_wl, fx17_spec)

			# Corrected FX17 ≥ 900nm
			axes[2].plot(fx17_corr_wl_masked, fx17_corr_spec_masked,
						color="tab:purple")
			axes[2].set_title("FX17 corrected ROI mean")
			axes[2].set_ylabel("Reflectance")
			axes[2].set_xlabel("Wavelength")
			self._mark_extrema(axes[2],
							fx17_corr_wl_masked,
							fx17_corr_spec_masked)

			plt.tight_layout()
			plt.show()

		except Exception as e:
			QMessageBox.warning(self, "Plot Error",
								f"Failed to plot spectra: {e}")

	#function for saving the corrected fx17 
	def save_corrected_fx17(self):
		if self.cross_results is None:
			QMessageBox.warning(self, "Error", "No corrected spectrum available")
			return

		try:
			import spectral
			import time

			# --------------------------------------------------
			# Create /corrected folder beside FX17 file
			# --------------------------------------------------
			parent_dir = os.path.dirname(self.fx17_path)
			corrected_dir = os.path.join(parent_dir, "corrected")
			os.makedirs(corrected_dir, exist_ok=True)

			output_dir = QFileDialog.getExistingDirectory(
				self,
				"Select directory to save corrected FX17",
				corrected_dir
			)

			if not output_dir:
				return

			# --------------------------------------------------
			# Load original FX17 cube
			# --------------------------------------------------
			img = spectral.open_image(self.fx17_path)
			cube = img.load()  # H x W x B
			cube = np.array(cube, dtype=np.float32)

			H, W, B = cube.shape

			# --------------------------------------------------
			# Load correction spectrum
			# --------------------------------------------------
			fx17_corr_spec = self.cross_results["fx17_corr_spec"]
			fx17_corr_spec = np.array(fx17_corr_spec, dtype=np.float32)

			if len(fx17_corr_spec) != B:
				raise ValueError("Corrected spectrum size does not match FX17 bands")

			# --------------------------------------------------
			# Apply correction to every pixel
			# --------------------------------------------------
			# corrected_cube = cube * fx17_corr_spec  # broadcast across H,W
			corrected_cube = self.cross_results["a"]*cube + self.cross_results["b"]  # linear correction

			# --------------------------------------------------
			# Copy metadata
			# --------------------------------------------------
			meta = img.metadata.copy()

			meta["description"] = f"Corrected FX17 cube saved at {time.ctime()}"
			meta["samples"] = W
			meta["lines"] = H
			meta["bands"] = B
			meta["data type"] = 4  # float32
			meta["interleave"] = "bil"

			# --------------------------------------------------
			# Output file name
			# --------------------------------------------------
			raw_name = os.path.basename(self.fx17_path)[:-4]

			out_hdr = os.path.join(
				output_dir,
				f"{raw_name}_corrected-FX17.hdr"
			)

			# --------------------------------------------------
			# Save full hyperspectral cube
			# --------------------------------------------------
			spectral.envi.save_image(
				out_hdr,
				corrected_cube,
				dtype=np.float32,
				interleave="bil",
				metadata=meta,
				force=True
			)

			# Store the corrected FX17 path
			self.corrected_fx17_path = out_hdr
			self.overlay_btn.setEnabled(True)

			QMessageBox.information(
				self,
				"Success",
				f"Corrected FX17 cube saved to:\n{out_hdr}"
			)

		except Exception as e:
			QMessageBox.warning(self, "Save Error", f"Failed to save:\n{e}")

	def show_overlay_preview(self):
		"""Show overlay preview by performing spatial registration."""
		if not self.fx10_path or not self.corrected_fx17_path:
			QMessageBox.warning(self, "Error", "FX10 path or corrected FX17 path not available")
			return

		try:
			print(f"Registering FX10: {self.fx10_path}")
			print(f"Registering corrected FX17: {self.corrected_fx17_path}")
			
			# Call the registration function
			cube17_registered, M, inliers = register_fx10_fx17(
				self.fx10_path,
				self.corrected_fx17_path
			)

			# Store registration results
			self.registration_results = {
				"cube": cube17_registered,
				"M": M,
				"inliers": inliers
			}

			# Enable save wrap matrix button
			self.save_warp_btn.setEnabled(True)
			self.save_registered_btn.setEnabled(True)

			# Display wrap matrix and inliers information
			msg = f"Registration Complete!\n\n"
			msg += f"Wrap Matrix (Affine 2x3):\n{M}\n\n"
			msg += f"Number of Affine Inliers: {np.sum(inliers)}\n"
			msg += f"Total Inliers: {np.sum(inliers)}"
			
			QMessageBox.information(self, "Registration Results", msg)

		except Exception as e:
			QMessageBox.warning(self, "Registration Error", f"Failed to register:\n{e}")

	def save_wrap_matrix(self):
		"""Save the wrap matrix to a file selected by user."""
		if self.registration_results is None:
			QMessageBox.warning(self, "Error", "No wrap matrix available")
			return

		try:
			# Get output file path from user
			file_path, _ = QFileDialog.getSaveFileName(
				self,
				"Save Wrap Matrix",
				"",
				"Text Files (*.txt);;NumPy Files (*.npy);;CSV Files (*.csv)"
			)

			if not file_path:
				return

			M = self.registration_results["M"]
			inliers = self.registration_results["inliers"]

			# Save based on file extension
			if file_path.endswith('.npy'):
				np.save(file_path, M)
				np.save(file_path.replace('.npy', '_inliers.npy'), inliers)
				QMessageBox.information(
					self,
					"Success",
					f"Wrap matrix saved to:\n{file_path}\n\nInliers saved to:\n{file_path.replace('.npy', '_inliers.npy')}"
				)
			elif file_path.endswith('.csv'):
				np.savetxt(file_path, M, delimiter=',')
				np.savetxt(file_path.replace('.csv', '_inliers.csv'), inliers, delimiter=',')
				QMessageBox.information(
					self,
					"Success",
					f"Wrap matrix saved to:\n{file_path}\n\nInliers saved to:\n{file_path.replace('.csv', '_inliers.csv')}"
				)
			else:  # .txt or default
				with open(file_path, 'w') as f:
					f.write("Affine Transformation Matrix (2x3):\n")
					f.write(str(M))
					f.write("\n\nInliers:\n")
					f.write(str(inliers))
				QMessageBox.information(
					self,
					"Success",
					f"Wrap matrix saved to:\n{file_path}"
				)

		except Exception as e:
			QMessageBox.warning(self, "Save Error", f"Failed to save:\n{e}")

	def save_registered_image(self):
		"""Save the merged registered image (FX10 + registered FX17) with wavelengths."""
		if self.registration_results is None or not self.fx10_path or not self.corrected_fx17_path:
			QMessageBox.warning(self, "Error", "Registration results or file paths not available")
			return

		try:
			# Get output directory from user
			output_dir = QFileDialog.getExistingDirectory(
				self,
				"Select directory to save registered image"
			)

			if not output_dir:
				return

			print("Loading FX10 cube...")
			# Load FX10 cube and metadata
			fx10_img = spectral.open_image(self.fx10_path)
			fx10_cube = fx10_img.load().astype(np.float32)
			fx10_meta = fx10_img.metadata.copy()
			fx10_wl = np.array([float(w) for w in fx10_meta.get("wavelength", [])])
			fx10_bands = fx10_cube.shape[2]

			print("Loading FX17 corrected cube...")
			# Load FX17 corrected cube and metadata
			fx17_img = spectral.open_image(self.corrected_fx17_path)
			fx17_cube = fx17_img.load().astype(np.float32)
			fx17_meta = fx17_img.metadata.copy()
			fx17_wl = np.array([float(w) for w in fx17_meta.get("wavelength", [])])
			fx17_bands = fx17_cube.shape[2]

			print(f"FX10 cube shape: {fx10_cube.shape}, wavelength range: {fx10_wl[0]:.2f}-{fx10_wl[-1]:.2f} nm")
			print(f"FX17 cube shape: {fx17_cube.shape}, wavelength range: {fx17_wl[0]:.2f}-{fx17_wl[-1]:.2f} nm")

			# Get registered FX17 cube from registration results
			fx17_registered = self.registration_results["cube"]
			print(f"Registered FX17 cube shape: {fx17_registered.shape}")

			# Ensure same spatial dimensions
			H, W, _ = fx10_cube.shape
			if fx17_registered.shape[:2] != (H, W):
				raise ValueError(f"Spatial dimension mismatch: FX10 {(H, W)} vs FX17 registered {fx17_registered.shape[:2]}")

			# Find band indices for wavelength ranges
			# FX10: 400-900nm
			fx10_900_idx = np.where(fx10_wl <= 900)[0]
			# FX17: 900-1700nm
			fx17_900_idx = np.where(fx17_wl >= 900)[0]
			fx17_1000_idx = np.where(fx17_wl >= 1000)[0]

			print(f"FX10 bands up to 900nm: {len(fx10_900_idx)}")
			print(f"FX17 bands from 900nm: {len(fx17_900_idx)}")
			print(f"FX17 bands from 1000nm: {len(fx17_1000_idx)}")

			# Build merged wavelength list and cube
			merged_wl_list = []
			merged_bands = []

			# Part 1: FX10 400-900nm
			merged_wl_list.extend(fx10_wl[fx10_900_idx].tolist())
			for idx in fx10_900_idx:
				merged_bands.append(fx10_cube[:, :, idx])

			# Part 2: FX17 900-1000nm (overlap region - use FX10 wavelengths but could add both)
			overlap_900_1000_fx10 = fx10_wl[(fx10_wl >= 900) & (fx10_wl <= 1000)]
			overlap_900_1000_fx17 = fx17_wl[(fx17_wl >= 900) & (fx17_wl <= 1000)]
			
			# For overlap: include unique wavelengths from both, prioritizing FX10
			overlap_all_wl = np.unique(np.concatenate([overlap_900_1000_fx10, overlap_900_1000_fx17]))
			
			for wl in overlap_all_wl:
				# Prefer FX10 if available
				if wl in fx10_wl:
					idx = np.where(fx10_wl == wl)[0][0]
					merged_wl_list.append(wl)
					merged_bands.append(fx10_cube[:, :, idx])
				elif wl in fx17_wl:
					idx = np.where(fx17_wl == wl)[0][0]
					merged_wl_list.append(wl)
					merged_bands.append(fx17_registered[:, :, idx])

			# Part 3: FX17 1000-1700nm
			merged_wl_list.extend(fx17_wl[fx17_1000_idx].tolist())
			for idx in fx17_1000_idx:
				merged_bands.append(fx17_registered[:, :, idx])

			# Stack all bands
			merged_cube = np.stack(merged_bands, axis=2).astype(np.float32)
			merged_wl_list = np.array(merged_wl_list)

			print(f"Merged cube shape: {merged_cube.shape}")
			print(f"Merged wavelength range: {merged_wl_list[0]:.2f}-{merged_wl_list[-1]:.2f} nm")

			# Update metadata
			meta = fx10_meta.copy()
			meta["description"] = "Merged FX10 (400-900nm) + Registered FX17 (900-1700nm)"
			meta["samples"] = W
			meta["lines"] = H
			meta["bands"] = merged_cube.shape[2]
			meta["data type"] = 4  # float32
			meta["interleave"] = "bil"
			meta["wavelength"] = merged_wl_list.tolist()

			# Output file name
			timestamp = __import__('time').strftime("%Y%m%d_%H%M%S")
			out_hdr = os.path.join(
				output_dir,
				f"merged_fx10_fx17_registered_{timestamp}.hdr"
			)

			print(f"Saving merged image to: {out_hdr}")

			# Save merged hyperspectral cube
			spectral.envi.save_image(
				out_hdr,
				merged_cube,
				dtype=np.float32,
				interleave="bil",
				metadata=meta,
				force=True
			)

			QMessageBox.information(
				self,
				"Success",
				f"Registered merged image saved to:\n{out_hdr}\n\n"
				f"Shape: {merged_cube.shape}\n"
				f"Wavelength range: {merged_wl_list[0]:.2f}-{merged_wl_list[-1]:.2f} nm"
			)

		except Exception as e:
			import traceback
			QMessageBox.warning(self, "Save Error", f"Failed to save:\n{str(e)}\n{traceback.format_exc()}")