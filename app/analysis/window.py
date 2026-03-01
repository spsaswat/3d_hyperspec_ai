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

		#storage variable for fx17 corrected varaible
		self.cross_results = None

		layout.addLayout(h1)
		layout.addLayout(h2)
		layout.addWidget(self.plot_btn)
		layout.addWidget(self.save_fx17_btn)

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

			# Open dialog starting inside corrected folder
			output_dir = QFileDialog.getExistingDirectory(
				self,
				"Select directory to save corrected FX17",
				corrected_dir
			)

			if not output_dir:
				return

			# --------------------------------------------------
			# Prepare corrected spectrum
			# --------------------------------------------------
			fx17_wl = self.cross_results["fx17_corr_wl"]
			fx17_corr_spec = self.cross_results["fx17_corr_spec"]

			data = np.array(fx17_corr_spec, dtype=np.float32)
			data = data.reshape((1, 1, -1))  # ENVI requires 3D

			# --------------------------------------------------
			# Copy metadata from original FX17
			# --------------------------------------------------
			original = spectral.open_image(self.fx17_path)
			meta = original.metadata.copy()

			meta["description"] = f"Corrected FX17 spectrum saved at {time.ctime()}"
			meta["samples"] = 1
			meta["lines"] = 1
			meta["bands"] = len(fx17_wl)
			meta["wavelength"] = [str(w) for w in fx17_wl]
			meta["interleave"] = "bil"
			meta["data type"] = 4  # float32

			raw_name = os.path.basename(self.fx17_path)[:-4]
			out_hdr = os.path.join(output_dir, f"{raw_name}_corrected-FX17.hdr")

			# --------------------------------------------------
			# Save
			# --------------------------------------------------
			spectral.envi.save_image(
				out_hdr,
				data,
				dtype=np.float32,
				interleave="bil",
				metadata=meta,
				force=True,
			)

			QMessageBox.information(
				self,
				"Success",
				f"Corrected FX17 saved to:\n{out_hdr}",
			)

		except Exception as e:
			QMessageBox.warning(self, "Save Error", f"Failed to save:\n{e}")
