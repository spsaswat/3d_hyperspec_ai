import numpy as np
import matplotlib.pyplot as plt
import spectral


def cross_calibrate_and_plot(fx10_path, fx17_path, overlap=(900, 1000), overlap_grid_n=300):
    """Cross-calibrate FX17 to FX10 using overlap and plot continuous reflectance.
    Steps:
    - Load FX10 and FX17 HDRs and compute spatial mean spectra.
    - Interpolate both spectra over a dense grid inside `overlap` and fit
      FX17_corrected = a * FX17 + b to match FX10 on the overlap (least-squares).
    - Apply correction to all FX17 bands.
    - Build a continuous spectrum using FX10 where available and FX17_corrected
      for wavelengths beyond the FX10 range.
    Parameters:
    - fx10_path (str): path to FX10 .hdr file
    - fx17_path (str): path to FX17 .hdr file
    - overlap (tuple): (start_nm, stop_nm) overlap for fitting
    - overlap_grid_n (int): number of points in dense overlap grid for fitting
    Returns:
    - dict with keys: 'fx10_wl','fx10_spec','fx17_wl','fx17_spec','fx17_corr_wl',
      'fx17_corr_spec','combined_wl','combined_spec','a','b'
    """

    def _load_mean(hdr_path):
        img = spectral.open_image(hdr_path)
        data = img.load()
        mean_spec = np.nanmean(data, axis=(0, 1))
        wls = img.metadata.get("wavelength", None)
        if wls is None:
            wls = np.arange(mean_spec.size)
        else:
            try:
                wls = np.array([float(w) for w in wls])
            except Exception:
                wls = np.arange(mean_spec.size)
        return np.array(wls), np.array(mean_spec)

    fx10_wl, fx10_spec = _load_mean(fx10_path)
    fx17_wl, fx17_spec = _load_mean(fx17_path)

    # Prepare dense overlap grid
    s, e = overlap
    ol_grid = fx17_wl[(fx17_wl >= s) & (fx17_wl <= e)]

    # Interpolate spectra onto the overlap grid
    fx10_ol = np.interp(ol_grid, fx10_wl, fx10_spec)
    fx17_ol = fx17_spec[(fx17_wl >= s) & (fx17_wl <= e)]

    # Solve least-squares for a and b: a * fx17_ol + b ~= fx10_ol
    A = np.vstack([fx17_ol, np.ones_like(fx17_ol)]).T
    a, b = np.linalg.lstsq(A, fx10_ol, rcond=None)[0]

    # Apply correction to full FX17 spectrum
    fx17_corr_spec = a * fx17_spec + b

    # Build combined continuous spectrum:
    # - Use FX10 for wavelengths < overlap start
    # - Use corrected FX17 for wavelengths >= overlap start
    overlap_start = s

    mask_fx10_before = fx10_wl < overlap_start
    mask_fx17_after = fx17_wl >= overlap_start

    combined_wl = np.concatenate([
        fx10_wl[mask_fx10_before],
        fx17_wl[mask_fx17_after]
    ])
    combined_spec = np.concatenate([
        fx10_spec[mask_fx10_before],
        fx17_corr_spec[mask_fx17_after]
    ])

    # Sort combined by wavelength (in case inputs are unordered)
    order = np.argsort(combined_wl)
    combined_wl = combined_wl[order]
    combined_spec = combined_spec[order]

    # Plotting: FX10, original FX17, corrected FX17, combined continuous curve
    plt.figure(figsize=(9, 5))
    plt.plot(fx10_wl, fx10_spec, label="FX10 (reference)", color="tab:blue")
    plt.plot(fx17_wl, fx17_spec, label="FX17 (original)", color="tab:orange", alpha=0.6)
    plt.plot(fx17_wl, fx17_corr_spec, label="FX17 (corrected)", color="tab:green", linestyle="--")
    plt.plot(combined_wl, combined_spec, label="Combined continuous", color="k", linewidth=1)
    plt.axvspan(s, e, color="gray", alpha=0.08)
    plt.title("Cross-calibrated reflectance (FX10 reference)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (combined_wl, combined_spec)

    # return {
    #     "fx10_wl": fx10_wl,
    #     "fx10_spec": fx10_spec,
    #     "fx17_wl": fx17_wl,
    #     "fx17_spec": fx17_spec,
    #     "fx17_corr_wl": fx17_wl,
    #     "fx17_corr_spec": fx17_corr_spec,
    #     "combined_wl": combined_wl,
    #     "combined_spec": combined_spec,
    #     "a": float(a),
    #     "b": float(b),
    # }


if __name__ == "__main__":
    # Example usage:
    fx10_path = r"D:\anu-files\ANU\data\csiro_fx10_17_2_wheat\no-background\004-specim-fx10_no_background.hdr"
    fx17_path = r"D:\anu-files\ANU\data\csiro_fx10_17_2_wheat\no-background\004-specim-fx17_no_background.hdr"
    results = cross_calibrate_and_plot(fx10_path, fx17_path)
