# PRISM // cloud GPU reconstruction (Google Colab)

Use **`prism_turbo_reconstruction.ipynb`** (PRISM-TURBO 1.2). The engine source is `src/prism_engine.py`;
the notebook's Cell 2 contains exactly the same file.

1. Open the notebook in Google Colab, `Runtime → Change runtime type → T4 GPU` (L4 / A100 are faster).
2. Put the video and its DJI `.SRT` in Google Drive (same name, e.g. `DJI_0001.MP4` + `DJI_0001.SRT`).
   Optional: an RTK / PPK file (`RTK_PATH`) and a camera calibration (`INTRINSICS_PATH`).
3. Run the cells in order. `QUALITY = "sih"` meets the "10-minute video in < 15 minutes" target on a T4;
   `balanced` / `max` spend more time for denser models. No flight altitude has to be typed in: it is read
   from the log.
4. Download `prism_colab_bundle.zip` and import it: **Mission ▸ New mission ▸ Colab package**.

What 1.2 adds: in-decoder 4K resize (faster ingest), vehicle / people / animal masking with an epipolar
motion test (moving objects are not fused, parked ones stay), RTK / PPK and lab-calibration inputs, and the
processing / video-time ratio in `recon_report.json`.

`prism_reconstruction.ipynb` is the original, slower CPU notebook, kept for reference.
