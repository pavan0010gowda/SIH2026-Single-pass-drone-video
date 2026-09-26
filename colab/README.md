# PRISM // Cloud GPU Photogrammetry Pipeline (Google Colab / Kaggle)

This directory contains the ready-to-use cloud reconstruction notebook for offloading heavy photogrammetry workloads (such as 3.5GB+ 4K UAV videos) from consumer laptops to high-performance cloud GPUs.

## Files:
- [`prism_reconstruction.ipynb`](file:///c:/Users/pavan/PRISM-Defense-3D%20-%20Anti/colab/prism_reconstruction.ipynb): 
  Complete Jupyter notebook containing 8 cells for running on Google Colab or Kaggle with GPU acceleration (NVIDIA T4 / V100 / A100).
- [`../COLAB_INSTRUCTIONS.txt`](file:///c:/Users/pavan/PRISM-Defense-3D%20-%20Anti/COLAB_INSTRUCTIONS.txt): 
  Comprehensive step-by-step instructions on switching GPUs, adding 3.5GB videos via Google Drive, running cells, downloading the output bundle, and importing into PRISM with 1 click.

## Quick Workflow:
1. Open [Google Colab](https://colab.research.google.com) and upload `prism_reconstruction.ipynb`.
2. Select **Runtime -> Change runtime type -> T4 GPU** (or A100).
3. Mount Google Drive and run all 8 cells.
4. Download `prism_colab_bundle.zip`.
5. Open PRISM Dashboard -> Click **Ingest Mission** -> Choose **Google Colab Package** -> Drop `prism_colab_bundle.zip` -> Click **Deploy Cloud Mission**.
