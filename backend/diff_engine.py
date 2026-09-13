import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def detect_3d_changes(baseline_ply, new_ply, output_ply, distance_threshold=0.5):
    """
    Compares two 3D point clouds. Any points in 'new_ply' that are further than 
    'distance_threshold' (in meters) from 'baseline_ply' are flagged as new structures.
    """
    print(f"Loading Baseline Scan: {baseline_ply}")
    pcd_base = o3d.io.read_point_cloud(baseline_ply)
    
    print(f"Loading Day-2 Recon Scan: {new_ply}")
    pcd_new = o3d.io.read_point_cloud(new_ply)

    # Convert point cloud geometry into fast numpy arrays
    base_points = np.asarray(pcd_base.points)
    new_points = np.asarray(pcd_new.points)
    
    if len(base_points) == 0 or len(new_points) == 0:
        print("Error: One of the point clouds is empty.")
        return

    # Build a K-D Tree for lightning-fast spatial distance calculations
    print("Building spatial index for rapid cross-referencing...")
    tree = cKDTree(base_points)

    # Calculate the exact distance from every new point to the closest baseline point
    print("Calculating temporal spatial geometry diff...")
    distances, _ = tree.query(new_points, k=1)

    # Flag any points that are further away than our threshold (e.g., 0.5 meters)
    # These are points that literally did not exist yesterday (like a new tent or truck)
    anomaly_mask = distances > distance_threshold
    num_anomalies = np.sum(anomaly_mask)
    
    print(f"Detected {num_anomalies} new anomaly points!")

    # Repaint the new point cloud: Keep old structures grey, paint new threats BRIGHT RED
    colors = np.zeros_like(new_points)
    colors[~anomaly_mask] = [0.6, 0.6, 0.6]  # Safe/Unchanged (Grey)
    colors[anomaly_mask] = [1.0, 0.0, 0.0]   # Threat/New (Red)
    
    pcd_new.colors = o3d.utility.Vector3dVector(colors)

    # Save the flagged model for the PRISM Dashboard
    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    o3d.io.write_point_cloud(output_ply, pcd_new)
    print(f"Actionable 3D Threat Map saved to: {output_ply}")

if __name__ == "__main__":
    # For future testing, you will pass your Day 1 and Day 2 .ply models here
    BASELINE = "../data/models/day1_baseline.ply"
    NEW_SCAN = "../data/models/day2_recon.ply"
    OUTPUT_MAP = "../data/models/actionable_threat_map.ply"
    
    # If the test files exist, run the diff!
    if os.path.exists(BASELINE) and os.path.exists(NEW_SCAN):
        detect_3d_changes(BASELINE, NEW_SCAN, OUTPUT_MAP, distance_threshold=0.75)
    else:
        print("Waiting for Day 1 and Day 2 .ply files to be generated...")