#!/usr/bin/env python3
"""
Convert VIPE camera poses + pre-existing intrinsics to Dyn-HaMR cameras.npz format.

Usage:
    python convert_vipe_cameras.py /path/to/data_folder [--output cameras.npz]
    
Expected folder structure:
    data_folder/
        intrinsics.json         # Pre-existing intrinsics
        video1.mp4              # Video file
        video2.mp4              # Another video file
        ...
        
Output:
    data_folder/
        video1/cameras.npz      # Cameras file for video1
        video2/cameras.npz      # Cameras file for video2
        ...
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger


def load_intrinsics(intrinsics_path: str) -> dict:
    """
    Load intrinsics from JSON file.
    
    Returns:
        dict with keys: fx, fy, cx, cy, width, height
    """
    with open(intrinsics_path, 'r') as f:
        data = json.load(f)
    
    return {
        'fx': data['fx'],
        'fy': data['fy'],
        'cx': data['cx'],
        'cy': data['cy'],
        'width': data['width'],
        'height': data['height'],
    }


def find_video_files(folder: str) -> list[str]:
    """Find all video files in the folder."""
    videos = []
    for f in os.listdir(folder):
        if f.endswith('.mp4'):
            videos.append(os.path.join(folder, f))
    
    if not videos:
        raise FileNotFoundError(f"No .mp4 video files found in {folder}")
    
    return sorted(videos)


def run_vipe(video_path: str, output_dir: str) -> int:
    """
    Run VIPE on the video.
    
    Args:
        video_path: Path to input video
        output_dir: Directory for VIPE outputs
        
    Returns:
        Exit code (0 for success)
    """
    logger.info(f"Running VIPE on {video_path}")
    logger.info(f"Output dir: {output_dir}")
    
    cmd = f"vipe infer {video_path} --output {output_dir}"
    logger.info(f"Executing: {cmd}")
    
    result = subprocess.call(cmd, shell=True)
    
    if result != 0:
        logger.error(f"VIPE failed with exit code {result}")
    else:
        logger.info("VIPE completed successfully")
    
    return result


def load_vipe_poses(vipe_dir: str, seq_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load VIPE pose outputs.
    
    Args:
        vipe_dir: Path to VIPE results directory
        seq_name: Sequence name (video filename without extension)
        
    Returns:
        c2w: (N, 4, 4) camera-to-world matrices
        inds: (N,) frame indices
    """
    pose_path = os.path.join(vipe_dir, "pose", f"{seq_name}.npz")
    
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"VIPE pose file not found: {pose_path}")
    
    data = np.load(pose_path)
    c2w = data['data']  # (N, 4, 4) camera-to-world
    inds = data['inds']  # (N,) frame indices
    
    logger.info(f"Loaded {len(c2w)} poses from VIPE")
    return c2w, inds


def save_cameras_npz(output_path: str, w2c: np.ndarray, intrinsics: dict):
    """
    Save cameras in Dyn-HaMR format.
    
    Args:
        output_path: Path to save cameras.npz
        w2c: (N, 4, 4) world-to-camera matrices
        intrinsics: dict with fx, fy, cx, cy, width, height
    """
    N = len(w2c)
    
    # Build intrinsics array: (N, 4) [fx, fy, cx, cy]
    intrins = np.zeros((N, 4), dtype=np.float32)
    intrins[:, 0] = intrinsics['fx']
    intrins[:, 1] = intrinsics['fy']
    intrins[:, 2] = intrinsics['cx']
    intrins[:, 3] = intrinsics['cy']
    
    W = intrinsics['width']
    H = intrinsics['height']
    focal = (intrinsics['fx'] + intrinsics['fy']) / 2
    
    logger.info(f"Saving cameras to {output_path}")
    logger.info(f"Image size: {W}x{H}, focal: {focal:.2f}")
    logger.info(f"Frames: {N}")
    
    np.savez(
        output_path,
        height=H,
        width=W,
        focal=focal,
        intrins=intrins,
        w2c=w2c,
    )
    
    logger.info(f"Saved cameras.npz with {N} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VIPE poses + pre-existing intrinsics to Dyn-HaMR cameras.npz"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to data folder containing intrinsics.json and videos"
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default="intrinsics.json",
        help="Name of intrinsics JSON file (default: intrinsics.json)"
    )
    # Output argument is no longer used for filename, but could be used for filename inside the subdir
    # keeping it for compatibility but maybe it should be fixed to cameras.npz
    parser.add_argument(
        "--output",
        type=str,
        default="cameras.npz",
        help="Output filename within each video subdirectory (default: cameras.npz)"
    )
    parser.add_argument(
        "--vipe-dir",
        type=str,
        default=None,
        help="VIPE output directory (default: folder/vipe_results)"
    )
    parser.add_argument(
        "--skip-vipe",
        action="store_true",
        help="Skip running VIPE (use existing results)"
    )
    
    args = parser.parse_args()
    
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        logger.error(f"Folder not found: {folder}")
        sys.exit(1)
    
    # Load intrinsics
    intrinsics_path = os.path.join(folder, args.intrinsics)
    if not os.path.exists(intrinsics_path):
        logger.error(f"Intrinsics file not found: {intrinsics_path}")
        sys.exit(1)
    
    logger.info(f"Loading intrinsics from {intrinsics_path}")
    intrinsics = load_intrinsics(intrinsics_path)
    logger.info(f"Intrinsics: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
                f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
    
    # Find video files
    video_paths = find_video_files(folder)
    logger.info(f"Found {len(video_paths)} videos")
    
    # VIPE output directory
    vipe_dir = args.vipe_dir or os.path.join(folder, "vipe_results")
    
    for video_path in video_paths:
        video_name = Path(video_path).stem
        logger.info(f"Processing {video_name}...")
        
        # Run VIPE if needed
        pose_path = os.path.join(vipe_dir, "pose", f"{video_name}.npz")
        
        if not args.skip_vipe and not os.path.exists(pose_path):
            logger.info("VIPE results not found, running VIPE...")
            result = run_vipe(video_path, vipe_dir)
            if result != 0:
                logger.error("VIPE failed, traversing to next video")
                continue
        elif os.path.exists(pose_path):
            logger.info(f"Using existing VIPE results from {vipe_dir}")
        else:
            logger.warning(f"VIPE results not found for {video_name} and --skip-vipe specified. Skipping.")
            continue
        
        try:
            # Load VIPE poses
            c2w, inds = load_vipe_poses(vipe_dir, video_name)
            
            # Convert c2w to w2c
            w2c = np.linalg.inv(c2w)
            
            # Create video subdirectory
            video_out_dir = os.path.join(folder, video_name)
            os.makedirs(video_out_dir, exist_ok=True)
            
            # Save cameras.npz
            output_path = os.path.join(video_out_dir, args.output)
            save_cameras_npz(output_path, w2c, intrinsics)
            
        except Exception as e:
            logger.error(f"Failed to process {video_name}: {e}")
            continue

    logger.info("All processing complete!")


if __name__ == "__main__":
    main()
