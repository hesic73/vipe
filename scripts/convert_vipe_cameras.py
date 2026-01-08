import argparse
import json
import os
import subprocess
import sys
import shutil
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
        # It's possible we only have chunks from a previous run, but let's warn if empty
        logger.warning(f"No .mp4 video files found in {folder}")
    
    return sorted(videos)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(output)
    except Exception as e:
        logger.error(f"Failed to get duration for {video_path}: {e}")
        return 0.0


def split_video(video_path: str, chunk_duration: int, output_base_dir: str) -> list[str]:
    """
    Split video into chunks.
    
    Args:
        video_path: Path to source video
        chunk_duration: Max duration in seconds
        output_base_dir: Directory where the 'chunks' folder will be created
        
    Returns:
        List of paths to the generated video chunks
    """
    video_name = Path(video_path).stem
    
    # Create a dedicated directory for chunks of this video to avoid clutter
    # structure: output_base_dir/video_name_chunks/chunk_000.mp4
    chunks_dir = os.path.join(output_base_dir, f"{video_name}_chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Check if chunks already exist (simple check: if dir is not empty)
    # We do this check to avoid re-splitting on re-runs
    existing_chunks = sorted([
        os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) 
        if f.endswith('.mp4') and f.startswith(f"{video_name}_chunk_")
    ])
    
    if existing_chunks:
        logger.info(f"Found {len(existing_chunks)} existing chunks in {chunks_dir}, skipping split.")
        return existing_chunks

    logger.info(f"Splitting {video_name} into {chunk_duration}s chunks...")
    
    output_pattern = os.path.join(chunks_dir, f"{video_name}_chunk_%03d.mp4")
    
    # ffmpeg splitting
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(chunk_duration),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern
    ]
    
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg splitting failed: {e}")
        return []
        
    # Gather produced files
    chunks = sorted([
        os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) 
        if f.endswith('.mp4') and f.startswith(f"{video_name}_chunk_")
    ])
    
    # Check if the last chunk is too short (< 5s) and merge if necessary
    if len(chunks) > 1:
        last_chunk = chunks[-1]
        last_dur = get_video_duration(last_chunk)
        
        if last_dur < 5.0:
            logger.info(f"Last chunk is too short ({last_dur:.2f}s < 5s). Merging with previous chunk...")
            
            prev_chunk = chunks[-2]
            
            # Create concat file
            concat_list_path = os.path.join(chunks_dir, "concat_list.txt")
            with open(concat_list_path, "w") as f:
                f.write(f"file '{os.path.abspath(prev_chunk)}'\n")
                f.write(f"file '{os.path.abspath(last_chunk)}'\n")
            
            merged_path = os.path.join(chunks_dir, f"{video_name}_chunk_merged_temp.mp4")
            
            # FFmpeg concat
            # ffmpeg -f concat -safe 0 -i list.txt -c copy merged.mp4
            concat_cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                "-y",
                merged_path
            ]
            
            try:
                subprocess.check_call(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Replace prev_chunk with merged file
                shutil.move(merged_path, prev_chunk)
                
                # Remove last chunk
                os.remove(last_chunk)
                os.remove(concat_list_path)
                
                # Update chunks list
                chunks.pop()
                logger.info(f"Merged successfully. New last chunk: {prev_chunk}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to merge chunks: {e}")
                # If merge fails, we just keep the small chunk, better than crashing
    
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks


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
    
    cmd = f"{sys.executable} -m vipe.cli.main infer {video_path} --output {output_dir}"
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


def process_video(
    video_path: str, 
    folder: str, 
    vipe_dir: str, 
    intrinsics: dict, 
    output_filename: str, 
    skip_vipe: bool
):
    """Process a single video (original or chunk)."""
    video_name = Path(video_path).stem
    logger.info(f"Processing {video_name}...")
    
    # Run VIPE if needed
    pose_path = os.path.join(vipe_dir, "pose", f"{video_name}.npz")
    
    if not skip_vipe and not os.path.exists(pose_path):
        logger.info("VIPE results not found, running VIPE...")
        result = run_vipe(video_path, vipe_dir)
        if result != 0:
            logger.error("VIPE failed, cannot process further")
            return
    elif os.path.exists(pose_path):
        logger.info(f"Using existing VIPE results from {vipe_dir}")
    else:
        logger.warning(f"VIPE results not found for {video_name} and --skip-vipe specified. Skipping.")
        return
    
    try:
        # Load VIPE poses
        c2w, inds = load_vipe_poses(vipe_dir, video_name)
        
        # Convert c2w to w2c
        w2c = np.linalg.inv(c2w)
        
        # Create output subdirectory matching the video name (or chunk name)
        video_out_dir = os.path.join(folder, video_name)
        os.makedirs(video_out_dir, exist_ok=True)
        
        # Save cameras.npz
        output_path = os.path.join(video_out_dir, output_filename)
        save_cameras_npz(output_path, w2c, intrinsics)
        
    except Exception as e:
        logger.error(f"Failed to process {video_name}: {e}")
        # import traceback
        # logger.error(traceback.format_exc())


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
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=120,
        help="Max video duration in seconds before splitting (default: 120)"
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
    
    # Find video files (exclude files that are likely chunks if they are in the root)
    # Actually, we should probably only process "source" videos in the root. 
    # If chunks are generated, they go into subdirs, so find_video_files logic of os.listdir(folder) handles root only.
    video_paths = find_video_files(folder)
    logger.info(f"Found {len(video_paths)} videos in {folder}")
    
    # VIPE output directory
    vipe_dir = args.vipe_dir or os.path.join(folder, "vipe_results")
    
    for video_path in video_paths:
        
        # Check duration
        duration = get_video_duration(video_path)
        
        if duration > args.chunk_duration:
            logger.info(f"Video {Path(video_path).name} is longer than {args.chunk_duration}s ({duration:.2f}s). Splitting...")
            
            # Split and get list of chunk paths
            chunk_paths = split_video(video_path, args.chunk_duration, folder)
            
            # Process each chunk
            for chunk_path in chunk_paths:
                process_video(
                    chunk_path, 
                    folder, 
                    vipe_dir, 
                    intrinsics, 
                    args.output, 
                    args.skip_vipe
                )
                
        else:
            # Process normally
            process_video(
                video_path, 
                folder, 
                vipe_dir, 
                intrinsics, 
                args.output, 
                args.skip_vipe
            )

    logger.info("All processing complete!")


if __name__ == "__main__":
    main()
