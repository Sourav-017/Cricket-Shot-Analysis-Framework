import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import os
import json
from pathlib import Path

# ------------------ SETTINGS ------------------
INPUT_FOLDER = "/kaggle/input/video-drive"
OUTPUT_FOLDER = "/kaggle/working/pose_coordinates_drive"
MODEL_PATH = "/kaggle/input/metrabs-model"  # Local path for MeTRAbs
SKELETON_TYPE = 'smpl+head_30'

# FRAME SHARPENING
APPLY_SHARPENING = False
SHARPEN_STRENGTH = 1.5

# SMOOTHING
APPLY_SMOOTHING = False
SMOOTHING_METHOD = 'savgol'
WINDOW_SIZE = 7
SIGMA = 2.0
ALPHA = 0.3
POLYORDER = 3

# PROCESSING OPTIONS
SAVE_FORMAT = 'both'
PROCESS_ALL_FRAMES = True  # NEW: Set to False to limit frames
MAX_FRAMES_PER_VIDEO = None  # Set to a number to limit (e.g., 15, 30, 100)
CONFIDENCE_THRESHOLD = 0.3

# PROGRESS REPORTING
REPORT_EVERY_N_FRAMES = 100  # Print progress every N frames
# ----------------------------------------------

def sharpen_frame(frame, strength=1.5):
    gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
    return cv2.addWeighted(frame, 1.0 + strength, gaussian, -strength, 0)

def smooth_poses(poses_list, method='savgol', window_size=7, sigma=2.0, alpha=0.3, polyorder=3):
    if len(poses_list) < window_size:
        print(f"⚠️ Not enough frames for smoothing")
        return poses_list
    
    valid_idx = [i for i, p in enumerate(poses_list) if p.shape[0] > 0]
    if not valid_idx:
        return poses_list
    
    max_people = max(poses_list[i].shape[0] for i in valid_idx)
    n_frames = len(poses_list)
    n_joints = poses_list[valid_idx[0]].shape[1]
    coord_dim = poses_list[valid_idx[0]].shape[2]  # Auto detect: 2D or 3D
    
    padded = np.zeros((n_frames, max_people, n_joints, coord_dim))
    for i in valid_idx:
        padded[i, :poses_list[i].shape[0]] = poses_list[i]
    
    smoothed = padded.copy()
    for person in range(max_people):
        for j in range(n_joints):
            for c in range(coord_dim):
                signal = padded[:, person, j, c]
                if np.all(signal == 0):
                    continue
                
                if method == 'savgol':
                    filtered = savgol_filter(signal, window_size, polyorder)
                elif method == 'gaussian':
                    filtered = gaussian_filter1d(signal, sigma=sigma)
                elif method == 'moving_average':
                    filtered = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
                elif method == 'exponential':
                    filtered = np.zeros_like(signal)
                    filtered[0] = signal[0]
                    for t in range(1, len(signal)):
                        filtered[t] = alpha * signal[t] + (1 - alpha) * filtered[t-1]
                else:
                    filtered = signal
                
                smoothed[:, person, j, c] = filtered
    
    final = []
    for i in range(n_frames):
        valid_people = poses_list[i].shape[0]
        if valid_people > 0:
            final.append(smoothed[i, :valid_people])
        else:
            final.append(np.zeros((0, n_joints, coord_dim)))
    
    return final

def process_video(video_path, model, output_dir):
    video_name = os.path.basename(video_path)
    print(f"\n=== Processing: {video_name} ===")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Total frames: {total_frames} | FPS: {fps} | Resolution: {width}x{height}")
    
    all_poses_3d, all_poses_2d, frame_nums = [], [], []
    frame_id, processed = 0, 0
    
    # Determine which frames to process
    if PROCESS_ALL_FRAMES or MAX_FRAMES_PER_VIDEO is None:
        target_frames = set(range(1, total_frames + 1))
        print(f" Processing ALL {total_frames} frames")
    else:
        if total_frames <= MAX_FRAMES_PER_VIDEO:
            target_frames = set(range(1, total_frames + 1))
        else:
            indices = np.linspace(0, total_frames - 1, MAX_FRAMES_PER_VIDEO, dtype=int)
            target_frames = set([i + 1 for i in indices])
        print(f" Processing {len(target_frames)} frames (sampled)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        if frame_id not in target_frames:
            continue
        
        processed += 1
        
        # Progress reporting
        if processed % REPORT_EVERY_N_FRAMES == 0:
            print(f" Processed {processed}/{len(target_frames)} frames...")
        
        if APPLY_SHARPENING:
            frame = sharpen_frame(frame, SHARPEN_STRENGTH)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = tf.convert_to_tensor(rgb)
        
        try:
            pred = model.detect_poses(img_tensor, skeleton=SKELETON_TYPE)
            boxes = pred.get('boxes', None)
            
            if boxes is not None:
                conf = boxes[:, 4].numpy()
                mask = conf >= CONFIDENCE_THRESHOLD
                poses3d = pred['poses3d'].numpy()[mask]
                poses2d = pred['poses2d'].numpy()[mask]
            else:
                poses3d = pred['poses3d'].numpy()
                poses2d = pred['poses2d'].numpy()
        
        except Exception as e:
            print(f"⚠ Error frame {frame_id}: {e}")
            poses3d = np.zeros((0, 30, 3))
            poses2d = np.zeros((0, 30, 2))
        
        all_poses_3d.append(poses3d)
        all_poses_2d.append(poses2d)
        frame_nums.append(frame_id)
    
    cap.release()
    print(f"✔ Extracted {processed} frames")
    
    # Smoothing
    if APPLY_SMOOTHING:
        print(f" Applying {SMOOTHING_METHOD} smoothing...")
        all_poses_3d = smooth_poses(all_poses_3d, SMOOTHING_METHOD, WINDOW_SIZE, SIGMA, ALPHA, POLYORDER)
        all_poses_2d = smooth_poses(all_poses_2d, SMOOTHING_METHOD, WINDOW_SIZE, SIGMA, ALPHA, POLYORDER)
    
    # Save output folder
    out_dir = os.path.join(output_dir, Path(video_name).stem)
    os.makedirs(out_dir, exist_ok=True)
    
    return save_pose_outputs(model, video_name, out_dir, fps, width, height, frame_nums,
                             all_poses_3d, all_poses_2d)

def save_pose_outputs(model, video_name, out_dir, fps, width, height,
                      frame_nums, poses3d_list, poses2d_list):
    joint_names = model.per_skeleton_joint_names[SKELETON_TYPE].numpy().astype(str).tolist()
    
    data_json = {
        "video_name": video_name,
        "fps": fps,
        "resolution": [width, height],
        "total_frames_processed": len(frame_nums),
        "joint_names": joint_names,
        "frames": []
    }
    
    for f, p3, p2 in zip(frame_nums, poses3d_list, poses2d_list):
        data_json["frames"].append({
            "frame_number": int(f),
            "num_people": int(p3.shape[0]),
            "poses_3d": p3.tolist(),
            "poses_2d": p2.tolist()
        })
    
    if SAVE_FORMAT in ["json", "both"]:
        json_path = os.path.join(out_dir, "pose_data.json")
        with open(json_path, "w") as f:
            json.dump(data_json, f, indent=2)
        print(f" Saved JSON: {json_path}")
    
    if SAVE_FORMAT in ["npy", "both"]:
        npy_path = os.path.join(out_dir, "pose_data.npy")
        np.save(npy_path, {
            "video_name": video_name,
            "joint_names": joint_names,
            "frame_numbers": frame_nums,
            "poses_3d": poses3d_list,
            "poses_2d": poses2d_list
        }, allow_pickle=True)
        print(f" Saved NPY: {npy_path}")
    
    print(f"✔ Saved results for {video_name}")
    return True

def main():
    print("\n=== MeTRAbs Batch Pose Extractor (ALL FRAMES) ===\n")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load model ONCE
    print(" Loading model...")
    model = hub.load(MODEL_PATH)
    print(" Model loaded!")
    
    # Find videos
    video_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(INPUT_FOLDER)
        for f in files
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]
    
    if not video_files:
        print("❌ No videos found!")
        return
    
    print(f"✔ Found {len(video_files)} videos")
    
    success = 0
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(video_files)}] {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        if process_video(video_path, model, OUTPUT_FOLDER):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"=== PROCESSING COMPLETE ===")
    print(f"{'='*60}")
    print(f"✔ Successfully processed: {success}/{len(video_files)} videos")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
