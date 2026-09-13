#!/usr/bin/env python3
"""
shot_simulator.py

Loads a pose-capture JSON file (list of frames, each frame a list of
[x, y, z] keypoints -- same format as your SMPL-30 pose files) and plays
it back as a looping, mouse-rotatable 3D animation inside a bounding box.

A Kalman filter (forward pass, constant-velocity model) is applied to
every joint/axis independently to smooth out jitter/noise before playback.

If you don't know the actual bone connections for your 30 joints, the
script will auto-infer a plausible skeleton using a minimum-spanning-tree
over the average joint positions (closest joints get connected). You can
override this with your own edge list -- see SKELETON_EDGES below.

Run it:
    python shot_simulator.py
    (it will then ask you for the JSON path)

or:
    python shot_simulator.py path/to/file.json
    python shot_simulator.py path/to/file.json --no-smooth
    python shot_simulator.py path/to/file.json --person 1
    python shot_simulator.py path/to/file.json --save playback.gif
    python shot_simulator.py path/to/file.json --no-auto-skeleton

Requirements:
    pip install numpy matplotlib
    (pip install pillow  -> only needed if you use --save output.gif)
"""

import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# EDIT THIS if you know the real bone connections for your 30 joints.
#
# Indices 0-23 follow the standard SMPL kinematic tree (the near-universal
# joint order used by SMPL / SMPL-X and virtually every SMPL-based tool):
#
#   0 Pelvis (root)   6 Spine2         12 Neck           18 L_Elbow
#   1 L_Hip           7 L_Ankle        13 L_Collar       19 R_Elbow
#   2 R_Hip           8 R_Ankle        14 R_Collar       20 L_Wrist
#   3 Spine1          9 Spine3         15 Head            21 R_Wrist
#   4 L_Knee          10 L_Foot        16 L_Shoulder      22 L_Hand
#   5 R_Knee          11 R_Foot        17 R_Shoulder      23 R_Hand
#
# This was verified against your actual data: joint Y-coordinates rise
# monotonically from feet -> knees -> hips/spine -> neck -> head in exactly
# this order, which is a strong confirmation your file follows this layout.
#
# Indices 24-29 are 6 extra points beyond the standard 24. They sit ~87
# units from the Head joint on average (vs 250-1300+ units from every other
# joint), so they're head/face landmarks (eyes, ears, nose, etc.) rather
# than fingers or toes -- attached here to Head (15) as a fan. Their exact
# individual identities (which is "left eye" vs "nose" etc.) aren't
# determinable from geometry alone; edit below if you know the real order.
# ---------------------------------------------------------------------------
SKELETON_EDGES = [
    # -- standard SMPL 24-joint kinematic tree --
    (0, 1), (0, 2), (0, 3),        # pelvis -> hips, spine1
    (1, 4), (2, 5),                # hips -> knees
    (3, 6),                        # spine1 -> spine2
    (4, 7), (5, 8),                # knees -> ankles
    (6, 9),                        # spine2 -> spine3
    (7, 10), (8, 11),              # ankles -> feet
    (9, 12), (9, 13), (9, 14),     # spine3 -> neck, collars
    (12, 15),                      # neck -> head
    (13, 16), (14, 17),            # collars -> shoulders
    (16, 18), (17, 19),            # shoulders -> elbows
    (18, 20), (19, 21),            # elbows -> wrists
    (20, 22), (21, 23),            # wrists -> hands
    # -- extra 6 points: attached to Head (data-verified nearest joint) --
    (15, 24), (15, 25), (15, 26), (15, 27), (15, 28), (15, 29),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def classify_frame(frame):
    """Return 'single', 'multi', or None based on frame nesting depth."""
    if not isinstance(frame, list) or len(frame) == 0:
        return None
    first = frame[0]
    if isinstance(first, list) and all(isinstance(v, (int, float)) for v in first):
        return "single"
    if isinstance(first, list) and len(first) > 0 and isinstance(first[0], list):
        return "multi"
    return None


def load_pose_sequence(json_path, person_index=0):
    """
    Loads a pose JSON file and returns a numpy array of shape (T, J, 3).

    Handles both:
      - single-person files: frame = [[x,y,z], ...]  (J keypoints)
      - multi-person files:  frame = [[[x,y,z],...], [[x,y,z],...], ...]
        (in which case `person_index` selects which person to play back)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Expected a non-empty list of frames at the top level.")

    kind = classify_frame(data[0])
    if kind is None:
        raise ValueError("First frame doesn't look like valid pose data.")

    frames = []
    for i, frame in enumerate(data):
        this_kind = classify_frame(frame)
        if this_kind == "single":
            frames.append(frame)
        elif this_kind == "multi":
            if person_index >= len(frame):
                raise ValueError(
                    f"Frame {i} only has {len(frame)} people; "
                    f"--person {person_index} is out of range."
                )
            frames.append(frame[person_index])
        else:
            raise ValueError(f"Frame {i} has an unrecognized shape.")

    arr = np.array(frames, dtype=float)  # (T, J, 3)
    return arr


# ---------------------------------------------------------------------------
# Kalman smoothing (per joint, per axis, independent 1D constant-velocity filter)
# ---------------------------------------------------------------------------
class KalmanFilter1D:
    def __init__(self, dt=1.0, process_var=1e-2, measurement_var=5.0):
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = process_var * np.array(
            [[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]
        )
        self.R = np.array([[measurement_var]])
        self.x = None
        self.P = None

    def init_state(self, z0):
        self.x = np.array([[z0], [0.0]])
        self.P = np.eye(2) * 100.0

    def step(self, z):
        # predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # update
        y = np.array([[z]]) - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0]


def kalman_smooth_series(series, dt=1.0, process_var=1e-2, measurement_var=5.0):
    kf = KalmanFilter1D(dt, process_var, measurement_var)
    kf.init_state(series[0])
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = kf.step(series[i])
    return out


def smooth_pose_sequence(frames, dt=1.0, process_var=1e-2, measurement_var=5.0):
    """frames: (T, J, 3) -> smoothed (T, J, 3), each joint/axis filtered independently."""
    T, J, D = frames.shape
    smoothed = np.empty_like(frames)
    for j in range(J):
        for d in range(D):
            smoothed[:, j, d] = kalman_smooth_series(
                frames[:, j, d], dt=dt, process_var=process_var, measurement_var=measurement_var
            )
    return smoothed


# ---------------------------------------------------------------------------
# Auto-skeleton via minimum spanning tree (used when SKELETON_EDGES is None)
# ---------------------------------------------------------------------------
def build_auto_skeleton(frames, rigidity_weight=0.85):
    """
    frames: (T, J, 3) full sequence (post-smoothing).

    Real bones keep a roughly *constant length* across frames because the
    skeleton is (semi-)rigid. Two joints that are merely close together in
    a handful of poses -- e.g. a hand near a knee mid-swing -- will have a
    distance that swings around a lot instead. So instead of connecting
    whatever is closest *on average* (which is what caused hand-to-knee
    links), each candidate edge is scored on:

        score = rigidity_weight * (std of distance over time / mean distance)
              + (1 - rigidity_weight) * (normalized mean distance)

    i.e. mostly "how constant is this length", with a small nudge toward
    shorter bones to break ties sensibly. The MST is then built by always
    growing the tree with the lowest-scoring (most bone-like) edge.

    This is still a heuristic, not ground truth -- if you know the real
    joint layout for your data, set SKELETON_EDGES at the top of the file
    instead; it will always be used in preference to this.
    """
    T, J, _ = frames.shape

    diffs = frames[:, :, None, :] - frames[:, None, :, :]  # (T, J, J, 3)
    dist = np.linalg.norm(diffs, axis=-1)  # (T, J, J)

    mean_dist = dist.mean(axis=0)
    std_dist = dist.std(axis=0)
    cv = std_dist / np.maximum(mean_dist, 1e-6)  # coefficient of variation

    norm_mean_dist = mean_dist / np.maximum(mean_dist.max(), 1e-6)
    norm_cv = cv / np.maximum(cv.max(), 1e-6)

    score = rigidity_weight * norm_cv + (1 - rigidity_weight) * norm_mean_dist
    np.fill_diagonal(score, np.inf)

    in_tree = np.zeros(J, dtype=bool)
    in_tree[0] = True
    edges = []
    min_edge = score[0].copy()
    nearest = np.zeros(J, dtype=int)

    for _ in range(J - 1):
        min_edge_masked = np.where(in_tree, np.inf, min_edge)
        j = int(np.argmin(min_edge_masked))
        edges.append((nearest[j], j))
        in_tree[j] = True
        better = score[j] < min_edge
        min_edge = np.where(better, score[j], min_edge)
        nearest = np.where(better, j, nearest)

    return edges


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def animate_pose(frames, skeleton_edges=None, fps=30, save_path=None, title="Shot Simulator"):
    T, J, _ = frames.shape

    all_pts = frames.reshape(-1, 3)
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    center = (mins + maxs) / 2
    max_range = max((maxs - mins).max() / 2 * 1.2, 1e-3)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    scat = ax.scatter([], [], [], c="crimson", s=35, depthshade=True)
    lines = [ax.plot([], [], [], c="steelblue", lw=2)[0] for _ in (skeleton_edges or [])]

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    def update(frame_idx):
        pts = frames[frame_idx]
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        if skeleton_edges:
            for line, (i1, i2) in zip(lines, skeleton_edges):
                line.set_data([pts[i1, 0], pts[i2, 0]], [pts[i1, 1], pts[i2, 1]])
                line.set_3d_properties([pts[i1, 2], pts[i2, 2]])
        ax.set_title(f"{title}  |  frame {frame_idx + 1}/{T}")
        return (scat, *lines)

    anim = FuncAnimation(fig, update, frames=T, interval=1000.0 / fps, blit=False, repeat=True)

    if save_path:
        writer = "pillow" if save_path.lower().endswith(".gif") else "ffmpeg"
        anim.save(save_path, writer=writer, fps=fps)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

    return anim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Play back a pose-capture JSON file as a looping 3D animation.")
    parser.add_argument("json_path", nargs="?", default=None, help="Path to the pose JSON file.")
    parser.add_argument("--person", type=int, default=0, help="Person index to play back if the file has multiple people per frame (default: 0).")
    parser.add_argument("--fps", type=int, default=30, help="Playback frame rate (default: 30).")
    parser.add_argument("--no-smooth", action="store_true", help="Disable Kalman smoothing.")
    parser.add_argument("--process-var", type=float, default=1e-2, help="Kalman process noise variance (lower = smoother but laggier).")
    parser.add_argument("--measurement-var", type=float, default=5.0, help="Kalman measurement noise variance (higher = smoother but laggier).")
    parser.add_argument("--no-auto-skeleton", action="store_true", help="Don't auto-infer a skeleton; show points only (unless SKELETON_EDGES is set).")
    parser.add_argument("--rigidity-weight", type=float, default=0.85, help="Auto-skeleton: 0-1, how much to favor constant-length bones over short bones (default: 0.85).")
    parser.add_argument("--save", default=None, help="Save the animation to a file (.gif or .mp4) instead of showing it interactively.")
    args = parser.parse_args()

    json_path = args.json_path or input("Path to pose JSON file: ").strip()

    frames = load_pose_sequence(json_path, person_index=args.person)
    print(f"Loaded {frames.shape[0]} frames, {frames.shape[1]} keypoints each.")

    if not args.no_smooth:
        frames = smooth_pose_sequence(
            frames, dt=1.0 / args.fps, process_var=args.process_var, measurement_var=args.measurement_var
        )
        print("Applied Kalman smoothing.")

    edges = SKELETON_EDGES
    if edges is None and not args.no_auto_skeleton:
        edges = build_auto_skeleton(frames, rigidity_weight=args.rigidity_weight)
        print(f"Auto-inferred a {len(edges)}-edge skeleton (MST over bone-length rigidity).")

    animate_pose(frames, skeleton_edges=edges, fps=args.fps, save_path=args.save)


if __name__ == "__main__":
    main()