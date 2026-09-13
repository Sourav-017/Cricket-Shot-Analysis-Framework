<img width="400" height="400" alt="Pull Shot " src="https://github.com/user-attachments/assets/e3c054f0-b276-499f-94e8-cb32a954f930" />


## Dataset



Distributed as a single zip for easy upload, containing 4 folders — one per shot type:

```
dataset.zip
├── defense/   *.json
├── drive/     *.json
├── pull/      *.json
└── flick/     *.json
```

Each `.json` file contains a sequence of frames capturing a single (or multi-person) 3D pose in SMPL-30 format:

- **Frames**: a list of per-frame poses, e.g. `[frame_0, frame_1, ...]`
- **Per frame**: 30 keypoints as `[x, y, z]` coordinates
  - Joints **0–23** follow the standard **SMPL 24-joint kinematic tree** (pelvis → hips/spine → knees → ankles → feet, neck → head, shoulders → elbows → wrists → hands)
  - Joints **24–29** are 6 additional **head/face landmarks**, anchored near the Head joint (15)
- Multi-person frames (if present) are nested one level deeper: `[[person_1_keypoints], [person_2_keypoints], ...]`

## Shot Simulator

`shot_simulator.py` plays back a pose file as a looping, mouse-rotatable 3D animation, with optional Kalman smoothing to reduce jitter.

### Install
```bash
pip install numpy matplotlib
pip install pillow   # only needed for --save output.gif
```

### Run
```bash
python shot_simulator.py path/to/file.json
```

### Options
| Flag | Description |
|---|---|
| `--person N` | Person index to play back (multi-person files only) |
| `--fps N` | Playback frame rate (default: 30) |
| `--no-smooth` | Disable Kalman smoothing |
| `--save out.gif` / `out.mp4` | Save animation instead of showing it interactively |
| `--no-auto-skeleton` | Show points only, no bone lines |

Run without a path argument to be prompted for one interactively.
