# SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments


**Project Website**: [https://11chens.github.io/sea-nav](https://11chens.github.io/sea-nav/)

<p align="center">
  <img src="imgs/terser.jpg" width="80%">
</p>

---

## Installation

### 1. Environment Setup
Create a new Python virtual environment with Python 3.8:
```bash
conda create -n sea_nav python=3.8
conda activate sea_nav
```

### 2. Install Isaac Gym
- Download and install Isaac Gym Preview 4 from [NVIDIA Developer](https://developer.nvidia.com/isaac-gym).
- Install the python package:
```bash
cd isaacgym/python && pip install -e .
```

### 3. Install rsl_rl
- Clone this repository
- Install the package:
```bash
cd training/rsl_rl && pip install -e .
```

### 4. Install legged_gym
```bash
cd training/legged_gym && pip install -e .
```

---

## Usage

### Training
To start training in headless mode:
```bash
python training/legged_gym/legged_gym/scripts/train.py --headless
```

### Testing
Legacy Isaac Gym visualization and playback:
```bash
python training/legged_gym/legged_gym/scripts/play.py
```

This path is the original open-source Isaac Gym evaluation route. It is useful
as a semantic reference, but it is not the currently validated evaluation path
for the Isaac Lab migration.

Current Isaac Lab evaluation path:

- Final navigation evaluation:
```bash
python training/isaac_lab/manual_reward_probe.py \
  --scenario hard_room_eval \
  --controller-mode policy \
  --checkpoint <path-to-checkpoint>
```
- Low-level command-tracking evaluation:
```bash
python training/isaac_lab/low_level_tracking_probe.py
```

For the current Isaac Lab + RobotLab reproduction line, treat the scripts under
`training/isaac_lab/` as the active validation entrypoints.

---

## Deployment (Coming soon)
For instructions on deploying to real-world robots, please refer to the [deployment README](deployment/README.md).
