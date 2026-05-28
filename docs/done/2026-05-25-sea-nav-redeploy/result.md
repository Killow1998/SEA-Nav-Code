# SEA-Nav Redeploy Result

Date: 2026-05-25

Current note:

- this document is the historical migration/run record from the first Isaac Lab redeploy phase
- the current retained reproduction verdict, asset map, and cleanup record now live under:
  - `logs/isaac_lab/reproduction_audit_20260528.md`
  - `logs/isaac_lab/asset_index_20260528.md`
  - `logs/isaac_lab/cleanup_recommendations_20260528.md`

## Goal Pair

- Goal source: `docs/done/2026-05-25-sea-nav-redeploy/goal.md`
- Final status page: `docs/done/2026-05-25-sea-nav-redeploy/status.html`

## Final Runtime

- Persistent Python env: `/home/user/rl_redeploy/.venvs/sea-nav-ilab`
- `uv` cache only: `/tmp/sea-nav-uv-cache`
- USD cache only: `/tmp/sea-nav-usd-cache`
- Python: `3.11.14`
- Isaac Lab: `2.3.2.post1`
- Isaac Sim: `5.1.0.0`
- PyTorch: `2.7.0+cu128`
- torchvision: `0.22.0+cu128`
- GPU target: `NVIDIA GeForce RTX 5060 Ti`

`/tmp/sea-nav-ilab-venv` was used only as a transient bootstrap source during migration. The final runnable environment is the persistent path above.

## Initial Audit Conclusion

Before migration, this repo only had the Isaac Gym path:

- `README.md` and `training/legged_gym/.../train.py` were Isaac Gym based.
- `deployment/README.md` was still empty.
- There was no Isaac Lab launcher, task package, or log-view path for SEA-Nav.

That meant the repo was a migration source, not a ready-to-run Isaac Lab target.

## Final Implementation

- `training/isaac_lab/sea_nav_env.py`
  - ports the SEA-Nav observation, reward, terrain, and low-level controller flow into Isaac Lab
  - exports `bad_masks` before reset, matching the original Isaac Gym timing
  - reports episode reward terms with the original `rew_*` naming
- `training/isaac_lab/train.py`
  - launches Isaac Lab through `AppLauncher`
  - converts the local Go2 URDF to USD under `/tmp/sea-nav-usd-cache`
  - wraps the env for the repo-local `training/rsl_rl` PPO runner
  - defaults to the original `OnPolicyRunner.learn(...)` path for training semantics
  - keeps the custom stdout + TensorBoard loop only as an explicit `--training-loop metrics` fallback

## Persistent Environment Build Notes

Environment creation command:

```bash
uv venv /home/user/rl_redeploy/.venvs/sea-nav-ilab --python 3.11
```

What actually worked:

1. Build the persistent venv under `/home/user/rl_redeploy/.venvs`.
2. Reuse `/tmp/sea-nav-uv-cache` only as the download cache.
3. Sync the already verified temporary Isaac Lab runtime into the persistent env:

```bash
rsync -a /tmp/sea-nav-ilab-venv/ /home/user/rl_redeploy/.venvs/sea-nav-ilab/
```

4. Remove conflicting `trimesh` copies and keep the working version:

```bash
env UV_CACHE_DIR=/tmp/sea-nav-uv-cache uv pip uninstall \
  --python /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python \
  trimesh

env UV_CACHE_DIR=/tmp/sea-nav-uv-cache uv pip install \
  --python /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python \
  trimesh==4.5.1
```

The persistent env now freezes to the same Isaac Sim package set as the working runtime and uses the verified `torch==2.7.0+cu128` and `torchvision==0.22.0+cu128`.

## Verified Final Training Command

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --num-envs 4 \
  --num-steps-per-env 8 \
  --max-iterations 1 \
  --experiment-name Go2_pos_rough_isaaclab_gpu_persistent_venv
```

Verified result:

- `reset complete obs_shape=(4, 550) action_dim=3 device=cuda:0`
- `[TRAIN] iter=0 rollout_reward_mean=0.2568 value_loss=2.2655 surrogate_loss=0.0012 regularization_loss=0.0019 smooth_loss=0.0083 interv_loss=0.0000`
- `tensorboard_dir=/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_persistent_venv/05_25_15-06-13/tensorboard`
- `training finished log_dir=/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_persistent_venv/05_25_15-06-13`

Artifacts verified from that run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_persistent_venv/05_25_15-06-13/model_1.pt`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_persistent_venv/05_25_15-06-13/tensorboard/events.out.tfevents.1779692774.jammy.40397.0`

## Semantic Alignment Fix

After the first end-to-end migration worked, the default Isaac Lab entrypoint still differed from the original Isaac Gym semantics in two material places:

- the default training loop was a custom `_learn_with_metrics(...)` path rather than `OnPolicyRunner.learn(...)`
- `bad_masks` were being exported after reset, not before reset as in the original environment

That has now been corrected.

Current default behavior:

- `training/isaac_lab/train.py` now uses the original `runner.learn(...)` path by default
- `training/isaac_lab/sea_nav_env.py` now exposes `extras["bad_masks"]` from `_get_dones()` before reset happens
- the optional custom TensorBoard loop is still available, but only when explicitly requested with `--training-loop metrics`

Verified semantic-alignment command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --num-envs 32 \
  --num-steps-per-env 8 \
  --max-iterations 21 \
  --experiment-name Go2_pos_rough_isaaclab_semantic_replay_check
```

Verified output from the default original loop:

- `runner ready training_loop=original obs_dim=550 action_dim=3 device=cuda:0`
- `Iteration: 20`
- `Value function loss: 21660393.2500`
- `Regularization loss: 3.1955`
- `Smooth loss: 63.9024`
- `Mean episode rew_collision: -60.3179`
- `training finished log_dir=/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_semantic_replay_check/05_25_16-45-48`

Artifacts verified from that semantic-alignment run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_semantic_replay_check/05_25_16-45-48/model_21.pt`

## Joint Mapping And GUI Runtime Fix

After the semantic-alignment run above, two remaining non-trivial mismatches were confirmed:

- the Isaac Lab articulation joint order was not the same as the original Isaac Gym low-level controller order
- the non-headless GUI path mixed the venv `numpy 2.4.4` with Isaac Sim's bundled `numpy 1.26.0`, which caused the earlier `numpy.dtype size changed` crash

Confirmed live Isaac Lab joint order from the actual articulation:

```text
FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint
```

Original low-level controller order inferred from the original Isaac Gym `reindex([3,4,5,0,1,2,9,10,11,6,7,8])` semantics:

```text
FR_hip_joint, FR_thigh_joint, FR_calf_joint,
FL_hip_joint, FL_thigh_joint, FL_calf_joint,
RR_hip_joint, RR_thigh_joint, RR_calf_joint,
RL_hip_joint, RL_thigh_joint, RL_calf_joint
```

What was changed:

- `training/isaac_lab/sea_nav_env.py`
  - removed the incorrect hard-coded reuse of the Isaac Gym `REINDEX` permutation against Isaac Lab live articulation order
  - now builds the low-level observation/action permutations dynamically from `self._robot.joint_names`
- `training/isaac_lab/train.py`
  - delays `torch` and `SummaryWriter` imports until after `AppLauncher(args).app`
  - this prevents the GUI path from preloading the venv `numpy` before Isaac Sim installs its runtime import hooks

Important correction:

- the previous long run under `Go2_pos_rough_isaaclab_gpu_train_original_semantics/05_25_16-50-28` should not be used as the final quality judgment for the task, because it was executed before the joint-order fix and therefore did not preserve the original low-level controller semantics

Verified host-side headless smoke after the fix:

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --num-envs 1 \
  --smoke-steps 1 \
  --experiment-name Go2_pos_rough_isaaclab_mapping_smoke_host
```

Verified output:

- `low-level joint mapping live_joint_names=[...] low_level_joint_names=[...]`
- `reset complete obs_shape=(1, 550) action_dim=3 device=cuda:0`
- `[SMOKE] step=1 reward_mean=0.0252 terminated=0 truncated=0 obs_shape=(1, 550)`

Verified host-side non-headless GUI smoke after the fix:

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --num-envs 1 \
  --smoke-steps 5 \
  --experiment-name Go2_pos_rough_isaaclab_gui_smoke_verify
```

Verified GUI-path result:

- `isaacsim.sensors.camera`, `isaaclab_assets`, and `isaaclab_tasks` all started without the earlier `numpy.dtype size changed` failure
- the windowed path progressed into viewport startup and RTX shader compilation instead of crashing during Python extension import

Verified short training after the joint-order fix:

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --num-envs 32 \
  --num-steps-per-env 8 \
  --max-iterations 21 \
  --experiment-name Go2_pos_rough_isaaclab_jointfix_shortcheck
```

Verified iteration-20 result:

- `Value function loss: 11863.8458`
- `Regularization loss: 0.0017`
- `Smooth loss: 0.0316`
- `Interv loss: 0.0026`
- `Mean reward: -7731.84`
- `Mean episode rew_collision: -63.5814`
- `training finished log_dir=/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_jointfix_shortcheck/05_25_19-57-14`

Artifacts verified from the joint-fix short training run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_jointfix_shortcheck/05_25_19-57-14/model_21.pt`

## Post-Fix Long GUI Training Launch

This is the first full-length training relaunch after both critical fixes:

- live articulation joint mapping is now aligned to the original low-level controller semantics
- the non-headless GUI path no longer crashes during Isaac Sim extension startup from mixed `numpy` ABIs

Launch start time:

- `2026-05-25 20:01:44 CST`

Planned long GUI launch command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --training-loop original \
  --num-envs 256 \
  --num-steps-per-env 48 \
  --max-iterations 2000 \
  --experiment-name Go2_pos_rough_isaaclab_jointfix_gui_longrun
```

Expected operator-visible behavior:

- the Isaac Sim GUI window should open on the host desktop
- the run should use the repaired original-semantics path rather than the custom metrics loop
- logs should be tailed from a dedicated stdout file instead of relying on the interactive terminal session

Verified live launch status:

- active host OS PID: `143633`
- active Codex PTY session id: `50427`
- active run directory: `logs/isaac_lab/Go2_pos_rough_isaaclab_jointfix_gui_longrun/05_25_20-04-01`
- TensorBoard event file already created:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_jointfix_gui_longrun/05_25_20-04-01/tensorboard/events.out.tfevents.1779710642.jammy.143633.0`
- verified startup state:
  - `Creating window for environment.`
  - `Completed setting up the environment...`
  - `runner ready training_loop=original obs_dim=550 action_dim=3 device=cuda:0`
- first verified live training summary:
  - `Iteration: 20`
  - `Value function loss: 10231919.4500`
  - `Regularization loss: 0.3306`
  - `Smooth loss: 6.6127`
  - `Mean reward: -10009.14`
  - `Mean episode rew_collision: -85.4909`

Later observed state before termination:

- checkpoint files reached:
  - `model_900.pt`
  - `model_1000.pt`
- last complete printed training block reached `Iteration: 1010`
- `Mean episode rew_reach_pos_target_tight` was still `0.0000`
- the run did not end cleanly; the GUI path terminated with:
  - `omni.physx.plugin: Subscription cannot be changed during the event call.`
  - `carb.tasking Mutex::lock assertion failed: Recursion not allowed`
  - `terminate called without an active exception`

## Original-Semantics Long Training Relaunch

The next full training run should use the now-correct default original semantics rather than the older custom metrics loop.

Planned launch command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --training-loop original \
  --num-envs 256 \
  --num-steps-per-env 48 \
  --max-iterations 2000 \
  --experiment-name Go2_pos_rough_isaaclab_gpu_train_original_semantics
```

Live stdout log:

```text
/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/sea-nav-train-gpu-original.log
```

Historical launch status before the joint-order fix:

- launched in background on 2026-05-25 with detached PID `62420`
- active run directory: `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_train_original_semantics/05_25_16-50-28`
- first verified printed training summary reached `Iteration: 20`
- first verified summary values:
  - `Value function loss: 86422542.4000`
  - `Regularization loss: 5.6560`
  - `Smooth loss: 113.1202`
  - `Mean reward: -7981.73`
- after completion, TensorBoard events were backfilled for this run under `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_train_original_semantics/05_25_16-50-28/tensorboard`
- this run remains useful as an implementation milestone, but not as the final training-quality verdict because it predates the joint-order fix
- future original-semantics runs now write TensorBoard directly from `OnPolicyRunner` without changing training semantics

## Metric Access

Default training path:

```text
OnPolicyRunner.learn(...)
```

Optional TensorBoard path:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --training-loop metrics \
  --num-envs 32 \
  --num-steps-per-env 8 \
  --max-iterations 21 \
  --experiment-name Go2_pos_rough_isaaclab_metrics_check
```

If you use the metrics loop, TensorBoard root is still:

```bash
tensorboard --logdir logs/isaac_lab
```

Original loop stdout format:

```text
Iteration: ...
Value function loss: ...
Surrogate loss: ...
Regularization loss: ...
Smooth loss: ...
```

## Long Training Launched

Managed service:

```text
sea-nav-train-gpu.service
```

Live stdout log:

```text
/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/sea-nav-train-gpu.log
```

Current long-run directory:

```text
/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_train_rtx5060ti/05_25_15-20-53
```

Launch command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --num-envs 256 \
  --num-steps-per-env 48 \
  --max-iterations 2000 \
  --experiment-name Go2_pos_rough_isaaclab_gpu_train_rtx5060ti
```

Observed early training lines:

- `[TRAIN] iter=0 rollout_reward_mean=-205.8046 value_loss=42441827.4000 surrogate_loss=0.0152 regularization_loss=0.0001 smooth_loss=0.0012 interv_loss=0.0006 episode_reward_mean=-6504.3117 episode_length_mean=27.86`
- `[TRAIN] iter=19 rollout_reward_mean=-356.0815 value_loss=136928671.2000 surrogate_loss=0.0050 regularization_loss=1.6485 smooth_loss=32.9694 interv_loss=0.0008 episode_reward_mean=-8339.4184 episode_length_mean=26.58`

Observed speed on this machine:

- the first 20 iterations were already present while the managed service had been up for about 43 seconds
- that implies roughly `2.1` to `2.2` seconds per iteration including warm-up
- later, when the managed service had been up for about `2` minutes `12` seconds, the log had already reached `iter=75`
- that later sample suggests the long run is currently closer to about `1.7` seconds per iteration overall
- if that later pace stays roughly stable, `2000` iterations would be about `55` to `60` minutes, not "a few minutes"

Live TensorBoard directory for the long run:

```text
/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_train_rtx5060ti/05_25_15-20-53/tensorboard
```

Note:

- checkpoint files for this long run will appear from iteration `100` onward because the current save interval is `100`

## Notes

- CPU smoke is still useful only as an entry-path check, not as the success criterion.
- URDF import still warns about unresolved `Head_upper` and `Head_lower` visual references, but they did not block the verified GPU training start.
- A too-small PPO verification batch can still produce invalid mini-batches and `nan` loss terms; the verified command above avoids that case.

## Conclusion

The final deliverable is no longer tied to a reboot-unsafe `/tmp` environment. The persistent venv at `/home/user/rl_redeploy/.venvs/sea-nav-ilab` can start the migrated SEA-Nav Isaac Lab training path, emit reward/loss metrics, write TensorBoard logs, and save checkpoints.

## Manual Reward Probes

To avoid wasting more time on unstable PPO runs, training was stopped and replaced with deterministic scripted probes in simplified scenes using the same Isaac Lab environment and reward code, but without policy output.

Probe utility:

```text
/home/user/rl_redeploy/SEA-Nav-Code/training/isaac_lab/manual_reward_probe.py
```

What this probe does:

- builds a single-room preset map instead of random terrain generation
- uses `set_manual_start_and_goal(...)` to place the robot and goal deterministically
- uses zero settle steps before scripted motion to check for invalid spawn
- drives the navigation command directly from a scripted controller
- records per-step reward, distance, yaw, command, and termination state

### Straight Corridor Probe

Command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u \
  training/isaac_lab/manual_reward_probe.py \
  --headless \
  --scenario straight \
  --max-steps 420 \
  --settle-steps 10
```

Run directory:

```text
/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_straight/05_25_20-55-10
```

Observed result:

- `settle_invalid=false`: the deterministic spawn did not immediately fail
- `reach_reward_sum=0.0`: no goal-reaching reward was ever triggered
- `min_distance=0.7049`: the robot approached the goal but never crossed the `0.5m` success threshold
- `done_step=null`: no episode termination happened in 420 steps
- `total_reward_sum=-4166.43`: cumulative reward became negative even though the robot moved toward the goal

Key trace evidence:

- around steps `395-405`, the robot hovered near the goal at `0.70m` distance but still had `reach_reward=0.0`
- repeated large negative spikes appeared before and after that point:
  - `step=310 reward=-429.32 distance=1.0938`
  - `step=398 reward=-222.60 distance=0.7098`
  - `step=417 reward=-208.64 distance=0.7078`

Interpretation:

- because this run never terminated, the only plausible large negative term is collision penalty, not termination penalty
- the robot can earn positive dense progress reward while moving straight, but side-wall scraping wipes out that gain before the `0.5m` reach gate is satisfied

### Shortened L-Corridor Turn Probe

Command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u \
  training/isaac_lab/manual_reward_probe.py \
  --headless \
  --scenario turn \
  --max-steps 520 \
  --settle-steps 10
```

Run directory:

```text
/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn/05_25_20-55-09
```

Observed result:

- `settle_invalid=false`: deterministic spawn again started cleanly
- the scripted controller did switch into turn mode; trace rows around `step=207-247` show `command=[0.35, 0.0, 1.0]`
- `reach_reward_sum=0.0`: no goal-reaching reward was triggered
- `min_distance=3.0055`: the robot never got close to the goal
- `done_step=287`, `done_reason=terminated`: the episode eventually crashed out
- `total_reward_sum=-12356.92`: much worse than straight-line probing

Key trace evidence:

- strong negative spikes already appeared during the turn:
  - `step=215 reward=-427.77 yaw=0.3287`
  - `step=246 reward=-461.91 yaw=0.7321`
- final termination row:
  - `step=287 reward=-350.72 distance=7.7369 terminated=true`

Interpretation:

- the turn command is being issued, so this is not a "script never turned" false alarm
- however, turning in the corridor produces heavy collision penalties and then a terminal failure before the goal is reached

### Reset / Invalid Data Note

The user's GUI observation about the robot "falling from the sky" is real:

- the migrated Isaac Lab reset currently writes root height as a fixed `0.42m`
- it randomizes root linear and angular velocity in `[-0.5, 0.5]`
- it also randomizes yaw on every reset

Relevant code:

```text
/home/user/rl_redeploy/SEA-Nav-Code/training/isaac_lab/sea_nav_env.py
```

Specifically:

- `_reset_idx(...)` sets `root_state[i, 2] = 0.42 + env_origins[i, 2]`
- `_reset_idx(...)` sets `root_state[i, 7:13] = uniform(-0.5, 0.5)`
- `_reset_idx(...)` randomizes yaw in `[-pi, pi]`

This behavior is inherited from the original Isaac Gym training semantics rather than introduced by the migration. The deterministic scripted probe bypassed those random reset terms with `set_manual_start_and_goal(...)`, and both simplified probes reported `settle_invalid=false`, so the clean-start path itself is viable.

### Probe Takeaway

These manual probes show that the current bottleneck is not "PPO did not run long enough":

- straight-line progress reward exists, but the robot still fails the final `distance < 0.5m` reach gate
- turn behavior is materially worse: the scripted controller turns, but collision penalties dominate and the episode terminates
- further full-length PPO training would mostly optimize against a reward surface that the current low-level execution cannot reliably satisfy in these simple corridors

## Low-Level Tracking and Actuator Path Findings

To separate reward-design issues from execution-path issues, an open-space command-tracking probe was added:

```text
/home/user/rl_redeploy/SEA-Nav-Code/training/isaac_lab/low_level_tracking_probe.py
```

The probe uses a large mostly empty room, deterministic manual spawn, and short command windows to measure:

- requested navigation command
- filtered command after the task-side command filter
- actual base-frame `vx`, `vy`, and `wz`
- RMSE between the requested command and the realized base motion

### Important reset discovery

The tracking probe initially appeared to hang at `env.reset()`. The cause was not Isaac Lab itself. It came from the SEA-Nav task's room sampler:

- `place_robot_and_goal(...)` only returns when the line from start to goal is blocked by an obstacle
- a truly empty room therefore makes reset loop forever

For tracking probes, this was worked around by adding a far-corner obstacle island that does not affect the central test area but allows the original reset logic to complete.

### Implicit actuator tracking

Current default execution path before this investigation:

- Isaac Lab `ImplicitActuatorCfg`
- low-level action mapped to `set_joint_position_target(...)`

Relevant probe outputs:

- forward command, implicit actuator:
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/low_level_tracking_probe/05_25_21-39-27`
  - requested `vx=1.0`, filtered `vx=0.997`
  - actual mean `vx=0.487`, `vy=0.041`, `wz=-0.178`
  - this means only about half of the requested forward speed is realized, with noticeable spontaneous yaw drift
- lateral / yaw / forward-turn command, implicit actuator:
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/low_level_tracking_probe/05_25_21-39-49`
  - lateral requested `vy=0.5`, actual mean `vy=0.116`
  - yaw requested `wz=0.8`, actual mean `wz=0.674`
  - forward-turn requested `(vx=0.35, wz=1.0)`, actual mean `(vx=0.179, wz=0.899)`

Interpretation:

- yaw tracking is usable
- forward tracking is significantly compressed
- lateral tracking is especially weak
- this by itself explains why the navigation layer can command valid-looking motions while the robot still fails to center itself and enter the final `0.5m` success region

### Explicit PD actuator comparison

The Isaac Lab environment was updated to support actuator mode switching:

- `implicit`
- `ideal_pd`

Affected files:

```text
/home/user/rl_redeploy/SEA-Nav-Code/training/isaac_lab/sea_nav_env.py
/home/user/rl_redeploy/SEA-Nav-Code/training/isaac_lab/train.py
/home/user/rl_redeploy/SEA-Nav-Code/training/isaac_lab/manual_reward_probe.py
/home/user/rl_redeploy/SEA-Nav-Code/training/isaac_lab/low_level_tracking_probe.py
```

This allows direct A/B between the previous Isaac Lab path and a path that is closer to the original Isaac Gym semantics.

Probe outputs with `--actuator-mode ideal_pd`:

- forward command:
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/low_level_tracking_probe/05_25_21-41-34`
  - requested `vx=1.0`, filtered `vx=0.997`
  - actual mean `vx=0.521`, `vy=-0.034`, `wz=-0.029`
- lateral command:
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/low_level_tracking_probe/05_25_21-41-57`
  - requested `vy=0.5`, filtered `vy=0.498`
  - actual mean `vy=0.158`

Interpretation:

- switching to `ideal_pd` does not fully fix under-tracking
- however, it materially reduces unwanted yaw drift during straight motion
- this is strong evidence that the actuator path mismatch was a real blocker, not a coincidence
- at the same time, the persistent weak lateral tracking shows that actuator choice is not the only issue

### Straight and turn probes with IdealPD

After enabling `--actuator-mode ideal_pd`, the simplified corridor probes changed meaningfully.

- straight corridor with `ideal_pd`:
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_straight/05_25_21-42-47`
  - `reach_reward_sum=250.84`
  - `min_distance=0.1723`
  - `first_reach_step=390`
  - `total_reward_sum=2131.56`
  - this is the first deterministic probe that truly entered the goal region and collected reach reward
- shortened turn corridor with `ideal_pd`:
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn/05_25_21-43-08`
  - `reach_reward_sum=0.0`
  - `min_distance=3.0731`
  - `done_step=226`, `done_reason=terminated`
  - `total_reward_sum=475.44`

Interpretation:

- the actuator-path fix is sufficient to recover simple straight-line goal reaching
- it is not yet sufficient to recover turning navigation in the corridor task
- therefore, the next training attempt should not use the old implicit actuator path, but a simple switch to `ideal_pd` is also not the whole solution

### Default execution path switched to IdealPD

To prevent accidental fallback to the already-invalidated Isaac Lab execution path, the default actuator mode was changed from `implicit` to `ideal_pd` in both the CLI entrypoints and the environment builder:

- `training/isaac_lab/train.py`
- `training/isaac_lab/manual_reward_probe.py`
- `training/isaac_lab/low_level_tracking_probe.py`
- `training/isaac_lab/sea_nav_env.py`

This means future training and probe runs now default to the actuator mode that is closest to the original Isaac Gym semantics, unless explicitly overridden.

### Additional turn-scene takeover probes after IdealPD default

Three manual controllers were tested on the same simplified turn scene, all with `ideal_pd`:

- scripted baseline:
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn/05_25_21-51-58`
  - `reach_reward_sum=0.0`
  - `min_distance=3.0731`
  - `done_step=226`
- feedback heading controller (`goal_heading`):
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn/05_25_21-52-41`
  - `reach_reward_sum=0.0`
  - `min_distance=4.4002`
  - `done_step=129`
  - this controller was too conservative and failed even earlier
- piecewise pivot-turn controller (`pivot_turn`):
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn/05_25_21-53-24`
  - `reach_reward_sum=0.0`
  - `min_distance=3.0207`
  - `done_step=358`
  - `rew_collision=-78.06`

The `pivot_turn` trace is especially informative:

- it does complete a near-90-degree reorientation, reaching `yaw≈1.39rad` before resuming forward motion
- even after the turn is mostly complete, distance only improves from about `3.09m` to `3.02m`
- it then terminates with a large collision spike instead of entering the goal region

Interpretation:

- turn-scene failure is not just a consequence of one bad scripted policy
- after the actuator-path fix, the remaining gap is now concentrated in turning / corridor navigation rather than straight-line locomotion
- at this stage, continuing long PPO training without resolving the turn-scene gap would still be premature

### Runtime note for future Isaac Sim probe runs

Inside the current Codex sandbox, Isaac Sim headless probe runs can fail to enumerate the real GPU and report `no CUDA-capable device is detected`. The same commands succeed when re-run outside the sandbox. For this repo, GPU-backed Isaac Sim probe/training commands should be treated as requiring escalated execution when launched from Codex.

### Wider and more open turn scenes still fail

To check whether the original turn probe was simply too tight, two easier variants were added and tested with the same `pivot_turn` controller:

- widened L corridor:
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn_wide/05_25_21-55-08`
  - `reach_reward_sum=0.0`
  - `min_distance=3.9746`
  - `done_step=439`
  - `rew_collision=-445.46`
- open-corner turn scene:
  - log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn_open/05_25_21-57-44`
  - `reach_reward_sum=0.0`
  - `min_distance=4.1137`
  - `done_step=355`
  - `rew_collision=-58.92`

Interpretation:

- failure is not limited to one overly tight corner geometry
- even when the corner is widened or opened up, the current control stack still fails to drive the robot through a simple turn and into the goal region
- this makes it much less likely that the remaining issue is only about the exact wall placement in the probe scene

### PhysX velocity-stability tweak was tested and rejected

Isaac Lab warned that noisy velocities may improve if `enable_external_forces_every_iteration=True` and velocity iterations are increased. This was tested by temporarily:

- setting `SimulationCfg.physx.enable_external_forces_every_iteration=True`
- setting `SimulationCfg.physx.min_velocity_iteration_count=1`
- setting articulation `solver_velocity_iteration_count=1`

The result was worse, not better. Tracking probe log:

- `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/low_level_tracking_probe/05_25_21-58-43`

Key regressions compared with the earlier `ideal_pd` baseline:

- yaw command `wz=0.8` dropped from actual mean about `0.674` to `0.329`
- forward-turn command `(vx=0.35, wz=1.0)` dropped from actual mean about `(0.179, 0.899)` to `(0.120, 0.405)`

Therefore this PhysX tweak was reverted immediately and is not part of the retained code path.

### Open-corner turn still fails with a stronger hybrid takeover controller

To avoid blaming the result on a too-simple second-leg policy, a fourth takeover controller was added:

- `pivot_then_heading`
  - first leg: straight
  - corner: pivot in place until roughly aligned
  - second leg: switch to goal-heading feedback

Probe result on the open-corner scene:

- log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn_open/05_25_22-03-41`
- `reach_reward_sum=0.0`
- `min_distance=4.1581`
- `done_step=468`
- `rew_collision=-57.20`

Interpretation:

- failure is not explained away by using a weak scripted second leg
- even with a more reasonable staged controller, the current Isaac Lab port still cannot complete a simplified turn-and-reach task

### Probe accounting fix for terminal steps

The manual probe was updated to cache pre-reset state and reward terms inside the environment, because `DirectRLEnv.step()` computes rewards before reset but returns after reset. Without this, terminal-step diagnostics could accidentally read the new episode state instead of the last failing state.

Affected files:

- `training/isaac_lab/sea_nav_env.py`
- `training/isaac_lab/manual_reward_probe.py`

This change does not alter training behavior; it only makes the manual takeover diagnostics more faithful around termination.

### Native Go2 asset still fails the same open-corner turn

To check whether the remaining turn blocker was mainly caused by the converted URDF-based robot asset, I ran the corrected open-corner turn probe with Isaac Lab's native Go2 asset:

- log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn_open/05_25_22-13-01`
- controller: `pivot_then_heading`
- asset source: `native_go2`
- result:
  - `reach_reward_sum=0.0`
  - `min_distance=4.1347`
  - `done_step=488`
  - `done_reason=terminated`

This rules out the simplest explanation that only the converted USD asset was wrong. The open-corner turn blocker persists even on Isaac Lab's shipped native Go2 articulation.

### Manual pivot target was over-rotated, but turn is still not solved

The original hand-scripted turn controller pivoted toward `yaw=1.35rad`. For the current `turn_open` geometry, that target was too large and caused an avoidable oscillation. I parameterized the takeover probe and lowered the pivot target:

- log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn_open/05_25_22-15-46`
- controller: `pivot_then_heading`
- settings:
  - `pivot_yaw_target=0.9`
  - `max_steps=520`
- result:
  - `reach_reward_sum=0.0`
  - `min_distance=3.1585`
  - no termination within `520` steps

So the earlier turn-failure diagnosis was partly too pessimistic. Lowering the pivot target clearly improved the trajectory and removed the early death pattern.

However, extending the same tuned probe to a longer horizon still did not reach the goal:

- log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn_open/05_25_22-16-24`
- controller: `pivot_then_heading`
- settings:
  - `pivot_yaw_target=0.9`
  - `max_steps=1500`
- result:
  - `reach_reward_sum=0.0`
  - `min_distance=2.4030`
  - `done_step=669`
  - `done_reason=terminated`

Interpretation:

- the manual takeover script itself was indeed one source of error
- but fixing that script mistake does not close the simplified turn-and-reach task
- the current stack can get materially closer to the goal before failure, but still collides before collecting reach reward

### Lateral response exists but is weaker than commanded

Because the remaining blocker might require more than `vx+wz`, I ran a dedicated lateral low-level tracking probe:

- log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/low_level_tracking_probe/05_25_22-18-15`
- command: `[vx=0.0, vy=0.5, wz=0.0]`
- actual mean:
  - `vx=-0.0490`
  - `vy=0.2167`
  - `wz=0.0424`

This shows that lateral motion is not absent, but it is much weaker than commanded. A cautious vector-style takeover controller was then added and tested:

- log: `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/manual_reward_probe_turn_open/05_25_22-19-08`
- controller: `pivot_then_safe_vector`
- settings:
  - `pivot_yaw_target=0.9`
  - `max_steps=1500`
- result:
  - `reach_reward_sum=0.0`
  - `min_distance=2.7712`
  - `done_step=772`
  - `done_reason=terminated`

Updated conclusion after these follow-up probes:

- the blocker is not just "bad converted asset"
- it is not just "badly chosen hand-scripted pivot target"
- the remaining gap is consistent with limited low-level command-tracking fidelity during tight turning, especially near low front-clearance states

### Fixed `turn_open` room training shows transient reach, but not stable success

To stop guessing from hand-written takeover controllers alone, I added a deterministic fixed-room training entry path in `training/isaac_lab/train.py` using `--preset-room-scenario turn_open`. This keeps the normal PPO stack but removes random room generation, so the training question becomes much sharper: can the current Isaac Lab port learn the simplified open-corner task at all?

First short diagnostic run:

- command:
  - `env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py --headless --num-envs 64 --num-steps-per-env 16 --max-iterations 80 --save-interval 20 --preset-room-scenario turn_open --experiment-name Go2_pos_rough_isaaclab_turn_open_diag --run-name 80it_turn_open`
- log dir:
  - `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_diag/05_25_22-37-07_80it_turn_open`
- result:
  - reward improved strongly during the short run
  - but `Mean episode rew_reach_pos_target_tight` stayed at `0.0000`

Then a longer 400-iteration diagnostic:

- command:
  - `env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py --headless --num-envs 64 --num-steps-per-env 16 --max-iterations 400 --save-interval 100 --preset-room-scenario turn_open --experiment-name Go2_pos_rough_isaaclab_turn_open_diag --run-name 400it_turn_open`
- log dir:
  - `/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_diag/05_25_22-37-58_400it_turn_open`
- saved checkpoints:
  - `model_200.pt`
  - `model_300.pt`
  - `model_400.pt`

Important mid-run observations from the 400-iteration training log:

- `Iteration 220`:
  - `Mean reward = 857.99`
  - `Mean episode rew_reach_pos_target_tight = 1.2183`
- `Iteration 270`:
  - `Mean reward = 2039.34`
  - `Mean episode rew_reach_pos_target_tight = 0.4869`
- `Iteration 310`:
  - `Mean episode rew_reach_pos_target_tight = 2.4086`
- `Iteration 320`:
  - `Mean episode rew_reach_pos_target_tight = 7.9331`

But the end of the run did not keep that behavior:

- `Iteration 390`:
  - `Mean reward = 240.44`
  - `Mean episode rew_reach_pos_target_tight = 0.0000`
- training finished successfully, but the policy clearly did **not** converge to stable simplified-room success

Interpretation:

- the current port is no longer in the "completely broken, never reaches" regime
- it can transiently enter states that collect real reach reward in the simplified open-corner room
- but the learned behavior is still unstable and later training drifts back toward non-reaching dense-reward solutions

This is much stronger evidence than the earlier hand-coded probes:

- the task is not fundamentally impossible under the current port
- but current reward/termination balance and/or execution semantics still allow degenerate local optima

### Checkpoint evaluation path was added, but current GPU runtime on this machine failed before rollout

To separate "training curve looks promising" from "checkpoint can actually reproduce the fixed-room turn", I extended `training/isaac_lab/manual_reward_probe.py` with a `--controller-mode policy` path that:

- loads a saved checkpoint
- uses deterministic `act_inference`
- reuses the same fixed start/goal deterministic probe logic

The first checkpoint evaluation attempted was:

- command:
  - `env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/manual_reward_probe.py --headless --scenario turn_open --controller-mode policy --checkpoint /home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_diag/05_25_22-37-58_400it_turn_open/model_200.pt --max-steps 1500 --settle-steps 10`

This did **not** fail in policy loading or probe logic. Instead, Isaac Sim startup failed because the machine's GPU runtime had entered a bad state:

- `nvidia-smi` now reports:
  - `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.`
- `/dev/nvidia*` device nodes are currently absent
- but kernel modules are still loaded:
  - `nvidia`
  - `nvidia_modeset`
  - `nvidia_drm`
  - `nvidia_uvm`
- `/proc/driver/nvidia/version` still reports:
  - `NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 580.159.03`

This means the new blocker is **not** inside SEA-Nav code itself. The machine-level NVIDIA runtime is currently inconsistent: kernel modules exist, but user-space device nodes are gone, so fresh Isaac Sim GPU processes cannot acquire CUDA/Vulkan devices.

As of `2026-05-25 22:44:40 CST`, the correct conclusion is:

- fixed-room simplified training provided real progress evidence
- fixed-room checkpoint evaluation code is in place
- but new rollout evaluation is temporarily blocked by a machine-level NVIDIA runtime failure, not by the checkpoint-eval code path

### Original collision replay semantics were missing and have now been ported into the Isaac Lab env

While re-checking the original Isaac Gym code path against the current Isaac Lab port, I found a major remaining semantic mismatch that had not been ported at all:

- original `Go2PosRoughCfg.replay.enable_collision_replay = True`
- original env keeps a replay history of root state + joint state
- original env performs probabilistic early reset on new collisions
- original env can rewind to an earlier in-episode state instead of always doing a fresh spawn reset

The relevant original implementation lives in:

- `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py`
  - `_init_replay_buffers`
  - `_update_replay_buffer`
  - `_reset_collision_replay`
  - `reset_idx`
  - `check_termination`

Before this turn, the Isaac Lab port had none of that. It only had:

- collision penalty
- terminal contact reset
- normal fresh-spawn reset

That means previous Isaac Lab training runs were still missing one of the original training-stability mechanisms, so "runner semantics aligned" was still materially incomplete.

This turn I ported the missing replay path into:

- `training/isaac_lab/sea_nav_env.py`

Specifically, the Isaac Lab env now includes:

- replay history buffers for:
  - root state
  - joint position
  - joint velocity
- per-env replay/collision flags:
  - `collision_occurred`
  - `is_replay`
  - `last_collision_active`
- early collision reset logic in `_get_dones()`:
  - new penalized collision detection
  - probabilistic early reset using `collision_replay_early_reset_prob_range`
  - termination penalty applied to replay-triggered resets
- replay rewind logic in `_reset_idx()`:
  - choose replay vs normal reset
  - restore an older buffered state when replay is valid
  - fall back to normal reset if history is too short
- replay config defaults in `SeaNavEnvCfg`:
  - `replay_len = 100`
  - `enable_collision_replay = True`
  - `collision_replay_prob = 0.8`
  - `collision_replay_early_reset_prob_range = (0.1, 0.5)`
  - `collision_replay_undo_steps_range = (100, 150)`

Static verification completed:

- `python3 -m py_compile training/isaac_lab/sea_nav_env.py training/isaac_lab/manual_reward_probe.py training/isaac_lab/train.py`

What this means for the main objective:

- the current blocker is no longer just "evaluate checkpoints after reboot"
- once the NVIDIA runtime is healthy again, the next training runs will no longer be missing this obvious original-Gym replay mechanism
- that makes the next round of fixed-room and full-room training materially more faithful to the original method than the earlier Isaac Lab runs

### Original observation-noise semantics were also missing and have now been restored for training

During the same audit, I confirmed that the current Isaac Lab port had another important mismatch:

- original Isaac Gym training injects noise into the low-level controller observation buffer
- original Isaac Gym training also injects noise into the high-level navigation observation buffer
- current Isaac Lab port had neither

This matters because it changes the training distribution seen by both the low-level command-tracking stack and the high-level navigation policy. To reduce that mismatch, I restored the missing training-time observation noise in:

- `training/isaac_lab/sea_nav_env.py`

The Isaac Lab env now includes configurable noise terms matching the original rough magnitudes:

- low-level (`slr_obs_buf`)
  - angular velocity noise
  - projected gravity noise
  - dof position noise
  - dof velocity noise
- high-level (`prop_buf`)
  - projected gravity noise
  - base linear velocity noise
  - base angular velocity noise

To keep deterministic debugging usable, the simplified probe path explicitly disables this noise:

- `training/isaac_lab/manual_reward_probe.py`

So the new behavior is:

- training path: closer to original Isaac Gym semantics
- manual fixed-room probes / checkpoint probes: still deterministic and easier to interpret

### Training now saves best checkpoints instead of only periodic snapshots

Another concrete issue exposed by the fixed `turn_open` experiment was:

- transient real reach appeared around the middle of training
- but periodic checkpoints only kept `model_200.pt`, `model_300.pt`, `model_400.pt`
- the strongest transient behavior around `iter=320` could easily be lost if later training regressed

To prevent that, I updated:

- `training/rsl_rl/rsl_rl/runners/on_policy_runner.py`

The runner now additionally saves:

- `best_mean_reward.pt`
- `best_reach.pt`

where:

- `best_mean_reward.pt` tracks the best rolling episode reward
- `best_reach.pt` tracks the best observed `rew_reach_pos_target_tight`

This does not solve the runtime blocker, but it does remove an important evaluation failure mode:

- after the NVIDIA runtime is recovered, the next training run will not depend only on `save_interval`
- if the policy briefly learns a better simplified-room solution and later regresses, that best checkpoint will still be preserved for deterministic rollout evaluation

Static verification after these changes:

- `python3 -m py_compile training/isaac_lab/sea_nav_env.py training/rsl_rl/rsl_rl/runners/on_policy_runner.py training/isaac_lab/train.py training/isaac_lab/manual_reward_probe.py`

### Terrain curriculum and initial difficulty were also mismatched and have now been aligned more closely

Another concrete semantic gap appeared during the config audit:

- in the original Isaac Gym stack, `Go2PosRoughCfg.terrain` inherits `curriculum = True`
- the original terrain importer starts robots only in the first few terrain rows:
  - `max_init_terrain_level = 2`
- the original navigation env also keeps a separate `goal_levels` buffer and uses it in:
  - terrain progression logic
  - early collision replay probability

The Isaac Lab port previously differed in two ways:

- it effectively started from any terrain row
- it had no `goal_levels` curriculum state at all

This turn I aligned those pieces in `training/isaac_lab/sea_nav_env.py`:

- added `goal_levels` tracking
- added `_update_terrain_curriculum(...)` mirroring the original navigation env logic
- switched replay early-reset probability to depend on `goal_levels` instead of a terrain-level shortcut
- added episode logging for `goal_level`
- changed the initial terrain spawn bound in `make_sea_nav_env_cfg(...)` to:
  - `max_init_terrain_level = min(num_rows - 1, 2)`

This matters because the previous Isaac Lab runs were still biased away from the original training schedule:

- start-state difficulty distribution was too broad
- curriculum state was missing
- collision replay probability was tied to the wrong quantity

So after this turn, the Isaac Lab port is materially closer to the original training semantics than before, even though runtime validation is still waiting on the machine-level NVIDIA recovery.

### Host GPU runtime is healthy again; the earlier hard blocker was partly a sandbox artifact

The earlier conclusion that "the machine-level NVIDIA runtime is broken" turned out to be too broad.

What was verified later:

- root-level `nvidia-smi` works on the host again:
  - driver `580.159.03`
  - GPU `NVIDIA GeForce RTX 5060 Ti`
- `udevadm trigger --action=add` on the GPU PCI device restored the missing `/dev/nvidia*` and `/dev/dri/*` nodes
- a non-sandbox Isaac Lab GPU smoke run succeeded again on `cuda:0`

What actually caused the confusing symptoms:

- ordinary sandboxed commands still could not access the GPU correctly
- non-sandbox / escalated runs could

So the practical rule for the remaining SEA-Nav validation work is:

- Isaac Lab training / rollout verification that depends on the host GPU must be run outside the sandbox
- the codebase is no longer blocked by the earlier "no CUDA-capable device is detected" diagnosis

### Fixed `turn_open` training with repaired semantics now reaches real goal reward again

After restoring the missing semantics above, I reran a fixed-room training on:

- preset room: `turn_open`
- `32` envs
- `16` steps per env
- `400` iterations
- run directory:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_replaycheck/05_25_23-06-11_400it_replaycheck`

This run is materially healthier than the earlier unstable Isaac Lab attempts:

- `iter=20`
  - `value_loss=504.36`
  - `smooth=0.027`
  - `rew_collision=-19.18`
- training stayed numerically stable through `400` iterations
- real nonzero reach reward reappeared multiple times:
  - `iter=250`: `rew_reach_pos_target_tight=2.5095`
  - `iter=320`: `rew_reach_pos_target_tight=3.9989`
  - `iter=360`: `rew_reach_pos_target_tight=12.8544`
  - `iter=380`: `rew_reach_pos_target_tight=12.5938`

So the repaired Isaac Lab port is no longer in the earlier "can only learn dense shaping but never real reach" state.

### Deterministic fixed-room checkpoint rollout now has one real success case

I then evaluated several checkpoints from the same run with deterministic policy rollout in the same fixed `turn_open` room using:

- `training/isaac_lab/manual_reward_probe.py`
- `--controller-mode policy`
- `--scenario turn_open`
- `--max-steps 1500`
- `--actuator-mode ideal_pd`

Results:

- `model_320.pt`
  - log dir:
    - `logs/isaac_lab/manual_reward_probe_turn_open/05_25_23-09-34`
  - `reach_reward_sum=219.4657`
  - `min_distance=0.3483`
  - `first_reach_step=1411`
  - `done_step=null`
  - this is the first deterministic fixed-room policy rollout that actually entered the goal region and accumulated real reach reward
- `model_360.pt`
  - failed early
  - `done_step=50`
  - `min_distance=6.4443`
- `model_380.pt`
  - did not terminate, but also did not reach
  - `min_distance=1.8508`
  - `reach_reward_sum=0`
- `best_mean_reward.pt`
  - failed with contact termination
  - `done_step=174`
  - `min_distance=4.4911`
- `best_reach.pt`
  - failed with contact termination
  - `done_step=350`
  - `min_distance=3.5354`

What this means:

- the fixed-room chain is now genuinely closed at least once under deterministic policy rollout
- the policy selected by scalar training summaries is not necessarily the best deterministic fixed-room policy
- in this run, `model_320.pt` is a better fixed-room checkpoint than either `best_reach.pt` or `best_mean_reward.pt`

This is a critical milestone for the overall reproduction objective:

- earlier, even the simplified room could not be closed with a trained policy
- now, the repaired Isaac Lab stack can produce at least one checkpoint that really reaches the goal in the deterministic fixed `turn_open` probe

The next logical step from here is no longer "keep debugging whether the fixed room is fundamentally broken".
It is:

- move back toward broader reproduction training/evaluation on the repaired semantics
- keep deterministic fixed-room rollout as a hard regression gate

### Full repaired-semantics reproduction training has now been restarted on the host GPU

After the deterministic fixed-room gate passed with `model_320.pt`, I moved the main path back to full reproduction training.

Current long run:

- service:
  - `sea-nav-full-repaired.service`
- run directory:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_repro_repaired/05_25_23-13-50_2000it_full_repaired`
- command shape:
  - headless Isaac Lab training
  - `32` envs
  - `48` steps per env
  - `2000` iterations
  - `save_interval=100`

Reason for using a user systemd service instead of a shell background job:

- the earlier `nohup`-style launch did not stay alive reliably through Isaac Sim startup
- the transient user service survives the front-end shell lifecycle better and is easier to monitor

The service has already passed:

- simulation startup
- environment construction
- low-level joint mapping print
- `runner ready`

Early metrics from the first full-room iterations:

- `iter=20`
  - `mean_reward=-4652.41`
  - `rew_collision=-16.99`
  - `rew_reach_pos_target_tight=0.0`
  - `terrain_level=0.5033`
- `iter=30`
  - `mean_reward=-4590.91`
  - `rew_collision=-27.69`
  - `rew_reach_pos_target_tight=0.0`
- `iter=40`
  - `mean_reward=-3436.27`
  - `rew_collision=-77.86`
  - `rew_reach_pos_target_tight=0.0`
- `iter=50`
  - `mean_reward=-3922.18`
  - `rew_collision=-80.25`
  - `rew_reach_pos_target_tight=0.0`

Interpretation at this stage:

- this is still very early for the full curriculum run
- unlike the fixed-room replaycheck, the agent is back in the broader terrain / obstacle distribution
- the important fact right now is not reward quality yet, but that the repaired full training path is running end-to-end on this hardware again

So the main reproduction status is now:

- simplified fixed-room deterministic reach: achieved at least once
- repaired full reproduction training on host GPU: running

### The from-scratch full run did not preserve the fixed-room success gate and was not left as the main path

After the full repaired-semantics run started, I checked both its scalar metrics and deterministic fixed-room transfer behavior instead of letting it run blindly.

By `iter=220`, the full run still showed:

- `Episode/rew_reach_pos_target_tight = 0.0`
- best scalar `Train/mean_reward` only around `-2672.65`
- no sign of true goal-reaching in the broader curriculum

I then evaluated the current full-run checkpoints in the same deterministic `turn_open` gate:

- `best_mean_reward.pt`
  - log dir:
    - `logs/isaac_lab/manual_reward_probe_turn_open/05_25_23-16-13`
  - `reach_reward_sum=0`
  - `min_distance=6.3503`
  - `done_step=297`
- `model_200.pt`
  - log dir:
    - `logs/isaac_lab/manual_reward_probe_turn_open/05_25_23-16-58`
  - `reach_reward_sum=0`
  - `min_distance=6.3061`
  - `done_step=78`

So even though the broad full run was alive and numerically stable enough to continue, it had not yet retained the simplified-room capability that had already been demonstrated by the repaired fixed-room training.

This was enough evidence not to keep "from scratch full training" as the only main line.

### Resume support has now been added to the Isaac Lab training entrypoint

To make the next step explicit and reproducible, I added checkpoint resume support to:

- `training/isaac_lab/train.py`

New arguments:

- `--resume-from`
- `--resume-load-optimizer`

This allows two useful modes:

- fine-tune from a previous checkpoint while reusing only model weights
- resume more literally with optimizer state when desired

I also fixed an important bookkeeping bug in:

- `training/rsl_rl/rsl_rl/runners/on_policy_runner.py`

Before this fix:

- intermediate checkpoints such as `model_320.pt` were being saved with `iter=0`
- that made resume metadata misleading

Now:

- periodic and best-checkpoint saves record the actual loop iteration they came from

Static verification after these edits:

- `python3 -m py_compile training/isaac_lab/train.py training/rsl_rl/rsl_rl/runners/on_policy_runner.py`

### The main training path has now been switched to full finetune from the successful fixed-room checkpoint

After adding resume support, I verified the new path with a minimal host-GPU smoke load:

- `training/isaac_lab/train.py --headless --num-envs 1 --max-iterations 0 --num-steps-per-env 8 --resume-from .../model_320.pt`

That completed successfully and printed:

- `resumed runner checkpoint=...model_320.pt`
- `load_optimizer=False`

With that path verified, I stopped the less-promising from-scratch service and launched a new main run:

- service:
  - `sea-nav-full-finetune-turnopen320.service`
- run directory:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen320/05_25_23-20-06_2000it_finetune_noopt`
- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_replaycheck/05_25_23-06-11_400it_replaycheck/model_320.pt`
- resume mode:
  - load model weights
  - do not load optimizer state

The new finetune service has already passed:

- simulation startup
- environment construction
- checkpoint load
- `runner ready`

and has started training iterations on the full distribution.

Earliest observed finetune metrics:

- `iter=20`
  - `mean_reward=-3895.40`
  - `rew_collision=-63.42`
  - `rew_reach_pos_target_tight=0.0`
  - `terrain_level=0.3125`

This is still too early to claim improvement over the paper target.
But it is a more evidence-backed main path than continuing to trust the earlier from-scratch full run without the fixed-room success gate.

### A major remaining semantic gap was found in full training: terrain generation curriculum had been disabled

While comparing the original Isaac Gym config against the Isaac Lab port, I found a more serious mismatch than expected:

- original legged-gym base terrain config uses `curriculum = True`
- current Isaac Lab terrain generator had been instantiated with:
  - `TerrainGeneratorCfg(... curriculum=False, ...)`

This matters a lot in Isaac Lab's implementation:

- with `curriculum=True`, terrain difficulty increases by row
- with `curriculum=False`, terrain difficulty is sampled randomly across rows

I confirmed the installed Isaac Lab generator behavior directly from:

- `isaaclab/source/isaaclab/isaaclab/terrains/terrain_generator.py`

So the earlier broad full runs were not just "still training"; they were being trained on the wrong terrain-difficulty layout.

I corrected this in:

- `training/isaac_lab/sea_nav_env.py`

by changing the terrain generator config to:

- `curriculum=True`

Static verification after this fix:

- `python3 -m py_compile training/isaac_lab/sea_nav_env.py training/isaac_lab/train.py training/rsl_rl/rsl_rl/runners/on_policy_runner.py`

### The earlier broad full runs are now superseded by a corrected-curriculum finetune run

Because of the terrain-curriculum bug, I did not keep the previous broad full runs as the main path:

- `sea-nav-full-repaired.service`
- `sea-nav-full-finetune-turnopen320.service`

Both were superseded after the curriculum fix.

I then launched the new main run:

- service:
  - `sea-nav-full-finetune-turnopen320-curriculum.service`
- run directory:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen320_curriculumfix/05_25_23-25-53_2000it_finetune_noopt`
- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_replaycheck/05_25_23-06-11_400it_replaycheck/model_320.pt`

This new run already shows a very different early profile from the broken-curriculum full finetune:

- `iter=20`
  - `mean_reward=-842.95`
  - `rew_collision=-11.71`
  - `terrain_level=1.5436`
- `iter=30`
  - `mean_reward=-570.40`
  - `terrain_level=1.3750`
- `iter=40`
  - `mean_reward=-746.89`
  - `rew_collision=-6.18`
- `iter=70`
  - `mean_reward=-680.43`
  - `rew_collision=-7.38`
  - `rew_velo_dir=4.1102`
- `iter=80`
  - `rew_reach_pos_target_tight=3.6054`
  - `mean_reward=-853.40`
  - `rew_collision=-9.35`

Compared with the previous broken-curriculum full finetune:

- the reward scale is much less catastrophic
- collision is much lower
- `terrain_level` now behaves consistently with curriculum-based row sampling
- most importantly, true nonzero reach reward has already reappeared by `iter=80`

This is the strongest broad-training signal seen so far on the corrected Isaac Lab stack.

### Deterministic fixed-room gate is not fully preserved yet, but the corrected full run is closer than before

I also evaluated the current corrected-curriculum full-run `best_reach.pt` in the deterministic `turn_open` fixed-room probe:

- log dir:
  - `logs/isaac_lab/manual_reward_probe_turn_open/05_25_23-27-11`
- result:
  - `reach_reward_sum=0`
  - `min_distance=3.3183`
  - no termination

This means:

- the corrected full run is not yet strong enough to re-close the deterministic fixed-room gate
- but it is much closer than the previous broad runs that stayed around `6m` away or terminated quickly

So the current best interpretation is:

- the terrain-curriculum fix was necessary and materially improved the broad training path
- the new main finetune run is the correct line to keep pushing
- the reproduction objective is still not complete, because the broad run has not yet demonstrated stable deterministic goal-reaching near the paper target

### The corrected-curriculum full finetune is still only showing transient reach, not stable success

I continued monitoring the same corrected-curriculum full finetune run:

- service:
  - `sea-nav-full-finetune-turnopen320-curriculum.service`
- run directory:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen320_curriculumfix/05_25_23-25-53_2000it_finetune_noopt`

By `iter=600`, the run had still not stabilized into consistent reaching, but it also had not collapsed into a fully dead line. The strongest evidence is that `rew_reach_pos_target_tight` reappeared more than once:

- `iter=80`
  - `rew_reach_pos_target_tight=3.6054`
- `iter=110`
  - `rew_reach_pos_target_tight=7.6223`
- `iter=480`
  - `rew_reach_pos_target_tight=6.9602`
- `iter=540`
  - `rew_reach_pos_target_tight=1.9293`

At the same time, the run still spends long stretches at zero reach reward:

- `iter=120` through `iter=470`
  - mostly `rew_reach_pos_target_tight=0`
- `iter=490` through `iter=600`
  - again mostly `rew_reach_pos_target_tight=0`

Representative full-run metrics later in training:

- `iter=400`
  - `mean_reward=-1063.59`
  - `terrain_level=0.0312`
- `iter=480`
  - `mean_reward=-726.13`
  - `rew_reach_pos_target_tight=6.9602`
- `iter=540`
  - `mean_reward=-611.33`
  - `rew_reach_pos_target_tight=1.9293`
- `iter=600`
  - `mean_reward=-692.08`
  - `terrain_level=0.0312`

This means the corrected-curriculum full finetune is better described as:

- capable of occasional true goal-reaching under the broad training distribution
- not yet reliably preserving that behavior
- therefore still short of a completed reproduction

### A recurring machine-level GPU runtime issue blocks new Isaac Sim eval processes, even while the existing training process keeps running

While the main training service continued past `iter=600`, a fresh deterministic fixed-room probe process failed to initialize CUDA:

- `nvidia-smi`
  - failed with `couldn't communicate with the NVIDIA driver`
- `/dev/nvidia*`
  - absent at probe time
- new Isaac Sim process
  - reported `no CUDA-capable device is detected`
  - `NVML_ERROR_DRIVER_NOT_LOADED`

Important nuance:

- the already-running training process kept progressing
- but fresh Isaac Sim eval launches were blocked at the machine/runtime layer

Because of that, I could confirm that newer full-run checkpoints existed:

- `model_300.pt`
- `model_400.pt`
- `model_500.pt`

but I could not yet complete new deterministic fixed-room rollout checks for them in the same host state. So at this point the broad full run remains the active main line, but the final fixed-room validation for its newer checkpoints is temporarily blocked by host GPU runtime instability rather than by a newly identified SEA-Nav code bug.

I also inspected the checkpoint metadata directly:

- `best_reach.pt`
  - saved from `iter=67`
- `best_mean_reward.pt`
  - saved from `iter=205`

This matters because later printed nonzero reach events such as `iter=110`, `iter=480`, and `iter=540` did **not** overwrite `best_reach.pt`. So the current interpretation is not "best checkpoint saving is broken"; instead, an earlier unprinted iteration around `67` likely produced an even larger broad-distribution reach metric than those later logged reach spikes.

### The same corrected-curriculum run kept improving in broad-training reward, but still only reached intermittently

I continued monitoring the same run past `iter=900`.

Important updates:

- `best_mean_reward.pt`
  - was refreshed again
  - checkpoint metadata now shows `iter=859`
- periodic checkpoints now include:
  - `model_900.pt`

Late-training signals are stronger than earlier in the run, but still not stable enough to call the reproduction complete:

- `iter=860`
  - `mean_reward=-285.99`
  - `mean_episode_length=506.98`
  - `rew_collision=-14.9611`
  - `rew_close_obst_vel=7.4472`
  - `rew_velo_dir=6.7539`
  - `rew_reach_pos_target_tight=1.6347`
- `iter=870`
  - `mean_reward=-336.49`
  - `rew_reach_pos_target_tight=0`
- `iter=880`
  - `mean_reward=-595.56`
  - `rew_reach_pos_target_tight=0`
- `iter=890`
  - `mean_reward=-751.58`
  - `rew_reach_pos_target_tight=0`
- `iter=900`
  - periodic checkpoint saved as `model_900.pt`

So the current broad-training interpretation is:

- this line is still alive and improving in average reward
- it can still produce true nonzero reach reward late in training
- but the reach signal remains sparse and intermittent rather than stable

This is progress toward the paper-level target, but it is still not enough evidence to claim the target has been reproduced.

### The GPU runtime issue was recoverable by recreating missing device nodes

Later in the same session I rechecked the host runtime and found:

- NVIDIA kernel modules were still loaded
- the GPU still existed in `/proc/driver/nvidia/gpus/0000:01:00.0/information`
- major numbers for `nvidia`, `nvidiactl`, and `nvidia-uvm` were still present in `/proc/devices`
- but `/dev/nvidia*` device nodes had disappeared

The practical recovery step was:

- `nvidia-modprobe -u -c=0`

After that:

- `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, and related nodes reappeared
- `nvidia-smi` worked again
- fresh Isaac Sim evaluation processes could launch normally again

So this was a machine/runtime issue, but not a permanent one; it did not require changing SEA-Nav code.

### Fresh deterministic fixed-room evaluations on the late full-run checkpoints still failed

Once the GPU runtime was healthy again, I re-ran the `turn_open` deterministic fixed-room probe on the strongest late full-run candidates.

1. `best_mean_reward.pt` from the corrected-curriculum full run:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen320_curriculumfix/05_25_23-25-53_2000it_finetune_noopt/best_mean_reward.pt`
- probe log:
  - `logs/isaac_lab/manual_reward_probe_turn_open/05_25_23-40-29`
- result:
  - `reach_reward_sum=0.0`
  - `min_distance=3.7257`
  - `done_step=467`
  - `done_reason=terminated`

2. `model_900.pt` from the same run:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen320_curriculumfix/05_25_23-25-53_2000it_finetune_noopt/model_900.pt`
- probe log:
  - `logs/isaac_lab/manual_reward_probe_turn_open/05_25_23-41-55`
- result:
  - `reach_reward_sum=0.0`
  - `min_distance=4.0730`
  - `done_step=497`
  - `done_reason=terminated`

So even after the broad run improved in average reward, it still did not preserve deterministic `turn_open` success.

### I added a broader hard-room rollout evaluator to move validation closer to the project-page behavior

The single fixed-room gate was useful, but it was still narrower than the public project-page one-take behavior. To make the validation more aligned with that, I extended:

- `training/isaac_lab/manual_reward_probe.py`

with a new mode:

- `--scenario hard_room_eval`

This mode:

- runs a policy checkpoint in a single randomly generated hard-room layout
- samples new random start/goal pairs on each reset
- aggregates multiple episodes
- records success, collision, stand-still, and other terminal reasons

I also added minimal train-time terrain overrides to:

- `training/isaac_lab/train.py`

via:

- `--terrain-rows`
- `--terrain-cols`
- `--obstacle-level`

so I could launch one-room targeted follow-up experiments without patching constants again.

Static verification after these changes:

- `python3 -m py_compile training/isaac_lab/train.py training/isaac_lab/manual_reward_probe.py`

### The finished corrected-curriculum full run is still far from the project-page target

The corrected-curriculum full finetune eventually completed:

- final run directory:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen320_curriculumfix/05_25_23-25-53_2000it_finetune_noopt`
- final checkpoint:
  - `model_2000.pt`
- `best_mean_reward.pt` metadata:
  - `iter=859`
- final checkpoint metadata:
  - `iter=2000`

Late in training, the printed summaries had already returned to zero reach reward:

- `iter=1930`
  - `rew_reach_pos_target_tight=0.0000`
- `iter=1990`
  - `rew_reach_pos_target_tight=0.0000`

I then ran the broader `hard_room_eval` mode with `10` episodes, which is a much closer validation shape to the project-page one-take reporting style.

1. `best_mean_reward.pt` hard-room eval:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_25_23-46-57`
- result:
  - `success_count=0/10`
  - `collision_failures=7`
  - `collision_free_success_count=0/10`
  - `mean_reach_reward_sum=0.0`
  - `mean_min_distance=5.3837`

2. `model_2000.pt` hard-room eval:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_25_23-50-46`
- result:
  - `success_count=0/10`
  - `collision_failures=6`
  - `stand_failures=3`
  - `collision_free_success_count=0/10`
  - `mean_reach_reward_sum=0.0`
  - `mean_min_distance=5.3800`

This is the strongest current end-to-end evidence against claiming reproduction success:

- the full repaired Isaac Lab run does train and finish
- it does sometimes collect nonzero reach reward during training
- but when evaluated in a broader one-room hard-room rollout setting, it is still `0/10`, not "near the project-page result"

### Episode-level hard-room failures show the policy is not even entering the final success zone

From the episode-level traces:

- `best_mean_reward.pt`
  - closest episode only reached `min_distance=3.5057`
- `model_2000.pt`
  - closest episode only reached `min_distance=2.8994`

Neither checkpoint ever triggered a single success episode in the `10`-episode hard-room evaluation.

Failure modes were mainly:

- contact/collision failures
- generic terminal failures that still carried very large negative collision sums
- stand-still failures in the final `model_2000.pt`

So the remaining gap is not a subtle near-threshold problem. The broad policy is still far from consistently entering the `distance < 0.5m` success region.

### A one-room targeted follow-up experiment did not justify continuing past an early gate

Since the full repaired run was `0/10` in hard-room evaluation, I launched a more targeted follow-up:

- experiment:
  - `Go2_pos_rough_isaaclab_onehardroom_from_fullbest`
- run:
  - `05_25_23-53-34_400it_targeted`
- start checkpoint:
  - the full-run `best_mean_reward.pt`
- train setup:
  - `--terrain-rows 1 --terrain-cols 1 --obstacle-level 9`
  - same random start/goal mechanism, but on a single hard-room layout

The intention was to see whether the policy could at least recover nonzero reach under a narrower but still nontrivial one-room distribution before going back to the broader full setting.

What actually happened:

- by `iter=870` through `iter=1090`
  - `rew_reach_pos_target_tight` stayed at `0.0000`
- reward remained noisy and unstable
- checkpoints reached:
  - `model_1000.pt`
  - `model_1100.pt`

Because this branch still had zero reach after more than `200` resumed iterations, I stopped it early rather than wasting more GPU time.

Current interpretation:

- the one-room targeted fine-tune did not immediately recover success either
- so the next step should not be "just let another training branch cook longer"
- the remaining blocker is still at the policy/execution/task-behavior level, not basic runtime wiring

### I found and fixed an evaluation-path semantic mismatch around collision replay reset

After the broader hard-room evaluation returned `0/10`, I checked whether the evaluation semantics themselves were still harsher than the original Isaac Gym `play.py` path.

That inspection found a real mismatch:

- in the Isaac Gym `play.py` path, testing explicitly sets:
  - `env_cfg.asset.terminate_after_contacts_on = []`
  - `env_cfg.replay.enable_collision_replay = False`
  - `env_cfg.env.stay_time = 500`
- but in the Isaac Lab port, `_get_dones()` was still allowing `early_replay_reset` to terminate episodes even when:
  - `enable_collision_replay = False`

That meant the earlier "no contact termination" eval was still not faithfully matching the intended play/eval semantics.

I fixed this in:

- `training/isaac_lab/sea_nav_env.py`

by gating `early_replay_reset` on `self.cfg.enable_collision_replay`.

I also extended:

- `training/isaac_lab/manual_reward_probe.py`

to support:

- `--disable-contact-termination`
- `--stay-steps`

so the eval path can explicitly match the original `play.py` behavior more closely.

Static verification after the fix:

- `python3 -m py_compile training/isaac_lab/sea_nav_env.py training/isaac_lab/manual_reward_probe.py`

### Even under play-style semantics, the current best full-run checkpoint still does not succeed

After the replay-reset mismatch was fixed, I reran the broader hard-room evaluation on:

- `best_mean_reward.pt`

with a closer match to the Isaac Gym play/test path:

- contact termination disabled
- collision replay disabled
- `stay_steps = 500`
- `10` episodes

Command shape:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u \
  training/isaac_lab/manual_reward_probe.py \
  --headless \
  --scenario hard_room_eval \
  --controller-mode policy \
  --episodes 10 \
  --checkpoint .../best_mean_reward.pt \
  --max-steps 2000 \
  --settle-steps 0 \
  --stay-steps 500 \
  --disable-contact-termination \
  --actuator-mode ideal_pd \
  --robot-asset-source converted_urdf \
  --sim-device cuda:0
```

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_00-04-19`

Observed result:

- `success_count = 0/10`
- `collision_failures = 0`
- `stand_failures = 9`
- `max_step_failures = 1`
- `mean_reach_reward_sum = 0.0`
- `mean_min_distance = 5.1093`

This is a much stronger diagnostic result than the earlier contact-terminated evaluation:

- the policy still does not succeed even when contact termination is removed
- the failure mode shifts from "contact/terminated" to "stand still far from the goal"
- therefore, the remaining blocker is not mainly evaluation harshness

Episode-level evidence from the same log:

- all `10` episodes failed without a single reach event
- the closest episode only got to `min_distance = 2.9313`
- most episodes ended via `stand`
- one episode ran to `max_steps` without succeeding

So at this point the broad policy failure is better described as:

- it can move and accumulate dense reward
- it often collides repeatedly or gets trapped in a non-progressing local behavior
- and even after removing contact termination, it still cannot enter the final success zone reliably

### Native Go2 asset confirms the converted URDF path is not the only blocker

I then switched the same play-style hard-room evaluation to:

- `--robot-asset-source native_go2`

using the same corrected-curriculum full-run checkpoint:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen320_curriculumfix/05_25_23-25-53_2000it_finetune_noopt/best_mean_reward.pt`

Command shape:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u \
  training/isaac_lab/manual_reward_probe.py \
  --scenario hard_room_eval \
  --controller-mode policy \
  --episodes 10 \
  --checkpoint .../best_mean_reward.pt \
  --max-steps 2000 \
  --stay-steps 500 \
  --disable-contact-termination \
  --robot-asset-source native_go2 \
  --actuator-mode ideal_pd
```

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_00-09-10`

Observed result:

- `success_count = 0/10`
- `stand_failures = 5`
- `max_step_failures = 5`
- `mean_reach_reward_sum = 112.5019`
- `mean_min_distance = 3.2523`

This was materially better than the equivalent converted-URDF play-style eval:

- the policy still failed `0/10`
- but it no longer failed via contact termination
- and it did manage to enter the goal region in `2` of the `10` episodes

Episode-level evidence:

- episode `0`
  - `reach_reward_sum = 282.6143`
  - `min_distance = 0.2490`
  - `first_reach_step = 1233`
  - final `done_reason = stand`
- episode `5`
  - `reach_reward_sum = 842.4052`
  - `min_distance = 0.1298`
  - `first_reach_step = 552`
  - final `done_reason = stand`

Interpretation:

- `native_go2` clearly improves over the converted asset path
- but the main remaining failure is now sharper: the policy can sometimes enter the goal region and still fails to hold long enough to finish with `goal_hold`

### A native one-hard-room targeted fine-tune still did not close the goal-hold gap

Since the native-asset baseline showed real goal-region entry, I launched a narrower follow-up from the same full-run best checkpoint:

- experiment:
  - `Go2_pos_rough_isaaclab_onehardroom_native_from_fullbest`
- run:
  - `05_26_00-13-20_300it_targeted`
- start checkpoint:
  - the full-run `best_mean_reward.pt`
- setup:
  - `--terrain-rows 1 --terrain-cols 1 --obstacle-level 9`
  - `--robot-asset-source native_go2`

Artifacts produced by that run:

- `best_mean_reward.pt` with checkpoint metadata `iter = 869`
- `best_reach.pt` with checkpoint metadata `iter = 949`
- `model_1000.pt`
- `model_1100.pt`

I then evaluated the two most relevant checkpoints under the same play-style hard-room protocol.

#### `best_reach.pt` from the native targeted run

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_00-18-09`

Observed result:

- `success_count = 0/10`
- `stand_failures = 7`
- `max_step_failures = 3`
- `mean_reach_reward_sum = 22.6608`
- `mean_min_distance = 4.0489`

Episode-level evidence:

- only episode `3` entered the goal region
  - `reach_reward_sum = 226.6076`
  - `min_distance = 0.3476`
  - `first_reach_step = 1342`
  - final `done_reason = max_steps`

This means the targeted native branch still did not produce a single successful `goal_hold`, even at its own best-reach checkpoint.

#### Latest `model_1100.pt` from the native targeted run

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_00-25-40`

Observed result:

- `success_count = 0/10`
- `stand_failures = 9`
- `fall_failures = 1`
- `mean_reach_reward_sum = 0.0`
- `mean_min_distance = 4.7339`

Episode-level evidence:

- no episode entered the goal region at all
- closest episode only reached `min_distance = 2.2891`

Interpretation:

- the native one-hard-room targeted fine-tune did not solve the hold problem
- later in that branch, behavior actually regressed relative to:
  - the native full-run baseline `best_mean_reward.pt`
  - and the branch's own earlier `best_reach.pt`

So the current best understanding is:

- `native_go2` does reduce one part of the migration gap
- but even after narrowing training to a single hard-room distribution, the policy is still not reliably converting goal-region entry into `goal_hold`
- therefore the remaining blocker is not just the converted asset path; it is still a behavior-level failure around final approach, stabilization, or hold

### Native fixed-room `turn_open` training closes the deterministic gate again

Since the native hard-room targeted branch was still failing under play-style eval, I went back to the simpler fixed-room gate and reran the successful `turn_open` recipe on:

- `native_go2`
- `ideal_pd`
- the repaired Isaac Lab semantics already restored earlier

Training command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u \
  training/isaac_lab/train.py \
  --headless \
  --num-envs 32 \
  --num-steps-per-env 16 \
  --max-iterations 400 \
  --save-interval 100 \
  --preset-room-scenario turn_open \
  --robot-asset-source native_go2 \
  --actuator-mode ideal_pd \
  --experiment-name Go2_pos_rough_isaaclab_turn_open_native_replaycheck \
  --run-name 400it_native_turn_open
```

Run directory:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_native_replaycheck/05_26_00-37-23_400it_native_turn_open`

Training finished normally and produced:

- `best_mean_reward.pt`
- `best_reach.pt`
- `model_200.pt`
- `model_300.pt`
- `model_400.pt`

The fixed-room training signal was materially stronger than the earlier broad/full runs:

- `iter 80`
  - `Mean reward = 350.72`
  - `rew_reach_pos_target_tight = 9.7161`
- `iter 100`
  - `Mean reward = 634.42`
  - `rew_reach_pos_target_tight = 8.1627`
- `iter 280`
  - `rew_reach_pos_target_tight = 14.2948`
- `iter 350`
  - `rew_reach_pos_target_tight = 16.9782`
- `iter 370`
  - `rew_reach_pos_target_tight = 20.6158`
- `iter 390`
  - `rew_reach_pos_target_tight = 10.1109`

This confirmed that the repaired Isaac Lab path on `native_go2` is still capable of learning the simplified `turn_open` room strongly enough to accumulate repeated real reach reward.

### Deterministic fixed-room checkpoint evals show `model_400.pt` is the real native winner

I then evaluated the native fixed-room checkpoints with the deterministic `turn_open` policy rollout gate:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u \
  training/isaac_lab/manual_reward_probe.py \
  --headless \
  --scenario turn_open \
  --controller-mode policy \
  --checkpoint ... \
  --max-steps 1500 \
  --settle-steps 10 \
  --robot-asset-source native_go2 \
  --actuator-mode ideal_pd
```

#### `best_reach.pt`

Result log:

- `logs/isaac_lab/manual_reward_probe_turn_open/05_26_00-40-12`

Observed result:

- `reach_reward_sum = 0.0`
- `min_distance = 6.4437`
- `done_step = 134`
- `done_reason = terminated`
- `done_flags.contact = true`

So the scalar `best_reach.pt` snapshot was not the deterministic winner.

#### `model_300.pt`

Result log:

- `logs/isaac_lab/manual_reward_probe_turn_open/05_26_00-40-44`

Observed result:

- `reach_reward_sum = 0.0`
- `min_distance = 3.0588`
- `done_step = 1438`
- `done_reason = terminated`
- `done_flags.contact = true`

This was better than `best_reach.pt`, but still not enough to close the gate.

#### `model_400.pt`

Result log:

- `logs/isaac_lab/manual_reward_probe_turn_open/05_26_00-41-58`

Observed result:

- `total_reward_sum = 4434.2946`
- `reach_reward_sum = 460.4850`
- `min_distance = 0.1266`
- `first_reach_step = 1144`
- `done_step = null`
- `done_reason = null`

Interpretation:

- the native fixed-room chain does close the deterministic `turn_open` gate again
- `model_400.pt` is materially stronger than the earlier converted-asset fixed-room winner
- the repaired Isaac Lab path is not fundamentally broken on the simplified task

This is currently the strongest positive reproduction signal in the whole redeploy effort.

### New fixed-start hard-room replay support was added for behavior debugging

To diagnose why broad/hard-room behavior still fails after the simplified-room success, I also extended:

- `training/isaac_lab/manual_reward_probe.py`

with:

- `--fixed-start-cell`
- `--fixed-goal-cell`
- `--fixed-start-yaw`
- `--trace-steps`

This now allows replaying a single hard-room episode from a fixed layout and fixed yaw while logging per-step policy behavior, instead of only doing random `10`-episode aggregate eval.

Static verification passed:

```bash
python3 -m py_compile training/isaac_lab/manual_reward_probe.py
```

### A short full-distribution transfer diagnostic from the native fixed-room winner has started

Since `model_400.pt` from the native fixed-room run is the new strongest simplified-room checkpoint, I launched a short broad/full transfer diagnostic from it:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u \
  training/isaac_lab/train.py \
  --headless \
  --num-envs 32 \
  --num-steps-per-env 48 \
  --max-iterations 400 \
  --save-interval 100 \
  --robot-asset-source native_go2 \
  --actuator-mode ideal_pd \
  --resume-from /home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_native_replaycheck/05_26_00-37-23_400it_native_turn_open/model_400.pt \
  --experiment-name Go2_pos_rough_isaaclab_full_from_turnopen_native400 \
  --run-name 400it_transfer_diag
```

Current run directory:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400/05_26_00-43-34_400it_transfer_diag`

Early signal so far:

- `iter 480`
  - `Mean reward = 70.74`
  - `rew_reach_pos_target_tight = 1.1670`
- `iter 500`
  - `Mean reward = -95.55`
  - `rew_reach_pos_target_tight = 0.0000`
- `iter 510`
  - `Mean reward = -250.98`
  - `rew_reach_pos_target_tight = 0.0000`

So this transfer line is not a dead start. It has already produced one early non-zero reach event under the broad/full distribution, but the signal is still sparse and not yet strong enough to claim successful transfer.

### The native fixed-room to full-distribution transfer diagnostic finished with repeated non-zero reach signals

That short transfer run then finished normally:

- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400/05_26_00-43-34_400it_transfer_diag`
- final log line:
  - `training finished log_dir=/home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400/05_26_00-43-34_400it_transfer_diag`

Artifacts produced by the run:

- `best_mean_reward.pt` with checkpoint metadata `iter = 597`
- `best_reach.pt` with checkpoint metadata `iter = 584`
- `model_600.pt`
- `model_700.pt`
- `model_800.pt`

Important broad/full training signals from this run:

- `iter 480`
  - `Mean reward = 70.74`
  - `rew_reach_pos_target_tight = 1.1670`
- `iter 610`
  - `Mean reward = 148.26`
  - `rew_reach_pos_target_tight = 4.6045`
- `iter 650`
  - `Mean reward = 91.67`
  - `rew_reach_pos_target_tight = 1.7850`
- `iter 710`
  - `Mean reward = -103.30`
  - `rew_reach_pos_target_tight = 14.4099`
- `iter 720`
  - `Mean reward = -37.97`
  - `rew_reach_pos_target_tight = 4.6487`
- `iter 740`
  - `Mean reward = 339.16`
  - `rew_reach_pos_target_tight = 3.8726`

Interpretation:

- this line is materially more encouraging than the earlier broad/full runs that stayed near all-zero reach
- the repaired native fixed-room winner is transferring some genuine reach behavior into the broader training distribution
- but the signal is still intermittent rather than stable, so this by itself is not enough to declare successful reproduction

### Fresh host eval works, and the transfer run really is closer than the older full-run baselines

The temporary blocker right after the transfer run finished turned out not to be a broken host GPU runtime. The real issue was:

- inside the Codex sandbox, `nvidia-smi` and `/dev/nvidia*` were not visible
- outside the sandbox, the host NVIDIA runtime was still healthy

Host verification outside the sandbox:

```bash
nvidia-smi
```

returned:

- `Driver Version: 580.159.03`
- `CUDA Version: 13.0`
- normal visibility of the `NVIDIA GeForce RTX 5060 Ti`

So the correct conclusion is:

- fresh Isaac Sim evals for this repo must currently be run outside the sandbox
- the host GPU itself was still fine

I then reran the transfer checkpoints under the same play-style hard-room eval protocol on the host GPU:

- `native_go2`
- `ideal_pd`
- `--disable-contact-termination`
- `--stay-steps 500`
- `10` episodes

#### `best_reach.pt`

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_00-55-39`

Observed result:

- `success_count = 0/10`
- `max_step_failures = 10`
- `mean_reach_reward_sum = 203.7907`
- `mean_min_distance = 1.4376`

Episode-level evidence:

- `4/10` episodes entered the goal region and accumulated real reach reward
- closest four:
  - episode `0`: `min_distance = 0.0429`, `reach_reward_sum = 687.29`, `done_reason = max_steps`
  - episode `7`: `min_distance = 0.0613`, `reach_reward_sum = 397.50`, `done_reason = max_steps`
  - episode `6`: `min_distance = 0.1532`, `reach_reward_sum = 329.40`, `done_reason = max_steps`
  - episode `2`: `min_distance = 0.4110`, `reach_reward_sum = 623.72`, `done_reason = max_steps`

Interpretation:

- this checkpoint is much stronger than the earlier full-run baselines
- it is no longer mainly failing by collision or early stand
- the dominant failure is now: enter the goal region, but do not satisfy `goal_hold` before `max_steps`

#### `best_mean_reward.pt`

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_00-59-19`

Observed result:

- `success_count = 0/10`
- `stand_failures = 1`
- `max_step_failures = 9`
- `mean_reach_reward_sum = 237.5128`
- `mean_min_distance = 1.9528`

Episode-level evidence:

- `3/10` episodes entered the goal region
- closest three:
  - episode `7`: `min_distance = 0.0860`, `reach_reward_sum = 1219.02`, `done_reason = max_steps`
  - episode `9`: `min_distance = 0.1317`, `reach_reward_sum = 410.75`, `done_reason = max_steps`
  - episode `5`: `min_distance = 0.2289`, `reach_reward_sum = 745.35`, `done_reason = max_steps`

Interpretation:

- this checkpoint also clearly improves over the older baselines
- but compared with `best_reach.pt`, it is less consistent in getting close
- even when it enters the goal region deeply, it still fails to finish with `goal_hold`

#### `model_800.pt`

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_01-02-45`

Observed result:

- `success_count = 0/10`
- `stand_failures = 5`
- `fall_failures = 2`
- `max_step_failures = 3`
- `mean_reach_reward_sum = 13.8594`
- `mean_min_distance = 3.4015`

Episode-level evidence:

- only `1/10` episode entered the goal region
  - episode `7`: `min_distance = 0.3912`, `reach_reward_sum = 138.59`, `done_reason = stand`

Interpretation:

- the last checkpoint has already regressed relative to both `best_reach.pt` and `best_mean_reward.pt`
- so the best policy from this short transfer run is not the tail checkpoint; it is an earlier checkpoint

### Updated conclusion from the transfer diagnostic

This short transfer line did not complete the reproduction, but it changed the diagnosis materially:

- the simplified native fixed-room winner does transfer some real navigation skill into broad/full training
- the best transfer checkpoint is substantially better than the earlier broad/full baselines
- the remaining gap is now much narrower and more concrete:
  - not "cannot get near the goal at all"
  - but "can enter the goal region in several episodes and still fails to hold long enough to terminate successfully"

### A continuation run from the best broad/full transfer checkpoint is now active, with optimizer state restored

One practical issue uncovered by the short transfer diagnostic is that its original resume path did **not** load optimizer state:

- `training/isaac_lab/train.py` defaults `--resume-load-optimizer` to `False`

So the first `model_400.pt -> broad/full` transfer line improved, but then regressed by `model_800.pt`.

To test whether that regression is partly caused by resuming without optimizer state, I launched a new continuation from the strongest broad/full checkpoint:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400/05_26_00-43-34_400it_transfer_diag/best_reach.pt`
- new run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_opt/05_26_01-08-49_400it_resume_bestreach_opt`

Command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u \
  training/isaac_lab/train.py \
  --headless \
  --num-envs 32 \
  --num-steps-per-env 48 \
  --max-iterations 400 \
  --save-interval 100 \
  --robot-asset-source native_go2 \
  --actuator-mode ideal_pd \
  --resume-from /home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400/05_26_00-43-34_400it_transfer_diag/best_reach.pt \
  --resume-load-optimizer \
  --experiment-name Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_opt \
  --run-name 400it_resume_bestreach_opt
```

Verified startup:

- `resumed runner checkpoint=.../best_reach.pt load_optimizer=True current_iteration=584`

Early signal so far:

- `iter 670`
  - `Mean reward = 64.65`
  - `rew_reach_pos_target_tight = 2.0612`
- `iter 700`
  - `Mean reward = -452.77`
  - `rew_reach_pos_target_tight = 0.4357`

Interpretation at this point:

- this optimizer-restored continuation is not a dead start
- but through `iter 710`, its early reach signal is weaker than the earlier `model_400 -> broad/full` short transfer line
- so it is not yet clear that "resume from broad/full best with optimizer state" is a better path than restarting broad/full transfer from the native fixed-room winner

### The optimizer-restored continuation produced the first true broad/full hard-room success, but still drifted by the end

That continuation run then finished:

- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_opt/05_26_01-08-49_400it_resume_bestreach_opt`
- final checkpoint:
  - `model_984.pt`

Checkpoint metadata:

- `best_mean_reward.pt` at `iter = 601`
- `best_reach.pt` at `iter = 613`
- `model_984.pt` at `iter = 984`

I evaluated its `best_reach.pt` under the same host-GPU hard-room protocol:

- `native_go2`
- `ideal_pd`
- `10` episodes
- `--disable-contact-termination`
- `--stay-steps 500`

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_01-14-42`

Observed result:

- `success_count = 1/10`
- `collision_free_success_count = 1/10`
- `stand_failures = 3`
- `fall_failures = 1`
- `max_step_failures = 5`
- `mean_reach_reward_sum = 306.2148`
- `mean_min_distance = 1.9160`

This is the first true broad/full hard-room `goal_hold` success in the whole redeploy arc.

Episode-level evidence:

- success episode:
  - episode `8`
  - `min_distance = 0.1104`
  - `reach_reward_sum = 1356.31`
  - `done_reason = goal_hold`
- additional goal-region entry episodes:
  - episode `7`: `min_distance = 0.0439`, `reach_reward_sum = 1008.65`, `done_reason = max_steps`
  - episode `2`: `min_distance = 0.1247`, `reach_reward_sum = 483.03`, `done_reason = stand`
  - episode `5`: `min_distance = 0.3289`, `reach_reward_sum = 214.16`, `done_reason = max_steps`

Interpretation:

- compared with the earlier best transfer checkpoint, this branch did convert at least one broad/full rollout into a true `goal_hold` success
- however, the tail checkpoint still drifted

Final checkpoint eval:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_01-18-17`
- `model_984.pt` result:
  - `success_count = 0/10`
  - `stand_failures = 5`
  - `max_step_failures = 5`
  - `mean_reach_reward_sum = 51.6288`
  - `mean_min_distance = 3.2410`

So the branch proved that broad/full success is reachable, but it still regressed when trained too long.

### Lower-learning-rate fine-tuning from the new success checkpoint improved stability further

Because the optimizer-restored continuation proved that success is reachable but still drift-prone, I added CLI overrides in:

- `training/isaac_lab/train.py`

for:

- `--learning-rate`
- `--lr-schedule`

and also made resumed runs explicitly re-apply the requested learning rate after checkpoint load, so resumed fine-tunes are not silently stuck on the old optimizer LR.

Static verification passed:

```bash
python3 -m py_compile training/isaac_lab/train.py
```

I then launched a more conservative continuation from the new successful checkpoint:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_opt/05_26_01-08-49_400it_resume_bestreach_opt/best_reach.pt`
- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_lowlr/05_26_01-22-23_200it_lowlr_fixed`
- config:
  - `learning_rate = 1e-4`
  - `schedule = fixed`
  - `load_optimizer = False`

Checkpoint metadata from that run:

- `best_reach.pt` at `iter = 717`
- `best_mean_reward.pt` at `iter = 807`
- final `model_813.pt`

The training curve on this branch looked materially healthier in the middle:

- `iter 740`: `rew_reach_pos_target_tight = 13.7634`
- `iter 750`: `rew_reach_pos_target_tight = 14.3450`
- `iter 800`: `rew_reach_pos_target_tight = 6.8237`

#### `best_reach.pt` from the low-LR continuation

Host eval log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_01-25-32`

Observed result:

- `success_count = 1/10`
- `collision_free_success_count = 1/10`
- `stand_failures = 1`
- `max_step_failures = 8`
- `mean_reach_reward_sum = 351.7147`
- `mean_min_distance = 1.2470`

Episode-level evidence:

- success episode:
  - episode `1`
  - `min_distance = 0.1196`
  - `reach_reward_sum = 1210.71`
  - `done_reason = goal_hold`
- additional goal-region entry episodes:
  - episode `8`: `min_distance = 0.0082`, `reach_reward_sum = 749.33`, `done_reason = max_steps`
  - episode `0`: `min_distance = 0.1038`, `reach_reward_sum = 368.74`, `done_reason = max_steps`
  - episode `2`: `min_distance = 0.1072`, `reach_reward_sum = 353.54`, `done_reason = max_steps`
  - episode `9`: `min_distance = 0.1106`, `reach_reward_sum = 589.33`, `done_reason = max_steps`
  - episode `5`: `min_distance = 0.2798`, `reach_reward_sum = 245.48`, `done_reason = max_steps`

Interpretation:

- success rate remained `1/10`
- but the branch improved broad/full stability meaningfully:
  - from `4/10` goal-region entries on the earlier best transfer checkpoint
  - to `6/10` goal-region entries here
- it also improved the average distance and average reach reward

#### `best_mean_reward.pt` from the low-LR continuation

Host eval log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_01-29-11`

Observed result:

- `success_count = 0/10`
- `max_step_failures = 10`
- `mean_reach_reward_sum = 174.9407`
- `mean_min_distance = 1.4598`

So for this branch as well, the strongest checkpoint is `best_reach.pt`, not `best_mean_reward.pt`.

### A smaller-step `5e-5` continuation from the improved low-LR checkpoint is now running

Since the `1e-4` fixed-schedule branch improved goal-region entry consistency from `4/10` to `6/10` while preserving `1/10 goal_hold`, I started an even smaller-step continuation from its `best_reach.pt`:

- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_ultralow/05_26_01-33-13_150it_5e-5_fixed`
- config:
  - `learning_rate = 5e-5`
  - `schedule = fixed`
  - `load_optimizer = False`

Verified startup:

- `resumed runner checkpoint=.../best_reach.pt load_optimizer=False current_iteration=717 learning_rate=5e-05 schedule=fixed`

Early signal so far:

- `iter 720`
  - `rew_reach_pos_target_tight = 1.5320`
- `iter 730`
  - `rew_reach_pos_target_tight = 5.7052`
- `iter 740`
  - `rew_reach_pos_target_tight = 13.7634`
- `iter 750`
  - `rew_reach_pos_target_tight = 14.3450`
- `iter 800`
  - `rew_reach_pos_target_tight = 6.8237`

This is not enough to claim success yet, but it is the current best-aligned continuation path:

- it starts from the first true broad/full `goal_hold` checkpoint lineage
- it avoids the obvious late-stage drift seen in higher-LR continuations
- and it already preserves strong mid-run reach signals

### The `5e-5` fixed-schedule continuation raised broad/full hard-room success to `2/10`

That `5e-5` continuation then finished:

- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_ultralow/05_26_01-33-13_150it_5e-5_fixed`
- checkpoint metadata:
  - `best_reach.pt` at `iter = 785`
  - `best_mean_reward.pt` at `iter = 842`
  - final `model_867.pt`

I evaluated its `best_reach.pt` under the same host-GPU hard-room protocol.

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_01-35-44`

Observed result:

- `success_count = 2/10`
- `collision_free_success_count = 2/10`
- `max_step_failures = 8`
- `mean_reach_reward_sum = 439.6732`
- `mean_min_distance = 1.1127`

This is the strongest broad/full result so far.

Episode-level evidence:

- success episodes:
  - episode `1`
    - `min_distance = 0.0874`
    - `reach_reward_sum = 1298.90`
    - `done_reason = goal_hold`
  - episode `9`
    - `min_distance = 0.0112`
    - `reach_reward_sum = 1251.99`
    - `done_reason = goal_hold`
- additional goal-region entry episodes:
  - episode `0`: `min_distance = 0.0368`, `reach_reward_sum = 574.55`, `done_reason = max_steps`
  - episode `7`: `min_distance = 0.0565`, `reach_reward_sum = 453.46`, `done_reason = max_steps`
  - episode `5`: `min_distance = 0.1153`, `reach_reward_sum = 588.49`, `done_reason = max_steps`
  - episode `6`: `min_distance = 0.3347`, `reach_reward_sum = 229.35`, `done_reason = max_steps`

Interpretation:

- success rate improved from:
  - `0/10` on the earlier broad/full transfer baseline
  - to `1/10` on the optimizer-restored continuation
  - to `2/10` here
- goal-region entry frequency stayed high:
  - `6/10` episodes entered the goal region
- so the direction is now clearly positive, even though it is still far from paper-level robustness

I also evaluated the same branch's `best_mean_reward.pt`.

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_01-39-24`

Observed result:

- `success_count = 1/10`
- `max_step_failures = 8`
- `stand_failures = 1`
- `mean_reach_reward_sum = 410.1840`
- `mean_min_distance = 1.4509`

So this branch's strongest checkpoint remains `best_reach.pt`, not `best_mean_reward.pt`.

### A new stabilization branch with even lower LR and zero entropy bonus is now running

Since the best current failure mode is no longer "cannot reach", but rather "many episodes enter the goal region and only some convert to `goal_hold`", I added one more fine-tuning axis:

- `training/isaac_lab/train.py`
  - added CLI override:
    - `--entropy-coef`

Static verification passed:

```bash
python3 -m py_compile training/isaac_lab/train.py
```

Then I launched a hold-stabilization branch from the current strongest checkpoint:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_ultralow/05_26_01-33-13_150it_5e-5_fixed/best_reach.pt`
- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_stabilize_hold/05_26_01-43-31_120it_2e-5_ent0`
- config:
  - `learning_rate = 2e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.0`
  - `load_optimizer = False`

Verified startup:

- `resumed runner checkpoint=.../best_reach.pt load_optimizer=False current_iteration=785 learning_rate=2e-05 schedule=fixed`

Early signal so far:

- `iter 800`
  - `rew_reach_pos_target_tight = 6.7466`
- `iter 810`
  - `rew_reach_pos_target_tight = 0.0000`

This branch is intended specifically to improve conversion from:

- "enter goal region" -> "hold long enough to finish"

rather than to relearn broad/full navigation from scratch.

### The first `2e-5 + entropy=0` stabilization branch did not beat the current best checkpoint

That stabilization run finished:

- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_stabilize_hold/05_26_01-43-34_120it_2e-5_ent0`
- checkpoint metadata:
  - `best_reach.pt` at `iter = 825`
  - `best_mean_reward.pt` at `iter = 902`
  - final `model_905.pt`

Its training curve did show repeated non-zero reach signals:

- `iter 830`: `rew_reach_pos_target_tight = 15.5639`
- `iter 850`: `rew_reach_pos_target_tight = 3.1970`
- `iter 890`: `rew_reach_pos_target_tight = 16.1324`
- `iter 900`: `rew_reach_pos_target_tight = 9.8796`

But host hard-room eval of its `best_reach.pt` did **not** surpass the current strongest `5e-5` baseline.

Result log:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_01-45-52`

Observed result:

- `success_count = 1/10`
- `collision_free_success_count = 1/10`
- `stand_failures = 1`
- `max_step_failures = 8`
- `mean_reach_reward_sum = 395.2457`
- `mean_min_distance = 1.1756`

Interpretation:

- this branch is still clearly better than the older `0/10` era
- but it does **not** improve on the current strongest checkpoint, which remains:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_ultralow/05_26_01-33-13_150it_5e-5_fixed/best_reach.pt`
  - with `2/10` broad/full hard-room `goal_hold` success

So the evidence now points to a narrower next question:

- lower learning rate definitely helped
- but the previous `2e-5 + entropy=0` branch changed both LR and exploration at once
- therefore the clean next A/B is:
  - keep the stronger `5e-5` LR
  - change only `entropy_coef`
  - and measure whether success conversion improves over the current `2/10` baseline

### High-sample hard-room eval shows the pure-policy line is still plateaued around `4/30`

To reduce selection noise from `10`-episode rollouts, I reran the strongest broad/full checkpoints with `30`-episode host hard-room evals using the same play-style settings:

- `episodes = 30`
- `stay_steps = 500`
- `disable_contact_termination = true`
- `robot_asset_source = native_go2`
- `actuator_mode = ideal_pd`

Current best pure-policy baseline:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_ultralow/05_26_01-33-13_150it_5e-5_fixed/best_reach.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_02-10-27`
- observed result:
  - `success_count = 4/30`
  - `collision_free_success_count = 4/30`
  - `stand_failures = 1`
  - `max_step_failures = 25`
  - `mean_reward_sum = -5125.8216`
  - `mean_reach_reward_sum = 364.2581`
  - `mean_min_distance = 1.4534`
- trace-derived behavior summary:
  - `15/30` episodes entered the goal region (`min_distance < 0.5`)
  - `6/30` episodes reached `min_distance < 0.1`

The follow-up `bestgoalhold_probe` branch did not produce a pure-policy checkpoint that clearly beat this baseline.

Reach-focused checkpoint from that branch:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_bestgoalhold_probe/05_26_01-56-52_120it_5e-5_bestgoalhold/best_reach.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_02-20-22`
- observed result:
  - `success_count = 2/30`
  - `stand_failures = 2`
  - `fall_failures = 3`
  - `max_step_failures = 23`
  - `mean_reach_reward_sum = 440.5618`
  - `mean_min_distance = 1.4355`
- trace-derived behavior summary:
  - `19/30` episodes entered the goal region
  - `11/30` episodes reached `min_distance < 0.1`

Interpretation:

- this checkpoint is actually *better at reaching the goal region deeply*
- but it is *worse at converting that reach into stable `goal_hold` success*
- so the dominant failure is no longer reaching, but stopping/holding

Mean-reward checkpoint from the same branch:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_bestgoalhold_probe/05_26_01-56-52_120it_5e-5_bestgoalhold/best_mean_reward.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_02-29-23`
- observed result:
  - `success_count = 4/30`
  - `stand_failures = 3`
  - `max_step_failures = 23`
  - `mean_reward_sum = -7076.7152`
  - `mean_reach_reward_sum = 480.0604`
  - `mean_min_distance = 1.6428`
- trace-derived behavior summary:
  - `16/30` episodes entered the goal region
  - `9/30` episodes reached `min_distance < 0.1`

Interpretation:

- this checkpoint ties the best `4/30` pure-policy success count
- but it does not improve overall robustness enough to replace the current baseline

### A conservative low-LR, low-entropy hold-stabilization continuation still did not beat `4/30`

I launched one more short continuation from the branch checkpoint that tied the best high-sample success while reaching the goal region slightly more often:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_bestgoalhold_probe/05_26_01-56-52_120it_5e-5_bestgoalhold/best_mean_reward.pt`
- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_hold_stabilize_lowent/05_26_02-40-05_80it_1e-5_ent1e-3`
- config:
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `load_optimizer = False`

Checkpoint metadata:

- `best_goal_hold.pt` at `iter = 953`
- `best_reach.pt` at `iter = 953`
- `best_mean_reward.pt` at `iter = 921`
- final `model_973.pt`

Host hard-room eval of `best_goal_hold.pt`:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_02-41-39`
- observed result:
  - `success_count = 4/30`
  - `stand_failures = 5`
  - `fall_failures = 1`
  - `max_step_failures = 20`
  - `mean_reach_reward_sum = 499.2283`
  - `mean_min_distance = 1.6256`
- trace-derived behavior summary:
  - `18/30` episodes entered the goal region
  - `11/30` episodes reached `min_distance < 0.1`

Interpretation:

- this continuation again improved *goal-region entry frequency*
- but it still did **not** improve pure-policy `goal_hold` success beyond `4/30`
- the plateau is now very clear

### Stop-override diagnostics show the main blocker is high-level stopping, not low-level hold ability

To isolate whether the remaining problem is "cannot hold position physically" versus "policy does not command stop", I added a probe-only diagnostic flag:

- `training/isaac_lab/manual_reward_probe.py`
  - new CLI:
    - `--policy-stop-radius`
  - behavior:
    - when active, if the current policy rollout is already within the specified radius, the navigation command is overridden to zero for that step

Static verification passed:

```bash
python3 -m py_compile training/isaac_lab/manual_reward_probe.py
```

I then reran the current strongest pure-policy checkpoint with `policy_stop_radius = 0.45`.

Diagnostic eval:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_resume_bestreach_ultralow/05_26_01-33-13_150it_5e-5_fixed/best_reach.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_02-52-07`
- observed result:
  - `success_count = 17/30`
  - `collision_free_success_count = 17/30`
  - `stand_failures = 1`
  - `max_step_failures = 12`
  - `mean_reward_sum = -2414.8823`
  - `mean_reach_reward_sum = 666.4910`

I also reran the reach-heavy but pure-policy-weaker checkpoint from the `bestgoalhold_probe` branch under the same stop override.

Diagnostic eval:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_bestgoalhold_probe/05_26_01-56-52_120it_5e-5_bestgoalhold/best_reach.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_03-02-04`
- observed result:
  - `success_count = 18/30`
  - `collision_free_success_count = 18/30`
  - `stand_failures = 0`
  - `fall_failures = 0`
  - `max_step_failures = 12`
  - `mean_reach_reward_sum = 711.7659`

This is the strongest diagnostic result in the whole redeploy arc so far.

Interpretation:

- once the navigation command is forced to zero inside the goal region, success jumps from:
  - `4/30` pure policy
  - to `17/30` or `18/30` with the same policy plus stop override
- therefore the low-level controller and Isaac Lab execution path **can** hold the robot in the goal region when the command is stopped
- the main remaining blocker is now sharply isolated:
  - the high-level policy has learned to *reach* reasonably often
  - but it has **not** learned to *stop issuing motion commands cleanly enough* inside the goal region

Current exact status:

- pure-policy broad/full hard-room performance is still only about `4/30`
- a probe-only goal-stop prior raises that to about `17-18/30`
- so SEA-Nav on this machine is **not** fully reproduced yet under pure policy semantics
- but the remaining gap is now behaviorally localized to near-goal stopping rather than broad navigation reachability

### Integrating the hard stop prior into training improved broad/full hard-room success from `18/30` to `21/30`

Since the probe-only zero-command prior was the first intervention that clearly moved the success rate, I integrated the same behavior into the actual Isaac Lab environment and training entrypoint as an optional, default-off setting:

- `training/isaac_lab/sea_nav_env.py`
  - new config:
    - `goal_stop_radius`
    - `goal_stop_mode`
  - current implementation supports:
    - `goal_stop_mode = "zero"`
    - `goal_stop_mode = "linear"`
- `training/isaac_lab/train.py`
  - new CLI:
    - `--goal-stop-radius`
    - `--goal-stop-mode`
- `training/isaac_lab/manual_reward_probe.py`
  - matching probe-only controls:
    - `--policy-stop-radius`
    - `--policy-stop-mode`

Static verification passed:

```bash
python3 -m py_compile \
  training/isaac_lab/sea_nav_env.py \
  training/isaac_lab/train.py \
  training/isaac_lab/manual_reward_probe.py
```

I then launched a real continuation branch that used the hard stop prior inside training, not only in probe-time eval:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400_bestgoalhold_probe/05_26_01-56-52_120it_5e-5_bestgoalhold/best_reach.pt`
- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_finetune/05_26_03-21-10_120it_1e-5_ent1e-3`
- config:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`

Training signal was much stronger than the earlier pure-policy lines:

- `iter 900`
  - `goal_hold_success = 0.4792`
  - `rew_reach_pos_target_tight = 8.6296`
- `iter 950`
  - `goal_hold_success = 0.5833`
  - `rew_reach_pos_target_tight = 11.8018`

Checkpoint metadata:

- `best_goal_hold.pt` at `iter = 912`
- `best_reach.pt` at `iter = 951`
- `best_mean_reward.pt` at `iter = 900`

Host hard-room eval of `best_goal_hold.pt` under matching stop-prior semantics:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_03-24-46`
- observed result:
  - `success_count = 21/30`
  - `collision_free_success_count = 21/30`
  - `stand_failures = 1`
  - `fall_failures = 2`
  - `max_step_failures = 6`
  - `mean_reach_reward_sum = 863.6640`

This is the current strongest verified result in the entire redeploy arc.

Meaning:

- pure policy baseline:
  - `4/30`
- probe-only hard stop prior:
  - `17/30` to `18/30`
- trained hard stop prior:
  - `21/30`

So the hard stop prior is not just a probe-time trick. Once brought into training, it materially improves the learned policy under the same eval semantics.

### A more conservative `5e-6` hard-stop refinement regressed to `15/30`

To see whether the new `21/30` result could be pushed higher with a smaller update size, I launched a second continuation:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_finetune/05_26_03-21-10_120it_1e-5_ent1e-3/best_goal_hold.pt`
- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_refine/05_26_03-43-27_80it_5e-6_ent1e-3`
- config:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 5e-6`
  - `schedule = fixed`
  - `entropy_coef = 0.001`

Training-period `goal_hold_success` looked promising:

- `iter 940`
  - `goal_hold_success = 1.0000`
  - `rew_reach_pos_target_tight = 18.1062`
- `iter 960`
  - `goal_hold_success = 0.8542`
  - `rew_reach_pos_target_tight = 17.5423`

But fresh hard-room eval did **not** improve the real result.

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_03-45-26`
- observed result:
  - `success_count = 15/30`
  - `collision_free_success_count = 15/30`
  - `stand_failures = 3`
  - `fall_failures = 3`
  - `max_step_failures = 9`

Interpretation:

- more conservative continuation from the new `21/30` checkpoint did not help
- the current best remains:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_finetune/05_26_03-21-10_120it_1e-5_ent1e-3/best_goal_hold.pt`

### Linear slowdown is worse than hard zero for the current best checkpoint

I also tested whether a smoother near-goal prior could beat the hard zero stop.

Probe-only linear slowdown diagnostic:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_finetune/05_26_03-21-10_120it_1e-5_ent1e-3/best_goal_hold.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_03-53-06`
- config:
  - `policy_stop_mode = linear`
  - `policy_stop_radius = 0.8`
- observed result:
  - `success_count = 16/30`
  - `collision_free_success_count = 16/30`
  - `stand_failures = 2`
  - `fall_failures = 0`
  - `max_step_failures = 12`

This is clearly worse than the hard-zero prior on the same checkpoint:

- hard-zero:
  - `21/30`
- linear slowdown:
  - `16/30`

So among the tested stop priors so far:

- `zero` is better than `linear`
- and `0.45` versus `0.5` is effectively a wash on the current best checkpoint

Current exact status after this round:

- strict pure-policy broad/full hard-room result is still only `4/30`
- the best verified stop-prior-assisted training result is now `21/30`
- continuing to reduce LR further did not improve on that
- replacing hard-zero with linear slowdown also did not improve on that
- therefore the current strongest actionable branch is:
  - hard zero near-goal prior
  - around `goal_stop_radius = 0.45`
  - with the best checkpoint at:
    - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_finetune/05_26_03-21-10_120it_1e-5_ent1e-3/best_goal_hold.pt`

### One-hard-room targeted continuation did not beat the `21/30` mainline

I then tested whether the remaining failures were mostly caused by broad-distribution
generalization by continuing training directly on the same single hard-room
distribution used by host eval:

- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_onehardroom_goalstop045/05_26_04-36-56_120it_1e-5_ent1e-3_opt`
- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_finetune/05_26_03-21-10_120it_1e-5_ent1e-3/best_mean_reward.pt`
- config:
  - `terrain_rows = 1`
  - `terrain_cols = 1`
  - `obstacle_level = 9`
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

Training itself was not dead:

- `best_goal_hold.pt` landed at iter `930`
- `best_reach.pt` landed at iter `1000`
- `best_mean_reward.pt` landed at iter `1019`
- multiple training windows showed `goal_hold_success = 1.0000`

But fresh host hard-room eval still failed to beat the current `21/30` mainline.

`best_mean_reward.pt`:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_04-50-12`
- observed result:
  - `success_count = 19/30`
  - `collision_free_success_count = 19/30`
  - `stand_failures = 0`
  - `fall_failures = 0`
  - `max_step_failures = 11`
  - `mean_reach_reward_sum = 772.4511`
  - `mean_min_distance = 1.0044`

`best_reach.pt`:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_05-00-17`
- observed result:
  - `success_count = 19/30`
  - `collision_free_success_count = 19/30`
  - `stand_failures = 8`
  - `fall_failures = 0`
  - `max_step_failures = 3`
  - `mean_reach_reward_sum = 758.8214`
  - `mean_min_distance = 1.4459`

Interpretation:

- matching the eval distribution more aggressively did **not** produce a better
  checkpoint than the existing broad/full `hard-zero` mainline
- `best_mean_reward.pt` on this branch is cleaner than `best_reach.pt`, but both
  stall at `19/30`
- the current best verified result therefore remains:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_finetune/05_26_03-21-10_120it_1e-5_ent1e-3/best_goal_hold.pt`
  - `21/30 success`

Current exact status after this follow-up:

- strict pure-policy broad/full hard-room is still `4/30`
- the best verified `hard-zero` near-goal-prior result is still `21/30`
- one-hard-room targeted continuation reached `19/30`, so it does not replace the
  current mainline
- the next meaningful work should focus on the remaining reach-limited failures
  in the `21/30` branch, not on continuing this `19/30` one-hard-room branch

### Failure-case-only continuation matched `21/30`, but only by swapping success cases

To avoid another blind branch, I added a minimal continuation path that can load
exact start/goal/yaw cases directly from an existing hard-room eval trace.

Code changes:

- `training/isaac_lab/train.py`
  - new CLI:
    - `--preset-case-trace`
    - `--preset-case-episodes`
- `training/isaac_lab/sea_nav_env.py`
  - new cfg field:
    - `preset_start_goal_cases`
  - reset path can now sample explicit hard-room cases instead of always calling
    `place_robot_and_goal(...)`

I verified the new path with a real host smoke run before using it for training:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless --num-envs 4 --smoke-steps 1 \
  --preset-case-trace logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_04-03-38/trace.jsonl \
  --preset-case-episodes 4,7,8,12,17,18,20,21,27 \
  --goal-stop-radius 0.45 --goal-stop-mode zero \
  --robot-asset-source native_go2 --actuator-mode ideal_pd
```

Then I launched a short continuation from the current `21/30` mainline source
checkpoint, using only the `9` remaining failure episodes from the `21/30`
hard-room eval:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_finetune/05_26_03-21-10_120it_1e-5_ent1e-3/best_mean_reward.pt`
- failure trace:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_04-03-38/trace.jsonl`
- selected failure episodes:
  - `4,7,8,12,17,18,20,21,27`
- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failurecases_goalstop045/05_26_05-12-21_80it_1e-5_ent1e-3_opt`
- config:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

Training signal was not obviously strong:

- the resumed run started from `iter = 900`
- through `iter 920/930/940/950/960/970`, `rew_reach_pos_target_tight` stayed `0.0000`
- `best_mean_reward.pt` never refreshed past the source checkpoint

I still evaluated the new checkpoint instead of assuming failure:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failurecases_goalstop045/05_26_05-12-21_80it_1e-5_ent1e-3_opt/best_goal_hold.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_05-13-31`
- observed result:
  - `success_count = 21/30`
  - `collision_free_success_count = 21/30`
  - `stand_failures = 1`
  - `fall_failures = 0`
  - `max_step_failures = 8`
  - `mean_reach_reward_sum = 830.5730`
  - `mean_min_distance = 0.9701`

So this branch did **not** beat the current mainline; it only tied it.

More importantly, episode-by-episode comparison showed it was a swap, not a real
net improvement:

- mainline-only successes lost by the failure-case continuation:
  - `11, 14, 16, 22, 25`
- failure cases recovered by the continuation:
  - `7, 8, 17, 20, 21`
- still failing in both:
  - `4, 12, 18, 27`

Interpretation:

- single hard-room continuation (`19/30`) was too narrow
- failure-case-only continuation (`21/30`) is also too narrow
- the latter proves the new trace-driven continuation path works, but it mainly
  trades one subset of cases for another
- the next sensible direction is therefore **mixed curriculum**:
  keep the current broad/full `hard-zero` mainline behavior, while selectively
  rehearsing the unresolved hard-room failure cases, instead of training on the
  failure set alone

Current exact status after this branch:

- strict pure-policy broad/full hard-room is still `4/30`
- the best verified near-goal-prior-assisted result is still `21/30`
- one-hard-room-only continuation peaks at `19/30`
- failure-case-only continuation ties `21/30` but does not surpass it
- the remaining problem is no longer “can a targeted branch help at all”, but
  “how to improve the unresolved hard-room failures without regressing the
  already-solved ones”

### Mixed replay finally produced a real net improvement

The next step was to stop training on narrow slices alone and move to a true
mixed curriculum inside the same hard-room distribution:

- keep random hard-room resets alive
- replay the current unresolved failure cases only with a probability

Code changes:

- `training/isaac_lab/train.py`
  - new CLI:
    - `--preset-case-prob`
- `training/isaac_lab/sea_nav_env.py`
  - new cfg field:
    - `preset_start_goal_case_prob`
  - when preset cases exist, reset now samples:
    - replayed hard cases with probability `preset_start_goal_case_prob`
    - ordinary random hard-room starts/goals otherwise

I first verified the path with a real host smoke run:

```bash
env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless --num-envs 4 --smoke-steps 1 \
  --preset-case-trace logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_04-03-38/trace.jsonl \
  --preset-case-episodes 4,7,8,12,17,18,20,21,27 \
  --preset-case-prob 0.5 \
  --goal-stop-radius 0.45 --goal-stop-mode zero \
  --robot-asset-source native_go2 --actuator-mode ideal_pd
```

Then I launched the first mixed continuation:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_finetune/05_26_03-21-10_120it_1e-5_ent1e-3/best_mean_reward.pt`
- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045/05_26_05-24-32_80it_1e-5_ent1e-3_p05_opt`
- config:
  - `preset_case_episodes = 4,7,8,12,17,18,20,21,27`
  - `preset_case_prob = 0.5`
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

This branch already improved on the original `30`-episode gate:

- old broad/full hard-zero mainline:
  - `21/30`
- mixed replay `best_mean_reward.pt`:
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_05-25-31`
  - observed result:
    - `23/30`

The `best_goal_hold.pt` from the same run was worse:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_05-31-57`
- observed result:
  - `19/30`

To avoid overreacting to a `30`-episode wobble, I then tried a higher-sample
gate. Running two `100`-episode host evals concurrently turned out to be
operationally unreliable: both processes disappeared without writing summaries,
so I did **not** treat that attempt as evidence.

Instead, I switched to a more stable sequential `50`-episode gate:

Old mainline checkpoint:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_finetune/05_26_03-21-10_120it_1e-5_ent1e-3/best_mean_reward.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-19-49`
- observed result:
  - `success_count = 37/50`
  - `success_rate = 0.74`
  - `collision_free_success_count = 37`
  - `stand_failures = 2`
  - `max_step_failures = 11`

New mixed replay checkpoint:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045/05_26_05-24-32_80it_1e-5_ent1e-3_p05_opt/best_mean_reward.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-33-12`
- observed result:
  - `success_count = 39/50`
  - `success_rate = 0.78`
  - `collision_free_success_count = 39`
  - `stand_failures = 2`
  - `max_step_failures = 9`

This is the first continuation that survives a higher-sample gate as a real net
improvement:

- old mainline:
  - `37/50`
- new mixed replay:
  - `39/50`

Interpretation:

- `failure-case-only` was too narrow
- `one-hard-room-only` was too narrow
- `mixed replay` is the first continuation that improves the hard-room hold
  rate without collapsing back to the old baseline
- the improvement is still modest, but it is now supported by a stronger gate
  than the earlier `30`-episode checks

For completeness, I also checked the other `best_*` snapshots from this same
winner run:

- `best_goal_hold.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_05-31-57`
  - observed result:
    - `19/30`
- `best_reach.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_07-04-57`
  - observed result:
    - `19/30`

So for this winner run, `best_mean_reward.pt` is indeed the strongest snapshot,
not `best_goal_hold.pt` or `best_reach.pt`.

Current exact status after this round:

- strict pure-policy broad/full hard-room is still `4/30`
- the strongest verified near-goal-prior-assisted checkpoint is now:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045/05_26_05-24-32_80it_1e-5_ent1e-3_p05_opt/best_mean_reward.pt`
- its current verified gates are:
  - `23/30`
  - `39/50`
- this is still not close enough to claim “paper-level reproduction”, but it is
  a real step beyond the previous `21/30` and `37/50` lines

### Continuing the stronger mixed-replay line immediately caused regression

Because the first mixed-replay run was the only continuation that improved under
a stronger gate, I continued it with the exact same configuration rather than
changing multiple knobs at once:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045/05_26_05-24-32_80it_1e-5_ent1e-3_p05_opt/best_mean_reward.pt`
- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045_continue/05_26_06-44-45_80it_1e-5_ent1e-3_p05_opt`
- config:
  - same `preset_case_episodes = 4,7,8,12,17,18,20,21,27`
  - same `preset_case_prob = 0.5`
  - same `goal_stop_radius = 0.45`
  - same `goal_stop_mode = zero`
  - same `learning_rate = 1e-5`
  - same `schedule = fixed`
  - same `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

This continuation looked alive in training:

- by `iter 1020`:
  - `rew_reach_pos_target_tight = 5.9686`
  - `goal_hold_success = 0.3125`

But fresh eval contradicted that optimism.

Key checkpoint metadata:

- `best_goal_hold.pt` -> iter `976`
- `best_mean_reward.pt` -> iter `944`
- `best_reach.pt` -> iter `1008`

I evaluated the newly refreshed `best_reach.pt` first:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-46-49`
- observed result:
  - `success_count = 14/30`
  - `collision_free_success_count = 14/30`
  - `stand_failures = 3`
  - `fall_failures = 1`
  - `max_step_failures = 12`
  - `mean_reach_reward_sum = 561.8628`
  - `mean_min_distance = 1.5512`

Interpretation:

- extending the same mixed-replay line blindly did **not** help
- the current best checkpoint still remains the earlier mixed-replay winner:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045/05_26_05-24-32_80it_1e-5_ent1e-3_p05_opt/best_mean_reward.pt`
- that checkpoint is still the strongest verified point with:
  - `23/30`
  - `39/50`

Current exact status after this follow-up:

- `mixed replay` is still the only branch that has shown a stronger-gate uplift
- but simply continuing the same branch longer is not monotonic and already
  regressed sharply on a fresh `30`-episode gate
- the current strongest known checkpoint remains the earlier `39/50` mixed-replay
  winner, not the later continuation

### Lowering failure-case replay probability to `0.25` also regressed

I then tested whether the problem was not mixed replay itself, but the replay
ratio being too aggressive.

New continuation:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045/05_26_05-24-32_80it_1e-5_ent1e-3_p05_opt/best_mean_reward.pt`
- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045_p025/05_26_06-54-49_80it_1e-5_ent1e-3_p025_opt`
- config change:
  - `preset_case_prob = 0.25`
- all other knobs unchanged:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

This branch again showed some positive training-period signals:

- `iter 1000`
  - `rew_reach_pos_target_tight = 10.5579`
  - `goal_hold_success = 0.5625`

But fresh eval still failed to beat the earlier mixed-replay winner.

Key checkpoint metadata:

- `best_goal_hold.pt` -> iter `1001`
- `best_mean_reward.pt` -> iter `964`
- `best_reach.pt` -> iter `1016`

Fresh hard-room eval:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045_p025/05_26_06-54-49_80it_1e-5_ent1e-3_p025_opt/best_goal_hold.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-55-53`
- observed result:
  - `success_count = 18/30`
  - `collision_free_success_count = 18/30`
  - `stand_failures = 0`
  - `fall_failures = 0`
  - `max_step_failures = 12`
  - `mean_reach_reward_sum = 713.7564`
  - `mean_min_distance = 1.5198`

Interpretation:

- the more conservative replay ratio did **not** outperform the earlier
  `preset_case_prob = 0.5` winner
- reducing replay strength alone is therefore not the next obvious fix
- the current strongest verified checkpoint is still:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045/05_26_05-24-32_80it_1e-5_ent1e-3_p05_opt/best_mean_reward.pt`
  - with:
    - `23/30`
    - `39/50`

Current exact status after this additional branch:

- `mixed replay` remains the best known direction
- neither continuing the `p=0.5` branch longer nor lowering replay to `p=0.25`
  improved on the original mixed-replay winner
- the current best verified result remains:
  - `23/30`
  - `39/50`

### Replaying the newer `39/50` winner's own failures did not help

I then updated the replay source itself.

Instead of continuing to train against the older failure set from the `21/30`
era, I extracted the actual failures from the stronger `39/50` gate:

- source trace:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-33-12/trace.jsonl`
- failed episodes:
  - `2,3,5,8,13,18,28,35,36,39,41`
- count:
  - `11`

This produced a small but clean 2x2 replay matrix:

- old failure set + `p=0.5`
  - best branch
- old failure set + `p=0.25`
  - weaker
- current failure set + `p=0.5`
  - untested before this round
- current failure set + `p=0.25`
  - untested before this round

#### Current failure set + `p=0.5`

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045_curfail50/05_26_07-20-09_80it_1e-5_ent1e-3_p05_opt`

Config:

- `preset_case_trace = logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-33-12/trace.jsonl`
- `preset_case_episodes = 2,3,5,8,13,18,28,35,36,39,41`
- `preset_case_prob = 0.5`
- remaining knobs unchanged:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

Training looked promising for a moment:

- `iter 970`
  - `rew_reach_pos_target_tight = 6.6562`
  - `goal_hold_success = 0.3750`
- `iter 990`
  - `rew_reach_pos_target_tight = 18.0435`
  - `goal_hold_success = 1.0000`

But fresh eval contradicted the training-period optimism:

- `best_goal_hold.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_07-21-07`
  - observed result:
    - `16/30`
- `best_mean_reward.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_07-28-23`
  - observed result:
    - `15/30`

So `current failure set + p=0.5` is clearly worse than the earlier mixed-replay
winner.

#### Current failure set + `p=0.25`

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045_curfail50_p025/05_26_07-35-48_80it_1e-5_ent1e-3_p025_opt`

Config:

- same current failure trace and episode list
- `preset_case_prob = 0.25`

Again the training-period signal looked alive:

- `iter 980`
  - `rew_reach_pos_target_tight = 5.5831`
  - `goal_hold_success = 0.3125`

But fresh eval again failed to beat the original mixed-replay winner:

- `best_goal_hold.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-55-53`
  - observed result:
    - `18/30`
- `best_mean_reward.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_07-36-47`
  - observed result:
    - `19/30`

Interpretation:

- replaying the `39/50` winner's own current failures is **not** the immediate
  fix
- those failures appear to be sharper and easier to overfit than the older
  replay source
- the earlier mixed-replay winner still dominates every tested continuation:
  - `23/30`
  - `39/50`

Current exact status after completing this replay matrix:

- the strongest verified checkpoint remains:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045/05_26_05-24-32_80it_1e-5_ent1e-3_p05_opt/best_mean_reward.pt`
- verified gates for that checkpoint:
  - `23/30`
  - `39/50`
- all tested continuations from that checkpoint so far have regressed:
  - same stale failure set, `p=0.5` -> `14/30`
  - same stale failure set, `p=0.25` -> `18/30` / `19/30`
  - current `39/50` failure set, `p=0.5` -> `16/30` / `15/30`
  - current `39/50` failure set, `p=0.25` -> `18/30` / `19/30`

For completeness, I also evaluated the remaining `best_reach.pt` snapshot from
the `p=0.25` branch:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_07-12-03`
- observed result:
  - `18/30`
  - `collision_free_success_count = 18`
  - `stand_failures = 0`
  - `fall_failures = 0`
  - `max_step_failures = 12`

So the `p=0.25` branch is now fully checked and still does not beat the
original mixed-replay winner.

### 2026-05-26 08:10 CST: Hard-region-class mixed replay beats the old winner

I stopped treating the remaining `39/50` failures as an unstructured replay
pool and instead measured where they clustered in the fixed hard-room gate.

From the `50`-episode trace:

- source trace:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-33-12/trace.jsonl`
- remaining failures were not random:
  - `goal_x in [75, 99]` cases were only `4/10`
  - `start_y in [0, 24]` cases were only `3/8`
  - exact failure cases alone still overfit and dragged solved cases down

Based on that, I ran a new continuation that replays a **region-class mixed**
set instead of the raw failure list. The preset episodes were the union of:

- cases with `goal_x >= 75`
- cases with `start_y <= 24`

That union was:

- `1,2,3,5,6,8,13,15,18,19,26,28,35,36,39`

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt`

Config:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_failuremix_goalstop045/05_26_05-24-32_80it_1e-5_ent1e-3_p05_opt/best_mean_reward.pt`
- `preset_case_trace = logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-33-12/trace.jsonl`
- `preset_case_episodes = 1,2,3,5,6,8,13,15,18,19,26,28,35,36,39`
- `preset_case_prob = 0.5`
- remaining knobs unchanged:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

Training signal:

- `iter 970`
  - `rew_reach_pos_target_tight = 4.0799`
  - `goal_hold_success = 0.2292`
- `iter 1010`
  - `rew_reach_pos_target_tight = 2.6702`
  - `goal_hold_success = 0.1319`
- `iter 1020`
  - `rew_reach_pos_target_tight = 5.4108`
  - `goal_hold_success = 0.2778`

Fresh evals:

- `best_mean_reward.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-13-37`
  - observed result:
    - `19/30`
- `best_goal_hold.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-19-50`
  - observed result:
    - `20/30`
- `best_reach.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-25-42`
  - observed result:
    - `24/30`
    - `collision_free_success_count = 24`
    - `stand_failures = 0`
    - `fall_failures = 0`
    - `max_step_failures = 6`

This is the first continuation in this phase that **actually beats** the
current winner on the `30`-episode gate:

- old best:
  - `23/30`
- new best:
  - `24/30`

I then ran the stronger `50`-episode gate on that same `best_reach.pt`:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-31-55`
- observed result:
  - `40/50`
  - `collision_free_success_count = 40`
  - `fall_failures = 1`
  - `max_step_failures = 9`

So the current strongest verified checkpoint is now:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`

And its strongest verified gates are:

- `24/30`
- `40/50`

This is still not enough to call SEA-Nav reproduced near the paper result, but
it is a real improvement over the previous `23/30` and `39/50` winner.

Quick read on the new `40/50` failure tail:

- source trace:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-31-55/trace.jsonl`
- total failures:
  - `10`
- failure reasons:
  - `9 max_steps`
  - `1 fall`
- failures with positive reach reward:
  - `1`
- still notable:
  - `5/10` failures have `goal_x >= 75`

So the hard-region-class replay is directionally correct, but the remaining tail
is still dominated by zero-reach hard cases rather than hold-conversion alone.

### 2026-05-26 08:46 CST: Two tail-focused follow-up continuations both regressed

I then treated the new `40/50` winner as the source checkpoint and tested two
follow-up continuations that were supposed to target the remaining `10`-episode
tail without going back to exact failure-only replay.

The `40/50` tail from:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-31-55/trace.jsonl`

has these properties:

- total failures:
  - `10`
- reasons:
  - `9 max_steps`
  - `1 fall`
- positive reach among failures:
  - only `1`
- direction clustering:
  - `SE`: `4`
  - `E`: `3`
  - `NW`: `2`
  - `SW`: `1`
- `goal_x >= 75` among failures:
  - `5/10`

That led to two continuation hypotheses.

#### Follow-up A: direction-class replay (`E/SE/NW`)

Candidate class from the new `50`-episode trace:

- `2,4,5,6,7,9,11,12,14,15,16,17,18,19,20,23,24,26,28,32,38,40,41,42,43,44,47,48`

This was intentionally broad enough to include solved anchor cases:

- total cases:
  - `28`
- within-class trace breakdown:
  - `19` successes
  - `9` failures

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_harddirmix_goalstop045/05_26_08-46-17_80it_1e-5_ent1e-3_p05_opt`

Config:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- `preset_case_trace = logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-31-55/trace.jsonl`
- `preset_case_episodes = 2,4,5,6,7,9,11,12,14,15,16,17,18,19,20,23,24,26,28,32,38,40,41,42,43,44,47,48`
- `preset_case_prob = 0.5`
- remaining knobs unchanged:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

Training-period signals looked alive enough:

- `iter 1010`
  - `rew_reach_pos_target_tight = 3.6020`
  - `goal_hold_success = 0.1875`
- `iter 1020`
  - `rew_reach_pos_target_tight = 4.2480`
  - `goal_hold_success = 0.2396`

But the fresh gate failed hard:

- `best_reach.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-49-00`
  - observed result:
    - `16/30`
    - `stand_failures = 3`
    - `fall_failures = 1`
    - `max_step_failures = 10`

Conclusion:

- `E/SE/NW` is too broad.
- It drags down already-solved cases and is clearly worse than the current
  `24/30` winner.

#### Follow-up B: narrower hard-region replay (`goal_x >= 75 or dir in {E,SE}`)

Because the previous branch was too broad, I narrowed the class to:

- `1,2,5,7,12,15,16,17,19,20,24,28,38,41,42,44,47,48`

This covers `7/10` of the remaining failures while still keeping successful
anchor cases:

- total cases:
  - `18`
- within-class trace breakdown:
  - `11` successes
  - `7` failures

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardgoalx_dirESE_goalstop045/05_26_08-56-22_80it_1e-5_ent1e-3_p05_opt`

Config:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- `preset_case_trace = logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-31-55/trace.jsonl`
- `preset_case_episodes = 1,2,5,7,12,15,16,17,19,20,24,28,38,41,42,44,47,48`
- `preset_case_prob = 0.5`
- remaining knobs unchanged:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

Training-period signal looked better than Follow-up A:

- `iter 1040`
  - `rew_reach_pos_target_tight = 3.7134`
  - `goal_hold_success = 0.2083`
- `iter 1050`
  - `rew_reach_pos_target_tight = 5.1272`
  - `goal_hold_success = 0.2812`

But fresh eval still regressed:

- `best_reach.pt`
  - eval log:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-59-09`
  - observed result:
    - `16/30`
    - `stand_failures = 1`
    - `fall_failures = 0`
    - `max_step_failures = 13`

Conclusion:

- this narrower region class is less destructive than the `E/SE/NW` branch in
  training signal, but still loses badly on the authoritative fresh gate
- it does **not** beat the current `24/30` winner

### Current exact status after these two follow-ups

The strongest verified checkpoint is **still**:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`

The strongest verified gates remain:

- `24/30`
- `40/50`

So this turn did not produce a new winner. What it did establish is:

- the remaining `40/50` tail is real and structured
- direction-heavy replay can easily become too broad
- even the narrower `goal_x + E/SE` class still regresses
- the next move should not be “more obvious class subsets”; it needs a more
  structural change than the two replay classes tested here

### Tail-case controller diagnostics: simple geometry still fails, even without stand termination

To test whether the remaining `10/50` tail was just a “policy not planning well
enough” problem, I added one small extension to
`training/isaac_lab/manual_reward_probe.py`: `hard_room_eval` can now run
scripted controllers, as long as I explicitly provide the exact
`fixed_start_cell` and `fixed_goal_cell` for a sampled hard-room case.

That let me replay representative failures from the current `40/50` winner under
multiple deterministic controllers on the exact same room instances.

Representative remaining failures chosen from:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-31-55/trace.jsonl`
- episodes:
  - `2`
  - `7`
  - `20`
  - `43`
  - `47`

These cover the hardest surviving classes, including:

- long `SE` transitions
- a long `E` corridor-style case
- a difficult `NW` crossing
- one nearly-solved `E` case that still stopped at `min_distance = 0.627`

Runs:

- `logs/isaac_lab/hard_room_controller_diag/05_26_09-11-55`
  - `safe_vector`
  - `path_follow`
- `logs/isaac_lab/hard_room_controller_diag/05_26_09-14-29`
  - `pivot_then_safe_vector`
- `logs/isaac_lab/hard_room_controller_diag/05_26_09-15-57`
  - `pivot_then_path_follow`
- `logs/isaac_lab/hard_room_controller_diag/05_26_09-20-00`
  - `reverse_holonomic_path_follow_no_stand`

Observed result:

- `safe_vector`: `0/5`
- `path_follow`: `0/5`
- `pivot_then_safe_vector`: `0/5`
- `pivot_then_path_follow`: `0/5`

The key follow-up was disabling the usual “stand” termination to check whether
these scripted controllers were simply being killed too early.

For `reverse_holonomic_path_follow_no_stand`, I used:

- `stay_steps = 100000`
- `max_steps = 4000`
- `disable_contact_termination = true`
- `policy_stop_radius = 0.45`

Observed result:

- still `0/5`
- all `5/5` timed out
- all `5/5` had `mean_reach_reward_sum = 0.0`
- `stand_failures = 0`

Interpretation:

- the remaining tail is **not** explained by stand termination being too
  aggressive
- it is also **not** something that simple local vector control or even global
  path-follow can solve automatically on the exact same hard-room instances

So by this point the remaining question was no longer “does the policy need a
slightly better replay subset”, but “is there a deeper execution asymmetry that
these hard cases are exposing?”

### Open-space low-level tracking probe: the bottleneck is direction-dependent mixed-maneuver asymmetry

To answer that, I extended `training/isaac_lab/low_level_tracking_probe.py`
with a few missing command profiles:

- `reverse`
- `reverse_turn`
- `reverse_turn_neg`
- `lateral_turn`
- `lateral_turn_neg`

Then I ran fresh open-space probes on the current target stack:

- `robot_asset_source = native_go2`
- `actuator_mode = ideal_pd`

Logs:

- `logs/isaac_lab/low_level_tracking_probe/05_26_09-29-16`
- `logs/isaac_lab/low_level_tracking_probe/05_26_09-30-09`

The important split is:

Stable or acceptable profiles:

- `forward`
  - target `vx = 1.0`
  - actual mean `vx = 0.7817`
- `lateral`
  - target `vy = 0.5`
  - actual mean `vy = 0.4522`
- `yaw`
  - target `wz = 0.8`
  - actual mean `wz = 0.7541`
- `forward_turn`
  - target `[0.35, 0.0, 1.0]`
  - actual mean `[0.3288, 0.0032, 0.9854]`
- `reverse`
  - target `vx = -0.5`
  - actual mean `vx = -0.4479`

So pure forward, pure lateral, pure yaw, forward-turn, and even pure reverse
are not the real blocker anymore.

What breaks is the **negative-yaw mixed maneuvers**:

- `reverse_turn`
  - target `[-0.3, 0.0, +0.8]`
  - actual mean `[-0.2610, +0.1646, +0.7840]`
  - already shows large unintended lateral drift
- `reverse_turn_neg`
  - target `[-0.3, 0.0, -0.8]`
  - episode terminated at step `90`
  - actual mean `[-0.5449, +0.3692, +0.1542]`
  - this is not just inaccurate; the turn direction effectively collapses
- `lateral_turn`
  - target `[0.0, +0.35, +0.8]`
  - actual mean `[+0.0258, +0.2891, +0.7926]`
  - still usable
- `lateral_turn_neg`
  - target `[0.0, -0.35, -0.8]`
  - actual mean `[-0.5430, +0.2924, -0.1154]`
  - again, this is a severe sign/coupling failure, not a small tracking error

Interpretation:

- the remaining hard-room failure tail is **not** mainly a replay-curriculum
  problem anymore
- it lines up with a **direction-dependent execution asymmetry**
- positive-yaw mixed maneuvers are roughly trackable
- negative-yaw mixed maneuvers can collapse into the wrong lateral/forward
  response entirely

This is the first strong evidence that the current ceiling (`24/30`, `40/50`)
is probably being set by the low-level execution envelope, not by the high-level
replay recipe.

### Current exact status after the controller and low-level diagnostics

The strongest verified checkpoint is still:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`

The strongest verified gates are still:

- `24/30`
- `40/50`

But the working diagnosis is now much sharper:

- replay tuning alone is no longer the main lever
- stand termination is not the main reason the remaining tail fails
- the current limiting factor appears to be low-level mixed-maneuver asymmetry,
  especially on negative-yaw combinations

### Method correction: the first low-level probe was history-contaminated

After writing the section above, I caught a methodology bug in
`training/isaac_lab/low_level_tracking_probe.py`.

The first version ran multiple command profiles back-to-back inside the same env
instance without a fresh `env.reset()` before each profile. Since the low-level
controller carries observation history, this contaminated the symmetry check.

I fixed the probe so that **every profile now starts from a fresh reset** before
applying the manual start/goal pose.

Then I reran the full suite here:

- `logs/isaac_lab/low_level_tracking_probe/05_26_09-35-15`

This supersedes the earlier interpretation based on:

- `05_26_09-29-16`
- `05_26_09-30-09`

### Corrected low-level tracking diagnosis after fresh reset per profile

The corrected split is more specific than “all negative mixed maneuvers are
broken”.

Profiles that are still reasonably trackable from fresh reset:

- `forward`
  - target `[1.0, 0.0, 0.0]`
  - actual mean `[0.7901, -0.0194, 0.0196]`
- `lateral`
  - target `[0.0, 0.5, 0.0]`
  - actual mean `[0.0503, 0.4466, -0.0131]`
- `forward_turn`
  - target `[0.35, 0.0, 1.0]`
  - actual mean `[0.3098, -0.0041, 0.9882]`
- `forward_turn_neg`
  - target `[0.35, 0.0, -1.0]`
  - actual mean `[0.3009, -0.0346, -0.9815]`
- `reverse`
  - target `[-0.5, 0.0, 0.0]`
  - actual mean `[-0.4485, -0.0040, -0.0105]`
- `reverse_turn`
  - target `[-0.3, 0.0, 0.8]`
  - actual mean `[-0.2389, 0.0056, 0.7901]`
- `reverse_turn_neg`
  - target `[-0.3, 0.0, -0.8]`
  - actual mean `[-0.2966, 0.0562, -0.8006]`
- `lateral_turn`
  - target `[0.0, 0.35, 0.8]`
  - actual mean `[0.0229, 0.2877, 0.7896]`

So reverse-turn itself is **not** the dominant failure anymore once the probe is
run correctly.

What still breaks badly from fresh reset is:

- `yaw`
  - target `[0.0, 0.0, 0.8]`
  - actual mean `[-0.4952, 0.5590, 1.3274]`
- `yaw_neg`
  - target `[0.0, 0.0, -0.8]`
  - actual mean `[-0.4641, 0.3058, -0.0400]`

In other words, **turn-in-place from standstill is severely unstable**, and the
negative direction is worse.

There is also still one mixed maneuver with clear transient/mean-level trouble:

- `lateral_turn_neg`
  - target `[0.0, -0.35, -0.8]`
  - actual mean `[-0.5007, 0.2523, -0.3946]`
  - even though the final sample drifted closer to the target, the measurement
    window still shows very large unwanted forward/backward and lateral coupling

### Revised interpretation after the corrected probe

So the diagnosis is now:

- replay tuning is still no longer the main lever
- stand termination is still not the main reason the remaining tail fails
- but the sharper low-level bottleneck is **not** “all negative mixed maneuvers”
- the clearer bottleneck is:
  - poor turn-in-place behavior from standstill in both yaw directions
  - especially poor negative standstill turn
  - plus unstable leftward negative lateral-turn transients

This fits the remaining hard-room tail better than the earlier, broader claim:
those tail cases look increasingly like they require a reliable reorientation
primitive near standstill, and that is exactly where the corrected fresh-reset
probe is weakest.

### Final probe correction: noise had to be disabled as well

There was one more probe methodology issue to remove before trusting the
turning-threshold conclusion:

- even after switching to fresh reset per profile, the probe was still inheriting
  training-time observation noise

I fixed that in `training/isaac_lab/low_level_tracking_probe.py` by forcing:

- `env_cfg.add_noise = False`

Then I reran the focused turning sweep here:

- `logs/isaac_lab/low_level_tracking_probe/05_26_09-39-37`

Profiles tested:

- `yaw`
- `yaw_neg`
- `crawl_turn`
- `crawl_turn_neg`
- `slow_turn`
- `slow_turn_neg`
- `forward_turn`
- `forward_turn_neg`

### Noise-free turning threshold result

This is the cleanest version of the low-level diagnosis so far.

Pure turn-in-place from standstill is badly broken in both directions:

- `yaw`
  - target `[0.0, 0.0, +0.8]`
  - actual mean `[-0.4978, +0.5401, +1.2850]`
- `yaw_neg`
  - target `[0.0, 0.0, -0.8]`
  - actual mean `[-0.4976, +0.4322, +0.5502]`

So the low-level controller does **not** provide a usable in-place turning
primitive from standstill.

Positive-yaw reorientation becomes stable with only a small forward bias:

- `crawl_turn`
  - target `[+0.1, 0.0, +0.8]`
  - actual mean `[+0.1440, -0.0204, +0.7610]`
- `slow_turn`
  - target `[+0.2, 0.0, +0.8]`
  - actual mean `[+0.2219, -0.0090, +0.7730]`
- `forward_turn`
  - target `[+0.35, 0.0, +1.0]`
  - actual mean `[+0.3254, +0.0013, +0.9867]`

So for the positive direction, the threshold is low: even `vx = 0.1` is enough
to convert “bad standstill yaw” into a usable turning primitive.

Negative-yaw reorientation is the real asymmetry:

- `crawl_turn_neg`
  - target `[+0.1, 0.0, -0.8]`
  - actual mean `[-0.5522, +0.5260, +0.7535]`
- `slow_turn_neg`
  - target `[+0.2, 0.0, -0.8]`
  - actual mean `[-0.4570, +0.3263, +0.1491]`

These are not small tracking errors; they still collapse into the wrong motion
mode entirely.

Even at the larger command:

- `forward_turn_neg`
  - target `[+0.35, 0.0, -1.0]`
  - actual mean `[+0.1788, +0.0543, -1.0064]`
  - final sample eventually got close to the target, but the measurement window
    still shows a long transient with suppressed forward speed

### Final interpretation after the deterministic sweep

The current best explanation for the remaining `40/50` ceiling is now:

- replay tuning is not the main lever anymore
- stand termination is not the main reason the tail fails
- the decisive execution gap is:
  - no stable turn-in-place primitive from standstill
  - very strong left/right asymmetry in low-speed yaw reorientation
  - positive reorientation can be rescued by a tiny forward bias
  - negative reorientation remains badly broken until much larger forward motion,
    and even then it has a long transient

That is a much more actionable blocker than the earlier “generic mixed-maneuver
asymmetry” diagnosis, and it matches why the remaining hard-room failures keep
ending as zero-reach or max-step timeouts even after replay tuning and scripted
path-follow attempts.

### Eval-only asymmetric turn prior did not beat the current 40/50 winner

Once the noise-free sweep showed:

- positive `+yaw` can be stabilized by a tiny forward bias
- negative `-yaw` remains broken at low speed

the obvious next question was whether a **simple eval-only turn prior** would
already lift the current best hard-room checkpoint.

I added three probe-only knobs to `training/isaac_lab/manual_reward_probe.py`:

- `--policy-turn-yaw-threshold`
- `--policy-turn-forward-floor-pos`
- `--policy-turn-forward-floor-neg`

These only affect `manual_reward_probe.py` evaluation; they do **not** touch the
training path.

I then re-ran the authoritative `50`-episode hard-room gate on the current
winner:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- baseline gate:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-31-55`
  - `40/50`
- asymmetric turn-prior gate:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_09-43-05`
  - config:
    - `policy_turn_yaw_threshold = 0.6`
    - `policy_turn_forward_floor_pos = 0.1`
    - `policy_turn_forward_floor_neg = 0.35`

Observed result:

- `39/50`
- `stand_failures = 3`
- `fall_failures = 1`
- `max_step_failures = 7`

So this prior did **not** improve the winner. It was slightly worse than the
baseline `40/50`.

Interpretation:

- the low-level asymmetry diagnosis is still useful
- but a naive “force minimum forward speed during turning” patch is **not**
  sufficient by itself
- especially for the negative-yaw side, the failure mode is deeper than just
  “needs a little forward motion”

Current exact status after this follow-up:

- best verified hard-room gate is still:
  - `24/30`
  - `40/50`
- the current winner is still:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- the new evidence says:
  - replay tuning alone is no longer the bottleneck
  - low-level negative reorientation is still the real weakness
  - but the fix is not as simple as adding a hard forward floor during turning

### Failure-trace command probe: representative failing commands are mostly executable

The asymmetric low-speed turn diagnosis was strong, but it still left one open
question:

- are the remaining hard-room failures happening because the high-level policy is
  repeatedly asking for commands that the low-level cannot realize at all?

To answer that, I extended `training/isaac_lab/low_level_tracking_probe.py` to
accept arbitrary `name:vx,vy,wz` commands, then replayed representative commands
lifted directly from the traced hard-room failures.

Run:

- `logs/isaac_lab/low_level_tracking_probe/05_26_09-58-25`

Representative commands:

- from the far-field `episode 2` failure:
  - `fail_e2_early = [1.218, +0.341, -0.315]`
  - `fail_e2_mid = [0.751, -0.292, -0.746]`
- from the near-goal `episode 47`-style failure:
  - `fail_e47_near = [1.52, -0.25, -0.77]`
- plus mirrored variants for comparison

Observed results:

- `fail_e2_early`
  - actual mean `[0.9923, 0.2400, -0.3164]`
- `fail_e2_mid`
  - actual mean `[0.5898, -0.2882, -0.7133]`
- `fail_e47_near`
  - actual mean `[1.1423, -0.2647, -0.7557]`

Their mirrored counterparts were also mostly trackable at the same qualitative
level.

Interpretation:

- these representative failure-trace commands are **not** in the same category as
  the broken standstill-turn primitives
- the low-level can realize them reasonably well in open space
- so the remaining `40/50` ceiling cannot be explained purely as “the policy is
  commanding impossible motions”

### Two different surviving failure modes are now visible

I traced two concrete hard-room failures from the current `40/50` winner:

- far-field failure:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_09-54-54`
  - fixed case:
    - `start = [16, 62]`
    - `goal = [87, 24]`
  - result:
    - `max_step_failure`
    - `reach_reward_sum = 0`
    - `min_distance = 3.3175`
- near-goal failure:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_09-55-49`
  - fixed case:
    - `start = [15, 51]`
    - `goal = [49, 39]`
  - result:
    - `max_step_failure`
    - `reach_reward_sum = 542.78`
    - `min_distance = 0.3301`

What these traces show:

- the far-field failure is **not** dominated by low-speed negative yaw; early on,
  it uses large forward and lateral commands such as:
  - mean first-50 command `[1.218, 0.341, -0.315]`
  - which the open-space probe shows is executable
- the near-goal failure is a different class:
  - it does enter the reach zone
  - the stop prior eventually zeros the command
  - but in this traced short run it still does not finish as `goal_hold`

So the surviving tail is now clearly multi-modal:

- one class is early/path-level failure where the policy drifts into a bad
  trajectory even though the commanded motions are feasible
- another class is late-stage near-goal conversion / hold behavior

### Updated working diagnosis

The remaining `40/50` gap is now best explained as a combination of:

- a real low-level weakness in standstill and low-speed negative reorientation
- plus a separate high-level path-selection problem on some hard cases

That means the project is no longer in a state where a single “more replay” or a
single “simple turn prior” is likely to close the gap.

### Eval-only A* path blend also regressed the current winner

Because the far-field failure class now looked more like “early path choice
drift” than pure low-level infeasibility, I tested one more eval-only prior:

- blend the current policy with an A*-derived `path_follow` command while the
  robot is still far from the goal

This was added only to `training/isaac_lab/manual_reward_probe.py` through:

- `--policy-path-blend-weight`
- `--policy-path-blend-min-distance`

I did **not** put this into the training path.

Run:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-02-20`

Config:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- `episodes = 30`
- `policy_stop_radius = 0.45`
- `policy_stop_mode = zero`
- `policy_path_blend_weight = 0.7`
- `policy_path_blend_min_distance = 1.0`
- `path_lookahead = 12`

Observed result:

- `20/30`
- `stand_failures = 6`
- `max_step_failures = 4`

This is clearly worse than the baseline authoritative gate for the same winner:

- baseline:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-25-42`
  - `24/30`

Interpretation:

- the remaining far-field failures are **not** fixed by a simple global
  path-follow blend
- that means the surviving gap is now even less likely to be closed by generic
  priors layered on top of the current policy
- both of the obvious eval-only priors tested in this phase:
  - asymmetric turn floor
  - A* path-follow blend
  regressed relative to the current best winner

So the next useful step is no longer “try another generic blend weight”, but to
work on the two failure classes separately:

- early path-selection failure
- late near-goal conversion / hold failure

### Eval noise check: current hard-room gate was already aligned with original play semantics here

I explicitly checked whether the current hard-room eval might be artificially low
simply because it was still running with observation noise on.

Original Isaac Gym reference:

- `training/legged_gym/legged_gym/scripts/play.py`
- explicitly sets:
  - `env_cfg.noise.add_noise = False`

Current Isaac Lab eval path:

- `training/isaac_lab/manual_reward_probe.py`
- already sets:
  - `env_cfg.add_noise = False`

So the current verified gates such as:

- `24/30`
- `40/50`

are **not** pessimistic just because the eval path forgot to disable noise.

That rules out one more easy explanation for the remaining gap.

### 2026-05-26 10:19 CST: Training-side reward rebalance branch was wired correctly but regressed on fresh hard-room eval

Based on the current failure picture, the next justified lever was not another
eval-only prior, but a **training-side reward rebalance** targeting far-field
path-selection drift.

I first parameterized the Isaac Lab reward scales instead of leaving them
hardcoded:

- `training/isaac_lab/sea_nav_env.py`
  - added:
    - `reward_scale_termination`
    - `reward_scale_collision`
    - `reward_scale_close_obst_vel`
    - `reward_scale_stuck`
    - `reward_scale_velo_dir`
    - `reward_scale_reach_pos_target_tight`
  - `_get_rewards()` now uses `self.cfg.reward_scale_*`
- `training/isaac_lab/train.py`
  - added matching CLI flags:
    - `--reward-scale-termination`
    - `--reward-scale-collision`
    - `--reward-scale-close-obst-vel`
    - `--reward-scale-stuck`
    - `--reward-scale-velo-dir`
    - `--reward-scale-reach-pos-target-tight`
  - these are passed through `make_sea_nav_env_cfg(...)`
  - `runner.learn(..., config=...)` now reports the active reward scales from
    CLI instead of a fixed dict

Static validation passed:

- `python3 -m py_compile training/isaac_lab/sea_nav_env.py training/isaac_lab/train.py`

I also re-verified the GPU situation before launching the branch:

- inside the sandbox:
  - `nvidia-smi` still fails
  - `/dev/nvidia*` is absent
- host-side / escalated:
  - `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`
  - reported:
    - `NVIDIA GeForce RTX 5060 Ti, 580.159.03`

So fresh Isaac Sim runs are still forced to use the host-side escalated path;
the blocker is no longer "host GPU is down", but "sandbox cannot see the GPU".

#### Reward-rebalanced continuation

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045_rewardrebalance/05_26_10-19-45_80it_1e-5_ent1e-3_p05_c7_v2_opt`

Source checkpoint:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`

Command changes relative to the current `24/30`, `40/50` winner:

- kept:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - same hard-region mixed replay source
  - `preset_case_prob = 0.5`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`
- changed only:
  - `reward_scale_close_obst_vel: 5.0 -> 7.0`
  - `reward_scale_velo_dir: 4.0 -> 2.0`

Training-period signal looked alive:

- `iter 1020`
  - `rew_collision = -94.2480`
  - `rew_close_obst_vel = 24.9569`
  - `rew_velo_dir = 11.2716`
  - `rew_reach_pos_target_tight = 16.3764`
- `iter 1030`
  - `rew_reach_pos_target_tight = 18.8175`
  - `goal_hold_success = 1.0000`
- `iter 1050`
  - `rew_reach_pos_target_tight = 5.6206`
  - `goal_hold_success = 0.3125`

But fresh hard-room eval did **not** support the training-time optimism.

`best_reach.pt`:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-20-48`
- observed result:
  - `16/30`
  - `collision_free_success_count = 16`
  - `stand_failures = 5`
  - `fall_failures = 1`
  - `max_step_failures = 8`

`best_mean_reward.pt`:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-27-30`
- observed result:
  - `18/30`
  - `collision_free_success_count = 18`
  - `stand_failures = 4`
  - `fall_failures = 1`
  - `max_step_failures = 7`

Interpretation:

- the reward rebalance branch was **correctly wired** and **truly trained on the
  host GPU**
- but this particular reweighting:
  - stronger `close_obst_vel`
  - weaker `velo_dir`
  clearly regressed against the current authoritative winner
- current best remains unchanged:
  - `24/30`
  - `40/50`
  - checkpoint:
    - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`

So this direction is now checked and should **not** be continued as-is.

### 2026-05-26 10:36 CST: Training-side turn prior continuation also regressed

After closing out the reward-rebalance branch, I moved to the more structural
direction suggested by the low-level probes: **train with the same asymmetric
turn prior that was previously tested only in eval**.

The exact prior already existed in `training/isaac_lab/manual_reward_probe.py`:

- `policy_turn_yaw_threshold = 0.6`
- `policy_turn_forward_floor_pos = 0.1`
- `policy_turn_forward_floor_neg = 0.35`

I then ported that logic into the training path:

- `training/isaac_lab/sea_nav_env.py`
  - added config fields:
    - `turn_yaw_threshold`
    - `turn_forward_floor_pos`
    - `turn_forward_floor_neg`
  - `_pre_physics_step()` now applies the same forward-floor prior to
    `nav_actions_orig` before goal-stop shaping and low-level rollout
- `training/isaac_lab/train.py`
  - added matching CLI flags:
    - `--turn-yaw-threshold`
    - `--turn-forward-floor-pos`
    - `--turn-forward-floor-neg`
  - passes them through `make_sea_nav_env_cfg(...)`

Static validation again passed:

- `python3 -m py_compile training/isaac_lab/sea_nav_env.py training/isaac_lab/train.py`

#### Turn-prior continuation

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045_turnprior/05_26_10-36-43_80it_1e-5_ent1e-3_p05_tp06_fp01_fn035_opt`

Source checkpoint:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`

Config relative to the current winner:

- kept:
  - same hard-region mixed replay source
  - `preset_case_prob = 0.5`
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`
- added only:
  - `turn_yaw_threshold = 0.6`
  - `turn_forward_floor_pos = 0.1`
  - `turn_forward_floor_neg = 0.35`

Training-period signal was unstable and mostly weak:

- `iter 990`
  - `mean_reward = -179.69`
  - `rew_reach_pos_target_tight = 0.0`
- `iter 1010`
  - `rew_reach_pos_target_tight = 17.7879`
  - `goal_hold_success = 1.0000`
- `iter 1050`
  - `mean_reward = -323.42`
  - `rew_reach_pos_target_tight = 0.0`
  - `goal_hold_success = 0.0000`

Fresh hard-room eval for `best_reach.pt`:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-38-43`
- observed result:
  - `16/30`
  - `collision_free_success_count = 16`
  - `stand_failures = 1`
  - `fall_failures = 0`
  - `max_step_failures = 13`

Interpretation:

- moving the asymmetric turn prior from eval-only into training did **not**
  improve the current winner
- the branch regressed just like the reward-rebalance branch
- current strongest verified checkpoint still remains:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
  - with authoritative gates:
    - `24/30`
    - `40/50`

So the simple "bake the turn prior into training" hypothesis is now also
checked and rejected in this form.

### 2026-05-26 10:51 CST: Hard-room traces now save the exact room map, and exact-room failure replay is partially effective but still regresses overall

At this point, I found a concrete reproducibility gap in the tooling itself:

- `manual_reward_probe.py --scenario hard_room_eval` was saving:
  - `trace.jsonl`
  - `summary.json`
- but it was **not** saving the actual generated hard-room map

That meant all earlier `preset_case_trace` continuations were only replaying:

- `start_cell`
- `goal_cell`
- `start_yaw`

while still rebuilding a **different** hard-room at training time. So earlier
"failure replay does not help" conclusions were directionally useful, but not
yet exact-room experiments.

I fixed that by adding:

- `training/isaac_lab/manual_reward_probe.py`
  - every summary write can now also save `room.npy`
  - hard-room eval summaries now include:
    - `room_path`
- `training/isaac_lab/train.py`
  - when `--preset-case-trace` points to a trace whose directory contains
    `room.npy`, training now automatically loads that room and uses it as
    `preset_room`
- `training/isaac_lab/manual_reward_probe.py`
  - added:
    - `--preset-room-npy`
  - this allows same-room evaluation on an explicitly saved room

Static validation passed:

- `python3 -m py_compile training/isaac_lab/manual_reward_probe.py training/isaac_lab/train.py`

#### Rebuilt the current `24/30` gate as an exact-room reproducible asset

I reran the current strongest winner:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-51-04`

Observed result:

- `24/30`
- `collision_free_success_count = 24`
- `max_step_failures = 6`
- and now this run also saved:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-51-04/room.npy`

The six exact same-room failures were:

- `episodes = 2,7,12,15,20,26`

#### Exact-room failure replay continuation

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_exactroom_failmix_goalstop045/05_26_10-58-00_80it_1e-5_ent1e-3_p05_opt`

Config:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- exact-room replay source:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-51-04/trace.jsonl`
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-51-04/room.npy`
- replay episodes:
  - `2,7,12,15,20,26`
- `preset_case_prob = 0.5`
- all other continuation knobs unchanged:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

Training-period signal was at least alive:

- `iter 1040`
  - `rew_reach_pos_target_tight = 6.6121`
  - `goal_hold_success = 0.3750`

Fresh eval on the **same exact room**:

- checkpoint:
  - `best_reach.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-58-50`
- observed result:
  - `22/30`
  - `collision_free_success_count = 22`
  - `fall_failures = 1`
  - `max_step_failures = 7`

For completeness, `best_goal_hold.pt` produced the same same-room gate:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_11-05-30`
- observed result:
  - `22/30`

So exact-room replay **still did not beat** the same-room baseline `24/30`.

#### But exact-room replay revealed a more precise failure mode than before

Because both runs now used the same saved room and the same seed, their
`30`-episode traces were directly comparable episode-by-episode.

Comparing:

- baseline trace:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-51-04/trace.jsonl`
- exact-room replay trace:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-58-50/trace.jsonl`

showed this:

- the continuation **fixed 5 of the original 6 failures**:
  - `7`
  - `12`
  - `15`
  - `20`
  - `26`
- but it also **broke 7 previously solved cases**:
  - `3`
  - `4`
  - `6`
  - `18`
  - `25`
  - `27`
  - `28`

Interpretation:

- the older failure-replay regressions were **not** only caused by replaying the
  wrong room
- exact-room replay is genuinely more informative:
  - it can fix the intended hard tail
  - but it still introduces strong forgetting of already solved cases
- so the remaining problem is now sharper:
  - not "failure replay doesn't work"
  - but "failure replay fixes the tail by over-specializing and erasing nearby
    solved behavior"

This is a better target for the next branch than anything attempted so far,
because it finally separates:

- room mismatch
- from actual forgetting / policy drift

### 2026-05-26 11:13 CST: Exact-room anti-forgetting replay also regressed

I immediately followed the exact-room failure replay with a more balanced same-room continuation.

Instead of replaying only the six exact-room failures, I replayed:

- the `6` original failures:
  - `2,7,12,15,20,26`
- plus the `7` originally-solved episodes that the failure-only continuation
  later broke:
  - `3,4,6,18,25,27,28`

This gave a `13`-case same-room anchor set intended to reduce forgetting while
still pushing on the true tail.

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_exactroom_balancemix_goalstop045/05_26_11-13-41_80it_1e-5_ent1e-3_p05_opt`

Config:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- exact room:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-51-04/room.npy`
- preset trace:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_10-51-04/trace.jsonl`
- replay episodes:
  - `2,3,4,6,7,12,15,18,20,25,26,27,28`
- `preset_case_prob = 0.5`
- all other knobs unchanged from the same-room branches:
  - `goal_stop_radius = 0.45`
  - `goal_stop_mode = zero`
  - `learning_rate = 1e-5`
  - `schedule = fixed`
  - `entropy_coef = 0.001`
  - `resume_load_optimizer = True`

Training-period signal again looked alive enough to justify a fresh eval:

- `iter 1030`
  - `rew_reach_pos_target_tight = 7.8030`
  - `goal_hold_success = 0.4375`
- `iter 1050`
  - `rew_reach_pos_target_tight = 2.4628`
  - `goal_hold_success = 0.1250`

So I evaluated the branch on the **same exact room** using its
`best_goal_hold.pt`.

Eval:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_11-14-31`
- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_exactroom_balancemix_goalstop045/05_26_11-13-41_80it_1e-5_ent1e-3_p05_opt/best_goal_hold.pt`
- observed result:
  - `19/30`
  - `collision_free_success_count = 19`
  - `stand_failures = 2`
  - `fall_failures = 1`
  - `max_step_failures = 8`

Comparison across the three same-room lines is now:

- baseline current winner on exact room:
  - `24/30`
- exact-room failure replay:
  - `22/30`
- exact-room balance replay:
  - `19/30`

Interpretation:

- simply mixing the broken-success anchor set back into replay does **not**
  solve the forgetting problem
- the next step is no longer "add more cases to replay"
- the remaining requirement is likely some form of more selective preservation:
  - lower replay probability
  - different resume point
  - or explicit regularization / anchoring against the source winner

## 2026-05-26 actuator drift continuation: add a torque-faithful Isaac Lab mode

I stopped treating the remaining actuator gap as a documentation-only drift and
implemented a new repo-side actuator mode:

- `actuator_mode = gym_torque`

What changed:

- `training/isaac_lab/sea_nav_env.py`
  - adds `cfg.actuator_mode`, joint PD constants, and torque-limit buffers
  - adds a new `gym_torque` articulation branch
  - keeps `implicit` and `ideal_pd` untouched for A/B
  - computes low-level torques with the original Gym navigation law:
    - `tau = kp * (q_target - q) - kd * qdot`
    - `q_target = default_q + 0.25 * action`
  - recomputes torque inside `_apply_action()` so it is refreshed on every
    Isaac Lab decimation substep, instead of only once per RL step
  - writes torques through `set_joint_effort_target(...)`
- `training/isaac_lab/train.py`
  - exposes `--actuator-mode gym_torque`
- `training/isaac_lab/manual_reward_probe.py`
  - exposes `--actuator-mode gym_torque`
- `training/isaac_lab/low_level_tracking_probe.py`
  - exposes `--actuator-mode gym_torque`

This is intentionally narrower than a full control rewrite. The goal is to
close the most suspicious migration drift while preserving the existing
`ideal_pd` path for controlled comparisons.

### Runtime validation

Static validation:

- `python3 -m py_compile training/isaac_lab/train.py training/isaac_lab/sea_nav_env.py training/isaac_lab/manual_reward_probe.py training/isaac_lab/low_level_tracking_probe.py`

Result:

- passed

GPU runtime status during this check:

- `nvidia-smi`
- observed result:
  - `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`
- `lsmod | grep '^nvidia'`
- observed result:
  - `nvidia`, `nvidia_uvm`, `nvidia_modeset`, `nvidia_drm` are still loaded
- `ls -l /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm`
- observed result:
  - all three device nodes are missing
- recovery attempts:
  - `nvidia-modprobe -u -c=0`
  - `nvidia-modprobe -c 0 -u -m`
- observed result:
  - neither recreated the missing device nodes
  - `nvidia-smi` still failed afterward

So I did not treat GPU-launch failures as evidence against the new actuator
mode. They are machine runtime failures, not SEA-Nav control-chain failures.

CPU smoke command:

- `env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py --headless --num-envs 1 --smoke-steps 2 --actuator-mode gym_torque --robot-asset-source converted_urdf --sim-device cpu --rl-device cpu`

Observed result:

- environment setup completed
- low-level joint mapping still resolves correctly
- new torque path prints:
  - `joint_action_scale=0.25`
  - `joint_stiffness=30.0`
  - `joint_damping=0.75`
  - `torque_limit_min=80.000`
  - `torque_limit_max=80.000`
- reset succeeded with:
  - `obs_shape=(1, 550)`
- smoke rollout succeeded:
  - `step=1 reward_mean=2.1839 terminated=0 truncated=0`
  - `step=2 reward_mean=1.9572 terminated=0 truncated=0`

One additional portability fix was needed to make this CPU smoke meaningful:

- low-level JIT models are now loaded with explicit `map_location=self.device`
  so CPU smoke no longer fails just because the checkpoints were saved from a
  CUDA runtime

### Interpretation

This does **not** prove the actuator gap is now fully closed.

What it does prove:

- the repo now contains a torque-faithful actuator path much closer to the
  original Isaac Gym semantics
- the path is not just syntactically wired; it reaches real environment reset
  and rollout
- the current blocker for fresh GPU verification is again the host NVIDIA
  runtime, not an immediate control-chain crash inside `gym_torque`

So the next redeploy step is no longer "design the torque path".

The next real gate is:

- restore host GPU runtime
- then run the same smoke / tracking / hard-room eval on `gym_torque`
- and compare it directly against the current `ideal_pd` winner

## 2026-05-26 host GPU A/B: `ideal_pd` vs `gym_torque`

After correcting the earlier mistake of trusting sandbox GPU visibility, I reran
the new actuator path on the real host GPU.

### 1. Host GPU smoke for `gym_torque`

Command:

- `env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py --headless --num-envs 1 --smoke-steps 2 --actuator-mode gym_torque --robot-asset-source converted_urdf --sim-device cuda:0 --rl-device cuda:0`

Observed result:

- host GPU is visible:
  - `Driver Version: 580.159.03`
  - `GPU 0: NVIDIA GeForce RTX 5060 Ti`
- environment device is really `cuda:0`
- smoke passes:
  - `reset complete obs_shape=(1, 550) action_dim=3 device=cuda:0`
  - `step=1 reward_mean=0.0252 terminated=0 truncated=0`
  - `step=2 reward_mean=0.0251 terminated=0 truncated=0`

So the new torque-faithful path is not only a CPU wiring check. It runs on the
real target GPU.

### 2. Host GPU low-level tracking probe A/B

I then compared `ideal_pd` and `gym_torque` on the same negative-turn profiles
that were previously most suspicious:

- `yaw`
- `yaw_neg`
- `crawl_turn`
- `crawl_turn_neg`
- `slow_turn`
- `slow_turn_neg`
- `forward_turn`
- `forward_turn_neg`

Commands:

- `env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/low_level_tracking_probe.py --headless --robot-asset-source converted_urdf --sim-device cuda:0 --actuator-mode ideal_pd --warmup-steps 60 --measure-steps 120 --profiles yaw,yaw_neg,crawl_turn,crawl_turn_neg,slow_turn,slow_turn_neg,forward_turn,forward_turn_neg`
- `env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/low_level_tracking_probe.py --headless --robot-asset-source converted_urdf --sim-device cuda:0 --actuator-mode gym_torque --warmup-steps 60 --measure-steps 120 --profiles yaw,yaw_neg,crawl_turn,crawl_turn_neg,slow_turn,slow_turn_neg,forward_turn,forward_turn_neg`

Outputs:

- `ideal_pd`:
  - `logs/isaac_lab/low_level_tracking_probe/05_26_12-34-48/summary.json`
- `gym_torque`:
  - `logs/isaac_lab/low_level_tracking_probe/05_26_12-35-21/summary.json`

Observed result:

- across all probed profiles, the summary statistics are effectively identical
- the previously bad negative low-speed turn cases remain bad in exactly the
  same way
- representative examples:
  - `yaw_neg`: `actual_mean.wz = 0.3759` for both modes
  - `crawl_turn_neg`: `actual_mean.vx = -0.3785`, `actual_mean.wz = -0.4421` for both modes
  - `slow_turn_neg`: `actual_mean.vx = -0.2846`, `actual_mean.wz = -0.6508` for both modes
  - `forward_turn_neg`: `actual_mean.vx = 0.1142`, `actual_mean.wz = -1.0198` for both modes

Interpretation:

- on this converted-URDF open-space probe, `gym_torque` is behaviorally the
  same as the existing `ideal_pd` path
- the negative-turn weakness is real, but it is **not** being fixed merely by
  switching to direct effort writes

### 3. Host GPU hard-room gate A/B on the strongest checkpoint

I then evaluated the same strongest checkpoint that previously gave the current
best `24/30` hard-room gate, but switched only the actuator mode:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`

Reference baseline:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-25-42/summary.json`
- baseline result:
  - `24/30`
  - `collision_free_success_count = 24`
  - `mean_reach_reward_sum = 986.3669534365337`
  - `mean_min_distance = 0.7202126403649648`

New command:

- `env OMNI_KIT_ACCEPT_EULA=YES /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/manual_reward_probe.py --headless --scenario hard_room_eval --controller-mode policy --checkpoint /home/user/rl_redeploy/SEA-Nav-Code/logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt --episodes 30 --max-steps 2000 --stay-steps 500 --disable-contact-termination --policy-stop-radius 0.45 --policy-stop-mode zero --actuator-mode gym_torque --robot-asset-source native_go2 --sim-device cuda:0`

Output:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_12-36-22/summary.json`

Observed result:

- exactly the same gate outcome:
  - `24/30`
  - `collision_free_success_count = 24`
  - `mean_reach_reward_sum = 986.3669534365337`
  - `mean_min_distance = 0.7202126403649648`

### Updated interpretation

This is the strongest actuator-gap result so far:

- the repo now contains a more torque-faithful actuator path
- that path is verified on the host GPU
- but the current strongest policy, under the current hard-room gate, behaves
  identically under `ideal_pd` and `gym_torque`

So actuator semantics still count as a formal paper-to-code drift, but they are
no longer the leading verified explanation for the current `24/30`, `40/50`
ceiling.

The remaining main blocker stays where the behavioral evidence was already
pointing:

- low-level negative low-speed reorientation weakness
- plus a smaller set of high-level hard-case path-choice failures
- plus replay/fine-tune forgetting when pushing on the tail

## 2026-05-26 domain-randomization gap closure and host-GPU continuation

After actuator A/B had been downgraded as the main explanation, I moved to the
next migration-level gap that still existed in the repo:

- Gym-side friction randomization
- Gym-side base-mass perturbation
- the missing `ang_vel_xy` reward term

### Code changes

I ported these pieces into the Isaac Lab env:

- `training/isaac_lab/sea_nav_env.py`
  - added `reward_scale_ang_vel_xy = -0.05`
  - added repo-side `randomize_friction`, `friction_range`
  - added repo-side `randomize_base_mass`, `added_mass_range`
  - added `_apply_initial_domain_randomization()`:
    - per-env friction bucket sampling
    - per-env base-mass perturbation with inertia rescaling
  - added `_reward_ang_vel_xy()` back into reward composition
- `training/isaac_lab/train.py`
  - added CLI knob `--reward-scale-ang-vel-xy`
- `training/isaac_lab/manual_reward_probe.py`
  - eval now explicitly disables friction randomization and uses zero added mass
    to stay closer to original `play.py` semantics
- `training/isaac_lab/low_level_tracking_probe.py`
  - probe also disables friction randomization and uses zero added mass for
    deterministic controller diagnostics

Static validation passed:

```bash
python3 -m py_compile \
  training/isaac_lab/sea_nav_env.py \
  training/isaac_lab/train.py \
  training/isaac_lab/manual_reward_probe.py \
  training/isaac_lab/low_level_tracking_probe.py
```

### Important simulator-level finding: Gym friction lower bound is invalid in PhysX

The original Gym config uses:

- `friction_range = [-0.2, 1.25]`

When I first tried to continue training with this range ported literally, Isaac
Sim / PhysX raised repeated material creation errors:

- `dynamicFriction must be >= 0.`
- `material pointer 0 is NULL!`

So this is a real simulator-level drift, not a training hyperparameter issue:

- the original Gym config allows a negative lower bound
- Isaac Sim / PhysX does not

I fixed this by clamping the randomized friction lower bound to PhysX-safe
non-negative values before material creation:

- effective Isaac Lab rule now becomes:
  - `friction_lo = max(0.0, friction_lo)`

### Host GPU smoke after the PhysX-safe clamp

I verified the patched env on the real host GPU:

```bash
env OMNI_KIT_ACCEPT_EULA=YES \
/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless --num-envs 1 --smoke-steps 1 \
  --actuator-mode ideal_pd \
  --robot-asset-source native_go2 \
  --sim-device cuda:0 --rl-device cuda:0
```

Observed result:

- host GPU came up normally
- friction randomization was now valid:
  - `friction_min=0.248`
  - `friction_max=0.248`
- smoke passed:
  - `reset complete obs_shape=(1, 550) action_dim=3 device=cuda:0`
  - `step=1 reward_mean=0.0069 terminated=0 truncated=0`

### Short continuation from the current 24/30 winner

I then tested whether closing this missing training-side gap would actually push
the current strongest hard-room winner above `24/30`.

Source winner:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- reference gate:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-25-42/summary.json`
- reference result:
  - `24/30`

Continuation run:

- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045_domrandfix/05_26_12-56-08_80it_1e-5_ent1e-3_p05_opt`
- command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES \
/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --robot-asset-source native_go2 \
  --actuator-mode ideal_pd \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --num-envs 256 \
  --max-iterations 80 \
  --num-steps-per-env 48 \
  --learning-rate 1e-5 \
  --lr-schedule fixed \
  --entropy-coef 0.001 \
  --goal-stop-radius 0.45 \
  --goal-stop-mode zero \
  --preset-case-trace logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-33-12/trace.jsonl \
  --preset-case-episodes 1,2,3,5,6,8,13,15,18,19,26,28,35,36,39 \
  --preset-case-prob 0.5 \
  --resume-from logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt \
  --resume-load-optimizer \
  --experiment-name Go2_pos_rough_isaaclab_hardclassmix_goalstop045_domrandfix \
  --run-name 80it_1e-5_ent1e-3_p05_opt
```

Checkpoint metadata:

- `best_mean_reward.pt` at `iter = 1008`
- `best_goal_hold.pt` at `iter = 1057`
- `best_reach.pt` at `iter = 1057`

Training signal was not dead:

- `iter 1000`
  - `rew_reach_pos_target_tight = 0.3739`
  - `goal_hold_success = 0.0208`
- `iter 1020`
  - `rew_reach_pos_target_tight = 2.2824`
  - `goal_hold_success = 0.1146`

### Fresh hard-room evals

`best_reach.pt`:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_12-58-59/summary.json`
- observed result:
  - `16/30`
  - `collision_free_success_count = 16`
  - `stand_failures = 1`
  - `fall_failures = 1`
  - `max_step_failures = 12`

`best_mean_reward.pt`:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-06-06/summary.json`
- observed result:
  - `23/30`
  - `collision_free_success_count = 23`
  - `fall_failures = 1`
  - `max_step_failures = 6`

### Updated interpretation

This closes a previously real repo-side gap, but it does **not** improve the
current best hard-room winner:

- old best:
  - `24/30`
- new domain-randomization continuation:
  - `best_reach.pt = 16/30`
  - `best_mean_reward.pt = 23/30`

So the new evidence is:

- missing domain randomization is no longer an open repo-side omission
- the original Gym friction lower bound needs PhysX-safe clamping in Isaac Sim
- but after closing that gap and restoring `ang_vel_xy`, the current hard-room
  ceiling still does not move upward

This means the remaining blocker is still better explained by:

- low-level negative low-speed reorientation weakness
- some hard-case high-level path selection failures
- and forgetting when pushing on the tail

not by “the repo is still missing Gym-style friction / mass randomization”.

## 2026-05-26 source-level CBF geometry drift: align shield FOV with the 240-degree ray fan

After actuator A/B and repo-side domain-randomization repair had both failed to
push the ceiling above `24/30`, I closed one more high-value source-level
mismatch that already existed in the released Isaac Gym code:

- the environment ray fan is `240` degrees
- but the CBF layer internally precomputed ray geometry over `180` degrees

This was visible in:

- Gym / env side:
  - `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py`
  - `training/isaac_lab/sea_nav_env.py`
- shield side:
  - `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py`

### Repo-side repair

I updated the shared shield implementation so that Isaac Lab now uses the
correct `240`-degree geometry directly:

- `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py`
  - default `fov_deg` changed from `180.0` to `240.0`
  - the layer now stores `num_rays` / `fov_deg`
  - `ray_unit_vectors` are rebuilt from config instead of trusting checkpointed
    legacy buffer values
  - older checkpoints still load because `_load_from_state_dict(...)` ignores the
    stale saved buffer and refreshes geometry after load

I also added train-time isolation switches so I could test this geometry repair
without confounding it with the friction / base-mass continuation that had just
been introduced:

- `training/isaac_lab/train.py`
  - `--disable-friction-rand`
  - `--disable-base-mass-rand`

### Load-compatibility check

I verified that the old strongest checkpoint still loads under the repaired
shield geometry:

```bash
env /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('training/rsl_rl').resolve()))
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
import torch
model = DifferentiableSafeActorCritic(
    num_actions=3,
    actor_hidden_dims=[512,256,128],
    critic_hidden_dims=[512,256,128],
    init_noise_std=1.5,
    num_props=12,
    num_rays=41,
    his_len=10,
)
ckpt = torch.load(
    'logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt',
    map_location='cpu',
)
model.load_state_dict(ckpt['model_state_dict'])
print('loaded ok', model.cbf_layer.fov_deg, model.cbf_layer.ray_unit_vectors[0].tolist(), model.cbf_layer.ray_unit_vectors[-1].tolist())
PY
```

Observed output:

- `loaded ok 240.0 [-0.5000, -0.8660] [-0.5000, 0.8660]`

So the repair is real, and old checkpoints are no longer silently evaluated
through a mismatched `180`-degree shield geometry.

### Direct host-GPU re-eval of the current strongest winner

I first re-evaluated the existing `24/30` winner under the corrected shield
geometry, without retraining:

- source checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/summary.json`

Observed result:

- `24/30`
- `collision_free_success_count = 24`
- `mean_reach_reward_sum = 959.71`
- `mean_min_distance = 0.5868`

Compared with the original authoritative gate:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-25-42/summary.json`
- `24/30`
- `mean_reach_reward_sum = 986.37`
- `mean_min_distance = 0.7202`

So the corrected geometry did **not** raise the success count directly, but it
did improve the average closest approach distance of the old winner.

### Isolated continuation with only the CBF geometry repair

I then launched an isolated continuation from the same winner family, while
explicitly turning off the newly added friction / base-mass randomization and
setting `reward_scale_ang_vel_xy = 0.0` so this run would not be confounded by
the previous domain-randomization repair:

- run:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045_cbffov240_iso/05_26_13-25-45_80it_1e-5_ent1e-3_p05_opt`

Command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES \
/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --robot-asset-source native_go2 \
  --actuator-mode ideal_pd \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --num-envs 256 \
  --max-iterations 80 \
  --num-steps-per-env 48 \
  --learning-rate 1e-5 \
  --lr-schedule fixed \
  --entropy-coef 0.001 \
  --goal-stop-radius 0.45 \
  --goal-stop-mode zero \
  --reward-scale-ang-vel-xy 0.0 \
  --disable-friction-rand \
  --disable-base-mass-rand \
  --preset-case-trace logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_06-33-12/trace.jsonl \
  --preset-case-episodes 1,2,3,5,6,8,13,15,18,19,26,28,35,36,39 \
  --preset-case-prob 0.5 \
  --resume-from logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt \
  --resume-load-optimizer \
  --experiment-name Go2_pos_rough_isaaclab_hardclassmix_goalstop045_cbffov240_iso \
  --run-name 80it_1e-5_ent1e-3_p05_opt
```

Checkpoint metadata:

- `best_reach.pt` at `iter = 1028`
- `best_goal_hold.pt` at `iter = 1028`
- `best_mean_reward.pt` at `iter = 1046`
- `model_1059.pt` at `iter = 1059`

Training signal was live:

- `iter 1000`
  - `rew_reach_pos_target_tight = 2.0500`
  - `goal_hold_success = 0.1146`
- `iter 1030`
  - `rew_reach_pos_target_tight = 6.0271`
  - `goal_hold_success = 0.2500`

### Fresh hard-room gate for the isolated continuation

`best_reach.pt`:

- eval log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-28-36/summary.json`
- observed result:
  - `18/30`
  - `collision_free_success_count = 18`
  - `stand_failures = 3`
  - `max_step_failures = 9`
  - `mean_reach_reward_sum = 729.64`
  - `mean_min_distance = 1.3747`

### Updated interpretation

This makes the new status of the shield-geometry drift much clearer:

- the `240`-vs-`180` mismatch was real
- it has now been repaired in the local Isaac Lab fork
- old checkpoints now load under the corrected geometry
- direct re-eval of the old winner still stays at `24/30`
- and an isolated continuation from the same winner family regresses to `18/30`

So this source-level mismatch still matters as a paper-to-open-source caveat,
but it is **no longer a strong candidate for the current `24/30`, `40/50`
ceiling**. The remaining blocker is still better explained by:

- low-level negative low-speed reorientation weakness
- some hard-case high-level path-selection failures
- and replay / continuation forgetting

## 2026-05-26 low-level follow-up: boosted negative-yaw commands do not fix the asymmetry

Because the remaining hard-room tail still looked strongly correlated with
negative low-speed reorientation, I tested one more narrow low-level hypothesis
before touching the policy again:

- maybe the problem is simply that negative yaw commands are too small
- if so, boosting only negative-yaw commands should make the low-level tracker
  more symmetric

I tested this directly in open space on the host GPU:

- log:
  - `logs/isaac_lab/low_level_tracking_probe/05_26_13-44-22/summary.json`

Command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES \
/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/low_level_tracking_probe.py \
  --headless \
  --robot-asset-source native_go2 \
  --sim-device cuda:0 \
  --actuator-mode ideal_pd \
  --settle-steps 10 \
  --warmup-steps 60 \
  --measure-steps 120 \
  --profiles yaw,yaw_neg,crawl_turn,crawl_turn_neg,slow_turn,slow_turn_neg,forward_turn,forward_turn_neg \
  --custom-command yaw_neg_boost:0.0,0.0,-1.2 \
  --custom-command crawl_turn_neg_boost:0.1,0.0,-1.2 \
  --custom-command slow_turn_neg_boost:0.2,0.0,-1.2 \
  --custom-command forward_turn_neg_boost:0.35,0.0,-1.2
```

Key comparisons:

- baseline positive turn commands are healthy:
  - `crawl_turn -> actual_mean wz = 0.759`
  - `slow_turn -> actual_mean wz = 0.757`
  - `forward_turn -> actual_mean wz = 0.985`
- baseline negative turns are asymmetric:
  - `crawl_turn_neg -> actual_mean wz = 0.424`
  - `slow_turn_neg -> actual_mean wz = -0.527`
  - `forward_turn_neg -> actual_mean wz = -0.967`
- boosted negative yaw does **not** repair the weak low-speed cases:
  - `yaw_neg_boost -> actual_mean wz = -0.034`
  - `crawl_turn_neg_boost -> actual_mean wz = 0.684`
  - `slow_turn_neg_boost -> actual_mean wz = -0.121`
  - `forward_turn_neg_boost -> actual_mean wz = -0.963`

Interpretation:

- the failure is **not** a simple “negative yaw magnitude is too small” issue
- boosting negative yaw can make the low-speed cases even less stable
- the only negative-turn profile that remains healthy is the already
  high-forward-speed one (`forward_turn_neg`)

So the more accurate local conclusion now is:

- negative reorientation needs enough forward motion to stay in a good regime
- but a simple negative-yaw gain is the wrong fix

This sharply lowers the value of trying more “just make negative yaw bigger”
priors. If the next branch stays at the command level, it should be about
carefully controlled forward support during reorientation, not yaw amplification.

## 2026-05-26 exact-room tail diagnosis: the remaining failures split into two verified classes

To move from “likely blocker” to “verified blocker”, I used the current
exact-room asset produced by the corrected-geometry re-eval:

- gate:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/summary.json`
- room asset:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/room.npy`

That gate is still `24/30`, and its six failures are all:

- `done_reason = max_steps`
- `reach_reward_sum = 0.0`

So under the current strongest policy family, the remaining tail is already not
“entered goal region but failed hold”; it is “never truly reached”.

### Reproducible far-field failure

I replayed one of those failures exactly on the saved room:

- start:
  - `start_cell = [86, 24]`
  - `goal_cell = [36, 87]`
  - `start_yaw = -0.9471807621136622`
- trace log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-58-41`

Observed result:

- `max_steps`
- `reach_reward_sum = 0.0`
- `min_distance = 5.0116`

The step trace shows this is not a near-goal failure at all:

- minimum distance happened at the very end, still `5.01m` away
- mean command over the whole rollout:
  - `[vx, vy, wz] = [0.755, -0.186, -0.469]`
- last `200` steps:
  - mean command `[1.218, 0.120, -0.087]`
  - mean distance `6.488`
- there were `410` low-forward negative-turn steps globally, but `0` of them
  happened inside `2m` of the goal

Interpretation:

- this class is a **far-field path/trajectory drift** failure
- the policy never gets close enough for near-goal hold or stop logic to matter
- after adding trace-driven replay support to `manual_reward_probe.py`, I
  re-ran the same case via `--case-trace ... --case-episode 25`; that replay
  ended as `stand` instead of `max_steps`, but it still stayed in the **same
  far-field zero-reach failure class** (`min_distance ~= 5.74`, `reach_reward_sum = 0.0`)

### Reproducible near-goal zero-reach failure

I then replayed a different exact-room failure that was much closer to the goal:

- start:
  - `start_cell = [88, 64]`
  - `goal_cell = [32, 21]`
  - `start_yaw = 2.6622012074273256`
- trace log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_14-02-58`

Observed result:

- `max_steps`
- `reach_reward_sum = 0.0`
- `min_distance = 0.9148`

At the closest approach:

- step `1469`
- command:
  - `[-0.057, -0.605, -0.633]`
- local goal:
  - `[-0.068, 0.912]`
- front clearance:
  - `0.22`

Whole-episode statistics:

- mean command:
  - `[1.103, -0.015, 0.171]`
- last `200` steps:
  - mean command `[1.012, 0.008, 0.337]`
  - mean distance `5.390`
- low-forward negative-turn steps:
  - `57` total
  - `41` of them happened within `2m` of the goal

Interpretation:

- this class is **not** a hold-conversion failure
- it is a near-goal **reorientation / conversion failure before first reach**
- and it lands exactly in the regime already exposed by the low-level probe:
  - low forward support
  - negative yaw
  - tight clearance

### Updated blocker conclusion

This is the strongest behavior-level evidence so far:

1. Some tail cases are far-field path-selection / trajectory-drift failures.
2. Some tail cases are near-goal conversion failures that enter the same
   low-forward negative-yaw regime where the low-level tracker is already known
   to be weak.
3. The failed boosted-negative-yaw probe shows this is **not** fixable by simply
   increasing negative yaw magnitude.

So the remaining blocker is now better described as a **two-part behavioral
ceiling**:

- high-level path/trajectory drift on a small set of hard cases
- low-level weakness during near-goal negative reorientation under tight
  clearance and insufficient forward support

## 2026-05-26 batch replay of all remaining exact-room failures

With `case-trace` replay in place, I re-ran all six failures from the
`24/30` exact-room gate:

- source gate:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/summary.json`
- source trace:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/trace.jsonl`

Batch replay results:

- `episode 5`
  - replayed as `goal_hold`
  - `min_distance = 0.3276`
  - `reach_reward_sum = 1188.39`
- `episode 11`
  - `max_steps`
  - `min_distance = 1.3176`
  - class: near-goal zero-reach
- `episode 17`
  - `max_steps`
  - `min_distance = 2.3841`
  - class: far-field drift
- `episode 25`
  - `stand`
  - `min_distance = 5.7399`
  - class: far-field drift
- `episode 27`
  - `max_steps`
  - `min_distance = 0.9852`
  - class: near-goal zero-reach
- `episode 29`
  - `max_steps`
  - `min_distance = 0.9148`
  - class: near-goal zero-reach

This sharpens the interpretation of the current winner's tail:

- the original `24/30` gate is not made of six equally stable geometric failure
  cases
- one of the six (`episode 5`) is a rollout-level unstable case: under exact
  replay it consistently turns into a success
- the remaining **five** failures split into:
  - `3` near-goal zero-reach cases
  - `2` far-field drift cases

So the current post-migration ceiling is not just “some failures remain”. It is
more specifically:

- a small but real rollout-level instability band
- plus two stable behavioral failure classes

## 2026-05-26 exact-room replay ergonomics: add trace-driven case replay to manual_reward_probe

The exact-room diagnosis work had already exposed one practical problem:

- manually copying `start_cell`, `goal_cell`, and especially `start_yaw` out of
  `trace.jsonl` is error-prone
- using a rounded yaw can turn an apparent failure into a false success

To make the blocker diagnosis workflow trustworthy, I added a direct replay path
to `training/isaac_lab/manual_reward_probe.py`:

- new CLI:
  - `--case-trace <trace.jsonl>`
  - `--case-episode <episode_idx>`
- behavior:
  - loads exact `start_cell`, `goal_cell`, and full-precision `start_yaw`
    directly from the trace row
  - automatically loads sibling `room.npy` if `--preset-room-npy` is not passed

Static check:

```bash
python3 -m py_compile training/isaac_lab/manual_reward_probe.py
```

Host-GPU smoke:

```bash
env OMNI_KIT_ACCEPT_EULA=YES \
/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/manual_reward_probe.py \
  --headless \
  --scenario hard_room_eval \
  --controller-mode policy \
  --checkpoint logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt \
  --episodes 1 \
  --max-steps 2000 \
  --stay-steps 500 \
  --disable-contact-termination \
  --policy-stop-radius 0.45 \
  --policy-stop-mode zero \
  --actuator-mode ideal_pd \
  --robot-asset-source native_go2 \
  --sim-device cuda:0 \
  --case-trace logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/trace.jsonl \
  --case-episode 25 \
  --trace-steps
```

Observed result:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_14-07-12`
- parsed values were correct:
  - `fixed_start_cell = [86, 24]`
  - `fixed_goal_cell = [36, 87]`
  - `fixed_start_yaw = -0.9471807621136622`
- replay again landed in the same far-field zero-reach class:
  - `reach_reward_sum = 0.0`
  - `mean_min_distance = 5.7399`

This does not directly improve policy quality, but it materially improves the
quality of future blocker diagnosis because exact-room case replay no longer
depends on hand-transcribing floats.

## 2026-05-26 case-trace A/B: simple forward-support prior does not rescue the remaining tail

After the exact-room tail had been split into:

- far-field path/trajectory drift
- near-goal zero-reach reorientation failure

I tested one more direct causal hypothesis:

- maybe the near-goal class can be fixed by a simple command-level prior that
  enforces some minimum forward motion whenever the policy asks for negative yaw

I used the existing eval-only command prior:

- `--policy-turn-yaw-threshold 0.1`
- `--policy-turn-forward-floor-neg 0.35`

and applied it only to exact-room `case-trace` replays of the two reproduced
failure cases.

### Near-goal zero-reach case with forward-support prior

Case:

- `case-trace = logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/trace.jsonl`
- `case-episode = 29`

Baseline exact-room replay:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_14-02-58`
- `max_steps`
- `reach_reward_sum = 0.0`
- `min_distance = 0.9148`

Forward-support replay:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_14-09-56`
- `max_steps`
- `reach_reward_sum = 0.0`
- `min_distance = 0.9567`

Interpretation:

- this simple prior does **not** rescue the near-goal zero-reach class
- it does not even improve closest approach on this case

### Far-field drift case with forward-support prior

Case:

- `case-trace = logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/trace.jsonl`
- `case-episode = 25`

Baseline exact-room replay through the new case-trace interface:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_14-07-12`
- `stand`
- `reach_reward_sum = 0.0`
- `min_distance = 5.7399`

Forward-support replay:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_14-09-59`
- `max_steps`
- `reach_reward_sum = 0.0`
- `min_distance = 2.3024`

Interpretation:

- the same simple prior does alter the far-field trajectory materially
- but it still does **not** produce first reach
- so it is not a practical fix for the far-field failure class either

### Updated blocker conclusion

This pushes the diagnosis one step further:

- simple negative-yaw amplification already failed in the open-space probe
- simple negative-turn forward support now also fails on the reproduced
  near-goal zero-reach case
- and it still cannot convert the far-field drift case into a true reach

So the remaining ceiling is **not** explained by one more small command-level
prior. The current best explanation remains:

- high-level tail path / trajectory drift on some hard cases
- plus a lower-level near-goal negative reorientation weakness that is not fixed
  by trivial command shaping

## 2026-05-26 exact-room far-field-only continuation: no net improvement

After the exact-room `24/30` winner had been decomposed into:

- `1` rollout-level unstable case
- `3` stable near-goal zero-reach cases
- `2` stable far-field path / trajectory-drift cases

I ran one more targeted continuation that replayed only the two stable
far-field cases, at low probability, to test whether that subgroup alone could
move the exact-room ceiling upward without disturbing the rest of the policy too
much.

Training run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_exactroom_farfieldmix_goalstop045/05_26_14-21-00_80it_1e-5_ent1e-3_p025_opt`

Replay source:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/trace.jsonl`
- replayed episodes:
  - `17`
  - `25`
- `preset-case-prob = 0.25`

Resume checkpoint:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`

Online training statistics did briefly look promising:

- around `iter=1030`:
  - `Mean reward = 118.63`
  - `rew_reach_pos_target_tight = 7.1950`
  - `goal_hold_success = 0.3958`

But the authoritative same-room fresh gate still regressed relative to the
current exact-room baseline:

Baseline:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19`
- result:
  - `24/30`
  - `mean_min_distance = 0.5868`

New branch snapshots:

- `best_reach.pt`
  - eval:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_14-24-50`
  - result:
    - `17/30`
    - `mean_min_distance = 1.1593`
- `best_mean_reward.pt`
  - eval:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_14-32-32`
  - result:
    - `21/30`
    - `mean_min_distance = 0.8510`
- `best_goal_hold.pt`
  - eval:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_14-39-41`
  - result:
    - `22/30`
    - `mean_min_distance = 0.8429`

Conclusion:

- replaying only the two stable far-field drift cases does **not** improve the
  exact-room ceiling
- even the best snapshot of this branch stays below the existing `24/30`
  baseline
- therefore the remaining blocker is not “just fix the far-field subgroup with
  a small targeted continuation”
- this reinforces the current diagnosis:
  - a small rollout-level instability band
  - a stable far-field path / trajectory-drift subgroup
  - and a stable near-goal negative reorientation subgroup

## 2026-05-26 exact-room failure replay with policy anchoring: lightweight preservation also fails

After the targeted far-field-only continuation failed, I tested the next
structural hypothesis directly:

- maybe the replay branches are failing mainly because they forget solved cases
- and maybe a **small behavior-preservation anchor** against the current winner
  is enough to stop that forgetting

### Minimal preservation mechanism added to training

I added the smallest training-side preservation hook I could justify:

- new train CLI:
  - `--policy-anchor-coef`
  - `--policy-anchor-from`
- PPO now supports a frozen reference actor-critic
- during update, actor `mu_batch` is lightly regularized toward the reference
  policy's shielded action mean on the same observations

This is intentionally minimal:

- actor-side only
- no environment changes
- no reward changes
- no new replay semantics

### Experiment: exact-room failure replay + anchor

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_exactroom_failmix_anchor_goalstop045/05_26_14-53-14_80it_1e-5_ent1e-3_p05_anchor005`

Config:

- replay source:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/trace.jsonl`
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/room.npy`
- replayed episodes:
  - `5,11,17,25,27,29`
- `preset-case-prob = 0.5`
- resume checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
- policy anchor:
  - `policy_anchor_coef = 0.05`
  - `policy_anchor_from = resume checkpoint`

The training loss itself stayed numerically healthy:

- `policy_anchor_loss` stayed small:
  - `0.0006` at `iter=990`
  - `0.0018` at `iter=1000`
  - `0.0040` at `iter=1020`
  - `0.0069` at `iter=1050`

So this was not a trivial “anchor completely froze the policy” failure.

### Same-room fresh gate results

Baseline:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19`
- result:
  - `24/30`

Anchored branch snapshots:

- `best_goal_hold.pt`
  - eval:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_14-56-12`
  - result:
    - `14/30`
    - `stand_failures = 3`
    - `fall_failures = 3`
    - `max_step_failures = 10`
- `best_reach.pt`
  - eval:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_15-03-12`
  - result:
    - `14/30`
    - effectively the same gate as `best_goal_hold.pt`
- `best_mean_reward.pt`
  - eval:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_15-11-04`
  - result:
    - `18/30`
    - `stand_failures = 1`
    - `fall_failures = 1`
    - `max_step_failures = 10`

### Conclusion

This is a strong negative result:

- a lightweight behavior anchor does **not** preserve the current `24/30`
  exact-room winner under failure replay
- it actually regresses all tested fresh gates
- and because `policy_anchor_loss` stayed small, this is not well explained by
  “the anchor was just too strong and froze everything”

So the blocker statement becomes stricter again:

- the remaining gap is not solved by one more replay subset
- and it is also not solved by a small actor-side preservation anchor

At this point, any further anti-forgetting line would need to be materially more
structural than a small reference-action regularizer.

## 2026-05-26 exact-room failure replay with frozen policy trunk: stronger preservation still fails

After the lightweight actor-side anchor had already regressed the exact-room
gate, I tested one more structurally stronger but still minimal preservation
idea:

- freeze the actor representation trunk
- keep only the output-side policy heads trainable
- then rerun the exact-room failure replay branch

### Freeze design

New train-side switch:

- `--freeze-policy-trunk`

Implemented meaning:

- freeze:
  - `encoder.*`
  - `backbone.*`
- keep trainable:
  - `nav_head.*`
  - `alpha_head.*`
  - `critic.*`
  - `std`

On the actual run, this left:

- `trainable_params = 226440`
- `total_params = 875928`

So this is materially stronger than the small actor-anchor regularizer, but it
is still a constrained continuation rather than a redesigned algorithm.

### Experiment: exact-room failure replay + frozen policy trunk

Run:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_exactroom_failmix_freezetrunk_goalstop045/05_26_15-23-07_80it_1e-5_ent1e-3_p05_freezetrunk`

Config:

- replay source:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/trace.jsonl`
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19/room.npy`
- replayed episodes:
  - `5,11,17,25,27,29`
- `preset-case-prob = 0.5`
- resume checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`

Training-period metrics were not obviously dead:

- `iter=1010`
  - `goal_hold_success = 0.1562`
  - `rew_reach_pos_target_tight = 2.8680`
- `iter=1030`
  - `goal_hold_success = 0.1667`
  - `rew_reach_pos_target_tight = 5.4323`

So this branch was at least behaviorally active during training.

### Same-room fresh gate results

Baseline:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_13-18-19`
- result:
  - `24/30`

Frozen-trunk branch:

- `best_reach.pt`
  - eval:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_15-25-59`
  - result:
    - `16/30`
    - `stand_failures = 1`
    - `fall_failures = 1`
    - `max_step_failures = 12`
- `best_mean_reward.pt`
  - eval:
    - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_15-34-03`
  - result:
    - `19/30`
    - `fall_failures = 2`
    - `max_step_failures = 9`

`best_goal_hold.pt` was saved at the same training iteration as `best_reach.pt`
(`iter = 1049`), so the branch never produced a same-room winner that exceeded
`19/30`.

### Conclusion

This is another strong negative result:

- freezing the actor trunk does **not** preserve the `24/30` exact-room winner
  under failure replay
- it is less catastrophic than the small actor-anchor branch
- but it still regresses well below baseline

So the blocker statement tightens again:

- the remaining gap is not solved by:
  - replay-subset tuning
  - small command priors
  - small actor-side reference anchoring
  - freezing the actor trunk while fine-tuning policy heads

Any continuation from here would have to be more structural than the current
family of replay-plus-light-preservation experiments.

## 2026-05-26 broad hard-room multi-room audit of the strongest winner

After the exact-room preservation branches all failed, I stopped adding new
training tweaks and instead strengthened the final evaluation evidence on the
current strongest broad winner:

- checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`

The goal of this audit was to answer two questions more rigorously:

1. how stable is the current winner across different hard-room seeds?
2. how much of its success depends on the non-paper `goal_stop_radius = 0.45`
   prior?

### Audit setup

For each seed in `{1, 2, 3}` I ran:

- `hard_room_eval`
- `episodes = 30`
- `stay_steps = 500`
- `disable_contact_termination = true`
- `native_go2 + ideal_pd`

under two modes:

- with stop prior:
  - `policy_stop_radius = 0.45`
  - `policy_stop_mode = zero`
- without stop prior:
  - no `policy_stop_radius`

### With stop prior: large room-to-room variance

Seed 1:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_15-44-23`
- result:
  - `24/30`

Seed 2:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_15-50-29`
- result:
  - `25/30`

Seed 3:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_15-55-50`
- result:
  - `8/30`
  - `stand_failures = 8`
  - `fall_failures = 1`
  - `max_step_failures = 13`

Aggregate with stop prior:

- total success:
  - `57/90`
- success rate:
  - `63.3%`

Interpretation:

- the current strongest winner is **not** stable across hard-room seeds
- even with the non-paper stop prior, one room still collapses from the `24/30`
  to `25/30` range down to `8/30`

### Without stop prior: near-total collapse

Seed 1:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_16-03-46`
- result:
  - `2/30`

Seed 2:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_16-13-21`
- result:
  - `1/30`

Seed 3:

- log:
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_16-23-19`
- result:
  - `1/30`

Aggregate without stop prior:

- total success:
  - `4/90`
- success rate:
  - `4.4%`

Interpretation:

- the current strongest local winner depends very heavily on the non-paper stop
  prior
- once that prior is removed, broad hard-room success nearly disappears

### Final implication of the multi-room audit

This is the strongest high-level evidence gathered in the whole redeploy:

- the best local winner is not room-robust even **with** the stop prior
- and it collapses almost completely **without** the stop prior

So the current Isaac Lab migration on this machine cannot be honestly described
as “close to the paper demonstration effect”.

At this point, the negative conclusion is no longer based only on single-room
tails or failed continuation branches; it is also supported by a multi-room
broad hard-room audit of the strongest winner itself.

## Final determination

Under the current hardware and the current Isaac Lab migration in this repo:

- SEA-Nav has been ported far enough to train, save checkpoints, run GUI and
  headless eval, and produce partially competent policies
- but it has **not** been credibly reproduced near the paper's demonstrated
  effect

The strongest verified local winner is:

- same-room hard-room:
  - `24/30`
  - `40/50`
- multi-room hard-room with the non-paper stop prior:
  - `57/90`
- multi-room hard-room without that prior:
  - `4/90`

So the final answer to the objective is:

- on this machine, in the current Isaac Lab migration, the method does **not**
  reach a paper-near result
- and the failure is no longer best explained by one more missing Gym semantic
  or one more lightweight continuation trick

The strongest verified blocker set is:

- a small rollout-level instability band
- stable far-field path / trajectory-drift failures
- stable near-goal negative reorientation failures
- continuation / replay forgetting that is not fixed by:
  - replay-subset tuning
  - small command priors
  - small actor-side policy anchoring
  - freezing the actor trunk while tuning heads

## Deferred TODO

The faithful Isaac Lab reproduction line is intentionally paused here.

- Do **not** continue the same reward / replay / lightweight continuation tuning
  line by default.
- If a materially stronger low-level controller or locomotion primitive becomes
  available in Isaac Lab, resume from that new control contract rather than from
  another PPO tuning sweep.
- Before any new broad/full training restart, first re-check:
  - turn-in-place stability
  - near-goal negative reorientation
  - stop / hold behavior in tight spaces
- Only restart the reproduction attempt after the low-level contract is
  demonstrably stronger on those gates.

## 2026-05-27 RobotLab Low-Level Controller Restart

The faithful line was reopened after a new RobotLab-trained Go2 flat low-level
controller became available.

Source RobotLab checkpoint:

```text
/home/user/rl_ws/robot_lab/logs/rsl_rl/unitree_go2_flat/2026-05-27_13-38-03_codex_go2_flat_20260527/model_4999.pt
```

Copied runtime artifacts in this repo:

```text
training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/model_4999.pt
training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/policy.pt
training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/agent.yaml
training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/env.yaml
```

Important interface facts:

- RobotLab exported policy input is `45` dims.
- Observation order is:
  - base angular velocity, scaled by `0.25`
  - projected gravity
  - velocity command
  - relative joint position
  - relative joint velocity, scaled by `0.05`
  - previous action
- RobotLab action output is `12` joint position offsets.
- RobotLab joint order is:
  - `FR_hip`, `FR_thigh`, `FR_calf`
  - `FL_hip`, `FL_thigh`, `FL_calf`
  - `RR_hip`, `RR_thigh`, `RR_calf`
  - `RL_hip`, `RL_thigh`, `RL_calf`
- This matches the existing SEA-Nav low-level order after the earlier dynamic
  joint-name mapping fix.

Implementation changes:

- Added `--low-level-controller robotlab`.
- Added `--actuator-mode robotlab_dc`, matching RobotLab's Go2 `DCMotorCfg`
  style:
  - stiffness `25.0`
  - damping `0.5`
  - effort / saturation effort `23.5`
  - velocity limit `30.0`
- Added RobotLab per-joint action scaling:
  - hip joints: `0.125`
  - thigh / calf joints: `0.25`
- The RobotLab low-level command is clipped to its training range `[-1, 1]`
  for `vx`, `vy`, and `wz`.
- No low-level observation noise is added for RobotLab inference, matching
  RobotLab `play.py` behavior.

Host GPU smoke:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --num-envs 1 \
  --smoke-steps 8 \
  --robot-asset-source native_go2 \
  --actuator-mode robotlab_dc \
  --low-level-controller robotlab \
  --disable-friction-rand \
  --disable-base-mass-rand \
  --disable-obs-noise \
  --experiment-name Go2_pos_rough_isaaclab_robotlab_lowlevel_smoke
```

Verified result:

- environment device: `cuda:0`
- live joint order was dynamically mapped into RobotLab low-level order
- `policy.pt` loaded from the copied repo-local path
- `8/8` smoke steps completed without termination

Low-level tracking probe:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/low_level_tracking_probe.py \
  --headless \
  --robot-asset-source native_go2 \
  --actuator-mode robotlab_dc \
  --low-level-controller robotlab \
  --profiles yaw_neg,crawl_turn_neg,slow_turn_neg,forward_turn_neg,forward,strafe_left \
  --sim-device cuda:0 \
  --settle-steps 80 \
  --warmup-steps 80 \
  --measure-steps 240
```

Result directory:

```text
logs/isaac_lab/low_level_tracking_probe/05_27_15-38-26
```

Key tracking results:

- `yaw_neg`: no termination, actual mean `wz=-0.923`, yaw RMSE about `0.174`
- `crawl_turn_neg`: no termination, actual mean `wz=-0.886`, yaw RMSE about `0.144`
- `slow_turn_neg`: no termination, actual mean `wz=-0.854`, yaw RMSE about `0.128`
- `forward_turn_neg`: no termination, actual mean `wz=-1.010`, yaw RMSE about `0.126`

This directly improves the previous low-level blocker around negative low-speed
reorientation.

Short PPO check:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --headless \
  --robot-asset-source native_go2 \
  --actuator-mode robotlab_dc \
  --low-level-controller robotlab \
  --disable-friction-rand \
  --disable-base-mass-rand \
  --disable-obs-noise \
  --num-envs 32 \
  --num-steps-per-env 16 \
  --max-iterations 21 \
  --experiment-name Go2_pos_rough_isaaclab_robotlab_lowlevel_shortcheck
```

Verified result:

- run directory:
  `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_lowlevel_shortcheck/05_27_15-39-24`
- `Iteration: 20`
- value loss about `5.4e4`
- smooth loss about `1.34`
- mean reward about `-811.87`
- collision reward about `-16.57`
- no reach yet

Current long run:

```bash
systemd-run --user --unit sea-nav-robotlab-lowlevel-train --collect \
  --working-directory=/home/user/rl_redeploy/SEA-Nav-Code \
  --setenv=OMNI_KIT_ACCEPT_EULA=YES \
  --setenv=PYTHONDONTWRITEBYTECODE=1 \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
    --headless \
    --robot-asset-source native_go2 \
    --actuator-mode robotlab_dc \
    --low-level-controller robotlab \
    --disable-friction-rand \
    --disable-base-mass-rand \
    --disable-obs-noise \
    --num-envs 256 \
    --num-steps-per-env 48 \
    --max-iterations 2000 \
    --save-interval 100 \
    --experiment-name Go2_pos_rough_isaaclab_robotlab_lowlevel_full
```

Service:

```text
sea-nav-robotlab-lowlevel-train.service
```

Run directory:

```text
logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_lowlevel_full/05_27_15-40-11
```

First logged training point:

- `Iteration: 20`
- value loss about `6.4e4`
- smooth loss about `0.601`
- mean reward about `-629.42`
- collision reward about `-7.16`
- no reach yet

Latest early check before handoff:

- service still active at `Iteration: 60`
- `Iteration: 40` had `rew_reach_pos_target_tight = 0.2037`
- `Iteration: 60` had `rew_reach_pos_target_tight = 0.4143`
- `goal_hold_success` is still `0.0`

Interpretation:

- the new RobotLab low-level controller is wired into SEA-Nav correctly enough
  to run smoke, tracking, short PPO, and a long managed training job
- the earlier negative-yaw low-level failure is materially improved
- the remaining question is now empirical: whether broad/full high-level
  training can exploit this stronger low-level contract

### RobotLab low-level full run completed, but high-level navigation did not recover

The RobotLab low-level full training run completed to `model_2000.pt`.

Run directory:

```text
logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_lowlevel_full/05_27_15-40-11
```

Saved checkpoints:

- `best_mean_reward.pt`, metadata `iter = 84`
- `best_goal_hold.pt`, metadata `iter = 210`
- `best_reach.pt`, metadata `iter = 225`
- `model_2000.pt`, metadata `iter = 2000`

TensorBoard scalar summary:

- `Train/mean_reward`: latest `-1845.58`, tail-100 mean `-1697.64`, best `-629.42 @ step 20`
- `Episode/rew_reach_pos_target_tight`: latest `0.0`, tail-100 mean `0.0430`, best `3.143 @ step 180`
- `Episode/goal_hold_success`: latest `0.0`, tail-100 mean `0.00123`, best `0.1458 @ step 210`
- `Episode/rew_collision`: latest `-23.30`, tail-100 mean `-28.67`
- `Episode/terrain_level`: latest `0.0078`
- `Episode/goal_level`: latest `0.0`

Fresh hard-room eval on `best_reach.pt`:

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u \
  training/isaac_lab/manual_reward_probe.py \
  --headless \
  --scenario hard_room_eval \
  --controller-mode policy \
  --checkpoint logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_lowlevel_full/05_27_15-40-11/best_reach.pt \
  --episodes 10 \
  --max-steps 1200 \
  --settle-steps 10 \
  --disable-contact-termination \
  --robot-asset-source native_go2 \
  --actuator-mode robotlab_dc \
  --low-level-controller robotlab \
  --sim-device cuda:0
```

Result:

- log directory:
  `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_27_16-50-07`
- success count: `0 / 10`
- stand failures: `8`
- max-step failures: `2`
- collision failures: `0`
- mean reward sum: `754.44`
- mean reach reward sum: `54.48`
- mean minimum distance: `4.36`

Fresh hard-room eval on `model_2000.pt`:

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u \
  training/isaac_lab/manual_reward_probe.py \
  --headless \
  --scenario hard_room_eval \
  --controller-mode policy \
  --checkpoint logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_lowlevel_full/05_27_15-40-11/model_2000.pt \
  --episodes 10 \
  --max-steps 1200 \
  --settle-steps 10 \
  --disable-contact-termination \
  --robot-asset-source native_go2 \
  --actuator-mode robotlab_dc \
  --low-level-controller robotlab \
  --sim-device cuda:0
```

Result:

- log directory:
  `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_27_16-51-48`
- success count: `0 / 10`
- stand failures: `0`
- max-step failures: `10`
- collision failures: `0`
- mean reward sum: `-1654.59`
- mean reach reward sum: `0.0`
- mean minimum distance: `5.15`

Interpretation:

- the stronger RobotLab low-level controller did improve the previously
  isolated negative-yaw low-level tracking primitive
- however, broad/full high-level PPO did not exploit that improvement into a
  usable navigation policy
- the best scalar checkpoint appeared early, around `iter = 210` to `225`, and
  later training regressed
- under the same play-style hard-room validation shape, this branch is weaker
  than the previous best `24/30` and `40/50` SEA-Nav high-level checkpoints
- current evidence says the remaining blocker is not only low-level velocity
  tracking; high-level training semantics/objective/distribution still fail to
  produce robust navigation even when the low-level controller is replaced
