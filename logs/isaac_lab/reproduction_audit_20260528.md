# SEA-Nav Isaac Lab Reproduction Audit

Date: 2026-05-28
Repo: `/home/user/rl_redeploy/SEA-Nav-Code`
Scope: audit the current Isaac Lab reproduction state, identify the real code path and evidence chain, and separate functional reproduction from paper-equivalent reproduction.

## Verdict

- 2026-06-06 baseline decision:
  - The workspace now has a usable adapted Isaac Lab SEA-Nav baseline.
  - The baseline is suitable for future method comparisons and ablations on this Isaac Lab + RobotLab stack.
  - It should not be described as a paper-equivalent line-by-line reproduction of the original Isaac Gym SEA-Nav implementation.
  - Frozen baseline manifest: `logs/isaac_lab/baseline_manifest_20260606.md`.
  - Primary baseline checkpoint remains `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/model_3500.pt`.
- Functional reproduction on the current stack is working, but the old broad eval numbers were pre-fix adapted-room results, not strict original Gym difficulty results.
  - Best currently validated all-around checkpoint after the 2026-06-01 reliable eval remains `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/model_3500.pt`.
  - The newer `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_30_19-08-44_1024env_2500it_adaptive_bounded_lr/best_goal_hold.pt` is a competitive hard-level variant, but does not supersede `model_3500.pt` overall under the 100-episode broad eval.
  - Pre-fix broad 100-episode eval on 2026-06-01:
    - `model_3500.pt`: `292/300`, success-only effective speed `0.6073 m/s`
    - `1024env bounded best_goal_hold.pt`: `290/300`, success-only effective speed `0.5574 m/s`
  - Post-fix exact Gym-difficulty eval on 2026-06-01 for `model_3500.pt`:
    - easy level 3: `99/100`
    - medium level 6: `72/100`
    - hard level 9: `55/100`
    - overall: `226/300`, success-only effective speed about `0.581 m/s`
  - Post-fix exact Gym-difficulty screen on 2026-06-01 across retained trained checkpoints:
    - `acsi_off_best_reach`: `68/90`, best in the 30-episode screen
    - `model_3500.pt`: `66/90`
    - `1024env bounded best_goal_hold.pt`: `66/90`
    - `model_16000.pt`: `57/90`, clear hard-difficulty forgetting
    - `1024env unbounded best_goal_hold.pt`: `3/90`, failed checkpoint
    - Current interpretation: `model_3500.pt` remains the strongest 100-episode exact evidence; `acsi_off_best_reach` is the best screen-level candidate and should be promoted only after a larger exact eval.
  - Minimal ablation on 2026-06-01 supports keeping bounded LR and the current reward shaping:
    - `1024env` alone is not proven to improve the result.
    - bounded adaptive LR prevents the unbounded LR collapse observed in the `1024env` run.
    - original-like reward scales reached only `167/300` under the same 100-episode broad eval, with severe stand/max-step failures.
  - Broad 30-episode eval on 2026-05-31 for the newer checkpoint was optimistic: easy level 3 `29/30`, medium level 6 `30/30`, hard level 9 `30/30`, overall `89/90`.
  - Previous hard-room 30-episode eval on 2026-05-28: `28/30` success for `model_3500.pt`, versus `24/30` for `model_16000.pt`.
  - Broad difficulty evals for `model_3500.pt`:
    - easy level 3: `30/30`, `30/30`, `30/30`
    - medium level 6: `29/30`, `30/30`, `30/30`
    - hard level 9: `28/30`, `30/30`, `30/30`
- The 2026-06-06 highyaw low-level retrain is not promoted to baseline.
  - Full exact comparison output: `logs/isaac_lab/highyaw_retrain_comparison_20260606/comparison_summary.md`.
  - `highyaw_model_2500`: easy `100/100`, medium `67/100`, hard `31/100`, total `198/300`, mean success speed `0.5917 m/s`.
  - In the same comparison, `old_256_model_3500` reaches total `222/300`, and `bounded_1024_best_goal_hold` reaches total `221/300`.
  - Highyaw improves speed and collision avoidance, but hard-room stand failures dominate, so it is an exploratory low-level branch rather than the main navigation baseline.
- Paper-equivalent reproduction is not complete.
  - The simulator is now Isaac Lab rather than Isaac Gym.
  - After forcing exact difficulty `3/6/9`, medium and hard success rates drop substantially, so easy/middle/hard test-set equivalence is not solved yet.
  - The current low-level contract is RobotLab policy driven and clipped to `[-1, 1]^3`, not the original SEA-Nav JIT path that multiplies commands by `[2.0, 2.0, 0.25]`.
  - Actuator semantics, curriculum plumbing, eval stop behavior, and training scale all changed enough that this should be treated as an adapted reproduction, not a line-by-line original-semantic reproduction.

## Best Current Evidence

### Competitive 1024-env checkpoint, 2026-05-31

- Run directory:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_30_19-08-44_1024env_2500it_adaptive_bounded_lr`
- Checkpoint:
  - `best_goal_hold.pt`
- Key config:
  - `num_envs = 1024`
  - `max_iterations = 2500`
  - `num_steps_per_env = 48`
  - `low_level_controller = "robotlab"`
  - `actuator_mode = "robotlab_dc"`
  - `learning_rate = 3e-4`
  - `learning_rate_min = 5e-5`
  - `learning_rate_max = 5e-4`
  - `goal_stop_radius = 0.45`
- 30-episode broad eval:

| difficulty | eval obstacle level | eval output | success | collision failures | stand failures | fall failures | success-only effective speed |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| easy | 3 | `logs/isaac_lab/bounded_lr_1024_best_goal_hold_eval/level_3/manual_reward_probe_hard_room_eval/05_31_17-48-39` | 29/30 | 1 | 0 | 0 | 0.5915 |
| medium | 6 | `logs/isaac_lab/bounded_lr_1024_best_goal_hold_eval/level_6/manual_reward_probe_hard_room_eval/05_31_17-51-33` | 30/30 | 0 | 0 | 0 | 0.5704 |
| hard | 9 | `logs/isaac_lab/bounded_lr_1024_best_goal_hold_eval/level_9/manual_reward_probe_hard_room_eval/05_31_17-54-34` | 30/30 | 0 | 0 | 0 | 0.4806 |

- Updated reliable-eval interpretation:
  - The 30-episode result was optimistic.
  - At 100 episodes per difficulty, this checkpoint reaches `290/300`, versus `292/300` for `model_3500.pt`.
  - It is slightly better on hard success count (`95/100` versus `94/100`) but slower overall and worse on medium.
  - Keep this checkpoint as a useful hard-level variant, not as the current all-around best.

### 2026-06-01 reliable broad eval, pre-fix adapted protocol

| Checkpoint | Level 3 | Level 6 | Level 9 | Overall | Success-only effective speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_3500.pt` | 99/100 | 99/100 | 94/100 | 292/300 | 0.6073 |
| `1024env bounded best_goal_hold.pt` | 99/100 | 96/100 | 95/100 | 290/300 | 0.5574 |

Important: these rows were produced before the exact-difficulty fix below. They remain useful for comparing checkpoints under the adapted Isaac Lab protocol, but they must not be cited as exact original Gym easy/middle/hard reproduction results.

### 2026-06-01 minimal ablation result

| Question | Evidence | Conclusion |
| --- | --- | --- |
| Is `1024 env` the main improvement source? | `1024env bounded best_goal_hold.pt` reaches `290/300`; older `256env model_3500.pt` reaches `292/300`. | Not proven. Keep `model_3500.pt` as all-around best. |
| Is bounded LR needed? | unbounded `1024env` run ends at `learning_rate=0.0`, `goal_hold_success=0.0302`, `goal_level=0.0722`; bounded run remains competitive. | Yes, bounded adaptive LR is required on this stack. |
| Does reward shaping matter? | bounded `1024env` with original-like reward scales evaluates at `92/100`, `63/100`, `12/100` for levels `3/6/9`, overall `167/300`. | Yes, current reward shaping is necessary for robust eval. |

### 2026-06-01 difficulty-equivalence audit

- Original Isaac Gym room difficulty definitions were found in `training/legged_gym/legged_gym/utils/terrain.py`:
  - `easy_room_terrain_func`: `create_rand_room(3, grid_size=20, ...)`
  - `middle_room_terrain_func`: `create_rand_room(6, grid_size=20, ...)`
  - `hard_room_terrain_func`: `create_rand_room(9, grid_size=20, ...)`
- Isaac Lab copied the same room generator, but the evaluation wrapper previously let Isaac Lab's terrain curriculum apply an extra random `difficulty` multiplier:
  - `sea_nav_room_terrain`: `round(obstacle_level * difficulty)`
  - Isaac Lab single-row curriculum samples `difficulty=(row + U(0,1))/num_rows`; for `num_rows=1`, this is random in `[0, 1)`, not exactly `1`.
- Consequence:
  - previous `--eval-obstacle-level 3/6/9` logs were not strict Gym-equivalent `3/6/9`; the effective room density was lower and seed-dependent.
- Evidence from occupied-cell ratios:

| Source | Intended level | Occupied cells | Occupied ratio |
| --- | ---: | ---: | ---: |
| Original Gym generator, exact level 3, 100-seed mean | 3 | 2431.2 | 0.2431 |
| Original Gym generator, exact level 6, 100-seed mean | 6 | 2915.5 | 0.2915 |
| Original Gym generator, exact level 9, 100-seed mean | 9 | 3371.5 | 0.3372 |
| Old reliable eval logged as level 3 | 3 | 2250 | 0.2250 |
| Old reliable eval logged as level 6 | 6 | 2400 | 0.2400 |
| Old reliable eval logged as level 9 | 9 | 2725 | 0.2725 |

- Fix applied:
  - `manual_reward_probe.py` now passes `terrain_difficulty_range=(1.0, 1.0)` for `hard_room_eval`.
  - `make_sea_nav_env_cfg` now exposes `terrain_difficulty_range` while preserving training's default `(0.0, 1.0)` curriculum behavior.
- Post-fix exact-difficulty eval for `model_3500.pt`, seed `20260601`, `100` episodes per level, `max_steps=900`:

| Difficulty | Exact obstacle level | Output | Success | Collision fails | Stand fails | Fall fails | Max-step fails | Success-only effective speed | Mean min distance | Occupied ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 3 | `logs/isaac_lab/exact_difficulty_eval_20260601/model_3500/level_3/manual_reward_probe_hard_room_eval/06_01_16-41-16` | 99/100 | 0 | 1 | 0 | 0 | 0.6263 | 0.3384 | 0.2450 |
| medium | 6 | `logs/isaac_lab/exact_difficulty_eval_20260601/model_3500/level_6/manual_reward_probe_hard_room_eval/06_01_17-08-49` | 72/100 | 5 | 14 | 1 | 8 | 0.5908 | 1.1874 | 0.2875 |
| hard | 9 | `logs/isaac_lab/exact_difficulty_eval_20260601/model_3500/level_9/manual_reward_probe_hard_room_eval/06_01_17-39-28` | 55/100 | 9 | 28 | 4 | 4 | 0.5266 | 2.2626 | 0.3375 |

Overall exact-difficulty result: `226/300 = 75.3%`, with `14` collision failures, `43` stand failures, `5` fall failures, and `12` max-step failures. This is the current strongest evidence that the adapted Isaac Lab reproduction is not yet paper-equivalent under strict original Gym room difficulty.

### 2026-06-01 all-trained checkpoint exact screen

Protocol:
- exact Gym room levels `3/6/9`
- seed `20260601`
- `30` episodes per level
- `max_steps=900`
- RobotLab low-level controller, `robotlab_dc`, `native_go2`

| Model | easy 3 | medium 6 | hard 9 | Overall | Collision | Stand | Fall | Max-step | Success-only speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `acsi_off_best_reach` | 29/30 | 23/30 | 16/30 | 68/90 = 75.6% | 3 | 16 | 1 | 2 | 0.575 |
| `model_3500.pt` | 30/30 | 22/30 | 14/30 | 66/90 = 73.3% | 5 | 13 | 2 | 4 | 0.600 |
| `1024env bounded best_goal_hold.pt` | 30/30 | 18/30 | 18/30 | 66/90 = 73.3% | 3 | 17 | 0 | 4 | 0.543 |
| `model_16000.pt` | 30/30 | 23/30 | 4/30 | 57/90 = 63.3% | 1 | 32 | 0 | 0 | 0.576 |
| `acsi_off_model_8500.pt` | 30/30 | 18/30 | 12/30 | 60/90 = 66.7% | 1 | 26 | 0 | 3 | 0.523 |
| `1024env original-reward best_goal_hold.pt` | 29/30 | 15/30 | 5/30 | 49/90 = 54.4% | 4 | 26 | 2 | 9 | 0.500 |
| `1024env unbounded best_goal_hold.pt` | 1/30 | 1/30 | 1/30 | 3/90 = 3.3% | 7 | 35 | 1 | 44 | 0.229 |

Interpretation:
- `acsi_off_best_reach` is the best screen-level checkpoint, but its margin over `model_3500.pt` is small.
- `model_16000.pt` confirms long-run hard-difficulty forgetting; failures are mostly `stand`.
- Removing adaptive LR bounds produces a collapsed policy under exact eval.
- Original-like reward scales underperform the shaped reward run, so reward shaping is currently helping rather than causing the exact hard-room gap.

### Main training run

- Run directory:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2`
- Key config:
  - `low_level_controller = "robotlab"` in `args.json:30`
  - `lr_schedule = "adaptive"` in `args.json:31`
  - `max_iterations = 16000` in `args.json:32`
  - `goal_stop_radius = 0.45` in `args.json:21`
  - `action_reg_min/max = [-1, 1]^3` in `train_cfg.json:8-18`
  - `learning_rate_min = 5e-5`, `learning_rate_max = 5e-4` in `train_cfg.json:19-20`
- TensorBoard:
  - valid event file size `2185899` bytes
  - the three earlier curriculum siblings only have `88`-byte event files, so they should be treated as incomplete logging runs, not the main result.

### Hard-room checkpoint comparison

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-11-41/summary.json`
  - checkpoint `model_3500.pt`
  - `success_count=28`, `collision_failures=1`, `stand_failures=1`, `mean_min_distance=0.4965`
- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-15-02/summary.json`
  - checkpoint `model_16000.pt`
  - `success_count=24`, `collision_failures=0`, `stand_failures=6`, `mean_min_distance=1.0515`

This is the clearest evidence that longer training on the current setup degraded the policy.

### Broad difficulty evals for `model_3500.pt`

- easy:
  - `05_28_10-20-34`, `05_28_10-23-33`, `05_28_10-26-11`
  - all `30/30`
- medium:
  - `05_28_10-28-54`, `05_28_10-31-54`, `05_28_10-34-34`
  - `29/30`, `30/30`, `30/30`
- hard:
  - `05_28_10-37-31`, `05_28_10-40-56`, `05_28_10-43-55`
  - `28/30`, `30/30`, `30/30`

Representative summary rows with failure breakdown:

| difficulty | eval obstacle level | representative summary | success | collision-free success | collision failures | stand failures | fall failures | timeout failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 3 | `05_28_10-20-34/summary.json` | 30/30 | 30/30 | 0 | 0 | 0 | 0 |
| medium | 6 | `05_28_10-28-54/summary.json` | 29/30 | 29/30 | 0 | 0 | 1 | 0 |
| hard | 9 | `05_28_10-37-31/summary.json` | 28/30 | 28/30 | 1 | 1 | 0 | 0 |

Representative successful trace rows:

- easy: `05_28_10-20-34/trace.jsonl`
  - `start_cell=[10, 21]`, `goal_cell=[74, 38]`, `first_reach_step=298`, `done_step=447`
- medium: `05_28_10-28-54/trace.jsonl`
  - `start_cell=[74, 63]`, `goal_cell=[39, 79]`, `first_reach_step=181`, `done_step=330`
- hard: `05_28_10-37-31/trace.jsonl`
  - `start_cell=[26, 86]`, `goal_cell=[65, 84]`, `first_reach_step=450`, `done_step=599`

### Recording-chain assets

Recording-chain assets were useful during the GUI and OVD validation phase, but they are not part of the retained core reproduction package.

- They were only display and tooling assets:
  - `logs/isaac_lab/eval_case_mp4/...`
  - `logs/isaac_lab/ovd_recordings/...`
  - `logs/isaac_lab/topdown_videos/...`
  - `logs/isaac_lab/visual_progressive_cases/...`
  - `logs/isaac_lab/recording_probes/...`
- In the current cleanup pass, these assets are intentionally removed per user request.
- The retained proof of reproduction is the broad eval evidence under `logs/isaac_lab/manual_reward_probe_hard_room_eval/`, not the recording outputs.

## Actual Control Chain

### Training entry

- `training/isaac_lab/train.py:63-96`
  - builds PPO config
  - for `low_level_controller == "robotlab"`, default action regularization becomes `[-1, 1]^3`
- `training/isaac_lab/train.py:584-590`
  - passes `goal_stop_radius`, `nav_action_scale`, and `low_level_controller` into the Isaac Lab env

### High-level policy and CBF shield

- `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py:51-91`
  - `nav_head` outputs nominal navigation command `u_bar`
  - `alpha_head` outputs shield parameter
  - `cbf_layer = ExactLSECBFLayer(...)`
- `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py:140-149`
  - forward path computes `u_bar -> alpha -> u_s`
- `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:10-12`
  - `safe_radius=0.15`, `safety_margin=0.05`, `kappa=10.0`
- `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:33-38`
  - `_load_from_state_dict` rebuilds `ray_unit_vectors`, which is the local fix for old checkpoint geometry mismatch
- `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:45-80`
  - only the planar `(vx, vy)` part is modified by the shield
  - yaw is passed through unchanged

### Isaac Lab env action chain

- `training/isaac_lab/sea_nav_env.py:614-617`
  - holds both `slr_commands_scale = [2.0, 2.0, 0.25]` and the RobotLab command clip setup
- `training/isaac_lab/sea_nav_env.py:920-943`
  - high-level actions are filtered and scaled
  - final low-level command is clamped by `self.nav_clip_min/max`
- `training/isaac_lab/sea_nav_env.py:970-986`
  - RobotLab low-level path clamps commands to `[-robotlab_command_clip, robotlab_command_clip]`
  - current default `robotlab_command_clip = 1.0`
- `training/isaac_lab/sea_nav_env.py:989-996`
  - legacy SEA-Nav JIT path still multiplies by `self.slr_commands_scale`

In other words, `nav_actions * [2.0, 2.0, 0.25]` belongs to the legacy JIT path, not the active RobotLab path.

### Reward, termination, curriculum, and ACSI/collision replay

- `training/isaac_lab/sea_nav_env.py:369-377`
  - env default reward scales are defined here
- `training/isaac_lab/sea_nav_env.py:1244-1253`
  - active reward terms are assembled as:
  - `termination`
  - `collision`
  - `close_obst_vel`
  - `stuck`
  - `progress`
  - `far_goal_stand`
  - `velo_dir`
  - `reach_pos_target_tight`
  - `ang_vel_xy`
- `training/isaac_lab/sea_nav_env.py:383-386`
  - collision replay defaults:
  - `enable_collision_replay = True`
  - `collision_replay_prob = 0.8`
  - `collision_replay_early_reset_prob_range = (0.1, 0.5)`
  - `collision_replay_undo_steps_range = (100, 150)`
- `training/isaac_lab/sea_nav_env.py:1264-1265`
  - early replay reset probability scales with `goal_levels / 1.5`
- `training/isaac_lab/sea_nav_env.py:1327-1334`
  - new collisions can trigger early replay reset before the nominal episode end
- `training/isaac_lab/sea_nav_env.py:1361-1368`
  - reset can roll state back with collision replay when history is available
- `training/isaac_lab/sea_nav_env.py:1342-1351`
  - episode end conditions are:
  - `goal_hold`
  - `stand_still`
  - `fall_down`
  - collision/contact derived termination via `self._terminated`
  - `time_out`
- `training/isaac_lab/sea_nav_env.py:1276-1278`
  - curriculum updates `goal_levels` and then delegates environment-origin updates to the terrain system

For the current validated run, the env defaults above were overridden by `args.json`:

- `reward_scale_progress = 20.0`
- `reward_scale_far_goal_stand = -10.0`
- `reward_scale_stuck = -12.0`
- `reward_scale_velo_dir = 6.0`

This is one of the reasons the current result should be treated as an adapted reproduction rather than a frozen paper-default replay.

### Low-level policy source and contract

- policy files:
  - `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/policy.pt`
  - `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/model_4999.pt`
- source:
  - this is a RobotLab flat-velocity locomotion policy snapshot packaged into the current repo under `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/`
  - it is not the original SEA-Nav low-level JIT policy
- command ranges:
  - `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/env.yaml:1632-1640`
  - `lin_vel_x`, `lin_vel_y`, `ang_vel_z` are all `[-1.0, 1.0]`
- low-level PPO config:
  - `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/agent.yaml:38-43`
  - `learning_rate=0.001`, `schedule=adaptive`, `desired_kl=0.01`

### Low-level tracking evidence

- `logs/isaac_lab/low_level_tracking_probe/05_27_20-32-31/summary.json`
  - requested `vx2 = [2.0, 0.0, 0.0]`
  - filtered command mean becomes `vx=1.0`
  - actual mean becomes approximately `vx=1.19`
  - `vy1` actual mean approximately `vy=0.95`
  - `yaw1` actual mean approximately `wz=1.09`

Therefore, the current deployed contract is approximately `vx, vy, yaw in [-1, 1]`, not true direct 2.0 m/s forward command tracking.

## Where Current Training Semantics Differ From Original Isaac Gym

### Changed simulator and actuator semantics

- Isaac Lab env uses `DCMotorCfg` for `actuator_mode == "robotlab_dc"` at `training/isaac_lab/sea_nav_env.py:444-445`
- Original Gym path was built around the older legged-gym actuation and low-level wiring

### Changed command contract

- Isaac Lab RobotLab path:
  - `training/isaac_lab/sea_nav_env.py:970-986`
  - clipped `[-1, 1]^3`
- Original Gym JIT path:
  - `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:189-193`
  - multiplies by `slr_commands_scale`

### Changed curriculum plumbing

- Isaac Lab:
  - `training/isaac_lab/sea_nav_env.py:1276-1278`
  - updates `goal_levels`, then calls `self._terrain.update_env_origins(...)`
- Original Gym:
  - `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:423-438`
  - explicitly updates both `goal_levels` and `terrain_levels`

### Changed eval behavior

- Current run used `goal_stop_radius = 0.45` from `args.json:21`
- Original Gym play config disables collision replay and contact termination:
  - `training/legged_gym/legged_gym/scripts/play.py:57`
  - `training/legged_gym/legged_gym/scripts/play.py:79`

### Changed training scale

- Original Gym config base episode length:
  - `training/legged_gym/legged_gym/envs/base/legged_robot_pos_config.py:43`
  - `episode_length_s = 8`
- Current Isaac Lab main run:
  - `args.json` uses `256` envs and `max_iterations=16000`
  - the run name says `equiv_steps_v2`, but this is still not the same as re-running the original stack unchanged
- hardware training time:
  - the current workspace does not contain a reliable end-to-end wall-clock record for the validated run
  - therefore current hardware-vs-paper training time equivalence is not proven from workspace evidence alone

## Current Learning Dynamics Findings

- Adaptive LR exists and is bounded:
  - `training/rsl_rl/rsl_rl/algorithms/ppo.py:219-228`
- In the validated main run, `Train/learning_rate` stayed pinned at `5e-4` for all `1598` logged points.
- Other late-run signals from TensorBoard:
  - `Train/mean_reward`: `62.5 -> 851.3`, max `3962.1`
  - `Episode/goal_hold_success`: `0.0 -> 0.1354`, max `0.6667`
  - `Loss/smooth`: `0.0073 -> 3.3708`, max `7.1122`
  - `Loss/regularization`: `0.0005 -> 0.1687`, max `0.3556`
  - `Loss/intervention`: `0.0026 -> 0.00033`, max `0.00871`

This is consistent with the later-checkpoint degradation: the run kept learning, but not toward a more reliable navigation policy.

## Archived Historical Tuning Lessons

The following lesson summary is the archived replacement for the old targeted-tuning directories that are no longer required as retained assets.

- `exactroom_*`, `failuremix_*`, `hardclassmix_*`, `harddirmix_*`, `hardgoalx_*`
  - these branches explored whether hand-crafted hard-room sampling, failure-case mixing, direction-class mixing, reward rebalance, turn priors, or local CBF/domain-randomization tweaks could rescue end-to-end performance
  - the final lesson is that none of these single-branch targeted tweaks became the retained mainline; the stable retained path shifted to training directly against the RobotLab low-level contract with the later curriculum run
- `full_from_turnopen*`, `onehardroom_*`, `turn_open_*`
  - these branches were warm-start and replay-check scaffolding during the migration period
  - the final lesson is that these were useful stepping stones for bootstrapping and debugging, but they were not the final evaluation protocol and did not replace the retained broad hard-room evidence
- `full_goalstop045_*`, `full_hold_stabilize_lowent`
  - these branches explored stop-radius and near-goal stabilization behavior
  - the useful retained outcome is that the current validated run uses `goal_stop_radius = 0.45`, but the branches themselves are no longer needed once that choice is captured in `args.json` and the eval summaries
- `full_repro_repaired`
  - this was an earlier repaired full-run attempt before the later RobotLab-contract curriculum path
  - it remains part of the historical story, but not of the minimum retained reproduction package

Because these lessons are now summarized here and the final retained evidence lives in the active main run, broad evals, and low-level tracking probes, the directories themselves no longer need to be kept for continuation.

## Archived Long-Run Migration Lessons

The following summary is the archived replacement for the remaining non-core long-run directories from the migration period.

- `robotlab_lowlevel_full`
  - this was the most complete early training history after wiring the RobotLab low-level controller into the Isaac Lab stack
  - the retained lesson is not the checkpoint ladder itself, but the fact that low-level integration was trainable; the actionable proof is now preserved by the retained low-level policy directory and `logs/isaac_lab/low_level_tracking_probe/`
- `robotlab_contract_retrain`
  - this was an explicit intermediate retrain and resume branch under the RobotLab contract
  - the retained lesson is that a direct retrain branch still did not become the final mainline; it was superseded by the later `robotlab_contract_curriculum/...equiv_steps_v2` run
- `gpu_train_original_semantics`, `gpu_train_rtx5060ti`
  - these were earlier long training runs from the migration period before the final retained RobotLab-contract alignment
  - the retained lesson is that simply continuing earlier semantics or hardware bring-up lines was not enough to produce the current best hard-room result; later contract and curriculum changes mattered more than keeping those branches alive
- `turn_open_replaycheck`, `full_from_turnopen320_curriculumfix`
  - these were replay-check, warm-start, and curriculum-fix long runs used to bootstrap and debug the migration
  - the retained lesson is that they were useful scaffolding, but they were not the final evaluation protocol and were superseded by the later hard-room curriculum line and broad eval evidence

Because these long-run lessons are now summarized here, and because the retained continuation bundle already keeps the final high-level run, broad eval proof, low-level contract probe, and low-level policy asset, the old long-run directories themselves are no longer required.

## Recording Pipeline Status

- `training/isaac_lab/manual_reward_probe.py:61, 80-81, 102-104, 197, 1120-1141`
  - supports `--eval-room-profile`, `--case-trace`, top-down camera, markers, and deterministic replay from sibling `room.npy`
- `training/isaac_lab/manual_reward_probe.py:1106-1114`
  - contains the NumPy preload workaround needed before Isaac Sim GUI startup
- `training/isaac_lab/record_eval_cases_to_mp4.py:37-48, 224, 262, 315, 345, 379-432`
  - supports both `live` and `ovd` backends
  - default eval room profile is now `visual_progressive`

Status:

- OVD/MP4 conversion chain is functional.
- Earlier OVD videos were not trusted as visual proof because the rendered robot motion could look wrong even when the eval trace reached `goal_hold`.
- The corresponding historical recording assets are not retained in the core reproduction bundle after the user-requested recording cleanup.
- Older simplified scenario probes such as `straight`, `turn`, `turn_open`, `turn_wide`, and the separate `hard_room_controller_diag` traces are also not retained in the core bundle; the retained rollout evidence is the broad `hard_room_eval` set plus `low_level_tracking_probe`.

## Reproduction Command Pack

All commands below assume:

- repo cwd: `/home/user/rl_redeploy/SEA-Nav-Code`
- python env: `/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python`

### Training command

The exact original shell line was not logged. The command below is a reconstructed equivalent command from `args.json` and `train_cfg.json`.

```bash
cd /home/user/rl_redeploy/SEA-Nav-Code
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=YES \
/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --experiment-name Go2_pos_rough_isaaclab_robotlab_contract_curriculum \
  --run-name 256env_16000it_adaptive_limited_curriculum_equiv_steps_v2 \
  --num-envs 256 \
  --num-steps-per-env 48 \
  --max-iterations 16000 \
  --learning-rate 3e-4 \
  --learning-rate-min 5e-5 \
  --learning-rate-max 5e-4 \
  --lr-schedule adaptive \
  --entropy-coef 0.003 \
  --seed 1 \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --device cuda:0 \
  --active-gpu 0 \
  --physics-gpu 0 \
  --headless \
  --hide-ui \
  --actuator-mode robotlab_dc \
  --low-level-controller robotlab \
  --robot-asset-source native_go2 \
  --robotlab-low-level-policy training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/policy.pt \
  --terrain-rows 10 \
  --terrain-cols 10 \
  --obstacle-level 9 \
  --episode-length-s 120 \
  --goal-stop-radius 0.45 \
  --goal-stop-mode zero \
  --nav-action-scale 1.0 1.0 1.0 \
  --reward-scale-termination -100 \
  --reward-scale-collision -4 \
  --reward-scale-close-obst-vel 5 \
  --reward-scale-stuck -12 \
  --reward-scale-progress 20 \
  --reward-scale-far-goal-stand -10 \
  --reward-scale-velo-dir 6 \
  --reward-scale-reach-pos-target-tight 10 \
  --reward-scale-ang-vel-xy -0.05 \
  --save-interval 500
```

### Broad easy/medium/hard evaluation command

Change only `--eval-obstacle-level` between `3`, `6`, and `9`.

```bash
cd /home/user/rl_redeploy/SEA-Nav-Code
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=YES \
/home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/manual_reward_probe.py \
  --scenario hard_room_eval \
  --controller-mode policy \
  --checkpoint logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/model_3500.pt \
  --episodes 30 \
  --eval-obstacle-level 3 \
  --actuator-mode robotlab_dc \
  --low-level-controller robotlab \
  --robot-asset-source native_go2 \
  --max-steps 900 \
  --policy-stop-radius 0.45 \
  --policy-stop-mode zero \
  --sim-device cuda:0
```

## Reproduction Status Summary

- Current answer to "是否完成复现":
  - Engineering reproduction on the present Isaac Lab + RobotLab stack: yes, basically completed.
  - Paper-equivalent semantic reproduction: no.
- Current answer to "另一个人能否只根据文档和路径重新跑出 easy/medium/hard 评估":
  - yes, for the current workspace and current venv
  - the broad eval command is explicit in this document
  - no retained recording asset is required for this
- Current answer to "与原论文差多少":
  - the main gap is not just training time; it is the control contract and simulator semantics
  - the active low-level is narrower and different from the original JIT low-level path
  - evaluation and actuation are no longer paper-identical
  - exact current wall-clock training duration on this hardware is not proven from the run artifacts, so only step/config comparison is currently defensible
- Current answer to "当前难度是否已经等价于原论文":
  - not proven
  - missing comparable evidence includes:
  - an original-paper-aligned room density metric
  - a direct mapping from current `eval_obstacle_level` or `occupied interior cells` to the paper difficulty bins
  - a paper-equivalent timeout and success protocol on the same simulator stack
  - a paper-equivalent low-level controller contract
- Current answer to "为什么有这些修改":
  - they were needed to make the system stable on Isaac Lab and compatible with the RobotLab low-level controller
  - several recording additions are purely evaluation tooling and do not change training semantics
- Current answer to "哪些录制资产仍属于核心资料集":
  - none
  - recording-chain outputs were intentionally cleaned per user request and do not belong to the minimum retained reproduction set
- Current answer to "为什么当前版本能跑到现在的效果":
  - the high-level policy is now trained against the actual RobotLab low-level contract instead of the old wider JIT command assumption
  - the local CBF ray-geometry load fix prevents old checkpoint buffer mismatch from silently breaking shielding
  - the current curriculum run uses bounded adaptive LR and a reward mix that strongly favors progress and reach behavior
  - `goal_stop_radius = 0.45` reduces late-stage chattering near the target during current eval
- Current answer to "为什么仍不稳定":
  - later checkpoints degrade, so the present training objective is still not stably aligned with best eval performance
  - the active low-level saturates the forward command compared with the paper-like `2.0` expectation
  - collision replay and early replay reset change the optimization landscape relative to the original Gym implementation
  - current difficulty equivalence to the paper is still unproven, so robustness claims cannot yet be transferred one-to-one
