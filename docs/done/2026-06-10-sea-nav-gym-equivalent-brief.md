# SEA-Nav Isaac Lab gym-equivalent brief

Date: 2026-06-10

Branch: `repro-gym-equivalent-isaaclab`

## Goal

We are trying to reproduce the official SEA-Nav Gym performance in Isaac Lab, not merely tune a RobotLab-adapted baseline.

The intended paper-equivalent line is:

- low-level controller: official SEA-Nav JIT (`body_latest.jit`, `encoder_vel.jit`, `encoder_latent.jit`)
- robot asset / actuator: converted `go2_description_v8.urdf` plus manual Gym-style PD torque
- high-level action range: `vx [-0.5, 2.0]`, `vy [-1.0, 1.0]`, `wz [-1.0, 1.0]`
- CBF FOV: official-compatible `180 deg` first, with `240 deg` as an ablation
- no RobotLab low-level policy and no implicit stop wrapper for paper-equivalent eval

## Confirmed implementation issue

The initial Isaac Lab low-level parity issue was caused by our Lab implementation, not by the official SEA-Nav JIT controller itself.

Bug:

- Isaac Lab `gym_torque` clamped every joint to `80 Nm`.
- Official Gym URDF effort limits are:
  - hip: `23.7 Nm`
  - thigh: `23.7 Nm`
  - calf: `45.43 Nm`

Fix:

- `training/isaac_lab/sea_nav_env.py` now applies per-joint URDF effort limits for the `gym_torque` fallback clamp.
- Startup now prints `torque_limit_min=23.700 torque_limit_max=45.430`.

Interpretation:

- Native Gym low-level tracking passed.
- Corrected Lab clean-room low-level tracking is stable enough.
- The remaining reproduction gap should not be blamed primarily on the SEA-Nav JIT low-level controller.

## Eval protocol added

`training/isaac_lab/eval_checkpoint_comparison.py` now supports explicit eval protocols:

- `strict_env`: 60s episode, 150-step goal hold, contact termination enabled, no stop.
- `official_play_style`: 40s episode, 500-step goal hold, contact termination disabled, no stop.
- `assist_stop`: same as official play-style, plus explicit `policy_stop_radius=0.45`, `policy_stop_mode=zero`.

The important correction is that our earlier strict result used `max_steps=900`, only 18 seconds at 50 Hz, which was too short and produced misleading max-step failures.

## Repro harness fix

The first pushed matrix commit was incomplete as a reproducible harness:

- `eval_checkpoint_comparison.py` passed `--repro-mode` and `--cbf-fov-deg` into `manual_reward_probe.py`.
- The pushed `manual_reward_probe.py` did not yet register those arguments.
- `DifferentiableSafeActorCritic` instantiated `ExactLSECBFLayer(num_rays=num_rays)` without an explicit FOV, so a loaded checkpoint could not reliably reproduce `180 deg` versus `240 deg` CBF geometry.

Fix:

- `manual_reward_probe.py` now accepts `--repro-mode` and `--cbf-fov-deg`.
- `train.py`, `play.py`, and `eval_checkpoint_comparison.py` expose the same CBF FOV knob.
- `DifferentiableSafeActorCritic` now passes `cbf_fov_deg` into `ExactLSECBFLayer`.
- `--repro-mode gym_equiv` defaults to `180 deg`, while explicit `--cbf-fov-deg 240` remains supported for G1.
- `manual_reward_probe.py` no longer lets `--repro-mode gym_equiv` override an explicit assist-stop setting. The earlier `G0_fresh2500_assist_stop_100eps_20260610` run is invalid for assist-stop because this override was still present.
- `eval_checkpoint_comparison.py` preserves an explicitly provided `--policy-stop-radius` under `--repro-mode gym_equiv --eval-protocol custom`; default gym-equivalent custom eval still runs with no stop wrapper.

## Fresh G0 training

Fresh paper-equivalent G0 run:

- run dir: `logs/isaac_lab/G0_gym_equiv_training/06_10_17-58-56_1024env_2500it_torque_fix_20260610`
- checkpoint: `model_2500.pt`
- settings:
  - `converted_urdf`
  - `gym_torque`
  - `sea_nav_jit`
  - official high-level action range
  - adaptive LR bounds `1e-5` to `1e-2`
  - CBF FOV `180`
  - no goal-stop heuristic

## Four-path matrix

All rows below use:

- eval protocol: `official_play_style`
- `episode_length_s=40`
- `max_steps=2000`
- `stay_steps=500`
- contact termination disabled
- no policy stop wrapper
- 100 episodes at each obstacle level 3, 6, 9
- seed `20260610`

| Path | Low-level | Asset / actuator | CBF FOV | Checkpoint | Level 3 | Level 6 | Level 9 | Total | Collision | Stand | Fall | Timeout |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | `sea_nav_jit` | `converted_urdf` / `gym_torque` | 180 | fresh G0 `model_2500.pt` | 96/100 | 69/100 | 58/100 | 223/300 | 0 | 60 | 2 | 15 |
| G1 | `sea_nav_jit` | `converted_urdf` / `gym_torque` | 240 | same G0 checkpoint | 96/100 | 68/100 | 64/100 | 228/300 | 0 | 58 | 1 | 13 |
| G2 | `sea_nav_jit` | `native_go2` / `robotlab_dc` | 180 | same G0 checkpoint | 91/100 | 73/100 | 63/100 | 227/300 | 0 | 43 | 17 | 13 |
| R0 | `robotlab` | `native_go2` / `robotlab_dc` | 240 | RobotLab `best_goal_hold.pt` | 80/100 | 79/100 | 67/100 | 226/300 | 0 | 27 | 18 | 29 |

R0 adapted sanity check:

- Same R0 checkpoint, but with `assist_stop`.
- Result: `97/100`, `88/100`, `87/100`, total `272/300`.

## First-reach re-score

The traces record `first_reach_step` and `min_distance`, so the same episodes can be re-scored without changing the environment. This separates "reached but failed goal-hold" from "never reached the target radius".

| Run | Level | goal_hold | first_reach | min<0.5 | reached_then_failed | Stand | Fall | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 no-stop | 3 | 96/100 | 98/100 | 98/100 | 2/100 | 2 | 0 | 2 |
| G0 no-stop | 6 | 69/100 | 76/100 | 76/100 | 7/100 | 25 | 1 | 5 |
| G0 no-stop | 9 | 58/100 | 63/100 | 63/100 | 5/100 | 33 | 1 | 8 |
| G0 no-stop | all | 223/300 | 237/300 | 237/300 | 14/300 | 60 | 2 | 15 |
| G1 no-stop | 9 | 64/100 | 66/100 | 66/100 | 2/100 | 29 | 1 | 6 |
| R0 no-stop | 9 | 67/100 | 85/100 | 85/100 | 18/100 | 9 | 13 | 11 |
| R0 assist-stop | 9 | 87/100 | 87/100 | 87/100 | 0/100 | 7 | 3 | 3 |

Interpretation:

- R0 no-stop has a large success-semantics gap: hard `first_reach=85/100` but `goal_hold=67/100`.
- G0 no-stop does not. Hard `first_reach=63/100` and `goal_hold=58/100`; only `5/100` episodes reached then failed to hold.
- Therefore stop/hold semantics strongly explains the adapted RobotLab R0 drop, but it cannot explain most of the paper-equivalent G0 hard-room gap.

## Assist-stop check on G0/G1

After fixing the repro-mode override in `manual_reward_probe.py`, the true assist-stop runs are:

| Path | Eval protocol | CBF FOV | Level 3 | Level 6 | Level 9 | Total | Collision | Stand | Fall | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | official_play_style | 180 | 96/100 | 69/100 | 58/100 | 223/300 | 0 | 60 | 2 | 15 |
| G0 | assist_stop | 180 | 99/100 | 79/100 | 63/100 | 241/300 | 0 | 48 | 2 | 9 |
| G1 | official_play_style | 240 | 96/100 | 68/100 | 64/100 | 228/300 | 0 | 58 | 1 | 13 |
| G1 | assist_stop | 240 | 96/100 | 77/100 | 65/100 | 238/300 | 0 | 51 | 2 | 9 |
| R0 | official_play_style | 240 | 80/100 | 79/100 | 67/100 | 226/300 | 0 | 27 | 18 | 29 |
| R0 | assist_stop | 240 | 97/100 | 88/100 | 87/100 | 272/300 | 0 | 21 | 3 | 4 |

G0/G1 assist-stop gives only modest improvement, especially on hard rooms:

- G0 hard: `58 -> 63`
- G1 hard: `64 -> 65`
- R0 hard: `67 -> 87`

In the true G0/G1 assist-stop traces, hard first-reach equals goal-hold (`G0 63/100`, `G1 65/100`). That means the remaining hard-room failures are mostly not "reached but counted as failure"; they mostly never enter the reach radius.

## Current interpretation

1. The harness gap was real and is now fixed. Future G0/G1 evals can reproduce `repro_mode` and CBF FOV from committed code.
2. The low-level SEA-Nav JIT controller is not the primary remaining suspect. The torque-limit bug was ours, and the corrected low-level path is stable enough for this matrix.
3. Collision avoidance is not the blocker in this snapshot. All matrix rows have `collision=0`.
4. Stop / success semantics is the main issue for R0 no-stop, but not for G0/G1. G0 hard no-stop reaches only `63/100`; true G0 assist-stop succeeds `63/100`.
5. CBF `240 deg` is mildly helpful on hard no-stop (`58 -> 64`) but does not close the gap and should remain an ablation, not the official-compatible setting.
6. The next paper-equivalent work should focus on fixed-case Gym-vs-Lab parity for observation, done/stand timers, reward terms, and trajectory behavior before changing reward scales or PPO.

## Prompt for GPTPro

Please analyze this Isaac Lab SEA-Nav reproduction state.

The main question is:

What is the most likely remaining mismatch preventing the paper-equivalent Isaac Lab G0 line from matching official SEA-Nav Gym performance, given that the low-level torque-limit bug is fixed, the committed eval harness now supports `--repro-mode` / `--cbf-fov-deg`, collision failures are zero, and true G0/G1 assist-stop does not materially improve hard-room success?

Key evidence:

- G0 no-stop: `96/100`, `69/100`, `58/100`, total `223/300`.
- G0 assist-stop: `99/100`, `79/100`, `63/100`, total `241/300`.
- G1 no-stop: `96/100`, `68/100`, `64/100`, total `228/300`.
- G1 assist-stop: `96/100`, `77/100`, `65/100`, total `238/300`.
- R0 no-stop: `80/100`, `79/100`, `67/100`, total `226/300`.
- R0 assist-stop: `97/100`, `88/100`, `87/100`, total `272/300`.
- G0 hard first-reach: no-stop `63/100`, assist-stop `63/100`.
- R0 hard first-reach: no-stop `85/100`, assist-stop `87/100`.

Please reason from the matrix above and prioritize the next discriminating checks. In particular:

1. Should the next step be fixed-case Gym-vs-Lab observation/done/reward parity or longer G0 training?
2. Why does G0 fail mostly as stand/high-level-idle or moving-without-progress while collision remains zero and first-reach is low?
3. Which exact observation fields should be compared first: rays/log2 rays, goal-local position, projected gravity, base velocity, delayed obs history, or CBF output?
4. Which done/reward fields are most likely to diverge: `static`, `stand_still_flag`, `goal_hold_timer`, `fall_down`, timeout order, or reach reward?
5. What result would falsify "observation/done parity mismatch" and push us toward "training budget / PhysX residual" instead?

Desired output:

- a ranked list of likely remaining mismatches
- the first 3 concrete checks to run
- what result would falsify each hypothesis
- whether to keep CBF 180 for official compatibility or use 240 for geometry consistency
