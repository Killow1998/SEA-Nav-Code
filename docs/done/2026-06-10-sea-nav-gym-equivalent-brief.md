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

## Current interpretation

1. G0 is viable after the torque-limit fix, but it is not yet an official-performance reproduction. Hard level is still `58/100` under official play-style no-stop eval.
2. CBF FOV is not the primary explanation in this snapshot. Switching 180 to 240 improves total by only `+5/300`, mostly hard `58 -> 64`.
3. Native Go2 plus RobotLab DC actuator changes failure mode, especially falls, but does not dominate aggregate success in this one-checkpoint ablation.
4. R0's strong adapted result depends heavily on the explicit 0.45m stop wrapper. Without the stop wrapper, R0 is about the same aggregate as G0/G1/G2.
5. The remaining paper-equivalent blocker appears to be high-level goal-hold / stand behavior in medium-hard rooms, not collision avoidance and not the official JIT low-level controller.

## Prompt for GPTPro

Please analyze this Isaac Lab SEA-Nav reproduction state.

The main question is:

What is the most likely remaining mismatch preventing the paper-equivalent Isaac Lab G0 line from matching official SEA-Nav Gym performance, given that the low-level torque-limit bug is fixed and collision failures are zero?

Please reason from the matrix above and prioritize the next discriminating checks. In particular:

1. Should the next step be official Gym eval-semantics parity, observation/reward/done parity, or longer G0 training?
2. In official SEA-Nav Gym, how exactly should success be counted: first reach, `goal_reached_time`, `stay_time`, or a play-wrapper behavior?
3. Why would failures show mostly stand/high-level idle or moving-without-progress while collision remains zero?
4. Could the 40s/500-step official play setting mean that our `goal_hold` implementation is still semantically wrong?
5. Which code-level parity checks should be done before changing rewards or PPO?

Desired output:

- a ranked list of likely remaining mismatches
- the first 3 concrete checks to run
- what result would falsify each hypothesis
- whether to keep CBF 180 for official compatibility or use 240 for geometry consistency
