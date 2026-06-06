# SEA-Nav Isaac Lab Baseline Manifest

Date: 2026-06-06

## Decision

The current workspace is frozen as an adapted Isaac Lab SEA-Nav baseline, not a paper-equivalent line-by-line reproduction.

Use this baseline for future method comparisons unless a later run beats it under the same exact evaluation protocol.

## Primary Baseline

- Label: `isaaclab_robotlab_model_3500`
- Checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/model_3500.pt`
- Low-level policy:
  - `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/policy.pt`
- Controller stack:
  - Isaac Lab environment
  - `actuator_mode=robotlab_dc`
  - `low_level_controller=robotlab`
  - `robot_asset_source=native_go2`
  - `goal_stop_radius=0.45`
  - `policy_stop_mode=zero`

## Secondary Baseline

- Label: `isaaclab_robotlab_1024env_bounded_best_goal_hold`
- Checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_30_19-08-44_1024env_2500it_adaptive_bounded_lr/best_goal_hold.pt`
- Role:
  - Hard-room variant and robustness comparison point.
  - Not the main all-around baseline because total 100-episode exact-eval success is slightly lower and speed is lower.

## Frozen Evaluation Protocol

Use exact Gym-equivalent room difficulty:

- easy: `--eval-obstacle-level 3`
- medium: `--eval-obstacle-level 6`
- hard: `--eval-obstacle-level 9`
- episodes: `100` per difficulty
- seed: `20260601`
- max steps: `900`
- `manual_reward_probe.py` forces `terrain_difficulty_range=(1.0, 1.0)` for `hard_room_eval`

Command template:

```bash
cd /home/user/rl_redeploy/SEA-Nav-Code
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=YES \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/manual_reward_probe.py \
  --headless \
  --scenario hard_room_eval \
  --controller-mode policy \
  --checkpoint logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/model_3500.pt \
  --episodes 100 \
  --num-envs 16 \
  --seed 20260601 \
  --eval-obstacle-level 3 \
  --actuator-mode robotlab_dc \
  --low-level-controller robotlab \
  --robot-asset-source native_go2 \
  --max-steps 900 \
  --policy-stop-radius 0.45 \
  --policy-stop-mode zero \
  --sim-device cuda:0
```

Change only `--eval-obstacle-level` to `6` and `9` for medium and hard.

## Main Evidence

Exact 100-episode eval for `model_3500.pt`:

| Difficulty | Level | Success | Collision | Stand | Fall | Max-step | Success-only speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 3 | 99/100 | 0 | 1 | 0 | 0 | 0.6263 |
| medium | 6 | 72/100 | 5 | 14 | 1 | 8 | 0.5908 |
| hard | 9 | 55/100 | 9 | 28 | 4 | 4 | 0.5266 |
| total | - | 226/300 | 14 | 43 | 5 | 12 | about 0.581 |

Comparable 2026-06-06 exact eval:

| Model | easy | medium | hard | Total | Mean success speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old_256_model_3500` | 98/100 | 74/100 | 50/100 | 222/300 | 0.5684 |
| `bounded_1024_best_goal_hold` | 97/100 | 70/100 | 54/100 | 221/300 | 0.5287 |
| `highyaw_model_2500` | 100/100 | 67/100 | 31/100 | 198/300 | 0.5917 |

The small difference between `226/300` and `222/300` for `model_3500.pt` is expected from rerunning stochastic exact eval with the same seed but different comparison harness timing and current code state.

## HighYaw Low-Level Follow-Up Result

The highyaw low-level branch is not promoted to baseline.

- It improved successful-trajectory speed and eliminated collision failures in the comparison run.
- It reduced hard success substantially: `31/100` on hard versus `50/100` for `model_3500.pt` and `54/100` for the 1024-env bounded variant.
- TensorBoard shows no numerical collapse, but the high-difficulty phase plateaued/regressed:
  - `goal_hold_success` peaked at `0.7049` only at low curriculum difficulty around iter `140`
  - high-difficulty training averaged about `0.39` after iter `2000`
  - `smooth loss` increased from early low values to about `7.4` in the final segment

Interpretation:

The highyaw controller is useful evidence for low-level capability exploration, but it currently changes the high-level learning dynamics enough that it is worse as the main navigation baseline.

## Remaining Gap To Paper

This baseline is suitable for future Isaac Lab ablations and method comparisons, but should be described as an adapted baseline.

Do not claim paper-equivalent reproduction because:

- simulator changed from Isaac Gym to Isaac Lab
- low-level controller changed to RobotLab policy
- action/command clipping differs from the original SEA-Nav JIT low-level path
- exact Gym-difficulty hard-room success is still around `50-55%`, not near the optimistic adapted-protocol broad eval numbers
