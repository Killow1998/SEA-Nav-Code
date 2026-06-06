# 2026-06-06 HighYaw Low-Level Retrain After Crash

## Reason

The 2026-06-05 highyaw low-level high-level training run was interrupted by a PC crash before reaching `max_iterations=2500`. It produced only early checkpoints in:

`logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_highyaw_contract_ab/06_05_15-23-37_20260605_highyaw_ll_clip13_vx13_contract_ab/`

Because the run did not produce a final `model_2500.pt`, it is treated as interrupted evidence only. This record starts a fresh rerun with the same intended training configuration and a new dated run name.

## Objective

Train a SEA-Nav high-level policy on Isaac Lab using the new RobotLab highyaw low-level controller, then compare it against the existing high-level baselines.

Primary question:

Does the new low-level controller improve exact medium/hard success by giving the high-level policy stronger yaw/lateral tracking without increasing contact termination?

## Rerun Configuration

The rerun keeps the 2026-06-05 experimental configuration:

- `num_envs=1024`
- `max_iterations=2500`
- `num_steps_per_env=48`
- `save_interval=500`
- `learning_rate=3e-4`
- `learning_rate_min=5e-5`
- `learning_rate_max=5e-4`
- `lr_schedule=adaptive`
- `low_level_controller=robotlab`
- `robotlab_low_level_policy=training/isaac_lab/low_level_policies/robotlab_go2_highyaw_20260604/policy.pt`
- `robotlab_command_clip=1.3`
- `action_reg_min=[-1.0, -1.0, -1.0]`
- `action_reg_max=[1.3, 1.0, 1.0]`
- `goal_stop_radius=0.45`
- `goal_stop_mode=zero`
- `obstacle_level=9`
- `training_loop=original`
- collision replay, observation noise, friction randomization, and base-mass randomization enabled

New run name:

`20260606_highyaw_ll_clip13_vx13_contract_ab_rerun_after_crash`

Experiment directory:

`logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_highyaw_contract_ab/`

## Launch Command

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=YES \
  /home/user/rl_redeploy/.venvs/sea-nav-ilab/bin/python -u training/isaac_lab/train.py \
  --num-envs 1024 \
  --max-iterations 2500 \
  --num-steps-per-env 48 \
  --save-interval 500 \
  --learning-rate 3e-4 \
  --learning-rate-min 5e-5 \
  --learning-rate-max 5e-4 \
  --lr-schedule adaptive \
  --entropy-coef 0.003 \
  --experiment-name Go2_pos_rough_isaaclab_robotlab_highyaw_contract_ab \
  --run-name 20260606_highyaw_ll_clip13_vx13_contract_ab_rerun_after_crash \
  --actuator-mode robotlab_dc \
  --low-level-controller robotlab \
  --robotlab-low-level-policy training/isaac_lab/low_level_policies/robotlab_go2_highyaw_20260604/policy.pt \
  --robotlab-command-clip 1.3 \
  --robot-asset-source native_go2 \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --terrain-rows 10 \
  --terrain-cols 10 \
  --obstacle-level 9 \
  --goal-stop-radius 0.45 \
  --goal-stop-mode zero \
  --action-reg-min -1.0 -1.0 -1.0 \
  --action-reg-max 1.3 1.0 1.0 \
  --reward-scale-stuck -12.0 \
  --reward-scale-progress 20.0 \
  --reward-scale-far-goal-stand -10.0 \
  --reward-scale-velo-dir 6.0 \
  --training-loop original
```

## Comparison Plan

After the rerun finishes, evaluate and compare:

- New highyaw rerun: best checkpoint among `best_reach.pt`, `best_goal_hold.pt`, `best_mean_reward.pt`, and final `model_2500.pt`.
- Current best bounded RobotLab baseline: `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_30_19-08-44_1024env_2500it_adaptive_bounded_lr/best_goal_hold.pt`.
- Old main baseline: `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/model_3500.pt`.

Required exact evaluation metrics:

- easy/medium/hard success rate
- contact/collision termination
- stand
- fall
- timeout
- average successful trajectory speed
- average arrival time
- minimum distance to goal for failures

## Status

Started as systemd user service:

`sea-nav-highyaw-20260606.service`

Actual run directory:

`logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_highyaw_contract_ab/06_06_01-32-40_20260606_highyaw_ll_clip13_vx13_contract_ab_rerun_after_crash/`

Start verification:

- 2026-06-06 01:32 CST: systemd user service launched.
- 2026-06-06 01:35 CST: service verified active with PID `111172`.
- 2026-06-06 01:35 CST: PPO loop already emitting episode metrics.
- 2026-06-06 01:35 CST: early `best_goal_hold.pt`, `best_reach.pt`, `best_mean_reward.pt`, `args.json`, `train_cfg.json`, and TensorBoard event file were present.
- 2026-06-06 01:37 CST: runtime resource guard applied with `MemoryMax=24G` and `MemorySwapMax=8G`; service remained active after the limit was applied.

TaskWatch:

- Workspace monitor installed at `.codex_monitor/`.
- Timer installed as `sea-nav-highyaw-20260606-monitor.timer`.
- First monitor dry run produced `.codex_monitor/reports/hourly_report_20260606_013535.md`.
- The training process is a transient systemd service and does not write a dedicated stdout log file. Post-training comparison must rely on checkpoints, TensorBoard events, explicit evaluation JSON/trace outputs, and `journalctl --user -u sea-nav-highyaw-20260606.service` if raw training stdout is needed.

Post-training evaluation automation:

- A one-off wait-and-eval wrapper was started as systemd user service `sea-nav-highyaw-20260606-posteval.service`.
- That wrapper was deleted after completion because the reusable entry point is now `training/isaac_lab/eval_checkpoint_comparison.py`.
- The post-eval service waited until `sea-nav-highyaw-20260606.service` was no longer active, then ran exact easy/medium/hard evaluation with `episodes=100`, `num_envs=16`, `seed=20260601`, and `max_steps=900`.
- Comparison output target: `logs/isaac_lab/highyaw_retrain_comparison_20260606/`.
- The comparison uses `training/isaac_lab/eval_checkpoint_comparison.py`, which wraps the existing `manual_reward_probe.py` rather than introducing a new evaluation environment.
- Highyaw checkpoints are evaluated with `robotlab_low_level_policy=training/isaac_lab/low_level_policies/robotlab_go2_highyaw_20260604/policy.pt` and `robotlab_command_clip=1.3`; baseline checkpoints use the default flat RobotLab low-level controller and `robotlab_command_clip=1.0`.

Completion gate:

Do not treat this run as complete until either `model_2500.pt` exists or the service has exited with a documented earlier stopping reason. After completion, run the comparison plan above before making a quality claim.
