# SEA-Nav Paper-to-Code Drift Review

Date: 2026-05-26  
Scope: paper/project claims vs open-source Isaac Gym code vs current Isaac Lab migration in this repo

Current note:

- this document is the historical drift-analysis record from the migration phase
- the current retained reproduction verdict, asset map, and cleanup record now live under:
  - `logs/isaac_lab/reproduction_audit_20260528.md`
  - `logs/isaac_lab/asset_index_20260528.md`
  - `logs/isaac_lab/cleanup_recommendations_20260528.md`

## Sources reviewed

- Paper: `arXiv:2603.09460v1`, "SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments"
- Project page: `https://11chens.github.io/sea-nav/`
- Open-source Isaac Gym path in this repo:
  - `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py`
  - `training/legged_gym/legged_gym/envs/base/legged_robot.py`
  - `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py`
  - `training/legged_gym/legged_gym/scripts/play.py`
  - `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py`
  - `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py`
  - `training/rsl_rl/rsl_rl/algorithms/ppo.py`
- Current Isaac Lab migration path in this repo:
  - `training/isaac_lab/train.py`
  - `training/isaac_lab/sea_nav_env.py`
  - `training/isaac_lab/manual_reward_probe.py`

## Executive verdict

This repo is no longer failing because of a single missing port or a single obvious bug. The current state is:

- Many early migration drifts have already been repaired:
  - `bad_masks` / reset semantics
  - joint order mapping
  - collision replay
  - observation noise
  - terrain and goal curriculum
- The actuator execution path still differs structurally from the paper/open-source Gym path:
  - paper and open-source Isaac Gym semantics are low-level controller -> torque PD -> `set_dof_actuation_force_tensor`
  - current Isaac Lab default semantics are low-level controller -> joint position targets -> Isaac Lab actuator abstraction
- However, after adding a repo-side `gym_torque` mode and testing it on the host GPU, actuator drift no longer looks like the dominant explanation for the current performance ceiling:
  - `ideal_pd` and `gym_torque` are effectively identical on the negative-turn low-level tracking probe
  - the current strongest hard-room gate remains exactly `24/30` when the same checkpoint is evaluated under `gym_torque`
- The previously missing domain-randomization block is now also ported into the repo:
  - per-env friction bucket randomization
  - per-env base-mass perturbation
  - the missing `ang_vel_xy` reward term
- But this domain-randomization closure also did not improve the current winner:
  - after a same-family continuation from the `24/30` winner, `best_reach.pt` regressed to `16/30`
  - `best_mean_reward.pt` recovered only to `23/30`
- This work also exposed one real simulator-level incompatibility:
  - the original Gym config uses `friction_range = [-0.2, 1.25]`
  - Isaac Sim / PhysX does not allow negative dynamic friction
  - so the migrated repo must clamp the lower bound to `0.0` to stay PhysX-safe
- Some mismatches are not Isaac Lab migration bugs at all. They already exist between the paper text/appendix and the open-source Isaac Gym code:
  - reward weights in the appendix do not match the open-source config
  - the appendix `alpha_min` / shield loss details do not match the open-source PPO implementation
  - the paper presents the safe command as a 3D body-velocity command, but the open-source shield actually only modifies planar `vx, vy` and passes `yaw` through unchanged
- The current strongest local result is not a paper-pure reproduction:
  - strongest verified checkpoint is `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_goalstop045/05_26_08-10-51_80it_1e-5_ent1e-3_p05_opt/best_reach.pt`
  - its strongest gates are `24/30` and `40/50`
  - those gates were run with `policy_stop_radius = 0.45`, which is a repo-local prior not present in the paper or original Isaac Gym default path

## Current best local reproduction status

The current strongest gate files are:

- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-25-42/summary.json`
- `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_26_08-31-55/summary.json`

Important details from those gate files:

- `success_count = 24 / 30` and `40 / 50`
- `collision_free_success_count = 24 / 30` and `40 / 50`
- `disable_contact_termination = true`
- `stay_steps = 500`
- `policy_stop_radius = 0.45`

This matters because:

- the `disable_contact_termination = true` and `stay_steps = 500` settings are consistent with the original Isaac Gym `play.py` eval semantics
- but `policy_stop_radius = 0.45` is not part of the paper method or the original Isaac Gym default evaluation path

So the best local result is meaningful, but it is not yet a strict paper-faithful win.

For comparison, the public project page claims:

- a one-take demo with `10/10` successful trials and zero collisions
- visual results after roughly `30` minutes and `60` minutes of training

The current local winner is below that public bar, and it also depends on the non-paper stop prior described above.

## Drift matrix

### 1. Training platform and reproduction conditions

Paper:

- training is explicitly described as being done on Isaac Gym
- the paper claims minute-level training, specifically "tens of minutes on a single RTX 4090 GPU"

Open-source Isaac Gym code:

- README remains Gym-first and documents only the Gym train/play entrypoints: `README.md:44-59`
- default Gym env count is `2048`: `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:57`

Current Isaac Lab migration:

- Isaac Lab train entrypoint default is `--num-envs 256`: `training/isaac_lab/train.py:241-251`
- current work was run on a different GPU generation and a different simulator stack

Verdict:

- major reproduction-condition drift
- this alone does not prove the algorithm is wrong, but it makes "paper-level minute training" non-comparable by default

### 2. Observation and timing interface

Paper:

- observation includes base linear velocity, base angular velocity, projected gravity, 2D local goal, and `41` LiDAR rays
- history length is `10`
- exteroception is updated at `10 Hz`
- proprioception is maintained at `50 Hz`

Open-source Isaac Gym code:

- rays are `41`, with `theta_start = -2*pi/3`, `theta_end = 2*pi/3`, `theta_step = pi/30`: `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:182-190`
- history buffers and delayed exteroception are implemented in `legged_robot_pos.py`: `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:649-715`

Current Isaac Lab migration:

- `41` rays, `10`-frame history, delayed exteroception, and the same `2*pi/3` FOV are present: `training/isaac_lab/sea_nav_env.py:306-343`, `training/isaac_lab/sea_nav_env.py:528-533`, `training/isaac_lab/sea_nav_env.py:841-846`, `training/isaac_lab/sea_nav_env.py:882-889`

Verdict:

- largely aligned

### 3. Low-level controller and actuator semantics

Paper:

- the high-level policy outputs safe body velocity commands
- these commands are sent to a pre-trained low-level locomotion controller and converted into target joint torques

Open-source Isaac Gym code:

- low-level controller outputs are converted to torques: `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:164-179`
- the final actuator write is torque-based: `training/legged_gym/legged_gym/envs/base/legged_robot.py:96-103`, `training/legged_gym/legged_gym/envs/base/legged_robot.py:351-374`

Current Isaac Lab migration:

- default actuator mode is `ideal_pd`: `training/isaac_lab/train.py:243-246`
- the migrated env builds an `IdealPDActuatorCfg` or `ImplicitActuatorCfg`, not a direct torque write path: `training/isaac_lab/sea_nav_env.py:361-417`
- the migrated env applies `set_joint_position_target(...)`: `training/isaac_lab/sea_nav_env.py:736-741`

Verdict:

- formal migration drift still exists
- but after the new host-GPU A/B, it is no longer the leading verified explanation for the remaining behavior gap

### 4. Adaptive Collision-State Initialization (ACSI) / collision replay

Paper:

- collision-state replay before collision is a core method contribution
- replay probability increases with curriculum progress

Open-source Isaac Gym code:

- replay config exists in the Go2 config: `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:62-67`
- history replay and early replay reset are implemented in `legged_robot_pos.py`: `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:310-374`, `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:583-589`

Current Isaac Lab migration:

- replay config is present: `training/isaac_lab/sea_nav_env.py:339-343`
- replay buffers, early replay probability, replay reset, and replay-on-reset are all implemented: `training/isaac_lab/sea_nav_env.py:660-677`, `training/isaac_lab/sea_nav_env.py:996-1038`, `training/isaac_lab/sea_nav_env.py:1060-1105`

Verdict:

- aligned at the algorithm level
- this was a real migration drift earlier, but it is now ported back in

### 5. Goal-stay / hold-at-goal mechanism

Paper:

- training does not end immediately at first goal contact
- the robot must stay near goal for a period to increase sparse goal-reaching experience

Open-source Isaac Gym code:

- `goal_reached_time = 150`, `stay_time = 150`: `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:47-49`
- hold timer and stand-still timer are implemented in `legged_robot_pos.py`: `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:614-623`

Current Isaac Lab migration:

- `goal_reached_steps = 150`, `stay_steps = 150`: `training/isaac_lab/sea_nav_env.py:309-311`
- the same timer logic is implemented in `_get_dones()`: `training/isaac_lab/sea_nav_env.py:1073-1085`

Verdict:

- aligned

### 6. CBF shield structure

Paper:

- the actor outputs nominal command `u_bar` and adaptive gain `alpha`
- the differentiable LSE-CBF shield computes the final safe command `u_s`
- the layer uses smooth LSE aggregation and damping `epsilon_d`

Open-source Isaac Gym code:

- actor outputs `u_bar` and `alpha`, then applies the CBF layer: `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py:126-155`
- the CBF layer uses LSE aggregation and damping factor `1.0`: `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:27-66`

Current Isaac Lab migration:

- it started from the same shared `rsl_rl` implementation
- the local Isaac Lab fork now repairs the internal shield geometry to `240`
  degrees and forces checkpoint loads to rebuild `ray_unit_vectors` from config
  rather than trusting older saved `180`-degree buffers:
  `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:6-41`

Verdict:

- there are still two paper-to-open-source caveats:
  - the paper describes the final safe command as `[vx,s, vy,s, wz,s]`, while the implementation only shields planar `vx, vy` and passes `yaw` through unchanged: `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:33-35`, `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:62-64`
  - the original release had rays spanning `240` degrees in the env while the
    shield internally built unit vectors over `180` degrees:
    `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:182-190`,
    `training/isaac_lab/sea_nav_env.py:532`,
    `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:10-25`

The second item was a high-value source-level mismatch. It exists in the
open-source Isaac Gym code already, so it is not an Isaac Lab migration bug.
I repaired it locally and then verified two things:

- direct host-GPU re-eval of the current strongest winner still stayed at
  `24/30`, though mean closest-approach distance improved
- an isolated continuation with only the `240`-degree geometry repair regressed
  to `18/30`

So this mismatch is real and worth documenting, but it no longer looks like the
main explanation for the current ceiling.

### 7. Kinematic regularization and shield intervention losses

Paper:

- `L_total = L_PPO + lambda_shield * L_shield + lambda_reg * L_reg`
- appendix values list `lambda_shield = 0.1`, `lambda_reg = 1.0`, `lambda_pi = 0.05`, `lambda_V = 0.005`, `alpha_min = 0.1`

Open-source Isaac Gym code:

- the smoothness loss composes actor and critic terms as `1.0` and `0.1`, and then the caller multiplies smoothness by `0.05`; this reproduces `0.05` for the actor term and `0.005` for the critic term: `training/rsl_rl/rsl_rl/algorithms/ppo.py:104-130`, `training/rsl_rl/rsl_rl/algorithms/ppo.py:234-236`
- the range loss bounds are `[-0.5, -0.8, -1.0]` to `[1.7, 0.8, 1.0]`: `training/rsl_rl/rsl_rl/algorithms/ppo.py:230-232`
- intervention loss is weighted by `0.1`: `training/rsl_rl/rsl_rl/algorithms/ppo.py:244-248`
- but `alpha_loss` is applied separately with `alpha_min = 1.0`, not `0.1`: `training/rsl_rl/rsl_rl/algorithms/ppo.py:97-102`, `training/rsl_rl/rsl_rl/algorithms/ppo.py:238-242`

Current Isaac Lab migration:

- it uses the same shared `rsl_rl` implementation

Verdict:

- current Isaac Lab is aligned with the open-source code
- but the open-source code is not fully aligned with the appendix hyperparameter statement

### 8. Reward design

Paper appendix:

- `r_term = -100`
- `r_reach = +10`
- `r_velo = +15`
- `r_clear = +15`
- `r_stuck = -5`
- `r_coll = -4`
- `r_omega = -0.05`

Open-source Isaac Gym code:

- default scales are `termination=-100`, `collision=-4`, `close_obst_vel=5`, `stuck=-5`, `velo_dir=4`, `reach_pos_target_tight=10`, `ang_vel_xy=-0.05`: `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:214-223`
- reward function structure for `reach`, `velo_dir`, `close_obst_vel`, `stuck`, and `collision` is implemented in `legged_robot_pos.py`: `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:970-1030`

Current Isaac Lab migration:

- default scales are now `termination=-100`, `collision=-4`, `close_obst_vel=5`, `stuck=-5`, `velo_dir=4`, `reach_pos_target_tight=10`, `ang_vel_xy=-0.05`: `training/isaac_lab/sea_nav_env.py`
- the migrated reward functions match the open-source Gym formulas for those terms
- a short continuation with the restored `ang_vel_xy` term did not beat the current `24/30` winner

Verdict:

- paper vs open-source Gym: significant reward-scale mismatch remains
- open-source Gym vs current Isaac Lab: the dropped `ang_vel_xy` term has now been restored
- restoring it did not move the current hard-room ceiling upward

This means "reward drift" must be separated into two layers:

- some reward differences are already present in the source release
- one additional reward term was then lost in the Isaac Lab migration

### 9. Domain randomization

Paper appendix:

- ray delay: `U(40, 80) ms`
- gravity noise: `U(-0.05, 0.05)`
- linear velocity noise: `U(-0.1, 0.1)`
- angular velocity noise: `U(-0.1, 0.1)`
- friction coefficient factor
- base mass perturbation

Open-source Isaac Gym code:

- delay, noise, friction randomization, and base mass randomization are configured in `go2_pos_config.py`: `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:159-180`, `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:205-212`
- friction and base mass randomization are actually applied in `legged_robot.py`: `training/legged_gym/legged_gym/envs/base/legged_robot.py:258-281`, `training/legged_gym/legged_gym/envs/base/legged_robot.py:312-316`

Current Isaac Lab migration:

- delay and observation noise are present
- per-env friction bucket randomization is now present in the repo-side env
- per-env base-mass perturbation is now present in the repo-side env
- eval tooling disables friction randomization and uses zero added mass to stay closer to original `play.py`

Verdict:

- this migration gap is now closed at the repo level
- but the literal Gym friction lower bound `-0.2` is not PhysX-valid and had to be clamped to `0.0`
- after closing this gap, a same-family continuation still failed to beat the current `24/30` winner

### 10. Terrain and goal curriculum

Paper:

- replay probability is coupled to curriculum progress
- training uses dense rooms with curriculum over difficulty

Open-source Isaac Gym code:

- `goal_levels` and `terrain_levels` are updated together in `_update_terrain_curriculum(...)`: `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:423-445`
- early replay probability uses `goal_levels`: `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:583-589`
- default max initial terrain level is low: `training/legged_gym/legged_gym/envs/base/legged_robot_config.py:68`, `training/legged_gym/legged_gym/envs/base/legged_robot.py:776-782`

Current Isaac Lab migration:

- terrain generator curriculum is enabled: `training/isaac_lab/sea_nav_env.py:477-489`
- max initial terrain level is restored to `min(..., 2)`: `training/isaac_lab/sea_nav_env.py:498`
- `goal_levels` drive early replay probability and terrain-origin updates: `training/isaac_lab/sea_nav_env.py:618`, `training/isaac_lab/sea_nav_env.py:996-1011`

Verdict:

- largely aligned after repair

### 11. Evaluation semantics

Paper and project page:

- the public story emphasizes one-take deployment and fast training
- the paper itself also notes a limitation: the algorithm can still get stuck in complex mazes and dead ends

Open-source Isaac Gym eval path:

- `play.py` disables collision replay, disables noise, removes contact termination, and uses `stay_time = 500`: `training/legged_gym/legged_gym/scripts/play.py:57-79`

Current Isaac Lab eval path:

- `manual_reward_probe.py` can emulate these eval semantics: `training/isaac_lab/manual_reward_probe.py:59-60`, `training/isaac_lab/manual_reward_probe.py:842-848`

Verdict:

- the stricter early eval mismatch around replay/contact termination has already been repaired
- current remaining failures are less likely to be caused by "overly harsh eval semantics"

### 12. Non-paper additions currently present in the Isaac Lab migration

These are real code paths now in the repo, but they are not paper mechanisms:

- `goal_stop_radius` / `goal_stop_mode`: `training/isaac_lab/sea_nav_env.py:312-313`, `training/isaac_lab/sea_nav_env.py:725-733`
- `turn_yaw_threshold` / forward-floor turn priors: `training/isaac_lab/sea_nav_env.py:314-316`, `training/isaac_lab/sea_nav_env.py:715-724`
- exact-room replay and `room.npy`-based continuation tooling in the manual probes and train entrypoint

Important note:

- these additions are default-off in code
- but the strongest local checkpoint and gate currently come from the `goalstop045` family, not from a paper-pure branch

## Ranked drift list

### Highest-confidence unresolved drifts

1. The current strongest local success still relies on a non-paper near-goal stop prior.
2. The remaining hard-room tail is still dominated by low-speed negative reorientation weakness and a smaller set of hard-case path failures.
3. Replay or targeted continuation still causes forgetting conflicts.
4. The released shield still only modifies planar velocities and leaves `yaw` unchanged.

### Source-level mismatches already present before Isaac Lab migration

1. Paper appendix reward weights do not match the open-source Gym config.
2. Paper appendix `alpha_min` / shield-loss description does not match the open-source PPO code.
3. The shield layer only modifies planar velocities and leaves `yaw` unchanged.
4. The original release used `240`-degree env rays with `180`-degree internal shield geometry; the local Isaac Lab fork now patches this to `240`, but the isolated continuation still did not beat `24/30`.

### Lower-priority or already-closed migration drifts

1. `bad_masks` / reset semantics
2. joint order mapping
3. collision replay semantics
4. observation noise
5. terrain and goal curriculum
6. eval replay / contact-termination mismatch

## What this review implies

The current ceiling is probably not explained by one more easy porting fix.

The most likely remaining explanation is a combination of:

- behavior-level issues already seen in host probes and gates:
  - low-level negative low-speed reorientation weakness
  - a smaller set of hard-case path-selection failures
  - forgetting when replaying or targeting the tail
- several source-level inconsistencies already present in the released Gym code:
  - reward-weight mismatch
  - shield-loss hyperparameter mismatch
  - shield only passes `yaw` through unchanged
- and one local experimental crutch:
  - the current strongest result depends on a near-goal stop prior that is not part of the paper method

## Recommended next work order

1. Decide whether to pursue a strict paper-faithful line or a best-local-performance line.
   - strict paper-faithful line: remove `goal_stop_radius` prior from the success criterion
   - best-local-performance line: keep it, but stop calling the result paper-near reproduction

2. If strict paper fidelity is still the target, audit the source-level mismatches too.
   - especially the fact that the released shield still leaves `yaw` unshielded
   - and the appendix reward / hyperparameter mismatch

3. If the goal is better local performance, prioritize behavior-level fixes over more drift closure.
   - negative low-speed reorientation
   - hard-case path-selection tail
   - anti-forgetting continuation strategies

## Bottom line

The repo is no longer blocked by setup or obvious missing Isaac Lab ports. The remaining gap is now a mix of:

- behavior-level control weaknesses
- replay/continuation forgetting
- several paper-to-open-source source-level mismatches
- and a current best result that depends on a non-paper stop prior

So the answer to "is this only because of the Isaac Sim transfer" is no.

The transfer still matters, but the latest evidence says it is not enough to explain the ceiling by itself. After host-GPU actuator A/B, after restoring repo-side friction/mass randomization plus `ang_vel_xy`, and after locally repairing the released `240`-degree env vs `180`-degree shield-geometry mismatch, the current strongest hard-room gate still stays below the paper bar and still depends on a non-paper stop prior.

The latest exact-room step traces also make the behavioral blocker more concrete:

- one reproduced tail case never gets close at all (`min_distance ~= 5.01m` after
  `2000` steps), so it is a far-field path / trajectory-drift failure
- another reproduced tail case gets much closer (`min_distance ~= 0.915m`) but
  still never reaches, and at the closest point it is already issuing a
  low-forward negative-yaw command under tight clearance

Together with the failed boosted-negative-yaw probe, this strongly suggests that
the current ceiling is a combination of:

- some high-level tail cases that choose poor trajectories early
- and a low-level weakness in near-goal negative reorientation that is **not**
  fixed by simply increasing negative yaw magnitude

The latest exact-room `case-trace` A/B makes that last point stricter:

- adding a simple negative-turn forward-support prior
  (`policy_turn_forward_floor_neg = 0.35`) does **not** rescue the reproduced
  near-goal zero-reach case
- and it still does not convert the reproduced far-field drift case into first
  reach either

So the remaining ceiling is no longer well explained by any small command-level
prior. It is better described as a genuine behavior-level limitation of the
current migrated stack.

Batch replay of the full `24/30` exact-room tail strengthens this further:

- the original `6` failures do not behave as six equally stable geometric
  failures
- `1` of them replays as success, indicating a small rollout-level instability
  band
- the remaining stable failures split into:
  - `3` near-goal zero-reach cases
  - `2` far-field path / trajectory-drift cases

One more targeted continuation makes the blocker statement even stricter:

- replaying only the `2` stable far-field exact-room cases at low probability
  (`p = 0.25`) produced:
  - `best_reach = 17/30`
  - `best_mean_reward = 21/30`
  - `best_goal_hold = 22/30`
- all of these remain below the same-room `24/30` baseline

So the remaining gap is not well explained by one more easy replay subset
either. Even when the continuation is narrowed to the reproduced far-field
subgroup, the best fresh gate still regresses.

The next anti-forgetting test also failed:

- I added a minimal actor-side policy anchor against the current winner
  checkpoint
- then reran the exact-room failure replay branch with:
  - the `6` exact-room failures replayed at `p = 0.5`
  - `policy_anchor_coef = 0.05`
- same-room fresh gates regressed to:
  - `best_goal_hold = 14/30`
  - `best_reach = 14/30`
  - `best_mean_reward = 18/30`

So a small reference-action anchor is not enough to solve the forgetting
problem either.

One stronger preservation variant also failed:

- freezing the actor trunk (`encoder + backbone`) while replaying the exact-room
  failure set still produced only:
  - `best_reach = 16/30`
  - `best_mean_reward = 19/30`
- so even a more structural “preserve representation, only tune heads” variant
  remains below the `24/30` baseline

This means the remaining gap is not well explained by easy replay tuning or by
lightweight preservation variants.

The strongest final evaluation evidence is now a multi-room broad hard-room
audit of the current best local winner:

- with the non-paper stop prior:
  - seed 1: `24/30`
  - seed 2: `25/30`
  - seed 3: `8/30`
  - aggregate: `57/90`
- without the stop prior:
  - seed 1: `2/30`
  - seed 2: `1/30`
  - seed 3: `1/30`
  - aggregate: `4/90`

So two things are now directly established:

1. the current best local winner is not room-robust even with the stop prior
2. its performance depends heavily on that non-paper prior

That is strong enough to support the final negative conclusion: under the
current hardware and current migrated stack, this method has **not** been shown
to approach the paper's demonstration effect in Isaac Lab.
