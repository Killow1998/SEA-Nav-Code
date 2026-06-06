# SEA-Nav Isaac Lab Asset Index

Date: 2026-05-28
Rule: this index is for current audit and cleanup planning only. It is not a deletion script.

## Code and Config Assets

| Path | Size | Role | Status | Notes |
| --- | ---: | --- | --- | --- |
| `training/isaac_lab/` | `276K` | Isaac Lab training and eval env code | keep | Active code path for current reproduction |
| `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/` | `5.2M` | RobotLab low-level policy and exported config | keep | Contains `policy.pt`, `model_4999.pt`, `env.yaml`, `agent.yaml` |
| `training/isaac_lab/low_level_policies/robotlab_go2_highyaw_20260604/` | small | High-yaw RobotLab low-level policy export | archive keep | Explored on 2026-06-06; not the main SEA-Nav baseline low-level policy |
| `training/isaac_lab/eval_checkpoint_comparison.py` | tracked/new | exact eval comparison wrapper | keep | Reusable wrapper around `manual_reward_probe.py` for multi-checkpoint comparison |
| `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py` | tracked | High-level actor + CBF shield | keep | Active policy module |
| `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py` | tracked | LSE-CBF layer | keep | Includes local geometry-load compatibility fix |
| `training/rsl_rl/rsl_rl/algorithms/ppo.py` | tracked | PPO with adaptive LR and extra losses | keep | Active optimizer path |
| `training/rsl_rl/rsl_rl/runners/on_policy_runner.py` | tracked | checkpoint and TensorBoard logging | keep | Writes `best_*` and `model_*` files |
| `training/rsl_rl/rsl_rl/storage/rollout_storage.py` | tracked | rollout storage with `bad_masks` | keep | Active masking path |

## Primary Training Assets

| Path | Size | Role | Status | Notes |
| --- | ---: | --- | --- | --- |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/` | main share of `385M` group | Main validated high-level run | keep | Best current checkpoint lives here |
| `.../model_3500.pt` | included above | Best currently validated checkpoint | keep | Best hard-room evidence on 2026-05-28 |
| `.../best_mean_reward.pt` | included above | best reward checkpoint | keep | Useful for regression checks |
| `.../best_reach.pt` | included above | best reach checkpoint | keep | Useful for ablation/eval |
| `.../best_goal_hold.pt` | included above | best goal-hold checkpoint | keep | Useful for hold behavior comparison |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-04-32_2048env_2000it_adaptive_limited_curriculum/` | small within `385M` group | early curriculum attempt | archive keep | TensorBoard event is only `88` bytes |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-19-09_1024env_4000it_adaptive_limited_curriculum/` | small within `385M` group | early curriculum attempt | archive keep | TensorBoard event is only `88` bytes |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-26-47_256env_16000it_adaptive_limited_curriculum_equiv_steps/` | small within `385M` group | first equivalent-step run | archive keep | TensorBoard event is only `88` bytes |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_*` excluding robotlab curriculum | removed after follow-up cleanup | historical exploratory training runs | deleted after lesson archiving | The retained conclusions now live in `reproduction_audit_20260528.md`; only the robotlab curriculum family remains |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_highyaw_contract_ab/06_06_01-32-40_20260606_highyaw_ll_clip13_vx13_contract_ab_rerun_after_crash/` | current highyaw run | High-yaw low-level high-level retrain | archive keep | Full 2500-iter run completed; not promoted to main baseline because hard exact eval regressed |

## Historical Go2 Training Directory Breakdown

This section only classifies the `logs/isaac_lab/Go2_pos_rough_isaaclab*` family.

| Category | Paths or families | Evidence | Current handling |
| --- | --- | --- | --- |
| core active | `Go2_pos_rough_isaaclab_robotlab_contract_curriculum` | 4 subruns, but only the `...equiv_steps_v2` child has a valid `2.1M` TensorBoard event file; it is the active reproduction line | keep |
| important history | `Go2_pos_rough_isaaclab_robotlab_lowlevel_full`, `Go2_pos_rough_isaaclab_robotlab_contract_retrain`, `Go2_pos_rough_isaaclab_gpu_train_original_semantics`, `Go2_pos_rough_isaaclab_gpu_train_rtx5060ti`, `Go2_pos_rough_isaaclab_turn_open_replaycheck`, `Go2_pos_rough_isaaclab_full_from_turnopen320_curriculumfix` | these were the nontrivial long runs with the deepest checkpoint ladders or clearest migration milestones; their lessons are now preserved in `reproduction_audit_20260528.md`; total about `1.2G` before deletion | deleted after lesson archiving |
| targeted tuning history | `Go2_pos_rough_isaaclab_exactroom_*`, `Go2_pos_rough_isaaclab_failure*`, `Go2_pos_rough_isaaclab_full_from_turnopen320`, `Go2_pos_rough_isaaclab_full_from_turnopen_native400*`, `Go2_pos_rough_isaaclab_full_goalstop045_*`, `Go2_pos_rough_isaaclab_full_hold_stabilize_lowent`, `Go2_pos_rough_isaaclab_full_repro_repaired`, `Go2_pos_rough_isaaclab_hardclassmix_*`, `Go2_pos_rough_isaaclab_harddirmix_goalstop045`, `Go2_pos_rough_isaaclab_hardgoalx_dirESE_goalstop045`, `Go2_pos_rough_isaaclab_onehardroom_*`, `Go2_pos_rough_isaaclab_turn_open_diag`, `Go2_pos_rough_isaaclab_turn_open_native_replaycheck` | most are single-run experiment branches of `41M` to `71M`, usually with only `4` to `6` `.pt` files total and very small TensorBoard files (`~6KB` to `40KB`); total about `1.7G` | deleted after lesson archiving |
| highly likely historical garbage | `Go2_pos_rough_isaaclab`, `Go2_pos_rough_isaaclab_cpu_smoke`, `Go2_pos_rough_isaaclab_gpu_smoke`, `Go2_pos_rough_isaaclab_gpu_metrics`, `Go2_pos_rough_isaaclab_gpu_metrics_valid`, `Go2_pos_rough_isaaclab_gpu_metrics_verify`, `Go2_pos_rough_isaaclab_gpu_persistent_venv`, `Go2_pos_rough_isaaclab_idealpd_shorttrain`, `Go2_pos_rough_isaaclab_jointfix_shortcheck`, `Go2_pos_rough_isaaclab_semantic_replay_check`, `Go2_pos_rough_isaaclab_robotlab_contract_smoke`, `Go2_pos_rough_isaaclab_robotlab_lowlevel_shortcheck` | these are smoke, metrics, or short-check runs; they usually contain only `model_0.pt` or `model_1.pt`, sometimes no TensorBoard at all, otherwise `88` bytes to `10KB`; total about `225M` | deleted in follow-up cleanup |

## Evaluation Assets

| Path | Size | Role | Status | Notes |
| --- | ---: | --- | --- | --- |
| `logs/isaac_lab/manual_reward_probe_hard_room_eval/` | `22M` | broad eval history | keep | Contains current 2026-05-28 success evidence |
| `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-11-41/` | small | `model_3500.pt` hard-room 30-episode eval | keep | `28/30` success |
| `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-15-02/` | small | `model_16000.pt` hard-room 30-episode eval | keep | `24/30`, degraded |
| `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-20-34/` to `05_28_10-43-55/` | small | easy/medium/hard broad eval set | keep | Main public-facing performance evidence |
| `logs/isaac_lab/low_level_tracking_probe/` | `8.1M` | low-level contract verification history | keep | Needed to justify the active command range claim |
| `logs/isaac_lab/highyaw_retrain_comparison_20260606/` | small | 2026-06-06 exact comparison across highyaw and baselines | keep | Confirms highyaw is exploratory and not the main baseline |

## Removed During This Audit

| Path | Previous Size | Why Removed | Safety Basis |
| --- | ---: | --- | --- |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_150310/` | `16K` | empty aborted directory tree | it contained no files at all, only empty directories |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_150342/` | `420K` | trace-only recording debug residue | user explicitly dropped recording-chain debug leftovers |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_150452/` | `420K` | trace-only recording debug residue | user explicitly dropped recording-chain debug leftovers |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_150602/` | `420K` | trace-only recording debug residue | user explicitly dropped recording-chain debug leftovers |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_154532/` | `540K` | trace-only recording debug residue | user explicitly dropped recording-chain debug leftovers |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_141413/` | `126M` | historical OVD/MP4 output set | user explicitly dropped historical recording outputs |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_144958/` | `125M` | historical OVD/MP4 output set | user explicitly dropped historical recording outputs |
| `logs/isaac_lab/ovd_recordings/2026_05_28_132630/` | `12M` | OVD pipeline artifact | display/tooling only, not core reproduction evidence |
| `logs/isaac_lab/topdown_videos/model_3500_easy_isaac_gui_screen_topdown.mp4` | `680K` | external screen recording | user explicitly dropped external recordings |
| `logs/isaac_lab/visual_progressive_cases/2026_05_28_154503/` | `2.4M` | recording-only case pack | display-only asset, not retained as core evidence |
| `logs/isaac_lab/recording_probes/*.md` | `24K` total | recording-chain notes | recording-only notes removed with the display pipeline |
| `logs/isaac_lab/Go2_pos_rough_isaaclab`, `..._cpu_smoke`, `..._gpu_smoke`, `..._gpu_metrics*`, `..._gpu_persistent_venv`, `..._idealpd_shorttrain`, `..._jointfix_shortcheck`, `..._semantic_replay_check`, `..._robotlab_contract_smoke`, `..._robotlab_lowlevel_shortcheck` | `225M` total | smoke / metrics / short-check training history | user approved removing the clearly non-core short-run group |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_exactroom_*`, `..._failure*`, `..._full_from_turnopen320`, `..._full_from_turnopen_native400*`, `..._full_goalstop045_*`, `..._full_hold_stabilize_lowent`, `..._full_repro_repaired`, `..._hardclassmix_*`, `..._harddirmix_goalstop045`, `..._hardgoalx_dirESE_goalstop045`, `..._onehardroom_*`, `..._turn_open_diag`, `..._turn_open_native_replaycheck` | `1.7G` total | targeted tuning history | lesson summary preserved in `reproduction_audit_20260528.md`, so the directories were removed |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_lowlevel_full`, `..._robotlab_contract_retrain`, `..._gpu_train_original_semantics`, `..._gpu_train_rtx5060ti`, `..._turn_open_replaycheck`, `..._full_from_turnopen320_curriculumfix` | `1.2G` total | long-run migration history | lesson summary preserved in `reproduction_audit_20260528.md`, so the directories were removed |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_jointfix_gui_longrun` | `91M` | early joint-fix GUI long-run residue | it only preserved a partial checkpoint ladder plus one TensorBoard file, and it is fully superseded by the retained RobotLab curriculum run |
| `logs/isaac_lab/manual_reward_probe_turn_open`, `..._turn_wide`, `..._straight`, `..._turn` | `12M` total | simplified scenario probe history | the reusable probe script remains in `training/isaac_lab/manual_reward_probe.py`, but these old probe logs are no longer part of the retained bundle |
| `logs/isaac_lab/hard_room_controller_diag` | `304K` | controller blame-analysis traces | the user chose to retain only the low-level contract probe and the final hard-room eval evidence |

After the payloads above were removed, the empty wrapper directories `logs/isaac_lab/eval_case_mp4/`, `logs/isaac_lab/ovd_recordings/`, `logs/isaac_lab/topdown_videos/`, `logs/isaac_lab/visual_progressive_cases/`, and `logs/isaac_lab/recording_probes/` were also removed.

## Notes and Reports

| Path | Size | Role | Status | Notes |
| --- | ---: | --- | --- | --- |
| `logs/isaac_lab/reproduction_audit_20260528.md` | text | main reproduction report | keep | Core retained audit document |
| `logs/isaac_lab/asset_index_20260528.md` | text | asset map and retained bundle | keep | Core retained audit document |
| `logs/isaac_lab/cleanup_recommendations_20260528.md` | text | cleanup rationale and deletion record | keep | Core retained audit document |
| `logs/isaac_lab/baseline_manifest_20260606.md` | text | frozen adapted baseline manifest | keep | Primary entry point for future baseline usage |

## Weights Quick Index

- Current high-level best checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/model_3500.pt`
- Secondary high-level baseline checkpoint:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_30_19-08-44_1024env_2500it_adaptive_bounded_lr/best_goal_hold.pt`
- Current low-level RobotLab policy:
  - `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/policy.pt`
- Exploratory highyaw low-level policy:
  - `training/isaac_lab/low_level_policies/robotlab_go2_highyaw_20260604/policy.pt`

## Command Quick Index

- Training entry:
  - `training/isaac_lab/train.py`
  - equivalent reconstructed command is documented in `reproduction_audit_20260528.md`
- Broad eval entry:
  - `training/isaac_lab/manual_reward_probe.py`
  - current best checkpoint path:
    - `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/model_3500.pt`

## Minimum Reproduction Bundle

If another person only needs to continue current easy/medium/hard evaluation work, the minimum retained bundle is:

- code:
  - `training/isaac_lab/train.py`
  - `training/isaac_lab/sea_nav_env.py`
  - `training/isaac_lab/manual_reward_probe.py`
  - `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py`
  - `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py`
  - `training/rsl_rl/rsl_rl/algorithms/ppo.py`
- high-level run assets:
  - `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/`
- low-level policy assets:
  - `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/`
- eval evidence:
  - `logs/isaac_lab/baseline_manifest_20260606.md`
  - `logs/isaac_lab/exact_difficulty_eval_20260601/model_3500/`
  - `logs/isaac_lab/highyaw_retrain_comparison_20260606/`
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-11-41/`
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-15-02/`
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-20-34/`
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-28-54/`
  - `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-37-31/`
- low-level contract evidence:
  - `logs/isaac_lab/low_level_tracking_probe/`
- audit docs:
  - `logs/isaac_lab/reproduction_audit_20260528.md`
  - `logs/isaac_lab/asset_index_20260528.md`
  - `logs/isaac_lab/cleanup_recommendations_20260528.md`

Recording-chain outputs are intentionally not part of the retained minimum bundle.

The broad `Go2_pos_rough_isaaclab*` history outside the active RobotLab curriculum run is also not part of the minimum bundle.
