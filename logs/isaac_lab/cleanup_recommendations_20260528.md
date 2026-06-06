# SEA-Nav Isaac Lab Cleanup Recommendations

Date: 2026-05-28
Policy used in this audit:

- no broad deletion during audit
- only classify first
- only propose deletion when the asset is clearly redundant, regenerable, and already summarized elsewhere

## Recommended Decision Table

| Path or Pattern | Recommendation | Why | Safe To Regenerate |
| --- | --- | --- | --- |
| `training/isaac_lab/` | keep | Active implementation | no |
| `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/` | keep | Only verified low-level policy used by current reproduction | no |
| `training/isaac_lab/low_level_policies/robotlab_go2_highyaw_20260604/` | archive keep | High-yaw low-level exploration asset; not promoted to main baseline | partially |
| `training/isaac_lab/eval_checkpoint_comparison.py` | keep | Reusable exact-eval comparison wrapper | yes |
| `training/rsl_rl/rsl_rl/{algorithms/ppo.py,modules/cbf_actor_critic.py,modules/cbf_lse_layer.py,runners/on_policy_runner.py,storage/rollout_storage.py}` | keep | Active high-level training stack | no |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/` | keep | Main validated run and checkpoint family | no |
| `logs/isaac_lab/baseline_manifest_20260606.md` | keep | Frozen adapted baseline definition | no |
| `logs/isaac_lab/highyaw_retrain_comparison_20260606/` | keep | Evidence that highyaw is not the main baseline | yes, but costly |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_highyaw_contract_ab/06_06_01-32-40_20260606_highyaw_ll_clip13_vx13_contract_ab_rerun_after_crash/` | archive keep | Completed 2500-iter highyaw exploration run; useful for diagnosis but not baseline | partially |
| `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-11-41/` and `05_28_10-15-02/` | keep | Best checkpoint comparison proof | difficult |
| `logs/isaac_lab/manual_reward_probe_hard_room_eval/05_28_10-20-34/` through `05_28_10-43-55/` | keep | Broad easy/medium/hard eval proof | yes, but costly |
| `logs/isaac_lab/low_level_tracking_probe/` | keep | Evidence for the active low-level contract and capabilities | yes, but moderately expensive |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_jointfix_gui_longrun` | deleted in this cleanup | Early joint-fix GUI long-run residue with only partial checkpoints and one event file | yes |
| `logs/isaac_lab/manual_reward_probe_turn_open`, `..._turn_wide`, `..._straight`, `..._turn` | deleted in this cleanup | Old simplified-scenario probe logs; the reusable script remains in `training/isaac_lab/manual_reward_probe.py` | yes |
| `logs/isaac_lab/hard_room_controller_diag` | deleted in this cleanup | Non-core blame-analysis traces after the user chose to retain only low-level contract evidence and final hard-room eval evidence | yes |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-04-32_*`, `05_27_22-19-09_*`, `05_27_22-26-47_*` | archive keep | Incomplete logging runs, but still important provenance for the curriculum chain | partially |
| `logs/isaac_lab/Go2_pos_rough_isaaclab_*` exploratory runs from 2026-05-25/26 | deleted after lesson archiving | Historical conclusions are preserved in `reproduction_audit_20260528.md`, so the heavy non-core payloads no longer need to remain | partially |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_150310/` | deleted in this audit | Empty aborted directory tree with no files | yes |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_150342/` | deleted in this cleanup | Trace-only easy attempt during live-record debugging | yes |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_150452/` | deleted in this cleanup | Trace-only easy attempt during live-record debugging | yes |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_150602/` | deleted in this cleanup | Trace-only easy attempt during live-record debugging | yes |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_154532/` | deleted in this cleanup | Trace-only easy attempt during live-record debugging | yes |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_141413/` | deleted in this cleanup | Historical OVD/MP4 output set | yes |
| `logs/isaac_lab/eval_case_mp4/2026_05_28_144958/` | deleted in this cleanup | Historical OVD/MP4 output set | yes |
| `logs/isaac_lab/ovd_recordings/2026_05_28_132630/` | deleted in this cleanup | OVD pipeline artifact | yes |
| `logs/isaac_lab/topdown_videos/model_3500_easy_isaac_gui_screen_topdown.mp4` | deleted in this cleanup | External screen recording | yes |
| `logs/isaac_lab/visual_progressive_cases/2026_05_28_154503/` | deleted in this cleanup | Recording-only case pack | yes |
| `logs/isaac_lab/recording_probes/*.md` | deleted in this cleanup | Recording-chain notes | yes |

## Recording-chain Cleanup Policy

This cleanup pass intentionally removes display-only and tooling-only recording outputs.

What remains protected:

1. The main high-level run directory and best checkpoints.
2. The frozen baseline manifest `logs/isaac_lab/baseline_manifest_20260606.md`.
3. The exact-difficulty eval evidence under `logs/isaac_lab/exact_difficulty_eval_20260601/` and highyaw comparison evidence under `logs/isaac_lab/highyaw_retrain_comparison_20260606/`.
4. The broad eval evidence under `logs/isaac_lab/manual_reward_probe_hard_room_eval/`.
5. The low-level contract evidence under `logs/isaac_lab/low_level_tracking_probe/`.
6. The main low-level policy directory under `training/isaac_lab/low_level_policies/robotlab_go2_flat_20260527/`.
7. The three original audit documents in `logs/isaac_lab/`.

After the latest cleanup, these are also the only retained probe/eval log families. Older simplified probe logs and controller-diagnostic traces are intentionally removed.

## 2026-06-06 Baseline Freeze Addendum

The current baseline is frozen as an adapted Isaac Lab + RobotLab SEA-Nav baseline.

Keep:

- primary checkpoint: `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2/model_3500.pt`
- secondary checkpoint: `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_curriculum/05_30_19-08-44_1024env_2500it_adaptive_bounded_lr/best_goal_hold.pt`
- baseline manifest: `logs/isaac_lab/baseline_manifest_20260606.md`
- exact eval evidence: `logs/isaac_lab/exact_difficulty_eval_20260601/`
- highyaw comparison evidence: `logs/isaac_lab/highyaw_retrain_comparison_20260606/`

Highyaw assets are archive-keep, not baseline-keep:

- `training/isaac_lab/low_level_policies/robotlab_go2_highyaw_20260604/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_highyaw_contract_ab/06_06_01-32-40_20260606_highyaw_ll_clip13_vx13_contract_ab_rerun_after_crash/`

Reason:

- `highyaw_model_2500` is faster and has fewer collision failures, but total exact success is `198/300`, below `222/300` for `model_3500.pt`.
- hard difficulty regresses to `31/100`, so it should not replace the frozen baseline.

## Historical Training Run Cleanup Record

This section records the follow-up cleanup batches that were executed after the initial audit.

### Batch A: highly likely historical garbage

Executed in the follow-up cleanup after user approval.

Deleted paths:

- `logs/isaac_lab/Go2_pos_rough_isaaclab/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_cpu_smoke/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_smoke/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_metrics/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_metrics_valid/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_metrics_verify/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_persistent_venv/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_idealpd_shorttrain/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_jointfix_shortcheck/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_semantic_replay_check/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_smoke/`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_lowlevel_shortcheck/`

Evidence used before deletion:

- total size about `225M`
- usually only `model_0.pt` or `model_1.pt`
- TensorBoard is absent, near-empty, or only `88` bytes to about `10KB`
- these are smoke, metrics, or short-check runs rather than substantive training history

### Batch B: archive-but-not-delete by default

Executed after archiving the useful lessons into `reproduction_audit_20260528.md`.

Deleted paths:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_exactroom_*`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_failure*`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen320`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen_native400*`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_full_goalstop045_*`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_full_hold_stabilize_lowent`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_full_repro_repaired`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardclassmix_*`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_harddirmix_goalstop045`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_hardgoalx_dirESE_goalstop045`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_onehardroom_*`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_diag`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_native_replaycheck`

Evidence used before deletion:

- total size about `1.7G`
- most are single-run targeted branches with only `4` to `6` checkpoint files
- TensorBoard files are tiny, usually around `6KB` to `40KB`
- the useful conclusions are now preserved in prose, so the heavy directory payload is no longer needed

### Batch C: important long-run history

Executed after archiving the useful lessons into `reproduction_audit_20260528.md`.

Deleted paths:

- `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_lowlevel_full`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_robotlab_contract_retrain`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_train_original_semantics`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_gpu_train_rtx5060ti`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_turn_open_replaycheck`
- `logs/isaac_lab/Go2_pos_rough_isaaclab_full_from_turnopen320_curriculumfix`

Evidence used before deletion:

- total size about `1.2G`
- these were the directories with the deepest checkpoint ladders or clearest migration milestones
- the useful migration conclusions are now preserved in prose, while the retained continuation bundle already keeps the final main run, broad eval evidence, low-level policy, and low-level tracking proof

## Guardrails Maintained In This Audit

- No core reproduction asset deletion was performed.
- The only already-cleaned artifact before this document set was `training/isaac_lab/__pycache__`, which was safe generated bytecode.
- One additional low-risk cleanup was executed in this audit:
  - deleted `logs/isaac_lab/eval_case_mp4/2026_05_28_150310/`
  - reason: it was an empty aborted directory tree with no files
- The current recording-chain cleanup also removes historical recording outputs that still contained `trace.jsonl`, `summary.json`, and `room.npy`, because the user explicitly confirmed that none of these recording-display assets should be retained.
- After the recording payloads were removed, the empty wrapper directories `logs/isaac_lab/eval_case_mp4/`, `logs/isaac_lab/ovd_recordings/`, `logs/isaac_lab/topdown_videos/`, `logs/isaac_lab/visual_progressive_cases/`, and `logs/isaac_lab/recording_probes/` were also removed.
- A later cleanup also removed the old `jointfix_gui_longrun`, `manual_reward_probe_turn_open`, `manual_reward_probe_turn_wide`, `manual_reward_probe_straight`, `manual_reward_probe_turn`, and `hard_room_controller_diag` payloads after the user confirmed that only `low_level_tracking_probe`, `manual_reward_probe_hard_room_eval`, and trained weights should remain from the probe/eval side.
