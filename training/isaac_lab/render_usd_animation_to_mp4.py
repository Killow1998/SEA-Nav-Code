from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_isaaclab_imports():
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import isaaclab

    source_root = Path(isaaclab.__file__).resolve().parent / "source"
    isaaclab_pkg_root = source_root / "isaaclab" / "isaaclab"
    if isaaclab_pkg_root.exists() and str(isaaclab_pkg_root) not in isaaclab.__path__:
        isaaclab.__path__.append(str(isaaclab_pkg_root))
    return isaaclab


_ensure_isaaclab_imports()
from isaaclab.app import AppLauncher


def _parse_args():
    parser = argparse.ArgumentParser(description="Render a USD animation to an MP4 using Isaac Sim viewport capture.")
    parser.add_argument("--usd", required=True, help="Input USD/USDA/USDC animation stage.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--camera-height", type=float, default=16.0)
    parser.add_argument("--camera-target-x", type=float, default=0.0)
    parser.add_argument("--camera-target-y", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _wait_for_stage_load(app, usd_context, timeout_s: float):
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        app.update()
        stage = usd_context.get_stage()
        if stage is not None and stage.GetDefaultPrim().IsValid():
            return stage
        if stage is not None and stage.GetPrimAtPath("/World").IsValid():
            return stage
    raise TimeoutError("Timed out waiting for USD stage to load")


def _set_topdown_view(camera_height: float, target_x: float, target_y: float):
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport is available for MP4 capture")
    eye = np.array([target_x, target_y - 0.01, camera_height], dtype=float)
    target = np.array([target_x, target_y, 0.0], dtype=float)
    set_camera_view(eye, target, camera_prim_path="/OmniverseKit_Persp", viewport_api=viewport)
    viewport.camera_path = "/OmniverseKit_Persp"
    return viewport


def _capture_mp4(args, app):
    import carb
    import omni.kit.app
    import omni.timeline
    import omni.usd

    usd_path = Path(args.usd).resolve()
    output_path = Path(args.output).resolve()
    print(f"[MP4] opening_stage={usd_path}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    usd_context = omni.usd.get_context()
    if not usd_context.open_stage(str(usd_path)):
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    stage = _wait_for_stage_load(app, usd_context, args.timeout_s)
    print(f"[MP4] stage_loaded={stage.GetRootLayer().identifier}", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(args.start_time)
    timeline.set_end_time(args.end_time)
    timeline.set_current_time(args.start_time)
    print(
        "[MP4] timeline "
        f"start={args.start_time} end={args.end_time} fps={args.fps}",
        flush=True,
    )

    _set_topdown_view(args.camera_height, args.camera_target_x, args.camera_target_y)
    for _ in range(10):
        app.update()

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("omni.kit.capture.viewport", True)
    for _ in range(5):
        app.update()
    from omni.kit.capture.viewport import (
        CaptureExtension,
        CaptureOptions,
        CaptureRangeType,
        CaptureRenderPreset,
    )

    capture = CaptureExtension().get_instance()
    capture.show_default_progress_window = False
    capture.options = CaptureOptions(
        camera="/OmniverseKit_Persp",
        range_type=CaptureRangeType.SECONDS,
        start_time=args.start_time,
        end_time=args.end_time,
        fps=args.fps,
        animation_fps=float(args.fps),
        res_width=args.video_width,
        res_height=args.video_height,
        render_preset=CaptureRenderPreset.RAY_TRACE,
        spp_per_iteration=1,
        path_trace_spp=1,
        output_folder=str(output_path.parent),
        file_name=output_path.stem,
        file_type=".mp4",
        overwrite_existing_frames=True,
        app_level_capture=False,
    )

    if not capture.start():
        raise RuntimeError("Isaac viewport MP4 capture failed to start")
    print(
        "[MP4] capture_started "
        f"output={output_path} resolution={args.video_width}x{args.video_height}",
        flush=True,
    )

    start = time.monotonic()
    last_progress = -1.0
    while not capture.done:
        app.update()
        progress = float(capture.progress.progress)
        if progress - last_progress >= 0.1:
            carb.log_info(f"[MP4] capture_progress={progress:.2f}")
            last_progress = progress
        if time.monotonic() - start > args.timeout_s:
            capture.cancel()
            raise TimeoutError("Timed out while capturing MP4")

    for _ in range(30):
        app.update()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"MP4 was not created or is empty: {output_path}")

    print(
        "[MP4] done "
        f"stage={stage.GetRootLayer().identifier} "
        f"output={output_path} "
        f"bytes={output_path.stat().st_size}"
    )


def main():
    args = _parse_args()
    print("[MP4] launching_isaac", flush=True)
    launcher = AppLauncher(args)
    app = launcher.app
    print("[MP4] isaac_launched", flush=True)
    try:
        _capture_mp4(args, app)
    finally:
        app.close()


if __name__ == "__main__":
    main()
