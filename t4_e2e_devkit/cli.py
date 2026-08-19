"""The ``t4e2e`` command line.

One place to discover everything the devkit can do::

    t4e2e agents                 # what agents are registered
    t4e2e datalist ...           # build a data list
    t4e2e train ...              # train an agent (hydra)
    t4e2e score ...              # score an agent (hydra)
    t4e2e evaluate-closed-loop ... # evaluate sensor-replay closed loop
    t4e2e merge-closed-loop ... # merge completed closed-loop rank reports
    t4e2e report-closed-loop ... # render a local HTML report
    t4e2e inspect LIST           # explain a data list, filter policy included
    t4e2e rigs SCENE...          # what cameras a scene has, and how they are stored
    t4e2e visualize SCENE         # render a window: BEV, cameras, or both
    t4e2e visualize-video ...    # render planning videos from a data list
    t4e2e check                  # verify the install and the vendored sources

``train`` and ``score`` forward their arguments to the hydra entry points
unchanged, so a documented ``python -m t4_e2e_devkit.script.run_training ...``
command works identically as ``t4e2e train ...``.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence


def _cmd_agents(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.agents.registry import agent_registry

    registry = agent_registry()
    if not registry:
        print("no agents registered")
        return 1
    width = max(len(name) for name in registry)
    for name in sorted(registry):
        constructor = registry[name]
        module = getattr(constructor, "__module__", "?")
        print(f"{name:<{width}}  {module}.{getattr(constructor, '__qualname__', constructor)}")
    return 0


def _cmd_inspect(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e inspect")
    parser.add_argument("data_list", help="path to a data list")
    parser.add_argument(
        "--no-subtree-check", action="store_true",
        help="skip the annotation-free subtree guard",
    )
    args = parser.parse_args(argv)

    from t4_e2e_devkit.dataset.datalist import describe_data_list, load_data_list

    data_list = load_data_list(args.data_list, check_subtree=not args.no_subtree_check)
    print(describe_data_list(data_list))
    return 0


def _cmd_rigs(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="t4e2e rigs",
        description="Report each scene's camera register, its storage and which "
        "profiles fit. There is no single T4 rig, so this is what to run before "
        "choosing a camera profile for a run.",
    )
    parser.add_argument("scenes", nargs="*", help="T4 scene directories")
    parser.add_argument("--group", action="store_true", help="group the scenes by rig instead")
    parser.add_argument(
        "--timing", metavar="LIST",
        help="check camera timing consistency across a data list's scenes",
    )
    args = parser.parse_args(argv)
    if not args.scenes and not args.timing:
        parser.error("give at least one scene directory, or --timing LIST")

    if args.timing:
        from t4_e2e_devkit.dataset.datalist import load_data_list
        from t4_e2e_devkit.dataset.sync import data_list_timing_report

        report = data_list_timing_report(load_data_list(args.timing))
        print(f"scenes checked: {report['scenes_checked']}   "
              f"vehicles: {len(report['vehicles'])}")
        print()
        print(f"{'channel':<24} {'min_ms':>8} {'max_ms':>8} "
              f"{'across-veh':>11} {'worst-within':>13} {'vehicles':>9}")
        for channel, values in report["channels"].items():
            # Across-vehicle spread is a property of the fleet, not a defect.
            # Within-vehicle spread means one rig's timing wandered, which is.
            flag = "  <-- within-vehicle drift" if channel in report["suspect"] else ""
            print(
                f"{channel:<24} {values['min_ms']:>8.1f} {values['max_ms']:>8.1f} "
                f"{values['across_vehicle_spread_ms']:>11.1f} "
                f"{values['worst_within_vehicle_spread_ms']:>13.2f} "
                f"{values['vehicles']:>9}{flag}"
            )
        print()
        print("per-vehicle offsets are read from each scene, so the correction is "
              "exact regardless of this spread; the table is for knowing what a "
              "training set contains.")
        if report["suspect"]:
            print(
                f"\n{len(report['suspect'])} channel(s) drift by more than "
                f"{report['tolerance_ms']} ms WITHIN a single vehicle, which points "
                "at a converter or recording defect rather than fleet variation."
            )
            return 1
        return 0

    from t4_e2e_devkit.dataset.camera_source import available_cameras
    from t4_e2e_devkit.dataset.rigs import (
        describe_rig,
        group_scenes_by_rig,
        readable_camera_names,
    )

    if args.group:
        groups = group_scenes_by_rig(args.scenes)
        for signature, scenes in sorted(groups.items(), key=lambda item: -len(item[1])):
            print(f"{len(scenes):5d} scene(s)  {signature.replace('|', ', ')}")
            print(f"        e.g. {scenes[0]}")
        return 0

    for scene in args.scenes:
        print(f"=== {scene}")
        try:
            print("  " + describe_rig(readable_camera_names(scene)).replace("\n", "\n  "))
            storage = available_cameras(scene)
            for name, kind in sorted(storage.items()):
                print(f"    {name:<24} {kind}")
        except Exception as error:  # noqa: BLE001
            print(f"  unreadable: {error}")
    return 0


def _cmd_visualize(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="t4e2e visualize",
        description="Render one window. Frames are decoded through the same "
        "camera source the data loader uses, so the image is what a model sees.",
    )
    parser.add_argument("scene", help="T4 scene directory")
    parser.add_argument("--root", default=None, help="dataset root; inferred when omitted")
    parser.add_argument("--center", type=int, default=None, help="centre frame; middle by default")
    parser.add_argument("--out", default="window.png", help="output image path")
    parser.add_argument(
        "--mode", default="summary", choices=("bev", "cameras", "summary"),
        help="what to render (default: %(default)s)",
    )
    parser.add_argument("--agent", default=None, help="also plan with this registered agent")
    parser.add_argument("--no-lidar", action="store_true", help="skip the LiDAR overlay")
    parser.add_argument("--view-range", type=float, default=None, help="BEV half-extent, metres")
    args = parser.parse_args(argv)

    from pathlib import Path

    from t4_e2e_devkit.dataset.rigs import sensor_config_for_scene
    from t4_e2e_devkit.dataset.window import T4WindowBuilder
    from t4_e2e_devkit.visualization import (
        plot_bev_frame,
        plot_cameras_frame,
        plot_scene_summary,
        save_figure,
    )

    scene_dir = Path(args.scene).resolve()
    if args.root is not None:
        root = Path(args.root).resolve()
    else:
        # A scene lives at <root>/<subtree>/<uuid>/<date>/<time>, so the root is
        # four levels up. Stated rather than guessed per call.
        root = scene_dir.parents[3]

    needs_cameras = args.mode in ("cameras", "summary")
    sensor_config = (
        sensor_config_for_scene(scene_dir, lidar=not args.no_lidar)
        if needs_cameras
        else None
    )
    builder = T4WindowBuilder(scene_dir, root, sensor_config=sensor_config)
    try:
        centers = builder.valid_centers()
        if not len(centers):
            print(f"{scene_dir}: too short for a full window", file=sys.stderr)
            return 1
        center = args.center if args.center is not None else centers[len(centers) // 2]
        scene = builder.build(center)

        trajectories = {}
        if scene.future_ego_poses is not None:
            trajectories["ground_truth"] = scene.get_future_trajectory()
        if args.agent:
            from t4_e2e_devkit.agents.registry import build_agent

            agent = build_agent(args.agent)
            agent.initialize()
            trajectories["prediction"] = (
                agent.compute_trajectory_from_scene(scene)
                if agent.requires_scene
                else agent.compute_trajectory(scene.get_agent_input())
            )

        config = {"view_range": args.view_range} if args.view_range else None
        if args.mode == "bev":
            figure, _ = plot_bev_frame(scene, trajectories, config)
        elif args.mode == "cameras":
            figure, _ = plot_cameras_frame(
                scene, with_annotations=True, with_lidar=not args.no_lidar
            )
        else:
            figure, _ = plot_scene_summary(scene, trajectories, config)
        print(save_figure(figure, args.out))
    finally:
        builder.close()
    return 0


def _cmd_visualize_video(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="t4e2e visualize-video",
        description="Render one mp4 per scene: camera and BEV side by side, with the "
        "recorded future and any number of labelled prediction manifests overlaid. "
        "Frames are the data list's windows, which is also the key space of a "
        "prediction manifest, so a video shows exactly what was scored.",
    )
    parser.add_argument("--data-list", required=True, help="data list naming the windows")
    parser.add_argument(
        "--scene", nargs="*", default=None,
        help="relative scene directories to render (default: every scene in the list)",
    )
    parser.add_argument(
        "--manifest", action="append", default=[], metavar="LABEL=PATH",
        help="prediction manifest to overlay, repeatable; the label names the model",
    )
    parser.add_argument("--out", required=True, help="output directory for the mp4 files")
    parser.add_argument("--camera", default=None, help="camera to show; front-most by default")
    parser.add_argument("--fps", type=float, default=10.0, help="frames per second")
    parser.add_argument("--no-lidar", action="store_true", help="skip the LiDAR overlay")
    parser.add_argument("--view-range", type=float, default=None, help="BEV half-extent, metres")
    args = parser.parse_args(argv)

    from pathlib import Path

    from t4_e2e_devkit.dataset.datalist import load_data_list
    from t4_e2e_devkit.dataset.rigs import readable_camera_names, sensor_config_for_scene
    from t4_e2e_devkit.dataset.window import T4WindowBuilder
    from t4_e2e_devkit.evaluation.prediction_manifest import load_prediction_manifest
    from t4_e2e_devkit.visualization.planning_video import (
        front_camera_name,
        render_planning_video,
    )

    data_list = load_data_list(args.data_list)

    manifests = {}
    for entry in args.manifest:
        label, separator, path = entry.partition("=")
        if not separator or not label or not path:
            parser.error(f"--manifest expects LABEL=PATH, got {entry!r}")
        if label in manifests:
            parser.error(f"duplicate manifest label {label!r}")
        manifests[label] = load_prediction_manifest(path)

    scene_dirs = data_list.scene_dirs
    if args.scene:
        missing = sorted(set(args.scene) - set(scene_dirs))
        if missing:
            parser.error(
                f"scene(s) not in the data list: {missing}; run 't4e2e inspect' to see it"
            )
        scene_dirs = list(args.scene)

    out_dir = Path(args.out)
    for scene_rel in scene_dirs:
        centers = sorted({center for scene, center in data_list.rows if scene == scene_rel})
        scene_dir = data_list.absolute_scene_dir(scene_rel)
        camera = args.camera or front_camera_name(readable_camera_names(scene_dir))
        builder = T4WindowBuilder(
            scene_dir,
            data_list.root,
            sensor_config=sensor_config_for_scene(
                scene_dir, cameras=[camera], lidar=not args.no_lidar
            ),
        )
        try:
            windows = (builder.build(center) for center in centers)
            out_path = out_dir / (scene_rel.replace("/", "_") + ".mp4")
            print(
                render_planning_video(
                    windows,
                    out_path,
                    manifests,
                    camera=camera,
                    fps=args.fps,
                    view_range=args.view_range,
                )
            )
        finally:
            builder.close()
    return 0


def _cmd_check(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e check")
    parser.add_argument("--vendor", action="store_true", help="also diff vendored files vs sources")
    args = parser.parse_args(argv)

    ok = True
    print("imports:")
    for module in (
        "t4_e2e_devkit.common.dataclasses",
        "t4_e2e_devkit.dataset.window",
        "t4_e2e_devkit.dataset.dataset",
        "t4_e2e_devkit.agents",
        "t4_e2e_devkit.evaluation.navsim_score",
        "t4_e2e_devkit.evaluation.gpu.oracle",
        "t4_e2e_devkit.planning.simulation.closed_loop",
        "t4_e2e_devkit.planning.simulation.planner.pdm_planner.scoring.pdm_scorer",
        "t4_e2e_devkit.visualization",
    ):
        try:
            __import__(module)
            print(f"  ok       {module}")
        except Exception as error:  # noqa: BLE001
            ok = False
            print(f"  FAILED   {module}: {error}")

    # The devkit deliberately carries its own geometry/state vocabulary; a
    # nuplan import appearing here means a vendored file was edited by hand.
    print("nuplan independence:")
    if "nuplan" in sys.modules:
        ok = False
        print("  FAILED   nuplan was imported; the devkit must not depend on it")
    else:
        print("  ok       no nuplan import")

    print("scoring backends:")
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - report an actionable check failure
        ok = False
        print(f"  FAILED   torch unavailable: {error}")
    else:
        print("  cpu      available (reference judge)")
        if torch.cuda.is_available():
            print(f"  gpu      available ({torch.cuda.device_count()} device(s))")
        else:
            print("  gpu      unavailable (no CUDA); scoring must use backend=cpu")

    if args.vendor:
        print("vendored sources:")
        import subprocess
        from pathlib import Path

        tool = Path(__file__).resolve().parent.parent / "tools" / "vendor.py"
        if not tool.is_file():
            print(f"  skipped  {tool} not present (installed package)")
        else:
            result = subprocess.run(
                [sys.executable, str(tool), "check"], capture_output=True, text=True
            )
            print("  " + result.stdout.strip().replace("\n", "\n  "))
            ok = ok and result.returncode == 0

    return 0 if ok else 1


def _forward(module: str, argv: Sequence[str]) -> int:
    """Run a hydra entry point with ``argv`` as its command line."""
    import runpy

    sys.argv = [module, *argv]
    runpy.run_module(module, run_name="__main__")
    return 0


def _cmd_datalist(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.build_datalist import main as build_main

    return build_main(list(argv))


def _cmd_evaluate_closed_loop(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.evaluate_closed_loop import main as evaluate_main

    return evaluate_main(list(argv))


def _cmd_merge_closed_loop(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.merge_closed_loop import main as merge_main

    return merge_main(list(argv))


def _cmd_merge_workers(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.merge_workers import main as merge_main

    return merge_main(list(argv))


def _cmd_distribute(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.distributed import main as distribute_main

    return distribute_main(list(argv))


def _cmd_submit(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.submit import main as submit_main

    return submit_main(list(argv))


def _cmd_merge_submission(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.merge_submission import main as merge_main

    return merge_main(list(argv))


def _cmd_score_submission(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.score_submission import main as score_main

    return score_main(list(argv))


def _cmd_score_manifest(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.score_prediction_manifest import main as score_main

    return score_main(list(argv))


def _cmd_merge_score_submission(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.merge_submission_score import main as merge_main

    return merge_main(list(argv))


def _cmd_leaderboard(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.leaderboard import main as leaderboard_main

    return leaderboard_main(list(argv))


def _cmd_run_config(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.run_experiment import main as run_main

    return run_main(list(argv))


def _cmd_evaluate(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.evaluate import main as evaluate_main

    return evaluate_main(list(argv))


def _cmd_merge_evaluation(argv: Sequence[str]) -> int:
    from t4_e2e_devkit.script.merge_evaluation import main as merge_main

    return merge_main(list(argv))


def _cmd_dashboard(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e dashboard")
    parser.add_argument("results_dir", nargs="+", help="one or more ignored results/report directories")
    parser.add_argument("--out", default=None, help="HTML output path")
    parser.add_argument("--title", default="T4 experiment analysis")
    args = parser.parse_args(argv)
    if len(args.results_dir) == 1:
        from t4_e2e_devkit.visualization.dashboard import write_results_dashboard

        print(write_results_dashboard(args.results_dir[0], args.out, title=args.title))
    else:
        from t4_e2e_devkit.visualization.experiment_dashboard import write_experiment_dashboard

        print(write_experiment_dashboard(args.results_dir, args.out, title=args.title))
    return 0


def _cmd_report_closed_loop(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e report-closed-loop")
    parser.add_argument("report_dir", help="evaluation report directory")
    parser.add_argument("--out", default=None, help="HTML output path; defaults to report.html")
    args = parser.parse_args(argv)
    from t4_e2e_devkit.evaluation.closed_loop_report import write_static_html_report

    print(write_static_html_report(args.report_dir, args.out))
    return 0


COMMANDS = {
    "agents": (_cmd_agents, "list registered agents"),
    "datalist": (_cmd_datalist, "build a data list from a T4 root"),
    "evaluate": (_cmd_evaluate, "evaluate independent metric families"),
    "merge-evaluation": (_cmd_merge_evaluation, "merge evaluation rank reports"),
    "evaluate-closed-loop": (
        _cmd_evaluate_closed_loop,
        "evaluate sensor-replay closed loop",
    ),
    "merge-closed-loop": (_cmd_merge_closed_loop, "merge closed-loop rank reports"),
    "merge-workers": (_cmd_merge_workers, "merge distributed worker manifests"),
    "distribute": (_cmd_distribute, "launch and merge all evaluation ranks"),
    "submit": (_cmd_submit, "write a validated trajectory submission"),
    "score-manifest": (_cmd_score_manifest, "score an external prediction manifest"),
    "merge-submission": (_cmd_merge_submission, "merge rank trajectory submissions"),
    "score-submission": (_cmd_score_submission, "score a trajectory submission"),
    "merge-score-submission": (_cmd_merge_score_submission, "merge rank submission scores"),
    "leaderboard": (_cmd_leaderboard, "rank completed result directories"),
    "run-config": (_cmd_run_config, "run a typed experiment configuration"),
    "report-closed-loop": (_cmd_report_closed_loop, "render a local closed-loop HTML report"),
    "dashboard": (_cmd_dashboard, "render a local results dashboard"),
    "train": (
        lambda argv: _forward("t4_e2e_devkit.script.run_training", argv),
        "train an agent (hydra)",
    ),
    "score": (
        lambda argv: _forward("t4_e2e_devkit.script.run_pdm_score", argv),
        "score an agent over a data list (hydra)",
    ),
    "inspect": (_cmd_inspect, "explain a data list and the policy that built it"),
    "rigs": (_cmd_rigs, "report a scene's camera register and how it is stored"),
    "visualize": (_cmd_visualize, "render a window: BEV, cameras, or both"),
    "visualize-video": (
        _cmd_visualize_video,
        "render per-frame planning videos from a data list",
    ),
    "check": (_cmd_check, "verify the install and the vendored sources"),
}


def main(argv: Optional[List[str]] = None) -> int:
    """
    :param argv: argument vector; ``sys.argv[1:]`` by default.
    :return: process exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        width = max(len(name) for name in COMMANDS)
        print(__doc__.split("\n\n")[0])
        print("\nusage: t4e2e <command> [args]\n\ncommands:")
        for name, (_, description) in COMMANDS.items():
            print(f"  {name:<{width}}  {description}")
        print("\nRun 't4e2e <command> --help' for a command's own options.")
        return 0

    command, *rest = argv
    if command not in COMMANDS:
        print(f"unknown command {command!r}; expected one of {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    return COMMANDS[command][0](rest) or 0


if __name__ == "__main__":
    sys.exit(main())
