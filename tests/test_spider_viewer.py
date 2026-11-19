"""Unit tests for multi-seed support and Web UI seed switching in spider_viewer.py."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

import spider_viewer as sv


def _write_dummy_video(video_dir: Path, filename: str, payload: bytes, meta: dict) -> Path:
    video_dir.mkdir(parents=True, exist_ok=True)
    mp4 = video_dir / filename
    mp4.write_bytes(payload)
    mp4.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return mp4


def test_discover_seeds_numeric_sorting_and_layouts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    ckpt_dir = run_dir / "checkpoints"
    for name in ("seed_100", "seed_9", "seed_42", "not_a_seed", "seed_abc"):
        (ckpt_dir / name).mkdir(parents=True)
    # Direct fallback seed (only under run_dir)
    (run_dir / "seed_43").mkdir(parents=True)
    # File named seed_7 should be ignored (not a directory)
    (ckpt_dir / "seed_7").write_text("file", encoding="utf-8")

    discovered = sv.discover_seeds(run_dir)
    assert discovered == [
        ("seed_9", 9),
        ("seed_42", 42),
        ("seed_43", 43),
        ("seed_100", 100),
    ]


def test_parse_args_seed_optional() -> None:
    args_all = sv.parse_args(["-d", "outputs/19_11_08"])
    assert args_all.input_dir == "outputs/19_11_08"
    assert args_all.seed is None

    args_single = sv.parse_args(["-d", "outputs/19_11_08", "--seed", "42"])
    assert args_single.seed == "42"
    assert sv.normalize_seed(args_single.seed) == ("seed_42", 42)


def test_viewer_state_multi_seed_isolation_and_dynamic_registration(tmp_path: Path) -> None:
    run_dir = tmp_path / "demo_run"
    (run_dir / "checkpoints" / "seed_42").mkdir(parents=True)
    (run_dir / "checkpoints" / "seed_43").mkdir(parents=True)
    videos_root = tmp_path / "videos"

    v42_dir = sv.resolve_video_dir(videos_root, run_dir, "seed_42", "latest")
    v43_dir = sv.resolve_video_dir(videos_root, run_dir, "seed_43", "latest")
    p42 = _write_dummy_video(
        v42_dir,
        "standard_ant_ep_00010_step_00010000_20261119_090000.mp4",
        b"SEED42_MP4_" + b"x" * 2048,
        {"episode": 10, "rendered_at": "2026-11-19 09:00:00", "rollout": {"return": 123.0}},
    )
    p43 = _write_dummy_video(
        v43_dir,
        "standard_ant_ep_00020_step_00020000_20261119_090500.mp4",
        b"SEED43_MP4_" + b"y" * 2048,
        {"episode": 20, "rendered_at": "2026-11-19 09:05:00", "rollout": {"return": 456.0}},
    )

    state = sv.ViewerState(
        input_dir=run_dir,
        seeds=[("seed_43", 43), ("seed_42", 42)],
        ckpt_arg="latest",
        scenario_queries=["standard"],
        videos_root=videos_root,
        env_type="ant",
        watch_all_seeds=True,
    )

    assert state.seed_order == ["seed_42", "seed_43"]
    assert state.get_video_path(seed_name="seed_42", scenario_key="standard_ant") == p42
    assert state.get_video_path(seed_name="seed_43", scenario_key="standard_ant") == p43
    # Default seed fallback when seed_name omitted
    assert state.get_video_path(scenario_key="standard_ant") == p42
    # Path traversal rejected
    assert state.get_video_path(seed_name="seed_42", filename="../seed_43/standard_ant.mp4") is None

    # Dynamic registration keeps numeric order
    state.ensure_seed("seed_10", 10)
    assert state.seed_order == ["seed_10", "seed_42", "seed_43"]

    snap = state.snapshot()
    assert snap["watch_all_seeds"] is True
    assert snap["seed_order"] == ["seed_10", "seed_42", "seed_43"]
    assert snap["seeds"]["seed_42"]["metadata"]["episode"] == 10
    assert snap["seeds"]["seed_43"]["metadata"]["episode"] == 20


def test_watcher_discovers_and_renders_all_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "live_run"
    for s in ("seed_42", "seed_43"):
        d = run_dir / "checkpoints" / s
        d.mkdir(parents=True)
        (d / "latest_checkpoint.pt").write_bytes(b"fake_ckpt_bytes")

    state = sv.ViewerState(
        input_dir=run_dir,
        seeds=[],
        ckpt_arg="latest",
        scenario_queries=["standard"],
        videos_root=tmp_path / "videos",
        env_type="ant",
        watch_all_seeds=True,
    )

    rendered_seeds = []

    def fake_render_once(st: sv.ViewerState, seed_st: sv.SeedState, ckpt_stat, args) -> None:
        rendered_seeds.append(seed_st.seed_name)
        with st.lock:
            seed_st.last_stat = ckpt_stat
            seed_st.status = "watching"
            seed_st.version += 1

    monkeypatch.setattr(sv, "render_once", fake_render_once)
    monkeypatch.setattr(sv, "wait_for_stable_stat", lambda p, s, settle_sec=0.5: True)

    args = sv.parse_args(["-d", str(run_dir), "--poll-interval", "0.05"])
    stop_event = threading.Event()
    t = threading.Thread(
        target=sv.checkpoint_watcher_loop,
        args=(state, args, stop_event),
        daemon=True,
    )
    t.start()
    try:
        deadline = time.time() + 3.0
        while len(rendered_seeds) < 2 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        stop_event.set()
        t.join(timeout=2.0)

    assert rendered_seeds == ["seed_42", "seed_43"]
    assert state.seed_order == ["seed_42", "seed_43"]
    assert state.status == "watching"


def test_http_endpoints_serve_per_seed_videos_and_ui(tmp_path: Path) -> None:
    run_dir = tmp_path / "http_run"
    videos_root = tmp_path / "videos"
    v42_dir = sv.resolve_video_dir(videos_root, run_dir, "seed_42", "latest")
    v43_dir = sv.resolve_video_dir(videos_root, run_dir, "seed_43", "latest")
    bytes_42 = b"AAA42_" + b"0" * 2048
    bytes_43 = b"BBB43_" + b"1" * 2048
    _write_dummy_video(
        v42_dir,
        "standard_ant_ep_00005_step_00005000_20261119_090000.mp4",
        bytes_42,
        {"episode": 5, "rendered_at": "2026-11-19 09:00:00", "rollout": {"return": 50.0}},
    )
    _write_dummy_video(
        v43_dir,
        "standard_ant_ep_00009_step_00009000_20261119_090100.mp4",
        bytes_43,
        {"episode": 9, "rendered_at": "2026-11-19 09:01:00", "rollout": {"return": 90.0}},
    )

    state = sv.ViewerState(
        input_dir=run_dir,
        seeds=[("seed_42", 42), ("seed_43", 43)],
        ckpt_arg="latest",
        scenario_queries=["standard"],
        videos_root=videos_root,
        env_type="ant",
        watch_all_seeds=True,
    )

    class _TestHandler(sv.SpiderViewerHandler):
        viewer_state = state

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _TestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"

        # HTML page includes seed switcher elements
        with urlopen(f"{base}/") as resp:
            html = resp.read().decode("utf-8")
            assert resp.status == 200
            assert 'id="seedBar"' in html
            assert 'id="seedTabs"' in html
            assert "selectSeed(" in html

        # Status API returns both seeds
        with urlopen(f"{base}/api/status") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["seed_order"] == ["seed_42", "seed_43"]
            assert "seed_42" in data["seeds"]
            assert "seed_43" in data["seeds"]

        # Per-seed video streaming returns distinct bytes
        with urlopen(f"{base}/video?seed=seed_42&scenario=standard_ant") as resp:
            assert resp.status == 200
            assert resp.read() == bytes_42

        with urlopen(f"{base}/video?seed=seed_43&scenario=standard_ant") as resp:
            assert resp.status == 200
            assert resp.read() == bytes_43

        # HTTP 206 byte range request on seed_43
        req = Request(f"{base}/video?seed=seed_43&scenario=standard_ant", headers={"Range": "bytes=0-5"})
        with urlopen(req) as resp:
            assert resp.status == 206
            assert resp.read() == b"BBB43_"
    finally:
        server.shutdown()
        server.server_close()
