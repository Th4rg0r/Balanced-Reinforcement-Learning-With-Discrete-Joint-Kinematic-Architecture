"""Live Checkpoint Watcher & Web Video Viewer ("spider_viewer") for BRL-GMCTS.

Watches a training run's checkpoint (e.g. outputs/validation/checkpoints/seed_42/latest_checkpoint.pt),
immediately renders a validation scenario MP4 to:
    videos/<run_name>/<seed>/<checkpoint_key>/<scenario_key>.mp4
and automatically regenerates the video whenever the checkpoint updates and is ready to read.

Serves an externally accessible HTTP dashboard (0.0.0.0) with an embedded HTML5 video player
(supporting HTTP 206 byte-range requests) and live checkpoint / rollout / benchmark statistics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html as html_lib
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.video_recorder import (
    create_rendering_env,
    filter_validation_scenarios,
    get_validation_scenarios,
    render_scenario_video,
)


# ==============================================================================
# Path & Argument Helpers
# ==============================================================================

def normalize_seed(seed_arg: str) -> Tuple[str, int]:
    """Normalizes 'seed_42' or '42' into ('seed_42', 42)."""
    raw = str(seed_arg).strip()
    m = re.fullmatch(r"(?:seed_)?(\d+)", raw, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid seed '{seed_arg}'. Expected e.g. 'seed_42' or '42'.")
    seed_num = int(m.group(1))
    return f"seed_{seed_num}", seed_num


def discover_seeds(input_dir: Path) -> List[Tuple[str, int]]:
    """Discovers all seed_<N> subdirectories under input_dir/checkpoints or input_dir, sorted numerically."""
    found: Dict[int, str] = {}
    search_dirs: List[Path] = []
    ckpt_dir = input_dir / "checkpoints"
    if ckpt_dir.is_dir():
        search_dirs.append(ckpt_dir)
    if input_dir.is_dir():
        search_dirs.append(input_dir)

    for base in search_dirs:
        try:
            entries = list(base.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            m = re.fullmatch(r"seed_(\d+)", entry.name, flags=re.IGNORECASE)
            if m:
                seed_num = int(m.group(1))
                found.setdefault(seed_num, f"seed_{seed_num}")

    return [(found[num], num) for num in sorted(found)]


def resolve_checkpoint_path(input_dir: Path, seed_name: str, ckpt_arg: str) -> Tuple[Path, str]:
    """Resolves checkpoint argument ('latest', 'best', 'final', or filename/stem) to (path, key)."""
    ckpt_arg_clean = ckpt_arg.strip()
    low = ckpt_arg_clean.lower()

    alias_files = {
        "latest": ("latest_checkpoint.pt", "latest"),
        "latest_checkpoint": ("latest_checkpoint.pt", "latest"),
        "best": ("best_model.pt", "best"),
        "best_model": ("best_model.pt", "best"),
        "final": ("final_model.pt", "final"),
        "final_model": ("final_model.pt", "final"),
    }

    ckpt_base = input_dir / "checkpoints" / seed_name
    if not ckpt_base.exists() and (input_dir / seed_name).exists():
        ckpt_base = input_dir / seed_name

    if low in alias_files:
        fname, key = alias_files[low]
        return ckpt_base / fname, key

    fname = ckpt_arg_clean if low.endswith(".pt") else f"{ckpt_arg_clean}.pt"
    key = Path(fname).stem
    candidate = ckpt_base / fname
    if candidate.exists() or not (input_dir / fname).exists():
        return candidate, key
    return input_dir / fname, key


def resolve_video_dir(
    videos_root: Path,
    input_dir: Path,
    seed_name: str,
    ckpt_key: str,
) -> Path:
    """Returns videos/<run_name>/<seed_name>/<ckpt_key> directory."""
    run_name = input_dir.resolve().name or "run"
    return videos_root / run_name / seed_name / ckpt_key


def make_unique_video_paths(
    video_dir: Path,
    scenario_key: str,
    episode: int,
    total_env_steps: int,
) -> Tuple[Path, Path]:
    """Generates unique, persistent (video_path, metadata_json_path) for a render."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_stem = f"{scenario_key}_ep_{episode:05d}_step_{total_env_steps:08d}_{stamp}"
    stem = base_stem
    counter = 1
    while (video_dir / f"{stem}.mp4").exists():
        stem = f"{base_stem}_{counter}"
        counter += 1
    return video_dir / f"{stem}.mp4", video_dir / f"{stem}.json"


def find_latest_persisted_video(video_dir: Path, scenario_key: str) -> Tuple[Path, Path]:
    """Finds the most recently modified persisted video in video_dir, or returns a fallback path."""
    if video_dir.exists():
        candidates = [
            p for p in video_dir.glob(f"{scenario_key}*.mp4")
            if not p.name.endswith(".tmp.mp4") and p.is_file() and p.stat().st_size > 1024
        ]
        if candidates:
            latest_mp4 = max(candidates, key=lambda p: p.stat().st_mtime_ns)
            return latest_mp4, latest_mp4.with_suffix(".json")
    return video_dir / f"{scenario_key}.mp4", video_dir / f"{scenario_key}.json"


def list_persisted_videos(video_dir: Path, scenario_key: str) -> List[Dict[str, Any]]:
    """Lists all persisted videos for a scenario in video_dir, newest first."""
    if not video_dir.exists():
        return []
    candidates = [
        p for p in video_dir.glob(f"{scenario_key}*.mp4")
        if not p.name.endswith(".tmp.mp4") and p.is_file() and p.stat().st_size > 1024
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    items: List[Dict[str, Any]] = []
    for mp4 in candidates:
        meta_file = mp4.with_suffix(".json")
        meta: Dict[str, Any] = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        rollout = meta.get("rollout") or {}
        items.append({
            "filename": mp4.name,
            "episode": meta.get("episode"),
            "total_env_steps": meta.get("total_env_steps"),
            "rendered_at": meta.get("rendered_at")
            or datetime.fromtimestamp(mp4.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size_mb": round(mp4.stat().st_size / (1024.0 * 1024.0), 2),
            "return": rollout.get("return"),
            "total_distance": rollout.get("total_distance"),
            "metadata": meta,
        })
    return items


def resolve_scenario_choice(env_type: str, scenario_query: str) -> Tuple[str, Dict[str, Any]]:
    """Resolves a scenario query (e.g. 'standard', 'spider', 'icy') to (key, spec)."""
    all_scenarios = get_validation_scenarios(env_type)
    matched = filter_validation_scenarios(all_scenarios, scenario_query, env_type=env_type)
    key = next(iter(matched))
    return key, matched[key]


def infer_env_type(payload: Dict[str, Any], requested: str) -> str:
    """Infers creature env_type ('ant', 'humanoid', 'dog') from checkpoint payload."""
    if requested.lower().strip() != "auto":
        return requested.lower().strip()
    cfg_dict = payload.get("config", {})
    if isinstance(cfg_dict, dict):
        ant_id = str(cfg_dict.get("ant_morph", {}).get("env_id", "")).lower()
        if "humanoid" in ant_id:
            return "humanoid"
        if "dog" in ant_id or "go1" in ant_id:
            return "dog"
        if "dog_morph" in cfg_dict and not cfg_dict.get("ant_morph"):
            return "dog"
        if "humanoid_morph" in cfg_dict and not cfg_dict.get("ant_morph"):
            return "humanoid"
    return "ant"


def load_checkpoint_for_env(ckpt_path: Path, env_type: str, device: str):
    """Loads agent and payload using the creature-specific checkpoint loader."""
    et = env_type.lower().strip()
    if et == "ant":
        from experiments.train_ant_brl import load_training_checkpoint
    elif et == "humanoid":
        from experiments.train_humanoid_brl import load_training_checkpoint
    elif et in ("dog", "unitree_go1", "go1"):
        from experiments.train_dog_brl import load_training_checkpoint
    else:
        raise ValueError(f"Unknown env_type '{env_type}'.")
    return load_training_checkpoint(ckpt_path, device=device)


def get_local_ip() -> str:
    """Best-effort LAN IP lookup for displaying external http:// URL."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        return "127.0.0.1"


# ==============================================================================
# Shared Viewer State
# ==============================================================================

class SeedState:
    """Per-seed checkpoint, scenario, and persisted video state."""

    def __init__(
        self,
        input_dir: Path,
        seed_name: str,
        seed_num: int,
        ckpt_path: Path,
        ckpt_key: str,
        scenario_queries: List[str],
        videos_root: Path,
        env_type: str,
    ) -> None:
        self.input_dir = input_dir
        self.seed_name = seed_name
        self.seed_num = seed_num
        self.ckpt_path = ckpt_path
        self.ckpt_key = ckpt_key
        self.video_dir = resolve_video_dir(videos_root, input_dir, seed_name, ckpt_key)
        self.status: str = "starting"
        self.version: int = 0
        self.last_stat: Optional[Tuple[int, int]] = None
        self.last_render_at: Optional[str] = None
        self.last_error: Optional[str] = None

        init_env = "ant" if env_type == "auto" else env_type
        self.scenario_order: List[str] = []
        self.scenarios: Dict[str, Dict[str, Any]] = {}
        for q in scenario_queries:
            scen_key, scen_spec = resolve_scenario_choice(init_env, q)
            if scen_key in self.scenarios:
                continue
            self.scenario_order.append(scen_key)
            v_path, m_path = find_latest_persisted_video(self.video_dir, scen_key)
            meta: Dict[str, Any] = {}
            scen_ver = 0
            last_at = None
            if m_path.exists():
                try:
                    meta = json.loads(m_path.read_text(encoding="utf-8"))
                    if v_path.exists() and v_path.stat().st_size > 1024:
                        scen_ver = 1
                        last_at = meta.get("rendered_at")
                        self.version = max(self.version, 1)
                except Exception:
                    pass
            self.scenarios[scen_key] = {
                "scenario_key": scen_key,
                "display_name": scen_spec.get("display_name", scen_key),
                "video_path": v_path,
                "meta_path": m_path,
                "version": scen_ver,
                "last_render_at": last_at,
                "metadata": meta,
            }

        self.scenario_key = self.scenario_order[0] if self.scenario_order else "standard_ant"

    def snapshot(self) -> Dict[str, Any]:
        scenarios_snap: Dict[str, Any] = {}
        for key in self.scenario_order:
            entry = self.scenarios[key]
            vp = entry["video_path"]
            v_exists = vp.exists() and vp.stat().st_size > 1024
            v_size = vp.stat().st_size if v_exists else 0
            history = list_persisted_videos(self.video_dir, key)
            scenarios_snap[key] = {
                "scenario_key": key,
                "display_name": entry["display_name"],
                "version": entry["version"],
                "last_render_at": entry["last_render_at"],
                "video_exists": v_exists,
                "video_size_bytes": v_size,
                "video_path": str(vp),
                "metadata": entry["metadata"],
                "history": history,
            }

        primary = scenarios_snap.get(self.scenario_key, {})
        return {
            "status": self.status,
            "version": self.version,
            "last_render_at": self.last_render_at,
            "last_error": self.last_error,
            "ckpt_path": str(self.ckpt_path),
            "ckpt_key": self.ckpt_key,
            "seed_name": self.seed_name,
            "seed_num": self.seed_num,
            "video_dir": str(self.video_dir),
            "scenario_key": self.scenario_key,
            "scenario_order": list(self.scenario_order),
            "scenarios": scenarios_snap,
            "video_exists": primary.get("video_exists", False),
            "video_size_bytes": primary.get("video_size_bytes", 0),
            "video_path": primary.get("video_path", ""),
            "metadata": primary.get("metadata", {}),
        }


class ViewerState:
    """Thread-safe multi-seed state shared between the background watcher and HTTP handlers."""

    def __init__(
        self,
        input_dir: Path,
        seeds: List[Tuple[str, int]],
        ckpt_arg: str,
        scenario_queries: List[str],
        videos_root: Path,
        env_type: str,
        watch_all_seeds: bool = False,
    ) -> None:
        self.lock = threading.Lock()
        self.input_dir = input_dir
        self.ckpt_arg = ckpt_arg
        self.scenario_queries = list(scenario_queries)
        self.videos_root = videos_root
        self.env_type = env_type
        self.watch_all_seeds = watch_all_seeds

        _, self.ckpt_key = resolve_checkpoint_path(input_dir, "seed_0", ckpt_arg)
        self.status: str = "starting"
        self.active_rendering_seed: Optional[str] = None
        self.last_error: Optional[str] = None

        self.seed_order: List[str] = []
        self.seeds: Dict[str, SeedState] = {}
        for seed_name, seed_num in seeds:
            self._register_seed_unlocked(seed_name, seed_num)

    def _register_seed_unlocked(self, seed_name: str, seed_num: int) -> SeedState:
        if seed_name in self.seeds:
            return self.seeds[seed_name]
        ckpt_path, ckpt_key = resolve_checkpoint_path(self.input_dir, seed_name, self.ckpt_arg)
        self.ckpt_key = ckpt_key
        ss = SeedState(
            input_dir=self.input_dir,
            seed_name=seed_name,
            seed_num=seed_num,
            ckpt_path=ckpt_path,
            ckpt_key=ckpt_key,
            scenario_queries=self.scenario_queries,
            videos_root=self.videos_root,
            env_type=self.env_type,
        )
        self.seeds[seed_name] = ss
        self.seed_order = [
            name for name, _ in sorted(
                ((k, v.seed_num) for k, v in self.seeds.items()),
                key=lambda item: item[1],
            )
        ]
        return ss

    def ensure_seed(self, seed_name: str, seed_num: int) -> SeedState:
        with self.lock:
            return self._register_seed_unlocked(seed_name, seed_num)

    def get_seed_states(self) -> List[SeedState]:
        with self.lock:
            return [self.seeds[name] for name in self.seed_order]

    def get_video_path(
        self,
        seed_name: Optional[str] = None,
        scenario_key: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        with self.lock:
            if not self.seed_order:
                return None
            target_seed = seed_name if seed_name in self.seeds else self.seed_order[0]
            ss = self.seeds.get(target_seed)
            if ss is None:
                return None
            if filename:
                safe_name = Path(filename).name
                if (
                    safe_name == filename
                    and safe_name.endswith(".mp4")
                    and not safe_name.endswith(".tmp.mp4")
                ):
                    candidate = ss.video_dir / safe_name
                    if candidate.is_file():
                        return candidate
                return None
            key = scenario_key if scenario_key in ss.scenarios else ss.scenario_key
            entry = ss.scenarios.get(key)
            return entry["video_path"] if entry else None

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            seeds_snap: Dict[str, Any] = {
                name: self.seeds[name].snapshot() for name in self.seed_order
            }
            default_seed_name = self.seed_order[0] if self.seed_order else ""
            default_snap = seeds_snap.get(default_seed_name, {})
            total_version = sum(int(s.get("version", 0)) for s in seeds_snap.values())
            return {
                "run_name": self.input_dir.resolve().name,
                "status": self.status,
                "active_rendering_seed": self.active_rendering_seed,
                "watch_all_seeds": self.watch_all_seeds,
                "seed_order": list(self.seed_order),
                "seeds": seeds_snap,
                "version": total_version,
                "last_render_at": default_snap.get("last_render_at"),
                "last_error": self.last_error,
                "ckpt_path": default_snap.get("ckpt_path", ""),
                "ckpt_key": self.ckpt_key,
                "seed_name": default_seed_name,
                "scenario_key": default_snap.get("scenario_key", "standard_ant"),
                "scenario_order": default_snap.get("scenario_order", []),
                "scenarios": default_snap.get("scenarios", {}),
                "video_exists": default_snap.get("video_exists", False),
                "video_size_bytes": default_snap.get("video_size_bytes", 0),
                "video_path": default_snap.get("video_path", ""),
                "metadata": default_snap.get("metadata", {}),
            }


# ==============================================================================
# Checkpoint Watcher & Video Rendering Worker
# ==============================================================================

def is_checkpoint_ready(ckpt_path: Path) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """Checks that checkpoint exists, is non-empty, and has no active .pt.tmp write."""
    tmp_path = ckpt_path.with_suffix(".pt.tmp")
    if tmp_path.exists() or not ckpt_path.exists() or not ckpt_path.is_file():
        return False, None
    try:
        st = ckpt_path.stat()
    except OSError:
        return False, None
    if st.st_size <= 0:
        return False, None
    return True, (st.st_mtime_ns, st.st_size)


def wait_for_stable_stat(
    ckpt_path: Path, initial_stat: Tuple[int, int], settle_sec: float = 0.5
) -> bool:
    """Confirms checkpoint file size and mtime remain stable across settle_sec."""
    time.sleep(settle_sec)
    ready, new_stat = is_checkpoint_ready(ckpt_path)
    return ready and new_stat == initial_stat


def replace_file_with_retry(src: Path, dst: Path, retries: int = 10, delay: float = 0.25) -> None:
    """Atomically replaces dst with src, retrying briefly on Windows sharing locks."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def read_progress_json(ckpt_path: Path) -> Dict[str, Any]:
    """Reads sibling progress.json if present."""
    prog_path = ckpt_path.parent / "progress.json"
    if not prog_path.exists():
        return {}
    try:
        return json.loads(prog_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_metadata(
    seed_state: SeedState,
    ckpt_stat: Tuple[int, int],
    payload: Dict[str, Any],
    render_res: Dict[str, Any],
    scenario_spec: Dict[str, Any],
    env_type: str,
) -> Dict[str, Any]:
    """Combines checkpoint payload, progress.json, and video rollout stats."""
    mtime_sec = ckpt_stat[0] / 1e9
    ckpt_mtime_str = datetime.fromtimestamp(mtime_sec).strftime("%Y-%m-%d %H:%M:%S")
    rendered_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress = read_progress_json(seed_state.ckpt_path)

    return {
        "run_name": seed_state.input_dir.resolve().name,
        "seed": int(payload.get("seed", seed_state.seed_num)),
        "seed_name": seed_state.seed_name,
        "env_type": env_type,
        "checkpoint_key": seed_state.ckpt_key,
        "checkpoint_file": seed_state.ckpt_path.name,
        "checkpoint_mtime": ckpt_mtime_str,
        "checkpoint_size_mb": round(ckpt_stat[1] / (1024.0 * 1024.0), 2),
        "episode": int(payload.get("episode", 0)),
        "total_env_steps": int(payload.get("total_env_steps", 0)),
        "composite_score": payload.get("composite_score"),
        "benchmark_metrics": payload.get("benchmark_metrics") or {},
        "win_evaluator": payload.get("win_evaluator") or {},
        "vel_curriculum": payload.get("vel_curriculum") or {},
        "progress": {
            "completed_episodes": progress.get("completed_episodes"),
            "target_episodes": progress.get("target_episodes"),
            "best_composite_score": progress.get("best_composite_score"),
            "updated_at": progress.get("updated_at"),
        },
        "rollout": {
            "scenario_key": render_res.get("scenario", seed_state.scenario_key),
            "display_name": scenario_spec.get("display_name", seed_state.scenario_key),
            "steps": int(render_res.get("steps", 0)),
            "return": float(render_res.get("return", 0.0)),
            "total_distance": float(render_res.get("total_distance", 0.0)),
            "x_velocity": float(render_res.get("x_velocity", 0.0)),
            "wall_sec": float(render_res.get("wall_sec", 0.0)),
        },
        "rendered_at": rendered_at,
    }


def render_once(
    state: ViewerState,
    seed_state: SeedState,
    ckpt_stat: Tuple[int, int],
    args: argparse.Namespace,
) -> None:
    """Loads one seed's checkpoint, renders all configured scenarios to unique files, and publishes each."""
    with state.lock:
        state.status = "rendering"
        state.active_rendering_seed = seed_state.seed_name
        seed_state.status = "rendering"
        seed_state.last_error = None

    try:
        raw_payload = torch.load(seed_state.ckpt_path, map_location="cpu", weights_only=False)
        env_type = infer_env_type(raw_payload, args.env_type)

        resolved_scenarios: List[Tuple[str, Dict[str, Any]]] = []
        seen_keys = set()
        for q in state.scenario_queries:
            s_key, s_spec = resolve_scenario_choice(env_type, q)
            if s_key not in seen_keys:
                seen_keys.add(s_key)
                resolved_scenarios.append((s_key, s_spec))

        agent, _, _, cfg, payload = load_checkpoint_for_env(
            seed_state.ckpt_path, env_type=env_type, device=args.device
        )

        episode = int(payload.get("episode", 0))
        total_env_steps = int(payload.get("total_env_steps", 0))

        video_dir = resolve_video_dir(
            state.videos_root, state.input_dir, seed_state.seed_name, seed_state.ckpt_key
        )
        video_dir.mkdir(parents=True, exist_ok=True)

        env = create_rendering_env(env_type, cfg)
        try:
            for scenario_key, scenario_spec in resolved_scenarios:
                video_path, meta_path = make_unique_video_paths(
                    video_dir, scenario_key, episode, total_env_steps
                )
                tmp_video = video_path.with_suffix(".tmp.mp4")
                tmp_meta = meta_path.with_suffix(".tmp.json")

                render_res = render_scenario_video(
                    agent=agent,
                    env=env,
                    scenario_key=scenario_key,
                    scenario_spec=scenario_spec,
                    out_path=tmp_video,
                    seed=int(payload.get("seed", seed_state.seed_num)),
                    total_env_steps=total_env_steps,
                    episode=episode,
                    env_name=env_type,
                    max_steps=args.max_steps,
                    fps=args.fps,
                    overlay_hud=not args.no_hud,
                    deterministic=bool(getattr(args, "deterministic", False)),
                    orbit_camera=args.orbit_camera,
                    track_camera=not args.no_track_camera,
                )

                meta = build_metadata(
                    seed_state=seed_state,
                    ckpt_stat=ckpt_stat,
                    payload=payload,
                    render_res=render_res,
                    scenario_spec=scenario_spec,
                    env_type=env_type,
                )
                tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                replace_file_with_retry(tmp_video, video_path)
                replace_file_with_retry(tmp_meta, meta_path)

                with state.lock:
                    if scenario_key not in seed_state.scenarios:
                        seed_state.scenario_order.append(scenario_key)
                        seed_state.scenarios[scenario_key] = {"version": 0}
                    prev_ver = int(seed_state.scenarios[scenario_key].get("version", 0))
                    new_scen_ver = prev_ver + 1
                    seed_state.scenarios[scenario_key].update({
                        "scenario_key": scenario_key,
                        "display_name": scenario_spec.get("display_name", scenario_key),
                        "video_path": video_path,
                        "meta_path": meta_path,
                        "version": new_scen_ver,
                        "last_render_at": meta["rendered_at"],
                        "metadata": meta,
                    })
                    seed_state.video_dir = video_dir
                    seed_state.version += 1
                    seed_state.last_render_at = meta["rendered_at"]

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"[Render {seed_state.seed_name} | {scenario_key} v{new_scen_ver}] "
                    f"Ep {meta['episode']} | Step {meta['total_env_steps']:,} | "
                    f"Ret {meta['rollout']['return']:+.1f} | Dist {meta['rollout']['total_distance']:+.2f}m | "
                    f"Saved -> {video_path}",
                    flush=True,
                )
        finally:
            env.close()

        with state.lock:
            seed_state.last_stat = ckpt_stat
            seed_state.status = "watching"
            seed_state.last_error = None
            state.last_error = None
    finally:
        with state.lock:
            if state.active_rendering_seed == seed_state.seed_name:
                state.active_rendering_seed = None


def checkpoint_watcher_loop(
    state: ViewerState, args: argparse.Namespace, stop_event: threading.Event
) -> None:
    """Polls all watched seed checkpoints for updates, verifies readiness, and triggers rendering."""
    while not stop_event.is_set():
        if state.watch_all_seeds:
            for s_name, s_num in discover_seeds(state.input_dir):
                existed = s_name in state.seeds
                state.ensure_seed(s_name, s_num)
                if not existed:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] [Watcher] Discovered seed: {s_name}",
                        flush=True,
                    )

        seed_states = state.get_seed_states()
        if not seed_states:
            with state.lock:
                state.status = "waiting_for_checkpoint"
            stop_event.wait(args.poll_interval)
            continue

        for seed_state in seed_states:
            if stop_event.is_set():
                break

            ready, cur_stat = is_checkpoint_ready(seed_state.ckpt_path)
            if not ready or cur_stat is None:
                with state.lock:
                    if seed_state.status != "rendering":
                        seed_state.status = "waiting_for_checkpoint"
                continue

            if cur_stat != seed_state.last_stat:
                if not wait_for_stable_stat(seed_state.ckpt_path, cur_stat, settle_sec=0.5):
                    continue
                try:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] [Watcher] Detected ready checkpoint "
                        f"({seed_state.seed_name}): {seed_state.ckpt_path.name}",
                        flush=True,
                    )
                    render_once(state, seed_state, cur_stat, args)
                except Exception as exc:
                    with state.lock:
                        seed_state.status = "error"
                        seed_state.last_error = str(exc)
                        state.last_error = f"{seed_state.seed_name}: {exc}"
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"[Watcher Error {seed_state.seed_name}] {exc}",
                        flush=True,
                    )
            else:
                with state.lock:
                    if seed_state.status not in ("watching", "rendering"):
                        seed_state.status = "watching"

        with state.lock:
            statuses = [s.status for s in state.seeds.values()]
            if any(st == "rendering" for st in statuses):
                state.status = "rendering"
            elif any(st == "watching" for st in statuses):
                state.status = "watching"
            elif statuses and all(st == "error" for st in statuses):
                state.status = "error"
            else:
                state.status = "waiting_for_checkpoint"

        stop_event.wait(args.poll_interval)


# ==============================================================================
# Web UI HTML & HTTP Request Handler
# ==============================================================================

def build_html_page(snap: Dict[str, Any]) -> str:
    """Builds the responsive dark-mode spider_viewer dashboard HTML."""
    meta = snap.get("metadata") or {}
    rollout = meta.get("rollout") or {}
    seed_order = snap.get("seed_order") or []
    initial_seed = snap.get("seed_name") or (seed_order[0] if seed_order else "all seeds")
    title_seed = html_lib.escape(str(initial_seed))
    title_ckpt = html_lib.escape(str(snap.get("ckpt_key", "")))
    title_scen = html_lib.escape(str(rollout.get("display_name") or snap.get("scenario_key", "")))
    version = int(snap.get("version", 0))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Spider Viewer &mdash; {title_seed} / {title_ckpt}</title>
    <style>
        :root {{
            --bg: #0b1120;
            --surface: #111827;
            --surface-2: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --green: #22c55e;
            --amber: #f59e0b;
            --red: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 24px;
            line-height: 1.5;
        }}
        .wrap {{ max-width: 1280px; margin: 0 auto; }}
        .header {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding-bottom: 18px;
            margin-bottom: 18px;
            border-bottom: 1px solid var(--border);
        }}
        .header h1 {{ font-size: 1.55rem; font-weight: 700; }}
        .sub {{ color: var(--muted); font-size: 0.9rem; margin-top: 4px; }}
        .seed-bar {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            padding: 12px 14px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }}
        .seed-bar-label {{
            font-size: 0.82rem;
            color: var(--muted);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-right: 4px;
        }}
        .seed-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            flex: 1;
        }}
        .tab-btn.rendering-seed {{
            border-color: rgba(245, 158, 11, 0.55);
        }}
        .tab-badge.rendering {{
            background: rgba(245, 158, 11, 0.2);
            color: var(--amber);
        }}
        .tab-badge.error {{
            background: rgba(239, 68, 68, 0.2);
            color: var(--red);
        }}
        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            background: rgba(34, 197, 94, 0.14);
            color: var(--green);
            border: 1px solid rgba(34, 197, 94, 0.35);
        }}
        .pill.rendering {{
            background: rgba(245, 158, 11, 0.14);
            color: var(--amber);
            border-color: rgba(245, 158, 11, 0.35);
        }}
        .pill.error {{
            background: rgba(239, 68, 68, 0.14);
            color: var(--red);
            border-color: rgba(239, 68, 68, 0.35);
        }}
        .dot {{
            width: 8px; height: 8px; border-radius: 50%;
            background: currentColor;
        }}
        .grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.5fr) minmax(300px, 1fr);
            gap: 20px;
            align-items: start;
        }}
        @media (max-width: 960px) {{
            .grid {{ grid-template-columns: 1fr; }}
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }}
        .card-head {{
            padding: 14px 18px;
            background: var(--surface-2);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .card-head h2 {{ font-size: 1rem; font-weight: 600; }}
        .card-body {{ padding: 18px; }}
        video {{
            width: 100%;
            border-radius: 8px;
            background: #000;
            display: block;
        }}
        .scenario-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 14px;
        }}
        .tab-btn {{
            background: var(--surface-2);
            color: var(--muted);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 0.86rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}
        .tab-btn:hover {{ border-color: var(--accent); color: var(--text); }}
        .tab-btn.active {{
            background: rgba(56, 189, 248, 0.16);
            border-color: var(--accent);
            color: var(--accent);
        }}
        .tab-badge {{
            font-size: 0.76rem;
            padding: 2px 6px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            color: var(--text);
        }}
        .history-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 14px;
            padding: 10px 12px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
        }}
        .history-bar label {{
            font-size: 0.82rem;
            color: var(--muted);
            font-weight: 600;
        }}
        .history-select {{
            flex: 1;
            min-width: 220px;
            background: var(--bg);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 0.85rem;
        }}
        .toolbar {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
            align-items: center;
        }}
        .btn {{
            background: var(--surface-2);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 0.88rem;
            font-weight: 500;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .btn:hover {{ border-color: var(--accent); color: var(--accent); }}
        .btn.primary {{
            background: rgba(56, 189, 248, 0.16);
            border-color: rgba(56, 189, 248, 0.45);
            color: var(--accent);
        }}
        .toggle {{
            margin-left: auto;
            font-size: 0.85rem;
            color: var(--muted);
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
        }}
        .stats-stack {{ display: flex; flex-direction: column; gap: 16px; }}
        .kv-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }}
        .kv {{
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 12px;
        }}
        .kv .k {{ font-size: 0.76rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
        .kv .v {{ font-size: 1.05rem; font-weight: 600; margin-top: 2px; word-break: break-word; }}
        .empty-video {{
            padding: 64px 20px;
            text-align: center;
            color: var(--muted);
            border: 1px dashed var(--border);
            border-radius: 8px;
        }}
        .bench-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            margin-top: 10px;
        }}
        .bench-table th, .bench-table td {{
            text-align: left;
            padding: 6px 8px;
            border-bottom: 1px solid var(--border);
        }}
        .bench-table th {{ color: var(--muted); font-weight: 500; }}
        .path-note {{
            margin-top: 10px;
            font-size: 0.78rem;
            color: var(--muted);
            word-break: break-all;
        }}
    </style>
</head>
<body>
<div class="wrap">
    <div class="header">
        <div>
            <h1 id="headerTitle">Spider Viewer &mdash; {title_seed} ({title_ckpt})</h1>
            <div class="sub" id="subLine">Scenario: {title_scen}</div>
        </div>
        <div class="pill" id="statusPill"><span class="dot"></span><span id="statusText">Loading...</span></div>
    </div>

    <div class="seed-bar" id="seedBar" style="display:none;">
        <span class="seed-bar-label">Seed:</span>
        <div class="seed-tabs" id="seedTabs"></div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-head">
                <h2>Rollout Video</h2>
                <span class="sub" id="renderInfo">Version #{version}</span>
            </div>
            <div class="card-body">
                <div class="scenario-tabs" id="scenarioTabs"></div>
                <div class="history-bar">
                    <label for="historySelect">Saved Videos:</label>
                    <select id="historySelect" class="history-select" onchange="onHistoryChange(this.value)">
                        <option value="">Latest (Auto-update)</option>
                    </select>
                    <button class="btn" id="backToLatestBtn" onclick="selectHistoryFile('')" style="display:none;">Back to Latest</button>
                </div>
                <div id="videoContainer">
                    <video id="player" controls autoplay muted loop playsinline preload="metadata"
                           style="display:none;"></video>
                    <div id="emptyState" class="empty-video">
                        Waiting for initial video render...
                    </div>
                </div>
                <div class="toolbar">
                    <button class="btn primary" onclick="togglePlay()">Play / Pause</button>
                    <button class="btn" onclick="reloadVideo()">Reload Video</button>
                    <a class="btn" id="downloadBtn" href="/video?download=1" download>Download MP4</a>
                    <label class="toggle">
                        <input type="checkbox" id="autoReloadCb" checked />
                        Auto-update on new checkpoint
                    </label>
                </div>
                <div class="path-note" id="pathNote"></div>
            </div>
        </div>

        <div class="stats-stack">
            <div class="card">
                <div class="card-head"><h2>Checkpoint &amp; Run</h2></div>
                <div class="card-body">
                    <div class="kv-grid" id="ckptStats"></div>
                </div>
            </div>

            <div class="card">
                <div class="card-head"><h2>Rollout Performance</h2></div>
                <div class="card-body">
                    <div class="kv-grid" id="rolloutStats"></div>
                </div>
            </div>

            <div class="card">
                <div class="card-head"><h2>Training &amp; Benchmarks</h2></div>
                <div class="card-body">
                    <div class="kv-grid" id="trainStats"></div>
                    <div id="benchContainer"></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
const urlParams = new URLSearchParams(window.location.search);
let lastSnapshot = null;
let selectedSeed = urlParams.get("seed") || {json.dumps(snap.get("seed_name", ""))};
let selectedScenario = {json.dumps(snap.get("scenario_key", "standard_ant"))};
let selectedHistoryFile = "";
let shownSourceKey = null;
let shownVersions = {{}};

function fmtNum(v, digits = 2, signed = false) {{
    if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
    const n = Number(v);
    const s = n.toFixed(digits);
    return (signed && n > 0) ? "+" + s : s;
}}

function fmtInt(v) {{
    if (v === null || v === undefined) return "-";
    return Number(v).toLocaleString();
}}

function renderKV(containerId, items) {{
    const el = document.getElementById(containerId);
    el.innerHTML = items.map(([k, v]) =>
        `<div class="kv"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`
    ).join("");
}}

function getSelectedSeedData(data) {{
    if (!data) return {{}};
    const seeds = data.seeds || {{}};
    if (selectedSeed && seeds[selectedSeed]) {{
        return seeds[selectedSeed];
    }}
    const order = data.seed_order || [];
    if (order.length > 0 && seeds[order[0]]) {{
        return seeds[order[0]];
    }}
    return data;
}}

function applyVideo(seedName, scenKey, ver, exists, historyFile, forcePlay) {{
    const player = document.getElementById("player");
    const empty = document.getElementById("emptyState");
    const dl = document.getElementById("downloadBtn");
    const seedParam = seedName ? `seed=${{encodeURIComponent(seedName)}}&` : "";
    const fileParam = historyFile ? `&file=${{encodeURIComponent(historyFile)}}` : "";
    dl.href = `/video?${{seedParam}}scenario=${{encodeURIComponent(scenKey)}}${{fileParam}}&download=1`;

    if (!exists && !historyFile) {{
        player.style.display = "none";
        empty.style.display = "block";
        shownSourceKey = null;
        return;
    }}
    empty.style.display = "none";
    player.style.display = "block";

    const verKey = `${{seedName}}::${{scenKey}}`;
    const sourceKey = historyFile
        ? `${{seedName}}::${{scenKey}}::${{historyFile}}`
        : `${{seedName}}::${{scenKey}}::latest`;
    const sourceChanged = (shownSourceKey !== sourceKey);
    const versionChanged = (!historyFile && shownVersions[verKey] !== ver);
    const autoReload = document.getElementById("autoReloadCb").checked;
    if (forcePlay || sourceChanged || (versionChanged && (shownVersions[verKey] === undefined || autoReload))) {{
        const wasPaused = player.paused && shownSourceKey !== null && !sourceChanged;
        player.src = `/video?${{seedParam}}scenario=${{encodeURIComponent(scenKey)}}${{fileParam}}&v=${{ver}}&t=${{Date.now()}}`;
        player.load();
        if (!wasPaused || forcePlay || sourceChanged) {{
            player.play().catch(() => {{}});
        }}
        if (!historyFile) {{
            shownVersions[verKey] = ver;
        }}
        shownSourceKey = sourceKey;
    }}
}}

function selectSeed(seedName) {{
    selectedSeed = seedName;
    selectedHistoryFile = "";
    try {{
        const url = new URL(window.location.href);
        url.searchParams.set("seed", seedName);
        window.history.replaceState({{}}, "", url.toString());
    }} catch (_e) {{}}
    if (lastSnapshot) {{
        updateUI(lastSnapshot, true);
    }}
}}

function selectScenario(scenKey) {{
    selectedScenario = scenKey;
    selectedHistoryFile = "";
    if (lastSnapshot) {{
        updateUI(lastSnapshot, true);
    }}
}}

function onHistoryChange(val) {{
    selectHistoryFile(val);
}}

function selectHistoryFile(filename) {{
    selectedHistoryFile = filename || "";
    if (lastSnapshot) {{
        updateUI(lastSnapshot, true);
    }}
}}

function togglePlay() {{
    const player = document.getElementById("player");
    if (player.paused) player.play().catch(() => {{}});
    else player.pause();
}}

function reloadVideo() {{
    if (!lastSnapshot) return;
    const seedData = getSelectedSeedData(lastSnapshot);
    const scen = (seedData.scenarios || {{}})[selectedScenario] || {{}};
    applyVideo(selectedSeed, selectedScenario, Number(scen.version || 0), Boolean(scen.video_exists), selectedHistoryFile, true);
}}

function renderSeedTabs(data) {{
    const bar = document.getElementById("seedBar");
    const tabsEl = document.getElementById("seedTabs");
    const order = data.seed_order || [];
    const seeds = data.seeds || {{}};
    if (order.length === 0) {{
        bar.style.display = "none";
        tabsEl.innerHTML = "";
        return;
    }}
    bar.style.display = "flex";
    tabsEl.innerHTML = order.map(name => {{
        const s = seeds[name] || {{}};
        const active = name === selectedSeed ? "active" : "";
        const isRendering = (s.status === "rendering" || data.active_rendering_seed === name);
        const renderCls = isRendering ? "rendering-seed" : "";
        let badge = "Pending";
        let badgeCls = "";
        if (isRendering) {{
            badge = "Rendering...";
            badgeCls = "rendering";
        }} else if (s.status === "error") {{
            badge = "Error";
            badgeCls = "error";
        }} else if (s.metadata && s.metadata.episode !== undefined && s.metadata.episode !== null && s.version > 0) {{
            badge = `Ep ${{fmtInt(s.metadata.episode)}}`;
        }} else if (s.video_exists) {{
            badge = "Ready";
        }}
        return `<button class="tab-btn ${{active}} ${{renderCls}}" onclick="selectSeed('${{name}}')">
            <span>${{name}}</span>
            <span class="tab-badge ${{badgeCls}}">${{badge}}</span>
        </button>`;
    }}).join("");
}}

function renderHistorySelect(scen) {{
    const sel = document.getElementById("historySelect");
    const backBtn = document.getElementById("backToLatestBtn");
    const history = scen.history || [];
    if (selectedHistoryFile && !history.some(h => h.filename === selectedHistoryFile)) {{
        selectedHistoryFile = "";
    }}
    let opts = `<option value="">Latest (Auto-update)${{history.length ? ` — ${{history.length}} saved` : ""}}</option>`;
    for (const h of history) {{
        const ep = h.episode !== null && h.episode !== undefined ? `Ep ${{h.episode}}` : h.filename;
        const st = h.total_env_steps !== null && h.total_env_steps !== undefined ? `Step ${{fmtInt(h.total_env_steps)}}` : "";
        const ret = h.return !== null && h.return !== undefined ? `Ret ${{fmtNum(h.return, 1, true)}}` : "";
        const dist = h.total_distance !== null && h.total_distance !== undefined ? `${{fmtNum(h.total_distance, 2, true)}}m` : "";
        const label = [ep, st, ret, dist, h.rendered_at].filter(Boolean).join(" | ");
        const selectedAttr = h.filename === selectedHistoryFile ? " selected" : "";
        opts += `<option value="${{h.filename}}"${{selectedAttr}}>${{label}}</option>`;
    }}
    sel.innerHTML = opts;
    sel.value = selectedHistoryFile;
    backBtn.style.display = selectedHistoryFile ? "inline-flex" : "none";
}}

function renderScenarioTabs(seedData) {{
    const tabsEl = document.getElementById("scenarioTabs");
    const order = seedData.scenario_order || [];
    const scens = seedData.scenarios || {{}};
    tabsEl.innerHTML = order.map(key => {{
        const s = scens[key] || {{}};
        const r = (s.metadata || {{}}).rollout || {{}};
        const active = key === selectedScenario ? "active" : "";
        const badge = s.video_exists
            ? `Ret ${{fmtNum(r.return, 0, true)}} | ${{fmtNum(r.total_distance, 1, true)}}m`
            : "Pending";
        return `<button class="tab-btn ${{active}}" onclick="selectScenario('${{key}}')">
            <span>${{s.display_name || key}}</span>
            <span class="tab-badge">${{badge}}</span>
        </button>`;
    }}).join("");
}}

function updateUI(data, forceSwitch = false) {{
    lastSnapshot = data;
    const seedOrder = data.seed_order || [];
    if (seedOrder.length > 0 && !seedOrder.includes(selectedSeed)) {{
        selectedSeed = seedOrder[0];
    }}

    const seedData = getSelectedSeedData(data);
    const order = seedData.scenario_order || [];
    if (order.length > 0 && !order.includes(selectedScenario)) {{
        selectedScenario = order[0];
    }}

    const displaySeed = selectedSeed || (data.watch_all_seeds ? "all seeds" : "-");
    document.getElementById("headerTitle").textContent =
        `Spider Viewer — ${{displaySeed}} (${{data.ckpt_key || "-"}})`;
    document.title = `Spider Viewer — ${{displaySeed}} / ${{data.ckpt_key || "-"}}`;

    const pill = document.getElementById("statusPill");
    const statusText = document.getElementById("statusText");
    pill.className = "pill";
    if (data.status === "rendering") {{
        pill.classList.add("rendering");
        statusText.textContent = data.active_rendering_seed
            ? `Rendering ${{data.active_rendering_seed}}...`
            : "Rendering videos...";
    }} else if (data.status === "error") {{
        pill.classList.add("error");
        statusText.textContent = "Error: " + (data.last_error || "unknown");
    }} else if (data.status === "waiting_for_checkpoint") {{
        pill.classList.add("rendering");
        statusText.textContent = seedOrder.length === 0
            ? "Waiting for seed checkpoints..."
            : "Waiting for checkpoint...";
    }} else {{
        statusText.textContent = seedOrder.length > 1
            ? `Watching ${{seedOrder.length}} seeds`
            : "Watching checkpoint";
    }}

    renderSeedTabs(data);
    renderScenarioTabs(seedData);

    const scen = (seedData.scenarios || {{}})[selectedScenario] || {{}};
    renderHistorySelect(scen);

    const histEntry = selectedHistoryFile
        ? (scen.history || []).find(h => h.filename === selectedHistoryFile)
        : null;
    const m = (histEntry && histEntry.metadata && Object.keys(histEntry.metadata).length > 0)
        ? histEntry.metadata
        : (scen.metadata || seedData.metadata || {{}});
    const r = m.rollout || {{}};
    const p = m.progress || {{}};
    const win = m.win_evaluator || {{}};
    const vel = m.vel_curriculum || {{}};
    const benches = m.benchmark_metrics || {{}};
    const ver = Number(scen.version || 0);

    document.getElementById("renderInfo").textContent = selectedHistoryFile
        ? `${{displaySeed}} • ${{scen.display_name || selectedScenario}} • History: ${{selectedHistoryFile}}`
        : (ver > 0 ? `${{displaySeed}} • ${{scen.display_name || selectedScenario}} • v${{ver}} • Rendered ${{m.rendered_at || "-"}}` : "Not rendered yet");
    document.getElementById("subLine").textContent =
        `Run: ${{m.run_name || data.run_name || "-"}} • Seed: ${{displaySeed}} • Env: ${{(m.env_type || "-").toUpperCase()}} • Scenario: ${{r.display_name || selectedScenario}}`;
    document.getElementById("pathNote").textContent =
        `Video: ${{selectedHistoryFile || scen.video_path || "-"}} | Checkpoint: ${{seedData.ckpt_path || "-"}}`;

    renderKV("ckptStats", [
        ["Checkpoint", m.checkpoint_file || seedData.ckpt_key || data.ckpt_key || "-"],
        ["Episode", fmtInt(m.episode)],
        ["Total Steps", fmtInt(m.total_env_steps)],
        ["Ckpt Modified", m.checkpoint_mtime || "-"],
    ]);

    renderKV("rolloutStats", [
        ["Return", fmtNum(r.return, 1, true)],
        ["Distance (m)", fmtNum(r.total_distance, 2, true)],
        ["X Velocity (m/s)", fmtNum(r.x_velocity, 3, true)],
        ["Steps", `${{fmtInt(r.steps)}} (${{fmtNum(r.wall_sec, 1)}}s render)`],
    ]);

    const epProg = (p.completed_episodes !== undefined && p.target_episodes !== undefined)
        ? `${{fmtInt(p.completed_episodes)}} / ${{fmtInt(p.target_episodes)}}`
        : "-";
    renderKV("trainStats", [
        ["Composite Score", fmtNum(m.composite_score, 2)],
        ["Best Score", fmtNum(p.best_composite_score, 2)],
        ["Run Progress", epProg],
        ["EMA Win / Vel", `${{fmtNum(win.ema_win_rate, 3)}} / ${{fmtNum(vel.current_vel, 2)}} m/s`],
    ]);

    const benchKeys = Object.keys(benches);
    const benchEl = document.getElementById("benchContainer");
    if (benchKeys.length > 0) {{
        const rows = benchKeys.map(k => {{
            const b = benches[k] || {{}};
            return `<tr><td>${{k}}</td><td>${{fmtNum(b.return, 1, true)}}</td><td>${{fmtNum(b.distance, 2, true)}}m</td></tr>`;
        }}).join("");
        benchEl.innerHTML = `<table class="bench-table"><thead><tr><th>Benchmark</th><th>Return</th><th>Dist</th></tr></thead><tbody>${{rows}}</tbody></table>`;
    }} else {{
        benchEl.innerHTML = "";
    }}

    applyVideo(selectedSeed, selectedScenario, ver, Boolean(scen.video_exists), selectedHistoryFile, forceSwitch);
}}

async function pollStatus() {{
    try {{
        const resp = await fetch("/api/status", {{ cache: "no-store" }});
        if (!resp.ok) return;
        const data = await resp.json();
        updateUI(data);
    }} catch (_e) {{}}
}}

pollStatus();
setInterval(pollStatus, 4000);
</script>
</body>
</html>
"""


class SpiderViewerHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the dashboard, JSON status API, and Range-enabled MP4 stream."""

    viewer_state: ViewerState

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            snap = self.viewer_state.snapshot()
            body = build_html_page(snap).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/status":
            snap = self.viewer_state.snapshot()
            body = json.dumps(snap).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/video":
            qs = parse_qs(parsed.query or "")
            seed_param = qs.get("seed", [None])[0]
            scen_param = qs.get("scenario", [None])[0]
            file_param = qs.get("file", [None])[0]
            dl_param = qs.get("download", ["0"])[0] == "1"
            self._serve_video(
                seed_name=seed_param,
                scenario_key=scen_param,
                filename=file_param,
                download=dl_param,
            )
            return

        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _serve_video(
        self,
        seed_name: Optional[str] = None,
        scenario_key: Optional[str] = None,
        filename: Optional[str] = None,
        download: bool = False,
    ) -> None:
        video_path = self.viewer_state.get_video_path(
            seed_name=seed_name, scenario_key=scenario_key, filename=filename
        )
        if video_path is None or not video_path.exists() or video_path.stat().st_size == 0:
            self.send_error(HTTPStatus.NOT_FOUND, "Video not rendered yet")
            return

        file_size = video_path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
            if m:
                s_str, e_str = m.group(1), m.group(2)
                if s_str:
                    start = int(s_str)
                    end = int(e_str) if e_str else file_size - 1
                elif e_str:
                    suffix_len = int(e_str)
                    start = max(0, file_size - suffix_len)
                    end = file_size - 1
                end = min(end, file_size - 1)
                if start > end or start >= file_size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{video_path.name}"')
        self.end_headers()

        chunk_size = 256 * 1024
        with video_path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    break
                remaining -= len(chunk)

    def log_message(self, format: str, *args) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]} - {args[1]}", flush=True)


# ==============================================================================
# Entrypoint
# ==============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spider Viewer: live checkpoint video renderer and HTTP dashboard."
    )
    parser.add_argument(
        "--input-dir", "-d",
        type=str,
        required=True,
        help="Training output folder (e.g. outputs/validation).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=str,
        default=None,
        help="Seed folder or ID (e.g. seed_42 or 42). If omitted, watches and renders all seeds in --input-dir.",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="latest",
        help="Checkpoint name: 'latest' (default), 'best', 'final', or checkpoint stem/filename.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["standard", "spider", "hexapod", "harsh"],
        help="Validation scenarios to render and serve (default: standard spider hexapod harsh).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional single-scenario override (overrides --scenarios if provided).",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="auto",
        help="Environment type: auto (default), ant, humanoid, or dog.",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=str(ROOT_DIR / "videos"),
        help="Base directory where videos/<run>/<seed>/<ckpt>/ will be written (default: ./videos).",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", "-p", type=int, default=8081, help="HTTP port (default: 8081).")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between checkpoint change polls (default: 5.0).",
    )
    parser.add_argument("--max-steps", type=int, default=1000, help="Max rollout steps per clip (default: 1000).")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default: cpu).")
    parser.add_argument("--stochastic", action="store_true", default=True, help="Use stochastic action sampling matching training (default: True).")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic mode-center action selection.")
    parser.add_argument("--no-hud", action="store_true", help="Disable telemetry HUD overlay on video.")
    parser.add_argument("--orbit-camera", action="store_true", help="Orbit camera during video capture.")
    parser.add_argument("--no-track-camera", action="store_true", help="Disable torso-tracking camera.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    videos_root = Path(args.videos_dir)
    scenario_queries = [args.scenario] if args.scenario else list(args.scenarios)

    if args.seed is not None:
        initial_seeds = [normalize_seed(args.seed)]
        watch_all_seeds = False
    else:
        initial_seeds = discover_seeds(input_dir)
        watch_all_seeds = True

    state = ViewerState(
        input_dir=input_dir,
        seeds=initial_seeds,
        ckpt_arg=args.checkpoint,
        scenario_queries=scenario_queries,
        videos_root=videos_root,
        env_type=args.env_type,
        watch_all_seeds=watch_all_seeds,
    )

    stop_event = threading.Event()
    watcher = threading.Thread(
        target=checkpoint_watcher_loop,
        args=(state, args, stop_event),
        daemon=True,
    )
    watcher.start()

    SpiderViewerHandler.viewer_state = state
    server = ThreadingHTTPServer((args.host, args.port), SpiderViewerHandler)
    lan_ip = get_local_ip()

    seeds_desc = (
        f"{', '.join(state.seed_order) or '(none yet - auto-discovering)'}"
        if watch_all_seeds
        else (state.seed_order[0] if state.seed_order else "-")
    )
    run_videos_dir = videos_root / (input_dir.resolve().name or "run")

    print("=" * 78, flush=True)
    print(f"Spider Viewer running!", flush=True)
    print(f"  Local URL   : http://localhost:{args.port}", flush=True)
    print(f"  Network URL : http://{lan_ip}:{args.port}", flush=True)
    print(f"  Input Dir   : {input_dir.resolve()}", flush=True)
    print(f"  Seeds       : {seeds_desc}" + (" [all-seeds mode]" if watch_all_seeds else ""), flush=True)
    print(f"  Checkpoint  : {args.checkpoint} (key: {state.ckpt_key})", flush=True)
    print(f"  Videos Dir  : {run_videos_dir}", flush=True)
    print(f"  Scenarios   : {', '.join(scenario_queries)} | Poll: {args.poll_interval}s | Device: {args.device}", flush=True)
    print("=" * 78, flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down spider_viewer...", flush=True)
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
