"""Progress & Validation Visualization Tool for LA-MCTS and Balanced RL.

Ingests `progress.jsonl` files (and accompanying `.log` files) to generate
high-resolution publication-quality performance dashboards tracking:
  1. Episode Returns & Dynamic Win Threshold (r_ref)
  2. Locomotion Distance (Train & Eval Scenarios)
  3. Locomotion Velocity (vx, Train & Eval Scenarios)
  4. Survival & Episode Duration (Steps per Episode)
  5. Representation & Model Dynamics Loss (L_total)
  6. BRL Adversarial Balance (Win Rate & Balancer Budget)
  7. Validation & OOD Benchmark Suite Returns
  8. Aggregate Validation Score (Composite Score, Std-Ant, OOD & New Best Milestones)
  9. Compute Efficiency (Wall-Clock Episode Time)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def parse_accompanying_log(log_path: Path) -> Dict[str, Any]:
    """Parses evaluation benchmarks, aggregate scores, and distance/velocity metrics from a training .log file."""
    if not log_path.exists():
        return {}

    eval_by_ep: Dict[int, Dict[str, Dict[str, float]]] = {}
    eval_scores: Dict[int, float] = {}
    eval_summary: Dict[int, Dict[str, Any]] = {}

    # Regex patterns
    # e.g.: [2026-11-15 07:55:00] >>> [Seed 42 (1/10) | Eval @ Ep 2000 (1,906,499 steps)] Score: +952.13 | Std-Ant: +966.7 | OOD: +937.5 [★ NEW BEST]
    eval_hdr_re = re.compile(
        r"Eval @ Ep\s+(\d+)\s+\(\s*([\d,]+)\s+steps\)\s*\]\s+Score:\s*([+-]?\d+(?:\.\d+)?)"
        r"(?:\s*\|\s*Std-Ant:\s*([+-]?\d+(?:\.\d+)?))?"
        r"(?:\s*\|\s*OOD:\s*([+-]?\d+(?:\.\d+)?))?"
    )
    # e.g.: * Standard-Ant-v5 : Return = +966.72 | Steps = 1000/1000 | Dist =  +2.60m | Vel = +0.052 m/s
    suite_line_re = re.compile(
        r"\*\s+([\w\-]+)\s*:\s+Return\s*=\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*Steps\s*=\s*(\d+)/\d+\s*\|\s*Dist\s*=\s*([+-]?\d+(?:\.\d+)?)m\s*\|\s*Vel\s*=\s*([+-]?\d+(?:\.\d+)?)\s*m/s"
    )

    # Scenario lines are logged before the `>>> [Seed ... | Eval @ Ep ...]` summary line,
    # so buffer them until the matching Eval header arrives.
    pending_scenarios: Dict[str, Dict[str, float]] = {}
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                suite_m = suite_line_re.search(line)
                if suite_m:
                    scenario = suite_m.group(1).strip()
                    pending_scenarios[scenario] = {
                        "return": float(suite_m.group(2)),
                        "steps": float(int(suite_m.group(3))),
                        "total_distance": float(suite_m.group(4)),
                        "x_velocity": float(suite_m.group(5)),
                    }
                    continue

                hdr_m = eval_hdr_re.search(line)
                if hdr_m:
                    ep = int(hdr_m.group(1))
                    env_steps = int(hdr_m.group(2).replace(",", ""))
                    score = float(hdr_m.group(3))
                    std_ant = float(hdr_m.group(4)) if hdr_m.group(4) is not None else np.nan
                    ood = float(hdr_m.group(5)) if hdr_m.group(5) is not None else np.nan
                    new_best = "NEW BEST" in line

                    eval_scores[ep] = score
                    eval_summary[ep] = {
                        "env_steps": env_steps,
                        "score": score,
                        "std_ant": std_ant,
                        "ood": ood,
                        "new_best": new_best,
                    }
                    if pending_scenarios:
                        eval_by_ep[ep] = pending_scenarios
                        pending_scenarios = {}
                    elif ep not in eval_by_ep:
                        eval_by_ep[ep] = {}
                    continue
    except Exception:
        pass

    return {
        "eval_by_ep": eval_by_ep,
        "eval_scores": eval_scores,
        "eval_summary": eval_summary,
    }


def parse_progress_jsonl(jsonl_path: Path) -> Dict[str, Any]:
    """Reads a progress.jsonl file into structured NumPy arrays."""
    records: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return {}

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    if not records:
        return {}

    episodes = np.array([r.get("episode", i + 1) for i, r in enumerate(records)], dtype=np.int32)
    total_env_steps = np.array([r.get("total_env_steps", 0) for r in records], dtype=np.int64)
    returns = np.array([r.get("return", 0.0) for r in records], dtype=np.float64)
    has_agent_return = any("agent_return" in r for r in records)
    if has_agent_return:
        agent_returns = np.array(
            [float(r.get("agent_return", r.get("return", 0.0))) for r in records],
            dtype=np.float64,
        )
    else:
        agent_returns = np.full(len(records), np.nan, dtype=np.float64)
    steps = np.array([r.get("steps", 0) for r in records], dtype=np.int32)
    losses = np.array([r.get("loss_total", 0.0) for r in records], dtype=np.float64)
    win_rates = np.array([r.get("ema_win_rate", 0.0) for r in records], dtype=np.float64)
    r_refs = np.array([r.get("r_ref", 0.0) for r in records], dtype=np.float64)
    budgets = np.array([r.get("budget", 0.0) for r in records], dtype=np.float64)
    durations = np.array([r.get("ep_duration", 0.0) for r in records], dtype=np.float64)
    wins = np.array([1 if r.get("win", False) else 0 for r in records], dtype=np.int32)

    # Locomotion metrics (direct if present, otherwise default to nan)
    has_direct_dist = any("total_distance" in r for r in records)
    has_direct_vel = any("x_velocity" in r for r in records)
    has_ep_duration = any("ep_duration" in r and float(r.get("ep_duration", 0.0)) > 0.0 for r in records)
    if has_direct_dist:
        distances = np.array([float(r.get("total_distance", 0.0)) for r in records], dtype=np.float64)
    else:
        distances = np.full(len(records), np.nan, dtype=np.float64)

    if has_direct_vel:
        velocities = np.array([float(r.get("x_velocity", 0.0)) for r in records], dtype=np.float64)
    else:
        velocities = np.full(len(records), np.nan, dtype=np.float64)

    seed_val = records[0].get("seed", 0)

    # Check for accompanying log file
    log_cand1 = jsonl_path.with_name(f"seed_{seed_val}.log")
    log_cand2 = jsonl_path.parent / (jsonl_path.stem.replace("_progress", "") + ".log")
    log_path = log_cand1 if log_cand1.exists() else (log_cand2 if log_cand2.exists() else None)
    log_data = parse_accompanying_log(log_path) if log_path is not None else {}

    return {
        "seed": seed_val,
        "jsonl_path": str(jsonl_path),
        "log_path": str(log_path) if log_path else None,
        "episodes": episodes,
        "total_env_steps": total_env_steps,
        "returns": returns,
        "agent_returns": agent_returns,
        "has_agent_return": has_agent_return,
        "steps": steps,
        "losses": losses,
        "win_rates": win_rates,
        "r_refs": r_refs,
        "budgets": budgets,
        "durations": durations,
        "wins": wins,
        "distances": distances,
        "velocities": velocities,
        "has_direct_dist": has_direct_dist,
        "has_direct_vel": has_direct_vel,
        "has_ep_duration": has_ep_duration,
        "eval_by_ep": log_data.get("eval_by_ep", {}),
        "eval_scores": log_data.get("eval_scores", {}),
        "eval_summary": log_data.get("eval_summary", {}),
    }


SCENARIO_COLORS: Dict[str, str] = {
    "Standard-Ant-v5": "#2980b9",
    "Standard-Ant-v4": "#2980b9",
    "Tripod-Ant": "#d35400",
    "Spider-Ant": "#8e44ad",
    "Heavy-Ice-Ant": "#16a085",
    "Hexapod-Ant": "#c0392b",
}


def compute_rolling_mean(arr: np.ndarray, window: int = 20) -> np.ndarray:
    """Computes a causal rolling average with initial edge expansion."""
    if len(arr) == 0:
        return np.array([])
    window = max(1, min(window, len(arr)))
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.mean(arr[start : i + 1])
    return out


def _plot_eval_scenario_series(
    ax: plt.Axes,
    eval_by_ep: Dict[int, Dict[str, Dict[str, float]]],
    metric_key: str,
    marker: str = "o",
    label_suffix: str = "",
) -> bool:
    """Plots per-scenario evaluation series on the given axis; returns True if anything was plotted."""
    if not eval_by_ep:
        return False
    eval_eps = sorted(eval_by_ep.keys())
    scenarios = sorted({sc for ep in eval_eps for sc in eval_by_ep[ep].keys()})
    plotted = False
    for sc in scenarios:
        x_vals = [ep for ep in eval_eps if sc in eval_by_ep[ep] and metric_key in eval_by_ep[ep][sc]]
        y_vals = [eval_by_ep[ep][sc][metric_key] for ep in x_vals]
        if x_vals:
            ax.plot(
                x_vals,
                y_vals,
                marker=marker,
                markersize=4,
                linewidth=1.5,
                linestyle="-",
                color=SCENARIO_COLORS.get(sc, "#34495e"),
                label=f"{sc}{label_suffix}",
            )
            plotted = True
    return plotted


def plot_single_progress(
    data: Dict[str, Any],
    output_path: Path,
    title_suffix: str = "",
    rolling_window: int = 20,
) -> Path:
    """Renders a comprehensive 9-panel (3x3) progress dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), dpi=200)
    fig.patch.set_facecolor("#fcfcfc")

    seed = data.get("seed", "N/A")
    episodes = data["episodes"]
    env_steps = data["total_env_steps"]
    returns = data["returns"]
    agent_returns = data.get("agent_returns", np.full_like(returns, np.nan))
    has_agent_return = bool(data.get("has_agent_return", False)) and not np.all(np.isnan(agent_returns))
    steps = data["steps"]
    losses = data["losses"]
    win_rates = data["win_rates"]
    r_refs = data["r_refs"]
    budgets = data["budgets"]
    durations = data["durations"]
    distances = data["distances"]
    velocities = data["velocities"]
    has_direct_dist = data["has_direct_dist"]
    has_direct_vel = data.get("has_direct_vel", False)
    has_ep_duration = data.get("has_ep_duration", False)
    eval_by_ep = data["eval_by_ep"]
    eval_summary = data.get("eval_summary", {})

    ret_rolling = compute_rolling_mean(returns, rolling_window)

    # -------------------------------------------------------------
    # Panel 1: Episode Returns & Reference Target
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#ffffff")
    ax1.scatter(episodes, returns, color="#4a90e2", alpha=0.30, s=12, label="Raw Return")
    ax1.plot(episodes, ret_rolling, color="#1b4f72", linewidth=2.0, label=f"Raw Rolling (w={rolling_window})")
    if has_agent_return:
        ag_rolling = compute_rolling_mean(agent_returns, rolling_window)
        ax1.scatter(episodes, agent_returns, color="#9b59b6", alpha=0.20, s=10, label="Agent Return (Shaped)")
        ax1.plot(episodes, ag_rolling, color="#6c3483", linewidth=1.8, linestyle="-.", label=f"Agent Rolling (w={rolling_window})")
    if len(r_refs) > 0 and not np.all(r_refs == 0.0):
        ax1.plot(episodes, r_refs, color="#e67e22", linestyle="--", linewidth=1.5, label="Win Threshold (R_ref)")
    ax1.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)
    ax1.set_title("1. Cumulative Episode Return (Raw vs Shaped)", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="best", fontsize=7.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: Locomotion Distance (Meters Traveled)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#ffffff")
    has_plotted_dist = False
    if has_direct_dist and not np.all(np.isnan(distances)):
        dist_rolling = compute_rolling_mean(distances, rolling_window)
        ax2.scatter(episodes, distances, color="#27ae60", alpha=0.25, s=12, label="Train Dist")
        ax2.plot(episodes, dist_rolling, color="#196f3d", linewidth=2.0, label=f"Train Rolling (w={rolling_window})")
        has_plotted_dist = True

    if _plot_eval_scenario_series(ax2, eval_by_ep, "total_distance", marker="o", label_suffix=" (Eval)"):
        has_plotted_dist = True

    if not has_plotted_dist:
        ax2.text(
            0.5, 0.5,
            "Meters traveled not recorded in this run",
            horizontalalignment="center", verticalalignment="center",
            transform=ax2.transAxes, fontsize=10, color="#7f8c8d",
        )
    ax2.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)
    ax2.set_title("2. Locomotion Distance (Meters Traveled)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Distance (m)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    if has_plotted_dist:
        ax2.legend(loc="upper left", fontsize=7.5, framealpha=0.9, ncol=2)

    # -------------------------------------------------------------
    # Panel 3: Locomotion Velocity (v_x, m/s)
    # -------------------------------------------------------------
    ax3 = axes[0, 2]
    ax3.set_facecolor("#ffffff")
    has_plotted_vel = False
    if has_direct_vel and not np.all(np.isnan(velocities)):
        vel_rolling = compute_rolling_mean(velocities, rolling_window)
        ax3.scatter(episodes, velocities, color="#16a085", alpha=0.25, s=12, label="Train Vel")
        ax3.plot(episodes, vel_rolling, color="#0e6251", linewidth=2.0, label=f"Train Rolling (w={rolling_window})")
        has_plotted_vel = True

    if _plot_eval_scenario_series(ax3, eval_by_ep, "x_velocity", marker="^", label_suffix=" (Eval)"):
        has_plotted_vel = True

    if not has_plotted_vel:
        ax3.text(
            0.5, 0.5,
            "Forward velocity (v_x) not recorded in this run",
            horizontalalignment="center", verticalalignment="center",
            transform=ax3.transAxes, fontsize=10, color="#7f8c8d",
        )
    ax3.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)
    ax3.set_title("3. Locomotion Velocity (v_x, m/s)", fontsize=12, fontweight="bold", pad=8)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Forward Velocity (m/s)")
    ax3.grid(True, linestyle="--", alpha=0.4)
    if has_plotted_vel:
        ax3.legend(loc="upper left", fontsize=7.5, framealpha=0.9, ncol=2)

    # -------------------------------------------------------------
    # Panel 4: Stability & Survival (Steps per Episode)
    # -------------------------------------------------------------
    ax4 = axes[1, 0]
    ax4.set_facecolor("#ffffff")
    full_surv_pct = float(np.mean(steps >= 1000) * 100.0) if len(steps) > 0 else 0.0
    ax4.scatter(episodes, steps, color="#8e44ad", alpha=0.30, s=12, label=f"Steps Survived ({full_surv_pct:.1f}% @ 1000)")
    steps_rolling = compute_rolling_mean(steps, rolling_window)
    ax4.plot(episodes, steps_rolling, color="#5b2c6f", linewidth=2.0, label=f"Rolling Mean (w={rolling_window})")
    ax4.axhline(1000, color="#e74c3c", linestyle="--", linewidth=1.2, label="Max Cap (1000 steps)")
    ax4.set_title("4. Survival (Steps per Episode)", fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Steps Survived")
    ax4.set_ylim(-20, 1050)
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 5: Loss Progression & Latent Model Convergence
    # -------------------------------------------------------------
    ax5 = axes[1, 1]
    ax5.set_facecolor("#ffffff")
    valid_loss_mask = losses > 0
    if np.any(valid_loss_mask):
        ep_l = episodes[valid_loss_mask]
        ls_vals = losses[valid_loss_mask]
        ax5.plot(ep_l, ls_vals, color="#e74c3c", alpha=0.35, linewidth=1.0, label="Raw Loss")
        loss_rolling = compute_rolling_mean(ls_vals, rolling_window)
        ax5.plot(ep_l, loss_rolling, color="#922b21", linewidth=2.0, label=f"Rolling Loss (w={rolling_window})")
    ax5.set_title("5. Latent Model & Representation Loss (L_total)", fontsize=12, fontweight="bold", pad=8)
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Total Loss")
    ax5.grid(True, linestyle="--", alpha=0.4)
    ax5.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 6: BRL Meta-Game Balance (Win Rate & Budget)
    # -------------------------------------------------------------
    ax6 = axes[1, 2]
    ax6.set_facecolor("#ffffff")
    color_wr = "#2980b9"
    ax6.plot(episodes, win_rates * 100.0, color=color_wr, linewidth=2.0, label="Agent Win Rate (EMA)")
    ax6.axhline(50.0, color="#27ae60", linestyle="--", linewidth=1.5, label="Target Balance (50%)")
    ax6.set_xlabel("Episode")
    ax6.set_ylabel("Win Rate (%)", color=color_wr)
    ax6.tick_params(axis="y", labelcolor=color_wr)
    ax6.set_ylim(-2, 102)
    ax6.grid(True, linestyle="--", alpha=0.4)

    has_budget = np.any(budgets > 0)
    if has_budget:
        ax6_b = ax6.twinx()
        color_b = "#d35400"
        ax6_b.plot(episodes, budgets, color=color_b, linestyle=":", linewidth=1.5, label="Budget (B_e)")
        ax6_b.set_ylabel("Budget Allocation", color=color_b)
        ax6_b.tick_params(axis="y", labelcolor=color_b)

    ax6.set_title("6. BRL Adversarial Triad Balance", fontsize=12, fontweight="bold", pad=8)
    lines6_1, labels6_1 = ax6.get_legend_handles_labels()
    if has_budget:
        lines6_2, labels6_2 = ax6_b.get_legend_handles_labels()
        ax6.legend(lines6_1 + lines6_2, labels6_1 + labels6_2, loc="upper right", fontsize=8, framealpha=0.9)
    else:
        ax6.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 7: Validation & OOD Benchmark Returns (Per-Scenario)
    # -------------------------------------------------------------
    ax7 = axes[2, 0]
    ax7.set_facecolor("#ffffff")
    if _plot_eval_scenario_series(ax7, eval_by_ep, "return", marker="s"):
        ax7.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)
        ax7.legend(loc="lower right", fontsize=8, framealpha=0.9)
    else:
        ax7.text(
            0.5, 0.5,
            "Benchmark evaluations recorded periodically\n(Check seed_*.log for details)",
            horizontalalignment="center", verticalalignment="center",
            transform=ax7.transAxes, fontsize=10, color="#7f8c8d",
        )
    ax7.set_title("7. Validation & OOD Benchmark Returns", fontsize=12, fontweight="bold", pad=8)
    ax7.set_xlabel("Episode")
    ax7.set_ylabel("Evaluation Return")
    ax7.grid(True, linestyle="--", alpha=0.4)

    # -------------------------------------------------------------
    # Panel 8: Aggregate Validation Score (Composite, Std-Ant, OOD)
    # -------------------------------------------------------------
    ax8 = axes[2, 1]
    ax8.set_facecolor("#ffffff")
    best_eval_score_str = "N/A"
    if eval_summary:
        sum_eps = sorted(eval_summary.keys())
        scores = np.array([eval_summary[ep]["score"] for ep in sum_eps], dtype=np.float64)
        std_ants = np.array([eval_summary[ep]["std_ant"] for ep in sum_eps], dtype=np.float64)
        oods = np.array([eval_summary[ep]["ood"] for ep in sum_eps], dtype=np.float64)
        new_bests = [ep for ep in sum_eps if eval_summary[ep].get("new_best", False)]

        ax8.plot(sum_eps, scores, color="#1b4f72", linewidth=2.2, marker="o", markersize=4, label="Composite Score")
        if not np.all(np.isnan(std_ants)):
            ax8.plot(sum_eps, std_ants, color="#2980b9", linewidth=1.5, linestyle="--", label="Std-Ant")
        if not np.all(np.isnan(oods)):
            ax8.plot(sum_eps, oods, color="#8e44ad", linewidth=1.5, linestyle="-.", label="OOD Mean")
        if new_bests:
            nb_scores = [eval_summary[ep]["score"] for ep in new_bests]
            ax8.scatter(new_bests, nb_scores, color="#f1c40f", edgecolors="#b7950b", marker="*", s=130, zorder=5, label="New Best ★")
        best_idx = int(np.argmax(scores))
        best_eval_score_str = f"{scores[best_idx]:+.1f} @ Ep {sum_eps[best_idx]}"
        ax8.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)
        ax8.legend(loc="lower right", fontsize=8, framealpha=0.9)
    else:
        ax8.text(
            0.5, 0.5,
            "Aggregate Eval Scores not found in .log",
            horizontalalignment="center", verticalalignment="center",
            transform=ax8.transAxes, fontsize=10, color="#7f8c8d",
        )
    ax8.set_title("8. Aggregate Validation Score & Milestones", fontsize=12, fontweight="bold", pad=8)
    ax8.set_xlabel("Episode")
    ax8.set_ylabel("Validation Score")
    ax8.grid(True, linestyle="--", alpha=0.4)

    # -------------------------------------------------------------
    # Panel 9: Compute Efficiency (Episode Wall-Clock Time)
    # -------------------------------------------------------------
    ax9 = axes[2, 2]
    ax9.set_facecolor("#ffffff")
    if has_ep_duration:
        valid_dur = durations > 0
        ep_d = episodes[valid_dur]
        dur_vals = durations[valid_dur]
        ax9.scatter(ep_d, dur_vals, color="#7f8c8d", alpha=0.25, s=10, label="Raw EpTime (s)")
        dur_roll = compute_rolling_mean(dur_vals, rolling_window)
        ax9.plot(ep_d, dur_roll, color="#2c3e50", linewidth=2.0, label=f"Rolling Mean (w={rolling_window})")
        full_mask = valid_dur & (steps >= 1000)
        if np.any(full_mask):
            ax9.plot(
                episodes[full_mask],
                compute_rolling_mean(durations[full_mask], rolling_window),
                color="#e67e22",
                linewidth=1.6,
                linestyle="--",
                label="Full 1000-Step Rolling",
            )
        ax9.legend(loc="upper right", fontsize=8, framealpha=0.9)
        mean_dur_str = f"{np.mean(dur_vals):.1f}s"
    else:
        ax9.text(
            0.5, 0.5,
            "Episode wall-clock duration not recorded",
            horizontalalignment="center", verticalalignment="center",
            transform=ax9.transAxes, fontsize=10, color="#7f8c8d",
        )
        mean_dur_str = "N/A"
    ax9.set_title("9. Compute Efficiency (Wall-Clock EpTime)", fontsize=12, fontweight="bold", pad=8)
    ax9.set_xlabel("Episode")
    ax9.set_ylabel("Seconds / Episode")
    ax9.grid(True, linestyle="--", alpha=0.4)

    # Header and Footer
    total_steps_str = f"{env_steps[-1]:,}" if len(env_steps) > 0 else "0"
    dist_str = (
        f"Dist(Max/Last): {np.nanmax(distances):+.2f}m / {distances[-1]:+.2f}m"
        if has_direct_dist and not np.all(np.isnan(distances))
        else "Dist: N/A"
    )
    vel_str = (
        f" | Vel(Max/Last): {np.nanmax(velocities):+.3f} / {velocities[-1]:+.3f} m/s"
        if has_direct_vel and not np.all(np.isnan(velocities))
        else ""
    )
    ag_ret_str = (
        f" | AgRet(Last20/Last): {np.nanmean(agent_returns[-20:]):+.1f} / {agent_returns[-1]:+.1f}"
        if has_agent_return
        else ""
    )
    main_title = f"LA-MCTS + Balanced RL Performance Dashboard [Seed {seed}] {title_suffix}"
    sub_title = (
        f"Episodes: {len(episodes)} | Env Steps: {total_steps_str} | "
        f"Return(Min/Max/Last): {np.min(returns):.1f} / {np.max(returns):.1f} / {returns[-1]:.1f}"
        f"{ag_ret_str} | {dist_str}{vel_str} | Best Eval: {best_eval_score_str} | Mean EpTime: {mean_dur_str}"
    )
    plt.suptitle(f"{main_title}\n{sub_title}", fontsize=13.5, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0.015, 0.015, 0.985, 0.94])

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_multi_seed_progress(
    datasets: List[Dict[str, Any]],
    output_path: Path,
    rolling_window: int = 20,
) -> Path:
    """Renders a combined 9-panel multi-seed comparison dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), dpi=200)
    fig.patch.set_facecolor("#fcfcfc")

    palette = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#c0392b", "#16a085"]

    for idx, d in enumerate(datasets):
        color = palette[idx % len(palette)]
        seed = d.get("seed", idx)
        eps = d["episodes"]
        lbl = f"Seed {seed}"

        # 1. Returns (Raw + Shaped Agent Return when recorded)
        axes[0, 0].plot(eps, compute_rolling_mean(d["returns"], rolling_window), color=color, linewidth=2.0, label=lbl)
        if d.get("has_agent_return") and not np.all(np.isnan(d["agent_returns"])):
            axes[0, 0].plot(
                eps,
                compute_rolling_mean(d["agent_returns"], rolling_window),
                color=color,
                linewidth=1.4,
                linestyle="--",
                alpha=0.8,
                label=f"{lbl} (Shaped)",
            )

        # 2. Distance
        if d.get("has_direct_dist") and not np.all(np.isnan(d["distances"])):
            axes[0, 1].plot(eps, compute_rolling_mean(d["distances"], rolling_window), color=color, linewidth=2.0, label=lbl)

        # 3. Velocity
        if d.get("has_direct_vel") and not np.all(np.isnan(d["velocities"])):
            axes[0, 2].plot(eps, compute_rolling_mean(d["velocities"], rolling_window), color=color, linewidth=2.0, label=lbl)

        # 4. Steps per ep
        axes[1, 0].plot(eps, compute_rolling_mean(d["steps"], rolling_window), color=color, linewidth=2.0, label=lbl)

        # 5. Losses
        vmask = d["losses"] > 0
        if np.any(vmask):
            axes[1, 1].plot(
                eps[vmask],
                compute_rolling_mean(d["losses"][vmask], rolling_window),
                color=color,
                linewidth=2.0,
                label=lbl,
            )

        # 6. Win Rates
        axes[1, 2].plot(eps, d["win_rates"] * 100.0, color=color, linewidth=2.0, label=lbl)

        # 7. Balancer Budget
        if np.any(d["budgets"] > 0):
            axes[2, 0].plot(eps, compute_rolling_mean(d["budgets"], rolling_window), color=color, linewidth=2.0, label=lbl)

        # 8 & 9. Validation Composite Score & OOD Mean
        eval_sum = d.get("eval_summary", {})
        if eval_sum:
            s_eps = sorted(eval_sum.keys())
            axes[2, 1].plot(
                s_eps,
                [eval_sum[ep]["score"] for ep in s_eps],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                label=lbl,
            )
            ood_eps = [ep for ep in s_eps if not np.isnan(eval_sum[ep]["ood"])]
            if ood_eps:
                axes[2, 2].plot(
                    ood_eps,
                    [eval_sum[ep]["ood"] for ep in ood_eps],
                    color=color,
                    linewidth=2.0,
                    marker="s",
                    markersize=3.5,
                    label=lbl,
                )

    titles_and_labels = [
        (axes[0, 0], f"1. Episode Returns (Rolling {rolling_window})", "Return"),
        (axes[0, 1], f"2. Locomotion Distance (Rolling {rolling_window})", "Distance (m)"),
        (axes[0, 2], f"3. Locomotion Velocity v_x (Rolling {rolling_window})", "Velocity (m/s)"),
        (axes[1, 0], f"4. Survival (Steps / Episode, Rolling {rolling_window})", "Steps"),
        (axes[1, 1], f"5. Model & Rep Loss (Rolling {rolling_window})", "Loss"),
        (axes[1, 2], "6. Agent EMA Win Rate (%)", "Win Rate (%)"),
        (axes[2, 0], f"7. Balancer Budget B_e (Rolling {rolling_window})", "Budget"),
        (axes[2, 1], "8. Validation Composite Score", "Eval Score"),
        (axes[2, 2], "9. Validation OOD Mean Score", "OOD Score"),
    ]
    for ax, title, ylabel in titles_and_labels:
        ax.set_facecolor("#ffffff")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, framealpha=0.9)

    axes[0, 0].axhline(0.0, color="#7f8c8d", linestyle=":", alpha=0.7)
    axes[0, 1].axhline(0.0, color="#7f8c8d", linestyle=":", alpha=0.7)
    axes[0, 2].axhline(0.0, color="#7f8c8d", linestyle=":", alpha=0.7)
    axes[1, 0].axhline(1000, color="#e74c3c", linestyle="--", alpha=0.7)
    axes[1, 2].axhline(50.0, color="#27ae60", linestyle="--", alpha=0.7)
    axes[1, 2].set_ylim(-2, 102)

    plt.suptitle("LA-MCTS + Balanced RL Multi-Seed Progression Overview", fontsize=15, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0.015, 0.015, 0.985, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_stats_json(data: Dict[str, Any], output_path: Path) -> Path:
    """Exports key numerical statistics into a JSON summary file."""
    returns = data.get("returns", np.array([]))
    losses = data.get("losses", np.array([]))
    steps = data.get("steps", np.array([]))
    distances = data.get("distances", np.array([]))
    velocities = data.get("velocities", np.array([]))
    durations = data.get("durations", np.array([]))
    wins = data.get("wins", np.array([]))
    eval_summary = data.get("eval_summary", {})
    eval_by_ep = data.get("eval_by_ep", {})

    stats: Dict[str, Any] = {
        "seed": int(data.get("seed", 0)),
        "completed_episodes": int(len(data.get("episodes", []))),
        "total_env_steps": int(data["total_env_steps"][-1]) if len(data.get("total_env_steps", [])) > 0 else 0,
        "return_mean": float(np.mean(returns)) if len(returns) > 0 else 0.0,
        "return_best": float(np.max(returns)) if len(returns) > 0 else 0.0,
        "return_min": float(np.min(returns)) if len(returns) > 0 else 0.0,
        "return_last20_mean": float(np.mean(returns[-20:])) if len(returns) >= 20 else (float(np.mean(returns)) if len(returns) > 0 else 0.0),
        "steps_mean": float(np.mean(steps)) if len(steps) > 0 else 0.0,
        "steps_last20_mean": float(np.mean(steps[-20:])) if len(steps) > 0 else 0.0,
        "full_survival_rate": float(np.mean(steps >= 1000)) if len(steps) > 0 else 0.0,
        "loss_start": float(losses[0]) if len(losses) > 0 else 0.0,
        "loss_min": float(np.min(losses[losses > 0])) if np.any(losses > 0) else 0.0,
        "loss_final": float(losses[-1]) if len(losses) > 0 else 0.0,
        "final_win_rate_ema": float(data["win_rates"][-1]) if len(data.get("win_rates", [])) > 0 else 0.0,
        "actual_win_rate": float(np.mean(wins)) if len(wins) > 0 else 0.0,
        "final_r_ref": float(data["r_refs"][-1]) if len(data.get("r_refs", [])) > 0 else 0.0,
        "final_budget": float(data["budgets"][-1]) if len(data.get("budgets", [])) > 0 else 0.0,
    }

    if data.get("has_agent_return") and not np.all(np.isnan(data.get("agent_returns", np.array([np.nan])))):
        ag_ret = data["agent_returns"]
        stats["agent_return_mean"] = float(np.nanmean(ag_ret))
        stats["agent_return_best"] = float(np.nanmax(ag_ret))
        stats["agent_return_min"] = float(np.nanmin(ag_ret))
        stats["agent_return_last20_mean"] = float(np.nanmean(ag_ret[-20:]))

    if data.get("has_direct_dist") and not np.all(np.isnan(distances)):
        stats["distance_mean"] = float(np.nanmean(distances))
        stats["distance_best"] = float(np.nanmax(distances))
        stats["distance_last20_mean"] = float(np.nanmean(distances[-20:]))
    if data.get("has_direct_vel") and not np.all(np.isnan(velocities)):
        stats["velocity_mean"] = float(np.nanmean(velocities))
        stats["velocity_best"] = float(np.nanmax(velocities))
        stats["velocity_last20_mean"] = float(np.nanmean(velocities[-20:]))
    if data.get("has_ep_duration") and np.any(durations > 0):
        valid_d = durations[durations > 0]
        stats["ep_duration_mean_sec"] = float(np.mean(valid_d))
        stats["ep_duration_last20_mean_sec"] = float(np.mean(valid_d[-20:]))
        stats["total_wall_time_hours"] = float(np.sum(valid_d) / 3600.0)

    if eval_summary:
        best_ep = max(eval_summary.keys(), key=lambda ep: eval_summary[ep]["score"])
        latest_ep = max(eval_summary.keys())
        stats["eval_best_ep"] = int(best_ep)
        stats["eval_best_score"] = float(eval_summary[best_ep]["score"])
        if not np.isnan(eval_summary[best_ep]["std_ant"]):
            stats["eval_best_std_ant"] = float(eval_summary[best_ep]["std_ant"])
        if not np.isnan(eval_summary[best_ep]["ood"]):
            stats["eval_best_ood"] = float(eval_summary[best_ep]["ood"])
        stats["eval_latest_ep"] = int(latest_ep)
        stats["eval_latest_score"] = float(eval_summary[latest_ep]["score"])
        if not np.isnan(eval_summary[latest_ep]["std_ant"]):
            stats["eval_latest_std_ant"] = float(eval_summary[latest_ep]["std_ant"])
        if not np.isnan(eval_summary[latest_ep]["ood"]):
            stats["eval_latest_ood"] = float(eval_summary[latest_ep]["ood"])

    if eval_by_ep:
        latest_eval_ep = max(eval_by_ep.keys())
        stats["eval_latest_by_scenario"] = eval_by_ep[latest_eval_ep]
        lat_vels = [v["x_velocity"] for v in eval_by_ep[latest_eval_ep].values() if "x_velocity" in v]
        lat_dists = [v["total_distance"] for v in eval_by_ep[latest_eval_ep].values() if "total_distance" in v]
        if lat_vels:
            stats["eval_latest_mean_vel"] = float(np.mean(lat_vels))
        if lat_dists:
            stats["eval_latest_mean_dist"] = float(np.mean(lat_dists))

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and validation charts from progress.jsonl")
    parser.add_argument("--input", "-i", type=str, help="Path to single progress.jsonl file")
    parser.add_argument("--dir", "-d", type=str, help="Directory containing one or more progress.jsonl files")
    parser.add_argument("--output", "-o", type=str, help="Path to output image (.png)")
    parser.add_argument("--rolling", "-r", type=int, default=20, help="Rolling window size (default: 20)")
    args = parser.parse_args()

    if not args.input and not args.dir:
        parser.error("Must provide either --input or --dir")

    input_paths: List[Path] = []
    if args.input:
        input_paths.append(Path(args.input))
    if args.dir:
        d = Path(args.dir)
        input_paths.extend(sorted(d.glob("*progress.jsonl")))

    if not input_paths:
        print(f"Error: No progress.jsonl files found in {args.dir}")
        return

    datasets: List[Dict[str, Any]] = []
    for p in input_paths:
        data = parse_progress_jsonl(p)
        if data and len(data.get("episodes", [])) > 0:
            datasets.append(data)
            out_img = Path(args.output) if (args.output and len(input_paths) == 1) else p.with_name(f"{p.stem}_chart.png")
            plot_single_progress(data, out_img, rolling_window=args.rolling)
            export_stats_json(data, out_img.with_suffix(".json"))
            print(f"[OK] Generated chart: {out_img}")

    if len(datasets) > 1:
        multi_out = Path(args.output) if args.output else Path(args.dir) / "multi_seed_overview.png"
        plot_multi_seed_progress(datasets, multi_out, rolling_window=args.rolling)
        print(f"[OK] Generated multi-seed overview: {multi_out}")


if __name__ == "__main__":
    main()
