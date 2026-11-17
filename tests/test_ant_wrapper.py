"""Verification Gate 3: MuJoCo Physics & Morphology Wrapper Tests (ant_morph_env/)."""

from __future__ import annotations

import numpy as np
import pytest

from ant_morph_env.wrapper import MorphologicalAntBRLWrapper


def test_gate3_50_extreme_episodes_and_morphologies() -> None:
    """Simulates 50 episodes under maximum boundary tweaks (w_i = 1000 individually & combined)
    and across 3-, 4-, and 6-legged creature variants.
    """
    rng = np.random.default_rng(2026)
    wrapper = MorphologicalAntBRLWrapper(
        max_action_capacity=12,
        fixed_action_space=True,
        default_num_legs=4,
    )
    assert "Ant-v5" in wrapper.env_id
    assert wrapper.observation_space.shape == (27,)

    num_episodes = 50
    steps_per_episode = 25

    for ep in range(num_episodes):
        tweak = np.zeros(12, dtype=np.float64)
        num_legs = 4

        if ep < 12:
            # Individual extreme tweak w_i = 1000
            tweak[ep] = 1000.0
        elif ep == 12:
            # All parameters simultaneously at maximum w = 1000
            tweak[:] = 1000.0
        elif ep == 13:
            # 3-legged creature variant
            num_legs = 3
            tweak[0] = 600.0
            tweak[3] = 1000.0
        elif ep == 14:
            # 6-legged Hexapod creature variant
            num_legs = 6
            tweak[0] = 400.0
            tweak[4] = 500.0
            tweak[11] = 800.0
        else:
            # Random extreme combinations across 3-, 4-, and 6-legged variants
            num_legs = int(rng.choice([3, 4, 6]))
            active_dims = rng.choice(12, size=4, replace=False)
            tweak[active_dims] = 1000.0
            tweak += rng.uniform(0.0, 500.0, size=12)
            tweak = np.clip(tweak, 0.0, 1000.0)

        obs, info = wrapper.reset(
            seed=100 + ep,
            options={"tweak_vector": tweak, "num_legs": num_legs},
        )
        assert np.all(np.isfinite(obs))
        assert obs.shape == (27,), f"Expected canonical obs shape (27,), got {obs.shape}"
        morph_mask = info["morphology_mask"]
        assert morph_mask.shape == (12,)

        # Verify action dimension masking for 3-, 4-, and 6-legged variants
        if num_legs == 3:
            assert np.all(morph_mask[:6] == 1.0)
            assert np.all(morph_mask[6:] == 0.0)
        elif num_legs == 4:
            assert np.all(morph_mask[:6] == 1.0)
            if tweak[3] >= 990.0:
                assert np.all(morph_mask[6:8] == 0.0)
            else:
                assert np.all(morph_mask[6:8] == 1.0)
            assert np.all(morph_mask[8:] == 0.0)
        elif num_legs == 6:
            assert np.all(morph_mask[:12] == 1.0)

        # Verify inertia matrices and masses are strictly positive and finite after mj_setConst
        assert np.all(np.isfinite(wrapper.mj_model.body_mass))
        assert np.all(wrapper.mj_model.body_mass[1:] > 0.0)
        assert np.all(np.isfinite(wrapper.mj_model.body_inertia))

        for _ in range(steps_per_episode):
            rand_action = rng.uniform(-1.0, 1.0, size=(12,)).astype(np.float32)
            obs, reward, terminated, truncated, step_info = wrapper.step(rand_action)
            assert np.all(np.isfinite(obs)), f"Non-finite obs in episode {ep}"
            assert np.isfinite(reward), f"Non-finite reward in episode {ep}"
            assert "brl_metrics" in step_info
            if terminated or truncated:
                break

    wrapper.close()


def test_standing_reward_is_unpenalized_and_win_evaluator() -> None:
    """Verifies the full env reward is returned (no stagnation penalty) and WinEvaluator still rejects zero-velocity wins."""
    from brl_core.interfaces import WinEvaluator

    wrapper = MorphologicalAntBRLWrapper(max_action_capacity=12, fixed_action_space=True)
    obs, info = wrapper.reset(seed=42)

    # Issue zero action (ant stands still)
    zero_action = np.zeros(12, dtype=np.float32)
    obs, reward, term, trunc, step_info = wrapper.step(zero_action)
    wrapper.close()

    # Raw env reward keeps the healthy bonus (~+1.0), while initial training reward (alpha=0) is custom_reward
    assert np.isfinite(reward)
    assert step_info["raw_env_reward"] > 0.5, (
        f"Expected unpenalized raw_env_reward (healthy bonus kept), got {step_info['raw_env_reward']}"
    )
    assert reward == pytest.approx(step_info["custom_reward"], abs=1e-6)

    # Verify WinEvaluator: premature unhealthy termination MUST be evaluated as LOSS (0) even with high return
    we = WinEvaluator(initial_r_ref=500.0)
    win_ind, dom, ema = we.evaluate_episode(
        episode_return=900.0,
        terminated_unhealthy=True,
        forward_velocity=0.01,
    )
    assert win_ind == 0, "Unhealthy termination must evaluate to LOSS (win_indicator = 0)"
    assert dom <= 0.5, f"Dominance score for unhealthy termination should be dampened, got {dom}"

    # Verify WinEvaluator: healthy survival WITH return >= r_ref MUST be WIN (1) without velocity gate
    win_ind_run, dom_run, ema_run = we.evaluate_episode(
        episode_return=900.0,
        terminated_unhealthy=False,
        forward_velocity=0.01,
    )
    assert win_ind_run == 1, "Healthy survival meeting r_ref must evaluate to WIN (win_indicator = 1)"


def test_hexapod_all_legs_point_downward() -> None:
    """Verifies that all 6 legs of the Hexapod morphological ant point downwards towards the ground.

    Ensures that Leg 6 (middle-right) has its ankle range oriented so that throughout its valid range
    [30, 70] deg, its geom position in z is strictly below the torso (rel_z < 0), maintaining bilateral
    downward symmetry with Leg 5 (middle-left).
    """
    import mujoco
    from ant_morph_env.xml_gen import ProceduralAntXMLGenerator

    gen = ProceduralAntXMLGenerator()
    xml_str = gen.generate_xml(num_legs=6)

    m = mujoco.MjModel.from_xml_string(xml_str)
    d = mujoco.MjData(m)

    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    assert torso_id >= 0

    # Test at min, mid, and max joint angle for each leg's ankle
    for frac in [0.0, 0.5, 1.0]:
        for j in range(1, m.njnt):
            rng = m.jnt_range[j]
            d.qpos[6 + j] = rng[0] + frac * (rng[1] - rng[0])
        mujoco.mj_forward(m, d)

        torso_z = float(d.xpos[torso_id][2])
        for leg_id in range(1, 7):
            geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"ankle_{leg_id}_geom")
            assert geom_id >= 0
            rel_z = float(d.geom_xpos[geom_id][2]) - torso_z
            assert rel_z < 0.0, (
                f"Leg {leg_id} points upwards at frac={frac}: geom_z={d.geom_xpos[geom_id][2]:.4f}, "
                f"torso_z={torso_z:.4f}, rel_z={rel_z:.4f}"
            )

    # Check bilateral symmetry between Leg 5 (middle-left) and Leg 6 (middle-right) at mid-range
    for j in range(1, m.njnt):
        rng = m.jnt_range[j]
        d.qpos[6 + j] = 0.5 * (rng[0] + rng[1])
    mujoco.mj_forward(m, d)

    torso_z = float(d.xpos[torso_id][2])
    g5_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ankle_5_geom")
    g6_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ankle_6_geom")
    rel_z_5 = float(d.geom_xpos[g5_id][2]) - torso_z
    rel_z_6 = float(d.geom_xpos[g6_id][2]) - torso_z

    assert abs(rel_z_5 - rel_z_6) < 1e-4, (
        f"Leg 5 (rel_z={rel_z_5:.4f}) and Leg 6 (rel_z={rel_z_6:.4f}) are not vertically symmetric!"
    )


def test_actuator_weakening_scales_linearly() -> None:
    """Verifies w_9 and w_4 scale effective actuator torque linearly (not quadratically)."""
    wrapper = MorphologicalAntBRLWrapper(max_action_capacity=12, fixed_action_space=True)
    nom_gear = wrapper._nom_actuator_gear[:, 0].copy()
    nom_gain = wrapper._nom_actuator_gainprm[:, 0].copy()

    # w_9 = 1000 -> act_scale = 0.40 (must not become 0.16 via gainprm * gear)
    tweak_w9 = np.zeros(18, dtype=np.float32)
    tweak_w9[8] = 1000.0
    wrapper.reset(seed=42, options={"tweak_vector": tweak_w9})
    model = wrapper.mj_model
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], nom_gain, rtol=1e-6)
    np.testing.assert_allclose(model.actuator_gear[:, 0], nom_gear * 0.40, rtol=1e-6)

    # w_4 = 500 + w_9 = 1000 -> leg 4 gear = 0.40 * 0.50 = 0.20 * nom_gear
    tweak_both = np.zeros(18, dtype=np.float32)
    tweak_both[3] = 500.0
    tweak_both[8] = 1000.0
    wrapper.reset(seed=42, options={"tweak_vector": tweak_both})
    model = wrapper.mj_model
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], nom_gain, rtol=1e-6)
    np.testing.assert_allclose(model.actuator_gear[:6, 0], nom_gear[:6] * 0.40, rtol=1e-6)
    np.testing.assert_allclose(model.actuator_gear[6:8, 0], nom_gear[6:8] * 0.20, rtol=1e-6)
    wrapper.close()


def test_fixed_smoothness_and_velocity_curriculum_reward_blending() -> None:
    """Verifies fixed smoothness coef, 1000-step gated rolling 20 raw-return curriculum (0-1000 torque, 0-3000 exploration), and no upside-down penalty."""
    wrapper = MorphologicalAntBRLWrapper(
        max_action_capacity=12,
        fixed_action_space=True,
        action_smoothness_coef=0.15,
        torque_return_ref=1000.0,
        exploration_return_ref=3000.0,
        vel_shift_factor=3000.0,
    )
    try:
        # 1. Smoothness coef is fixed regardless of total_env_steps
        assert wrapper.get_current_smoothness_coef() == pytest.approx(0.15, abs=1e-9)
        wrapper.set_total_env_steps(500000)
        assert wrapper.get_current_smoothness_coef() == pytest.approx(0.15, abs=1e-9)

        # 2. Episodes shorter than 1000 steps or unhealthy must NOT update target_return or enter history
        wrapper.update_return_curriculum(800.0, ep_steps=500, required_steps=1000, terminated_unhealthy=False)
        wrapper.update_return_curriculum(900.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=True)
        assert len(wrapper._recent_ep_returns) == 0
        assert wrapper.target_return == 0.0

        # 3. Rolling 20-episode mean of full 1000-step episodes -> target_return = mean(rolling window up to 20)
        wrapper.update_return_curriculum(500.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=False)
        wrapper.update_return_curriculum(550.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=False)
        wrapper.update_return_curriculum(650.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=False)
        wrapper.update_return_curriculum(700.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=False)
        assert len(wrapper._recent_ep_returns) == 4
        assert wrapper.target_return == pytest.approx(600.0, abs=1e-6)

        # 4. Adding a higher return episode continues to take the simple rolling mean (up to 20 episodes)
        wrapper.update_return_curriculum(1500.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=False)
        # [500, 550, 650, 700, 1500] -> mean is 780.0
        assert wrapper.target_return == pytest.approx(780.0, abs=1e-6)

        # 5. Each step shifts current_return toward target_return by target_return / 3000;
        #    torque/smoothness scales over 0..1000 and exploration scales over 0..3000
        wrapper.reset(seed=42)
        expected_step_shift = 780.0 / 3000.0
        _, rew, _, _, info_step = wrapper.step(np.ones(12, dtype=np.float32) * 0.5)
        assert wrapper.current_return == pytest.approx(expected_step_shift, abs=1e-7)
        alpha_torque = expected_step_shift / 1000.0
        alpha_explore = expected_step_shift / 3000.0
        assert info_step["reward_blend_alpha"] == pytest.approx(alpha_torque, abs=1e-7)
        assert info_step["exploration_progress"] == pytest.approx(alpha_explore, abs=1e-7)
        expected_rew = (1.0 - alpha_torque) * info_step["custom_reward"] + alpha_torque * info_step["raw_env_reward"]
        assert rew == pytest.approx(expected_rew, abs=1e-5)
        assert not info_step["upside_down_terminated"]

        # 6. action_ema_alpha (0.8) converges linearly to 1.0 as current_return approaches 1000.0,
        #    while exploration_progress converges to 1.0 as current_return approaches 3000.0
        assert info_step["action_ema_alpha"] == pytest.approx(0.8 + alpha_torque * 0.2, abs=1e-7)
        wrapper.current_return = 0.0
        assert wrapper.get_current_action_ema_alpha() == pytest.approx(0.8, abs=1e-7)
        assert wrapper.get_exploration_progress() == pytest.approx(0.0, abs=1e-7)
        wrapper.current_return = 500.0
        assert wrapper.get_current_action_ema_alpha() == pytest.approx(0.9, abs=1e-7)
        assert wrapper.get_exploration_progress() == pytest.approx(500.0 / 3000.0, abs=1e-7)
        wrapper.current_return = 1000.0
        assert wrapper.get_reward_blend_alpha() == pytest.approx(1.0, abs=1e-7)
        assert wrapper.get_current_action_ema_alpha() == pytest.approx(1.0, abs=1e-7)
        assert wrapper.get_exploration_progress() == pytest.approx(1.0 / 3.0, abs=1e-7)
        wrapper.current_return = 3000.0
        assert wrapper.get_reward_blend_alpha() == pytest.approx(1.0, abs=1e-7)
        assert wrapper.get_exploration_progress() == pytest.approx(1.0, abs=1e-7)

        # 7. Negative returns clamp to 0.0 for curriculum / torque-penalty control, and rolling 20 window caps at 20
        wrapper._recent_ep_returns.clear()
        wrapper.target_return = 0.0
        wrapper.current_return = 0.0
        wrapper.update_return_curriculum(-450.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=False)
        wrapper.update_return_curriculum(300.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=False)
        wrapper.update_return_curriculum(300.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=False)
        wrapper.update_return_curriculum(300.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=False)
        assert list(wrapper._recent_ep_returns) == pytest.approx([0.0, 300.0, 300.0, 300.0], abs=1e-7)
        assert wrapper.target_return == pytest.approx(225.0, abs=1e-6)
        # Fill 20 episodes with 50.0 -> rolling 20 mean becomes 50.0
        for _ in range(20):
            wrapper.update_return_curriculum(50.0, ep_steps=1000, required_steps=1000, terminated_unhealthy=False)
        assert len(wrapper._recent_ep_returns) == 20
        assert wrapper.target_return == pytest.approx(50.0, abs=1e-6)
        wrapper.current_return = -200.0
        assert wrapper.get_reward_blend_alpha() == pytest.approx(0.0, abs=1e-7)
        assert wrapper.get_exploration_progress() == pytest.approx(0.0, abs=1e-7)
        assert wrapper.get_current_action_ema_alpha() == pytest.approx(0.8, abs=1e-7)
    finally:
        wrapper.close()


def test_front_back_left_right_leg_scaling() -> None:
    """Verifies w[0] (s_front) scales +x legs (1, 4), w[1] (s_back) scales -x legs (2, 3), and w[2] scales +y/-y."""
    wrapper = MorphologicalAntBRLWrapper(max_action_capacity=12, fixed_action_space=True)
    try:
        # 1. Front scaling w[0] = 800 -> Legs 1 (+x,+y) and 4 (+x,-y) elongate; Legs 2, 3 stay nominal
        t_front = np.zeros(18, dtype=np.float32)
        t_front[0] = 800.0
        _, info_f = wrapper.reset(seed=42, options={"tweak_vector": t_front})
        kin_f = info_f["true_limb_kinematics"]
        # Anchor x/y norms: legs 1 & 4 should be > 0.5, legs 2 & 3 should be ~0.283 (sqrt(0.2^2 + 0.2^2))
        norms_f = np.linalg.norm(kin_f[:, :2], axis=1)
        assert norms_f[0] > 0.50 and norms_f[3] > 0.50
        assert norms_f[1] < 0.35 and norms_f[2] < 0.35

        # 2. Back scaling w[1] = 800 -> Legs 2 (-x,+y) and 3 (-x,-y) shrink; Legs 1, 4 stay nominal
        t_back = np.zeros(18, dtype=np.float32)
        t_back[1] = 800.0
        _, info_b = wrapper.reset(seed=42, options={"tweak_vector": t_back})
        norms_b = np.linalg.norm(info_b["true_limb_kinematics"][:, :2], axis=1)
        assert norms_b[1] < 0.20 and norms_b[2] < 0.20
        assert norms_b[0] > 0.25 and norms_b[3] > 0.25

        # 3. Left/right scaling w[2] = 800 -> +y legs (1, 2) elongate; -y legs (3, 4) shrink
        t_lr = np.zeros(18, dtype=np.float32)
        t_lr[2] = 800.0
        _, info_lr = wrapper.reset(seed=42, options={"tweak_vector": t_lr})
        norms_lr = np.linalg.norm(info_lr["true_limb_kinematics"][:, :2], axis=1)
        assert norms_lr[0] > norms_lr[3] and norms_lr[1] > norms_lr[2]
    finally:
        wrapper.close()


def test_velocity_gated_healthy_reward_and_win_evaluator() -> None:
    """Verifies training-only velocity-gated healthy_reward and WinEvaluator min_win_velocity + r_ref_target_mult."""
    from brl_core.interfaces import WinEvaluator

    # 1. WinEvaluator denies stationary wins and raises r_ref past episode_return
    ev = WinEvaluator(
        initial_r_ref=500.0,
        benchmark_r_ref=6000.0,
        beta_w=0.25,
        adaptive_ref_rate=0.5,
        min_win_velocity=0.05,
        r_ref_target_mult=1.02,
    )
    ev.ema_win_rate = 0.60
    # Stationary episode (vel=0.01 < 0.05) loses even with high return
    win_stat, _, _ = ev.evaluate_episode(800.0, terminated_unhealthy=False, forward_velocity=0.01)
    assert win_stat == 0

    # Moving episode (vel=0.20 >= 0.05) wins and pushes r_ref toward 1.02 * 800 = 816 (> 0.90 * 800 = 720)
    ev.ema_win_rate = 0.60
    win_move, _, _ = ev.evaluate_episode(800.0, terminated_unhealthy=False, forward_velocity=0.20)
    assert win_move == 1
    assert ev.r_ref > 500.0
    # After repeated wins at 800, r_ref converges above 800 (not capped at 720)
    for _ in range(25):
        ev.ema_win_rate = 0.80
        ev.evaluate_episode(800.0, terminated_unhealthy=False, forward_velocity=0.20)
    assert ev.r_ref > 800.0

    # 2. Wrapper withholds healthy_reward during stationary training steps, preserves it in eval_mode
    wrapper = MorphologicalAntBRLWrapper(
        max_action_capacity=12,
        fixed_action_space=True,
        action_ema_alpha=1.0,
        action_smoothness_coef=0.0,
        min_healthy_vel=0.10,
    )
    try:
        wrapper.reset(seed=42, options={"eval_mode": False})
        zero_act = np.zeros(12, dtype=np.float32)
        _, train_r, _, _, train_info = wrapper.step(zero_act)
        raw_r = float(train_info["brl_metrics"]["raw_episode_return"])
        survive_r = float(train_info.get("reward_survive", 0.0))
        x_vel = float(train_info["brl_metrics"]["x_velocity"])
        assert survive_r > 0.0
        assert x_vel < 0.10
        expected_gate = float(np.clip(x_vel / 0.10, 0.0, 1.0))
        assert train_r == pytest.approx(raw_r - survive_r * (1.0 - expected_gate), abs=1e-5)

        wrapper.reset(seed=42, options={"eval_mode": True})
        _, eval_r, _, _, eval_info = wrapper.step(zero_act)
        assert eval_r == pytest.approx(float(eval_info["brl_metrics"]["raw_episode_return"]), abs=1e-6)
    finally:
        wrapper.close()


def test_stationary_penalty_smoothed_gate_and_mcts_gradients() -> None:
    """Stationary ant gets net-negative shaped reward (fed to MCTS); eval stays raw; gradients stay finite."""
    import torch
    from config import LAMCTSConfig
    from la_mcts.agent import LatentActionMCTSAgent

    wrapper = MorphologicalAntBRLWrapper(
        max_action_capacity=12,
        fixed_action_space=True,
        action_ema_alpha=1.0,
        action_smoothness_coef=0.0,
        min_healthy_vel=0.10,
        stationary_penalty_coef=0.50,
        stationary_vel_window=20,
    )
    try:
        zero_act = np.zeros(12, dtype=np.float32)
        wrapper.reset(seed=42, options={"eval_mode": False})
        agent_sum, raw_sum = 0.0, 0.0
        infos = []
        for _ in range(300):  # beyond the ~100-step post-reset settling rock
            _, r, term, trunc, info = wrapper.step(zero_act)
            agent_sum += float(r)
            raw_sum += float(info["raw_env_reward"])
            infos.append(info)
            if term or trunc:
                break
        # Standing still: raw Ant-v5 reward is positive, shaped agent reward is net negative
        assert raw_sum > 0.0
        assert agent_sum < 0.0
        last = infos[-1]
        assert last["stationary_penalty"] > 0.0
        assert len(wrapper._recent_step_vels) == 20
        assert last["smooth_x_vel"] == pytest.approx(float(np.mean(list(wrapper._recent_step_vels))), abs=1e-9)

        # Single-step jitter cannot bypass the smoothed gate: one big positive spike in a stationary window
        wrapper._recent_step_vels.clear()
        for v in [0.0] * 19:
            wrapper._recent_step_vels.append(v)
        wrapper._recent_step_vels.append(1.0)  # instantaneous spike
        smooth = float(np.mean(list(wrapper._recent_step_vels)))
        assert smooth == pytest.approx(0.05, abs=1e-9)
        assert float(np.clip(smooth / 0.10, 0.0, 1.0)) < 1.0

        # Eval mode keeps the raw Ant-v5 reward (no healthy/stationary shaping)
        wrapper.reset(seed=42, options={"eval_mode": True})
        assert len(wrapper._recent_step_vels) == 0
        _, eval_r, _, _, eval_info = wrapper.step(zero_act)
        assert eval_info["stationary_penalty"] == 0.0
        assert eval_r == pytest.approx(float(eval_info["raw_env_reward"]), abs=1e-6)
    finally:
        wrapper.close()

    # Negative stationary rewards flow into MCTS training without NaNs
    cfg = LAMCTSConfig(batch_size=4, n_sim=4, k_act=4, unroll_steps=2, td_steps=2)
    agent = LatentActionMCTSAgent(config=cfg, device="cpu")
    obs_dim = cfg.obs_dim
    obs = np.zeros(obs_dim, dtype=np.float32)
    for _ in range(3):
        for t in range(12):
            act = np.zeros(cfg.max_action_dim, dtype=np.float32)
            nxt = obs + 0.01
            agent.observe_transition(obs, act, -0.5, nxt, False, t == 11, {})
            obs = nxt
    metrics = agent.train_step()
    if metrics:
        assert all(np.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float)))
    for p in agent.parameters() if hasattr(agent, "parameters") else []:
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_training_shaping_preserves_standard_benchmark_eval() -> None:
    """Verifies forward_reward_weight / ctrl_cost_weight shape training reward only; eval/raw stays standard Ant-v5."""
    wrapper = MorphologicalAntBRLWrapper(
        max_action_capacity=12,
        fixed_action_space=True,
        action_ema_alpha=1.0,
        action_smoothness_coef=0.0,
        min_healthy_vel=0.0,
        stationary_penalty_coef=0.0,
        forward_reward_weight=2.5,
        ctrl_cost_weight=0.15,
        forward_return_ref=5000.0,
    )
    try:
        assert wrapper.get_current_forward_reward_weight() == pytest.approx(2.5, abs=1e-6)
        assert wrapper.get_current_ctrl_cost_weight() == pytest.approx(0.15, abs=1e-6)
        wrapper.current_return = 2500.0
        assert wrapper.get_current_forward_reward_weight() == pytest.approx(1.75, abs=1e-6)
        assert wrapper.get_current_ctrl_cost_weight() == pytest.approx(0.325, abs=1e-6)
        wrapper.current_return = 5000.0
        assert wrapper.get_current_forward_reward_weight() == pytest.approx(1.0, abs=1e-6)
        assert wrapper.get_current_ctrl_cost_weight() == pytest.approx(0.5, abs=1e-6)
        wrapper.current_return = 0.0

        act = np.full(12, 0.5, dtype=np.float32)
        wrapper.reset(seed=42, options={"eval_mode": False})
        _, train_r, _, _, train_info = wrapper.step(act)
        raw_r = float(train_info["raw_env_reward"])
        r_fwd = float(train_info["reward_forward"])
        r_ctrl = float(train_info["reward_ctrl"])
        expected_train = raw_r + (2.5 - 1.0) * r_fwd + (1.0 - 0.15 / 0.5) * (-r_ctrl)
        assert train_r == pytest.approx(expected_train, abs=1e-5)
        assert float(train_info["ctrl_cost_weight"]) == pytest.approx(0.15, abs=1e-6)

        # Mid-curriculum blending in training mode moves both weights halfway toward benchmark
        wrapper.target_return = 2500.0
        wrapper.current_return = 2500.0
        _, mid_r, _, _, mid_info = wrapper.step(act)
        mid_raw_r = float(mid_info["raw_env_reward"])
        mid_fwd = float(mid_info["reward_forward"])
        mid_ctrl = float(mid_info["reward_ctrl"])
        expected_mid = mid_raw_r + (1.75 - 1.0) * mid_fwd + (1.0 - 0.325 / 0.5) * (-mid_ctrl)
        assert mid_r == pytest.approx(expected_mid, abs=1e-5)
        assert float(mid_info["ctrl_cost_weight"]) == pytest.approx(0.325, abs=1e-6)
        wrapper.target_return = 0.0
        wrapper.current_return = 0.0

        # Standard Ant-v5 control cost for 8 active dims at 0.5 is -0.5 * (8 * 0.25) = -1.0
        assert r_ctrl == pytest.approx(-1.0, abs=1e-5)

        wrapper.reset(seed=42, options={"eval_mode": True})
        _, eval_r, _, _, eval_info = wrapper.step(act)
        assert eval_r == pytest.approx(float(eval_info["raw_env_reward"]), abs=1e-6)
        assert float(eval_info["reward_ctrl"]) == pytest.approx(-1.0, abs=1e-5)
        assert float(eval_info["ctrl_cost_weight"]) == pytest.approx(0.5, abs=1e-6)
    finally:
        wrapper.close()


def test_use_real_returns_bypasses_all_reward_shaping_from_step_one() -> None:
    """Verifies use_real_returns=True passes raw Ant-v5 rewards from step 1 while preserving exploration_progress."""
    wrapper = MorphologicalAntBRLWrapper(
        max_action_capacity=12,
        fixed_action_space=True,
        action_ema_alpha=0.75,
        action_smoothness_coef=0.15,
        min_healthy_vel=0.10,
        stationary_penalty_coef=0.50,
        forward_reward_weight=4.0,
        ctrl_cost_weight=0.30,
        exploration_return_ref=3000.0,
        use_real_returns=True,
    )
    try:
        wrapper.reset(seed=42, options={"eval_mode": False})
        act = np.full(12, 0.2, dtype=np.float32)
        _, rew, _, _, info = wrapper.step(act)
        raw_r = float(info["raw_env_reward"])
        assert rew == pytest.approx(raw_r, abs=1e-6)
        assert float(info["agent_reward"]) == pytest.approx(raw_r, abs=1e-6)
        assert float(info["healthy_penalty"]) == 0.0
        assert float(info["stationary_penalty"]) == 0.0
        assert float(info["forward_reward_weight"]) == pytest.approx(1.0, abs=1e-6)
        assert float(info["ctrl_cost_weight"]) == pytest.approx(0.5, abs=1e-6)
        assert float(info["reward_blend_alpha"]) == pytest.approx(1.0, abs=1e-6)

        wrapper.current_return = 1500.0
        assert wrapper.get_exploration_progress() == pytest.approx(0.5, abs=1e-6)
    finally:
        wrapper.close()


def test_orchestrator_forwards_raw_returns_and_gates_exploration_by_velocity() -> None:
    """Verifies BRLTrainingOrchestrator forwards raw_env_reward when use_real_returns=True and keeps exploration alive at v_x=0."""
    from brl_core.balancer import DDPGBalancer
    from brl_core.orchestrator import BRLTrainingOrchestrator
    from brl_core.tweaker import StochasticTweakerActorCritic
    from config import SystemConfig

    class _MockEnv:
        def reset(self, options=None):
            return np.zeros(27, dtype=np.float32), {"morphology_mask": np.ones(8, dtype=np.float32)}

        def step(self, action):
            return (
                np.zeros(27, dtype=np.float32),
                99.0,  # shaped dummy reward
                False,
                True,
                {
                    "raw_env_reward": 1.25,
                    "exploration_progress": 0.4,
                    "brl_metrics": {"x_velocity": 0.0, "total_distance": 0.0},
                },
            )

        def get_exploration_progress(self):
            return 0.4

    class _MockAgent:
        def __init__(self) -> None:
            self.observed_rewards: list[float] = []
            self.exp_progress_calls: list[float] = []

        def set_exploration_progress(self, progress: float) -> None:
            self.exp_progress_calls.append(float(progress))

        def select_action(self, obs, deterministic=False, limb_kinematics=None):
            return np.zeros(8, dtype=np.float32)

        def observe_transition(self, obs, act, rew, next_obs, term, trunc, info):
            self.observed_rewards.append(float(rew))

        def train_step(self):
            return {}

    cfg = SystemConfig()
    cfg.brl.warmup_episodes = 5
    cfg.brl.brl_episodes = 10
    cfg.ant_morph.max_episode_steps = 1
    cfg.ant_morph.use_real_returns = True

    agent = _MockAgent()
    orch = BRLTrainingOrchestrator(
        env=_MockEnv(),
        agent=agent,
        tweaker=StochasticTweakerActorCritic(cfg.brl, device="cpu"),
        balancer=DDPGBalancer(cfg.brl, device="cpu"),
        config=cfg,
    )
    summary = orch.run_episode(episode_idx=1, max_steps=1, train_agent_every=10)
    assert agent.observed_rewards == pytest.approx([1.25], abs=1e-6)
    assert summary["agent_episode_return"] == pytest.approx(1.25, abs=1e-6)
    assert summary["episode_return"] == pytest.approx(1.25, abs=1e-6)
    # Velocity-gated exploration stays at 0.0 decay when mean_x_vel=0.0
    assert agent.exp_progress_calls[-1] == pytest.approx(0.0, abs=1e-6)

    # When use_real_returns=False, orchestrator forwards shaped step reward & un-gated exploration
    cfg_shaped = SystemConfig()
    cfg_shaped.brl.warmup_episodes = 5
    cfg_shaped.brl.brl_episodes = 10
    cfg_shaped.ant_morph.max_episode_steps = 1
    cfg_shaped.ant_morph.use_real_returns = False
    agent_shaped = _MockAgent()
    orch_shaped = BRLTrainingOrchestrator(
        env=_MockEnv(),
        agent=agent_shaped,
        tweaker=StochasticTweakerActorCritic(cfg_shaped.brl, device="cpu"),
        balancer=DDPGBalancer(cfg_shaped.brl, device="cpu"),
        config=cfg_shaped,
    )
    summary_s = orch_shaped.run_episode(episode_idx=1, max_steps=1, train_agent_every=10)
    assert agent_shaped.observed_rewards == pytest.approx([99.0], abs=1e-6)
    assert summary_s["agent_episode_return"] == pytest.approx(99.0, abs=1e-6)
    assert agent_shaped.exp_progress_calls[-1] == pytest.approx(0.4, abs=1e-6)



def test_ctrl_cost_cancellation_holds_until_1000_and_ramps_to_3500() -> None:
    """Verifies ctrl_cost_weight=0.0 cancels 100% of torque penalty up to 1000 return, then ramps linearly to 0.5 at 3500."""
    wrapper = MorphologicalAntBRLWrapper(
        max_action_capacity=12,
        fixed_action_space=True,
        action_ema_alpha=1.0,
        action_smoothness_coef=0.0,
        min_healthy_vel=0.0,
        stationary_penalty_coef=0.0,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.0,
        ctrl_start_return_ref=1000.0,
        ctrl_return_ref=3500.0,
        forward_return_ref=5000.0,
        use_real_returns=False,
    )
    try:
        wrapper.reset(seed=42, options={"eval_mode": False})
        act = np.full(12, 0.5, dtype=np.float32)  # 8 active actuators -> ||a||^2 = 2.0, reward_ctrl = -1.0

        # 1) At 0 and 1000 real return: 100% torque penalty cancelled
        for ret in (0.0, 1000.0):
            wrapper.target_return = ret
            wrapper.current_return = ret
            assert wrapper.get_current_ctrl_cost_weight() == pytest.approx(0.0, abs=1e-6)
            _, r, _, _, info = wrapper.step(act)
            assert float(info["ctrl_cost_weight"]) == pytest.approx(0.0, abs=1e-6)
            assert float(info["reward_ctrl"]) == pytest.approx(-1.0, abs=1e-5)
            assert r - float(info["raw_env_reward"]) == pytest.approx(1.0, abs=1e-5)

        # 2) At 2250 real return (midpoint between 1000 and 3500): 50% cancelled (eff_ctrl_weight = 0.25)
        wrapper.target_return = 2250.0
        wrapper.current_return = 2250.0
        assert wrapper.get_current_ctrl_cost_weight() == pytest.approx(0.25, abs=1e-6)
        _, mid_r, _, _, mid_info = wrapper.step(act)
        assert float(mid_info["ctrl_cost_weight"]) == pytest.approx(0.25, abs=1e-6)
        assert mid_r - float(mid_info["raw_env_reward"]) == pytest.approx(0.5, abs=1e-5)

        # 3) At 3500+ real return: full standard Ant-v5 torque penalty (eff_ctrl_weight = 0.5)
        wrapper.target_return = 3500.0
        wrapper.current_return = 3500.0
        assert wrapper.get_current_ctrl_cost_weight() == pytest.approx(0.5, abs=1e-6)
        _, full_r, _, _, full_info = wrapper.step(act)
        assert float(full_info["ctrl_cost_weight"]) == pytest.approx(0.5, abs=1e-6)
        assert full_r == pytest.approx(float(full_info["raw_env_reward"]), abs=1e-5)

        # 4) Eval mode always keeps standard 0.5 regardless of low training return
        wrapper.current_return = 0.0
        wrapper.reset(seed=42, options={"eval_mode": True})
        _, eval_r, _, _, eval_info = wrapper.step(act)
        assert float(eval_info["ctrl_cost_weight"]) == pytest.approx(0.5, abs=1e-6)
        assert eval_r == pytest.approx(float(eval_info["raw_env_reward"]), abs=1e-5)
    finally:
        wrapper.close()
