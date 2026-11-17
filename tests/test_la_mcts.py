"""Verification Gate 1: GMM & MCTS Engine Unit Tests (la_mcts/)."""

from __future__ import annotations

import time
import numpy as np
import torch

from la_mcts.backbone import SelfAttentionBackbone, combine_cumulative_rep
from la_mcts.gmm_heads import MultiPeakGMMActorHead, SupportCriticHead
from la_mcts.rep_trainer import ModelRepresentationTrainer
from la_mcts.sampled_mcts import SampledContinuousMCTS
from config import LAMCTSConfig
from la_mcts.agent import LatentActionMCTSAgent


def test_gate1_backbone_normalization_bounds() -> None:
    """Verifies z_t in [0, 1]^(B x 256) strictly bounded across random and extreme inputs."""
    torch.manual_seed(42)
    backbone = SelfAttentionBackbone(
        obs_dim=27, k_tok=8, d_model=256, n_heads=4, l_enc=3, latent_dim=256
    )
    backbone.eval()

    obs = torch.randn(16, 27) * 25.0
    context = torch.rand(16, 12)
    z_t = backbone(obs, context=context)

    assert z_t.shape == (16, 256)
    assert torch.all(torch.isfinite(z_t))
    assert float(z_t.min().item()) >= 0.0
    assert float(z_t.max().item()) <= 1.0
    # Min-Max scaling maps each row's minimum to 0.0 and maximum to ~1.0
    row_mins = z_t.min(dim=-1)[0]
    row_maxs = z_t.max(dim=-1)[0]
    assert torch.allclose(row_mins, torch.zeros_like(row_mins), atol=1e-5)
    assert torch.allclose(row_maxs, torch.ones_like(row_maxs), atol=1e-4)


def test_gate1_gmm_actor_head_1000_points() -> None:
    """Tests MultiPeakGMMActorHead(M=5, N=8) on 1000 points a in (-1, 1)^8 for NaN/Inf freedom."""
    torch.manual_seed(42)
    actor_head = MultiPeakGMMActorHead(
        latent_dim=256, action_dim=8, num_peaks=5, log_std_bounds=(-5.0, 2.0)
    )
    # Verify architectural rule: strictly 3 linear layers directly from z_t
    linear_children = [m for m in actor_head.children() if isinstance(m, torch.nn.Linear)]
    assert len(linear_children) == 3

    z_t = torch.rand(1000, 256)
    # Generate 1000 continuous actions across (-1, 1)^8 including near-boundary points
    actions = torch.empty(1000, 8).uniform_(-0.9999, 0.9999)

    log_probs = actor_head.log_prob(z_t, actions)
    assert log_probs.shape == (1000,)
    assert not torch.any(torch.isnan(log_probs)), "log_prob produced NaN"
    assert not torch.any(torch.isinf(log_probs)), "log_prob produced Inf"

    # Also test sampling K_act = 16 actions
    sampled, priors = actor_head.sample_actions(z_t[:8], k_act=16, temperature=1.25)
    assert sampled.shape == (8, 16, 8)
    assert priors.shape == (8, 16)
    assert float(sampled.min().item()) >= -1.0
    assert float(sampled.max().item()) <= 1.0
    assert torch.allclose(priors.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_gate1_rep_trainer_and_sampled_mcts_latency() -> None:
    """Verifies ModelRepresentationTrainer and SampledContinuousMCTS (< 15 ms, sum(pi) == 1)."""
    torch.manual_seed(42)
    np.random.seed(42)

    backbone = SelfAttentionBackbone(
        obs_dim=27, k_tok=8, d_model=256, n_heads=4, l_enc=3, latent_dim=256
    )
    actor_head = MultiPeakGMMActorHead(
        latent_dim=256, action_dim=8, num_peaks=5, log_std_bounds=(-5.0, 2.0)
    )
    critic_head = SupportCriticHead(latent_dim=256, use_categorical=True)
    rep_trainer = ModelRepresentationTrainer(
        latent_dim=256, action_dim=8, num_blocks=3, proj_dim=512, pred_dim=256
    )

    backbone.eval()
    actor_head.eval()
    critic_head.eval()
    rep_trainer.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    actor_head = actor_head.to(device)
    critic_head = critic_head.to(device)
    rep_trainer = rep_trainer.to(device)

    obs = torch.randn(1, 27, device=device)
    with torch.inference_mode():
        z_t = backbone(obs).squeeze(0)
        a_test = torch.zeros(1, 8, device=device)
        z_next, r_hat = rep_trainer(z_t.unsqueeze(0), a_test)

    assert z_next.shape == (1, 256)
    assert r_hat.shape == (1, 1)
    assert float(z_next.min().item()) >= 0.0
    assert float(z_next.max().item()) <= 1.0

    mcts = SampledContinuousMCTS(
        actor_head=actor_head,
        critic_head=critic_head,
        rep_trainer=rep_trainer,
        n_sim=50,
        k_act=16,
    )

    # Warmup call for PyTorch kernel caching
    mcts.search(z_t, deterministic=False)

    # Timed benchmark over 5 runs (take median latency)
    timings_ms = []
    chosen_action = None
    policy_weights = None
    for _ in range(5):
        t0 = time.perf_counter()
        chosen_action, policy_weights, root_samples, root_val, _ = mcts.search(
            z_t, deterministic=False
        )
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    median_ms = float(np.median(timings_ms))
    assert chosen_action is not None and policy_weights is not None
    assert chosen_action.shape == (8,)
    assert np.all(chosen_action >= -1.0) and np.all(chosen_action <= 1.0)
    assert policy_weights.shape == (16,)
    assert abs(float(np.sum(policy_weights)) - 1.0) <= 1e-6
    max_latency = 50.0 if device.type == "cuda" else 2500.0
    assert median_ms < max_latency, f"MCTS latency {median_ms:.2f} ms exceeded {max_latency} ms limit"


def test_gate1_mcts_root_dirichlet_noise() -> None:
    """Verifies Dirichlet noise mixing on root prior weights during search and search_batch."""
    torch.manual_seed(123)
    np.random.seed(123)

    actor_head = MultiPeakGMMActorHead(
        latent_dim=64, action_dim=4, num_peaks=3, log_std_bounds=(-5.0, 2.0)
    )
    critic_head = SupportCriticHead(latent_dim=64, use_categorical=False)
    rep_trainer = ModelRepresentationTrainer(
        latent_dim=64, action_dim=4, num_blocks=1, proj_dim=128, pred_dim=64
    )

    actor_head.eval()
    critic_head.eval()
    rep_trainer.eval()

    z_root = torch.rand(64)

    # 1. Non-deterministic search with Dirichlet noise enabled (eps = 0.25, alpha = 0.3)
    mcts_noisy = SampledContinuousMCTS(
        actor_head=actor_head,
        critic_head=critic_head,
        rep_trainer=rep_trainer,
        n_sim=5,
        k_act=8,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
    )

    _, _, _, _, root_node_noisy = mcts_noisy.search(z_root, deterministic=False)
    assert root_node_noisy.prior_weights.shape == (8,)
    assert np.all(root_node_noisy.prior_weights >= 0.0)
    assert abs(float(np.sum(root_node_noisy.prior_weights)) - 1.0) <= 1e-6

    # 2. Deterministic search disables Dirichlet noise and evaluates noise-free peak centers
    _, expected_det_priors = actor_head.sample_actions(
        z_root, k_act=8, temperature=mcts_noisy.root_tau_beta, deterministic_candidates=True
    )
    expected_det_priors_np = expected_det_priors.detach().cpu().numpy()

    _, _, _, _, root_node_det = mcts_noisy.search(z_root, deterministic=True)
    np.testing.assert_allclose(root_node_det.prior_weights, expected_det_priors_np, atol=1e-6)

    # 3. Fraction = 0.0 disables noise even when deterministic=False
    torch.manual_seed(42)
    np.random.seed(42)
    _, expected_priors = actor_head.sample_actions(
        z_root, k_act=8, temperature=mcts_noisy.root_tau_beta
    )
    expected_priors_np = expected_priors.detach().cpu().numpy()

    mcts_no_frac = SampledContinuousMCTS(
        actor_head=actor_head,
        critic_head=critic_head,
        rep_trainer=rep_trainer,
        n_sim=5,
        k_act=8,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.0,
    )
    torch.manual_seed(42)
    np.random.seed(42)
    _, _, _, _, root_node_no_frac = mcts_no_frac.search(z_root, deterministic=False)
    np.testing.assert_allclose(root_node_no_frac.prior_weights, expected_priors_np, atol=1e-6)

    # 4. Search batch testing: verify simplex bounds and shapes
    z_batch = torch.rand(3, 64)
    acts, pi_weights, root_acts, root_vals = mcts_noisy.search_batch(z_batch, deterministic=False)
    assert acts.shape == (3, 4)
    assert pi_weights.shape == (3, 8)
    assert root_acts.shape == (3, 8, 4)
    assert root_vals.shape == (3,)
    for b in range(3):
        assert abs(float(np.sum(pi_weights[b])) - 1.0) <= 1e-5
        assert np.all(pi_weights[b] >= 0.0)


def test_combine_cumulative_rep_bounds_and_fixed_point() -> None:
    """Verifies combine_cumulative_rep normalization, fixed-point identity, and zero-history mask."""
    torch.manual_seed(0)
    cur = torch.rand(4, 32)
    assert torch.equal(combine_cumulative_rep(cur, None, alpha=0.4), cur)
    assert torch.equal(combine_cumulative_rep(cur, cur, alpha=0.0), cur)

    # Fixed point: min_max_normalize(cur + 0.4 * cur) == min_max_normalize(cur)
    # when cur is already min-max normalized in [0, 1]
    c_min = cur.min(dim=-1, keepdim=True).values
    c_max = cur.max(dim=-1, keepdim=True).values
    cur_norm = (cur - c_min) / (c_max - c_min)
    fp = combine_cumulative_rep(cur_norm, cur_norm, alpha=0.4)
    torch.testing.assert_close(fp, cur_norm, atol=1e-5, rtol=1e-5)

    # Zero prev row is skipped per batch item, non-zero row is combined & bounded in [0, 1]
    prev = torch.rand(4, 32)
    prev[1].zero_()
    out = combine_cumulative_rep(cur, prev, alpha=0.4)
    assert float(out.min().item()) >= 0.0
    assert float(out.max().item()) <= 1.0
    torch.testing.assert_close(out[1], cur[1])


def test_compute_prev_cumulative_matches_sequential_ema() -> None:
    """Verifies compute_prev_cumulative skips zero history and matches sequential accumulation."""
    torch.manual_seed(7)
    backbone = SelfAttentionBackbone(
        obs_dim=10, max_action_dim=4, context_dim=6, k_tok=4, d_model=32, n_heads=2, l_enc=1, latent_dim=32, history_len=3
    )
    backbone.eval()

    zero_hist = torch.zeros(2, 3 * (10 + 4))
    ctx = torch.randn(2, 6)
    assert backbone.compute_prev_cumulative(ctx, zero_hist, alpha=0.4) is None
    assert backbone.compute_prev_cumulative(ctx, None, alpha=0.4) is None

    # Populate batch 0 with 2 valid older steps (step index 1 and 2) and batch 1 empty
    hist = torch.zeros(2, 3, 14)
    hist[0, 1, :10] = torch.randn(10)
    hist[0, 1, 10:] = torch.randn(4)
    hist[0, 2, :10] = torch.randn(10)
    hist[0, 2, 10:] = torch.randn(4)
    flat_hist = hist.reshape(2, -1)

    prev_cum = backbone.compute_prev_cumulative(ctx, flat_hist, alpha=0.4)
    assert prev_cum is not None
    assert prev_cum.shape == (2, 32)
    torch.testing.assert_close(prev_cum[1], torch.zeros(32))

    with torch.no_grad():
        sub_hist_1 = torch.zeros_like(hist[0:1])
        sub_hist_2 = torch.zeros_like(hist[0:1])
        sub_hist_2[0, 2] = hist[0, 1]
        z1 = backbone(hist[0:1, 1, :10], context=ctx[0:1], history=sub_hist_1.reshape(1, -1))
        z2 = backbone(hist[0:1, 2, :10], context=ctx[0:1], history=sub_hist_2.reshape(1, -1))
        expected_0 = combine_cumulative_rep(z2, z1, alpha=0.4).squeeze(0)
    torch.testing.assert_close(prev_cum[0], expected_0, atol=1e-5, rtol=1e-5)
    assert backbone.compute_prev_cumulative(ctx, flat_hist, alpha=0.0) is None


def test_mcts_search_uses_on_demand_cumulative_rep_and_keeps_cur_rep_dynamics() -> None:
    """Verifies MCTS keeps rep_trainer on cur_rep -> next_rep while feeding cum_rep to heads."""
    torch.manual_seed(11)
    actor_head = MultiPeakGMMActorHead(latent_dim=32, action_dim=4, num_peaks=3)
    critic_head = SupportCriticHead(latent_dim=32, use_categorical=True, support_min=-50, support_max=50, num_bins=101)
    rep_trainer = ModelRepresentationTrainer(
        latent_dim=32, action_dim=4, num_blocks=1, proj_dim=32, pred_dim=32, latent_ema_decay=0.4
    )
    mcts = SampledContinuousMCTS(
        actor_head=actor_head,
        critic_head=critic_head,
        rep_trainer=rep_trainer,
        n_sim=10,
        k_act=4,
        latent_ema_decay=0.4,
    )

    z_root = torch.rand(32)
    prev_cum = torch.rand(32)
    _, _, _, _, root = mcts.search(z_root, deterministic=True, prev_cum_state=prev_cum)

    assert root.z_state is not None
    assert root.z_cum_state is not None
    torch.testing.assert_close(root.z_state, z_root)
    expected_root_cum = combine_cumulative_rep(z_root, prev_cum, alpha=0.4)
    torch.testing.assert_close(root.z_cum_state, expected_root_cum)
    assert not torch.allclose(root.z_cum_state, root.z_state)

    expanded_children = list(root.children.values())
    assert len(expanded_children) > 0
    child = expanded_children[0]
    assert child.z_state is not None and child.z_cum_state is not None
    # Verify child.z_state was predicted from root.z_state (cur_rep), NOT root.z_cum_state
    child_edge_idx = next(idx for idx, c in root.children.items() if c is child)
    with torch.no_grad():
        expected_child_z, _ = rep_trainer.forward(
            root.z_state, root.sampled_actions[child_edge_idx]
        )
    torch.testing.assert_close(child.z_state, expected_child_z, atol=1e-5, rtol=1e-5)
    expected_child_cum = combine_cumulative_rep(child.z_state, root.z_cum_state, alpha=0.4)
    torch.testing.assert_close(child.z_cum_state, expected_child_cum, atol=1e-5, rtol=1e-5)


def test_agent_train_step_with_latent_ema_decay() -> None:
    """Verifies LatentActionMCTSAgent unroll training and reanalyze with latent_ema_decay=0.4."""
    torch.manual_seed(19)
    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=4,
        context_dim=6,
        history_len=3,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        n_sim=8,
        n_sim_eval=6,
        k_act=4,
        unroll_steps=2,
        batch_size=2,
        support_min=-20,
        support_max=20,
        num_bins=41,
        latent_ema_decay=0.4,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    agent.set_episode_context(tweak_vector=np.zeros(6, dtype=np.float32), action_mask=np.ones(4, dtype=np.float32))

    # Collect a short episode via select_action + observe_transition
    obs = np.random.randn(12).astype(np.float32)
    for step in range(8):
        act = agent.select_action(obs, deterministic=False)
        next_obs = np.random.randn(12).astype(np.float32)
        agent.observe_transition(
            obs, act, float(0.5), next_obs, terminated=(step == 7), truncated=False, info={}
        )
        obs = next_obs

    metrics = agent.train_step()
    assert "loss_total" in metrics
    assert np.isfinite(metrics["loss_total"])
    assert agent.rep_trainer.latent_ema_decay == 0.4
    assert agent.mcts.latent_ema_decay == 0.4
    agent.reanalyze_batch(num_states=2)


def test_velocity_driven_exploration_annealing() -> None:
    """Verifies velocity-progress exploration annealing across Agent, Balancer, and Tweaker."""
    from config import BRLConfig
    from brl_core.balancer import DDPGBalancer
    from brl_core.tweaker import StochasticTweakerActorCritic

    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=4,
        context_dim=6,
        history_len=2,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        n_sim=4,
        k_act=4,
        tau_act_train=1.0,
        tau_act_train_end=0.25,
        root_exploration_fraction=0.30,
        root_exploration_fraction_end=0.10,
        min_policy_std_target=0.22,
        min_policy_std_target_end=0.12,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))

    # Progress = 0.0 -> initial exploration parameters
    agent.set_exploration_progress(0.0)
    assert np.isclose(agent.get_current_tau_act_train(), 1.0)
    assert np.isclose(agent.get_current_root_exploration_fraction(), 0.30)
    assert np.isclose(agent.get_current_std_target(), 0.22)

    # Progress = 0.5 -> midpoint interpolation (root_exploration_fraction is step-based, not return-based)
    agent.set_exploration_progress(0.5)
    assert np.isclose(agent.get_current_tau_act_train(), 0.625)
    assert np.isclose(agent.get_current_root_exploration_fraction(), 0.30)
    assert np.isclose(agent.get_current_std_target(), 0.17)

    # Progress = 1.0 -> end-of-curriculum precision parameters
    agent.set_exploration_progress(1.0)
    assert np.isclose(agent.get_current_tau_act_train(), 0.25)
    assert np.isclose(agent.get_current_root_exploration_fraction(), 0.30)
    assert np.isclose(agent.get_current_std_target(), 0.12)

    # Can also move back up if velocity progress decreases (clamped to [0.0, 1.0])
    agent.set_exploration_progress(-0.2)
    assert np.isclose(agent.exploration_progress, 0.0)
    assert np.isclose(agent.get_current_std_target(), 0.22)

    # Verify Tweaker beta_ent and Balancer noise annealing
    brl_cfg = BRLConfig(
        num_params=6,
        balancer_noise_std=0.05,
        balancer_noise_std_end=0.01,
        beta_ent=0.01,
        beta_ent_end=0.002,
    )
    tweaker = StochasticTweakerActorCritic(brl_cfg, device="cpu")
    balancer = DDPGBalancer(brl_cfg, device="cpu")

    assert np.isclose(tweaker.get_current_beta_ent(0.0), 0.01)
    assert np.isclose(tweaker.get_current_beta_ent(0.5), 0.006)
    assert np.isclose(tweaker.get_current_beta_ent(1.0), 0.002)

    bal_state = balancer.build_state(ema_win_rate=0.5, last_win=1, dominance_score=0.5)
    limits, budget, costs, _ = balancer.select_constraints(
        state=bal_state, warmup=False, deterministic=False, exploration_progress=1.0
    )
    assert limits.shape == (6,) and costs.shape == (6,) and np.isfinite(budget)


def test_needle_point_a_coherent_latent_exploration() -> None:
    """Point A: Coherent global latent perturbations feed actor without corrupting critic/dynamics."""
    torch.manual_seed(7)
    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=4,
        context_dim=6,
        history_len=2,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        n_sim=6,
        k_act=6,
        latent_noise_std=0.08,
        latent_noise_std_end=0.0,
        mode_hold_steps=4,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    agent.set_episode_context()
    agent.set_exploration_progress(0.0)

    obs = np.random.randn(12).astype(np.float32)
    agent.select_action(obs, deterministic=False)
    noise_step0 = agent._latent_noise_unit.clone()
    assert noise_step0 is not None
    assert agent._mode_hold_remaining == 3

    # Held constant inside the active 4-step window
    agent.select_action(obs, deterministic=False)
    torch.testing.assert_close(agent._latent_noise_unit, noise_step0)
    assert agent._mode_hold_remaining == 2

    # Clean z_cum vs perturbed z_actor_root bounds and critic purity
    z_root = agent.backbone(
        torch.as_tensor(obs).unsqueeze(0),
        context=torch.zeros(1, 6),
        history=torch.zeros(1, 2 * 16),
    ).squeeze(0)
    z_actor = agent._get_actor_latent_root(z_root, prev_cum_root=None, ema_decay=0.4)
    assert z_actor is not None
    assert float(z_actor.min().item()) >= 0.0 and float(z_actor.max().item()) <= 1.0
    assert not torch.allclose(z_actor, z_root)

    with torch.no_grad():
        expected_v = float(agent.critic_head(z_root.unsqueeze(0)).squeeze().item())
    _, _, _, _, root = agent.mcts.search(
        z_root, deterministic=False, n_sim=4, z_actor_root=z_actor
    )
    assert np.isclose(root.v_value, expected_v, atol=1e-5)

    # Anneals completely to zero at exploration_progress = 1.0
    agent.set_exploration_progress(1.0)
    assert np.isclose(agent.get_current_latent_noise_std(), 0.0)
    assert agent._get_actor_latent_root(z_root, prev_cum_root=None, ema_decay=0.4) is None


def test_needle_point_b_mode_commitment_and_hold() -> None:
    """Point B: Composite GMM commits to peak_k and MCTS agent holds/anneals the commitment horizon."""
    torch.manual_seed(21)
    actor_head = MultiPeakGMMActorHead(latent_dim=32, action_dim=4, num_peaks=4)
    z_t = torch.rand(32)

    # Strong commitment onto peak 2 should make almost all sampled indices equal 2 while keeping valid priors
    sampled, priors, peaks = actor_head.sample_actions(
        z_t,
        k_act=64,
        temperature=1.2,
        committed_peak=2,
        mode_commitment_prob=0.90,
        return_peaks=True,
    )
    assert sampled.shape == (64, 4)
    assert peaks.shape == (64,)
    assert torch.allclose(priors.sum(), torch.tensor(1.0), atol=1e-5)
    frac_peak2 = float((peaks == 2).float().mean().item())
    assert frac_peak2 > 0.70

    # Agent hold horizon anneals from mode_hold_steps=8 down to 1 and resets at episode boundary
    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=4,
        context_dim=6,
        history_len=2,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        n_sim=4,
        k_act=4,
        mode_hold_steps=8,
        mode_commitment_prob=0.70,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    agent.set_exploration_progress(0.0)
    assert agent.get_current_mode_hold_steps() == 8
    assert np.isclose(agent.get_current_mode_commitment_prob(), 0.70)

    agent.set_exploration_progress(1.0)
    assert agent.get_current_mode_hold_steps() == 1
    assert np.isclose(agent.get_current_mode_commitment_prob(), 0.0)

    agent.set_exploration_progress(0.0)
    obs = np.random.randn(12).astype(np.float32)
    act = agent.select_action(obs, deterministic=False)
    assert agent._committed_peak is not None
    assert agent._mode_hold_remaining == 7
    agent.observe_transition(obs, act, 1.0, obs, terminated=True, truncated=False, info={})
    assert agent._committed_peak is None
    assert agent._mode_hold_remaining == 0


def test_needle_point_c_mixture_weight_entropy_floor() -> None:
    """Point C: Decaying categorical entropy bonus + floor prevents GMM peak collapse."""
    torch.manual_seed(33)
    backbone = SelfAttentionBackbone(
        obs_dim=10, max_action_dim=4, context_dim=6, k_tok=4, d_model=32, n_heads=2, l_enc=1, latent_dim=32, history_len=2
    )
    actor_head = MultiPeakGMMActorHead(latent_dim=32, action_dim=4, num_peaks=4)
    critic_head = SupportCriticHead(latent_dim=32, use_categorical=False)
    rep_trainer = ModelRepresentationTrainer(
        latent_dim=32, action_dim=4, num_blocks=1, proj_dim=32, pred_dim=32
    )

    # Deliberately collapse global mixture logits toward peak 0
    with torch.no_grad():
        actor_head.logits_net.bias.copy_(torch.tensor([8.0, -8.0, -8.0, -8.0]))

    obs_seq = torch.randn(2, 3, 10)
    act_seq = torch.zeros(2, 2, 4)
    rew_seq = torch.zeros(2, 2)
    val_seq = torch.zeros(2, 3)
    samples = torch.zeros(2, 3, 4, 4)
    weights = torch.full((2, 3, 4), 0.25)

    actor_head.zero_grad()
    total_loss, metrics = rep_trainer.compute_unrolled_loss(
        obs_seq=obs_seq,
        action_seq=act_seq,
        reward_seq=rew_seq,
        target_value_seq=val_seq,
        mcts_policy_samples=samples,
        mcts_policy_weights=weights,
        backbone=backbone,
        actor_head=actor_head,
        critic_head=critic_head,
        lambda_p=0.0,
        lambda_v=0.0,
        lambda_r=0.0,
        lambda_rep=0.0,
        lambda_ent=0.0,
        c_reg=0.0,
        lambda_peak_ent=0.15,
        peak_entropy_floor_ratio=0.70,
    )
    assert "loss_peak_ent" in metrics and "peak_ent_floor" in metrics
    assert np.isclose(metrics["peak_ent_floor"], 0.70 * np.log(4), atol=1e-4)
    assert metrics["peak_entropy"] < metrics["peak_ent_floor"]
    assert metrics["loss_peak_ent"] > 0.0

    # Gradient on collapsed peak 0 logit bias and per-limb weight head should push dominant mode down
    total_loss.backward()
    bias_grad = actor_head.logits_net.bias.grad
    assert bias_grad is not None
    assert float(bias_grad[0].item()) > 0.0
    assert float(bias_grad[1].item()) < 0.0
    limb_w_grad = actor_head.limb_head.weight_head.weight.grad
    assert limb_w_grad is not None and float(torch.norm(limb_w_grad).item()) > 0.0


def test_needle_point_d_critic_warmup_and_config_defaults() -> None:
    """Point D: Critic warmup scales actor sharpening and persists transition count; default_config loads cleanly."""
    import warnings
    from config import load_config

    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=4,
        context_dim=6,
        history_len=2,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        critic_warmup_steps=100,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    assert np.isclose(agent.get_critic_warmup_scale(), 0.15)
    assert np.isclose(agent.get_current_q_advantage_weight(), 0.60)

    agent.total_transitions_observed = 50
    assert np.isclose(agent.get_critic_warmup_scale(), 0.50)
    assert np.isclose(agent.get_current_q_advantage_weight(), 2.00)

    agent.total_transitions_observed = 200
    assert np.isclose(agent.get_critic_warmup_scale(), 1.0)
    assert np.isclose(agent.get_current_q_advantage_weight(), 4.0)

    sd = agent.state_dict()
    agent2 = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    agent2.load_state_dict(sd)
    assert agent2.total_transitions_observed == 200
    assert np.isclose(agent2.get_critic_warmup_scale(), 1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sys_cfg = load_config("config/default_config.yaml")
    assert not caught, f"Unexpected config warnings: {[str(w.message) for w in caught]}"
    assert sys_cfg.la_mcts.actor_type == "deterministic"
    assert np.isclose(sys_cfg.la_mcts.lambda_dpg, 0.05)
    assert sys_cfg.la_mcts.dpg_horizon == 1
    assert np.isclose(sys_cfg.la_mcts.action_explore_std, 0.40)
    assert np.isclose(sys_cfg.la_mcts.latent_noise_std, 0.25)
    assert sys_cfg.la_mcts.mode_hold_steps == 6
    assert np.isclose(sys_cfg.la_mcts.lambda_peak_ent, 0.15)
    assert np.isclose(sys_cfg.la_mcts.lambda_peak_sep, 0.25)
    assert sys_cfg.la_mcts.k_act == 8
    assert sys_cfg.la_mcts.critic_warmup_steps == 8000
    assert np.isclose(sys_cfg.la_mcts.lr, 1e-4)
    assert np.isclose(sys_cfg.la_mcts.c_1, 2.50)
    assert np.isclose(sys_cfg.la_mcts.tau_act_train, 1.25)
    assert sys_cfg.la_mcts.use_gait_phase is False
    assert sys_cfg.la_mcts.gait_phase_period == 16
    assert np.isclose(sys_cfg.ant_morph.ctrl_cost_weight, 0.0)
    assert np.isclose(sys_cfg.ant_morph.ctrl_start_return_ref, 1000.0)
    assert np.isclose(sys_cfg.ant_morph.ctrl_return_ref, 3500.0)
    assert sys_cfg.ant_morph.min_joint_vel > 0.0
    assert sys_cfg.ant_morph.joint_vel_penalty_coef > 0.0
    assert np.isclose(sys_cfg.ant_morph.joint_vel_start_return_ref, 1000.0)
    assert np.isclose(sys_cfg.ant_morph.joint_vel_end_return_ref, 3000.0)
    assert sys_cfg.ant_morph.pure_velocity_reward_mode is True
    assert np.isclose(sys_cfg.ant_morph.pure_vel_start_return_ref, 1000.0)
    assert np.isclose(sys_cfg.ant_morph.pure_vel_end_return_ref, 3000.0)
    assert 0.0 < sys_cfg.ant_morph.action_ema_alpha <= 1.0
    assert 0.0 < sys_cfg.la_mcts.root_exploration_fraction_decay_constant <= 1.0
    assert sys_cfg.brl.warmup_episodes == 150


def test_deterministic_eval_searches_noise_free_mode_centers() -> None:
    """Verifies deterministic=True MCTS uses exact GMM peak centers without stochastic jitter."""
    torch.manual_seed(55)
    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=4,
        context_dim=6,
        history_len=2,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        actor_type="gmm",
        n_sim=8,
        n_sim_eval=8,
        k_act=8,
        num_peaks=4,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    obs = np.ones(12, dtype=np.float32) * 0.25

    agent.set_episode_context()
    a1 = agent.select_action(obs, deterministic=True)
    agent.set_episode_context()
    a2 = agent.select_action(obs, deterministic=True)
    np.testing.assert_allclose(a1, a2, atol=1e-6)

    # Also verify include_peak_means anchors slots 0..num_peaks-1 during training sampling
    z = torch.rand(32)
    _, mu_targ, _ = agent.actor_head.get_distribution_params(z)
    expected_centers = torch.tanh(mu_targ).squeeze(0)
    sampled, priors = agent.actor_head.sample_actions(
        z, k_act=8, temperature=1.0, include_peak_means=True
    )
    assert sampled.shape == (8, 4)
    torch.testing.assert_close(sampled[:4], expected_centers, atol=1e-5, rtol=1e-5)
    assert torch.allclose(priors.sum(), torch.tensor(1.0), atol=1e-5)


def test_peak_mean_separation_regularization_gradient() -> None:
    """Verifies lambda_peak_sep produces nonzero repulsion gradient when peak means overlap."""
    torch.manual_seed(77)
    backbone = SelfAttentionBackbone(
        obs_dim=10, max_action_dim=4, context_dim=6, k_tok=4, d_model=32, n_heads=2, l_enc=1, latent_dim=32, history_len=2
    )
    actor_head = MultiPeakGMMActorHead(latent_dim=32, action_dim=4, num_peaks=4)
    critic_head = SupportCriticHead(latent_dim=32, use_categorical=False)
    rep_trainer = ModelRepresentationTrainer(
        latent_dim=32, action_dim=4, num_blocks=1, proj_dim=32, pred_dim=32
    )

    # Force all peaks to have nearly identical means initially (tiny non-zero weight so diff > 0)
    with torch.no_grad():
        actor_head.limb_head.mu_head.weight.fill_(1e-3)
        actor_head.limb_head.mu_head.bias.zero_()

    obs_seq = torch.randn(2, 3, 10)
    act_seq = torch.zeros(2, 2, 4)
    rew_seq = torch.zeros(2, 2)
    val_seq = torch.zeros(2, 3)
    samples = torch.zeros(2, 3, 4, 4)
    weights = torch.full((2, 3, 4), 0.25)

    actor_head.zero_grad()
    loss, metrics = rep_trainer.compute_unrolled_loss(
        obs_seq=obs_seq,
        action_seq=act_seq,
        reward_seq=rew_seq,
        target_value_seq=val_seq,
        mcts_policy_samples=samples,
        mcts_policy_weights=weights,
        backbone=backbone,
        actor_head=actor_head,
        critic_head=critic_head,
        lambda_p=0.0,
        lambda_v=0.0,
        lambda_r=0.0,
        lambda_rep=0.0,
        lambda_ent=0.0,
        lambda_peak_ent=0.0,
        c_reg=0.0,
        lambda_peak_sep=0.25,
        min_peak_separation=0.40,
    )
    assert metrics["peak_mean_sep"] < 0.05
    assert metrics["loss_peak_sep"] > 0.05
    loss.backward()
    mu_w_grad = actor_head.limb_head.mu_head.weight.grad
    assert mu_w_grad is not None and float(mu_w_grad.abs().sum().item()) > 0.0


def test_balancer_slew_rate_and_low_win_gating() -> None:
    """Verifies Balancer never spikes budget post-warmup and refuses upward climbs at low win rate."""
    from config import BRLConfig
    from brl_core.balancer import DDPGBalancer

    brl_cfg = BRLConfig(
        num_params=6,
        budget_max=8500.0,
        warmup_limit=30.0,
        warmup_budget=90.0,
    )
    balancer = DDPGBalancer(brl_cfg, device="cpu")
    # Force actor final linear bias strongly positive (wanting budget ~ 8500)
    with torch.no_grad():
        balancer.actor.net[-2].bias.fill_(5.0)

    # Low win rate (0.05 < 0.35) must not increase budget above warmup_budget (90.0)
    bal_state_low = balancer.build_state(ema_win_rate=0.05, last_win=0, dominance_score=0.1)
    _, budget_low, _, _ = balancer.select_constraints(
        state=bal_state_low,
        warmup=False,
        deterministic=False,
        exploration_progress=0.1,
        ema_win_rate=0.05,
    )
    assert budget_low <= 90.0 + 1e-5

    # Healthy win rate (0.55) allows gradual slew-limited progression, capped by exploration_progress
    bal_state_ok = balancer.build_state(ema_win_rate=0.55, last_win=1, dominance_score=0.6)
    _, budget_step1, _, _ = balancer.select_constraints(
        state=bal_state_ok,
        warmup=False,
        deterministic=False,
        exploration_progress=0.0,
        ema_win_rate=0.55,
    )
    assert 90.0 < budget_step1 <= 190.0 + 1e-5  # 90 + max(100, 0.15*90) = 190
    assert budget_step1 <= 250.0  # exploration_progress=0.0 cap


def test_deterministic_limb_actor_head_and_single_limb_sharing() -> None:
    """Verifies DeterministicActorHead outputs exact morphology-invariant actions without GMM noise."""
    from la_mcts.gmm_heads import DeterministicActorHead

    torch.manual_seed(91)
    for act_dim, joints in [(8, 2), (12, 3), (17, None)]:
        actor = DeterministicActorHead(
            latent_dim=64, action_dim=act_dim, hidden_dim=32, joints_per_limb=joints
        )
        actor.eval()
        z = torch.randn(2, 64)
        a1 = actor(z)
        a2 = actor(z)
        assert a1.shape == (2, act_dim)
        assert float(a1.min().item()) >= -1.0 and float(a1.max().item()) <= 1.0
        torch.testing.assert_close(a1, a2)

    # Verify single-limb query matches corresponding joints in vectorized forward pass
    ant_actor = DeterministicActorHead(latent_dim=64, action_dim=8, hidden_dim=32, joints_per_limb=2)
    ant_actor.eval()
    z1 = torch.randn(64)
    kin = ant_actor.default_limb_kinematics
    all_act = ant_actor(z1.unsqueeze(0), limb_kinematics=kin).squeeze(0)
    leg0_single = ant_actor.get_limb_action(
        z1,
        anchor=kin[0, :3],
        yaw=kin[0, 3],
        pitch=kin[0, 4],
        num_joints=2,
        all_limbs_kinematics=kin,
    )
    torch.testing.assert_close(leg0_single, all_act[:2], atol=1e-5, rtol=1e-5)

    # Verify phase_film and mu_head biases initialize at 0.0 (preventing static stance offset)
    torch.testing.assert_close(ant_actor.limb_head.phase_film.bias, torch.zeros_like(ant_actor.limb_head.phase_film.bias))
    torch.testing.assert_close(ant_actor.limb_head.mu_head.bias, torch.zeros_like(ant_actor.limb_head.mu_head.bias))

    # Verify FiLM z-conditioning keeps DeterministicActorHead responsive to latent state z
    z_grid = torch.rand(32, 64)
    a_grid = ant_actor(z_grid)
    assert float(a_grid.std(dim=0).mean().item()) > 0.08

    # Verify held_action_noise places +held into slot 1, antithetic -held into slot 2, and uses uniform priors
    held = torch.ones(8) * 0.8
    cands, priors = ant_actor.sample_actions(
        z1, k_act=8, temperature=1.0, action_noise_std=0.35, held_action_noise=held
    )
    assert cands.shape == (8, 8)
    expected_slot1 = torch.clamp(all_act + 0.35 * held, -1.0, 1.0)
    expected_slot2 = torch.clamp(all_act - 0.35 * held, -1.0, 1.0)
    torch.testing.assert_close(cands[1], expected_slot1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cands[2], expected_slot2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(priors, torch.full_like(priors, 1.0 / 8.0), atol=1e-6, rtol=1e-6)
    assert float(cands.std(dim=0).mean().item()) > 0.15


def test_model_based_dpg_gradients_and_world_model_freeze() -> None:
    """Verifies Model-Based DPG gives analytical per-limb gradients without modifying world model/critic."""
    from la_mcts.gmm_heads import DeterministicActorHead

    torch.manual_seed(101)
    actor = DeterministicActorHead(latent_dim=32, action_dim=4, hidden_dim=32)
    critic = SupportCriticHead(latent_dim=32, use_categorical=True, support_min=-20, support_max=20, num_bins=41)
    rep = ModelRepresentationTrainer(latent_dim=32, action_dim=4, num_blocks=1, proj_dim=32, pred_dim=32)

    z_0 = torch.rand(4, 32)
    cum_0 = torch.rand(4, 32)
    actor.zero_grad()
    rep.zero_grad()
    critic.zero_grad()

    loss_dpg, dpg_metrics = rep.compute_deterministic_policy_loss(
        z_0=z_0,
        cum_0=cum_0,
        actor_head=actor,
        critic_head=critic,
        gamma=0.99,
        dpg_horizon=2,
        lambda_dpg=0.50,
    )
    assert np.isfinite(dpg_metrics["loss_dpg"])
    assert np.isfinite(dpg_metrics["dpg_q_mean"])
    assert dpg_metrics["dpg_q_scale"] >= 1.0

    loss_dpg.backward()
    actor_grad_norm = sum(
        float(p.grad.abs().sum().item()) for p in actor.parameters() if p.grad is not None
    )
    assert actor_grad_norm > 0.0
    # World model and critic weights must remain completely untouched by DPG backward pass
    assert all(p.grad is None for p in rep.parameters())
    assert all(p.grad is None for p in critic.parameters())


def test_deterministic_agent_end_to_end_mcts_and_dpg_step() -> None:
    """Verifies Deterministic Network 2 agent runs O(1) eval, latent MCTS search, reanalyze, and DPG training."""
    torch.manual_seed(111)
    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=4,
        context_dim=6,
        history_len=2,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        actor_type="deterministic",
        lambda_dpg=0.50,
        dpg_horizon=2,
        n_sim=8,
        n_sim_eval=8,
        k_act=6,
        unroll_steps=2,
        batch_size=2,
        support_min=-20,
        support_max=20,
        num_bins=41,
        critic_warmup_steps=10,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    assert getattr(agent.actor_head, "is_deterministic", False) is True

    agent.set_episode_context()
    obs = np.random.randn(12).astype(np.float32)
    det_a1 = agent.select_action(obs, deterministic=True)
    det_a_step1 = agent.select_action(obs, deterministic=True)
    # 2D periodic gait-phase clock prevents identical static pose across consecutive steps
    assert float(np.abs(det_a_step1 - det_a1).mean()) > 1e-4
    agent.set_episode_context()
    det_a2 = agent.select_action(obs, deterministic=True)
    np.testing.assert_allclose(det_a1, det_a2, atol=1e-6)

    held_noise_ref = None
    for step in range(6):
        act = agent.select_action(obs, deterministic=False)
        if step == 0:
            held_noise_ref = agent._det_explore_noise.clone()
        else:
            torch.testing.assert_close(agent._det_explore_noise, held_noise_ref)
        next_obs = np.random.randn(12).astype(np.float32)
        agent.observe_transition(
            obs, act, 0.5, next_obs, terminated=(step == 5), truncated=False, info={}
        )
        obs = next_obs
    assert agent._det_explore_noise is None

    # Verify loading pre-z_film legacy deterministic state dict succeeds cleanly
    sd_legacy = agent.state_dict()
    sd_legacy["actor_head"] = {
        k: v
        for k, v in sd_legacy["actor_head"].items()
        if "z_film" not in k and "phase_film" not in k
    }
    agent_compat = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    agent_compat.load_state_dict(sd_legacy)

    metrics = agent.train_step()
    assert "loss_dpg" in metrics and np.isfinite(metrics["loss_dpg"])
    assert "dpg_q_mean" in metrics and np.isfinite(metrics["dpg_q_mean"])
    agent.reanalyze_batch(num_states=2)


def test_step_based_root_exploration_fraction_decay() -> None:
    """Verifies root_exploration_fraction decays exponentially by steps taken and clips at end."""
    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=4,
        context_dim=6,
        history_len=2,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        root_exploration_fraction=0.75,
        root_exploration_fraction_end=0.12,
        root_exploration_fraction_decay_constant=0.99,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    assert np.isclose(agent.get_current_root_exploration_fraction(), 0.75)

    # Return/exploration_progress changes must NOT affect root_exploration_fraction
    agent.set_exploration_progress(1.0)
    assert np.isclose(agent.get_current_root_exploration_fraction(), 0.75)

    agent.total_transitions_observed = 50
    expected_50 = 0.75 * (0.99 ** 50)
    assert np.isclose(agent.get_current_root_exploration_fraction(), expected_50, atol=1e-7)

    # Large step count must clip cleanly at root_exploration_fraction_end
    agent.total_transitions_observed = 5000
    assert np.isclose(agent.get_current_root_exploration_fraction(), 0.12, atol=1e-7)

    # State dict persistence preserves decayed schedule across resume
    sd = agent.state_dict()
    agent_resumed = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    agent_resumed.load_state_dict(sd)
    assert np.isclose(agent_resumed.get_current_root_exploration_fraction(), 0.12, atol=1e-7)








def test_deterministic_awr_distills_positive_advantage_actions_over_slot0_collapse() -> None:
    """Verifies AWR pulls DeterministicActorHead toward high-return executed actions even when Slot 0 dominates MCTS weights."""
    from la_mcts.gmm_heads import DeterministicActorHead, compute_gait_phase
    from la_mcts.backbone import SelfAttentionBackbone

    torch.manual_seed(202)
    backbone = SelfAttentionBackbone(
        obs_dim=12, k_tok=4, d_model=32, n_heads=2, l_enc=1, latent_dim=32
    )
    actor = DeterministicActorHead(latent_dim=32, action_dim=4, hidden_dim=32)
    critic = SupportCriticHead(
        latent_dim=32, use_categorical=True, support_min=-50, support_max=50, num_bins=101
    )
    rep = ModelRepresentationTrainer(
        latent_dim=32, action_dim=4, num_blocks=1, proj_dim=32, pred_dim=32
    )
    opt = torch.optim.Adam(actor.parameters(), lr=5e-3)

    bsz, unroll_k, k_act = 4, 2, 4
    obs_seq = torch.randn(bsz, unroll_k + 1, 12)
    with torch.no_grad():
        z0 = backbone(obs_seq[:, 0])
        ph0 = compute_gait_phase(torch.zeros(bsz), period_steps=20, dtype=z0.dtype)
        a_slot0_before = actor(z0, phase=ph0)

    # Winning executed action in trajectory 0, static/zero action in trajectories 1..3
    a_win = torch.tensor([0.85, -0.85, 0.85, -0.85], dtype=torch.float32)
    act_seq = torch.zeros(bsz, unroll_k, 4)
    act_seq[0, :, :] = a_win.unsqueeze(0)
    rew_seq = torch.zeros(bsz, unroll_k)
    rew_seq[0, :] = 10.0
    target_val_seq = torch.zeros(bsz, unroll_k + 1)
    target_val_seq[0, :] = 25.0  # Strong positive advantage over critic (~0) and batch (~0)

    # Simulate Slot 0 winning MCTS weights (argmax=0), which previously produced zero actor pull toward a_win
    samples_seq = torch.zeros(bsz, unroll_k + 1, k_act, 4)
    samples_seq[:, 0, 0, :] = a_slot0_before
    weights_seq = torch.zeros(bsz, unroll_k + 1, k_act)
    weights_seq[:, :, 0] = 0.85
    weights_seq[:, :, 1:] = 0.05

    dist_before = float((a_slot0_before[0] - a_win).abs().mean().item())
    phase_seq = compute_gait_phase(
        torch.arange(unroll_k + 1, dtype=torch.float32).unsqueeze(0).expand(bsz, -1),
        period_steps=20,
    )

    for _ in range(15):
        opt.zero_grad()
        loss, metrics = rep.compute_unrolled_loss(
            obs_seq=obs_seq,
            action_seq=act_seq,
            reward_seq=rew_seq,
            target_value_seq=target_val_seq,
            mcts_policy_samples=samples_seq,
            mcts_policy_weights=weights_seq,
            backbone=backbone,
            actor_head=actor,
            critic_head=critic,
            lambda_p=1.0,
            lambda_v=0.1,
            lambda_r=0.1,
            lambda_rep=0.1,
            phase_seq=phase_seq,
        )
        assert metrics["loss_p_awr"] > 0.0
        loss.backward()
        opt.step()

    with torch.no_grad():
        a_slot0_after = actor(backbone(obs_seq[:, 0]), phase=ph0)
    dist_after = float((a_slot0_after[0] - a_win).abs().mean().item())
    assert dist_after < dist_before - 0.1


def test_deterministic_limb_head_dynamic_cosine_and_dpg_grad_clip() -> None:
    """Verifies per-limb phase harmonics eliminate static 0.975 cosine similarity and DPG grad is clipped."""
    from la_mcts.gmm_heads import DeterministicActorHead, compute_gait_phase

    torch.manual_seed(303)
    actor = DeterministicActorHead(latent_dim=64, action_dim=8, hidden_dim=64, joints_per_limb=2)
    lh = actor.limb_head
    z_rand = torch.rand(16, 64)
    phases = compute_gait_phase(torch.arange(16, dtype=torch.float32), period_steps=20)
    kin = actor.default_limb_kinematics.unsqueeze(0).expand(16, -1, -1)
    geom = lh.encode_kinematics(kin)

    with torch.no_grad():
        h = lh._encode_limbs(z_rand, geom, phase=phases)
        for i in range(4):
            hi = torch.nn.functional.normalize(h[:, i, :], dim=-1)
            sim_mat = hi @ hi.T
            off_diag = float(((sim_mat.sum() - sim_mat.diag().sum()) / (16 * 15)).item())
            assert off_diag < 0.35, f"Limb {i} off-diagonal cosine similarity too high: {off_diag}"

        a_phase = actor(z_rand[0:1].expand(16, -1), phase=phases)
        assert float(a_phase.std(dim=0).mean().item()) > 0.08


def test_use_gait_phase_false_disables_clock_and_centers_cross_limb_kin_emb() -> None:
    """Verifies use_gait_phase=False disables the phase clock and centered kin_emb differentiates all 4 limbs."""
    from la_mcts.gmm_heads import DeterministicActorHead

    torch.manual_seed(404)
    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=8,
        context_dim=6,
        history_len=2,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        actor_type="deterministic",
        use_gait_phase=False,
        gait_phase_period=16,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    obs = np.random.randn(12).astype(np.float32)

    agent.set_episode_context()
    agent._episode_step = 0
    a_step0 = agent.select_action(obs, deterministic=True)

    agent.set_episode_context()
    agent._episode_step = 5
    a_step5 = agent.select_action(obs, deterministic=True)

    # With use_gait_phase=False and identical history, _episode_step has zero effect on action
    np.testing.assert_allclose(a_step0, a_step5, atol=1e-6)

    # Verify centered limb geometry produces distinct (low/negative cosine similarity) kin_emb across the 4 Ant legs
    actor = DeterministicActorHead(latent_dim=64, action_dim=8, hidden_dim=64, joints_per_limb=2)
    lh = actor.limb_head
    kin = actor.default_limb_kinematics.unsqueeze(0)  # (1, 4, 11)
    geom = lh.encode_kinematics(kin)
    geom_centered = geom.clone()
    geom_centered[..., 0:3] = geom_centered[..., 0:3] * 4.0
    geom_centered[..., 3:4] = (geom_centered[..., 3:4] - 0.5) * 2.0
    geom_centered[..., 6:7] = (geom_centered[..., 6:7] - 0.5) * 2.0
    geom_centered[..., 7:8] = geom_centered[..., 7:8] - 1.0
    geom_centered[..., 9::3] = geom_centered[..., 9::3] - 1.0
    with torch.no_grad():
        kin_emb = torch.nn.functional.normalize(lh.kin_proj(geom_centered).squeeze(0), dim=-1)
        cross_limb_sim = kin_emb @ kin_emb.T
        off_diag_limb_sim = float(((cross_limb_sim.sum() - cross_limb_sim.diag().sum()) / (4 * 3)).item())
        assert off_diag_limb_sim < 0.25, f"Cross-limb kin_emb cosine similarity too high: {off_diag_limb_sim}"


def test_deterministic_eval_runs_mcts_lookahead_and_unroll_isolates_world_model() -> None:
    """Verifies deterministic=True expands k_act candidates via MCTS lookahead and loss_p uses detached real latents."""
    from la_mcts.gmm_heads import DeterministicActorHead

    torch.manual_seed(505)
    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=4,
        context_dim=6,
        history_len=2,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        actor_type="deterministic",
        use_gait_phase=False,
        n_sim=8,
        n_sim_eval=12,
        k_act=6,
        unroll_steps=2,
        batch_size=2,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    agent.set_episode_context()
    obs = np.random.randn(12).astype(np.float32)

    # 1) deterministic=True expands k_act=6 structured candidates and runs n_sim_eval=12 MCTS simulations
    act_eval = agent.select_action(obs, deterministic=True)
    assert act_eval.shape == (4,)
    assert agent._last_mcts_info is not None
    root_samples, policy_weights, _ = agent._last_mcts_info
    assert root_samples.shape == (6, 4)
    assert policy_weights.shape == (6,)
    assert float(np.std(root_samples, axis=0).mean()) > 0.05

    # 2) In compute_unrolled_loss, policy loss (lambda_p=1.0, others=0.0) must only update actor_head,
    #    keeping backbone and rep_trainer world-model gradients strictly zero.
    bsz, unroll_k = 2, 2
    obs_seq = torch.randn(bsz, unroll_k + 1, 12)
    act_seq = torch.randn(bsz, unroll_k, 4).clamp(-1.0, 1.0)
    rew_seq = torch.ones(bsz, unroll_k)
    val_seq = torch.ones(bsz, unroll_k + 1) * 5.0
    samps = torch.randn(bsz, unroll_k + 1, 6, 4).clamp(-1.0, 1.0)
    weights = torch.full((bsz, unroll_k + 1, 6), 1.0 / 6.0)

    agent.backbone.zero_grad()
    agent.rep_trainer.zero_grad()
    agent.actor_head.zero_grad()
    loss_p_only, _ = agent.rep_trainer.compute_unrolled_loss(
        obs_seq=obs_seq,
        action_seq=act_seq,
        reward_seq=rew_seq,
        target_value_seq=val_seq,
        mcts_policy_samples=samps,
        mcts_policy_weights=weights,
        backbone=agent.backbone,
        actor_head=agent.actor_head,
        critic_head=agent.critic_head,
        lambda_v=0.0,
        lambda_p=1.0,
        lambda_r=0.0,
        lambda_rep=0.0,
        c_reg=0.0,
    )
    loss_p_only.backward()
    actor_gn = sum(
        float(p.grad.abs().sum().item())
        for p in agent.actor_head.parameters()
        if p.grad is not None
    )
    bb_gn = sum(
        float(p.grad.abs().sum().item())
        for p in agent.backbone.parameters()
        if p.grad is not None
    )
    rep_gn = sum(
        float(p.grad.abs().sum().item())
        for p in agent.rep_trainer.parameters()
        if p.grad is not None
    )
    assert actor_gn > 0.0
    assert bb_gn == 0.0
    assert rep_gn == 0.0



