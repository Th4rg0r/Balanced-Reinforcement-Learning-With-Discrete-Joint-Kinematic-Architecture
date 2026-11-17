"""Model Representation Trainer and SimSiam Consistency Head for LA-MCTS."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from la_mcts.backbone import min_max_normalize, SelfAttentionBackbone, combine_cumulative_rep
from la_mcts.gmm_heads import (
    DeterministicActorHead,
    MultiPeakGMMActorHead,
    SupportCriticHead,
)


class SimSiamConsistencyHead(nn.Module):
    """SimSiam-style Projector and Predictor MLPs for self-supervised representation consistency."""

    def __init__(
        self,
        latent_dim: int = 256,
        proj_dim: int = 512,
        pred_dim: int = 256,
        beta_mse: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta_mse = beta_mse
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.LayerNorm(pred_dim),
            nn.SiLU(),
            nn.Linear(pred_dim, proj_dim),
        )

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return self.projector(z)

    def predict(self, p: torch.Tensor) -> torch.Tensor:
        return self.predictor(p)

    def forward(
        self,
        z_hat: torch.Tensor,
        z_true: torch.Tensor,
        target_projector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Computes hybrid SimSiam negative cosine similarity + stop-gradient MSE loss."""
        p_hat = self.projector(z_hat)
        q_hat = self.predictor(p_hat)
        proj = target_projector if target_projector is not None else self.projector
        with torch.no_grad():
            p_true = proj(z_true.detach()).detach()

        cos_sim = F.cosine_similarity(q_hat, p_true, dim=-1, eps=1e-8)
        cos_loss = (1.0 - cos_sim).mean()
        mse_loss = F.mse_loss(z_hat, z_true.detach())
        return cos_loss + self.beta_mse * mse_loss


class FiLMResidualBlock(nn.Module):
    """Action-Conditioned FiLM Residual Block: h + W2 * SiLU(gamma * LN(W1 * h) + beta)."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(latent_dim, latent_dim)

    def forward(
        self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        mod = gamma * self.ln(self.fc1(h)) + beta
        return h + self.fc2(self.act(mod))


class ModelRepresentationTrainer(nn.Module):
    """Latent dynamics and reward model g_psi(z_t, a_t) -> (z_hat_{t+1}, r_hat_t)."""

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 8,
        num_blocks: int = 3,
        proj_dim: int = 512,
        pred_dim: int = 256,
        beta_mse: float = 1.0,
        latent_ema_decay: float = 0.4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_blocks = num_blocks
        self.proj_dim = proj_dim
        self.pred_dim = pred_dim
        self.latent_ema_decay = float(latent_ema_decay)

        # Action encoder producing FiLM modulation parameters (gamma_a, beta_a)
        self.action_emb = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim),
        )

        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(latent_dim=latent_dim) for _ in range(num_blocks)]
        )
        self.out_ln = nn.LayerNorm(latent_dim)

        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.SiLU(),
            nn.Linear(latent_dim // 2, 1),
        )

        self.consistency_head = SimSiamConsistencyHead(
            latent_dim=latent_dim,
            proj_dim=proj_dim,
            pred_dim=pred_dim,
            beta_mse=beta_mse,
        )
        self._dpg_q_scale: float = 1.0
        self._last_root_latents: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _align_action_dim(self, action: torch.Tensor) -> torch.Tensor:
        cur_dim = action.shape[-1]
        if cur_dim == self.action_dim:
            return action
        if cur_dim < self.action_dim:
            pad = torch.zeros(
                *action.shape[:-1],
                self.action_dim - cur_dim,
                device=action.device,
                dtype=action.dtype,
            )
            return torch.cat([action, pad], dim=-1)
        return action[..., : self.action_dim]

    def forward(
        self, z_t: torch.Tensor, a_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts next penultimate representation z_hat_{t+1} in [0, 1]^d and reward r_hat_t."""
        squeeze_out = False
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
            squeeze_out = True
        if a_t.dim() == 1:
            a_t = a_t.unsqueeze(0)

        a_t = self._align_action_dim(a_t)
        film_params = self.action_emb(a_t)
        gamma_a, beta_a = torch.chunk(film_params, 2, dim=-1)
        # Center gamma around 1.0 for stable identity initialization
        gamma_a = 1.0 + gamma_a

        h = z_t
        for block in self.blocks:
            h = block(h, gamma_a, beta_a)

        z_next = min_max_normalize(self.out_ln(h), eps=1e-6)
        r_hat = self.reward_head(h)

        if squeeze_out:
            return z_next.squeeze(0), r_hat.squeeze(0)
        return z_next, r_hat

    def compute_unrolled_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        target_value_seq: torch.Tensor,
        mcts_policy_samples: torch.Tensor,
        mcts_policy_weights: torch.Tensor,
        backbone: SelfAttentionBackbone,
        actor_head: MultiPeakGMMActorHead | DeterministicActorHead,
        critic_head: SupportCriticHead,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        limb_kinematics: Optional[torch.Tensor] = None,
        lambda_v: float = 0.5,
        lambda_p: float = 1.0,
        lambda_r: float = 1.0,
        lambda_rep: float = 4.0,
        lambda_ent: float = 0.05,
        c_reg: float = 1e-4,
        target_backbone: Optional[SelfAttentionBackbone] = None,
        target_projector: Optional[nn.Module] = None,
        min_policy_std_target: float = 0.22,
        lambda_peak_ent: float = 0.15,
        peak_entropy_floor_ratio: float = 0.70,
        lambda_peak_sep: float = 0.25,
        min_peak_separation: float = 0.40,
        phase_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes K-step unrolled total loss with SimSiam representation consistency.

        Args:
            obs_seq: (B, K+1, D_s)
            action_seq: (B, K, N)
            reward_seq: (B, K)
            target_value_seq: (B, K+1)
            mcts_policy_samples: (B, K+1, K_act, N)
            mcts_policy_weights: (B, K+1, K_act)
        """
        bsz, k_plus_1, obs_dim = obs_seq.shape
        unroll_k = k_plus_1 - 1

        # Encode step k = 0 with online backbone so gradients flow into representation
        hist_0 = history[:, 0, :] if (history is not None and history.dim() == 3) else history
        z_hat = backbone(obs_seq[:, 0], context=context, history=hist_0)
        prev_cum_0 = backbone.compute_prev_cumulative(
            context, hist_0, alpha=self.latent_ema_decay
        )
        cum_hat = combine_cumulative_rep(z_hat, prev_cum_0, alpha=self.latent_ema_decay)
        self._last_root_latents = (z_hat.detach(), cum_hat.detach())

        # Encode k = 1..K target representations under no_grad using target_backbone when provided
        z_target_seq: Optional[torch.Tensor] = None
        if unroll_k > 0:
            targ_bb = target_backbone if target_backbone is not None else backbone
            with torch.no_grad():
                flat_obs_targ = obs_seq[:, 1:].reshape(bsz * unroll_k, obs_dim)
                flat_ctx_targ = (
                    context.unsqueeze(1)
                    .expand(bsz, unroll_k, -1)
                    .reshape(bsz * unroll_k, -1)
                    if context is not None
                    else None
                )
                flat_hist_targ = None
                if history is not None:
                    if history.dim() == 3:
                        flat_hist_targ = history[:, 1:].reshape(bsz * unroll_k, -1)
                    elif history.dim() == 2:
                        flat_hist_targ = (
                            history.unsqueeze(1)
                            .expand(bsz, unroll_k, -1)
                            .reshape(bsz * unroll_k, -1)
                        )
                z_target_seq = targ_bb(
                    flat_obs_targ, context=flat_ctx_targ, history=flat_hist_targ
                ).view(bsz, unroll_k, -1)

        # Active action dimension scaling for pre-tanh log-prob (avoids negative Jacobian offset)
        if action_mask is not None:
            active_dims = (
                action_mask.sum().clamp_min(1.0)
                if action_mask.dim() == 1
                else action_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            )
        else:
            active_dims = float(actor_head.action_dim)
        loss_r = torch.tensor(0.0, device=obs_seq.device, dtype=obs_seq.dtype)
        loss_rep = torch.tensor(0.0, device=obs_seq.device, dtype=obs_seq.dtype)
        cum_list = [cum_hat]

        for k in range(1, unroll_k + 1):
            z_hat, r_hat = self.forward(z_hat, action_seq[:, k - 1])
            # MuZero / EfficientZero 0.5 gradient scaling across unroll steps
            z_hat = 0.5 * z_hat + 0.5 * z_hat.detach()
            cum_hat = combine_cumulative_rep(z_hat, cum_hat, alpha=self.latent_ema_decay)
            cum_list.append(cum_hat)

            z_target_k = z_target_seq[:, k - 1, :]
            loss_rep = loss_rep + self.consistency_head(
                z_hat, z_target_k, target_projector=target_projector
            )
            loss_r = loss_r + F.smooth_l1_loss(
                r_hat.squeeze(-1), reward_seq[:, k - 1], beta=1.0
            )

        n_steps = unroll_k + 1
        cum_seq = torch.stack(cum_list, dim=1)  # (B, K+1, D)
        cum_flat = cum_seq.reshape(bsz * n_steps, -1)
        target_v_flat = target_value_seq[:, :n_steps].reshape(bsz * n_steps)
        loss_v = critic_head.compute_loss(cum_flat, target_v_flat) * float(n_steps)

        if limb_kinematics is None:
            kin_flat = None
        elif limb_kinematics.dim() == 4:
            kin_flat = limb_kinematics[:, :n_steps].reshape(
                bsz * n_steps, limb_kinematics.size(2), limb_kinematics.size(3)
            )
        elif limb_kinematics.dim() == 3:
            kin_flat = (
                limb_kinematics.unsqueeze(1)
                .expand(bsz, n_steps, limb_kinematics.size(1), limb_kinematics.size(2))
                .reshape(bsz * n_steps, limb_kinematics.size(1), limb_kinematics.size(2))
            )
        else:
            kin_flat = limb_kinematics

        if action_mask is None or action_mask.dim() == 1:
            mask_flat = action_mask
        else:
            mask_flat = (
                action_mask.unsqueeze(1)
                .expand(bsz, n_steps, action_mask.size(-1))
                .reshape(bsz * n_steps, action_mask.size(-1))
            )

        samples_flat = mcts_policy_samples[:, :n_steps].reshape(
            bsz * n_steps, mcts_policy_samples.size(2), mcts_policy_samples.size(3)
        )
        weights_seq = mcts_policy_weights[:, :n_steps]
        is_det_actor = bool(getattr(actor_head, "is_deterministic", False))

        if is_det_actor:
            phase_flat = None
            if phase_seq is not None:
                if phase_seq.dim() == 3:
                    phase_flat = phase_seq[:, :n_steps].reshape(bsz * n_steps, -1)
                elif phase_seq.dim() == 2:
                    phase_flat = (
                        phase_seq.unsqueeze(1)
                        .expand(bsz, n_steps, phase_seq.size(-1))
                        .reshape(bsz * n_steps, -1)
                    )
                else:
                    phase_flat = phase_seq
            a_det_flat, pre_tanh_flat = actor_head(
                cum_flat,
                limb_kinematics=kin_flat,
                action_mask=mask_flat,
                return_pre_tanh=True,
                phase=phase_flat,
            )
            a_det_seq = a_det_flat.view(bsz, n_steps, -1)
            # 1) Sharpened MCTS visit/Q barycenter avoids brittle Slot-0 argmax lock-in
            w_sharp = weights_seq.pow(2.0)
            w_sharp = w_sharp / w_sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            a_mcts_target = torch.sum(
                w_sharp.unsqueeze(-1) * mcts_policy_samples[:, :n_steps], dim=2
            )
            sq_err = (a_det_seq - a_mcts_target).pow(2)
            pre_tanh_sq = pre_tanh_flat.view(bsz, n_steps, -1).pow(2)
            if action_mask is not None:
                m_seq = action_mask.view(1, 1, -1) if action_mask.dim() == 1 else action_mask.unsqueeze(1)
                loss_p_mcts = ((sq_err * m_seq).sum(dim=-1) / active_dims).mean(dim=0).sum()
                pre_tanh_reg = ((pre_tanh_sq * m_seq).sum(dim=-1) / active_dims).mean()
            else:
                m_seq = None
                loss_p_mcts = (sq_err.sum(dim=-1) / active_dims).mean(dim=0).sum()
                pre_tanh_reg = pre_tanh_sq.mean()

            # 2) Direct Advantage-Weighted Regression (AWR) on executed replay actions so
            #    high-return exploratory motions distill straight into the deterministic actor
            n_awr = min(unroll_k, action_seq.size(1), target_value_seq.size(1)) if unroll_k > 0 else 0
            if n_awr > 0 and action_seq.numel() > 0:
                with torch.no_grad():
                    v_pred_seq = critic_head(
                        cum_seq[:, :n_awr].detach().reshape(bsz * n_awr, -1)
                    ).view(bsz, n_awr)
                    adv = target_value_seq[:, :n_awr] - v_pred_seq
                    adv_scale = adv.std().clamp_min(1.0)
                    adv_score = torch.relu((adv - adv.mean()) / adv_scale) * (
                        adv > -0.5 * adv_scale
                    ).to(adv.dtype)
                    w_awr = (torch.exp(adv_score.clamp(max=3.0)) - 1.0).clamp(max=6.0)
                awr_sq = (a_det_seq[:, :n_awr] - action_seq[:, :n_awr]).pow(2)
                if m_seq is not None:
                    awr_mse = (awr_sq * m_seq).sum(dim=-1) / active_dims
                else:
                    awr_mse = awr_sq.sum(dim=-1) / active_dims
                loss_p_awr = 2.0 * (w_awr.detach() * awr_mse).mean()
            else:
                loss_p_awr = torch.tensor(0.0, device=obs_seq.device, dtype=obs_seq.dtype)

            loss_p = loss_p_mcts + loss_p_awr * float(n_steps) + 1e-4 * pre_tanh_reg

            # Normalize over unroll horizon
            loss_v = loss_v / float(k_plus_1)
            loss_p = loss_p / float(k_plus_1)
            if unroll_k > 0:
                loss_r = loss_r / float(unroll_k)
                loss_rep = loss_rep / float(unroll_k)

            reg_loss = torch.tensor(0.0, device=obs_seq.device, dtype=obs_seq.dtype)
            if c_reg > 0.0:
                for p in self.parameters():
                    reg_loss = reg_loss + p.pow(2).sum()
                for p in backbone.parameters():
                    reg_loss = reg_loss + p.pow(2).sum()

            total_loss = (
                lambda_v * loss_v
                + lambda_p * loss_p
                + lambda_r * loss_r
                + lambda_rep * loss_rep
                + c_reg * reg_loss
            )
            metrics = {
                "loss_total": float(total_loss.detach().item()),
                "loss_v": float(loss_v.detach().item()),
                "loss_p": float(loss_p.detach().item()),
                "loss_p_mcts": float((loss_p_mcts / float(k_plus_1)).detach().item()),
                "loss_p_awr": float(loss_p_awr.detach().item()),
                "loss_r": float(loss_r.detach().item()),
                "loss_rep": float(loss_rep.detach().item()),
                "loss_peak_ent": 0.0,
                "loss_peak_sep": 0.0,
                "policy_entropy": 0.0,
                "peak_entropy": 0.0,
                "peak_ent_floor": 0.0,
                "peak_mean_sep": 0.0,
                "policy_std": 0.0,
            }
            return total_loss, metrics

        log_probs_flat, policy_entropy, peak_entropy, mean_policy_std, mu_flat = (
            actor_head.compute_log_prob_pre_tanh_and_entropy(
                cum_flat,
                samples_flat,
                action_mask=mask_flat,
                limb_kinematics=kin_flat,
                return_mu=True,
            )
        )
        log_probs_seq = log_probs_flat.view(bsz, n_steps, -1)
        loss_p = (
            -torch.sum(weights_seq * log_probs_seq, dim=-1) / active_dims
        ).mean(dim=0).sum()

        # Normalize over unroll horizon
        loss_v = loss_v / float(k_plus_1)
        loss_p = loss_p / float(k_plus_1)
        if unroll_k > 0:
            loss_r = loss_r / float(unroll_k)
            loss_rep = loss_rep / float(unroll_k)

        # L2 weight regularization
        reg_loss = torch.tensor(0.0, device=obs_seq.device, dtype=obs_seq.dtype)
        if c_reg > 0.0:
            for p in self.parameters():
                reg_loss = reg_loss + p.pow(2).sum()
            for p in backbone.parameters():
                reg_loss = reg_loss + p.pow(2).sum()

        # Mixture-weight categorical entropy regularization (Point C):
        # keeps all GMM modes alive and distinct while the critic warms up,
        # without forcing destructive per-joint Gaussian variance.
        max_peak_ent = math.log(max(1, int(actor_head.num_peaks)))
        clamped_floor_ratio = min(1.0, max(0.0, float(peak_entropy_floor_ratio)))
        peak_ent_floor = clamped_floor_ratio * max_peak_ent
        peak_floor_penalty = F.relu(peak_ent_floor - peak_entropy).mean() * 2.0
        eff_lambda_peak = max(0.0, float(lambda_peak_ent))
        loss_peak_ent = eff_lambda_peak * (peak_floor_penalty - peak_entropy.mean())

        # Pairwise geometric separation between GMM peak means tanh(mu_i), tanh(mu_j)
        # so multi-step mode commitment explores genuinely distinct gaits/postures.
        m_peaks = mu_flat.size(1)
        if m_peaks >= 2:
            squash_mu = torch.tanh(mu_flat)
            diff = squash_mu.unsqueeze(2) - squash_mu.unsqueeze(1)
            if mask_flat is not None:
                if mask_flat.dim() == 1:
                    m_exp = mask_flat.view(1, 1, 1, -1)
                    denom_sep = mask_flat.sum().clamp_min(1.0)
                else:
                    m_exp = mask_flat.unsqueeze(1).unsqueeze(1)
                    denom_sep = mask_flat.sum(dim=-1).clamp_min(1.0).view(-1, 1, 1)
                rms_dist = torch.sqrt((diff.pow(2) * m_exp).sum(dim=-1) / denom_sep + 1e-6)
            else:
                rms_dist = torch.sqrt(diff.pow(2).mean(dim=-1) + 1e-6)
            triu_mask = torch.triu(
                torch.ones((m_peaks, m_peaks), device=mu_flat.device, dtype=torch.bool),
                diagonal=1,
            )
            pair_dist = rms_dist[:, triu_mask]
            peak_mean_sep = pair_dist.mean()
            eff_lambda_sep = max(0.0, float(lambda_peak_sep))
            loss_peak_sep = eff_lambda_sep * F.relu(float(min_peak_separation) - pair_dist).mean()
        else:
            peak_mean_sep = torch.tensor(0.0, device=obs_seq.device, dtype=obs_seq.dtype)
            loss_peak_sep = torch.tensor(0.0, device=obs_seq.device, dtype=obs_seq.dtype)

        # Per-joint Gaussian std bounds [std_floor, std_ceil]
        std_floor = max(0.05, float(min_policy_std_target))
        std_ceil = max(std_floor + 0.15, 0.45)
        std_floor_bonus = -F.relu(std_floor - mean_policy_std).mean() * 5.0
        std_ceil_penalty = F.relu(mean_policy_std - std_ceil).pow(2).mean() * 5.0
        loss_ent = std_floor_bonus - std_ceil_penalty

        total_loss = (
            lambda_v * loss_v
            + lambda_p * loss_p
            + lambda_r * loss_r
            + lambda_rep * loss_rep
            - lambda_ent * loss_ent
            + loss_peak_ent
            + loss_peak_sep
            + c_reg * reg_loss
        )

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_v": float(loss_v.detach().item()),
            "loss_p": float(loss_p.detach().item()),
            "loss_r": float(loss_r.detach().item()),
            "loss_rep": float(loss_rep.detach().item()),
            "loss_peak_ent": float(loss_peak_ent.detach().item()),
            "loss_peak_sep": float(loss_peak_sep.detach().item()),
            "policy_entropy": float(policy_entropy.mean().detach().item()),
            "peak_entropy": float(peak_entropy.mean().detach().item()),
            "peak_ent_floor": float(peak_ent_floor),
            "peak_mean_sep": float(peak_mean_sep.detach().item()),
            "policy_std": float(mean_policy_std.mean().detach().item()),
        }
        return total_loss, metrics

    def compute_deterministic_policy_loss(
        self,
        z_0: torch.Tensor,
        cum_0: torch.Tensor,
        actor_head: MultiPeakGMMActorHead | DeterministicActorHead,
        critic_head: SupportCriticHead,
        action_mask: Optional[torch.Tensor] = None,
        limb_kinematics: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        dpg_horizon: int = 2,
        lambda_dpg: float = 0.50,
        phase_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes analytical Deterministic Policy Gradient loss over a frozen latent world-model rollout.

        Backpropagates d/d_phi [sum gamma^h r_hat + gamma^H V(z_cum)] directly into `actor_head`
        without modifying `rep_trainer` or `critic_head` weights.
        """
        z_det = z_0.detach()
        cum_det = cum_0.detach()
        horizon = max(1, int(dpg_horizon))

        frozen_params = list(self.parameters()) + list(critic_head.parameters())
        prev_flags = [p.requires_grad for p in frozen_params]
        for p in frozen_params:
            p.requires_grad_(False)

        try:
            is_det = bool(getattr(actor_head, "is_deterministic", False))
            bsz = z_det.shape[0]
            q_total = torch.zeros(bsz, device=z_det.device, dtype=z_det.dtype)
            pre_tanh_reg = torch.tensor(0.0, device=z_det.device, dtype=z_det.dtype)
            discount = 1.0
            z_cur = z_det
            cum_cur = cum_det

            if is_det:
                for h in range(horizon):
                    if limb_kinematics is not None and limb_kinematics.dim() == 4:
                        kin_h = limb_kinematics[:, min(h, limb_kinematics.size(1) - 1)]
                    else:
                        kin_h = limb_kinematics
                    if phase_seq is not None and phase_seq.dim() == 3:
                        phase_h = phase_seq[:, min(h, phase_seq.size(1) - 1)]
                    else:
                        phase_h = phase_seq
                    act_h, pre_h = actor_head(
                        cum_cur,
                        limb_kinematics=kin_h,
                        action_mask=action_mask,
                        return_pre_tanh=True,
                        phase=phase_h,
                    )
                    pre_tanh_reg = pre_tanh_reg + pre_h.pow(2).mean()
                    z_next, r_hat = self.forward(z_cur, act_h)
                    cum_next = combine_cumulative_rep(
                        z_next, cum_cur, alpha=self.latent_ema_decay, assume_nonzero=True
                    )
                    q_total = q_total + discount * r_hat.squeeze(-1)
                    discount *= float(gamma)
                    z_cur = z_next
                    cum_cur = cum_next
                v_boot = critic_head(cum_cur).squeeze(-1)
                q_total = q_total + discount * v_boot
            else:
                kin_0 = (
                    limb_kinematics[:, 0]
                    if (limb_kinematics is not None and limb_kinematics.dim() == 4)
                    else limb_kinematics
                )
                logits, mu, _ = actor_head.get_distribution_params(
                    cum_cur, temperature=1.0, limb_kinematics=kin_0
                )
                probs = F.softmax(logits, dim=-1).detach()  # (B, M)
                best_k = torch.argmax(probs, dim=-1)
                idx = best_k.unsqueeze(-1).unsqueeze(-1).expand(bsz, 1, mu.shape[-1])
                mu_best = torch.gather(mu, dim=1, index=idx).squeeze(1)
                act_0 = torch.tanh(mu_best)
                if action_mask is not None:
                    act_0 = act_0 * action_mask
                pre_tanh_reg = mu_best.pow(2).mean()
                z_next, r_hat = self.forward(z_cur, act_0)
                cum_next = combine_cumulative_rep(
                    z_next, cum_cur, alpha=self.latent_ema_decay, assume_nonzero=True
                )
                v_boot = critic_head(cum_next).squeeze(-1)
                q_total = r_hat.squeeze(-1) + float(gamma) * v_boot

            with torch.no_grad():
                if q_total.numel() >= 2:
                    q_span = float(
                        max(1.0, (torch.quantile(q_total, 0.95) - torch.quantile(q_total, 0.05)).item())
                    )
                else:
                    q_span = max(1.0, float(q_total.abs().mean().item()))
                self._dpg_q_scale = 0.95 * self._dpg_q_scale + 0.05 * q_span
                scale = max(1.0, float(self._dpg_q_scale))

            loss_dpg = -float(lambda_dpg) * (q_total / scale).mean() + 1e-4 * (
                pre_tanh_reg / float(horizon)
            )
            dpg_metrics = {
                "loss_dpg": float(loss_dpg.detach().item()),
                "dpg_q_mean": float(q_total.mean().detach().item()),
                "dpg_q_scale": float(scale),
            }
            return loss_dpg, dpg_metrics
        finally:
            for p, flag in zip(frozen_params, prev_flags):
                p.requires_grad_(flag)
