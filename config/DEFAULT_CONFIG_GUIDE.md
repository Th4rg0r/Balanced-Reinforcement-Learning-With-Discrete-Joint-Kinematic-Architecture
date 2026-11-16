# Balanced Reinforcement Learning & LA-MCTS: Configuration Guide (`default_config.yaml`)

This document provides a comprehensive reference for every configuration parameter in [`config/default_config.yaml`](default_config.yaml). The parameters are grouped into four primary subsystems:
1. **`la_mcts`**: Generic Latent-Action Monte Carlo Tree Search, Transformer World Model, and Categorical Value Critic.
2. **`brl`**: Balanced Reinforcement Learning Tri-Agent Curriculum (Agent, Tweaker, Balancer).
3. **`ant_morph`**: Morphological MuJoCo Ant Environment, Dynamic XML Engine, and Rolling Raw-Return Curriculum.
4. **`run`**: Experiment Execution, Parallelism, and Hardware Resource Allocation.

---

## 1. Latent-Action MCTS (`la_mcts`)

### 1.1 Architecture & Representation Dimensions

#### `obs_dim`
- **Type**: `int` | **Default**: `27`
- **Purpose**: Observation space dimensionality for the nominal 4-legged Ant ($z$-position, 4D orientation quaternion, 8 joint angles, 6D torso velocities, 8 joint velocities).
- **Training Impact**:
  - Must match the underlying environment's observation vector length.
  - Increasing or mismatching causes shape mismatch errors in the observation projection layer.

#### `max_action_dim`
- **Type**: `int` | **Default**: `8`
- **Purpose**: Number of active continuous control channels for the nominal creature (2 actuators per leg $\times$ 4 legs = 8 torques).
- **Training Impact**:
  - Determines the action output dimensionality of the policy prior and the action input dimension for the dynamics model.
  - If morphology scaling is enabled (e.g. 6-legged Ant with 12 actuators), `max_action_capacity` in `ant_morph` determines total capacity, while `max_action_dim` defines the nominal baseline.

#### `k_tok`
- **Type**: `int` | **Default**: `8`
- **Purpose**: Number of token slices into which the continuous input vector is divided for the Self-Attention Transformer representation network $h_\theta(o, w_{\text{ctx}})$.
- **Training Impact**:
  - **Higher**: Provides finer-grained attention between proprioceptive sub-groups (torso vs. individual limbs). Increases computational cost quadratically ($O(k_{\text{tok}}^2)$ attention matrix).
  - **Lower**: Blends features earlier, reducing expressiveness of spatial relationships across limbs.

#### `d_model`
- **Type**: `int` | **Default**: `256`
- **Purpose**: Internal embedding width of the Transformer representation backbone.
- **Training Impact**:
  - **Higher (e.g. 512)**: Increases capacity for complex multimodal dynamics and contact interactions, but slows down inference and MCTS expansion steps; risks overfitting if replay buffer is small.
  - **Lower (e.g. 128)**: Faster MCTS simulations, but may underfit complex terrain and contact dynamics.

#### `n_heads`
- **Type**: `int` | **Default**: `4`
- **Purpose**: Number of parallel attention heads in multi-head self-attention. Each head has dimension $d_k = d_{\text{model}} / n_{\text{heads}} = 64$.
- **Training Impact**:
  - Allows the agent to jointly attend to information from different representation subspaces (e.g. one head tracking velocity drift, another tracking leg contact phase).
  - Must evenly divide `d_model`.

#### `l_enc`
- **Type**: `int` | **Default**: `2`
- **Purpose**: Number of stacked Transformer encoder layers in the observation representation network $h_\theta$.
- **Training Impact**:
  - **Higher**: Deeper hierarchical reasoning over history and context, but increases latency during MCTS root node evaluation.
  - **Lower**: Minimal inference latency; 2 layers provides an optimal balance between representation depth and fast MCTS simulation throughput while `MorphologicalLimbHead` handles inter-limb attention.

#### `latent_dim`
- **Type**: `int` | **Default**: `256`
- **Purpose**: Dimensionality of the abstract latent state vector $s \in \mathbb{R}^{\text{latent\_dim}}$ traversed by the MCTS tree search.
- **Training Impact**:
  - All recurrent tree unrolls happen in this latent space without simulating raw MuJoCo physics.
  - Should match `d_model` to avoid bottlenecking information transfer between the representation and dynamics networks.

#### `context_dim`
- **Type**: `int` | **Default**: `18`
- **Purpose**: Dimensionality of the perceived context/tweak vector $w_{\text{ctx}} \in \mathbb{R}^{18}$ appended to the state representation.
- **Training Impact**:
  - Must match `brl.num_params`. Informs the Transformer of current body scale distortions and environmental perturbations.

#### `history_len`
- **Type**: `int` | **Default**: `6`
- **Purpose**: Temporal frame-stack horizon. The Transformer ingests a sliding window of past $(o_{t-H:t}, a_{t-H:t})$ transitions.
- **Training Impact**:
  - **Higher**: Enables robust system identification under noisy or delayed sensors (e.g., estimating leg mass and friction parameters from motion history). Increases memory footprint in replay buffer.
  - **Lower**: Struggles to disambiguate unobserved physical parameters (e.g. slippery ice vs. weak actuators) from instantaneous states.

#### `latent_ema_decay`
- **Type**: `float` | **Default**: `0.4`
- **Purpose**: Exponential moving average weight $\alpha \in [0, 1]$ used to blend cumulative historical latent representations ($z^{\text{cum}}_t = \text{norm}(z_t + \alpha z^{\text{cum}}_{t-1})$) for Actor and Critic readout while keeping latent dynamics transitions un-smeared ($z_t \to z_{t+1}$).

---

### 1.2 Policy Architecture & Actor Optimization

#### `actor_type`, `lambda_dpg`, `dpg_horizon`, `action_explore_std`, and `action_explore_std_end`
- **Defaults**:
  - `actor_type: "deterministic"`: Selects Network 2 implementation:
    - `"deterministic"` (Default): Uses `DeterministicActorHead` (`DeterministicLimbHead`), mapping `(z_t, c_i) -> a_i in [-1, 1]^J` directly via Pre-LN self-attention without composite GMM sampling, trained via analytical Model-Based DPG (`lambda_dpg`) plus MCTS anchor MSE (`lambda_p`).
    - `"gmm"`: Legacy `MultiPeakGMMActorHead` outputting multi-peak `(logits, mu, log_std)` distribution parameters.
  - `lambda_dpg: 0.50`: Weight of the analytical Deterministic Policy Gradient (`d/d_phi [sum gamma^h r_hat + gamma^H V(z_cum)]`) backpropagated per limb through the frozen world model and critic (gated by `critic_warmup_steps`).
  - `dpg_horizon: 2`: Latent rollout horizon (`2 * 50ms = 100ms`) differentiated per DPG update.
  - `action_explore_std: 0.12` $\to$ `action_explore_std_end: 0.02`: Exploration noise std around deterministic MCTS proposals during training, annealing linearly over `0 -> exploration_return_ref` (`3000`).

#### `num_peaks`
- **Type**: `int` | **Default**: `4`
- **Purpose**: Number of Gaussian mixture modes in the Latent-Action policy prior network $p_\theta(a \mid s)$.
- **Training Impact**:
  - Locomotion often features multiple discrete gaits (trot, gallop, crawl, turn left, turn right). A multi-modal mixture of Gaussians prevents mode averaging (where the policy would output zero torque trying to average left and right turns).
  - **Higher**: Captures rich multi-gait behaviors; slightly increases policy sampling time.
  - **Lower (e.g. 1)**: Collapses to a single unimodal Gaussian, which can cause policy stagnation in complex obstacle or terrain traversal.

#### `log_std_bounds`
- **Type**: `list[float]` | **Default**: `[-1.8, 0.25]`
- **Purpose**: Clamping bounds $[\log \sigma_{\min}, \log \sigma_{\max}]$ applied to the predicted Gaussian standard deviations ($\sigma \in [e^{-1.8}, e^{0.25}] \approx [0.165, 1.284]$).
- **Training Impact**:
  - **Lower bound (-1.8)**: Prevents premature covariance collapse while allowing low-noise precision gaits as the agent speeds up.
  - **Upper bound (0.25)**: Prevents excessive Gaussian action jitter during multi-step locomotion.

---

### 1.3 Value Critic & Return Distribution

#### `use_categorical_critic`
- **Type**: `bool` | **Default**: `true`
- **Purpose**: When `true`, enables a two-hot categorical distributional value critic (similar to MuZero / DreamerV3) instead of scalar mean-squared error (MSE).
- **Training Impact**:
  - **`true` (Recommended)**: Dramatically stabilizes value estimation across varied reward scales and prevents outlier returns from destabilizing gradient updates.
  - **`false`**: Uses scalar MSE regression, which is prone to gradient explosions when the agent falls or transitions between high-reward locomotion and severe fall penalties.

#### `support_min` and `support_max`
- **Type**: `int` | **Default**: `support_min: -150`, `support_max: 150`
- **Purpose**: Discrete value boundaries for the `symlog`-transformed return distribution $[V_{\min}, V_{\max}]$.
- **Training Impact**:
  - Returns falling outside $[support\_min, support\_max]$ are clipped to the boundary bins.
  - Because `SupportCriticHead` applies `symlog_transform` before binning, `[-150, 150]` covers raw cumulative returns well beyond `±20,000` while keeping the critic head compact.

#### `num_bins`
- **Type**: `int` | **Default**: `301`
- **Purpose**: Number of discrete probability atoms spanning $[support\_min, support\_max]$.
- **Training Impact**:
  - Bin resolution is $\Delta = (150 - (-150)) / (301 - 1) = 1.0$ `symlog` units per bin.
  - **Higher**: Finer value granularity at the cost of larger softmax layers.
  - **Lower**: Coarser value predictions; 301 bins provides unit-level `symlog` precision while halving critic head size compared to 601 bins.

#### `gamma`
- **Type**: `float` | **Default**: `0.99`
- **Purpose**: Temporal discount factor $\gamma \in [0, 1)$ for future rewards: $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$.
- **Training Impact**:
  - Effective planning horizon is $1 / (1 - \gamma) = 100$ simulation steps ($\approx 5.0$ seconds at $\Delta t = 0.05$s).
  - **Higher (e.g. 0.995)**: Encourages long-term forward velocity and energy preservation, but increases value variance.
  - **Lower (e.g. 0.95)**: Focuses myopically on immediate balance; may sacrifice forward progress to avoid falling in the next 10 steps.

#### `td_steps`
- **Type**: `int` | **Default**: `5`
- **Purpose**: Multi-step bootstrapping horizon for TD target values: $z_t = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V(s_{t+k})$.
- **Training Impact**:
  - **Higher (e.g. 10)**: Lower value bias, but higher variance due to environment stochasticity.
  - **Lower (e.g. 1)**: Classic TD(0); low variance but slow propagation of sparse distal rewards. 5 steps is the canonical sweet spot for continuous locomotion.

#### `target_tau`
- **Type**: `float` | **Default**: `0.01`
- **Purpose**: Polyak EMA update rate ($\theta^- \leftarrow (1 - \tau)\theta^- + \tau\theta$) for the Agent's target backbone, target critic head, and target SimSiam projector.

---

### 1.4 Dynamics & Self-Supervision Loss Weights

#### `l_dyn`
- **Type**: `int` | **Default**: `3`
- **Purpose**: Layer depth of the recurrent latent dynamics MLP $g_\theta(s_t, a_t) \to (\hat{s}_{t+1}, \hat{r}_t)$.
- **Training Impact**:
  - Must accurately model nonlinear contact mechanics, ground collisions, and friction. 3 FiLM residual blocks provide expressive capacity without vanishing gradients across unroll steps.

#### `proj_dim` and `pred_dim`
- **Type**: `int` | **Default**: `proj_dim: 512`, `pred_dim: 256`
- **Purpose**: Dimensions of the non-linear projection and prediction heads used in the self-supervised representation consistency loss (SimSiam / BYOL architecture).
- **Training Impact**:
  - Prevents representation collapse (where all states map to a constant latent vector). The asymmetric predictor head ($pred\_dim < proj\_dim$) guarantees non-trivial latent representations.

#### `unroll_steps`
- **Type**: `int` | **Default**: `5`
- **Purpose**: Number of consecutive future latent steps unrolled during training from replay buffer trajectories.
- **Training Impact**:
  - Forces the dynamics model $g_\theta$ to remain accurate over multi-step trajectory rollouts, preventing compounding error during MCTS planning.

#### `lambda_rep`, `beta_mse`, `lambda_v`, `lambda_r`, `lambda_p`, `lambda_ent`
- **Type**: `float`
- **Defaults**:
  - `lambda_rep: 2.0`: Weight of the self-supervised latent representation consistency loss.
  - `beta_mse: 1.0`: Stop-gradient MSE weight inside the hybrid SimSiam cosine + MSE consistency loss.
  - `lambda_v: 0.5`: Weight of the value critic loss.
  - `lambda_r: 1.0`: Weight of the step reward prediction loss.
  - `lambda_p: 0.50`: Weight of the policy anchor loss against MCTS search targets (masked MSE to MCTS winning candidate when `actor_type: "deterministic"`, or cross-entropy log-prob when `"gmm"`, ramped by `critic_warmup_steps`).
  - `lambda_ent: 0.03`: Weight of the per-joint Gaussian standard-deviation floor/ceiling regularization (`std_floor` $\to$ `std_ceil`).
  - `lambda_peak_ent: 0.15` $\to$ `lambda_peak_ent_end: 0.02`: Decaying weight on the GMM categorical mixture-weight entropy $H(\pi)$ (with `peak_entropy_floor_ratio: 0.70` $\to$ `0.0`) to prevent premature mode collapse without forcing destructive per-joint variance.
  - `lambda_peak_sep: 0.25` $\to$ `lambda_peak_sep_end: 0.05` (`min_peak_separation: 0.40`): Decaying pairwise RMS separation penalty between GMM peak means $\tanh(\mu_i), \tanh(\mu_j)$ so modes represent genuinely distinct gaits/postures.
- **Training Impact**:
  - Balanced multi-task loss objective:
    $$\mathcal{L}_{\text{total}} = \lambda_p \mathcal{L}_{\text{policy}} + \lambda_v \mathcal{L}_{\text{value}} + \lambda_r \mathcal{L}_{\text{reward}} + \lambda_{\text{rep}} \mathcal{L}_{\text{rep}} - \lambda_{\text{ent}} \mathcal{L}_{\text{ent}} + \mathcal{L}_{\text{peak\_ent}} + \mathcal{L}_{\text{peak\_sep}} + c_{\text{reg}} \|\theta\|_2^2$$

#### `c_reg`
- **Type**: `float` | **Default**: `0.0001` ($10^{-4}$)
- **Purpose**: $L_2$ weight decay regularization penalty applied across network parameters.

---

### 1.5 MCTS Tree Search & Exploration Annealing

Exploration in LA-MCTS is dynamically annealed using the environment's **rolling 20-episode raw return** over `0 -> exploration_return_ref` (`3000.0`), via progress $\alpha_{\text{explore}} = \text{clip}(\text{current\_return} / 3000.0, 0.0, 1.0)$.

#### `n_sim` and `n_sim_eval`
- **Type**: `int` | **Default**: `n_sim: 32`, `n_sim_eval: 64`
- **Purpose**: Number of MCTS simulation trajectories executed per environment step during training (`n_sim`) and evaluation (`n_sim_eval`).
- **Training Impact**:
  - **Training (32)**: `4` sequential PUCT waves (`batch_leaf_eval = 8`) per step for multi-step tree depth without hurting throughput.
  - **Eval (64)**: Used when evaluating `"gmm"` peak centers (`"deterministic"` mode evaluates `DeterministicActorHead` noise-free in O(1)).

#### `k_act`
- **Type**: `int` | **Default**: `8`
- **Purpose**: Number of candidate continuous actions per tree node (anchoring the `num_peaks` deterministic peak centers alongside exploratory Gaussian samples during training, and evaluating exact noise-free mode centers during `deterministic=True` evaluation).

#### `c_1` and `c_2`
- **Type**: `float` | **Default**: `c_1: 1.50`, `c_2: 19652.0`
- **Purpose**: Hyperparameters governing the polynomial upper confidence bound (pUCT) exploration bonus:
  $$U(s, a) = P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \cdot \left(c_1 + \log\left(\frac{\sum_b N(s, b) + c_2 + 1}{c_2}\right)\right)$$

#### `root_tau_beta` and `inner_tau_beta`
- **Type**: `float` | **Default**: `root_tau_beta: 1.0`, `inner_tau_beta: 0.8`
- **Purpose**: Softmax/proposal temperature applied when sampling continuous actions from the GMM actor at the root node (`1.0`) vs. interior tree nodes (`0.8`).

#### `root_dirichlet_alpha`, `root_exploration_fraction`, and `root_exploration_fraction_end`
- **Type**: `float` | **Default**: `root_dirichlet_alpha: 0.35`, `root_exploration_fraction: 0.40`, `root_exploration_fraction_end: 0.12`
- **Purpose**: Injects Dirichlet noise $\text{Dir}(\alpha)$ into the root node policy prior, with mixing fraction $\epsilon$ annealing linearly from `0.40` (at `0` return) down to `0.12` (at `3000` rolling raw return).

#### `tau_act_train`, `tau_act_train_end`, and `tau_act_eval`
- **Type**: `float` | **Default**: `tau_act_train: 1.0`, `tau_act_train_end: 0.15`, `tau_act_eval: 0.05`
- **Purpose**: Temperature used when selecting the executed environment action from root visit counts and normalized Q-values:
  - **Training (`tau_act_train` $\to$ `tau_act_train_end`)**: Starts at `1.0` for broad root action exploration at `0` return and anneals linearly to `0.15` at `3000` rolling raw return so high-speed gaits execute high-value MCTS branches reliably.
  - **Evaluation (`tau_act_eval: 0.05`)**: Near-deterministic greedy selection for benchmark evaluation.

#### `adaptive_exploration`, `min_policy_std_target`, `min_policy_std_target_end`, and `exploration_boost_max`
- **Type**: `bool` / `float` | **Default**: `adaptive_exploration: true`, `min_policy_std_target: 0.22`, `min_policy_std_target_end: 0.12`, `exploration_boost_max: 2.75`
- **Purpose**:
  - `min_policy_std_target` $\to$ `min_policy_std_target_end`: Sets the moderate target floor for each mode's per-joint Gaussian standard deviation (`0.22` at `0` return down to `0.12` at `3000` return), avoiding uncorrelated per-limb jitter while structured exploration is handled in latent/mode space.
  - When `adaptive_exploration: true` and the actor's current std drops below the annealed target, MCTS temperatures are boosted up to `exploration_boost_max` (`2.75`, softening as exploration progress approaches `1.0`).

#### Structured Exploration & Pre-Conditioning (`needle.md` Points A–D)
- **`latent_noise_std` / `latent_noise_std_end`** (`0.08` $\to$ `0.0`, Point A): Scale of temporally correlated OU perturbation $\delta_z$ added to the shared root latent representation (`min_max_normalize(z_cum + delta_z)`) seen by `MorphologicalLimbHead` (while Critic and dynamics keep clean $z$), so all limbs explore coherent whole-body intentions simultaneously.
- **`mode_hold_steps` / `mode_commitment_prob`** (`8` $\to$ `1` steps / `0.70` $\to$ `0.0`, Point B): Holds the active global latent intention $\delta_z$ and biases MCTS root candidate sampling toward the committed GMM peak for `mode_hold_steps`, letting MuJoCo physics accumulate torso momentum across steps.
- **`lambda_peak_ent` / `lambda_peak_ent_end` / `peak_entropy_floor_ratio`** (`0.15` $\to$ `0.02`, floor `0.70` $\to$ `0.0` of $\log M$, Point C): Decaying categorical entropy regularization and entropy floor on the GMM mixture weights $H(\pi)$ across all unroll steps, keeping distribution modes alive without inflating per-joint Gaussian variance.
- **`critic_warmup_steps`** (`5000`, Point D): Transitions over which Critic and World Model train at full weight while actor policy cross-entropy (`lambda_p`) and MCTS Q-advantage weight ramp from `0.15` to `1.0`.

---

### 1.6 Replay Buffer & MuZero Reanalyze

#### `lr`
- **Type**: `float` | **Default**: `0.00018` ($1.8 \times 10^{-4}$)
- **Purpose**: AdamW optimizer learning rate for the Agent networks (Transformer representation, dynamics, value critic, and policy prior).

#### `batch_size`
- **Type**: `int` | **Default**: `128`
- **Purpose**: Minibatch size sampled from the replay buffer per gradient update step.

#### `buffer_capacity`
- **Type**: `int` | **Default**: `50000`
- **Purpose**: Maximum number of step transitions retained in the trajectory replay buffer (~50 full 1,000-step episodes).

#### `train_every_steps`
- **Type**: `int` | **Default**: `50`
- **Purpose**: Frequency of agent gradient descent updates during episode execution (1 update every 50 environment steps, plus 1 final update at episode end).

#### `reanalyze_ratio`, `reanalyze_every`, `reanalyze_states`
- **Type**: `float` / `int` | **Default**: `reanalyze_ratio: 0.25`, `reanalyze_every: 4`, `reanalyze_states: 4`
- **Purpose**: MuZero Reanalyze schedule. Every `4` episodes, `4` historical states in the replay buffer have their MCTS policy and value targets refreshed using the latest networks.

---

## 2. Balanced Reinforcement Learning (`brl`)

### 2.1 Perturbation & Knapsack Parameter Space

#### `num_params`
- **Type**: `int` | **Default**: `18`
- **Purpose**: Dimensionality of the environmental and perceptual tweak vector $w \in [0, 1000]^{18}$ (12 physics/morphology parameters $w_1..w_{12}$ and 6 perception distortion parameters $w_{13}..w_{18}$).

#### `param_max`
- **Type**: `float` | **Default**: `1000.0`
- **Purpose**: Maximum allowable intensity on each normalized perturbation channel ($w_i \in [0, 1000]$).

#### `budget_max`
- **Type**: `float` | **Default**: `4600.0`
- **Purpose**: Maximum environmental budget $B_{\max}$ that the Balancer can allocate to the Tweaker.

#### `cost_min` and `cost_max`
- **Type**: `float` | **Default**: `cost_min: 0.1`, `cost_max: 10.0`
- **Purpose**: Cost bounds for individual tweak dimensions in the continuous knapsack problem.

#### `tau_s`
- **Type**: `float` | **Default**: `500.0`
- **Purpose**: Temperature scaling parameter for the continuous dominance score sigmoid $\sigma((R - R_{\text{ref}}) / \tau_s)$.

#### `tweaker_lr`, `beta_score`, `eta_viol`, `beta_ent`, and `beta_ent_end`
- **Type**: `float`
- **Defaults**:
  - `tweaker_lr: 0.0003`: Learning rate for the Tweaker's PPO actor-critic network.
  - `beta_score: 1.0`: Dominance score penalty weight in the Tweaker's reward.
  - `eta_viol: 2.0`: Pre-projection budget excess penalty factor.
  - `beta_ent: 0.01` $\to$ `beta_ent_end: 0.002`: Entropy bonus coefficient for the Tweaker's Beta distribution policy, annealing linearly from `0.01` to `0.002` as the agent's rolling 20 raw return progresses from `0` to `3000`.

---

### 2.2 Equilibrium, Win Criteria & ZPD Tracking

#### `target_win_rate`
- **Type**: `float` | **Default**: `0.50`
- **Purpose**: Target equilibrium win rate $\rho^* = 0.50$ for the Zone of Proximal Development (ZPD).

#### `ema_beta_w`
- **Type**: `float` | **Default**: `0.05`
- **Purpose**: Exponential moving average smoothing weight for the agent's tracked win rate ($\approx 20$-episode window).

#### `initial_r_ref` and `benchmark_r_ref`
- **Type**: `float` | **Default**: `initial_r_ref: -900.0`, `benchmark_r_ref: 6169.54`
- **Purpose**: Initial and asymptotic reference returns $R_{\text{ref}}$ used by `WinEvaluator` (calibrated at the end of warmup to the median of the last 20 warmup episode returns).

#### `window_k_w`
- **Type**: `int` | **Default**: `20`
- **Purpose**: Sliding window size (in episodes) used by the Balancer to track return statistics and failure correlations.

#### `sigma_zpd`
- **Type**: `float` | **Default**: `0.15`
- **Purpose**: Standard deviation of the ZPD Gaussian gate centered at `target_win_rate = 0.50`.

---

### 2.3 Balancer Meta-Controller (DDPG/TD3)

#### `balancer_lr` and `balancer_tau`
- **Type**: `float` | **Default**: `balancer_lr: 0.0001`, `balancer_tau: 0.005`
- **Purpose**: Learning rate and Polyak target smoothing factor for the Balancer's actor and twin Q-critic networks.

#### `balancer_noise_std` and `balancer_noise_std_end`
- **Type**: `float` | **Default**: `balancer_noise_std: 0.05`, `balancer_noise_std_end: 0.01`
- **Purpose**: Gaussian exploration noise std added to the Balancer's normalized constraint actions, annealing linearly from `0.05` to `0.01` as the agent's rolling 20 raw return progresses from `0` to `3000`.

#### `alpha_eq`, `alpha_inst`, `alpha_prog`, `lambda_entropy`
- **Type**: `float` | **Default**: `alpha_eq: 10.0`, `alpha_inst: 2.0`, `alpha_prog: 5.0`, `lambda_entropy: 0.5`
- **Purpose**: Reward weights in the Balancer's objective for win-rate equilibrium (`alpha_eq`), per-episode dominance stability (`alpha_inst`), difficulty progression inside the ZPD (`alpha_prog`), and budget allocation Shannon entropy (`lambda_entropy`).

---

### 2.4 Curriculum Schedule & Warmup

#### `warmup_episodes`, `brl_episodes`, and `total_episodes`
- **Type**: `int` | **Default**: `warmup_episodes: 150`, `brl_episodes: 10000`, `total_episodes: 10000`
- **Purpose**:
  - Episodes `1 .. 150`: Extended warmup phase with mild, fixed constraints (`warmup_limit`, `warmup_budget`) so Network 2 (`MorphologicalLimbHead`) and the Critic pre-condition over diverse nominal locomotion trajectories before co-evolutionary BRL ramps up difficulty.
  - Episodes `151 .. 10000`: Active co-evolutionary BRL phase.

#### `warmup_limit` and `warmup_budget`
- **Type**: `float` | **Default**: `warmup_limit: 30.0`, `warmup_budget: 90.0`
- **Purpose**: Per-parameter limit ($L_i = 30.0$) and total budget ($B_e = 90.0$) applied during the warmup phase.

---

## 3. Ant Morphology Environment (`ant_morph`)

#### `env_id`, `use_contact_forces`, `max_episode_steps`, `xml_rebuild_threshold`, `healthy_z_range`, `max_action_capacity`, `default_num_legs`
- **Defaults**:
  - `env_id: "Ant-v5"`
  - `use_contact_forces: false`
  - `max_episode_steps: 1000`
  - `xml_rebuild_threshold: 1.0`
  - `healthy_z_range: [0.2, 1.0]`
  - `max_action_capacity: 12` (supports 3-, 4-, and 6-legged morphologies)
  - `default_num_legs: 4`

#### `action_ema_alpha` and `action_smoothness_coef`
- **Type**: `float` | **Default**: `action_ema_alpha: 0.35`, `action_smoothness_coef: 0.05`
- **Purpose**:
  - `action_ema_alpha`: Initial exponential moving average action filter weight ($a_t^{\text{env}} = \alpha_{\text{ema}} a_t + (1 - \alpha_{\text{ema}}) a_{t-1}^{\text{env}}$). Interpolates linearly from `action_ema_alpha` to `1.0` (unfiltered agility) as `current_return` progresses from `0` to `torque_return_ref` (`1000.0`).
  - `action_smoothness_coef`: Penalty weight on squared action changes $\|a_t - a_{t-1}\|_2^2$. Scaled by $(1 - \alpha_{\text{torque}})$ so the smoothness penalty vanishes completely once `current_return` reaches `torque_return_ref` (`1000.0`).

#### `torque_return_ref` and `exploration_return_ref`
- **Type**: `float` | **Default**: `torque_return_ref: 1000.0`, `exploration_return_ref: 3000.0`
- **Purpose**: Reference thresholds for the **rolling 20-episode raw environment return** curriculum (`self._recent_ep_returns`, recorded across full 1,000-step healthy episodes with negative returns clamped to `0.0`):
  - **`torque_return_ref: 1000.0`**: Scales $\alpha_{\text{torque}} = \text{clip}(\text{current\_return} / 1000.0, 0.0, 1.0)$ from `0` to `1000` return, reducing the smoothness penalty to `0` and opening `action_ema_alpha` to `1.0`.
  - **`exploration_return_ref: 3000.0`**: Scales $\alpha_{\text{explore}} = \text{clip}(\text{current\_return} / 3000.0, 0.0, 1.0)$ from `0` to `3000` return, annealing MCTS action temperature (`tau_act_train`), Dirichlet noise (`root_exploration_fraction`), policy std target (`min_policy_std_target`), Balancer noise, and Tweaker entropy bonus.

#### `vel_shift_factor`
- **Type**: `float` | **Default**: `3000.0`
- **Purpose**: Per-step smoothing divisor governing how quickly `current_return` shifts toward the rolling 20-episode mean `target_return` ($\Delta = \text{target\_return} / \text{vel\_shift\_factor}$ per environment step, taking ~3 full episodes to traverse a shift).

#### `action_smoothness_end_step`, `benchmark_vel_ref`, `vel_std_threshold`, `upside_down_time_limit`, `upside_down_penalty`
- **Defaults**: `action_smoothness_end_step: 500000`, `benchmark_vel_ref: 0.20`, `vel_std_threshold: 0.20`, `upside_down_time_limit: 0.0`, `upside_down_penalty: 0.0`
- **Purpose**: Kept for backward compatibility; upside-down penalties/terminations are disabled (`0.0`) so only native `Ant-v5` `healthy_z_range` terminates episodes, and curriculum progression is driven directly by the rolling 20 raw return (`torque_return_ref` and `exploration_return_ref`).

---

## 4. Run Execution & Infrastructure (`run`)

#### `seeds` and `seed_offset`
- **Type**: `int` | **Default**: `seeds: 10`, `seed_offset: 42`
- **Purpose**: Number of independent training seeds and starting seed offset (`42 .. 51`).

#### `episodes` and `max_steps`
- **Type**: `int` | **Default**: `0` (automatic: uses `brl.total_episodes = 10000` and `ant_morph.max_episode_steps = 1000`).

#### `device`
- **Type**: `str` | **Default**: `"cuda"`
- **Purpose**: PyTorch compute device (`"cuda"` or `"cpu"`).

#### `eval_interval` and `checkpoint_interval`
- **Type**: `int` | **Default**: `eval_interval: 25`, `checkpoint_interval: 500`
- **Purpose**: Runs the 5-scenario benchmark evaluation suite every `25` episodes and saves permanent milestone checkpoints every `500` episodes (in addition to `latest_checkpoint.pt` and `best_model.pt`).

#### `parallel_seeds` and `gpu_mem_fraction`
- **Type**: `int` / `float` | **Default**: `parallel_seeds: 3`, `gpu_mem_fraction: 0.0`
- **Purpose**: Number of seeds trained concurrently in parallel worker processes (`3`) and optional per-process CUDA memory cap (`0.0` = dynamic).
