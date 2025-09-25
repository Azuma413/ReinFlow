import torch
from torch import nn
import copy
import torch.nn.functional as F
from torch import Tensor
import logging
log = logging.getLogger(__name__)
from collections import namedtuple
from typing import Tuple
from torch.distributions.normal import Normal
from model.flow.noisy_pi0 import NoisyPi0
from openpi_Azuma413.src.openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi_Azuma413.src.openpi.models.pi0_config import Pi0Config

Sample = namedtuple("Sample", "trajectories chains")

class PPOPi0(nn.Module):
    def __init__(
        self,
        device,
        pi0_policy_path,
        pi0_config,
        critic,
        act_dim,
        horizon_steps,
        act_min,
        act_max,
        obs_dim,
        cond_steps,
        noise_scheduler_type,
        inference_steps,
        ft_denoising_steps,
        randn_clip_value,
        min_sampling_denoising_std,
        min_logprob_denoising_std,
        logprob_min,
        logprob_max,
        clip_ploss_coef,
        clip_ploss_coef_base,
        clip_ploss_coef_rate,
        clip_vloss_coef,
        denoised_clip_value,
        max_logprob_denoising_std,
        time_dim_explore,
        learn_explore_time_embedding,
        use_time_independent_noise,
        noise_hidden_dims,
        logprob_debug_sample,
        logprob_debug_recalculate,
        explore_net_activation_type,
        pooling_mode="cls"
    ):
        super().__init__()
        self.device = device
        self.inference_steps = inference_steps
        self.ft_denoising_steps = ft_denoising_steps
        self.action_dim = act_dim  # This is the robot action dim (7)
        self.pi0_action_dim = 32   # PI0's native action dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim  # Robot actions
        self.pi0_act_dim_total = self.horizon_steps * self.pi0_action_dim  # PI0 actions
        self.act_min = act_min
        self.act_max = act_max
        
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        
        self.noise_scheduler_type = noise_scheduler_type
        self.randn_clip_value = randn_clip_value
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        self.max_logprob_denoising_std = max_logprob_denoising_std
        
        self.logprob_min = logprob_min
        self.logprob_max = logprob_max
        
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        
        self.denoised_clip_value = denoised_clip_value
        self.logprob_debug_sample = logprob_debug_sample
        self.logprob_debug_recalculate = logprob_debug_recalculate
        # Load PI0 policy
        model_cfg = Pi0Config(
            dtype=pi0_config.pytorch_training_precision,
            action_dim=pi0_config.model.action_dim,
            action_horizon=pi0_config.model.action_horizon,
            max_token_len=pi0_config.model.max_token_len,
            paligemma_variant=getattr(pi0_config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(pi0_config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(pi0_config.model, "pi05", False),
        )
        self.pi0_policy = PI0Pytorch(model_cfg)
        if pi0_policy_path:
            log.info(f"Loading PI0 policy from {pi0_policy_path}")
            checkpoint = torch.load(pi0_policy_path, map_location=self.device)
            self.pi0_policy.load_state_dict(checkpoint["model"])
            log.info("Loaded PI0 policy successfully")
        # Freeze original policy for reference
        self.actor_old = copy.deepcopy(self.pi0_policy)
        for param in self.actor_old.parameters():
            param.requires_grad = False
        self.actor_old.to(self.device)
        # Create fine-tuning policy with NoisyPi0 wrapper
        self.actor_ft = NoisyPi0(
            policy=self.pi0_policy,
            denoising_steps=inference_steps,
            learn_explore_noise_from=inference_steps - ft_denoising_steps,
            inital_noise_scheduler_type=noise_scheduler_type,
            min_logprob_denoising_std=min_logprob_denoising_std,
            max_logprob_denoising_std=max_logprob_denoising_std,
            learn_explore_time_embedding=learn_explore_time_embedding,
            time_dim_explore=time_dim_explore,
            use_time_independent_noise=use_time_independent_noise,
            device=device,
            noise_hidden_dims=noise_hidden_dims,
            activation_type=explore_net_activation_type,
            pooling_mode=pooling_mode
        )
        
        self.critic = critic.to(self.device)
        
        self.report_network_params()

    def report_network_params(self):
        logging.info(
            f"Number of network parameters: Total: {sum(p.numel() for p in self.parameters())/1e6} M. "
            f"PI0 Policy: {sum(p.numel() for p in self.pi0_policy.parameters())/1e6} M. "
            f"Actor (finetune): {sum(p.numel() for p in self.actor_ft.parameters())/1e6} M. "
            f"Critic: {sum(p.numel() for p in self.critic.parameters())/1e6} M"
        )

    @torch.no_grad()
    def sample_first_point(self, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample initial point for PI0 (32-dimensional actions)"""
        dist = Normal(torch.zeros(B, self.horizon_steps * self.pi0_action_dim), 1.0)
        xt = dist.sample()
        log_prob = dist.log_prob(xt).sum(-1).to(self.device)
        xt = xt.reshape(B, self.horizon_steps, self.pi0_action_dim).to(self.device)
        return xt, log_prob

    def convert_pi0_to_robot_actions(self, pi0_actions: torch.Tensor) -> torch.Tensor:
        """Convert PI0's 32-dimensional actions to robot's 7-dimensional actions"""
        # Extract first 7 dimensions from PI0 actions
        robot_actions = pi0_actions[:, :, :self.action_dim]  # [B, horizon_steps, 7]
        return robot_actions

    def convert_robot_to_pi0_actions(self, robot_actions: torch.Tensor) -> torch.Tensor:
        """Convert robot's 7-dimensional actions to PI0's 32-dimensional actions"""
        B, T, _ = robot_actions.shape
        pi0_actions = torch.zeros(B, T, self.pi0_action_dim, device=robot_actions.device)
        pi0_actions[:, :, :self.action_dim] = robot_actions
        return pi0_actions

    def get_logprobs(self, 
                     cond: dict, 
                     x_chain: Tensor, 
                     get_entropy=False, 
                     normalize_denoising_horizon=False, 
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     debug=True,
                     account_for_initial_stochasticity=False,
                     get_chains_stds=True
                     ):
        """
        Get log probabilities for PI0 actions
        x_chain: [B, inference_steps+1, horizon_steps, pi0_action_dim=32]
        """
        logprob = 0.0
        joint_entropy = 0.0
        entropy_rate_est = 0.0
        logprob_steps = 0
        
        B = x_chain.shape[0]
        chains_prev = x_chain[:, :-1, :, :].flatten(-2, -1)  # [B, inference_steps, horizon*32]
        chains_next = x_chain[:, 1:, :, :].flatten(-2, -1)   # [B, inference_steps, horizon*32]
        chains_stds = torch.zeros_like(chains_prev, device=self.device)
        
        # Initial probability (for 32-dim actions)
        init_dist = Normal(torch.zeros(B, self.horizon_steps * self.pi0_action_dim, device=self.device), 1.0)
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)
        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)
        if account_for_initial_stochasticity:
            logprob += logprob_init
            if get_entropy:
                joint_entropy += entropy_init
            logprob_steps += 1
        
        # Transition probabilities
        chains_vel = torch.zeros_like(chains_prev, device=self.device)
        
        dt = 1.0 / self.inference_steps
        steps = torch.linspace(0, 1-dt, self.inference_steps).repeat(B, 1).to(self.device)
        
        for i in range(self.inference_steps):
            t = steps[:, i]
            xt = x_chain[:, i]  # [B, horizon_steps, 32]
            vt, nt = self.actor_ft.forward(xt, t, cond, True, i)  # Velocity and noise std
            chains_vel[:, i] = vt.flatten(-2, -1)
            chains_stds[:, i] = nt
            logprob_steps += 1
            
        chains_mean = (chains_prev + chains_vel * dt)
        if clip_intermediate_actions:
            chains_mean = chains_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)
        
        # Transition distribution
        chains_dist = Normal(chains_mean, chains_stds)
        
        # Log probability and entropy of transitions
        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)  # [B, inference_steps]
        if get_entropy:
            entropy_trans = chains_dist.entropy().sum(-1)
        
        # Log probability of whole Markov chain
        logprob += logprob_trans.sum(-1)
        if self.logprob_debug_recalculate:
            log.info(f"logprob_init={logprob_init.mean().item()}, logprob_trans={logprob_trans.mean().item()}")
        
        # Entropy rate estimate
        if get_entropy:
            joint_entropy += entropy_trans.sum(-1)
        
        if get_entropy:
            entropy_rate_est = joint_entropy / logprob_steps
            
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
            
        if normalize_act_space_dimension:
            logprob = logprob / self.pi0_act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est / self.pi0_act_dim_total
        
        if verbose_entropy_stats and get_entropy:
            log.info(f"entropy_rate_est={entropy_rate_est.shape} Entropy Percentiles: "
                    f"10%={entropy_rate_est.quantile(0.1):.2f}, 50%={entropy_rate_est.median():.2f}, "
                    f"90%={entropy_rate_est.quantile(0.9):.2f}")
        
        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, chains_stds.mean()
            return logprob, entropy_rate_est
        else:
            if get_chains_stds:
                return logprob, chains_stds.mean()
            return logprob

    @torch.no_grad()
    def get_actions(self, 
                    cond: dict, 
                    eval_mode: bool, 
                    save_chains=False, 
                    normalize_denoising_horizon=False, 
                    normalize_act_space_dimension=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True,
                    ret_logprob=True
                    ):
        """
        Sample actions using PI0, then convert to robot actions
        Returns robot actions (7-dim) and optionally PI0 action chains (32-dim)
        """
        B = cond["rgb"].shape[0]  # Use rgb instead of state for PI0
        dt = (1/self.inference_steps) * torch.ones(B, self.horizon_steps, self.pi0_action_dim, device=self.device)
        steps = torch.linspace(0, 1-1/self.inference_steps, self.inference_steps).repeat(B, 1).to(self.device)
        
        if save_chains:
            x_chain = torch.zeros((B, self.inference_steps+1, self.horizon_steps, self.pi0_action_dim), device=self.device)
        if ret_logprob:
            log_prob = 0.0
            log_prob_steps = 0
            if self.logprob_debug_sample:
                log_prob_list = []
        
        # Sample first point (32-dim)
        xt, log_prob_init = self.sample_first_point(B)
        if ret_logprob and account_for_initial_stochasticity:
            log_prob += log_prob_init
            log_prob_steps += 1
            if self.logprob_debug_sample:
                log_prob_list.append(log_prob_init.mean().item())
        
        if save_chains:
            x_chain[:, 0] = xt
        
        for i in range(self.inference_steps):
            t = steps[:, i]
            vt, nt = self.actor_ft.forward(xt, t, cond, learn_exploration_noise=False, step=i)
            xt += vt * dt
            if clip_intermediate_actions:
                xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)
            
            # Add noise during training
            std = nt.unsqueeze(-1).reshape(xt.shape)
            std = torch.clamp(std, min=self.min_sampling_denoising_std)
            dist = Normal(xt, std)
            if not eval_mode:
                xt = dist.sample().clamp_(dist.loc - self.randn_clip_value * dist.scale,
                                         dist.loc + self.randn_clip_value * dist.scale).to(self.device)
            
            # Prevent last action overflow - convert to robot action space for clipping
            if i == self.inference_steps - 1:
                robot_actions = self.convert_pi0_to_robot_actions(xt)
                robot_actions = robot_actions.clamp_(self.act_min, self.act_max)
                xt[:, :, :self.action_dim] = robot_actions  # Update first 7 dims
            
            if ret_logprob:
                logprob_transition = dist.log_prob(xt).sum(dim=(-2, -1)).to(self.device)
                if self.logprob_debug_sample:
                    log_prob_list.append(logprob_transition.mean().item())
                log_prob += logprob_transition
                log_prob_steps += 1
                
            if save_chains:
                x_chain[:, i+1] = xt
        
        # Convert final actions from PI0 (32-dim) to robot (7-dim)
        robot_actions = self.convert_pi0_to_robot_actions(xt)
        
        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob = log_prob / log_prob_steps
            if normalize_act_space_dimension:
                log_prob = log_prob / self.pi0_act_dim_total
            if self.logprob_debug_sample:
                transform_logprob = torch.log(1-torch.tanh(x_chain[:, -1])**2+1e-7).sum(dim=(-2,-1)).mean().item()
                print(f"log_prob_list={log_prob_list}, transform={transform_logprob}")
        
        if ret_logprob:
            if save_chains:
                return (robot_actions, x_chain, log_prob)  # robot actions + PI0 chains
            return (robot_actions, log_prob)
        else:
            if save_chains:
                return (robot_actions, x_chain)
            return robot_actions

    def loss(self,
             obs,
             chains,  # PI0 action chains (32-dim)
             returns,
             oldvalues,
             advantages,
             oldlogprobs,
             use_bc_loss=False,
             bc_loss_type='W2',
             normalize_denoising_horizon=False,
             normalize_act_space_dimension=False,
             verbose=True,
             clip_intermediate_actions=True,
             account_for_initial_stochasticity=True
             ):
        """
        PPO loss for PI0
        chains: (B, K+1, Ta, 32)  # PI0 action chains
        """
        newlogprobs, entropy, noise_std = self.get_logprobs(obs, 
                                                            chains, 
                                                            get_entropy=True, 
                                                            normalize_denoising_horizon=normalize_denoising_horizon,
                                                            normalize_act_space_dimension=normalize_act_space_dimension, 
                                                            verbose_entropy_stats=verbose, 
                                                            clip_intermediate_actions=clip_intermediate_actions,
                                                            account_for_initial_stochasticity=account_for_initial_stochasticity)
        if verbose:
            log.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std of oldlogprobs={oldlogprobs.std():5.3f}")
            log.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std of newlogprobs={newlogprobs.std():5.3f}")
        
        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        
        if verbose:
            if oldlogprobs.min() < self.logprob_min: 
                log.info(f"WARNING: old logprobs too low, potential policy collapse detected")
            if newlogprobs.min() < self.logprob_min: 
                log.info(f"WARNING: new logprobs too low, potential policy collapse detected")
            if newlogprobs.max() > self.logprob_max: 
                log.info(f"WARNING: new logprobs too high")
            if oldlogprobs.max() > self.logprob_max: 
                log.info(f"WARNING: old logprobs too high")
        
        # Batch normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                advantage_stats = {
                    "mean": f"{advantages.mean().item():2.3f}",
                    "std": f"{advantages.std().item():2.3f}",
                    "max": f"{advantages.max().item():2.3f}",
                    "min": f"{advantages.min().item():2.3f}"
                }
                log.info(f"Advantage stats: {advantage_stats}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0, 1].item()
                log.info(f"Advantage-Reward Correlation: {corr:.2f}")
        
        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()
        
        # Get KL difference and whether value clipped
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef:
            v_clipped = torch.clamp(newvalues, oldvalues - self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 * torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                log.info(f"Value/Reward alignment: MSE={mse.item():.3f}")
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        if verbose:
            with torch.no_grad():
                log.info(f"Entropy Percentiles: 10%={entropy.quantile(0.1):.2f}, 50%={entropy.median():.2f}, 90%={entropy.quantile(0.9):.2f}")
        
        # BC loss (if enabled)
        bc_loss = 0.0
        if use_bc_loss:
            if bc_loss_type == 'W2':
                # Add Wasserstein divergence loss via action supervision
                z = torch.zeros((obs['rgb'].shape[0], self.horizon_steps, self.pi0_action_dim), device=self.device)
                # Note: This would require implementing sample_action for the old policy
                # For now, we skip BC loss for PI0
                log.warning("BC loss not implemented for PI0 - skipping")
                bc_loss = torch.tensor(0.0, device=self.device)
            else:
                raise NotImplementedError
        
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            bc_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            oldlogprobs.min(),
            oldlogprobs.max(),
            oldlogprobs.std(),
            newlogprobs.min(),
            newlogprobs.max(),
            newlogprobs.std(),
            noise_std.item(),
            newvalues.mean().item(),  # Q function
        )
