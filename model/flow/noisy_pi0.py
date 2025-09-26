import torch
import torch.nn as nn
from typing import Tuple
from .mlp_flow import ExploreNoiseNet
from openpi_Azuma413.src.openpi.models_pytorch.pi0_pytorch import PI0Pytorch
import openpi_Azuma413.src.openpi.models.gemma as _gemma
import numpy as np
import logging
log = logging.getLogger(__name__)

class NoisyPi0(nn.Module):
    """
    PI0PytorchとExploreNoiseNetをラップし，ReinFlowのアクターとして機能させるクラス
    """
    def __init__(
        self,
        policy: PI0Pytorch,
        denoising_steps: int,
        learn_explore_noise_from: int,
        inital_noise_scheduler_type: str,
        min_logprob_denoising_std: float,
        max_logprob_denoising_std: float,
        learn_explore_time_embedding: bool,
        time_dim_explore: int,
        use_time_independent_noise: bool,
        device,
        noise_hidden_dims=None,
        activation_type='Tanh',
        pooling_mode="cls",
    ):
        super().__init__()
        self.device = device
        self.policy = policy.to(self.device)
        self.denoising_steps = denoising_steps
        self.learn_explore_noise_from = learn_explore_noise_from
        self.initial_noise_scheduler_type = inital_noise_scheduler_type
        if min_logprob_denoising_std > max_logprob_denoising_std:
            raise ValueError(f"min_logprob_denoising_std must not exceed max_logprob_denoising_std, but received min_logprob_denoising_std={min_logprob_denoising_std} > max_logprob_denoising_std={max_logprob_denoising_std}. Revise your configuration file!")
        self.min_logprob_denoising_std = min_logprob_denoising_std
        self.max_logprob_denoising_std = max_logprob_denoising_std
        self.learn_explore_time_embedding = learn_explore_time_embedding
        self.pi0_config = self.policy.config
        self.set_logprob_noise_levels()
        self.noise_hidden_dims=noise_hidden_dims
        self.use_time_independent_noise = use_time_independent_noise
        self.time_dim_explore =time_dim_explore
        self.noise_activation_type=activation_type
        self.act_dim_total = self.pi0_config.action_horizon * 32
        self.init_exploration_noise_net()
        self.pooling_mode = pooling_mode

    def init_exploration_noise_net(self):
        cond_enc_dim = _gemma.get_config(self.pi0_config.paligemma_variant).width
        time_emb_dim = _gemma.get_config(self.pi0_config.action_expert_variant).width
        if self.use_time_independent_noise:
            noise_input_dim = cond_enc_dim
            if not self.noise_hidden_dims:
                self.noise_hidden_dims = [16]
        else:
            if self.learn_explore_time_embedding:
                noise_input_dim = self.time_dim_explore + cond_enc_dim
                self.time_embedding_explore = nn.Embedding(
                    num_embeddings=self.denoising_steps,
                    embedding_dim=self.time_dim_explore,
                    device=self.device
                )
            else:
                noise_input_dim = time_emb_dim + cond_enc_dim
                if not self.noise_hidden_dims:
                    self.noise_hidden_dims = [int(np.sqrt(noise_input_dim**2 + self.act_dim_total**2))]
        self.explore_noise_net=ExploreNoiseNet(
            in_dim=noise_input_dim,
            out_dim=self.act_dim_total,
            logprob_denoising_std_range=[self.min_logprob_denoising_std, self.max_logprob_denoising_std],
            device=self.device,
            hidden_dims=self.noise_hidden_dims,
            activation_type=self.noise_activation_type
        )

    def forward(
        self,
        action,
        time,
        cond,
        learn_exploration_noise=False,
        step=-1,
        verbose=False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = action.shape[0]
        vel, time_emb, prefix_embs = self.policy.get_velocity_and_embedding(
            observation=cond,
            actions=action,
            time=time,
        )
        if self.pooling_mode == "cls":
            # prefix_embsの最初のトークンを取得
            cond_emb = prefix_embs[:, 0, :]
        elif self.pooling_mode == "mean":
            # prefix_embsの平均を取得
            cond_emb = prefix_embs.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
        # noise head (for exploration). allow gradient flow.
        if self.initial_noise_scheduler_type=='const' or step < self.learn_explore_noise_from:
            noise_std = self.logprob_noise_levels[:, step].repeat(B,1)
        else:
            if self.use_time_independent_noise:
                noise_feature = cond_emb
            else:
                if self.learn_explore_time_embedding:
                    step_ts = torch.tensor(step, device = self.device).repeat(B)
                    time_emb_explore = self.time_embedding_explore(step_ts)
                    noise_feature = torch.cat([time_emb_explore, cond_emb], dim=-1)
                else:
                    noise_feature = torch.cat([time_emb.detach(), cond_emb], dim=-1)
            noise_std = self.explore_noise_net.forward(noise_feature=noise_feature)
            if verbose:
                log.info(f"step={step}, learnable noise = {noise_std.mean()}")
        if verbose:
            log.info(f"step={step}, set to learn from {self.learn_explore_noise_from}, will learn exploration noise ? {step >= self.learn_explore_noise_from}, noise_std={noise_std.mean()}require_grad={noise_std.requires_grad}")
        return vel, noise_std if learn_exploration_noise else noise_std.detach()

    @torch.no_grad()
    def stochastic_interpolate(self,t):
        valid_noise_schedulers=['vp', 'lin', 'const', 'const_schedule_itr', 'learn_decay']
        if self.initial_noise_scheduler_type == 'vp':
            a = 0.2 #2.0
            std = torch.sqrt(a * t * (1 - t))
        elif self.initial_noise_scheduler_type == 'lin':
            k=0.1
            b=0.0
            std = k*t+b
        elif self.initial_noise_scheduler_type == 'const' or 'const_schedule_itr':
            std = torch.ones_like(t) * self.min_logprob_denoising_std
        else:
            raise ValueError(f"Invalid noise scheduler type {self.initial_noise_scheduler_type}, must be in the following: {valid_noise_schedulers}")
        return std

    @torch.no_grad()
    def set_logprob_noise_levels(self, force_level=None, verbose=False):
        '''
        create noise std for logrporbability calcualion.
        generate a tensor `self.logprob_noise_levels` of shape `[1, self.denoising_steps,  self.policy.horizion_steps x self.policy.act_dim]`
        '''
        self.logprob_noise_levels = torch.zeros(self.denoising_steps, device=self.device, requires_grad=False)
        steps = torch.linspace(0, 1-1 /self.denoising_steps, self.denoising_steps, device=self.device)
        for i, t in enumerate(steps):
            if force_level:
                self.logprob_noise_levels[i] = torch.tensor(force_level, device=self.device)
            else:
                self.logprob_noise_levels[i] = self.stochastic_interpolate(t)
        self.logprob_noise_levels = self.logprob_noise_levels.clamp(min=self.min_logprob_denoising_std, max=self.max_logprob_denoising_std)
        self.logprob_noise_levels = self.logprob_noise_levels.unsqueeze(0).unsqueeze(-1).repeat(1, 1, self.pi0_config.action_horizon *  32)
        if verbose:
            log.info(f"Set logprob noise levels. self.logprob_noise_levels={self.logprob_noise_levels}")