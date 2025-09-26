"""
PPO fine-tuning for PI0 policy on SimplerEnv.
"""
from tqdm import tqdm as tqdm
import torch
import logging
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
log = logging.getLogger(__name__)
from agent.finetune.reinflow.train_ppo_flow_img_agent import TrainPPOImgFlowAgent
from model.flow.ft_ppo.ppopi0 import PPOPi0
from agent.finetune.reinflow.buffer import PPOFlowImgBuffer, PPOFlowImgBufferGPU

@dataclass
class PI0ObservationBatch:
    """Batch of observations in PI0 format"""
    images: Dict[str, torch.Tensor]
    image_masks: Dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    state: torch.Tensor
    token_ar_mask: torch.Tensor
    token_loss_mask: torch.Tensor

class TrainPPOPi0Agent(TrainPPOImgFlowAgent):
    def __init__(self, cfg):
        # Initialize parent class but we'll override most methods
        super().__init__(cfg)
        # PI0 specific parameters
        self.pi0_config = cfg.model.pi0_config
        self.pi0_policy_path = cfg.model.get('pi0_policy_path', None)
        self.pi0_action_dim = 32  # PI0's native action dimension
        self.robot_action_dim = cfg.model.act_dim  # Robot action dimension (7)
        # Override some settings for PI0
        self.initial_ratio_error_threshold = 1e-4  # More relaxed for PI0
        # PI0 observation processing setup
        self.max_token_len = cfg.model.pi0_config.model.max_token_len
        self.robot_type = cfg.robot_type
        log.info(f"Initialized PI0 training agent with robot_action_dim={self.robot_action_dim}, pi0_action_dim={self.pi0_action_dim}")

    def convert_simplerenv_to_pi0_observation(self, simplerenv_obs: Dict) -> PI0ObservationBatch:
        """
        Convert SimplerEnv observation format to PI0 format
        SimplerEnv obs format:
        - "rgb": image data [B, C, H, W]
        - "robot_type": robot type
        - "state": robot state [B, 8] (x,y,z,qw,qx,qy,qz,gripper)
        - "scene_info": scene information
        PI0 obs format:
        - images: Dict[str, Tensor] - image data
        - image_masks: Dict[str, Tensor] - image masks
        - tokenized_prompt: Tensor - tokenized language prompts
        - tokenized_prompt_mask: Tensor - prompt masks
        - state: Tensor - state information (32-dim for PI0)
        """
        # print("rgb shape:", simplerenv_obs["rgb"].shape) # (1, 1, 480, 640, 3)
        # print("state shape:", simplerenv_obs.get("state", None).shape) # (1, 1, 8)
        batch_size = simplerenv_obs["rgb"].shape[0]
        # Process images
        rgb_images = simplerenv_obs["rgb"]  # [B, C, H, W]
        # Ensure images are in correct format and device
        if not isinstance(rgb_images, torch.Tensor):
            rgb_images = torch.from_numpy(rgb_images).float()
        if rgb_images.dim() == 4 and rgb_images.shape[3] in [1, 3, 4]:
            rgb_images = rgb_images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        rgb_images = rgb_images.to(self.device)
        # Create image dictionary (PI0 expects named images)
        images = {
            "base_0_rgb": rgb_images,
            "left_wrist_0_rgb": torch.zeros_like(rgb_images),
            "right_wrist_0_rgb": torch.zeros_like(rgb_images)
        }
        # Create image masks (all True since we have valid images)
        image_masks = {}
        # Get tokenized prompts from observation (already tokenized by the environment wrapper)
        tokenized_prompt = torch.from_numpy(simplerenv_obs["tokenized_prompt"]).to(self.device)
        tokenized_prompt_mask = torch.from_numpy(simplerenv_obs["tokenized_prompt_mask"]).to(self.device)
        # Ensure batch dimension
        if tokenized_prompt.dim() == 1:
            tokenized_prompt = tokenized_prompt.unsqueeze(0)
            tokenized_prompt_mask = tokenized_prompt_mask.unsqueeze(0)
        # Process robot state
        processed_state = simplerenv_obs.get("state", None)
        if processed_state is not None:
            if not isinstance(processed_state, torch.Tensor):
                processed_state = torch.from_numpy(processed_state).float()
            processed_state = processed_state.to(self.device)
            # Expand 8-dimensional processed state to 32-dimensional PI0 format
            pi0_state = self.expand_state_to_pi0_dimension(processed_state)
        else:
            # Fallback: create dummy state
            log.warning("No robot state found in observation, using dummy state")
            pi0_state = torch.zeros(batch_size, 32, dtype=torch.float32, device=self.device)
        return PI0ObservationBatch(
            images=images,
            image_masks=image_masks,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            state=pi0_state,
            token_ar_mask=None,
            token_loss_mask=None,
        )

    def expand_state_to_pi0_dimension(self, processed_state: torch.Tensor) -> torch.Tensor:
        """
        Expand processed 8-dimensional state to 32-dimensional PI0 format
        (環境側で前処理済みの8次元状態を32次元に拡張するだけ)
        Args:
            processed_state: [B, 8] tensor with preprocessed robot state
        Returns:
            pi0_state: [B, 32] tensor in PI0 format
        """
        batch_size = processed_state.shape[0]
        
        # Check for NaN/inf in input
        if torch.any(~torch.isfinite(processed_state)):
            log.warning(f"Warning: Invalid values in processed_state: {processed_state}")
            # Use safe default state
            return torch.zeros(batch_size, 32, dtype=torch.float32, device=self.device)
        
        try:
            # Simply pad the 8-dimensional processed state to 32 dimensions
            pi0_states = []
            for i in range(batch_size):
                state_8d = processed_state[i].cpu().numpy()  # [8] - already processed by environment
                
                # Check for valid input values
                if not np.all(np.isfinite(state_8d)):
                    log.warning(f"Invalid state_8d at batch {i}: {state_8d}")
                    pi0_states.append(np.zeros(32, dtype=np.float32))
                    continue
                
                # Pad to 32 dimensions (environment already did the robot-specific processing)
                pi0_state = np.concatenate([state_8d, np.zeros(32 - 8)])
                
                # Final check for NaN values
                if not np.all(np.isfinite(pi0_state)):
                    log.warning(f"Final pi0_state contains invalid values, using zeros")
                    pi0_state = np.zeros(32, dtype=np.float32)
                
                pi0_states.append(pi0_state)
            
            pi0_states = np.stack(pi0_states)
            return torch.from_numpy(pi0_states).float().to(self.device)
            
        except Exception as e:
            log.warning(f"Error expanding state to PI0 dimension: {e}, using zero state")
            return torch.zeros(batch_size, 32, dtype=torch.float32, device=self.device)

    def pi0_observation_to_dict(self, pi0_obs: PI0ObservationBatch) -> Dict:
        """Convert PI0ObservationBatch to dictionary format expected by model"""
        return {
            "images": pi0_obs.images,
            "image_masks": pi0_obs.image_masks,
            "tokenized_prompt": pi0_obs.tokenized_prompt,
            "tokenized_prompt_mask": pi0_obs.tokenized_prompt_mask,
            "state": pi0_obs.state
        }

    def init_buffer(self):
        """Initialize buffer for PI0 training"""
        log.info(f"self.buffer_device={self.buffer_device}")
        log_prob_cfg_dict = {
            'normalize_denoising_horizon': self.normalize_denoising_horizon,
            'normalize_act_space_dimension': self.normalize_act_space_dim, 
            'clip_intermediate_actions': self.clip_intermediate_actions,
            'account_for_initial_stochasticity': self.account_for_initial_stochasticity
        }
        # Set observation dimensions for PI0
        # We need to store both SimplerEnv obs and PI0 obs
        obs_dims = {
            "rgb": self.obs_dims["rgb"],  # Image observations from SimplerEnv
            "state": (8,),  # Robot state (x,y,z,qw,qx,qy,qz,gripper)
            # "robot_type": (1,),   # Placeholder for robot type
            "tokenized_prompt": (self.max_token_len,),  # Tokenized prompts
            "tokenized_prompt_mask": (self.max_token_len,),  # Token masks
        }
        if self.buffer_device == 'cpu':
            self.buffer = PPOFlowImgBuffer(
                n_steps=self.n_steps,
                n_envs=self.n_envs,
                n_ft_denoising_steps=self.inference_steps, 
                horizon_steps=self.horizon_steps,
                act_steps=self.act_steps,
                action_dim=self.pi0_action_dim,  # Use PI0 action dimension for chains
                n_cond_step=self.n_cond_step,
                obs_dim=obs_dims,
                save_full_observation=self.save_full_observations,
                furniture_sparse_reward=self.furniture_sparse_reward,
                best_reward_threshold_for_success=self.best_reward_threshold_for_success,
                reward_scale_running=self.reward_scale_running,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                reward_scale_const=self.reward_scale_const,
                aug=self.aug if hasattr(self, 'aug') else None,
                fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug,
                device=self.device,
                log_prob_cfg_dict=log_prob_cfg_dict
            )
        else:
            self.buffer = PPOFlowImgBufferGPU(
                n_steps=self.n_steps,
                n_envs=self.n_envs,
                n_ft_denoising_steps=self.inference_steps,
                horizon_steps=self.horizon_steps,
                act_steps=self.act_steps,
                action_dim=self.pi0_action_dim,  # Use PI0 action dimension for chains
                n_cond_step=self.n_cond_step,
                obs_dim=obs_dims,
                save_full_observation=self.save_full_observations,
                furniture_sparse_reward=self.furniture_sparse_reward,
                best_reward_threshold_for_success=self.best_reward_threshold_for_success,
                reward_scale_running=self.reward_scale_running,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                reward_scale_const=self.reward_scale_const,
                aug=self.aug if hasattr(self, 'aug') else None,
                fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug,
                device=self.device,
                log_prob_cfg_dict=log_prob_cfg_dict
            )
        log.info(f"created buffer: {self.buffer.__class__} on {self.buffer_device}")

    @torch.no_grad()
    def get_samples(self,
                    cond: dict,
                    ret_device='cpu',
                    save_chains=True,
                    normalize_denoising_horizon=False,
                    normalize_act_space_dimension=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True):
        """
        Get action samples from PI0 model
        Returns robot actions (7-dim) and PI0 action chains (32-dim)
        """
        if save_chains:
            robot_actions, pi0_chains = self.model.get_actions(
                cond, 
                eval_mode=self.eval_mode, 
                save_chains=save_chains, 
                normalize_denoising_horizon=normalize_denoising_horizon, 
                normalize_act_space_dimension=normalize_act_space_dimension, 
                clip_intermediate_actions=clip_intermediate_actions,
                account_for_initial_stochasticity=account_for_initial_stochasticity,
                ret_logprob=False
            )
            robot_actions_np = robot_actions.cpu().numpy() if ret_device == 'cpu' else robot_actions
            pi0_chains_out = pi0_chains.cpu().numpy() if ret_device == 'cpu' else pi0_chains
            return robot_actions_np, pi0_chains_out
        else:
            robot_actions = self.model.get_actions(
                cond, 
                eval_mode=self.eval_mode, 
                save_chains=save_chains, 
                normalize_denoising_horizon=normalize_denoising_horizon, 
                normalize_act_space_dimension=normalize_act_space_dimension, 
                clip_intermediate_actions=clip_intermediate_actions,
                account_for_initial_stochasticity=account_for_initial_stochasticity,
                ret_logprob=False
            )
            return robot_actions.cpu().numpy()

    def run(self):
        """Main training loop for PI0"""
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset()
        if self.resume:
            self.resume_training()
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env(buffer_device=self.buffer_device)
            self.prev_obs_venv["rgb"] = self.prev_obs_venv["rgb"].squeeze(1)
            self.prev_obs_venv["state"] = self.prev_obs_venv["state"].squeeze(1)
            self.prev_obs_venv["tokenized_prompt"] = self.prev_obs_venv["tokenized_prompt"].squeeze(1)
            self.prev_obs_venv["tokenized_prompt_mask"] = self.prev_obs_venv["tokenized_prompt_mask"].squeeze(1)
            self.buffer.update_full_obs()
            for step in tqdm(range(self.n_steps)) if self.verbose else range(self.n_steps):
                if not self.verbose and step % 100 == 0: 
                    print(f"Processed {step} of {self.n_steps}")
                with torch.no_grad():
                    # Convert SimplerEnv observation to PI0 format
                    pi0_obs_batch = self.convert_simplerenv_to_pi0_observation(self.prev_obs_venv)
                    # pi0_cond = self.pi0_observation_to_dict(pi0_obs_batch)
                    # Get actions (robot actions + PI0 chains)
                    robot_actions, pi0_chains = self.get_samples(
                        cond=pi0_obs_batch, # pi0_cond,
                        ret_device=self.buffer_device,
                        normalize_denoising_horizon=self.normalize_denoising_horizon,
                        normalize_act_space_dimension=self.normalize_act_space_dim,
                        clip_intermediate_actions=self.clip_intermediate_actions,
                        account_for_initial_stochasticity=self.account_for_initial_stochasticity
                    )
                # Apply multi-step action (use robot actions for environment)
                action_venv = robot_actions[:, :self.act_steps]  # [n_envs, act_steps, 7]
                
                # Critical: Final check for NaN/inf in actions before sending to environment
                if isinstance(action_venv, torch.Tensor):
                    if torch.any(~torch.isfinite(action_venv)):
                        log.error(f"NaN/inf detected in action_venv at step {step}: {action_venv}")
                        # Replace NaN/inf with zero actions
                        action_venv = torch.nan_to_num(action_venv, nan=0.0, posinf=1.0, neginf=-1.0)
                        # Clamp to valid range
                        action_venv = torch.clamp(action_venv, -1.0, 1.0)
                        log.info(f"Replaced with safe actions: {action_venv[0][0]}")
                    
                    action_venv_numpy = action_venv.cpu().numpy()
                else:
                    action_venv_numpy = action_venv
                
                # Additional check for numpy array
                if np.any(~np.isfinite(action_venv_numpy)):
                    log.error(f"NaN/inf detected in action_venv_numpy at step {step}: {action_venv_numpy}")
                    action_venv_numpy = np.nan_to_num(action_venv_numpy, nan=0.0, posinf=1.0, neginf=-1.0)
                    action_venv_numpy = np.clip(action_venv_numpy, -1.0, 1.0)
                    log.info(f"Replaced with safe numpy actions: {action_venv_numpy[0][0]}")
                
                print("action_venv: ", action_venv_numpy[0][0])
                
                # Validate action shape
                expected_shape = (self.n_envs, self.act_steps, self.robot_action_dim)
                if action_venv_numpy.shape != expected_shape:
                    log.error(f"Invalid action shape: {action_venv_numpy.shape}, expected: {expected_shape}")
                    # Create safe default actions
                    action_venv_numpy = np.zeros(expected_shape, dtype=np.float32)
                    log.info(f"Using zero actions due to shape mismatch")
                
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv_numpy)
                squeezed_obs_venv = {}
                squeezed_obs_venv["rgb"] = obs_venv["rgb"].squeeze(1)
                squeezed_obs_venv["state"] = obs_venv["state"].squeeze(1)
                squeezed_obs_venv["tokenized_prompt"] = obs_venv["tokenized_prompt"].squeeze(1)
                squeezed_obs_venv["tokenized_prompt_mask"] = obs_venv["tokenized_prompt_mask"].squeeze(1)
                # Store in buffer (store PI0 chains for training)
                self.buffer.add(step, self.prev_obs_venv, pi0_chains, reward_venv, terminated_venv, truncated_venv)
                self.prev_obs_venv = squeezed_obs_venv
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0
            self.buffer.summarize_episode_reward()
            if not self.eval_mode:
                # Update buffer with final observations
                self.buffer.update_img(obs_venv, self.model)
                self.agent_update(verbose=self.verbose)
            self.log()
            self.update_lr()
            self.adjust_finetune_schedule()
            self.save_model()
            self.itr += 1
            # Early stopping
            if self.use_early_stop and (self.buffer.success_rate < 0.05 or self.buffer.avg_episode_reward < 2.0):
                log.info(f"Your finetuning failed. success_rate={self.buffer.success_rate*100:.2f}% and avg_episode_reward={self.buffer.avg_episode_reward:.2f}")
                exit()
            self.clear_cache()
            self.inspect_memory()

    def agent_update(self, verbose=True):
        """Agent update for PI0 training"""
        clipfracs_list = []
        noise_std_list = []
        actor_norm = 0.0
        critic_norm = 0.0
        for update_epoch, batch_id, minibatch in self.minibatch_generator() if not self.repeat_samples else self.minibatch_generator_repeat():
            # Minibatch gradient descent
            self.model: PPOPi0
            # Convert observations to PI0 format for loss calculation
            obs_dict, pi0_chains, returns, oldvalues, advantages, oldlogprobs = minibatch
            # Convert SimplerEnv observations to PI0 format
            pi0_obs_batch = self.convert_simplerenv_to_pi0_observation(obs_dict)
            # pi0_obs_dict = self.pi0_observation_to_dict(pi0_obs_batch)
            pg_loss, entropy_loss, v_loss, bc_loss, \
            clipfrac, approx_kl, ratio, \
            oldlogprob_min, oldlogprob_max, oldlogprob_std, \
            newlogprob_min, newlogprob_max, newlogprob_std, \
            noise_std, newQ_values = self.model.loss(
                pi0_obs_batch,  # Use PI0 format observations
                pi0_chains,    # PI0 action chains (32-dim)
                returns,
                oldvalues,
                advantages,
                oldlogprobs,
                use_bc_loss=self.use_bc_loss,
                bc_loss_type=self.bc_loss_type,
                normalize_denoising_horizon=self.normalize_denoising_horizon,
                normalize_act_space_dimension=self.normalize_act_space_dim,
                verbose=verbose,
                clip_intermediate_actions=self.clip_intermediate_actions,
                account_for_initial_stochasticity=self.account_for_initial_stochasticity
            )
            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")
            if update_epoch == 0 and batch_id == 0 and np.abs(ratio - 1.00) > self.initial_ratio_error_threshold:
                log.info(f"Warning: ratio={ratio} not 1.00 when update_epoch ==0  and batch_id ==0, there must be some bugs in your code not related to hyperparameters!")
            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)
            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
            clipfracs_list += [clipfrac]
            noise_std_list += [noise_std]
            loss.backward()
            # Gradient accumulation support
            if (batch_id + 1) % self.grad_accumulate == 0:
                # Debug the losses
                actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
                critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))
                if verbose:
                    log.info(f"before clipping: actor_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}")
                # Update actor: after critic warmup update the actor less frequently but more times. 
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                # Update critic
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                # Release gradient accumulation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                # Report
                log.info(f"run grad update at batch {batch_id}")
                log.info(f"approx_kl: {approx_kl}, update_epoch: {update_epoch}/{self.update_epochs}, num_batch: {self.total_steps //self.batch_size}")
        clip_fracs = np.mean(clipfracs_list)
        noise_stds = np.mean(noise_std_list)
        self.train_ret_dict = {
            "loss": loss,
            "pg loss": pg_loss,
            "value loss": v_loss,
            "entropy_loss": entropy_loss,
            "bc_loss": bc_loss,
            "approx kl": self.approx_kl,
            "ratio": ratio,
            "clipfrac": clip_fracs,
            "explained variance": self.explained_var,
            "old_logprob_min": oldlogprob_min,
            "old_logprob_max": oldlogprob_max,
            "old_logprob_std": oldlogprob_std,
            "new_logprob_min": newlogprob_min,
            "new_logprob_max": newlogprob_max,
            "new_logprob_std": newlogprob_std,
            "actor_norm": actor_norm,
            "critic_norm": critic_norm,
            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic lr": self.critic_optimizer.param_groups[0]["lr"],
            "min_logprob_noise_std": self.model.min_logprob_denoising_std,
            "min_sampling_noise_std": self.model.min_sampling_denoising_std,
            "noise_std": noise_stds,
            "Q_values": self.Q_values
        }

    def minibatch_generator_repeat(self):
        """Generate minibatches for PI0 training"""
        self.approx_kl = 0.0
        obs, chains, returns, oldvalues, advantages, oldlogprobs = self.buffer.make_dataset()
        # Explained variation of future rewards using value function
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        self.Q_values = oldvalues.mean().item()
        duplicate_multiplier = self.minibatch_duplicate_multiplier 
        self.total_steps = self.n_steps * self.n_envs * duplicate_multiplier
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            if self.lr_schedule == 'fixed' and self.kl_change_too_much:
                break
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b,
                    (self.n_steps * self.n_envs, duplicate_multiplier),
                )
                minibatch = (
                    {k: obs[k][batch_inds_b] for k in obs},  # SimplerEnv observations
                    chains[batch_inds_b],  # PI0 action chains (32-dim)
                    returns[batch_inds_b], 
                    oldvalues[batch_inds_b],
                    advantages[batch_inds_b],
                    oldlogprobs[batch_inds_b] 
                )
                if (self.lr_schedule == 'fixed' 
                    and self.target_kl 
                    and self.approx_kl > self.target_kl
                    and self.itr >= self.n_critic_warmup_itr):
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                yield update_epoch, batch_id, minibatch
