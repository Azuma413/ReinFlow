#!/usr/bin/env python3
"""
SimplerEnv環境でPI0を動かすテストスクリプト

SimplerEnvImageWrapperをMultiStepでラップし、PPOPi0モデルを使用して
環境内でエージェントを動作させるテストを行います。
"""
import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from omegaconf import OmegaConf
import argparse
from dataclasses import dataclass
from typing import Dict, Any
import cv2
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["MUJOCO_GL"] = "egl"
from env.gym_utils.wrapper.simplerenv_image import SimplerEnvImageWrapper
from env.gym_utils.wrapper.multi_step import MultiStep
from model.flow.ft_ppo.ppopi0 import PPOPi0
from SimplerEnv.simpler_env.utils.visualization import write_video
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class SimplerEnvTester:
    """SimplerEnv + PI0のテストクラス"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.env = self._create_environment()
        self.model = self._create_model()

    def _create_environment(self):
        """環境の作成"""
        print("Creating SimplerEnv environment...")
        # 設定から環境パラメータを取得
        env_config = self.config.env.wrappers.simplerenv_image
        multi_step_config = self.config.env.wrappers.multi_step
        base_env = SimplerEnvImageWrapper(
            env_name="GraspSingleRandomObjectInScene-v0",  # デフォルト環境
            scene_name=env_config.scene_name,
            robot_type=env_config.robot_type,
            robot_variant=env_config.get("robot_variant", None),
            enable_random_scene=env_config.get("enable_random_scene", False),
            max_episode_steps=env_config.max_episode_steps,
            control_freq=env_config.get("control_freq", 3),
            sim_freq=env_config.get("sim_freq", 513),
            shape_meta=self.config.get("shape_meta", None),
        )
        env = MultiStep(
            env=base_env,
            n_obs_steps=multi_step_config.n_obs_steps,
            n_action_steps=multi_step_config.n_action_steps,
            max_episode_steps=multi_step_config.max_episode_steps,
            reset_within_step=True,
        )
        print(f"Environment created successfully")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        return env

    def _create_model(self):
        """PPOPi0モデルの作成"""
        print("Creating PPOPi0 model...")
        # ダミーのクリティック（テスト用）
        class DummyCritic(torch.nn.Module):
            def __init__(self, obs_dim):
                super().__init__()
                self.linear = torch.nn.Linear(obs_dim, 1)
            def forward(self, obs):
                # 状態から価値を予測（ダミー実装）
                if isinstance(obs, dict):
                    state = obs.get("state", torch.zeros(obs["rgb"].shape[0], 8))
                    if state.dim() > 2:
                        state = state.flatten(1)
                else:
                    state = obs
                return self.linear(state)
        critic = DummyCritic(obs_dim=self.config.obs_dim)
        # PPOPi0モデルの作成（新しい設定構造に対応）
        model = PPOPi0(
            device=self.device,
            pi0_policy_path=self.config.pi0_policy_path,
            pi0_config=self.config.model.pi0_config,  # yamlのpi0_configを直接渡す
            critic=critic,
            act_dim=self.config.action_dim,
            horizon_steps=self.config.horizon_steps,
            act_min=self.config.model.act_min,
            act_max=self.config.model.act_max,
            obs_dim=self.config.obs_dim,
            cond_steps=self.config.cond_steps,
            noise_scheduler_type=self.config.model.noise_scheduler_type,
            inference_steps=self.config.model.inference_steps,
            ft_denoising_steps=self.config.model.ft_denoising_steps,
            randn_clip_value=self.config.model.randn_clip_value,
            min_sampling_denoising_std=self.config.model.min_sampling_denoising_std,
            min_logprob_denoising_std=self.config.model.min_logprob_denoising_std,
            max_logprob_denoising_std=self.config.model.max_logprob_denoising_std,
            logprob_min=self.config.model.logprob_min,
            logprob_max=self.config.model.logprob_max,
            clip_ploss_coef=self.config.model.clip_ploss_coef,
            clip_ploss_coef_base=self.config.model.clip_ploss_coef_base,
            clip_ploss_coef_rate=self.config.model.clip_ploss_coef_rate,
            clip_vloss_coef=self.config.model.get("clip_vloss_coef", None),
            denoised_clip_value=self.config.model.denoised_clip_value,
            time_dim_explore=self.config.model.time_dim_explore,
            learn_explore_time_embedding=self.config.model.learn_explore_time_embedding,
            use_time_independent_noise=self.config.model.use_time_independent_noise,
            noise_hidden_dims=self.config.model.noise_hidden_dims,
            logprob_debug_sample=self.config.model.logprob_debug_sample,
            logprob_debug_recalculate=self.config.model.logprob_debug_recalculate,
            explore_net_activation_type=self.config.model.explore_net_activation_type,
            pooling_mode=self.config.model.get("pooling_mode", "cls"),
        )
        model.to(self.device)
        model.eval()  # 推論モード
        print("PPOPi0 model created successfully")
        return model

    def convert_simplerenv_to_pi0_observation(self, simplerenv_obs: Dict) -> PI0ObservationBatch:
        """
        Convert SimplerEnv observation format to PI0 format
        """
        batch_size = 1  # Single observation for testing
        rgb_images = simplerenv_obs["rgb"]  # [n_obs_steps, H, W, C] or [H, W, C]
        if rgb_images.ndim == 4:  # (n_obs_steps, H, W, C)
            rgb_images = rgb_images[-1]  # 最新の観測を使用
        # Ensure images are in correct format and device
        if not isinstance(rgb_images, torch.Tensor):
            rgb_images = torch.from_numpy(rgb_images).float()
        # Convert to [C, H, W] format if needed
        if rgb_images.dim() == 3 and rgb_images.shape[2] in [1, 3, 4]:
            rgb_images = rgb_images.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        # Add batch dimension and move to device
        rgb_images = rgb_images.unsqueeze(0).to(self.device)  # [1, C, H, W]
        # Create image dictionary (PI0 expects named images)
        images = {
            "base_0_rgb": rgb_images,
            "left_wrist_0_rgb": torch.zeros_like(rgb_images),
            "right_wrist_0_rgb": torch.zeros_like(rgb_images)
        }
        # Create image masks (all empty since we don't use masks in testing)
        image_masks = {}
        # Get tokenized prompts from observation
        tokenized_prompt = torch.from_numpy(simplerenv_obs["tokenized_prompt"]).to(self.device)
        tokenized_prompt_mask = torch.from_numpy(simplerenv_obs["tokenized_prompt_mask"]).to(self.device)
        # Handle multi-step observations
        if tokenized_prompt.dim() == 2:  # (n_obs_steps, max_token_len)
            tokenized_prompt = tokenized_prompt[-1]  # 最新のプロンプトを使用
            tokenized_prompt_mask = tokenized_prompt_mask[-1]
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
            # Handle multi-step observations
            if processed_state.dim() == 2:  # (n_obs_steps, state_dim)
                processed_state = processed_state[-1]  # 最新の状態を使用
            # Expand 8-dimensional processed state to 32-dimensional PI0 format
            pi0_state = self.expand_state_to_pi0_dimension(processed_state.unsqueeze(0))
        else:
            # Fallback: create dummy state
            logger.warning("No robot state found in observation, using dummy state")
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
        """
        batch_size = processed_state.shape[0]
        # Check for NaN/inf in input
        if torch.any(~torch.isfinite(processed_state)):
            logger.warning(f"Warning: Invalid values in processed_state: {processed_state}")
            # Use safe default state
            return torch.zeros(batch_size, 32, dtype=torch.float32, device=self.device)
        try:
            # Simply pad the 8-dimensional processed state to 32 dimensions
            pi0_states = []
            for i in range(batch_size):
                state_8d = processed_state[i].cpu().numpy()  # [8] - already processed by environment
                # Check for valid input values
                if not np.all(np.isfinite(state_8d)):
                    logger.warning(f"Invalid state_8d at batch {i}: {state_8d}")
                    pi0_states.append(np.zeros(32, dtype=np.float32))
                    continue
                # Pad to 32 dimensions (environment already did the robot-specific processing)
                pi0_state = np.concatenate([state_8d, np.zeros(32 - 8)])
                # Final check for NaN values
                if not np.all(np.isfinite(pi0_state)):
                    logger.warning(f"Final pi0_state contains invalid values, using zeros")
                    pi0_state = np.zeros(32, dtype=np.float32)
                pi0_states.append(pi0_state)
            pi0_states = np.stack(pi0_states)
            return torch.from_numpy(pi0_states).float().to(self.device)
        except Exception as e:
            logger.warning(f"Error expanding state to PI0 dimension: {e}, using zero state")
            return torch.zeros(batch_size, 32, dtype=torch.float32, device=self.device)

    def run_test(self, num_episodes=3, max_steps_per_episode=50):
        """テストの実行"""
        print(f"Starting test with {num_episodes} episodes...")
        total_rewards = []
        success_count = 0
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            # エピソードの初期化
            obs, info = self.env.reset(seed=episode)
            episode_reward = 0.0
            step_count = 0
            # RGB画像を保存するためのリスト
            rgb_images = []
            # 言語指示の表示
            if hasattr(self.env.env, 'get_language_instruction'):
                instruction = self.env.env.get_language_instruction()
                print(f"Language instruction: {instruction}")
            # 観測の形状確認
            print(f"Observation keys: {obs.keys()}")
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {type(value)}")
            # 初期観測のRGB画像を保存
            if "rgb" in obs:
                # print("reset image shape: ", obs["rgb"].shape) # reset image shape:  (1, 512, 640, 3)
                rgb_images : np.ndarray = obs["rgb"].copy()
            while step_count < max_steps_per_episode:
                try:
                    # SimplerEnv観測をPI0形式に変換
                    pi0_obs_batch = self.convert_simplerenv_to_pi0_observation(obs)
                    # PI0でアクションを生成
                    with torch.no_grad():
                        # アクション生成（評価モード）
                        robot_actions = self.model.get_actions(
                            cond=pi0_obs_batch,
                            eval_mode=True,
                            save_chains=False,
                            ret_logprob=False
                        )
                        # アクションの形状確認
                        if step_count == 0:
                            print(f"Generated action shape: {robot_actions.shape}")
                        # PI0のactionを環境用actionに変換
                        # robot_actions: [batch_size, horizon_steps, action_dim]
                        pi0_action_chunk = robot_actions[0]  # [horizon_steps, action_dim]
                        env_action_chunk = self.env.env.postprocess_pi0_action_for_env(pi0_action_chunk)
                        print(f"original action: {pi0_action_chunk[0]}\nconverted action: {env_action_chunk[0]}")
                        if step_count == 0:
                            print(f"Converted action chunk shape: {env_action_chunk.shape}")
                        action_chunk = env_action_chunk[:self.config.act_steps]
                    # 環境でステップ実行（MultiStepがアクションチャンク全体を処理）
                    obs, reward, terminated, truncated, info = self.env.step(action_chunk)
                    # RGB画像を保存
                    intermediate_frames = self.env._get_obs(self.config.act_steps)["rgb"].copy()
                    # print(f"Step {step_count} intermediate frames shape: {intermediate_frames.shape}")
                    rgb_images = np.concatenate([rgb_images, intermediate_frames], axis=0)
                    episode_reward += reward
                    step_count += self.config.act_steps
                    if step_count % 10 == 0:
                        print(f"  Step {step_count}: reward={reward:.3f}, total_reward={episode_reward:.3f}")
                    if terminated or truncated:
                        if terminated:
                            success_count += 1
                            print(f"  Episode completed successfully!")
                        else:
                            print(f"  Episode truncated.")
                        break
                except Exception as e:
                    logger.error(f"Error at step {step_count}: {e}")
                    break
            # エピソード終了後、RGB画像を動画として保存
            if rgb_images is not None:
                video_path = self.get_next_video_path()
                print(f"Saving {len(rgb_images)} frames as video...")
                rgb_images = np.array(rgb_images)
                if rgb_images.ndim == 5:
                    rgb_images = rgb_images.squeeze(1)
                rgb_images = rgb_images.astype(np.uint8)
                write_video(video_path, rgb_images, fps=10)
            else:
                logger.warning(f"No RGB images collected for episode {episode + 1}")
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1} finished: reward={episode_reward:.3f}, steps={step_count}")
        # 結果の表示
        print(f"\n=== Test Results ===")
        print(f"Episodes completed: {num_episodes}")
        print(f"Success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
        print(f"Average reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
        print(f"Reward range: [{np.min(total_rewards):.3f}, {np.max(total_rewards):.3f}]")
        return {
            "success_rate": success_count / num_episodes,
            "average_reward": np.mean(total_rewards),
            "reward_std": np.std(total_rewards),
            "total_rewards": total_rewards
        }

    def get_next_video_path(self, base_dir="log/eval"):
        """重複を避けるための次の動画パスを取得"""
        test_idx = 0
        while True:
            test_dir = Path(base_dir) / f"test{test_idx}"
            video_idx = 0
            while True:
                video_path = test_dir / f"video{video_idx}.mp4"
                if not video_path.exists():
                    # ディレクトリを作成
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    return str(video_path)
                video_idx += 1
            test_idx += 1

    def cleanup(self):
        """リソースのクリーンアップ"""
        if hasattr(self, 'env'):
            self.env.close()
        print("Cleanup completed")

def create_default_config():
    """デフォルト設定の作成（ft_ppo_pi0.yamlの構造に合わせる）"""
    config = OmegaConf.load("cfg/ft_ppo_pi0.yaml")
    return config

def main():
    # ログレベル設定
    logging.getLogger().setLevel(logging.DEBUG)
    config = create_default_config()
    # テスト実行
    tester = None
    try:
        print("Initializing SimplerEnv + PI0 tester...")
        tester = SimplerEnvTester(config)
        print("Running test...")
        results = tester.run_test(
            num_episodes=10,
            max_steps_per_episode=config.env.max_episode_steps
        )
        print("Test completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if tester:
            tester.cleanup()

if __name__ == "__main__":
    main()
