"""
Environment wrapper for SimplerEnv environments with image observations and language instructions.

This wrapper supports:
- Google Robot and WidowX robot selection
- Random scene generation across episodes
- Image and language instruction observations
- Action normalization/unnormalization
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import imageio
from typing import Dict, List, Optional, Any

# SimplerEnv imports
from SimplerEnv.simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from SimplerEnv.simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


class SimplerEnvImageWrapper(gym.Env):
    def __init__(
        self,
        env_name: str = "GraspSingleRandomObjectInScene-v0",
        scene_name: str = "google_pick_coke_can_1_v4",
        robot_type: str = "google_robot",  # "google_robot" or "widowx"
        robot_variant: Optional[str] = None,  # For WidowX: None, "bridge_dataset", "sink_camera"
        shape_meta: Dict = None,
        normalization_path: Optional[str] = None,
        enable_random_scene: bool = False,
        env_name_pool: Optional[List[str]] = None,
        scene_name_pool: Optional[List[str]] = None,
        additional_env_build_kwargs: Optional[Dict] = None,
        clamp_obs: bool = False,
        control_freq: int = 3,
        sim_freq: int = 513,
        max_episode_steps: int = 80,
        obs_camera_name: Optional[str] = None,
        render_camera_name: Optional[str] = None,
        **env_kwargs
    ):
        # Store configuration
        self.base_env_name = env_name
        self.base_scene_name = scene_name
        self.robot_type = robot_type
        self.robot_variant = robot_variant
        self.enable_random_scene = enable_random_scene
        self.env_name_pool = env_name_pool or [env_name]
        self.scene_name_pool = scene_name_pool or [scene_name]
        self.additional_env_build_kwargs = additional_env_build_kwargs or {}
        self.clamp_obs = clamp_obs
        self.control_freq = control_freq
        self.sim_freq = sim_freq
        self.max_episode_steps = max_episode_steps
        self.env_kwargs = env_kwargs

        # Configure robot-specific settings
        self._setup_robot_config()
        
        # Set camera names
        self.obs_camera_name = obs_camera_name or self.default_obs_camera
        self.render_camera_name = render_camera_name or self.default_render_camera

        # Set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization.get("obs_min", None)
            self.obs_max = normalization.get("obs_max", None) 
            self.action_min = normalization.get("action_min", None)
            self.action_max = normalization.get("action_max", None)

        # Initialize environment
        self.env = None
        self.current_env_name = self.base_env_name
        self.current_scene_name = self.base_scene_name
        self.current_env_kwargs = self.additional_env_build_kwargs.copy()
        self._create_environment()

        # Setup observation and action spaces
        self._setup_spaces(shape_meta)
        
        # Video recording
        self.video_writer = None

    def _setup_robot_config(self):
        """Configure robot-specific parameters"""
        if self.robot_type == "google_robot":
            self.robot_name = "google_robot_static"
            self.default_obs_camera = "overhead_camera"
            self.default_render_camera = "overhead_camera"
            self.control_mode = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
            
        elif self.robot_type == "widowx":
            if self.robot_variant == "bridge_dataset":
                self.robot_name = "widowx_bridge_dataset_camera_setup"
            elif self.robot_variant == "sink_camera":
                self.robot_name = "widowx_sink_camera_setup"
            else:
                self.robot_name = "widowx"
            self.default_obs_camera = "3rd_view_camera"
            self.default_render_camera = "3rd_view_camera"
            self.control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
            
        else:
            raise ValueError(f"Unsupported robot_type: {self.robot_type}")

    def _create_environment(self):
        """Create the ManiSkill2 environment"""
        env_kwargs = dict(
            obs_mode="rgbd",
            robot=self.robot_name,
            sim_freq=self.sim_freq,
            control_mode=self.control_mode,
            control_freq=self.control_freq,
            max_episode_steps=self.max_episode_steps,
            scene_name=self.current_scene_name,
            camera_cfgs={"add_segmentation": True},
            **self.current_env_kwargs,
            **self.env_kwargs
        )
        
        self.env = build_maniskill2_env(
            self.current_env_name,
            **env_kwargs
        )

    def _setup_spaces(self, shape_meta: Optional[Dict]):
        """Setup observation and action spaces"""
        # Action space - normalized to [-1, 1]
        if hasattr(self.env, 'action_dimension'):
            action_dim = self.env.action_dimension
        else:
            # Fallback: typical robot action dimension (3 pos + 3 rot + 1 gripper)
            action_dim = 7
            
        low = np.full(action_dim, fill_value=-1)
        high = np.full(action_dim, fill_value=1)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float32,
        )

        # Observation space
        observation_space = spaces.Dict()
        
        if shape_meta is not None:
            # Use provided shape_meta
            for key, value in shape_meta["obs"].items():
                shape = value["shape"]
                if key.endswith("rgb"):
                    min_value, max_value = 0, 255  # Image values
                elif key.endswith("instruction"):
                    # For text instructions, we'll use a dummy space
                    # In practice, this will be handled separately
                    continue
                else:
                    min_value, max_value = -1, 1
                
                observation_space[key] = spaces.Box(
                    low=min_value,
                    high=max_value,
                    shape=shape,
                    dtype=np.float32 if not key.endswith("rgb") else np.uint8,
                )
        else:
            # Default observation space
            # RGB image space - will be determined dynamically
            observation_space["rgb"] = spaces.Box(
                low=0,
                high=255,
                shape=(3, 256, 256),  # Default shape, will be updated
                dtype=np.uint8,
            )
        
        # Add instruction space as a dummy - actual instructions are strings
        # We'll handle this separately in get_observation
        
        self.observation_space = observation_space

    def _select_random_scene(self):
        """Randomly select environment and scene configuration"""
        if self.enable_random_scene and len(self.env_name_pool) > 1:
            self.current_env_name = random.choice(self.env_name_pool)
        else:
            self.current_env_name = self.base_env_name
            
        if self.enable_random_scene and len(self.scene_name_pool) > 1:
            self.current_scene_name = random.choice(self.scene_name_pool)
        else:
            self.current_scene_name = self.base_scene_name
            
        # Optionally add random environment variations
        if self.enable_random_scene:
            variations = {}
            # Add some random lighting/orientation variations
            if random.random() < 0.3:  # 30% chance
                if random.random() < 0.5:
                    variations["slightly_darker_lighting"] = True
                else:
                    variations["slightly_brighter_lighting"] = True
                    
            # Random object orientation for applicable environments
            if "Object" in self.current_env_name and random.random() < 0.5:
                orientation_options = [
                    {"lr_switch": True},
                    {"upright": True}, 
                    {"laid_vertically": True},
                ]
                variations.update(random.choice(orientation_options))
                
            self.current_env_kwargs = {**self.additional_env_build_kwargs, **variations}
        else:
            self.current_env_kwargs = self.additional_env_build_kwargs.copy()

    def normalize_obs(self, obs):
        """Normalize observations if normalization is enabled"""
        if self.obs_min is not None and self.obs_max is not None:
            obs = 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)  # -> [-1, 1]
            if self.clamp_obs:
                obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        """Unnormalize action from [-1, 1] to environment range"""
        if self.action_min is not None and self.action_max is not None:
            action = (action + 1) / 2  # [-1, 1] -> [0, 1]
            return action * (self.action_max - self.action_min) + self.action_min
        return action

    def get_observation(self, raw_obs):
        """Process raw observation into formatted output"""
        # Get image
        image = get_image_from_maniskill2_obs_dict(
            self.env, raw_obs, camera_name=self.obs_camera_name
        )
        
        # Get language instruction
        instruction = self.env.get_language_instruction()
        
        # Prepare observation dictionary
        obs = {
            "rgb": image.astype(np.float32),  # Convert to float32, keep [0-255] range
            "instruction": instruction,
            "robot_type": self.robot_type,
            "scene_info": {
                "env_name": self.current_env_name,
                "scene_name": self.current_scene_name,
                "robot_name": self.robot_name,
                "camera_name": self.obs_camera_name
            }
        }
        
        # Apply observation normalization if needed (not typically for images)
        # obs["rgb"] would normally stay in [0, 255] range
        
        return obs

    def seed(self, seed=None):
        """Set random seed"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        else:
            random.seed()
            np.random.seed()

    def reset(self, options=None, **kwargs):
        """Reset environment and return initial observation"""
        if options is None:
            options = {}
            
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Handle seed
        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)

        # Select random scene if enabled
        if self.enable_random_scene:
            self._select_random_scene()
            # Recreate environment with new configuration
            self._create_environment()

        # Reset environment
        env_reset_options = options.copy()
        env_reset_options.pop("video_path", None)
        env_reset_options.pop("seed", None)
        
        raw_obs, info = self.env.reset(options=env_reset_options, **kwargs)
        
        return self.get_observation(raw_obs), info

    def step(self, action):
        """Execute action and return observation, reward, done, info"""
        # Unnormalize action if needed
        if self.normalize:
            action = self.unnormalize_action(action)
        
        # Execute action
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get processed observation
        obs = self.get_observation(raw_obs)
        
        # Render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array", width: int = 256, height: int = 256):
        """Render environment with unified interface"""
        return self.env.render(
            mode=mode,
            height=height,
            width=width,
            camera_name=self.render_camera_name,
        )

    def close(self):
        """Close environment"""
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if self.env is not None:
            self.env.close()

    @property 
    def robot_uid(self):
        """Get current robot UID for compatibility"""
        if hasattr(self.env, 'robot_uid'):
            return self.env.robot_uid
        return self.robot_name

    def get_language_instruction(self):
        """Get current language instruction"""
        return self.env.get_language_instruction()


if __name__ == "__main__":
    # Example usage
    import os
    os.environ["MUJOCO_GL"] = "egl"
    
    # Google Robot example
    print("Testing Google Robot...")
    wrapper = SimplerEnvImageWrapper(
        env_name="GraspSingleRandomObjectInScene-v0",
        scene_name="google_pick_coke_can_1_v4", 
        robot_type="google_robot",
        enable_random_scene=False,
        max_episode_steps=50
    )
    
    wrapper.seed(42)
    obs, info = wrapper.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"RGB shape: {obs['rgb'].shape}")
    print(f"Instruction: {obs['instruction']}")
    print(f"Robot type: {obs['robot_type']}")
    
    for i in range(5):
        action = wrapper.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)
        print(f"Step {i}: reward={reward}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            break
            
    wrapper.close()
    
    # WidowX example
    print("\nTesting WidowX Robot...")
    wrapper = SimplerEnvImageWrapper(
        env_name="PutSpoonOnTableClothInScene-v0",
        scene_name="bridge_table_1_v1",
        robot_type="widowx", 
        robot_variant="bridge_dataset",
        enable_random_scene=False,
        max_episode_steps=50
    )
    
    wrapper.seed(42)
    obs, info = wrapper.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"RGB shape: {obs['rgb'].shape}")
    print(f"Instruction: {obs['instruction']}")
    print(f"Robot type: {obs['robot_type']}")
    
    wrapper.close()
    print("Test completed!")
