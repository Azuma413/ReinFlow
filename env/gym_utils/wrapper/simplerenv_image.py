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
import torch
import gymnasium as gym
from gymnasium import spaces
import imageio
from typing import Dict, List, Optional, Any

# SimplerEnv imports
from SimplerEnv.simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from SimplerEnv.simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# PI0 imports for tokenization
from openpi_Azuma413.src.openpi.models.tokenizer import PaligemmaTokenizer


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
        additional_env_build_kwargs: Optional[Dict] = None,
        clamp_obs: bool = False,
        control_freq: int = 3,
        sim_freq: int = 513,
        max_episode_steps: int = 80,
        obs_camera_name: Optional[str] = None,
        render_camera_name: Optional[str] = None,
        max_token_len: int = 512,
        **env_kwargs
    ):
        # Store configuration
        self.base_env_name = env_name
        self.base_scene_name = scene_name
        self.robot_type = robot_type
        self.robot_variant = robot_variant
        self.enable_random_scene = enable_random_scene
        self.additional_env_build_kwargs = additional_env_build_kwargs or {}
        self.clamp_obs = clamp_obs
        self.control_freq = control_freq
        self.sim_freq = sim_freq
        self.max_episode_steps = max_episode_steps
        self.max_token_len = max_token_len
        self.env_kwargs = env_kwargs
        # Initialize PI0 tokenizer
        self.instruction_tokenizer = PaligemmaTokenizer(max_len=self.max_token_len)
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
                    observation_space[key] = spaces.Box(
                        low=min_value,
                        high=max_value,
                        shape=shape,
                        dtype=np.uint8,
                    )
                elif key.endswith("instruction"):
                    # For text instructions, use Text space
                    observation_space[key] = spaces.Text(max_length=1000)
                else:
                    min_value, max_value = -1, 1
                    observation_space[key] = spaces.Box(
                        low=min_value,
                        high=max_value,
                        shape=shape,
                        dtype=np.float32,
                    )
        else:
            # Default observation space
            observation_space["rgb"] = spaces.Box(
                low=0,
                high=255,
                shape=(3, 256, 256),  # Default shape, will be updated
                dtype=np.uint8,
            )
            # Add instruction space
            observation_space["instruction"] = spaces.Text(max_length=1000)
            # Add state space (robot state: x,y,z,qw,qx,qy,qz,gripper)
            observation_space["state"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(8,),
                dtype=np.float32,
            )
            # Add robot_type space (categorical, but we'll use Text for simplicity)
            observation_space["robot_type"] = spaces.Text(max_length=50)
        # Always add tokenized prompt spaces
        observation_space["tokenized_prompt"] = spaces.Box(
            low=np.iinfo(np.int64).min,
            high=np.iinfo(np.int64).max,
            shape=(self.max_token_len,),
            dtype=np.int64,
        )
        observation_space["tokenized_prompt_mask"] = spaces.Box(
            low=0,
            high=1,
            shape=(self.max_token_len,),
            dtype=np.bool_,
        )
        self.observation_space = observation_space

    def _select_random_scene(self):
        """Randomly select environment and scene configuration based on robot type"""
        if not self.enable_random_scene:
            self.current_env_name = self.base_env_name
            self.current_scene_name = self.base_scene_name
            self.current_env_kwargs = self.additional_env_build_kwargs.copy()
            return
        # Define robot-type specific environment-scene pairs
        robot_compatible_pairs = self._get_robot_compatible_pairs()
        if robot_compatible_pairs:
            # Select a random compatible pair
            selected_pair = random.choice(robot_compatible_pairs)
            self.current_env_name = selected_pair["env_name"]
            self.current_scene_name = selected_pair["scene_name"]
        else:
            # Fallback to base configuration if no compatible pairs found
            self.current_env_name = self.base_env_name
            self.current_scene_name = self.base_scene_name
        # Add random environment variations
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

    def _get_robot_compatible_pairs(self):
        """Get environment-scene pairs compatible with the current robot type"""
        # Define robot-type specific compatible environment-scene pairs
        google_robot_pairs = [
            # Google Robot grasp tasks
            {"env_name": "GraspSingleRandomObjectInScene-v0", "scene_name": "google_pick_coke_can_1_v4"},
            {"env_name": "GraspSingleRandomObjectInScene-v0", "scene_name": "google_pick_coke_can_1_v4_alt_background"},
            {"env_name": "GraspSingleRandomObjectInScene-v0", "scene_name": "google_pick_coke_can_1_v4_alt_background_2"},
            {"env_name": "GraspSingleRandomObjectInScene-v0", "scene_name": "Baked_sc1_staging_objaverse_cabinet1_h870"},
            {"env_name": "GraspSingleRandomObjectInScene-v0", "scene_name": "Baked_sc1_staging_objaverse_cabinet2_h870"},
            {"env_name": "GraspSingleRandomObjectAltGoogleCameraInScene-v0", "scene_name": "google_pick_coke_can_1_v4"},
            {"env_name": "GraspSingleRandomObjectAltGoogleCamera2InScene-v0", "scene_name": "google_pick_coke_can_1_v4"},
            {"env_name": "GraspSingleRandomObjectDistractorInScene-v0", "scene_name": "google_pick_coke_can_1_v4"},
            # Google Robot move near tasks
            {"env_name": "MoveNearGoogleInScene-v0", "scene_name": "google_pick_coke_can_1_v4"},
            {"env_name": "MoveNearGoogleBakedTexInScene-v0", "scene_name": "google_pick_coke_can_1_v4"},
            {"env_name": "MoveNearAltGoogleCameraInScene-v0", "scene_name": "google_pick_coke_can_1_v4"},
            {"env_name": "MoveNearAltGoogleCamera2InScene-v0", "scene_name": "google_pick_coke_can_1_v4"},
            # Google Robot drawer tasks
            {"env_name": "OpenTopDrawerCustomInScene-v0", "scene_name": "dummy_drawer"},
            {"env_name": "OpenMiddleDrawerCustomInScene-v0", "scene_name": "dummy_drawer"},
            {"env_name": "OpenBottomDrawerCustomInScene-v0", "scene_name": "dummy_drawer"},
            {"env_name": "CloseTopDrawerCustomInScene-v0", "scene_name": "dummy_drawer"},
            {"env_name": "CloseMiddleDrawerCustomInScene-v0", "scene_name": "dummy_drawer"},
            {"env_name": "CloseBottomDrawerCustomInScene-v0", "scene_name": "dummy_drawer"},
            {"env_name": "PlaceIntoClosedTopDrawerCustomInScene-v0", "scene_name": "dummy_drawer"},
            # Alternative drawer scenes
            {"env_name": "OpenTopDrawerCustomInScene-v0", "scene_name": "frl_apartment_stage_simple"},
            {"env_name": "OpenMiddleDrawerCustomInScene-v0", "scene_name": "frl_apartment_stage_simple"},
            {"env_name": "OpenBottomDrawerCustomInScene-v0", "scene_name": "frl_apartment_stage_simple"},
            {"env_name": "CloseTopDrawerCustomInScene-v0", "scene_name": "frl_apartment_stage_simple"},
            {"env_name": "CloseMiddleDrawerCustomInScene-v0", "scene_name": "frl_apartment_stage_simple"},
            {"env_name": "CloseBottomDrawerCustomInScene-v0", "scene_name": "frl_apartment_stage_simple"},
            {"env_name": "PlaceIntoClosedTopDrawerCustomInScene-v0", "scene_name": "frl_apartment_stage_simple"},
            {"env_name": "OpenTopDrawerCustomInScene-v0", "scene_name": "modern_bedroom_no_roof"},
            {"env_name": "OpenMiddleDrawerCustomInScene-v0", "scene_name": "modern_bedroom_no_roof"},
            {"env_name": "OpenBottomDrawerCustomInScene-v0", "scene_name": "modern_bedroom_no_roof"},
            {"env_name": "CloseTopDrawerCustomInScene-v0", "scene_name": "modern_bedroom_no_roof"},
            {"env_name": "CloseMiddleDrawerCustomInScene-v0", "scene_name": "modern_bedroom_no_roof"},
            {"env_name": "CloseBottomDrawerCustomInScene-v0", "scene_name": "modern_bedroom_no_roof"},
            {"env_name": "PlaceIntoClosedTopDrawerCustomInScene-v0", "scene_name": "modern_bedroom_no_roof"},
            {"env_name": "OpenTopDrawerCustomInScene-v0", "scene_name": "modern_office_no_roof"},
            {"env_name": "OpenMiddleDrawerCustomInScene-v0", "scene_name": "modern_office_no_roof"},
            {"env_name": "OpenBottomDrawerCustomInScene-v0", "scene_name": "modern_office_no_roof"},
            {"env_name": "CloseTopDrawerCustomInScene-v0", "scene_name": "modern_office_no_roof"},
            {"env_name": "CloseMiddleDrawerCustomInScene-v0", "scene_name": "modern_office_no_roof"},
            {"env_name": "CloseBottomDrawerCustomInScene-v0", "scene_name": "modern_office_no_roof"},
            {"env_name": "PlaceIntoClosedTopDrawerCustomInScene-v0", "scene_name": "modern_office_no_roof"},
        ]
        widowx_pairs = [
            # Bridge dataset tasks
            {"env_name": "PutCarrotOnPlateInScene-v0", "scene_name": "bridge_table_1_v1"},
            {"env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0", "scene_name": "bridge_table_1_v1"},
            {"env_name": "PutSpoonOnTableClothInScene-v0", "scene_name": "bridge_table_1_v1"},
            {"env_name": "PutEggplantInBasketScene-v0", "scene_name": "bridge_table_1_v2"},
        ]
        # Filter pairs based on robot type and available pools
        if self.robot_type == "google_robot":
            compatible_pairs = google_robot_pairs
        elif self.robot_type == "widowx":
            compatible_pairs = widowx_pairs
        else:
            # Unknown robot type, return empty list
            return []
        return compatible_pairs

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
        image = get_image_from_maniskill2_obs_dict(
            self.env, raw_obs, camera_name=self.obs_camera_name
        )
        instruction = self.env.get_language_instruction()
        if not isinstance(instruction, str):
            print("Warning: instruction is not a string, using default")
            instruction = "pick up something"
        tokens, mask = self.instruction_tokenizer.tokenize(instruction, state=None)
        tokenized_prompt = torch.from_numpy(tokens).long().cpu().numpy()
        tokenized_prompt_mask = torch.from_numpy(mask).bool().cpu().numpy()
        # Get robot state information
        state = self._get_robot_state(raw_obs)
        obs = {
            "rgb": image.astype(np.float32),  # Convert to float32, keep [0-255] range
            # "robot_type": self.robot_type,
            "state": state,  # Add robot state
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
            # "scene_info": {
            #     "env_name": self.current_env_name,
            #     "scene_name": self.current_scene_name,
            #     "robot_name": self.robot_name,
            #     "camera_name": self.obs_camera_name
            # }
        }
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
            self._setup_spaces(shape_meta=None)
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
        return self.env.render()

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

    def _get_robot_state(self, raw_obs):
        """Extract robot state information from raw observation"""
        try:
            # Get robot agent from environment
            if hasattr(self.env, 'agent'):
                agent = self.env.agent
            elif hasattr(self.env, '_agent'):
                agent = self.env._agent
            else:
                # Try to get from unwrapped environment
                env = self.env
                while hasattr(env, 'env'):
                    env = env.env
                if hasattr(env, 'agent'):
                    agent = env.agent
                else:
                    raise AttributeError("Cannot find robot agent")
            
            # Get end-effector pose
            if hasattr(agent, 'tcp'):
                tcp_pose = agent.tcp.pose
                # Position (3D)
                position = tcp_pose.p  # xyz position
                # Quaternion (4D) - ManiSkill uses wxyz format
                quaternion = tcp_pose.q  # wxyz quaternion
                
                # Get gripper state
                if hasattr(agent, 'robot') and hasattr(agent.robot, 'get_qpos'):
                    qpos = agent.robot.get_qpos()
                    # For most robots, gripper is the last joint(s)
                    if self.robot_type == "google_robot":
                        # Google robot typically has gripper as last joint
                        gripper_pos = qpos[-1] if len(qpos) > 0 else 0.0
                        # Convert to gripper width (0=closed, 1=open)
                        gripper_width = np.clip(gripper_pos, 0.0, 1.0)
                    elif self.robot_type == "widowx":
                        # WidowX typically has gripper as last joint
                        gripper_pos = qpos[-1] if len(qpos) > 0 else 0.0
                        # Convert to gripper width (0=closed, 1=open)
                        gripper_width = np.clip(gripper_pos, 0.0, 1.0)
                    else:
                        gripper_width = 0.5  # Default middle position
                else:
                    gripper_width = 0.5  # Default if can't get gripper state
                
                # Combine into eef_pos format: [x, y, z, qw, qx, qy, qz, gripper]
                eef_pos = np.concatenate([
                    position,  # [x, y, z]
                    quaternion,  # [qw, qx, qy, qz] (wxyz format)
                    [gripper_width]  # gripper openness
                ])
                
                return eef_pos.astype(np.float32)
                
            else:
                # Fallback: try to get from observation
                if 'agent' in raw_obs:
                    agent_obs = raw_obs['agent']
                    if 'qpos' in agent_obs and 'qvel' in agent_obs:
                        qpos = agent_obs['qpos']
                        # Use first 7 elements as rough approximation
                        if len(qpos) >= 7:
                            return qpos[:8].astype(np.float32)  # Include gripper
                
                # Last resort: return zeros
                print(f"Warning: Could not extract robot state, returning zeros")
                return np.zeros(8, dtype=np.float32)
                
        except Exception as e:
            print(f"Error extracting robot state: {e}")
            # Return default state
            return np.zeros(8, dtype=np.float32)

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
