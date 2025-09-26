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
from SimplerEnv.simpler_env.policies.openpi.geometry import quat2mat, mat2euler
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
        # Tokenization optimization: cache tokenized prompts per episode
        self.cached_tokenized_prompt = None
        self.cached_tokenized_prompt_mask = None
        self.cached_instruction = None
        # PI0 preprocessing setup (for WidowX Bridge specific rotation matrix)
        self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    def _setup_robot_config(self):
        """Configure robot-specific parameters"""
        if self.robot_type == "google_robot":
            self.robot_name = "google_robot_static"
            self.default_obs_camera = "overhead_camera"
            self.default_render_camera = "overhead_camera"
        elif self.robot_type == "widowx":
            if self.robot_variant == "bridge_dataset":
                self.robot_name = "widowx_bridge_dataset_camera_setup"
            elif self.robot_variant == "sink_camera":
                self.robot_name = "widowx_sink_camera_setup"
            else:
                self.robot_name = "widowx"
            self.default_obs_camera = "3rd_view_camera"
            self.default_render_camera = "3rd_view_camera"
        else:
            raise ValueError(f"Unsupported robot_type: {self.robot_type}")
        
        # Use SimplerEnv's standard function to get control mode
        # This ensures compatibility with SimplerEnv's robot configurations
        self.control_mode = get_robot_control_mode(self.robot_name, "pi0")
        
        # Debug output to verify robot configuration
        print(f"Robot configuration: type={self.robot_type}, variant={self.robot_variant}, name={self.robot_name}")
        print(f"Control mode: {self.control_mode}")

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
            observation_space["state"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(8,),
                dtype=np.float32,
            )
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
        variations = {}
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

    def get_observation(self, raw_obs, is_reset=False):
        """Process raw observation into formatted output"""
        image = get_image_from_maniskill2_obs_dict(
            self.env, raw_obs, camera_name=self.obs_camera_name
        )
        instruction = self.env.get_language_instruction()
        if not isinstance(instruction, str):
            print("Warning: instruction is not a string, using default")
            instruction = "pick up something"
        # Optimization: Only tokenize if instruction changed or on reset
        if (is_reset or 
            self.cached_instruction != instruction or 
            self.cached_tokenized_prompt is None or 
            self.cached_tokenized_prompt_mask is None):
            # Tokenize and cache the instruction
            tokens, mask = self.instruction_tokenizer.tokenize(instruction, state=None)
            self.cached_tokenized_prompt = torch.from_numpy(tokens).long().cpu().numpy()
            self.cached_tokenized_prompt_mask = torch.from_numpy(mask).bool().cpu().numpy()
            self.cached_instruction = instruction
            if not is_reset:
                print(f"Info: Instruction changed, re-tokenizing: '{instruction}'")
        # Use cached tokenized values
        tokenized_prompt = self.cached_tokenized_prompt
        tokenized_prompt_mask = self.cached_tokenized_prompt_mask
        # Get robot state information
        raw_state = self._get_robot_state(raw_obs)
        # Apply PI0 preprocessing (次元拡張以外の処理)
        processed_state = self.preprocess_robot_state_for_pi0(raw_state)
        obs = {
            "rgb": image.astype(np.float32),  # Convert to float32, keep [0-255] range
            "state": processed_state,  # Add processed robot state (8次元)
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
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
            print(f"***********************\nrobot_type: {self.robot_type}\nenv: {self.current_env_name}, scene: {self.current_scene_name}\nkwargs: {self.current_env_kwargs}\n***********************")
            self.env.close()
            # Recreate environment with new configuration
            self._create_environment()
            self._setup_spaces(shape_meta=None)
        try:
            import gc
            gc.collect()
            # Reset environment with enhanced error handling
            env_reset_options = options.copy()
            env_reset_options.pop("video_path", None)
            env_reset_options.pop("seed", None)
            raw_obs, info = self.env.reset(options=env_reset_options, **kwargs)
            # Validate reset observation before processing
            if raw_obs is None:
                print("Warning: Reset returned None observation, retrying...")
                raw_obs, info = self.env.reset(options=env_reset_options, **kwargs)
            obs = self.get_observation(raw_obs, is_reset=True)
            if "state" in obs and obs["state"] is not None:
                if np.any(~np.isfinite(obs["state"])):
                    print(f"Critical: NaN/inf detected in observation state after reset: {obs['state']}")
                    # Force safe state
                    obs["state"] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
                    print(f"Replaced with safe state: {obs['state']}")
            return obs, info
            
        except Exception as e:
            print(f"Critical error during environment reset: {e}")
            # Emergency fallback: attempt one more reset with basic parameters
            try:
                raw_obs, info = self.env.reset()
                obs = self.get_observation(raw_obs, is_reset=True)
                # Ensure safe state
                obs["state"] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
                print(f"Emergency fallback reset completed with safe state")
                return obs, info
            except Exception as fallback_error:
                print(f"Emergency fallback also failed: {fallback_error}")
                raise RuntimeError(f"Complete reset failure: {e}")

    def step(self, action):
        """Execute action and return observation, reward, done, info"""
        # Unnormalize action if needed
        if self.normalize:
            action = self.unnormalize_action(action)
        # Execute action
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        # Get processed observation (use cached tokenization)
        obs = self.get_observation(raw_obs, is_reset=False)
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
                
                # Critical: Check for NaN/inf values in position and quaternion
                if np.any(~np.isfinite(position)) or np.any(~np.isfinite(quaternion)):
                    print(f"Warning: Invalid pose values detected - position: {position}, quaternion: {quaternion}")
                    raise ValueError("Invalid pose values")
                
                # Validate quaternion norm
                quat_norm = np.linalg.norm(quaternion)
                if not (0.9 < quat_norm < 1.1):  # Allow some numerical tolerance
                    print(f"Warning: Invalid quaternion norm {quat_norm}, normalizing")
                    quaternion = quaternion / (quat_norm + 1e-8)  # Normalize with epsilon for safety
                
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
                
                # Ensure gripper width is valid
                if not np.isfinite(gripper_width):
                    print(f"Warning: Invalid gripper width {gripper_width}, using default")
                    gripper_width = 0.5
                
                # Combine into eef_pos format: [x, y, z, qw, qx, qy, qz, gripper]
                eef_pos = np.concatenate([
                    position,  # [x, y, z]
                    quaternion,  # [qw, qx, qy, qz] (wxyz format)
                    [gripper_width]  # gripper openness
                ])
                
                # Final check for NaN/inf values
                if np.any(~np.isfinite(eef_pos)):
                    print(f"Warning: Final eef_pos contains invalid values: {eef_pos}")
                    raise ValueError("Invalid final state")
                
                return eef_pos.astype(np.float32)
                
            else:
                # Fallback: try to get from observation
                if 'agent' in raw_obs:
                    agent_obs = raw_obs['agent']
                    if 'qpos' in agent_obs and 'qvel' in agent_obs:
                        qpos = agent_obs['qpos']
                        # Use first 7 elements as rough approximation
                        if len(qpos) >= 7:
                            fallback_state = qpos[:8].astype(np.float32)  # Include gripper
                            # Check for valid values
                            if np.any(~np.isfinite(fallback_state)):
                                raise ValueError("Invalid fallback state")
                            return fallback_state
                
                # Last resort: return safe default
                print(f"Warning: Could not extract robot state, returning safe default")
                raise ValueError("Could not extract valid state")
                
        except Exception as e:
            print(f"Error extracting robot state: {e}")
            # Return safe default state with valid quaternion
            safe_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
            print(f"Using safe default state: {safe_state}")
            return safe_state

    def preprocess_robot_state_for_pi0(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Convert robot state to PI0 format using robot-specific preprocessing
        (次元拡張以外の処理のみ - 8次元のまま返す)
        Args:
            raw_state: [8] array with [x, y, z, qw, qx, qy, qz, gripper]
        Returns:
            processed_state: [8] array in PI0-compatible format
        """
        try:
            # Check for NaN/inf in input
            if not np.all(np.isfinite(raw_state)):
                print(f"Warning: Invalid values in raw_state: {raw_state}")
                # Use safe default state
                return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
            
            if self.robot_type == "widowx":
                # WidowX preprocessing: xyz + rpy + gripper_openness (8次元のまま)
                try:
                    # Import geometry functions (matching reference implementation)
                    
                    # Extract position and quaternion
                    position = raw_state[:3]  # xyz
                    quat_wxyz = raw_state[3:7]  # [qw, qx, qy, qz]
                    gripper_openness = raw_state[7]  # gripper openness
                    
                    # Convert quaternion to rotation matrix with numerical stability
                    try:
                        rm_bridge = quat2mat(quat_wxyz)
                        
                        # Apply default rotation transformation for WidowX Bridge
                        # EE pose in Bridge data was relative to a top-down pose
                        rm_final = rm_bridge @ self.default_rot.T
                        
                        # Convert to Euler angles with numerical stability
                        rpy_bridge_converted = mat2euler(rm_final)
                        
                        # Check for valid Euler angles
                        if not np.all(np.isfinite(rpy_bridge_converted)):
                            print("Invalid Euler angles computed, using zero rotation")
                            rpy_bridge_converted = np.array([0.0, 0.0, 0.0])
                        
                    except Exception as e:
                        print(f"Quaternion to Euler conversion failed: {e}, using zero rotation")
                        rpy_bridge_converted = np.array([0.0, 0.0, 0.0])
                    
                    # Ensure gripper value is valid
                    if not np.isfinite(gripper_openness):
                        print(f"Invalid gripper value {gripper_openness}, using 0.5")
                        gripper_openness = 0.5
                    
                    # Create processed state: [xyz, rpy, pad, gripper_openness] (8次元)
                    processed_state = np.concatenate([
                        position,  # [x, y, z]
                        rpy_bridge_converted,  # [roll, pitch, yaw]
                        np.zeros(1),  # pad
                        [gripper_openness],  # gripper openness
                    ])
                    
                    # Final check for NaN values
                    if not np.all(np.isfinite(processed_state)):
                        print(f"Final processed_state contains invalid values, using zeros")
                        processed_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
                    
                    return processed_state.astype(np.float32)
                    
                except Exception as e:
                    print(f"Error processing WidowX state: {e}, using safe state")
                    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
                    
            elif self.robot_type == "google_robot":
                # Google Robot preprocessing: xyz + quat_xyzw + gripper_closedness (8次元)
                try:
                    # Extract position and quaternion
                    position = raw_state[:3]  # xyz
                    quat_wxyz = raw_state[3:7]  # [qw, qx, qy, qz]
                    gripper_width = raw_state[7]  # gripper openness (0=closed, 1=open)
                    
                    # Validate quaternion norm
                    quat_norm = np.linalg.norm(quat_wxyz)
                    if quat_norm < 1e-6:
                        print(f"Zero quaternion detected, using identity quaternion")
                        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
                    elif not (0.9 < quat_norm < 1.1):
                        # print(f"Normalizing quaternion with norm {quat_norm}")
                        quat_wxyz = quat_wxyz / (quat_norm + 1e-8)
                    
                    # Convert wxyz to xyzw format
                    quat_xyzw = np.roll(quat_wxyz, -1)  # [qx, qy, qz, qw]
                    
                    # Ensure gripper value is valid
                    if not np.isfinite(gripper_width):
                        print(f"Invalid gripper value {gripper_width}, using 0.5")
                        gripper_width = 0.5
                    
                    # Convert gripper openness to closedness
                    gripper_closedness = 1 - gripper_width
                    
                    # Create processed state: [xyz, quat_xyzw, gripper_closedness] (8次元)
                    processed_state = np.concatenate([
                        position,  # [x, y, z]
                        quat_xyzw,  # [qx, qy, qz, qw]
                        [gripper_closedness],  # gripper closedness
                    ])
                    
                    # Final check for NaN values
                    if not np.all(np.isfinite(processed_state)):
                        print(f"Final processed_state contains invalid values, using zeros")
                        processed_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5], dtype=np.float32)
                    
                    return processed_state.astype(np.float32)
                    
                except Exception as e:
                    print(f"Error processing Google Robot state: {e}, using safe state")
                    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5], dtype=np.float32)
            else:
                print(f"Unknown robot_type: {self.robot_type}, using original state")
                return raw_state.astype(np.float32)
                
        except Exception as e:
            print(f"Error in preprocess_robot_state_for_pi0: {e}")
            return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    def postprocess_pi0_action_for_env(self, pi0_actions: np.ndarray) -> np.ndarray:
        """
        Convert PI0 action format to environment-compatible format
        Args:
            pi0_actions: [horizon_steps, action_dim] or [action_dim] array from PI0
        Returns:
            env_actions: [horizon_steps, action_dim] array for environment
        """
        try:
            # Ensure input is numpy array
            if isinstance(pi0_actions, torch.Tensor):
                pi0_actions = pi0_actions.cpu().numpy()
            
            # Handle single action case
            if pi0_actions.ndim == 1:
                pi0_actions = pi0_actions[None, :]  # [action_dim] -> [1, action_dim]
            
            horizon_steps, action_dim = pi0_actions.shape
            env_actions = []
            
            for step in range(horizon_steps):
                pi0_action = pi0_actions[step]  # [action_dim]
                
                if self.robot_type == "widowx":
                    # WidowX processing: similar to pi0_or_fast.py
                    env_action = self._postprocess_widowx_action(pi0_action)
                elif self.robot_type == "google_robot":
                    # Google Robot processing: similar to pi0_or_fast.py
                    env_action = self._postprocess_google_robot_action(pi0_action)
                else:
                    # Fallback: use action as-is
                    env_action = np.clip(pi0_action, -1.0, 1.0)
                
                env_actions.append(env_action)
            
            return np.array(env_actions, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in postprocess_pi0_action_for_env: {e}")
            # Fallback: return clipped actions
            if pi0_actions.ndim == 1:
                pi0_actions = pi0_actions[None, :]
            return np.clip(pi0_actions, -1.0, 1.0).astype(np.float32)

    def _postprocess_widowx_action(self, pi0_action: np.ndarray) -> np.ndarray:
        """
        Convert PI0 action to WidowX environment format
        Based on pi0_or_fast.py widowx_bridge processing
        """
        try:
            # PI0 action format: [world_vector(3), rotation_delta(3), open_gripper(1)]
            world_vector = pi0_action[:3]
            rotation_delta = pi0_action[3:6] 
            open_gripper = pi0_action[6] if len(pi0_action) > 6 else 0.0
            
            # Apply action scale (similar to pi0_or_fast.py)
            action_scale = 1.0  # Can be made configurable
            world_vector = world_vector * action_scale
            
            # Convert rotation delta to axis-angle format
            from transforms3d.euler import euler2axangle
            roll, pitch, yaw = rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle * action_scale
            
            # Convert gripper action: PI0 outputs [0,1] (0=close, 1=open)
            # WidowX expects [-1,1] format
            gripper_action = 2.0 * (open_gripper > 0.5) - 1.0
            
            # Combine into environment action format
            env_action = np.concatenate([
                world_vector,           # [3] - xyz translation
                action_rotation_axangle, # [3] - axis-angle rotation
                [gripper_action]        # [1] - gripper
            ])
            
            return np.clip(env_action, -1.0, 1.0).astype(np.float32)
            
        except Exception as e:
            print(f"Error processing WidowX action: {e}")
            # Fallback: return clipped original action
            return np.clip(pi0_action[:7], -1.0, 1.0).astype(np.float32)

    def _postprocess_google_robot_action(self, pi0_action: np.ndarray) -> np.ndarray:
        """
        Convert PI0 action to Google Robot environment format
        Based on pi0_or_fast.py google_robot processing
        """
        try:
            # PI0 action format: [world_vector(3), rotation_delta(3), open_gripper(1)]
            world_vector = pi0_action[:3]
            rotation_delta = pi0_action[3:6]
            open_gripper = pi0_action[6] if len(pi0_action) > 6 else 0.0
            
            # Apply action scale
            action_scale = 1.0  # Can be made configurable
            world_vector = world_vector * action_scale
            
            # Convert rotation delta to axis-angle format
            from transforms3d.euler import euler2axangle
            roll, pitch, yaw = rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle * action_scale
            
            # Google Robot gripper processing (based on pi0_or_fast.py)
            # Uses relative gripper action with sticky behavior
            # For simplicity, we'll use direct conversion here
            # Note: Full sticky gripper logic would require state tracking
            gripper_action = open_gripper  # Keep as [0,1] range for now
            
            # Combine into environment action format
            env_action = np.concatenate([
                world_vector,           # [3] - xyz translation  
                action_rotation_axangle, # [3] - axis-angle rotation
                [gripper_action]        # [1] - gripper
            ])
            
            return np.clip(env_action, -1.0, 1.0).astype(np.float32)
            
        except Exception as e:
            print(f"Error processing Google Robot action: {e}")
            # Fallback: return clipped original action
            return np.clip(pi0_action[:7], -1.0, 1.0).astype(np.float32)

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
    print(f"State shape: {obs['state'].shape}")
    
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
    print(f"State shape: {obs['state'].shape}")
    wrapper.close()
    print("Test completed!")
