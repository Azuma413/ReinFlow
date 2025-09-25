# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Multi-step wrapper. 
Allow executing multiple environmnt steps. 
Returns stacked observation and optionally stacked previous action.

Modified from 
https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/gym_util/multistep_wrapper.py

TODO: allow cond_steps != img_cond_steps (should be implemented in training scripts, not here)
"""

import gymnasium as gym
from typing import Optional
from gymnasium import spaces
import numpy as np
from collections import defaultdict, deque
from typing import Any, Dict, List


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )


def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Text):
        # Text spaces don't need to be repeated (they're not stackable)
        return space
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method="max"):
    if method == "max":
        # equivalent to any
        return np.max(data)
    elif method == "min":
        # equivalent to all
        return np.min(data)
    elif method == "mean":
        return np.mean(data)
    elif method == "sum":
        return np.sum(data)
    else:
        raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    """Apply padding"""
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStep(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        n_obs_steps: int = 1,
        n_action_steps: int = 1,
        max_episode_steps: Optional[int] = None,
        reward_agg_method: str = "sum",
        prev_action: bool = True,
        reset_within_step: bool = False,
        pass_full_observations: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(env)
        self._single_action_space = env.action_space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.prev_action = prev_action
        self.reset_within_step = reset_within_step
        self.pass_full_observations = pass_full_observations
        self.verbose = verbose
        self.last_options = None

    # ###################### resetメソッドを修正 ######################
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, Dict[str, Any]]:
        """Resets the environment to a starting state."""
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        if self.prev_action:
            self.action = deque(
                [self._single_action_space.sample()], maxlen=self.n_obs_steps
            )
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        self._add_info(info)
        processed_obs = self._get_obs(self.n_obs_steps)
        self.cnt = 0
        processed_info = dict_take_last_n(self.info, self.n_obs_steps)
        self.last_options = options
        return processed_obs, processed_info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics."""
        if action.ndim == 1:
            action = action[None]
        episode_terminated = False
        episode_truncated = False
        for act_step, act in enumerate(action):
            self.cnt += 1
            if episode_terminated or episode_truncated:
                break
            observation, reward, terminated, truncated, info = self.env.step(act)
            self.obs.append(observation)
            self.action.append(act)
            self.reward.append(reward)
            if terminated:
                episode_terminated = True
            if truncated:
                episode_truncated = True
            self.done.append(terminated or truncated)
            self._add_info(info)
        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        info = dict_take_last_n(self.info, self.n_obs_steps)
        if self.pass_full_observations:
            info["full_obs"] = self._get_obs(act_step + 1)
        if self.reset_within_step and (episode_terminated or episode_truncated):
            if episode_truncated:
                info["final_obs"] = observation
            self.reset(options=self.last_options)
            self.verbose and print("Reset env within wrapper.")
        self.reward = list()
        self.done = list()
        return observation, reward, episode_terminated, episode_truncated, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert len(self.obs) > 0
        if isinstance(self._observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self._observation_space, spaces.Dict):
            result = dict()
            for key in self._observation_space.keys():
                obs_history_for_key = [obs[key] for obs in self.obs]
                # Check if this is a text/string observation or other non-numeric data
                if isinstance(self._observation_space[key], spaces.Text) or key in ['instruction', 'robot_type', 'scene_info']:
                    # For text/string data, just return the latest observation (no stacking)
                    result[key] = obs_history_for_key[-1]
                else:
                    # For numeric data (images, states), stack as usual
                    result[key] = stack_last_n_obs(obs_history_for_key, n_steps)
            return result
        else:
            raise RuntimeError("Unsupported space type")

    def get_prev_action(self, n_steps=None):
        if n_steps is None:
            n_steps = self.n_obs_steps - 1  # exclude current step
        assert len(self.action) > 0
        return stack_last_n_obs(self.action, n_steps)

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def render(self, **kwargs):
        """Not the best design"""
        return self.env.render(**kwargs)


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json
    os.environ["MUJOCO_GL"] = "egl"
    cfg = OmegaConf.load("cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp_img.yaml")
    shape_meta = cfg["shape_meta"]
    from env.gym_utils.wrapper.simplerenv_image import SimplerEnvImageWrapper
    import matplotlib.pyplot as plt
    wrappers = cfg.env.wrappers
    obs_modality_dict = {
        "low_dim": (
            wrappers.robomimic_image.low_dim_keys
            if "robomimic_image" in wrappers
            else wrappers.robomimic_lowdim.low_dim_keys
        ),
        "rgb": (
            wrappers.robomimic_image.image_keys
            if "robomimic_image" in wrappers
            else None
        ),
    }
    if obs_modality_dict["rgb"] is None:
        obs_modality_dict.pop("rgb")
    with open(cfg.robomimic_env_cfg_path, "r") as f:
        env_meta = json.load(f)
    wrapper = MultiStep(
            env=SimplerEnvImageWrapper(
            env_name="GraspSingleRandomObjectInScene-v0",
            scene_name="google_pick_coke_can_1_v4", 
            robot_type="google_robot",
            enable_random_scene=False,
            max_episode_steps=50
        ),
        n_obs_steps=1,
        n_action_steps=1,
    )
    wrapper.seed(0)
    obs, _ = wrapper.reset()
    print(obs.keys())
    print("obs['rgb']:", obs['rgb'].shape)
    # img = wrapper.render()
    wrapper.close()
    # plt.imshow(img)
    # plt.savefig("test.png")
