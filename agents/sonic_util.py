"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np
import sys

from math import floor
from random import randrange

from retro_contest.local import make
from baselines.common.atari_wrappers import WarpFrame, FrameStack
import gym_remote.client as grc


def make_sonic_env(game, state, remote_env, stack=False, scale_rew=True, video_dir=''):
    """
    Create an environment with some standard wrappers.
    """
    if remote_env:
        env = grc.RemoteEnv('tmp/sock')
    else:
        env = make(game=game,
                   state=state,
                   bk2dir=video_dir)
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    env = ShortLife(env)
    return env


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP",
                   "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """

    def reward(self, reward):
        return reward * 0.01


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):  # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

class ShortLife(gym.Wrapper):
    def __init__(self, env):
        super(ShortLife, self).__init__(env)
        self.steps_done = 0
        self.max_steps = 50
        self.death_count = 0
        self.curr_run_reward = 0

    def reset(self, **kwargs):  # pylint: disable=E0202
        self.steps_done = 0
        self.curr_run_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self.curr_run_reward += rew
        self.steps_done += 1
        if self.steps_done == self.max_steps:
            self.death_count += int(floor(self.curr_run_reward))
            # increment = int(floor(self.curr_run_reward)) + self.death_count + 15
            random_max_step = randrange(self.death_count, self.death_count + 200)
            self.max_steps = min(random_max_step, 4500)
            self.reset()
            done = True
        return obs, rew, done, info