import copy
import glob
import os
import time
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import reduce
from statistics import stdev

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from .envs import make_env
from .model import Policy
from .storage import RolloutStorage
from .visualize import visdom_plot

import agents.algo as algo


def silent_print(isSilent, msg):
    if not isSilent:
        print(msg)

class Agents:
    envs = None

    def __init__(self, args, game, level):
        self.args = args
        self.game = game
        self.level = level

        assert self.args.algo in ['a2c', 'ppo', 'acktr']
        if self.args.recurrent_policy:
            assert self.args.algo in ['a2c', 'ppo'], \
                'Recurrent policy is not implemented for ACKTR'

        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        try:
            os.makedirs(self.args.log_dir)
        except OSError:
            files = glob.glob(os.path.join(self.args.log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)

        torch.set_num_threads(1)

        if self.args.vis:
            from visdom import Visdom
            self.viz = Visdom(port=self.args.port)
            self.win = None

        self.envs = [make_env(self.args.game, self.args.level, self.args.env_name, self.args.seed, i, self.args.log_dir, self.args.add_timestep, self.args.remote_env, self.args.record)
                     for i in range(self.args.num_processes)]

        if self.args.num_processes > 1:
            self.envs = SubprocVecEnv(self.envs)
        else:
            self.envs = DummyVecEnv(self.envs)

        if len(self.envs.observation_space.shape) == 1:
            self.envs = VecNormalize(self.envs)

        obs_shape = self.envs.observation_space.shape
        obs_shape = (obs_shape[0] * self.args.num_stack, *obs_shape[1:])

        self.actor_critic = Policy(
            obs_shape, self.envs.action_space, self.args.recurrent_policy)

        if self.envs.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = self.envs.action_space.shape[0]

        if self.args.cuda:
            self.actor_critic.cuda()

        if self.args.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(self.actor_critic, self.args.value_loss_coef,
                                        self.args.entropy_coef, lr=self.args.lr,
                                        eps=self.args.eps, alpha=self.args.alpha,
                                        max_grad_norm=self.args.max_grad_norm)
        elif self.args.algo == 'ppo':
            self.agent = algo.PPO(self.actor_critic, self.args.clip_param, self.args.ppo_epoch, self.args.num_mini_batch,
                                  self.args.value_loss_coef, self.args.entropy_coef, lr=self.args.lr,
                                  eps=self.args.eps,
                                  max_grad_norm=self.args.max_grad_norm)
        elif self.args.algo == 'acktr':
            self.agent = algo.A2C_ACKTR(self.actor_critic, self.args.value_loss_coef,
                                        self.args.entropy_coef, acktr=True)

        self.rollouts = RolloutStorage(self.args.num_steps, self.args.num_processes,
                                       obs_shape, self.envs.action_space, self.actor_critic.base.state_size)
        self.current_obs = torch.zeros(self.args.num_processes, *obs_shape)

    def update_current_obs(self, obs):
        shape_dim0 = self.envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if self.args.num_stack > 1:
            self.current_obs[:, :-
                             shape_dim0] = self.current_obs[:, shape_dim0:]
        self.current_obs[:, -shape_dim0:] = obs

    def init_training(self):
        # print("#######")
        # print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see self.envs.py) or visdom plot to get true rewards")
        # print("#######")
        silent_print(self.args.silent, "Training...")
        sys.stdout.flush()
        obs = self.envs.reset()
        self.update_current_obs(obs)
        self.rollouts.observations[0].copy_(self.current_obs)

        # These variables are used to compute average rewards for all processes.
        self.episode_rewards = torch.zeros([self.args.num_processes, 1])
        self.final_rewards = torch.zeros([self.args.num_processes, 1])

        if self.args.cuda:
            self.current_obs = self.current_obs.cuda()
            self.rollouts.cuda()

        saved_state_path = os.path.join(
            f"./trained_models/{self.args.algo}", self.args.env_name + ".pt")

        if os.path.exists(saved_state_path):
            silent_print(self.args.silent, "Loading weights...")
            saved_state = torch.load(saved_state_path)
            self.actor_critic.load_state_dict(saved_state['state_dict'])
            self.agent.optimizer.load_state_dict(saved_state['optimizer'])
        
        self.final_scores = []
        self.x_vals = np.array([])
        self.y_vals = np.array([])

        self.start = time.time()

    def compute_steps(self):
        for step in range(self.args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = self.actor_critic.act(
                    self.rollouts.observations[step],
                    self.rollouts.states[step],
                    self.rollouts.masks[step])
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = self.envs.step(cpu_actions)
            reward = torch.from_numpy(
                np.expand_dims(np.stack(reward), 1)).float()
            self.episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            self.final_rewards *= masks
            self.final_rewards += (1 - masks) * self.episode_rewards

            self.episode_rewards *= masks

            for i in range(len(masks)):
                    if masks[i] == 0.0:
                        float_reward = float(self.final_rewards[i][0])
                        self.final_scores.append(float_reward)

            if self.args.cuda:
                masks = masks.cuda()

            if self.current_obs.dim() == 4:
                self.current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                self.current_obs *= masks

            self.update_current_obs(obs)
            self.rollouts.insert(self.current_obs, states.data, action.data,
                                 action_log_prob.data, value.data, reward, masks)

        with torch.no_grad():
            next_value = self.actor_critic.get_value(self.rollouts.observations[-1],
                                                     self.rollouts.states[-1],
                                                     self.rollouts.masks[-1]).detach()

        self.rollouts.compute_returns(
            next_value, self.args.use_gae, self.args.gamma, self.args.tau)

        self.value_loss, self.action_loss, self.dist_entropy = self.agent.update(
            self.rollouts)

        self.rollouts.after_update()

        return self.actor_critic.named_parameters(), self.final_rewards.mean()

    def save_model(self):
        save_path = os.path.join(self.args.save_dir, self.args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        state = {
            'state_dict': self.actor_critic.state_dict(),
            'optimizer': self.agent.optimizer.state_dict()
        }

        if self.args.full_set or self.args.testing:
            filename = f"{self.args.env_name}-{self.args.level}.pt"
        else:
            filename = self.args.env_name + ".pt"
            
        torch.save(state, os.path.join(
            save_path, filename))

    def print_progress(self, curr):
        end = time.time()
        total_num_steps = (
                    curr + 1) * self.args.num_processes * self.args.num_steps
        # print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
        #       format(curr, total_num_steps,
        #              int(total_num_steps / (end - self.start)),
        #              self.final_rewards.mean(),
        #              self.final_rewards.median(),
        #              self.final_rewards.min(),
        #              self.final_rewards.max(), self.dist_entropy,
        #              self.value_loss, self.action_loss))
        silent_print(self.args.silent, "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
                      format(curr, total_num_steps,
                             int(total_num_steps / (end - self.start)),
                             self.final_rewards.mean(),
                             self.final_rewards.median(),
                             self.final_rewards.min(),
                             self.final_rewards.max()))
        sys.stdout.flush()

    def plot(self, curr):
        # try:
        #     # Sometimes monitor doesn't properly flush the outputs
        #     self.win = visdom_plot(self.viz, self.win, self.args.log_dir, self.args.env_name,
        #                            self.args.algo, self.args.num_frames)
        # except IOError:
        #     pass
        plot_title = "{} - {} - {}".format(self.game,
                                                self.level,
                                                self.args.algo)
        total_num_steps = (
            curr + 1) * self.args.num_processes * self.args.num_steps

        self.x_vals = np.append(self.x_vals, [total_num_steps])
        self.y_vals = np.append(self.y_vals, [self.final_rewards.mean()])

        opts = dict(
            width=400,
            height=300,
            xlabel='Timesteps',
            ylabel='Score',
            title=plot_title,
            xtickmax=self.args.num_frames
        )
        
        if self.win:
            self.win = self.viz.line(self.y_vals, self.x_vals, win=self.win, update='append', opts=opts)
        else:
            self.win = self.viz.line(self.y_vals, self.x_vals, opts=opts)

        self.x_vals = np.array([])
        self.y_vals = np.array([])

    def substitute_params(self, params):
        self.actor_critic.load_state_dict(params)
