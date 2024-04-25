import copy
import datetime
import json
import os
import time
import imageio
from argparse import Namespace

import gym
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from buffer.shared_buffer import SharedReplayBuffer
# from buffer.separated_buffer import SeparatedReplayBuffer
from envs.make_env import make_env
from utils import util as utl, pytorch_utils as ptu


class Learner:
    def __init__(self, cfg: DictConfig):
        self.cfg = Namespace(**OmegaConf.to_container(cfg, resolve=True))
        utl.seed(self.cfg.seed)  # set seed for random, torch and np

        # 1. env
        self.train_envs = make_env(cfg=self.cfg)
        print("initial train envs: %s, done" % cfg.save_name)
        self.n_agents = self.cfg.num_agents
        self.max_ep_len = self.cfg.max_ep_len

        self.obs_dim_n = [self.train_envs.observation_space[i].shape[0] for i in range(self.n_agents)]
        self.action_dim_n = [self.train_envs.action_space[i].n if isinstance(self.train_envs.action_space[i], gym.spaces.Discrete)
                             else self.train_envs.action_space[i].shape[0] for i in range(self.n_agents)]
        self.cfg.action_dim_n = self.action_dim_n
        self.cfg.obs_dim_n = self.obs_dim_n

        # 2. rl agent
        self.use_centralized_V = self.cfg.use_centralized_V
        self.use_obs_instead_of_state = self.cfg.use_obs_instead_of_state
        self.algo_hidden_size = self.cfg.algo_hidden_size
        self.recurrent_N = self.cfg.recurrent_N
        if self.use_centralized_V:
            self.share_observation_space = self.train_envs.share_observation_space[0]
        else:
            self.share_observation_space = self.train_envs.observation_space[0]

        from algos.mappo import MAPPOTrainer, MAPPOPolicy
        self.policy = MAPPOPolicy(
            self.cfg,
            self.train_envs.observation_space[0],
            self.share_observation_space,
            self.train_envs.action_space[0], )
        self.trainer = MAPPOTrainer(
            cfg=self.cfg,
            policy=self.policy,
        )
        print("initial agent: shared mappo, done")

        self.rl_buffer = SharedReplayBuffer(
            self.cfg,
            self.train_envs.observation_space[0],
            self.share_observation_space,
            self.train_envs.action_space[0]
        )
        print("initial rl buffer, done")

        if self.cfg.n_eval_rollout_threads > 0:
            test_cfg = copy.deepcopy(self.cfg)
            test_cfg.n_rollout_threads = self.cfg.n_eval_rollout_threads
            self.test_envs = make_env(test_cfg)
            self.test_buffer = SharedReplayBuffer(
                test_cfg,
                self.train_envs.observation_space[0],
                self.share_observation_space,
                self.train_envs.action_space[0]
            )
            print("initial test envs and buffer, done")

        if self.cfg.n_render_rollout_threads > 0:
            assert self.cfg.n_render_rollout_threads == 1
            render_cfg = copy.deepcopy(self.cfg)
            render_cfg.n_rollout_threads = self.cfg.n_render_rollout_threads
            self.render_envs = make_env(render_cfg)
            self.render_buffer = SharedReplayBuffer(
                render_cfg,
                self.train_envs.observation_space[0],
                self.share_observation_space,
                self.train_envs.action_space[0]
            )
            print("initial render env and buffer, done")

        # 4. 读取cfg中train相关的参数
        self.use_linear_lr_decay = self.cfg.use_linear_lr_decay
        self.n_iters = self.cfg.n_iters
        self.n_rollout_threads = self.cfg.n_rollout_threads
        self.n_eval_rollout_threads = self.cfg.n_eval_rollout_threads
        self.eval_interval = self.cfg.eval_interval
        self.render_interval = self.cfg.render_interval
        self.save_gifs = self.cfg.save_gifs

        # 5. 存储/读取model, rl_buffer
        self.is_save_model = self.cfg.save_model
        self.save_interval = self.cfg.save_interval
        env_dir = self.cfg.save_name

        if self.cfg.load_model:
            self.load_model(self.cfg.load_model_path)
            print("!!!!!Note: Load model, done!!!!!")

        date_dir = datetime.datetime.now().strftime("%m%d_%H%M_")
        seed_dir = 'sd{}'.format(self.cfg.seed)
        self.expt_name = date_dir + seed_dir
        if self.is_save_model:
            self.output_path = str(os.path.join(self.cfg.main_save_path, env_dir, self.expt_name))
            self.cfg.output_path = self.output_path
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, "config.json"), 'w') as f:
                config_json = vars(self.cfg)
                json.dump(config_json, f, indent=4)

        # 6. 读取cfg中的log相关的参数, 并创建logger
        self.is_log_wandb = self.cfg.log_wandb
        self.log_interval = self.cfg.log_interval
        if self.is_log_wandb:
            wandb.init(project=env_dir, group="mappo",
                       name=self.expt_name, config=config_json, )
        print("initial learner, done")
        self._start_time = time.time()
        self._check_time = time.time()

    def train(self):
        self.warmup(self.rl_buffer, self.train_envs)

        for iter_ in range(1, self.n_iters + 1):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(iter_, self.n_iters)

            rollout_info = self.rollout(self.rl_buffer, self.train_envs)

            rl_train_info = self.rl_update()

            if iter_ % self.eval_interval == 0:
                test_rollout_info = self.rollout(self.test_buffer, self.test_envs)
            else:
                test_rollout_info = {}

            if iter_ % self.render_interval == 0:
                self.rollout(self.render_buffer, self.render_envs, is_render=True, iter_=iter_)

            if iter_ % self.log_interval == 0:
                self.log(
                    iter_=iter_,
                    rollout_info=rollout_info,
                    rl_train_info=rl_train_info,
                    test_rollout_info=test_rollout_info,
                )

            if self.is_save_model and (iter_ % self.save_interval == 0):
                save_path = os.path.join(self.output_path, 'models_%d.pt' % iter_)
                if self.is_save_model:
                    os.makedirs(save_path, exist_ok=True)
                    self.save_model(save_path)
                    print("model saved in %s" % save_path)

        if self.is_log_wandb:
            wandb.finish()
            print("wandb run has finished")
            print("")

        self.train_envs.close()
        print("multi processing train_envs have been closed")
        if self.cfg.n_eval_rollout_threads > 0:
            self.test_envs.close()
            print("eval_envs have been closed")

    # region functions 4 collect
    def rollout(self, r_buffer, r_envs, is_render=False, iter_=0):
        self.warmup(r_buffer, r_envs)
        _rew = 0.
        _sr = np.array([0. for _ in range(r_buffer.n_rollout_threads)])
        frames = []

        for cur_step in range(self.max_ep_len):
            (values, actions, action_log_probs, rnn_states,
             rnn_states_critic, actions_env) = self.collect(cur_step, r_buffer)
            obs, rewards, dones, infos = r_envs.step(actions_env)
            data = (obs, rewards, dones, infos, values, actions,
                    action_log_probs, rnn_states, rnn_states_critic,)
            self.insert(data, r_buffer)
            _rew += np.mean(rewards)
            sr = np.array([info["coverage_rate"] for info in infos])
            _sr = np.max(np.vstack((_sr, sr)), axis=0)

            if is_render:
                r_envs.render()
                time.sleep(0.025)
                if self.save_gifs:
                    frame = r_envs.render("rgb_array")
                    frames.append(frame[0][0])  # 并行环境的list, render本身返回的也是list

        self.compute(r_buffer)

        if self.is_save_model and is_render and self.save_gifs:
            imageio.mimsave(
                uri=os.path.join(self.output_path, "models_%d.gif" % iter_),
                ims=frames,
                format='GIF',
                duration=0.1
            )
        return {
            "reward": _rew,
            "coverage_rate": np.mean(_sr),
        }

    def warmup(self, r_buffer, r_envs):
        obs = r_envs.reset()
        if self.use_centralized_V:
            share_obs = obs.reshape(r_buffer.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
        else:
            share_obs = obs

        r_buffer.share_obs[0] = share_obs.copy()
        r_buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, cur_step, r_buffer):
        self.trainer.prep_rollout()

        (value, action, action_log_prob, rnn_states, rnn_states_critic) = \
            self.trainer.policy.get_actions(
                np.concatenate(r_buffer.share_obs[cur_step]),
                np.concatenate(r_buffer.obs[cur_step]),
                np.concatenate(r_buffer.rnn_states[cur_step]),
                np.concatenate(r_buffer.rnn_states_critic[cur_step]),
                np.concatenate(r_buffer.masks[cur_step]),
            )  # [n_envs, n_agents, dim]

        values = np.array(np.split(ptu.get_numpy(value), r_buffer.n_rollout_threads))
        actions = np.array(np.split(ptu.get_numpy(action), r_buffer.n_rollout_threads))
        action_log_probs = np.array(np.split(ptu.get_numpy(action_log_prob), r_buffer.n_rollout_threads))
        rnn_states = np.array(np.split(ptu.get_numpy(rnn_states), r_buffer.n_rollout_threads))
        rnn_states_critic = np.array(np.split(ptu.get_numpy(rnn_states_critic), r_buffer.n_rollout_threads))
        # [n_envs, n_agents, dim]

        if self.train_envs.action_space[0].__class__.__name__ == "Discrete":
            actions_env = np.eye(self.train_envs.action_space[0].n)[actions.reshape(-1)].reshape(*actions.shape[:2], -1)
        else:
            actions_env = actions.copy()

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data, r_buffer):
        (obs, rewards, dones, infos, values, actions,
         action_log_probs, rnn_states, rnn_states_critic,) = data

        rnn_states[dones] = np.zeros(
            (dones.sum(), self.recurrent_N, self.algo_hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones] = np.zeros(
            (dones.sum(), self.recurrent_N, self.algo_hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((r_buffer.n_rollout_threads, self.n_agents, 1), dtype=np.float32)  # 用于屏蔽envs中已经结束的episodes
        masks[dones] = np.zeros((dones.sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(r_buffer.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
        else:
            share_obs = obs

        r_buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                        actions, action_log_probs, values, rewards, masks, )

    @torch.no_grad()
    def compute(self, r_buffer):
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            cent_obs=np.concatenate(r_buffer.share_obs[-1]),
            rnn_states_critic=np.concatenate(r_buffer.rnn_states_critic[-1]),
            masks=np.concatenate(r_buffer.masks[-1])
        )
        next_values = np.array(np.split(ptu.get_numpy(next_values), r_buffer.n_rollout_threads))
        r_buffer.compute_returns(next_values, self.trainer.value_normalizer)

    # endregion

    # region functions 4 update
    def rl_update(self):
        self.trainer.prep_training()
        update_info = self.trainer.train(
            buffer=self.rl_buffer,
            update_actor=True,
        )
        self.rl_buffer.after_update()

        return update_info

    # endregion

    # region functions 4 log and s/l
    def log(self, iter_, **kwargs):
        if self.is_log_wandb:
            for key, value in kwargs.items():
                wandb.log(value, step=iter_)

        print("")
        print("******** iter: %d, iter_time: %.2fs, total_time: %.2fs" %
              (iter_, time.time() - self._check_time, time.time() - self._start_time))
        for key, value in kwargs.items():
            print("%s" % key + "".join([", %s: %.4f" % (k, v) for k, v in value.items()]))
        self._check_time = time.time()

    def save_model(self, save_path):
        self.trainer.save_model(save_path)

    def load_model(self, load_path):
        self.trainer.load_model(load_path)
    # endregion


