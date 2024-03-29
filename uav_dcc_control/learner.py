import datetime
import json
import os
import time
from argparse import Namespace

import gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from buffer.shared_buffer import SharedReplayBuffer
# from buffer.separated_buffer import SeparatedReplayBuffer
from envs.make_env import make_env
from utils import util as utl, pytorch_utils as ptu

class Learner:
    # region Learner Init
    def __init__(self, cfg: DictConfig):
        self.cfg = Namespace(**OmegaConf.to_container(cfg, resolve=True))
        utl.seed(self.cfg.seed)  # set seed for random, torch and np

        # 1. env & task
        self.envs = make_env(cfg=self.cfg)
        print("initial envs: %s, done" % cfg.save_name)
        # self.envs.reset_task(0)
        self.n_agents = self.cfg.num_agents
        # self.n_tasks = self.cfg.n_tasks
        self.max_ep_len = self.cfg.max_ep_len

        self.obs_dim_n = [self.envs.observation_space[i].shape[0] for i in range(self.n_agents)]
        self.action_dim_n = [self.envs.action_space[i].n if isinstance(self.envs.action_space[i], gym.spaces.Discrete)
                             else self.envs.action_space[i].shape[0] for i in range(self.n_agents)]
        self.cfg.action_dim_n = self.action_dim_n
        self.cfg.obs_dim_n = self.obs_dim_n

        # 2. rl agent
        self.use_centralized_V = self.cfg.use_centralized_V
        self.use_obs_instead_of_state = self.cfg.use_obs_instead_of_state
        self.algo_hidden_size = self.cfg.algo_hidden_size
        self.recurrent_N = self.cfg.recurrent_N
        if self.use_centralized_V:
            self.share_observation_space = self.envs.share_observation_space[0]
        else:
            self.share_observation_space = self.envs.observation_space[0]

        # self.agents = make_algo(cfg=self.cfg)
        from algos.mappo import MAPPOTrainer, MAPPOPolicy
        self.policy = MAPPOPolicy(
            self.cfg,
            self.envs.observation_space[0],
            self.share_observation_space,
            self.envs.action_space[0], )
        self.trainer = MAPPOTrainer(
            cfg=self.cfg,
            policy=self.policy,
        )
        print("initial agent: shared mappo, done")

        self.rl_buffer = SharedReplayBuffer(self.cfg,
                                            self.envs.observation_space[0],
                                            self.share_observation_space,
                                            self.envs.action_space[0])
        print("initial rl buffer, done")

        # 4. 读取cfg中train相关的参数
        self.use_linear_lr_decay = self.cfg.use_linear_lr_decay
        self.n_iters = self.cfg.n_iters
        self.n_rollout_threads = self.cfg.n_rollout_threads
        self.n_eval_rollout_threads = self.cfg.n_eval_rollout_threads
        self.eval_interval = self.cfg.eval_interval

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

        for iter_ in range(self.n_iters):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(iter_, self.n_iters)
            _rew = self.rollout(self.rl_buffer, self.envs)
            rollout_info = {"reward": _rew, "rl-collect-rps": _rew / self.max_ep_len}

            rl_train_info = self.rl_update()  # 此时的meta_tasks与采样时保持一致


            if (iter_ + 1) % self.log_interval == 0:
                self.log(iter_ + 1,
                         rollout_info=rollout_info,
                         rl_train_info=rl_train_info, )

            if self.is_save_model and (iter_ + 1) % self.save_interval == 0:
                save_path = os.path.join(self.output_path, 'models_%d.pt' % (iter_ + 1))
                if self.is_save_model:
                    os.makedirs(save_path, exist_ok=True)
                    self.save_model(save_path)
                    print("model saved in %s" % save_path)

        if self.is_log_wandb:
            wandb.finish()
            print("wandb run has finished")
            print("")

        self.envs.close()
        # self.dummy_envs.close()
        print("multi processing envs have been closed")
        print("")

    # region functions 4 collect

    def rollout(self, r_buffer, r_envs):

        _rew, _sr = 0., 0.
        self.warmup(r_buffer, r_envs)

        for cur_step in range(self.max_ep_len):

            (values, actions, action_log_probs, rnn_states,
             rnn_states_critic, actions_ae) = self.collect(cur_step, r_buffer)
            obs, rewards, dones, infos = r_envs.step(actions)
            data = (obs, rewards, dones, infos, values, actions,
                    action_log_probs, rnn_states, rnn_states_critic,)
            self.insert(data, r_buffer)
            _rew += np.mean(rewards)

        coverage_rate = np.array([info["coverage_rate"] for info in infos])
        # print(coverage_rate)
        print(coverage_rate.mean())
        self.compute(r_buffer)
        return _rew

    def warmup(self, r_buffer, r_envs):
        # r_envs.meta_reset_task(meta_tasks)
        obs = r_envs.reset()  # [env_num, agent_num, obs_dim]
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)  # [env_num, agent_num * obs_dim]
            share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)  # [ne, na, na*od]
        else:
            share_obs = obs
        # share_obs = obs.reshape(obs.shape[0], -1).copy()  # shape = [env_num, agent_num * obs_dim]

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

        values = np.array(np.split(ptu.get_numpy(value), self.n_rollout_threads))
        actions = np.array(np.split(ptu.get_numpy(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(ptu.get_numpy(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(ptu.get_numpy(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(ptu.get_numpy(rnn_states_critic), self.n_rollout_threads))
        # [n_envs, n_agents, dim]

        if self.envs.action_space[0].__class__.__name__ == "Discrete":
            actions_env = np.eye(self.envs.action_space[0].n)[actions.reshape(-1)].reshape(*actions.shape[:2], -1)
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
            share_obs = obs.reshape(self.n_rollout_threads, -1)
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
        next_values = np.array(np.split(ptu.get_numpy(next_values), self.n_rollout_threads))
        r_buffer.compute_returns(next_values, self.trainer.value_normalizer)


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

    def log(self, iter_, **kwargs):
        if self.is_log_wandb:
            for key, value in kwargs.items():
                wandb.log(value, step=iter_)

        print("")
        print("******** iter: %d, iter_time: %.2fs, total_time: %.2fs" %
              (iter_, time.time() - self._check_time, time.time() - self._start_time))
        # print("meta_tasks: ", meta_tasks)
        for key, value in kwargs.items():
            print("%s" % key + "".join([", %s: %.4f" % (k, v) for k, v in value.items()]))
        self._check_time = time.time()

    def save_model(self, save_path):
        self.trainer.save_model(save_path)

    def load_model(self, load_path):
        self.trainer.load_model(load_path)
