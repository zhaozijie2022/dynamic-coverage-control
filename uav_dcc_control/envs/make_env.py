import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
import importlib
from envs.wrappers import DummyVecEnv, SubprocVecEnv


def make_env(cfg):
    def get_env_fn(rank):
        def init_env():
            if "mpe" in cfg.env_file:
                env_file = importlib.import_module("envs." + cfg.env_file)
                Env = getattr(env_file, cfg.env_class)
                env = Env(
                    scenario=cfg.scenario_name,
                    num_agents=cfg.num_agents,
                    num_pois=cfg.num_pois,
                    max_ep_len=cfg.max_ep_len,
                    r_cover=cfg.r_cover,
                    r_comm=cfg.r_comm,
                    comm_r_scale=cfg.comm_r_scale,
                    comm_force_scale=cfg.comm_force_scale,
                )

                if cfg.seed is not None:
                    env.env.seed(cfg.seed + 1024 * rank)
                return env
            else:
                raise NotImplementedError("env_file: %s not found" % cfg.env_file)

        return init_env

    if cfg.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(cfg.n_rollout_threads)])
