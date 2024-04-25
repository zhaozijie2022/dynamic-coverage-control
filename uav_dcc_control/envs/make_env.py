import importlib
import os

import numpy as np
from envs.wrappers import DummyVecEnv, SubprocVecEnv


def make_env(cfg, **kwargs):
    if kwargs is not None:
        for k, v in kwargs.items():
            setattr(cfg, k, v)

    def get_env_fn(rank, **kwargs):
        def init_env():
            if "uav_dcc" in cfg.env_file:
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
                    **kwargs,
                )

                if cfg.seed is not None:
                    env.env.seed(cfg.seed + 1024 * rank)
                return env
            else:
                raise NotImplementedError("env_file: %s not found" % cfg.env_file)

        return init_env

    # if "uav_dcc" in cfg.env_file:
    #     # TODO 在windows中并行环境随机生成的pos_pois各不相同, 算法无法收敛
    #     if os.name == 'nt':
    #         pos_pois = np.random.uniform(-1, 1, (cfg.num_pois, 2))
    #     else:
    #         pos_pois = None

    if cfg.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(cfg.n_rollout_threads)])
