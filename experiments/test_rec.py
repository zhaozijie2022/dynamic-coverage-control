import os
import numpy as np
import pickle
import tensorflow as tf
from maddpg.lib4cover import make_env, get_trainers, parse_args
import maddpg.common.tf_util as U
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

max_ep_len = 80
num_eps = 50
lr, gma = 1e-2, 0.95


def test_rec(scenario_name, trainers, save_path):
    save_path += "test_rec/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        pkl = []
        with open(save_path + "test_rec_cover.pkl", "wb") as fp:
            pickle.dump(pkl, fp)
        with open(save_path + "test_rec_connect.pkl", "wb") as fp:
            pickle.dump(pkl, fp)

    # 创建环境
    env = make_env(scenario_name, r_cover=0.2, r_comm=0.4, comm_r_scale=0.9, comm_force_scale=0.0)
    obs_n = env.reset()
    episode_step = 0
    num_ep = 0

    episode_coverage = []
    episode_connectivity = [0]

    while True:
        episode_connectivity[-1] += env.world.connect
        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]

        new_obs_n, rew_n, done_n = env.step(action_n)
        episode_step += 1
        terminal = (episode_step >= 80) or all(done_n)
        obs_n = new_obs_n

        if terminal:
            num_ep += 1
            episode_connectivity[-1] /= episode_step
            episode_coverage.append(env.world.coverage_rate)
            episode_connectivity.append(0)
            episode_step = 0
            obs_n = env.reset()

        if num_ep >= num_eps:
            print("Test:  Cover: %.3f, Connect: %.3f"
                  % (float(np.mean(episode_coverage)),
                     float(np.mean(episode_connectivity))
                     ))
            with open(save_path + "test_rec_cover.pkl", "rb") as fp:
                pkl = pickle.load(fp)
                pkl += [float(np.mean(episode_coverage))]
            with open(save_path + "test_rec_cover.pkl", "wb") as fp:
                pickle.dump(pkl, fp)
            with open(save_path + "test_rec_connect.pkl", "rb") as fp:
                pkl = pickle.load(fp)
                pkl += [float(np.mean(episode_connectivity))]
            with open(save_path + "test_rec_connect.pkl", "wb") as fp:
                pickle.dump(pkl, fp)
            break



