import os
import numpy as np
import tensorflow as tf
from maddpg.lib4cover import make_env, get_trainers, parse_args
import maddpg.common.tf_util as U
import matplotlib.pyplot as plt
import pickle

scenario_name = "coverage2"
load_path = "./coverage2/#done_policy/"
# load_path = "./coverage_2/coverage_2_03_23_13_47/policy/"

max_ep_len = 80  # episode最多有多少状态转换
num_eps = 1
lr, gma = 1e-2, 0.95


if __name__ == "__main__":
    arglist = parse_args(scenario_name, max_ep_len, num_eps, lr, gma, num_units=128)
    tf_config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
    )
    with tf.compat.v1.Session(config=tf_config):  # 单个CPU的session
        # 创建环境
        env = make_env(scenario_name, r_cover=0.25, r_comm=0.5, comm_r_scale=0.9, comm_force_scale=0.0)
        # 创建trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, obs_shape_n, arglist)
        U.initialize()

        U.load_state(load_path)
        print("Load previous state, done!")

        obs_n = env.reset()
        episode_step = 0
        plan = [[] for i in range(4)]
        while True:
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            for i in range(4):
                plan[i].append(env.world.agents[i].state.p_pos * 1)
            new_obs_n, rew_n, done_n = env.step(action_n)
            episode_step += 1
            obs_n = new_obs_n

            if episode_step >= 50:
                obs_n = env.reset()
                episode_step = 0
                plan = [[] for i in range(4)]
            if all(done_n):
                pos_pois = np.load("./pos_PoIs.npy")
                x, y = [[] for i in range(4)], [[] for j in range(4)]
                for i in range(4):
                    for pos in plan[i]:
                        x[i].append(pos[0])
                        y[i].append(pos[1])

                color = ['r', 'g', 'b', 'm']

                plt.scatter(pos_pois[0, 0], pos_pois[0, 1], color="k", marker="*", label="PoIs")

                for i in range(1, 20):
                    plt.plot(pos_pois[i, 0], pos_pois[i, 1], color="k", marker="*", markersize=8)

                for i in range(4):
                    plt.plot(x[i], y[i], color[i] + "--", label="UAV%d" % i)

                plt.plot(x[0][0], y[0][0], color[0] + "--", marker="D", markersize=8, label="start")
                plt.plot(x[0][-1], y[0][-1], color[0] + "--", marker="s", markersize=8, label="terminal")

                for i in range(1, 4):
                    plt.plot(x[i][0], y[i][0], color[i] + "--", marker="D", markersize=8)

                for i in range(1, 4):
                    plt.plot(x[i][-1], y[i][-1], color[i] + "--", marker="s", markersize=8)

                for i in range(4):
                    plt.plot(x[i], y[i], color[i] + "-", marker="o", markersize=85, alpha=0.05, linewidth=5)

                plt.xlim([-1, 1])
                plt.ylim([-1, 1])
                plt.legend()
                plt.axis('off')
                plt.show()
                break



